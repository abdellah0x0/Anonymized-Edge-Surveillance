"""
detection/face_recognizer.py
Etape 4 - Reconnaissance faciale avec PyTorch + facenet-pytorch:
- MTCNN pour alignement
- InceptionResnetV1 (pretrained='vggface2') pour embeddings 512-D
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("[FaceRecognizer] facenet-pytorch/torch non installes.")

from config.settings import (
    WHITELIST_DIR,
    EMBEDDINGS_CACHE,
    FACE_RECOGNITION_TOLERANCE,
    FACE_RECOGNITION_MIN_MARGIN,
    FACE_RECOGNITION_MAX_FACES,
)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


class EmbeddingDB:
    def __init__(self):
        self.entries: list[dict] = []

    def _build_model(self):
        if not TORCH_AVAILABLE:
            return None, None, None
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            post_process=True,
            device=device,
            keep_all=False,
        )
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        return device, mtcnn, resnet

    def build_from_whitelist(self) -> int:
        if not TORCH_AVAILABLE:
            logger.error("[EmbeddingDB] Torch/facenet-pytorch indisponibles.")
            return 0

        whitelist_path = Path(WHITELIST_DIR)
        if not whitelist_path.exists():
            whitelist_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[EmbeddingDB] Dossier whitelist cree automatiquement: {whitelist_path}")
            return 0

        device, mtcnn, resnet = self._build_model()
        if mtcnn is None or resnet is None:
            return 0

        supported = {".png", ".jpg", ".jpeg", ".webp"}
        self.entries.clear()

        person_embs: dict[str, list[np.ndarray]] = {}
        for img_file in sorted(whitelist_path.rglob("*")):
            if not img_file.is_file() or img_file.suffix.lower() not in supported:
                continue
            try:
                rel = img_file.relative_to(whitelist_path)
                parts = rel.parts
                name = parts[0] if len(parts) >= 3 and parts[1] == "samples" else img_file.stem
            except Exception:
                name = img_file.stem

            img_bgr = cv2.imread(str(img_file))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            face_tensor = mtcnn(img_rgb)
            if face_tensor is None:
                logger.warning(f"[EmbeddingDB] visage non detecte: {img_file.name}")
                continue

            with torch.no_grad():
                emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()[0].astype(np.float32)
            emb = _l2_normalize(emb)
            person_embs.setdefault(name, []).append(emb)

        for name, vectors in person_embs.items():
            mean_emb = _l2_normalize(np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32))
            self.entries.append({"name": name, "embedding": mean_emb})
            logger.info(f"[EmbeddingDB] {name}: {len(vectors)} echantillons agreges")

        logger.info(f"[EmbeddingDB] Base construite: {len(self.entries)} personnes.")
        return len(self.entries)

    def save(self, path: Path | None = None) -> None:
        dest = Path(path or EMBEDDINGS_CACHE)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 2, "dim": 512, "entries": self.entries}
        with open(dest, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"[EmbeddingDB] Cache sauvegarde -> {dest}")

    def load(self, path: Path | None = None) -> bool:
        src = Path(path or EMBEDDINGS_CACHE)
        if not src.exists():
            return False
        try:
            with open(src, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "entries" in payload:
                entries = payload.get("entries", [])
                dim = int(payload.get("dim", 0))
                if entries and dim not in (0, 512):
                    logger.warning("[EmbeddingDB] Cache dimension incompatible, reconstruction requise.")
                    return False
                self.entries = entries
            else:
                # Ancien format: liste d'entrees (deepface/facenet 128)
                entries = payload if isinstance(payload, list) else []
                if entries and len(entries[0].get("embedding", [])) != 512:
                    logger.warning("[EmbeddingDB] Ancien cache detecte (dim != 512), reconstruction requise.")
                    return False
                self.entries = entries
            return True
        except Exception as e:
            logger.warning(f"[EmbeddingDB] Cache invalide: {e}")
            return False

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def names(self) -> list[str]:
        return [e["name"] for e in self.entries]

    def find_closest(self, embedding: np.ndarray) -> tuple[str | None, float, float]:
        if self.is_empty():
            return None, 1.0, 1.0
        db_matrix = np.stack([e["embedding"] for e in self.entries])
        similarities = db_matrix @ embedding
        distances = 1.0 - similarities
        order = np.argsort(distances)
        best_idx = int(order[0])
        best = float(distances[best_idx])
        second = float(distances[int(order[1])]) if len(order) > 1 else 1.0
        return self.entries[best_idx]["name"], best, second


class FaceRecognizer:
    def __init__(self, db: EmbeddingDB):
        self.db = db
        self.device = None
        self.mtcnn = None
        self.resnet = None
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                post_process=True,
                device=self.device,
                keep_all=False,
            )
            self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def _embed_from_bbox_crop(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x, y, w, h = bbox
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        if w < 80 or h < 80:
            return None
        padx, pady = int(w * 0.25), int(h * 0.25)
        x1 = max(0, x - padx)
        y1 = max(0, y - pady)
        x2 = min(frame.shape[1], x + w + padx)
        y2 = min(frame.shape[0], y + h + pady)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_tensor = self.mtcnn(crop_rgb) if self.mtcnn is not None else None
        if face_tensor is None:
            # Fallback rapide si MTCNN echoue localement
            resized = cv2.resize(crop_rgb, (160, 160))
            t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
            face_tensor = (t - 0.5) / 0.5

        with torch.no_grad():
            emb = self.resnet(face_tensor.unsqueeze(0).to(self.device)).cpu().numpy()[0].astype(np.float32)
        return _l2_normalize(emb)

    def identify_faces(self, frame: np.ndarray, bboxes: list) -> list[dict]:
        results: list[dict] = []
        if not bboxes or not TORCH_AVAILABLE or self.db.is_empty() or self.resnet is None:
            return [{"bbox": b, "name": None, "is_known": False, "distance": 1.0} for b in bboxes]

        indexed = list(enumerate(bboxes))
        indexed.sort(key=lambda item: int(item[1][2]) * int(item[1][3]), reverse=True)
        selected = indexed[: max(1, FACE_RECOGNITION_MAX_FACES)]
        selected_idx = {idx for idx, _ in selected}
        by_index: dict[int, dict] = {}

        for idx, bbox in selected:
            try:
                emb = self._embed_from_bbox_crop(frame, bbox)
                if emb is None:
                    by_index[idx] = {"bbox": bbox, "name": None, "is_known": False, "distance": 1.0}
                    continue
                name, dist, second_dist = self.db.find_closest(emb)
                margin = second_dist - dist
                is_known = (dist <= FACE_RECOGNITION_TOLERANCE) and (margin >= FACE_RECOGNITION_MIN_MARGIN)
                by_index[idx] = {
                    "bbox": bbox,
                    "name": name if is_known else None,
                    "is_known": is_known,
                    "distance": float(dist),
                }
            except Exception as e:
                logger.warning(f"[FaceRecognizer] Erreur bbox {bbox}: {e}")
                by_index[idx] = {"bbox": bbox, "name": None, "is_known": False, "distance": 1.0}

        for idx, bbox in enumerate(bboxes):
            if idx in selected_idx:
                results.append(by_index.get(idx, {"bbox": bbox, "name": None, "is_known": False, "distance": 1.0}))
            else:
                results.append({"bbox": bbox, "name": None, "is_known": False, "distance": 1.0})
        return results


_db_instance: EmbeddingDB | None = None
_recognizer_instance: FaceRecognizer | None = None


def get_db() -> EmbeddingDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = EmbeddingDB()
    return _db_instance


def get_recognizer() -> FaceRecognizer:
    global _recognizer_instance
    if _recognizer_instance is None:
        _recognizer_instance = FaceRecognizer(get_db())
    return _recognizer_instance


def identify_faces(frame: np.ndarray, bboxes: list) -> list[dict]:
    return get_recognizer().identify_faces(frame, bboxes)