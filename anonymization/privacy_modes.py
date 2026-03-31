"""
anonymization/privacy_modes.py
Façade qui orchestre les différents modes de confidentialité.

Modes disponibles:
  "blur"      → flou sur tous les visages (étape 3)
  "selective" → flou sélectif whitelist  (étape 4)
  "heatmap"   → heatmap de mouvement     (étape 5)
"""

import logging
import numpy as np

from anonymization.redactor import apply_anonymization, apply_selective_anonymization
from config.settings import SELECTIVE_REIDENTIFY_EVERY_N_FRAMES
from detection.face_detector import get_face_bboxes
from detection.face_recognizer import identify_faces
from detection.motion_processor import get_motion_processor

logger = logging.getLogger(__name__)

# Modes qui bypassent complètement la détection de visages
MOTION_MODES = {"heatmap"}

# Tous les modes supportés
ALL_MODES = {"blur", "selective", "heatmap"}


# Buffers pour le frame skipping et tracking
_last_bboxes = []
_last_face_results = []
_last_recognition_frame = 0


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _remap_results_to_current_bboxes(current_bboxes: list, previous_results: list[dict]) -> list[dict]:
    """
    Evite les inversions d'identite quand l'ordre des bboxes change entre 2 frames.
    On associe chaque bbox courante au resultat precedent ayant le meilleur IoU.
    """
    if not current_bboxes or not previous_results:
        return []

    remapped: list[dict] = []
    used_prev = set()

    for bbox in current_bboxes:
        best_idx = -1
        best_iou = 0.0
        for idx, prev in enumerate(previous_results):
            if idx in used_prev:
                continue
            iou = _bbox_iou(bbox, prev["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= 0.2:
            chosen = dict(previous_results[best_idx])
            chosen["bbox"] = bbox
            remapped.append(chosen)
            used_prev.add(best_idx)
        else:
            remapped.append({"bbox": bbox, "name": None, "is_known": False, "distance": 1.0})

    return remapped

def process_frame(
    frame: np.ndarray,
    timestamp_ms: int,
    mode: str = "blur",
    frame_count: int = 0
) -> tuple[np.ndarray, list[str]]:
    """
    Pipeline optimisé pour la latence (Point 5+ : Skip 1/10 + Tracking).
    """
    global _last_bboxes, _last_face_results, _last_recognition_frame

    if mode not in ALL_MODES:
        mode = "blur"

    alerts: list[str] = []

    # ── Modes mouvement (heatmap) ───────────────────────────────────────────
    if mode in MOTION_MODES:
        processor = get_motion_processor()
        return processor.process(frame, mode), alerts

    # ── 1. Détection : 1 frame sur 3 ─────────────────────────────────────────
    if frame_count % 3 == 0 or not _last_bboxes:
        _last_bboxes = get_face_bboxes(frame, timestamp_ms)

    # ── 2. Anonymisation simple (Blur) ───────────────────────────────────────
    if mode == "blur":
        frame_out = apply_anonymization(frame.copy(), _last_bboxes, mode="blur")

    # ── 3. Mode Whitelist (Sélectif) ─────────────────────────────────────────
    elif mode == "selective":
        # Re-identification plus frequente pour reduire les mauvaises attributions.
        should_reidentify = (frame_count - _last_recognition_frame >= SELECTIVE_REIDENTIFY_EVERY_N_FRAMES) or \
                            (len(_last_bboxes) != len(_last_face_results))
        
        if should_reidentify and _last_bboxes:
            _last_face_results = identify_faces(frame, _last_bboxes)
            _last_recognition_frame = frame_count
        
        # Si on a des resultats, on les remappe par IoU pour eviter le swap de noms.
        if _last_face_results and _last_bboxes:
            _last_face_results = _remap_results_to_current_bboxes(_last_bboxes, _last_face_results)

            frame_out, alerts = apply_selective_anonymization(
                frame.copy(), _last_face_results, mode="blur", draw_labels=True
            )
        else:
            # Fallback simple blur si pas encore de reconnaissance
            frame_out = apply_anonymization(frame.copy(), _last_bboxes, mode="blur")

    else:
        frame_out = apply_anonymization(frame.copy(), _last_bboxes, mode="blur")

    return frame_out, alerts