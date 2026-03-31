"""
Test offline de l'etape 4:
- charge les embeddings whitelist
- detecte les visages sur les images samples
- identifie chaque visage avec face_recognizer
- affiche precision et distances pour aider au tuning du seuil
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import FACE_RECOGNITION_TOLERANCE, WHITELIST_DIR
from detection.face_detector import get_face_bboxes
from detection.face_recognizer import get_db, identify_faces


def iter_sample_images(root: Path, per_person: int) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        samples_dir = person_dir / "samples"
        if not samples_dir.exists():
            continue

        count = 0
        for img_path in sorted(samples_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rows.append((person_dir.name, img_path))
            count += 1
            if count >= per_person:
                break
    return rows


def run(per_person: int, verbose: bool) -> int:
    db = get_db()
    if not db.load():
        print("Cache embeddings absent/invalide. Reconstruction whitelist en cours...")
        built = db.build_from_whitelist()
        if built <= 0:
            print("Echec de reconstruction embeddings.")
            return 2
        db.save()

    print(f"Embeddings charges: {len(db.entries)} | noms: {db.names()}")
    print(f"Tolerance actuelle: {FACE_RECOGNITION_TOLERANCE:.3f}")

    sample_rows = iter_sample_images(WHITELIST_DIR, per_person=per_person)
    if not sample_rows:
        print("Aucune image sample detectee dans whitelist/*/samples.")
        return 2

    stats = defaultdict(int)
    known_distances: list[float] = []
    unknown_distances: list[float] = []
    ts = 0

    for expected_name, img_path in sample_rows:
        frame = cv2.imread(str(img_path))
        if frame is None:
            stats["read_fail"] += 1
            continue

        ts += 33
        bboxes = get_face_bboxes(frame, ts)
        if not bboxes:
            # Les samples sont deja crops, fallback sur toute l'image.
            h, w = frame.shape[:2]
            bboxes = [(0, 0, w, h)]
            stats["no_face"] += 1
            if verbose:
                print(f"[NO_FACE_FALLBACK] {img_path.name} -> full-frame bbox")

        # Si bbox trop petite pour la logique de production, fallback egalement.
        x, y, w, h = bboxes[0]
        if w < 80 or h < 80:
            h0, w0 = frame.shape[:2]
            bboxes = [(0, 0, w0, h0)]

        results = identify_faces(frame, bboxes[:1])
        if not results:
            stats["no_result"] += 1
            continue

        r = results[0]
        pred_name = r["name"]
        is_known = bool(r["is_known"])
        dist = float(r["distance"])

        if is_known:
            known_distances.append(dist)
        else:
            unknown_distances.append(dist)

        if is_known and pred_name == expected_name:
            stats["correct"] += 1
            verdict = "OK"
        elif is_known and pred_name != expected_name:
            stats["wrong_person"] += 1
            verdict = "WRONG_PERSON"
        else:
            stats["unknown"] += 1
            verdict = "UNKNOWN"

        if verbose:
            print(
                f"[{verdict}] expected={expected_name:10s} "
                f"pred={str(pred_name):10s} dist={dist:.4f} file={img_path.name}"
            )

    total_eval = stats["correct"] + stats["wrong_person"] + stats["unknown"]
    print("\n=== RESULTATS ===")
    print(f"images_evaluees        : {total_eval}")
    print(f"correct                : {stats['correct']}")
    print(f"wrong_person           : {stats['wrong_person']}")
    print(f"unknown                : {stats['unknown']}")
    print(f"no_face                : {stats['no_face']}")
    print(f"read_fail              : {stats['read_fail']}")

    if total_eval:
        acc = 100.0 * stats["correct"] / total_eval
        print(f"accuracy (eval only)   : {acc:.2f}%")

    if known_distances:
        print(
            f"distance connus        : min={min(known_distances):.4f} "
            f"avg={sum(known_distances)/len(known_distances):.4f} "
            f"max={max(known_distances):.4f}"
        )
    if unknown_distances:
        print(
            f"distance inconnus      : min={min(unknown_distances):.4f} "
            f"avg={sum(unknown_distances)/len(unknown_distances):.4f} "
            f"max={max(unknown_distances):.4f}"
        )

    if known_distances and unknown_distances:
        print(
            "Suggestion: choisir un seuil entre max(connus) "
            "et min(inconnus) si possible."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test offline reconnaissance selective")
    parser.add_argument("--per-person", type=int, default=8, help="nb images par personne")
    parser.add_argument("--verbose", action="store_true", help="afficher chaque prediction")
    args = parser.parse_args()
    return run(per_person=max(1, args.per_person), verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
