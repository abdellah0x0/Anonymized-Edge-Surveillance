"""
Demo locale ETAPE 4:
- detection visage (MediaPipe)
- reconnaissance whitelist (DeepFace/FaceNet)
- anonymisation selective + alertes
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from anonymization.redactor import apply_selective_anonymization
from config.settings import FRAME_HEIGHT, FRAME_WIDTH
from detection.face_detector import get_face_bboxes
from detection.face_recognizer import get_db, identify_faces


def main() -> int:
    db = get_db()
    if not db.load() or db.is_empty():
        print("Embeddings non charges. Genere d'abord whitelist/.embeddings_cache.pkl")
        return 2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print("Webcam introuvable.")
        return 2

    print("Demo selective en cours. Touches: [q]=quit, [b]=blur, [x]=black_box")
    mode = "blur"
    frame_count = 0
    last_ms = int(time.time() * 1000)
    last_fps_time = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame_count += 1
            now_ms = int(time.time() * 1000)
            if now_ms <= last_ms:
                now_ms = last_ms + 1
            last_ms = now_ms

            bboxes = get_face_bboxes(frame, now_ms)
            results = identify_faces(frame, bboxes)
            out, alerts = apply_selective_anonymization(frame.copy(), results, mode=mode, draw_labels=True)

            if alerts:
                for alert in alerts:
                    print(alert)

            elapsed = time.time() - last_fps_time
            if elapsed >= 0.5:
                fps = frame_count / max(elapsed, 1e-6)
                frame_count = 0
                last_fps_time = time.time()

            cv2.putText(out, f"Mode: {mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2)
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2)
            cv2.imshow("Selective Redaction Demo", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                mode = "blur"
            if key == ord("x"):
                mode = "black_box"
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
