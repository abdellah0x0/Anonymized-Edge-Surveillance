"""
anonymization/redactor.py
Applique le mode d'anonymisation choisi sur les bounding boxes détectées.
Supporte le flou sélectif (whitelist) introduit à l'étape 4.
"""

import cv2
import numpy as np
from config.settings import BLUR_KERNEL, BLUR_SIGMA


def apply_anonymization(
    frame: np.ndarray,
    bboxes: list,
    mode: str = "blur",
) -> np.ndarray:
    """
    Flou ou boîte noire sur toutes les bboxes.
    Utilisé à l'étape 3 (toutes les faces).

    Args:
        frame : image BGR
        bboxes: liste de (x, y, w, h)
        mode  : "blur" | "black_box"
    """
    for (x, y, w, h) in bboxes:
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)

        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        match mode:
            case "blur":
                frame[y:y + h, x:x + w] = cv2.GaussianBlur(
                    roi, BLUR_KERNEL, BLUR_SIGMA
                )
            case "black_box":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

    return frame


def apply_selective_anonymization(
    frame: np.ndarray,
    face_results: list[dict],
    mode: str = "blur",
    draw_labels: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    ÉTAPE 4 - Flou sélectif basé sur la whitelist.

    Args:
        frame        : image BGR
        face_results : liste de dicts retournés par face_recognizer.identify_faces()
                       Chaque dict: {bbox, name, is_known, distance}
        mode         : "blur" | "black_box"
        draw_labels  : afficher le nom sous le visage connu

    Returns:
        (frame annoté, liste d'alertes pour DataChannel)
    """
    alerts: list[str] = []

    for result in face_results:
        x, y, w, h = result["bbox"]
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)

        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        if result["is_known"]:
            # Visage whitelisté → pas de flou, on dessine juste un cadre vert
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 80), 2)
            if draw_labels:
                label = result["name"] or "connu"
                cv2.putText(
                    frame, label,
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2,
                )
        else:
            # Visage inconnu → anonymisation + alerte
            match mode:
                case "blur":
                    frame[y:y + h, x:x + w] = cv2.GaussianBlur(
                        roi, BLUR_KERNEL, BLUR_SIGMA
                    )
                case "black_box":
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 0, 0), -1
                    )

            # Cadre rouge pour l'opérateur
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 220), 2)
            if draw_labels:
                cv2.putText(
                    frame, "INCONNU",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2,
                )

            # Génère une alerte DataChannel
            import time
            ts = time.strftime("%H:%M:%S")
            alerts.append(f"ALERT|{ts}|Visage inconnu détecté")

    return frame, alerts