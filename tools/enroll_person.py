"""
tools/enroll_person.py
Capture automatique de 30 images d'une personne pour la whitelist.
Stockage dans un dossier nominatif dans whitelist/.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Ajouter la racine au path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import WHITELIST_DIR, FRAME_WIDTH, FRAME_HEIGHT

# ── Constantes ────────────────────────────────────────────────────────────────
N_CAPTURES_DEFAULT = 30
CAPTURE_INTERVAL   = 0.15   # secondes entre deux captures
MIN_FACE_SIZE      = 80     # pixels
FACENET_INPUT_SIZE = 160    # taille pour le stockage des crops

# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def detect_face_opencv(frame: np.ndarray) -> tuple[np.ndarray | None, tuple]:
    """
    Détecte le visage principal avec Haar Cascade.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade  = cv2.CascadeClassifier(cascade_path)

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    if len(faces) == 0:
        return None, ()

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Marge 25 %
    mx = int(w * 0.25); my = int(h * 0.25)
    x1 = max(0, x - mx);              y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx); y2 = min(frame.shape[0], y + h + my)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, ()

    crop_resized = cv2.resize(crop, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE))
    return crop_resized, (x, y, w, h)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*40)
    print("   ENRÔLEMENT NOUVELLE PERSONNE")
    print("="*40 + "\n")
    
    name = input("Entrez le prénom de la personne : ").strip()
    if not name:
        print("❌ Erreur : Le nom est obligatoire.")
        sys.exit(1)

    # ── Préparer les dossiers ─────────────────────────────────────────────────
    person_dir = WHITELIST_DIR / name
    samples_dir = person_dir / "samples"
    person_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ Webcam introuvable.")
        sys.exit(1)

    print(f"\n📸 Prêt pour l'enrôlement de : {name.upper()}")
    print("➡️  Placez-vous face à la caméra.")
    print("   Appuyez sur [ESPACE] pour lancer la capture auto de 30 images.")
    print("   Appuyez sur [Q] pour annuler.\n")

    # ── Attente démarrage ─────────────────────────────────────────────────────
    start_capture = False
    while True:
        ret, frame = cap.read()
        if not ret: break

        face_crop, bbox = detect_face_opencv(frame)
        display = frame.copy()

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, "PRET - ESPACE pour demarrer", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Visage non detecte", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Enrolement : " + name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and bbox:
            start_capture = True
            break
        if key == ord('q'):
            break

    # ── Phase de capture automatique ──────────────────────────────────────────
    captured = 0
    last_cap_time = 0
    
    if start_capture:
        print(f"🔄 Capture en cours dans : {samples_dir} ...")
        while captured < N_CAPTURES_DEFAULT:
            ret, frame = cap.read()
            if not ret: break

            now = time.time()
            face_crop, bbox = detect_face_opencv(frame)
            display = frame.copy()

            if bbox and (now - last_cap_time >= CAPTURE_INTERVAL):
                img_path = samples_dir / f"{name}_{captured:02d}.jpg"
                cv2.imwrite(str(img_path), face_crop)
                captured += 1
                last_cap_time = now
                print(f"  Capture {captured}/{N_CAPTURES_DEFAULT}", end="\r")

            # Feedback visuel
            cv2.rectangle(display, (0, 450), (640, 480), (0,0,0), -1)
            cv2.putText(display, f"Progression : {captured}/{N_CAPTURES_DEFAULT}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Enrolement : " + name, display)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                print("\n❌ Interrompu par l'utilisateur.")
                break

    cap.release()
    cv2.destroyAllWindows()

    if captured >= N_CAPTURES_DEFAULT:
        print(f"\n\n✅ Enrôlement terminé !")
        print(f"📁 {captured} images sauvegardées dans : {samples_dir}")
        print("💡 Note : Les embeddings seront calculés lors de l'entraînement global.")
    else:
        print("\n⚠️  Enrôlement incomplet.")
