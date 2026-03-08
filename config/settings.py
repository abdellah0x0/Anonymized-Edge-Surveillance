from pathlib import Path

# Le dossier racine du projet (privacy-guardian/)
ROOT = Path(__file__).parent.parent

# Le chemin direct vers ton modèle
MODEL_PATH = ROOT / "models" / "blaze_face_short_range.tflite"

BLUR_KERNEL = (99, 99)
BLUR_SIGMA = 30