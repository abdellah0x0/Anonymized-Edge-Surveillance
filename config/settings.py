"""
config/settings.py
Toutes les constantes de configuration du projet.
"""

from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────

ROOT              = Path(__file__).parent.parent
MODEL_PATH        = ROOT / "models" / "blaze_face_short_range.tflite"
WHITELIST_DIR     = ROOT / "whitelist"
# Cache pickle des embeddings pré-calculés (généré par main.ipynb)
EMBEDDINGS_CACHE  = ROOT / "whitelist" / ".embeddings_cache.pkl"

# Crée automatiquement le dossier whitelist s'il n'existe pas.
WHITELIST_DIR.mkdir(parents=True, exist_ok=True)

# ── Capture vidéo ─────────────────────────────────────────────────────────────

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30

# ── Anonymisation (étape 3) ────────────────────────────────────────────────────

BLUR_KERNEL = (99, 99)
BLUR_SIGMA  = 30

# ── Reconnaissance faciale (étape 4) ──────────────────────────────────────────

# Seuil cosine distance FaceNet : 0.0 = identique, 1.0 = différent
# Seuil officiel DeepFace pour FaceNet+cosine ≈ 0.40
# Plus bas = plus strict (moins de faux positifs, plus de INCONNU)
FACE_RECOGNITION_TOLERANCE = 0.35

# Ecart minimum entre le meilleur match et le second.
# Si trop faible, on considere le resultat ambigu -> INCONNU.
FACE_RECOGNITION_MIN_MARGIN = 0.04

# Nombre max de visages a reconnaitre par frame (les plus grands).
# Permet de reduire la latence en scene chargee.
FACE_RECOGNITION_MAX_FACES = 2

# Frequence de re-identification en mode selective.
# Plus grand = moins de lag, plus petit = plus reactif.
SELECTIVE_REIDENTIFY_EVERY_N_FRAMES = 8

# ── Heatmap (étape 5) ─────────────────────────────────────────────────────────

HEATMAP_ALPHA = 0.6    # poids de la nouvelle diff dans l'accumulateur
HEATMAP_DECAY = 0.85   # décroissance (0.9 = longue mémoire)

# ── Mode par défaut ────────────────────────────────────────────────────────────

# "blur" | "selective" | "heatmap"
DEFAULT_PRIVACY_MODE = "blur"