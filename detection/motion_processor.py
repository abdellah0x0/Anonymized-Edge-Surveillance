"""
ÉTAPE 5 - Privacy-Preserving Motion Heatmaps
Mode alternatif à la vidéo brute :
  - "heatmap" : accumulation de mouvement -> image thermique
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

from config.settings import (
    HEATMAP_ALPHA,
    HEATMAP_DECAY,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


# ──────────────────────────────────────────────────────────────────────────────
# HEATMAP MODE
# ──────────────────────────────────────────────────────────────────────────────

class MotionHeatmapProcessor:
    """
    Calcule un heatmap de mouvement via la différence absolue entre frames.
    On accumule le mouvement dans un buffer flottant et on le colorie.
    Résultat: on voit 'où ça bouge' sans voir les visages.
    """

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.width  = width
        self.height = height
        # Accumulateur flottant [0..255]
        self._heat_map = np.zeros((height, width), dtype=np.float32)
        self._prev_gray: np.ndarray | None = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            frame: frame BGR (H, W, 3)
        Returns:
            frame BGR avec heatmap colorisée superposée
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.width, self.height))

        if self._prev_gray is not None:
            # Différence absolue entre frames consécutives
            diff = cv2.absdiff(self._prev_gray, gray).astype(np.float32)

            # Seuillage léger pour ignorer le bruit de capteur
            _, diff = cv2.threshold(diff, 15, 255, cv2.THRESH_TOZERO)

            # Accumulation avec décroissance (effet mémoire)
            self._heat_map = self._heat_map * HEATMAP_DECAY + diff * HEATMAP_ALPHA
            np.clip(self._heat_map, 0, 255, out=self._heat_map)

        self._prev_gray = gray

        # Convertir l'accumulateur en image couleur (COLORMAP_JET: bleu→rouge)
        heat_uint8  = self._heat_map.astype(np.uint8)
        heat_color  = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color  = cv2.resize(heat_color, (frame.shape[1], frame.shape[0]))

        # Fond noir : on n'expose PAS la vidéo brute, seulement le heatmap
        result = heat_color.copy()

        # Overlay semi-transparent pour rendre le heatmap lisible
        mask = heat_uint8 > 10
        mask_resized = cv2.resize(
            mask.astype(np.uint8) * 255,
            (frame.shape[1], frame.shape[0])
        ) > 127

        # Zones sans mouvement → noir
        result[~mask_resized] = (0, 0, 0)

        # Watermark mode
        cv2.putText(result, "MODE: HEATMAP", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return result

    def reset(self):
        self._heat_map[:] = 0
        self._prev_gray = None


# ──────────────────────────────────────────────────────────────────────────────
# Façade unifiée
# ──────────────────────────────────────────────────────────────────────────────

class MotionProcessor:
    """
    Point d'entrée unique pour les modes privacy-preserving.
    Utilisé par webrtc_server.py.
    """

    MODES = ("heatmap",)

    def __init__(self):
        self._heatmap = MotionHeatmapProcessor()

    def process(self, frame: np.ndarray, mode: str = "heatmap") -> np.ndarray:
        """
        Args:
            frame: frame BGR
            mode: "heatmap"
        Returns:
            frame transforme (fond noir + heatmap)
        """
        if mode == "heatmap":
            return self._heatmap.process(frame)
        else:
            logger.warning(f"[MotionProcessor] Mode inconnu '{mode}', fallback heatmap.")
            return self._heatmap.process(frame)

    def close(self):
        return None


# Singleton
_processor_instance: MotionProcessor | None = None


def get_motion_processor() -> MotionProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = MotionProcessor()
    return _processor_instance