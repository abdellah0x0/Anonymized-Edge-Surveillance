import cv2
# Importation des constantes depuis ton fichier config
from config.settings import BLUR_KERNEL, BLUR_SIGMA

def apply_anonymization(frame: cv2.Mat, bboxes: list, mode: str = "blur") -> cv2.Mat:
    for (x, y, w, h) in bboxes:
        # Sécurité pour ne pas dépasser les bords de l'image
        x, y = max(0, int(x)), max(0, int(y))
        w, h = int(w), int(h)
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: continue

        match mode:
            case "blur":
                # Utilisation des variables importées
                frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, BLUR_KERNEL, BLUR_SIGMA)
            case "black_box":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
                
    return frame