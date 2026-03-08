import cv2
import time
from detection.face_detector import get_face_bboxes
from anonymization.redactor import apply_anonymization # (Garde la version précédente)

def run_privacy_filter():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start_perf = time.perf_counter()
        
        ret, frame = cap.read()
        if not ret: break

        # L'API Task en mode VIDEO nécessite un timestamp en millisecondes
        timestamp_ms = int(time.time() * 1000)
        
        # 1. Détection
        faces = get_face_bboxes(frame, timestamp_ms)
        
        # 2. Anonymisation (le match/case reste identique)
        frame = apply_anonymization(frame, faces, mode="blur")

        # Calcul de la latence réelle
        latency = (time.perf_counter() - start_perf) * 1000
        cv2.putText(frame, f"{latency:.1f}ms", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Privacy Filter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_privacy_filter()