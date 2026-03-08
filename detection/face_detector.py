import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from config.settings import MODEL_PATH

# 1. Configuration simple
# Pas besoin de mp.tasks.python.BaseOptions, juste BaseOptions
base_options = BaseOptions(model_asset_path=str(MODEL_PATH))

options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

# 2. Création du détecteur
detector = vision.FaceDetector.create_from_options(options)

def get_face_bboxes(frame, timestamp_ms: int):
    # Conversion image pour MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    # Détection
    result = detector.detect_for_video(mp_image, timestamp_ms)
    
    bboxes = []
    if result.detections:
        for d in result.detections:
            b = d.bounding_box
            bboxes.append((b.origin_x, b.origin_y, b.width, b.height))
    return bboxes