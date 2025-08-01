from utils.eye_drowsiness import EyeStateDetector
from backend.config import MODEL_PATHS

# Load eye state detector once
eye_detector = EyeStateDetector(MODEL_PATHS["eye_drowsiness"])

def analyze_frame(frame):
    detections = eye_detector.detect_eye_landmarks(frame)
    is_drowsy = any(d["closed"] for d in detections)

    result = {
        "eye_detected": len(detections) > 0,
        "drowsy": is_drowsy,
    }
    return result

