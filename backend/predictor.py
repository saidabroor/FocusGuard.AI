from utils.eye_drowsiness import EyeStateDetector
from utils.head_pose import get_head_pose
from backend.config import MODEL_PATHS
from utils.focus_tracker import FocusEventTracker

# Load eye state detector once
eye_detector = EyeStateDetector(MODEL_PATHS["eye_drowsiness"])
focus_tracker = FocusEventTracker(threshold_seconds=6)


def determine_focus(eye_status, head_status):
    if not eye_status["eye_detected"]:
        return "No Eyes Detected"
    if head_status == "No Face Detected":
        return "No Face Detected"
    if eye_status["drowsy"] or head_status == "Looking Away":
        return "Not Focused"
    return "Focused"
def analyze_frame(frame):
    # Eye detection
    detections = eye_detector.detect_eye_landmarks(frame)
    is_drowsy = any(d["closed"] for d in detections)
    eye_status = {
        "eye_detected": len(detections) > 0,
        "drowsy": is_drowsy,
    }

    # Head pose detection
    pitch_yaw_roll = get_head_pose(frame)
    if pitch_yaw_roll:
        pitch, yaw, roll = pitch_yaw_roll
        
        # Adjust pitch to center around 0 (forward look)
        pitch -= 180
        if pitch < -180:
            pitch += 360
        
        # Normalize yaw and roll
        if yaw > 180:
            yaw -= 360
        if roll > 180:
            roll -= 360
        
        print(f"Pitch: {pitch:.2f}, Yaw: {yaw:.2f}, Roll: {roll:.2f}")
        
        if abs(yaw) > 30 or abs(pitch) > 20:
            head_status = "Looking Away"
        else:
            head_status = "Focused"
    else:
        head_status = "No Face Detected"

    # Final focus determination
    focus_status = determine_focus(eye_status, head_status)
    focus_tracker.update_status(focus_status)

    return {
        "eye_status": eye_status,
        "head_status": head_status,
        "focus_status": focus_status
    } 
