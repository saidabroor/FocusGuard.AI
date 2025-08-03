import cv2
import csv
import os
from datetime import datetime
from utils.eye_drowsiness import EyeStateDetector
from utils.head_pose import get_head_pose
from backend.config import MODEL_PATHS
from utils.focus_tracker import FocusEventTracker

# Initialize models
eye_detector = EyeStateDetector(MODEL_PATHS["eye_drowsiness"])
focus_tracker = FocusEventTracker(threshold_seconds=0)  

# This FocusEventTracker will be modified below to log every change instantly (no minimum duration)

def update_status_immediate(self, new_status):
    now = datetime.now().strftime("%H:%M:%S")  
    if new_status != self.current_status:
        # Log previous status end at now
        if self.current_status is not None:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.status_start_time, now, self.current_status])
            print(f"Logged: {self.current_status} from {self.status_start_time} to {now}")
        # Start new status
        self.current_status = new_status
        self.status_start_time = now

FocusEventTracker.update_status = update_status_immediate
focus_tracker.current_status = None
focus_tracker.status_start_time = None

def determine_focus(eye_status, head_status):
    if not eye_status["eye_detected"]:
        return "No Eyes Detected"
    if head_status == "No Face Detected":
        return "No Face Detected"
    if eye_status["drowsy"] or head_status == "Looking Away":
        return "Not Focused"
    return "Focused"

def analyze_video(video_path, csv_path):
    # Create/clear CSV file and write header
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Start Time", "End Time", "Status"])

    cap = cv2.VideoCapture(video_path)

    focus_tracker.csv_path = csv_path
    focus_tracker.current_status = None
    focus_tracker.status_start_time = None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

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
            pitch -= 180
            if pitch < -180:
                pitch += 360
            if yaw > 180:
                yaw -= 360
            if roll > 180:
                roll -= 360

            if abs(yaw) > 30 or abs(pitch) > 20:
                head_status = "Looking Away"
            else:
                head_status = "Focused"
        else:
            head_status = "No Face Detected"

        focus_status = determine_focus(eye_status, head_status)

        # Log status change immediately
        focus_tracker.update_status(focus_status)

    # Log the last state to close it out with current time
    now = datetime.now().strftime("%H:%M:%S")
    if focus_tracker.current_status is not None and focus_tracker.status_start_time != now:
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([focus_tracker.status_start_time, now, focus_tracker.current_status])
        print(f"Logged final: {focus_tracker.current_status} from {focus_tracker.status_start_time} to {now}")

    cap.release()
    print(f"Video processed, total frames: {frame_count}")

# Usage
video_path = "videos/test_video.mp4"  
csv_path = "logs/focus_timeline.csv"  # new CSV for timeline
analyze_video(video_path, csv_path)
