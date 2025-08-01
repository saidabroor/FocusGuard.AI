import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

class EyeStateDetector:
    def __init__(self, model_path, ear_threshold=0.2):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.ear_threshold = ear_threshold
        # Eye landmark indices from 68-point model
        self.left_eye_idx = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.right_eye_idx = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_eye_landmarks(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        results = []

        for face in faces:
            shape = self.predictor(gray, face)
            shape_np = face_utils.shape_to_np(shape)

            left_eye = shape_np[self.left_eye_idx[0]:self.left_eye_idx[1]]
            right_eye = shape_np[self.right_eye_idx[0]:self.right_eye_idx[1]]

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0
            is_closed = avg_ear < self.ear_threshold

            results.append({
                "landmarks": shape_np,
                "ear": avg_ear,
                "closed": is_closed
            })
        return results
