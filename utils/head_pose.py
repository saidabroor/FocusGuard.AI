import cv2
import dlib
import numpy as np
import os
import math

# Load the shape predictor
predictor_path = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Landmark indices
LANDMARK_INDEXES = [30, 8, 36, 45, 48, 54]  # Nose, Chin, Left eye, Right eye, Left mouth, Right mouth

import cv2
import numpy as np
import math

def get_head_pose(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    shape = predictor(gray, face)

    image_points = np.array([
        (shape.part(i).x, shape.part(i).y) for i in LANDMARK_INDEXES
    ], dtype="double")

    height, width = frame.shape[:2]
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rmat = cv2.Rodrigues(rotation_vector)[0]

    # Calculate Euler angles from rotation matrix
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0

    # Convert radians to degrees
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)

    return pitch, yaw, roll

