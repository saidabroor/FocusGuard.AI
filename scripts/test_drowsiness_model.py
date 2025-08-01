import cv2
import dlib
from imutils import face_utils

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/drowsiness_model.dat")  # <-- Update this!

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = face_utils.shape_to_np(shape)

        for (x, y) in shape_np:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Eye Landmarks", frame)

    # Break with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
