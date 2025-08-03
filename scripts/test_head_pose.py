import cv2
from utils.head_pose import get_head_pose

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = get_head_pose(frame)
    if result:
        pitch, yaw, roll = result
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Dlib Head Pose Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
