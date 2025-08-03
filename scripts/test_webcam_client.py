import cv2
import requests

URL = "http://localhost:5000/predict"

cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Encode frame to JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'frame': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

    try:
        # Send to backend
        response = requests.post(URL, files=files)
        print("Prediction:", response.json())
    except Exception as e:
        print("Error:", e)

    # Show frame
    cv2.imshow("Webcam - Press Q to Quit", frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
