import cv2
import requests

URL = "http://localhost:5000/predict"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Encode frame
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'frame': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

    try:
        # Send frame to backend
        response = requests.post(URL, files=files)
        result = response.json()
        print("Prediction:", result)

        # Get status values
        drowsy = result.get("eye_status", {}).get("drowsy", False)
        eye_detected = result.get("eye_status", {}).get("eye_detected", False)
        focus_status = result.get("focus_status", "")
        head_status = result.get("head_status", "")

        # Draw status texts
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Focus: {focus_status}", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Head: {head_status}", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Eyes: {'Detected' if eye_detected else 'Not Detected'}", (10, 90), font, 0.7, (255, 255, 255), 2)

        # Set border color based on status
        if not eye_detected or "No Face" in head_status:
            border_color = (0, 255, 255)  # Yellow
        elif drowsy or "Not Focused" in focus_status or "Looking Away" in head_status:
            border_color = (0, 0, 255)  # Red
        else:
            border_color = (0, 255, 0)  # Green

        # Draw border
        thickness = 10
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w-1, h-1), border_color, thickness)

        # Big alert message if distracted or drowsy
        if drowsy:
            cv2.putText(frame, "😴 DROWSY!", (w//4, h//2), font, 1.5, (0, 0, 255), 4)
        elif "Not Focused" in focus_status or "Looking Away" in head_status:
            cv2.putText(frame, "⚠️ NOT FOCUSED", (w//4, h//2), font, 1.2, (0, 0, 255), 3)
        elif not eye_detected:
            cv2.putText(frame, "❓ NO EYES DETECTED", (w//5, h//2), font, 1.0, (0, 255, 255), 3)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("FocusGuard.ai Webcam - Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
