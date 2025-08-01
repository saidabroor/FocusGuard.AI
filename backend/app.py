from flask import Flask, request, jsonify
import cv2
import numpy as np
from backend.predictor import analyze_frame

app = Flask(__name__)

@app.route("/")
def index():
    return "FocusGuard.AI Backend is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files["frame"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = analyze_frame(frame)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
