# FocusGuard.AI 👀🧠

**FocusGuard.AI** is a real-time focus detection system that uses computer vision to monitor eye states and head movements to determine whether a person is **focused**, **drowsy**, or **distracted** during study or work sessions.

This project is built using Python, Flask, OpenCV, dlib, and deep learning, with a frontend powered by Bolt AI for an interactive dashboard.

---

## 🚀 Features

- 🎥 **Webcam-based real-time monitoring**
- 🧠 **Focus detection using eye state and head pose**
- 📈 **Live line graph showing focus status over time**
- 📁 **CSV logging of focus states with timestamps**
- 🔔 **Optional real-time alerts for prolonged distraction or drowsiness**
- 📊 **Summary report with total focused time, drowsiness, and distractions**

---

## 🛠️ Project Structure

FocusGuard.AI/
├── backend/
│ ├── app.py # Flask server (API)
│ ├── predictor.py # Combines models and returns focus status
├── Models/
│ ├── eye_state_model.dat # Pretrained dlib model for eye detection
│ └── head_pose_model.py # Head pose estimation logic
├── Utils/
│ ├── eye_drowsiness.py # EAR calculation and eye state logic
│ ├── head_pose.py # Head direction and angle evaluation
├── Notebooks/
│ ├── video_analysis.ipynb # Analysis of recorded videos (visualizations, plots)
├── templates/
│ └── index.html # Frontend (if using with Flask locally)
├── static/
│ └── style.css # Styles for frontend
├── requirements.txt
├── README.md
└── .gitignore

yaml
Copy
Edit

---

## 📦 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/FocusGuard.AI.git
cd FocusGuard.AI
2. Create and Activate Virtual Environment
bash
Copy
Edit
conda create -n focusenv python=3.8
conda activate focusenv
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
# If using dlib:
conda install -c conda-forge dlib
▶️ Run the App
bash
Copy
Edit
# Start the Flask backend
cd backend
python app.py
Then, launch the frontend (Bolt AI app or local HTML dashboard) that connects to the API at http://localhost:5000/predict.

📊 Dashboard (Frontend)
The dashboard:

Captures webcam frames

Sends frames to Flask backend for prediction

Displays real-time status and live focus graph

Optionally logs to CSV

Generates summary at end of session

👉 You can run the frontend in Bolt AI or any modern web interface.

📚 How It Works
Eye Drowsiness Detection: Uses Eye Aspect Ratio (EAR) to detect whether eyes are closed.

Head Pose Estimation: Estimates yaw, pitch, and roll to detect if head is turned away.

Focus Status: Combined logic determines if user is focused, drowsy, or distracted.

📌 Future Improvements
Add voice feedback for alerts

User authentication and session tracking

Mobile support

Multi-user classroom view

🙌 Credits
Developed by [Your Name] using:

OpenCV

dlib

Flask

Python

Bolt AI (Frontend)

```
