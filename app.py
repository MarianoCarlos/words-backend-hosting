import os
import cv2
import base64
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
from inference_classifier import GestureClassifier

app = Flask(__name__)

# Detect environment
ENV = os.getenv("ENV", "development")

if ENV == "production":
    allowed_origins = [
        "https://www.insyncweb.site"  # 🔹 Replace with your real Vercel domain
    ]
else:
    allowed_origins = ["http://localhost:3000"] # allow all during local dev

socketio = SocketIO(app, cors_allowed_origins=allowed_origins)

# Load ASL model
classifier = GestureClassifier()

def decode_frame(img_base64):
    """Decode base64 frame from frontend into OpenCV image"""
    img_bytes = base64.b64decode(img_base64.split(",")[1])
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

@socketio.on("video_frame")
def handle_video_frame(data):
    """
    Expects:
    { "frame": "data:image/jpeg;base64,...." }
    """
    try:
        frame = decode_frame(data["frame"])
        prediction, _ = classifier.predict(frame)

        if prediction:
            emit("prediction", {"label": prediction}, broadcast=True)
        else:
            emit("prediction", {"label": "No hand detected"}, broadcast=True)
    except Exception as e:
        emit("prediction", {"error": str(e)})

@app.route("/")
def index():
    return f"ASL Backend Running 🚀 (ENV={ENV})"

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
