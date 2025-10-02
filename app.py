import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_classifier import GestureClassifier

# Initialize Flask app
app = Flask(__name__)

# ✅ Enable CORS for all origins (safe for dev / testing)
CORS(app)

# Load ASL model
classifier = GestureClassifier()

def decode_frame_file(file):
    """Decode uploaded file (FormData) into OpenCV frame"""
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return cv2.resize(frame, (224, 224))
    except Exception:
        return None

# -------------- REST API Endpoint --------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        frame = None

        # Only support FormData (file upload)
        if "file" in request.files:
            frame = decode_frame_file(request.files["file"])

        if frame is None:
            return jsonify({"error": "No valid frame provided"}), 400

        prediction, _ = classifier.predict(frame)
        return jsonify({"prediction": prediction or ""})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Root route
@app.route("/")
def home():
    return "ASL Backend (REST API) is running ✅"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
