import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_classifier import GestureClassifier

# Initialize Flask app
app = Flask(__name__)

# Detect environment for CORS
ENV = os.getenv("ENV", "development")
if ENV == "production":
    allowed_origins = [
        "https://www.insyncweb.site",   # 🔹 your deployed Vercel frontend
        "http://localhost:3000",  # 🔹 for local dev / testing
    ]
else:
    allowed_origins = ["*"]  # during local dev / testing

# ✅ Enable CORS
CORS(
    app,
    resources={r"/*": {"origins": allowed_origins}},
    supports_credentials=True,
    allow_headers="*",
    methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
)

# Load ASL model
classifier = GestureClassifier()

def decode_frame(img_base64):
    """Convert base64 string from frontend into OpenCV image"""
    try:
        img_bytes = base64.b64decode(img_base64.split(",")[1])
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # Downscale for lower CPU usage (good for free-tier hosting)
        frame = cv2.resize(frame, (224, 224))
        return frame
    except Exception:
        return None

# -------------- REST API Endpoint --------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        frame = None

        # ✅ Case 1: JSON with base64
        if request.is_json:
            data = request.get_json()
            img_base64 = data.get("frame")
            if img_base64:
                frame = decode_frame(img_base64)

        # ✅ Case 2: FormData with file
        if "file" in request.files:
            file = request.files["file"]
            file_bytes = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (224, 224))

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
