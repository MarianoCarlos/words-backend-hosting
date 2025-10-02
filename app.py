import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from inference_classifier import GestureClassifier

# Initialize Flask app
app = Flask(__name__)

# Detect environment for CORS
ENV = os.getenv("ENV", "development")
if ENV == "production":
    allowed_origins = [
        "https://www.insyncweb.site",   # 🔹 your Vercel frontend
    ]
else:
    allowed_origins = ["*"]  # during local dev

# Load ASL model
classifier = GestureClassifier()

def decode_frame(img_base64):
    """Convert base64 string from frontend into OpenCV image"""
    try:
        img_bytes = base64.b64decode(img_base64.split(",")[1])
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # Downscale to reduce CPU usage (faster inference on Render Free tier)
        frame = cv2.resize(frame, (224, 224))
        return frame
    except Exception:
        return None

# -------------- REST API Endpoint --------------
@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json
        img_base64 = data.get("frame")
        if not img_base64:
            return jsonify({"error": "No frame provided"}), 400

        frame = decode_frame(img_base64)
        if frame is None:
            return jsonify({"error": "Invalid frame data"}), 400

        prediction, _ = classifier.predict(frame)

        # Always return a string, even if prediction is None
        return jsonify({"prediction": prediction or ""})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Root route
@app.route("/")
def home():
    return "ASL Backend (REST API) is running ✅"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))