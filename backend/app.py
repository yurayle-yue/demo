# backend/app.py
import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import pathlib
import json

# For model loading
import tensorflow as tf
import numpy as np

# for downloading from Google Drive
try:
    import gdown
except Exception:
    gdown = None

UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
MODEL_FILENAME = "mrcnn_food_detection.h5"
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1sVzG18PNoUrwxctxIMIoQL79bCWjhnSY&export=download"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# ---- Simple in-memory "user DB" (demo only) ----
USERS_FILE = "users.json"
if os.path.exists(USERS_FILE):
    with open(USERS_FILE, "r") as f:
        USERS = json.load(f)
else:
    USERS = {}

def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump(USERS, f)

# ---- Model loader (lazy) ----
_model = None
def download_model_if_needed():
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if os.path.exists(model_path):
        return model_path
    if gdown is None:
        raise RuntimeError("gdown not installed. pip install gdown")
    print("Downloading model from Google Drive...")
    gdown.download(GDRIVE_FILE_URL, model_path, quiet=False)
    return model_path

def load_food_model():
    global _model
    if _model is None:
        model_path = download_model_if_needed()
        # If model was trained with custom optimizer, we set compile=False
        _model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded.")
    return _model

# ---- Auth endpoints (VERY BASIC -- for demo only) ----
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json or {}
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")
    if not email or not password or not username:
        return jsonify({"error":"missing fields"}), 400
    if email in USERS:
        return jsonify({"error":"user exists"}), 400
    USERS[email] = {"username": username, "password": password}
    save_users()
    return jsonify({"message":"registered"}), 200

@app.route("/api/login", methods=["POST"])
def login():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    user = USERS.get(email)
    if user and user.get("password") == password:
        # Demo token (NOT secure). In prod use real JWT.
        token = f"demo-token-for-{email}"
        return jsonify({"token": token, "username": user.get("username")})
    return jsonify({"error":"invalid credentials"}), 401

# ---- Upload kotlin file endpoint ----
@app.route("/api/upload_kotlin", methods=["POST"])
def upload_kotlin():
    # Expect multipart/form-data with 'file'
    if 'file' not in request.files:
        return jsonify({"error":"no file part"}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({"error":"no selected file"}), 400
    filename = secure_filename(f.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    f.save(save_path)
    return jsonify({"message":"file saved", "path": save_path})

# ---- Food detection predict endpoint (accepts image) ----
@app.route("/api/predict_food", methods=["POST"])
def predict_food():
    """
    Receive an image file (form field 'image'), run model.predict and return dummy response.
    NOTE: The actual Mask R-CNN inference code depends on how the model expects input.
    Here we assume the model accepts a (1, H, W, 3) float32 tensor and returns something.
    You need to adapt this block to match the model you have.
    """
    if 'image' not in request.files:
        return jsonify({"error":"no image"}), 400
    imgfile = request.files['image']
    filename = secure_filename(imgfile.filename)
    saved = os.path.join(UPLOAD_DIR, filename)
    imgfile.save(saved)

    model = load_food_model()

    # --- simplified preprocessing: load image, resize, normalize ---
    from PIL import Image
    img = Image.open(saved).convert("RGB")
    img_resized = img.resize((224,224))  # adapt to model input
    arr = np.array(img_resized).astype(np.float32)/255.0
    x = np.expand_dims(arr, axis=0)

    try:
        preds = model.predict(x)
        # adapt how you interpret preds depending on model
        # We'll return a placeholder
        return jsonify({"success": True, "pred_shape": str(np.array(preds).shape)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Serve uploaded files (for dev) ----
@app.route("/uploads/<path:filename>")
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_DIR, filename)

if __name__ == "__main__":
    # optionally download model in background
    def bg_download():
        try:
            download_model_if_needed()
        except Exception as e:
            print("Model download failed:", e)
    t = threading.Thread(target=bg_download, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=True)
