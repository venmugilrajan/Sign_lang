"""
Flask backend for real-time sign language translation.
Serves both ASL and ISL models (supports both fast Landmark MLP classifiers
and MobileNetV2 pixel CNNs), along with live spellchecking endpoints.
"""

import os
import io
import base64
import json
import pickle
import torch
import numpy as np
from PIL import Image, ImageEnhance
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from torchvision import transforms
from spellchecker import SpellChecker

from model import load_model, NORMALIZE_MEAN, NORMALIZE_STD, INPUT_SIZE

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEB_DIR = os.path.join(BASE_DIR, "web")

ASL_CNN_PATH = os.path.join(MODELS_DIR, "asl_model.pth")
ISL_CNN_PATH = os.path.join(MODELS_DIR, "isl_model.pth")

ASL_LANDMARK_PATH = os.path.join(MODELS_DIR, "asl_landmark_model.pkl")
ISL_LANDMARK_PATH = os.path.join(MODELS_DIR, "isl_landmark_model.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
CORS(app)

spell = SpellChecker()

# ─── Transforms & Enhancements ─────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
])


def enhance_for_inference(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.3)
    image = ImageEnhance.Brightness(image).enhance(1.1)
    return image


# ─── Registry ──────────────────────────────────────────────────────────────────
cnn_models = {}
cnn_labels = {}
landmark_models = {}
landmark_labels = {}


def load_all_models():
    global cnn_models, cnn_labels, landmark_models, landmark_labels

    # 1. Load Landmark Classifiers (.pkl)
    for mode, path in [("asl", ASL_LANDMARK_PATH), ("isl", ISL_LANDMARK_PATH)]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                landmark_models[mode] = data["model"]
                landmark_labels[mode] = data["classes"]
                print(f"[+] Landmark Classifier ready ({mode.upper()}) | {len(data['classes'])} classes | Test acc: {data.get('test_acc', 0):.1f}%")

    # 2. Load Legacy Pixel CNNs (.pth)
    for mode, path in [("asl", ASL_CNN_PATH), ("isl", ISL_CNN_PATH)]:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE)
            cnn_classes = ckpt["class_names"]
            cnn_num_classes = ckpt["num_classes"]
            m = load_model(path, cnn_num_classes, DEVICE)
            cnn_models[mode] = m
            cnn_labels[mode] = cnn_classes
            print(f"[+] Pixel CNN ready ({mode.upper()}) | {cnn_num_classes} classes")


def preprocess_image(image_b64: str) -> torch.Tensor:
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = enhance_for_inference(image)
    tensor = inference_transform(image).unsqueeze(0)
    return tensor.to(DEVICE)


# ─── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "landmark_models": list(landmark_models.keys()),
        "cnn_models": list(cnn_models.keys()),
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    mode = request.args.get("mode", "asl").lower()
    if mode in landmark_labels:
        return jsonify({"mode": mode, "classes": landmark_labels[mode], "count": len(landmark_labels[mode])})
    elif mode in cnn_labels:
        return jsonify({"mode": mode, "classes": cnn_labels[mode], "count": len(cnn_labels[mode])})
    return jsonify({"error": f"Model '{mode}' not loaded"}), 404


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    mode = data.get("mode", "asl").lower()
    features = data.get("landmarks", [])

    if mode not in landmark_models:
        return jsonify({"error": f"Landmark model '{mode}' not loaded"}), 404

    if not features or len(features) not in (78, 156):
        return jsonify({
            "error": f"Invalid landmark features vector. Expected 78 (single-hand) "
                     f"or 156 (dual-hand) floats, got {len(features) if features else 0}."
        }), 400

    clf = landmark_models[mode]
    expected = getattr(clf, "n_features_in_", len(features))
    if len(features) != expected:
        return jsonify({
            "error": f"Model '{mode}' expects {expected} features, got {len(features)}."
        }), 400

    try:
        # classes_ holds LabelEncoder integers; landmark_labels holds the real names.
        labels = [str(c) for c in landmark_labels[mode]]
        probs = clf.predict_proba([features])[0]
        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        top_conf = float(probs[top_idx])

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [{"label": labels[i], "confidence": round(float(probs[i]), 4)} for i in top3_indices]

        return jsonify({
            "mode": mode,
            "letter": top_label,
            "confidence": top_conf,
            "top3": top3
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    mode = data.get("mode", "asl").lower()
    img_b64 = data.get("image", "")

    if mode not in cnn_models:
        return jsonify({"error": f"Model '{mode}' not loaded"}), 404

    if not img_b64:
        return jsonify({"error": "No image data provided"}), 400

    try:
        tensor = preprocess_image(img_b64)
        model = cnn_models[mode]
        labels = cnn_labels[mode]

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top3_probs, top3_idxs = torch.topk(probs, k=min(3, len(labels)))
        top3 = [
            {"label": labels[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top3_probs, top3_idxs)
        ]

        return jsonify({
            "mode": mode,
            "letter": top3[0]["label"],
            "confidence": top3[0]["confidence"],
            "top3": top3,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


from word_buffer_engine import TwoTierSpellCorrector

two_tier_corrector = TwoTierSpellCorrector()


@app.route("/spellcheck", methods=["POST"])
def spellcheck():
    data = request.get_json() or {}
    raw_word = data.get("word", "").strip()
    confidences = data.get("confidences", None)
    if not raw_word:
        return jsonify({"original": "", "corrected": "", "suggestions": []})

    corrected, is_corr, suggestions = two_tier_corrector.correct(raw_word, confidences)
    return jsonify({
        "original": raw_word.upper(),
        "corrected": corrected,
        "is_corrected": is_corr,
        "suggestions": suggestions
    })


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Sign Language Translation API")
    print("=" * 60)
    load_all_models()
    print(f"\n[→] Server starting on http://localhost:5000")
    print(f"[→] Endpoints: GET /health | GET /classes | POST /predict_landmarks | POST /predict | POST /spellcheck\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
