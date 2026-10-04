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
import threading
import queue
from spellchecker import SpellChecker

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import load_model, NORMALIZE_MEAN, NORMALIZE_STD, INPUT_SIZE

# ─── Offline Text-to-Speech (Non-blocking Queue) ────────────────────────────────
_server_tts_queue = queue.Queue()

def _server_tts_worker():
    engine = None
    if pyttsx3:
        try:
            engine = pyttsx3.init()
        except Exception as e:
            print(f"[TTS] Warning: pyttsx3 init failed: {e}")
    while True:
        text = _server_tts_queue.get()
        if text is None:
            break
        if engine and text:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Speech error on '{text}': {e}")
        _server_tts_queue.task_done()

_server_tts_thread = threading.Thread(target=_server_tts_worker, daemon=True)
_server_tts_thread.start()

def server_speak(text: str):
    if text:
        _server_tts_queue.put(text)

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


# ─── Static Image Feature Extraction for Upload Studio ───────────────────────────
_static_detector = None

def get_static_detector():
    global _static_detector
    if _static_detector is None:
        task_path = os.path.join(BASE_DIR, "hand_landmarker.task")
        if os.path.exists(task_path):
            base_options = python.BaseOptions(model_asset_path=task_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.25,
                min_hand_presence_confidence=0.25
            )
            _static_detector = vision.HandLandmarker.create_from_options(options)
            print("[+] Static HandLandmarker initialized for Upload Studio")
    return _static_detector


def calculate_angle_3d_np(a, b, c):
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosine_angle = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle)) / 180.0


def extract_single_hand_features(landmarks, wp, hp, handedness, mode="asl"):
    pts = np.array([[lm.x * wp, lm.y * hp, lm.z * wp] for lm in landmarks], dtype=np.float32)

    # In single-hand ASL, mirror Left hand to Right hand coordinate space
    if mode == "asl" and handedness == "Left":
        pts[:, 0] = pts[0, 0] - (pts[:, 0] - pts[0, 0])

    wrist = pts[0]
    pts_centered = pts - wrist

    middle_mcp = pts_centered[9]
    scale = np.linalg.norm(middle_mcp[:2])
    if scale < 1e-4:
        scale = 1.0
    pts_scaled = pts_centered / scale

    angle = np.arctan2(pts_scaled[9, 0], -pts_scaled[9, 1])
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    if mode == "isl":
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    else:
        rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    pts_aligned = pts_scaled.copy()
    pts_aligned[:, :2] = np.dot(pts_scaled[:, :2], rot_matrix)

    angles = [
        calculate_angle_3d_np(pts[0], pts[1], pts[2]),
        calculate_angle_3d_np(pts[1], pts[2], pts[3]),
        calculate_angle_3d_np(pts[2], pts[3], pts[4]),
        calculate_angle_3d_np(pts[0], pts[5], pts[6]),
        calculate_angle_3d_np(pts[5], pts[6], pts[7]),
        calculate_angle_3d_np(pts[6], pts[7], pts[8]),
        calculate_angle_3d_np(pts[0], pts[9], pts[10]),
        calculate_angle_3d_np(pts[9], pts[10], pts[11]),
        calculate_angle_3d_np(pts[10], pts[11], pts[12]),
        calculate_angle_3d_np(pts[0], pts[13], pts[14]),
        calculate_angle_3d_np(pts[13], pts[14], pts[15]),
        calculate_angle_3d_np(pts[14], pts[15], pts[16]),
        calculate_angle_3d_np(pts[0], pts[17], pts[18]),
        calculate_angle_3d_np(pts[17], pts[18], pts[19]),
        calculate_angle_3d_np(pts[18], pts[19], pts[20]),
    ]
    return np.concatenate([pts_aligned.flatten(), angles])


@app.route("/predict_image", methods=["POST"])
def predict_image():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    mode = data.get("mode", "asl").lower()
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image data"}), 400

    if mode not in landmark_models:
        return jsonify({"error": f"Model '{mode}' not loaded"}), 404

    try:
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        raw_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"error": "Failed to decode image"}), 400

        orig_h, orig_w = img_bgr.shape[:2]
        pad = int(max(orig_h, orig_w) * 0.25)
        img_padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        hp, wp = img_padded.shape[:2]

        detector = get_static_detector()
        if not detector:
            return jsonify({"error": "Landmark detector not available"}), 500

        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        if not results.hand_landmarks:
            # Fallback to CNN if landmarks cannot be found on image
            if mode in cnn_models:
                tensor = preprocess_image(img_b64)
                with torch.no_grad():
                    logits = cnn_models[mode](tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                top3_probs, top3_idxs = torch.topk(probs, k=min(3, len(cnn_labels[mode])))
                labels = cnn_labels[mode]
                top3 = [{"label": labels[idx.item()], "confidence": round(prob.item(), 4)} for prob, idx in zip(top3_probs, top3_idxs)]
                return jsonify({
                    "mode": mode,
                    "letter": top3[0]["label"],
                    "confidence": top3[0]["confidence"],
                    "top3": top3,
                    "landmarks_detected": False,
                    "hands": []
                })
            return jsonify({"error": "No hand detected in image", "landmarks_detected": False, "hands": []}), 200

        # Convert detected landmarks back to original image coordinate space for drawing
        drawn_hands = []
        hands_data = []
        for idx, lms in enumerate(results.hand_landmarks):
            h_name = "Right"
            if results.handedness and idx < len(results.handedness):
                h_name = results.handedness[idx][0].category_name

            norm_pts = []
            for lm in lms:
                px = lm.x * wp - pad
                py = lm.y * hp - pad
                norm_pts.append({
                    "x": float(np.clip(px / orig_w, 0.0, 1.0)),
                    "y": float(np.clip(py / orig_h, 0.0, 1.0)),
                    "z": float(lm.z)
                })
            drawn_hands.append({"landmarks": norm_pts, "handedness": h_name})

            feat = extract_single_hand_features(lms, wp, hp, h_name, mode=mode)
            hands_data.append((lms[0].x, feat))

        # Build feature vector matching model requirements
        clf = landmark_models[mode]
        labels = [str(c) for c in landmark_labels[mode]]

        if mode == "isl" and getattr(clf, "n_features_in_", 78) == 156:
            hands_data.sort(key=lambda x: x[0])
            slot1 = hands_data[0][1]
            slot2 = hands_data[1][1] if len(hands_data) > 1 else np.zeros(78, dtype=np.float32)
            features = np.concatenate([slot1, slot2]).tolist()
        else:
            features = hands_data[0][1].tolist()

        probs = np.array(clf.predict_proba([features])[0], dtype=np.float32)

        # ISL hand count gating
        if mode == "isl" and len(features) == 156:
            ISL_SINGLE_HAND = {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'I', 'J', 'L', 'O', 'Q', 'U', 'V'}
            ISL_TWO_HAND = {'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z'}
            slot2 = features[78:]
            num_hands = 1 if all(abs(v) < 1e-4 for v in slot2) else 2
            if num_hands == 1:
                for i, c in enumerate(labels):
                    if c not in ISL_SINGLE_HAND:
                        probs[i] = 0.0
            else:
                for i, c in enumerate(labels):
                    if c not in ISL_TWO_HAND:
                        probs[i] = 0.0
            p_sum = float(np.sum(probs))
            if p_sum > 0:
                probs = probs / p_sum

        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        top_conf = float(probs[top_idx])

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [{"label": labels[i], "confidence": round(float(probs[i]), 4)} for i in top3_indices if probs[i] > 0.001]
        if not top3:
            top3 = [{"label": top_label, "confidence": round(top_conf, 4)}]

        print(f"[/predict_image] {mode.upper()} -> {top_label} ({top_conf:.2%})", flush=True)

        return jsonify({
            "mode": mode,
            "letter": top_label,
            "confidence": top_conf,
            "top3": top3,
            "landmarks_detected": True,
            "hands": drawn_hands
        })
    except Exception as e:
        print(f"[!] Error in /predict_image: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


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
        probs = np.array(clf.predict_proba([features])[0], dtype=np.float32)

        # ── ISL Hand-Count Constrained Gating ─────────────────────────────────
        # Prevents 1-hand gestures from falsely predicting 2-hand signs like 'R' or 'A',
        # and guarantees single-hand signs (5, 1, 2, C, L, etc.) fire with maximum precision.
        if mode == "isl" and len(features) == 156:
            ISL_SINGLE_HAND = {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'I', 'J', 'L', 'O', 'Q', 'U', 'V'}
            ISL_TWO_HAND = {'A', 'B', 'D', 'E', 'F', 'G', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z'}

            slot2 = features[78:]
            num_hands = 1 if all(abs(v) < 1e-4 for v in slot2) else 2

            if num_hands == 1:
                for i, c in enumerate(labels):
                    if c not in ISL_SINGLE_HAND:
                        probs[i] = 0.0
            else:
                for i, c in enumerate(labels):
                    if c not in ISL_TWO_HAND:
                        probs[i] = 0.0

            p_sum = float(np.sum(probs))
            if p_sum > 0:
                probs = probs / p_sum

        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        top_conf = float(probs[top_idx])

        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [{"label": labels[i], "confidence": round(float(probs[i]), 4)} for i in top3_indices if probs[i] > 0.001]
        if not top3:
            top3 = [{"label": top_label, "confidence": round(top_conf, 4)}]

        print(f"[/predict_landmarks] {mode.upper()} -> {top_label} ({top_conf:.2%}) | Top3: {[(t['label'], t['confidence']) for t in top3]}", flush=True)

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


@app.route("/speak", methods=["POST"])
def speak_route():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"status": "error", "message": "No text provided"}), 400
    server_speak(text)
    return jsonify({"status": "ok", "spoken": text})


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Sign Language Translation API")
    print("=" * 60)
    load_all_models()
    print(f"\n[->] Server starting on http://localhost:5000")
    print(f"[->] Endpoints: GET /health | GET /classes | POST /predict_landmarks | POST /predict | POST /spellcheck | POST /speak\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
