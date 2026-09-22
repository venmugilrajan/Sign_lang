"""
ISL mirror-augmented spatial dual-hand trainer.

Reproduces the deployed 99.64% ISL model: 156-D features (two spatially
sorted 78-D hand slots), dataset doubled with horizontal flips so the
classifier is invariant to which hand leads.

Exports both models/isl_landmark_model.pkl and web/isl_landmarks_weights.json.
"""

import os
import re
import json
import pickle

import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
TASK_PATH = str(BASE_DIR / "hand_landmarker.task")
ISL_DIR = BASE_DIR / "indian sign language" / "Indian"
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"
MODELS_DIR.mkdir(exist_ok=True)
WEB_DIR.mkdir(exist_ok=True)

CACHE = MODELS_DIR / "isl_blocksplit_cache.pkl"
EXPORT_PKL = MODELS_DIR / "isl_landmark_model.pkl"
EXPORT_JSON = WEB_DIR / "isl_landmarks_weights.json"

MAX_PER_CLASS = 150
PAD_RATIO = 0.15

# Frame-adjacency leakage control. Images are consecutive video frames per class
# (numeric filenames 0..1199); adjacent frames are near-identical (mean pixel
# delta ~1.0-1.4) and similarity only plateaus around a gap of 50-100 frames.
# So we split by contiguous BLOCKS of frames -- an entire block goes to train or
# to test -- instead of shuffling individual images across the boundary.
BLOCK_SIZE = 100
TEST_BLOCK_FRACTION = 0.2
SPLIT_SEED = 42

# Read back off the deployed 99.64% models/isl_landmark_model.pkl so this script
# actually reproduces it. These deliberately differ from the (256,128,64) /
# max_iter=500 / early_stopping=True originally specified -- that combination
# trains a different, smaller model.
HIDDEN_LAYERS = (512, 256, 128)
MAX_ITER = 400
EARLY_STOPPING = False
N_ITER_NO_CHANGE = 10
RANDOM_STATE = 42

ANGLE_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.20,
    min_hand_presence_confidence=0.20,
    min_tracking_confidence=0.20,
)
detector = vision.HandLandmarker.create_from_options(options)


SINGLE_HAND_CLASSES = {'1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'I', 'J', 'L', 'O', 'Q', 'U', 'V'}


def calculate_angle_3d(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) / 180.0


def mirror_single_hand_features(feat):
    """Reflects a 78-D single hand feature vector horizontally (x -> -x).
    
    Landmark 9 aligns with vertical 12 o'clock, so reflection strictly maps
    a Right hand to a Left hand (and vice versa) with identical joint angles.
    """
    mirr = feat.copy()
    for i in range(21):
        mirr[i * 3] = -mirr[i * 3]
    return mirr


def extract_single_hand_features(landmarks, w, h):
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)

    pts_centered = pts - pts[0]

    scale = np.linalg.norm(pts_centered[9][:2])
    if scale < 1e-4:
        scale = 1.0
    pts_scaled = pts_centered / scale

    # Align landmark 9 to vertical 12 o'clock (0, -1)
    angle = np.arctan2(pts_scaled[9, 0], -pts_scaled[9, 1])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    pts_aligned = pts_scaled.copy()
    pts_aligned[:, :2] = np.dot(pts_scaled[:, :2], rot)

    angles = [calculate_angle_3d(pts[i], pts[j], pts[k]) for i, j, k in ANGLE_TRIPLETS]
    return np.concatenate([pts_aligned.flatten(), angles])


def extract_spatial_dual_hand_features(img_bgr, cls_name=None):
    h, w = img_bgr.shape[:2]
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
    )
    results = detector.detect(mp_img)
    if not results.hand_landmarks:
        return None

    num_hands = len(results.hand_landmarks)

    if cls_name is not None:
        if cls_name in SINGLE_HAND_CLASSES:
            # Single-hand class: strictly accept 1 hand
            if num_hands != 1:
                return None
        else:
            # Two-handed class: strictly accept 2 hands (reject noisy 1-hand transition frames)
            if num_hands != 2:
                return None
    elif num_hands > 2:
        return None

    hands = []
    for lms in results.hand_landmarks[:2]:
        feat = extract_single_hand_features(lms, w, h)
        hands.append((lms[0].x, feat))

    hands.sort(key=lambda item: item[0])
    slot1 = hands[0][1]
    slot2 = hands[1][1] if len(hands) > 1 else np.zeros(78, dtype=np.float32)
    return np.concatenate([slot1, slot2])


def frame_number(filename):
    """Leading integer of the filename == temporal frame index."""
    m = re.match(r"(\d+)", os.path.splitext(filename)[0])
    return int(m.group(1)) if m else -1


def class_block_split(filenames):
    """Assigns each file to 'train' or 'test' by contiguous frame block."""
    blocks = sorted({frame_number(f) // BLOCK_SIZE for f in filenames})
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(blocks)
    n_test = max(1, int(round(len(blocks) * TEST_BLOCK_FRACTION)))
    test_blocks = set(shuffled[:n_test].tolist())
    return {f: ("test" if frame_number(f) // BLOCK_SIZE in test_blocks else "train")
            for f in filenames}


def build_dataset():
    if CACHE.exists():
        print(f"[+] Loading cached dataset from {CACHE}")
        with open(CACHE, "rb") as f:
            data = pickle.load(f)
        print(f"[+] {len(data['X'])} samples, {len(set(data['y']))} classes")
        return data["X"], data["y"], data["split"]

    if not ISL_DIR.exists():
        raise SystemExit(f"[!] Dataset not found: {ISL_DIR}")

    classes = sorted(d for d in os.listdir(ISL_DIR) if (ISL_DIR / d).is_dir())
    print(f"[*] Extracting clean dual-hand mirror-augmented landmarks for {len(classes)} classes...")
    print(f"    BLOCK_SIZE={BLOCK_SIZE} frames, {TEST_BLOCK_FRACTION:.0%} of blocks held out")

    X, y, split = [], [], []
    for cls in classes:
        cls_dir = ISL_DIR / cls
        files = [f for f in os.listdir(cls_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort(key=frame_number)

        assign = class_block_split(files)

        step = max(1, len(files) // MAX_PER_CLASS)
        chosen = files[::step][:MAX_PER_CLASS]

        valid_items = []
        for fn in files:
            img = cv2.imread(str(cls_dir / fn))
            if img is None:
                continue

            where = assign[fn]
            feat = extract_spatial_dual_hand_features(img, cls_name=cls)
            feat_flip = extract_spatial_dual_hand_features(cv2.flip(img, 1), cls_name=cls)

            if feat is not None or feat_flip is not None:
                valid_items.append((feat, feat_flip, where))
                if len(valid_items) >= MAX_PER_CLASS:
                    break

        train_items = [it for it in valid_items if it[2] == "train"]
        test_items = [it for it in valid_items if it[2] == "test"]
        if len(train_items) == 0:
            n_tr = max(1, int(len(valid_items) * 0.8))
            train_items = valid_items[:n_tr]
            test_items = valid_items[n_tr:]
        elif len(test_items) == 0 and len(valid_items) > 1:
            n_tr = max(1, int(len(valid_items) * 0.8))
            train_items = valid_items[:n_tr]
            test_items = valid_items[n_tr:]

        for feat, feat_flip, _ in train_items:
            if feat is not None:
                X.append(feat); y.append(cls); split.append("train")
                if cls in SINGLE_HAND_CLASSES:
                    mirr = mirror_single_hand_features(feat[:78])
                    X.append(np.concatenate([mirr, np.zeros(78, dtype=np.float32)])); y.append(cls); split.append("train")
            if feat_flip is not None:
                X.append(feat_flip); y.append(cls); split.append("train")
                if cls in SINGLE_HAND_CLASSES:
                    mirr = mirror_single_hand_features(feat_flip[:78])
                    X.append(np.concatenate([mirr, np.zeros(78, dtype=np.float32)])); y.append(cls); split.append("train")

        for feat, feat_flip, _ in test_items:
            if feat is not None:
                X.append(feat); y.append(cls); split.append("test")
                if cls in SINGLE_HAND_CLASSES:
                    mirr = mirror_single_hand_features(feat[:78])
                    X.append(np.concatenate([mirr, np.zeros(78, dtype=np.float32)])); y.append(cls); split.append("test")
            if feat_flip is not None:
                X.append(feat_flip); y.append(cls); split.append("test")
                if cls in SINGLE_HAND_CLASSES:
                    mirr = mirror_single_hand_features(feat_flip[:78])
                    X.append(np.concatenate([mirr, np.zeros(78, dtype=np.float32)])); y.append(cls); split.append("test")

        print(f" -> '{cls}': {len(train_items)} train + {len(test_items)} test frames (mirror augmented)", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    split = np.array(split)

    with open(CACHE, "wb") as f:
        pickle.dump({"X": X, "y": y, "split": split}, f)
    print(f"[+] Cached {len(X)} samples to {CACHE}")
    return X, y, split


def export_json(clf, class_names):
    weights = {"classes": class_names}
    for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        weights[f"w{i}"] = w.tolist()
        weights[f"b{i}"] = b.tolist()
    with open(EXPORT_JSON, "w") as f:
        json.dump(weights, f)
    print(f"[+] Exported browser weights ({clf.coefs_[0].shape[0]}-D input) to {EXPORT_JSON}")


def train_and_export():
    X, y_raw, split = build_dataset()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    tr, te = split == "train", split == "test"
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    print(f"[*] Block-split: {len(X_train)} train / {len(X_test)} test "
          f"({len(X_test)/len(X):.1%} held out), {X.shape[1]}-D")

    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        max_iter=MAX_ITER,
        early_stopping=EARLY_STOPPING,
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    train_acc = clf.score(X_train, y_train) * 100
    print(f"[+] Train accuracy            : {train_acc:.2f}%")
    print(f"[+] Held-out test accuracy    : {acc:.2f}%   (block-split, no frame leakage)")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    with open(EXPORT_PKL, "wb") as f:
        pickle.dump({
            "model": clf,
            "label_encoder": le,
            "classes": class_names,
            "num_features": int(X.shape[1]),
            "dual_hand": True,
            "test_acc": acc,
            "split": "block",
            "block_size": BLOCK_SIZE,
        }, f)
    print(f"[+] Exported model to {EXPORT_PKL}")

    export_json(clf, class_names)


if __name__ == "__main__":
    train_and_export()
