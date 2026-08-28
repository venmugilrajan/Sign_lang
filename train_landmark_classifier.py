"""
Sign Language Landmark Classifier Training Pipeline (High Accuracy Edition)

Key Improvements:
1. Aspect-Ratio Corrected Normalized Features (Independent of 1:1 square image vs 16:9 widescreen camera)
2. Invariant Hand Angle / Rotation Alignment (Aligns wrist-to-MCP axis to vertical)
3. Hand Joint Angles & Distance Features (Computes 15 key finger flexion & knuckle angles)
4. Border Padding Support for tightly cropped fist gestures (E, M, N, O, S, T, X)
5. Left/Right Handedness Mirror Normalization
6. 26-Letter ASL Alphabet Classifier + Extended Sign Classifiers
"""

import os
import sys
import cv2
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── File Paths & Config ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WEB_DIR = BASE_DIR / "web"
MODELS_DIR.mkdir(exist_ok=True)
WEB_DIR.mkdir(exist_ok=True)

MODEL_TASK_PATH = str(BASE_DIR / "hand_landmarker.task")
ASL_DIR = BASE_DIR / "asl-alphabet-train"
ISL_DIR = BASE_DIR / "indian sign language" / "Indian"
DS2_DIR = BASE_DIR / "sign language dataset 2" / "Gesture Image Data"

CACHE_ASL = MODELS_DIR / "asl_landmark_cache_v3.pkl"
CACHE_ISL = MODELS_DIR / "isl_landmark_cache_v3.pkl"

EXPORT_ASL_PKL = MODELS_DIR / "asl_landmark_model.pkl"
EXPORT_ISL_PKL = MODELS_DIR / "isl_landmark_model.pkl"

EXPORT_ASL_JSON = WEB_DIR / "asl_landmarks_weights.json"
EXPORT_ISL_JSON = WEB_DIR / "isl_landmarks_weights.json"

CM_ASL_PNG = BASE_DIR / "asl_confusion_matrix.png"
CM_ISL_PNG = BASE_DIR / "isl_confusion_matrix.png"

# Initialize MediaPipe Tasks HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    min_hand_detection_confidence=0.3,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# 26 Standard Alphabet letters + space + del
ASL_ALPHABET_CLASSES = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['del', 'space']


def calculate_angle_3d(a, b, c):
    """Calculates angle in degrees at vertex b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle)) / 180.0  # Normalized to [0, 1]


def extract_features_from_image(img_bgr: np.ndarray):
    """
    Robust Feature Extractor:
    1. Handles tightly cropped images with automatic contextual border padding.
    2. Converts normalized (0..1) coordinates to true pixel coordinates (w, h).
    3. Normalizes Left hand into Right hand geometric space.
    4. Centers on wrist (landmark 0) and aligns rotation along Y-axis.
    5. Appends 15 finger joint flexion angles (78 total features).
    """
    if img_bgr is None or img_bgr.size == 0:
        return None

    h, w = img_bgr.shape[:2]
    # Add border padding so closed-fist gestures with tight crops are easily localized by MediaPipe
    pad = int(max(h, w) * 0.25)
    img_padded = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    hp, wp = img_padded.shape[:2]

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)

    if not results.hand_landmarks:
        return None

    landmarks = results.hand_landmarks[0]
    handedness = results.handedness[0][0].category_name if results.handedness else "Right"

    # Step 1: Pixel coordinates (aspect ratio invariant)
    pts = np.array([[lm.x * wp, lm.y * hp, lm.z * wp] for lm in landmarks], dtype=np.float32)

    # Step 2: Left-to-Right Normalization
    if handedness == "Left":
        pts[:, 0] = pts[0, 0] - (pts[:, 0] - pts[0, 0])

    # Step 3: Center on wrist
    wrist = pts[0]
    pts_centered = pts - wrist

    # Step 4: Palm scale
    middle_mcp = pts_centered[9]
    scale = np.linalg.norm(middle_mcp[:2])
    if scale < 1e-4:
        scale = 1.0

    pts_scaled = pts_centered / scale

    # Step 5: 2D Palm Rotation Alignment
    angle = np.arctan2(pts_scaled[9, 0], -pts_scaled[9, 1])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    pts_aligned = pts_scaled.copy()
    pts_aligned[:, :2] = np.dot(pts_scaled[:, :2], rot_matrix)

    # Step 6: Finger Joint Angles (flexion & abduction)
    angles = [
        calculate_angle_3d(pts[0], pts[1], pts[2]),   # Thumb CMC
        calculate_angle_3d(pts[1], pts[2], pts[3]),   # Thumb MCP
        calculate_angle_3d(pts[2], pts[3], pts[4]),   # Thumb IP
        calculate_angle_3d(pts[0], pts[5], pts[6]),   # Index MCP
        calculate_angle_3d(pts[5], pts[6], pts[7]),   # Index PIP
        calculate_angle_3d(pts[6], pts[7], pts[8]),   # Index DIP
        calculate_angle_3d(pts[0], pts[9], pts[10]),  # Middle MCP
        calculate_angle_3d(pts[9], pts[10], pts[11]), # Middle PIP
        calculate_angle_3d(pts[10], pts[11], pts[12]),# Middle DIP
        calculate_angle_3d(pts[0], pts[13], pts[14]), # Ring MCP
        calculate_angle_3d(pts[13], pts[14], pts[15]),# Ring PIP
        calculate_angle_3d(pts[14], pts[15], pts[16]),# Ring DIP
        calculate_angle_3d(pts[0], pts[17], pts[18]), # Pinky MCP
        calculate_angle_3d(pts[17], pts[18], pts[19]),# Pinky PIP
        calculate_angle_3d(pts[18], pts[19], pts[20]),# Pinky DIP
    ]

    features = np.concatenate([pts_aligned.flatten(), angles])
    return features


def collect_landmark_dataset(dataset_dirs, cache_path, target_classes=None, max_per_class=300, name="Dataset"):
    """Scans dataset directories, extracts robust invariant landmarks, and caches dataset to disk."""
    if cache_path.exists():
        print(f"[+] Loading cached robust landmark dataset from: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            print(f"[+] Loaded {len(data['X'])} samples across {len(data['classes'])} classes.")
            return data["X"], data["y"], data["classes"]

    print(f"\n[*] Extracting robust landmarks for {name} from {dataset_dirs}...")
    X, y = [], []

    if not isinstance(dataset_dirs, (list, tuple)):
        dataset_dirs = [dataset_dirs]

    all_classes = set()
    for d in dataset_dirs:
        p = Path(d)
        if p.exists():
            for sub in p.iterdir():
                if sub.is_dir() and not sub.name.startswith("."):
                    if target_classes is None or sub.name in target_classes:
                        all_classes.add(sub.name)

    sorted_classes = sorted(list(all_classes))

    for c in sorted_classes:
        collected_for_c = 0
        print(f" -> Processing class '{c}'...", end=" ", flush=True)

        for d in dataset_dirs:
            class_folder = Path(d) / c
            if not class_folder.exists():
                continue

            img_files = [f for f in class_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            for img_p in img_files:
                if max_per_class is not None and collected_for_c >= max_per_class:
                    break

                img = cv2.imread(str(img_p))
                feats = extract_features_from_image(img)
                if feats is not None:
                    X.append(feats)
                    y.append(c)
                    collected_for_c += 1

            if max_per_class is not None and collected_for_c >= max_per_class:
                break

        print(f"Extracted {collected_for_c} samples.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    unique_classes = sorted(list(set(y)))

    print(f"[+] Extraction complete: {len(X)} samples across {len(unique_classes)} classes.")

    with open(cache_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "classes": unique_classes}, f)
    print(f"[+] Saved robust landmark cache to: {cache_path}")

    return X, y, unique_classes


def plot_and_save_confusion_matrix(y_true, y_pred, classes, save_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(14, 12), facecolor="#0e0e1a")
    ax = plt.subplot(111, facecolor="#0e0e1a")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, linecolor="#1a1a2e", ax=ax,
        cbar_kws={"label": "Sample Count"}
    )

    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color="white", fontsize=12)
    ax.set_ylabel("True Label", color="white", fontsize=12)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0e0e1a")
    plt.close()


def train_and_evaluate(X, y_raw, classes, model_name, pkl_export, json_export, cm_path):
    print(f"\n{'='*70}")
    print(f"  Training & Evaluating {model_name} Robust Classifier")
    print(f"{'='*70}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    val_rel_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_rel_size, random_state=42, stratify=y_train_val
    )

    print(f"[*] Dataset Splits:")
    print(f"   * Train set      : {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   * Validation set : {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   * Held-out Test  : {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # High-Performance Neural Network Classifier
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )

    print("\n[*] Training Robust Neural Network...")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train) * 100
    val_acc = clf.score(X_val, y_val) * 100
    test_preds_int = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds_int) * 100

    y_test_str = le.inverse_transform(y_test)
    test_preds_str = le.inverse_transform(test_preds_int)

    print(f"\n[+] Results for {model_name}:")
    print(f"   * Training Accuracy   : {train_acc:.2f}%")
    print(f"   * Validation Accuracy : {val_acc:.2f}%")
    print(f"   * Held-out TEST Acc   : {test_acc:.2f}%  (REAL-WORLD TEST ACCURACY)")

    print("\n[+] Detailed Classification Report (Held-out Test Set):")
    print(classification_report(y_test_str, test_preds_str, digits=3))

    plot_and_save_confusion_matrix(
        y_test_str, test_preds_str, class_names, cm_path,
        f"{model_name} Confusion Matrix (Held-out Test Set - Acc: {test_acc:.1f}%)"
    )

    with open(pkl_export, "wb") as f:
        pickle.dump({
            "model": clf,
            "label_encoder": le,
            "classes": class_names,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "num_features": X.shape[1],
        }, f)
    print(f"[+] Exported Python model package to: {pkl_export}")

    weights_json = {
        "classes": class_names,
        "w0": clf.coefs_[0].tolist(),
        "b0": clf.intercepts_[0].tolist(),
        "w1": clf.coefs_[1].tolist(),
        "b1": clf.intercepts_[1].tolist(),
        "w2": clf.coefs_[2].tolist(),
        "b2": clf.intercepts_[2].tolist(),
        "w3": clf.coefs_[3].tolist(),
        "b3": clf.intercepts_[3].tolist(),
    }
    with open(json_export, "w") as f:
        json.dump(weights_json, f)
    print(f"[+] Exported browser JSON weights to: {json_export}")

    return test_acc


def main():
    print("=" * 70)
    print("  Sign Language Landmark Classifier Training Pipeline (Balanced V3 Edition)")
    print("=" * 70)

    asl_sources = [ASL_DIR]
    if DS2_DIR.exists():
        asl_sources.append(DS2_DIR)

    X_asl, y_asl, classes_asl = collect_landmark_dataset(
        dataset_dirs=asl_sources,
        cache_path=CACHE_ASL,
        target_classes=ASL_ALPHABET_CLASSES,
        max_per_class=200,
        name="ASL Alphabet",
    )

    train_and_evaluate(
        X_asl, y_asl, classes_asl,
        model_name="ASL",
        pkl_export=EXPORT_ASL_PKL,
        json_export=EXPORT_ASL_JSON,
        cm_path=CM_ASL_PNG,
    )

    if ISL_DIR.exists():
        X_isl, y_isl, classes_isl = collect_landmark_dataset(
            dataset_dirs=[ISL_DIR],
            cache_path=CACHE_ISL,
            max_per_class=100,
            name="ISL",
        )
        train_and_evaluate(
            X_isl, y_isl, classes_isl,
            model_name="ISL",
            pkl_export=EXPORT_ISL_PKL,
            json_export=EXPORT_ISL_JSON,
            cm_path=CM_ISL_PNG,
        )


if __name__ == "__main__":
    main()
