"""
Sign Language Landmark Classifier Training Pipeline

- Uses MediaPipe HandLandmarker (21 3D hand keypoints)
- Normalizes landmarks relative to wrist position and scale
- Supports full dataset extraction with disk caching for high performance
- Evaluates on true stratified Train (70%) / Validation (15%) / Test (15%) splits
- Outputs classification reports and saves high-resolution confusion matrices
- Exports trained models to .pkl (for Python/native OpenCV) and .json (for zero-dependency browser JS)
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

CACHE_ASL = MODELS_DIR / "asl_landmark_cache.pkl"
CACHE_ISL = MODELS_DIR / "isl_landmark_cache.pkl"

EXPORT_ASL_PKL = MODELS_DIR / "asl_landmark_model.pkl"
EXPORT_ISL_PKL = MODELS_DIR / "isl_landmark_model.pkl"

EXPORT_ASL_JSON = WEB_DIR / "asl_landmarks_weights.json"
EXPORT_ISL_JSON = WEB_DIR / "isl_landmarks_weights.json"

CM_ASL_PNG = BASE_DIR / "asl_confusion_matrix.png"
CM_ISL_PNG = BASE_DIR / "isl_confusion_matrix.png"

# Initialize MediaPipe Tasks HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_TASK_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)


def extract_features_from_image(img_bgr: np.ndarray):
    """
    Extracts 63 normalized coordinates (21 keypoints * [x, y, z]) from an image.
    Normalized relative to wrist (landmark 0) and scaled by wrist-to-middle-MCP (landmark 9) distance.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None

    # Upsample small images if needed so MediaPipe detector performs reliably
    h, w = img_bgr.shape[:2]
    if max(h, w) < 180:
        scale_f = 200.0 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale_f), int(h * scale_f)), interpolation=cv2.INTER_CUBIC)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)

    if not results.hand_landmarks:
        return None

    landmarks = results.hand_landmarks[0]

    # Center & normalize hand size relative to distance between wrist (0) and middle MCP (9)
    wrist = landmarks[0]
    scale = np.sqrt((wrist.x - landmarks[9].x) ** 2 + (wrist.y - landmarks[9].y) ** 2)
    if scale < 1e-4:
        scale = 1.0

    features = []
    for lm in landmarks:
        features.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return features


def collect_landmark_dataset(dataset_dirs, cache_path, max_per_class=300, name="Dataset"):
    """
    Scans dataset directories, extracts MediaPipe landmarks, and caches the dataset to disk.
    """
    if cache_path.exists():
        print(f"[+] Loading cached landmark dataset from: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            print(f"[+] Loaded {len(data['X'])} samples across {len(data['classes'])} classes.")
            return data["X"], data["y"], data["classes"]

    print(f"\n[*] Extracting landmarks for {name} from {dataset_dirs}...")
    X, y = [], []

    if not isinstance(dataset_dirs, (list, tuple)):
        dataset_dirs = [dataset_dirs]

    # Gather all class folder names
    all_classes = set()
    for d in dataset_dirs:
        p = Path(d)
        if p.exists():
            for sub in p.iterdir():
                if sub.is_dir() and not sub.name.startswith("."):
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

    # Cache dataset
    with open(cache_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "classes": unique_classes}, f)
    print(f"[+] Saved landmark cache to: {cache_path}")

    return X, y, unique_classes


def plot_and_save_confusion_matrix(y_true, y_pred, classes, save_path, title):
    """Generates and saves a high-resolution dark-mode confusion matrix heatmap."""
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
    print(f"[+] Confusion matrix saved to: {save_path}")


def analyze_confusions(y_true, y_pred, classes):
    """Diagnoses the top confused pairs in the predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    confusions = []
    for i, true_cls in enumerate(classes):
        for j, pred_cls in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confusions.append((true_cls, pred_cls, cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)
    if confusions:
        print("\n[!] Top Confused Letter Pairs:")
        for true_c, pred_c, count in confusions[:8]:
            print(f"   * True '{true_c}' misclassified as '{pred_c}': {count} time(s)")
    else:
        print("\n[+] Perfect classification on test set! No confused pairs detected.")


def train_and_evaluate(X, y_raw, classes, model_name, pkl_export, json_export, cm_path):
    """
    Trains MLPClassifier on stratified Train/Val/Test splits and exports models.
    """
    print(f"\n{'='*70}")
    print(f"  Training & Evaluating {model_name} Landmark Classifier")
    print(f"{'='*70}")

    # Encode string classes to integer labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = [str(c) for c in le.classes_]

    # True 3-Way Split: 70% Train, 15% Validation, 15% Test
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

    # Neural Network Architecture: 3 hidden layers with ReLU & Early Stopping
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=42,
    )

    print("\n[*] Training MLP Neural Network...")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train) * 100
    val_acc = clf.score(X_val, y_val) * 100
    test_preds_int = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds_int) * 100

    # Convert test labels back to class strings for reports
    y_test_str = le.inverse_transform(y_test)
    test_preds_str = le.inverse_transform(test_preds_int)

    print(f"\n[+] Results for {model_name}:")
    print(f"   * Training Accuracy   : {train_acc:.2f}%")
    print(f"   * Validation Accuracy : {val_acc:.2f}%")
    print(f"   * Held-out TEST Acc   : {test_acc:.2f}%  (REAL-WORLD TEST ACCURACY)")

    print("\n[+] Detailed Classification Report (Held-out Test Set):")
    print(classification_report(y_test_str, test_preds_str, digits=3))

    analyze_confusions(y_test_str, test_preds_str, class_names)
    plot_and_save_confusion_matrix(
        y_test_str, test_preds_str, class_names, cm_path,
        f"{model_name} Confusion Matrix (Held-out Test Set - Acc: {test_acc:.1f}%)"
    )

    # 1. Export Standard Scikit-Learn Model & LabelEncoder without custom class dependencies
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

    # 2. Export Browser-Compatible JSON Weights
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
    print("  Sign Language Landmark Classifier Training Pipeline")
    print("=" * 70)

    # ── 1. ASL Training ──
    asl_sources = [ASL_DIR]
    if DS2_DIR.exists():
        asl_sources.append(DS2_DIR)

    X_asl, y_asl, classes_asl = collect_landmark_dataset(
        dataset_dirs=asl_sources,
        cache_path=CACHE_ASL,
        max_per_class=200,
        name="ASL",
    )

    train_and_evaluate(
        X_asl, y_asl, classes_asl,
        model_name="ASL",
        pkl_export=EXPORT_ASL_PKL,
        json_export=EXPORT_ASL_JSON,
        cm_path=CM_ASL_PNG,
    )

    # ── 2. ISL Training ──
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
