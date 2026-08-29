"""
ISL Dual-Hand Robust Trainer with Spatial Screen Sorting (Left Screen Hand + Right Screen Hand)
"""

import os
import cv2
import pickle
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
MODELS_DIR.mkdir(exist_ok=True)

CACHE_ISL_SPATIAL = MODELS_DIR / "isl_spatial_dual_cache.pkl"
EXPORT_ISL_PKL = MODELS_DIR / "isl_landmark_model.pkl"

# Initialize 2-hand detector
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.20,
    min_hand_presence_confidence=0.20,
    min_tracking_confidence=0.20
)
detector = vision.HandLandmarker.create_from_options(options)


def calculate_angle_3d(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle)) / 180.0


def extract_single_hand_features(landmarks, w, h):
    """Extracts 78-dim normalized features for one hand."""
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
    wrist = pts[0]
    pts_centered = pts - wrist
    middle_mcp = pts_centered[9]
    scale = np.linalg.norm(middle_mcp[:2])
    if scale < 1e-4:
        scale = 1.0
    pts_scaled = pts_centered / scale

    # Align vertical rotation
    angle = np.arctan2(pts_scaled[9, 0], -pts_scaled[9, 1])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    pts_aligned = pts_scaled.copy()
    pts_aligned[:, :2] = np.dot(pts_scaled[:, :2], rot_matrix)

    angles = [
        calculate_angle_3d(pts[0], pts[1], pts[2]),
        calculate_angle_3d(pts[1], pts[2], pts[3]),
        calculate_angle_3d(pts[2], pts[3], pts[4]),
        calculate_angle_3d(pts[0], pts[5], pts[6]),
        calculate_angle_3d(pts[5], pts[6], pts[7]),
        calculate_angle_3d(pts[6], pts[7], pts[8]),
        calculate_angle_3d(pts[0], pts[9], pts[10]),
        calculate_angle_3d(pts[9], pts[10], pts[11]),
        calculate_angle_3d(pts[10], pts[11], pts[12]),
        calculate_angle_3d(pts[0], pts[13], pts[14]),
        calculate_angle_3d(pts[13], pts[14], pts[15]),
        calculate_angle_3d(pts[14], pts[15], pts[16]),
        calculate_angle_3d(pts[0], pts[17], pts[18]),
        calculate_angle_3d(pts[17], pts[18], pts[19]),
        calculate_angle_3d(pts[18], pts[19], pts[20]),
    ]
    return np.concatenate([pts_aligned.flatten(), angles])


def extract_spatial_dual_hand_features(img_bgr):
    """
    Extracts 156-dim feature vector using screen spatial horizontal sorting:
    - Slot 1 (Leftmost hand in image, x_min): 78 features
    - Slot 2 (Rightmost hand in image, x_max): 78 features (zeros if only 1 hand)
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_img)

    if not results.hand_landmarks:
        return None

    # Sort detected hands strictly by horizontal screen coordinate (wrist x)
    hands_with_x = []
    for lms in results.hand_landmarks:
        wrist_x = lms[0].x
        feat = extract_single_hand_features(lms, w, h)
        hands_with_x.append((wrist_x, feat))

    hands_with_x.sort(key=lambda item: item[0])

    slot1_left_hand = hands_with_x[0][1]
    if len(hands_with_x) > 1:
        slot2_right_hand = hands_with_x[1][1]
    else:
        slot2_right_hand = np.zeros(78, dtype=np.float32)

    return np.concatenate([slot1_left_hand, slot2_right_hand])


def train_and_export():
    classes = sorted([d for d in os.listdir(ISL_DIR) if (ISL_DIR / d).is_dir()])
    print(f"[*] Extracting spatial dual-hand landmarks for {len(classes)} ISL classes...")

    X, y = [], []
    for cls in classes:
        cls_dir = ISL_DIR / cls
        img_files = sorted([f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        count = 0
        for img_name in img_files:
            img = cv2.imread(str(cls_dir / img_name))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            pad = int(max(h, w) * 0.15)
            img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            
            feat = extract_spatial_dual_hand_features(img_padded)
            if feat is not None:
                X.append(feat)
                y.append(cls)
                count += 1
                if count >= 150:
                    break

        print(f" -> Class '{cls}': Extracted {count} samples.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes_list = [str(c) for c in le.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    print(f"[*] Training Spatial Dual-Hand MLP Classifier on {len(X_train)} samples...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=400,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=20
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"[+] ISL Spatial Dual-Hand Held-out Test Accuracy: {acc:.2f}%")
    print(classification_report(y_test, y_pred, target_names=classes_list))

    with open(EXPORT_ISL_PKL, "wb") as f:
        pickle.dump({
            "model": clf,
            "label_encoder": le,
            "classes": classes_list,
            "num_features": 156,
            "dual_hand": True
        }, f)
    print(f"[+] Exported Spatial Dual-Hand ISL Model to: {EXPORT_ISL_PKL}")


if __name__ == "__main__":
    train_and_export()
