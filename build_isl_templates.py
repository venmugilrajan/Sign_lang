"""
Builds pristine, high-fidelity dual-hand and single-hand templates for ISL
from the official ISL chart (ISL_Hand_Signs_Official.png) and the Indian Sign Language dataset.
Preserves existing ASL templates in web/sign_templates.json.
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
CHART_PATH = BASE_DIR / "ISL_Hand_Signs_Official.png"
DATASET_DIR = BASE_DIR / "indian sign language" / "Indian"
TEMPLATES_JSON = BASE_DIR / "web" / "sign_templates.json"
TASK_PATH = str(BASE_DIR / "hand_landmarker.task")

TWO_HANDED_SIGNS = {
    'A', 'B', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M',
    'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z'
}

SINGLE_HANDED_SIGNS = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'C', 'I', 'L', 'O', 'U', 'V'
}

GESTURE_DESCRIPTIONS = {
    '1': "Single index finger pointing upward",
    '2': "Index and middle fingers raised in a 'V'",
    '3': "Thumb, index, and middle fingers extended",
    '4': "Four fingers extended upward with thumb tucked",
    '5': "All five fingers fully spread open",
    '6': "Thumb and pinky raised, middle fingers curled",
    '7': "Index finger hooked downward",
    '8': "Thumb, index, and middle spread wide",
    '9': "Thumb and index curled inward",
    'A': "Both index fingers touching at top to form tent apex",
    'B': "Both hands form two loops touching together",
    'C': "Single hand curved into a smooth 'C' arc",
    'D': "Left index vertical, right hand forms rounded 'D' loop",
    'E': "Left index horizontal, right index touching it",
    'F': "Two fingers crossed overlapping horizontally",
    'G': "Two closed fists stacked vertically",
    'H': "Open flat palm with other hand laid horizontally across",
    'I': "Single index finger pointing diagonally",
    'J': "Hand forming 'L' with other index drawing a hook",
    'K': "Two index fingers angled together to form 'K'",
    'L': "Thumb and index finger forming right angle 'L'",
    'M': "Three fingers resting across horizontal base hand",
    'N': "Two fingers resting across horizontal base hand",
    'O': "Fingers curled inward forming circular 'O'",
    'P': "Left index vertical, right hand forms upper loop",
    'Q': "Ring loop held with other index touching",
    'R': "Horizontal flat hand with other hand curled underneath",
    'S': "Both hands interlocking pinkies / fingers",
    'T': "Left index vertical, right index horizontal on top",
    'U': "Thumb and pinky extended upward in 'U' cup",
    'V': "Index and middle fingers spread in a clean 'V'",
    'W': "Both hands interlocking angled fingers in 'W'",
    'X': "Both index fingers crossed in the center",
    'Y': "Thumb and pinky spread with other index pointing inward",
    'Z': "One vertical palm with other hand fingers against it",
    'REST': "Hands relaxed naturally at resting position",
    'SPACE': "Pause between words, hands relaxed at resting position"
}

# 6x6 grid labels in ISL_Hand_Signs_Official.png
CHART_GRID = [
    ['1', '2', '3', '4', '5', '6'],
    ['7', '8', '9', 'A', 'B', 'C'],
    ['D', 'E', 'F', 'G', 'H', 'I'],
    ['J', 'K', 'L', 'M', 'N', 'O'],
    ['P', 'Q', 'R', 'S', 'T', 'U'],
    ['V', 'W', 'X', 'Y', 'Z', '']
]


def init_detector():
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_tracking_confidence=0.15
    )
    return vision.HandLandmarker.create_from_options(options)


def normalize_hand_landmarks(lms):
    """
    Centers hand on wrist [0, 0, 0] and normalizes hand span to natural unit scale (max span ~1.75).
    Guarantees hand proportions are always anatomically consistent regardless of finger curling.
    """
    raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    wrist = raw[0].copy()
    centered = raw - wrist

    # Find maximum 2D distance from wrist to any joint / fingertip
    distances = np.linalg.norm(centered[:, :2], axis=1)
    max_d = np.max(distances)
    if max_d < 1e-4:
        max_d = 1.0

    # Natural scale: full hand span from wrist to extended tip is ~1.75 units
    normalized = centered / (max_d / 1.75)

    # Ensure coordinates are cleanly rounded floats
    return [[round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)] for pt in normalized]


def extract_from_chart(detector, chart_img):
    cell_w = chart_img.shape[1] // 6
    cell_h = chart_img.shape[0] // 6

    crops = {}
    for r in range(6):
        for c in range(6):
            lbl = CHART_GRID[r][c]
            if not lbl:
                continue
            crop = chart_img[r * cell_h + 30:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            crops[lbl] = crop
    return crops


def detect_hands_in_image(detector, img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return detector.detect(mp_im)


def build_templates():
    print("[*] Initializing Hand Landmarker...")
    detector = init_detector()

    print(f"[*] Reading reference chart from: {CHART_PATH}")
    chart_img = cv2.imread(str(CHART_PATH))
    if chart_img is None:
        raise FileNotFoundError(f"Could not load chart from {CHART_PATH}")

    crops = extract_from_chart(detector, chart_img)
    isl_templates = {}

    all_labels = [c for row in CHART_GRID for c in row if c]

    for lbl in all_labels:
        is_two_handed = lbl in TWO_HANDED_SIGNS
        crop = crops[lbl]
        res = detect_hands_in_image(detector, crop)

        # Fallback to dataset if two hands were expected but < 2 were found
        if is_two_handed and len(res.hand_landmarks) < 2:
            cls_dir = DATASET_DIR / lbl
            if cls_dir.exists():
                for fn in sorted(os.listdir(cls_dir)):
                    sample_im = cv2.imread(str(cls_dir / fn))
                    if sample_im is None:
                        continue
                    r2 = detect_hands_in_image(detector, sample_im)
                    if len(r2.hand_landmarks) == 2:
                        res = r2
                        break

        num_found = len(res.hand_landmarks)
        print(f" -> Sign '{lbl}': detected {num_found} hands (expected 2-handed={is_two_handed})")

        if is_two_handed and num_found >= 2:
            # Sort by x coordinate of wrist: smaller x is viewer-left, larger x is viewer-right
            lms_a = res.hand_landmarks[0]
            lms_b = res.hand_landmarks[1]
            if lms_a[0].x <= lms_b[0].x:
                left_lms, right_lms = lms_a, lms_b
            else:
                left_lms, right_lms = lms_b, lms_a

            norm_left = normalize_hand_landmarks(left_lms)
            norm_right = normalize_hand_landmarks(right_lms)

            # Compute natural wrist offsets based on relative detection positions
            # Center of left wrist is around x=0.3, right around x=0.7 in 0..1 crop
            wL_dx = np.clip((left_lms[0].x - 0.35) * 0.5, -0.15, 0.15)
            wL_dy = np.clip((left_lms[0].y - 0.65) * 0.4, -0.15, 0.15)
            wR_dx = np.clip((right_lms[0].x - 0.65) * 0.5, -0.15, 0.15)
            wR_dy = np.clip((right_lms[0].y - 0.65) * 0.4, -0.15, 0.15)

            isl_templates[lbl] = {
                "twoHanded": True,
                "left": norm_left,
                "right": norm_right,
                "wristL": [round(float(wL_dx), 3), round(float(wL_dy), 3)],
                "wristR": [round(float(wR_dx), 3), round(float(wR_dy), 3)],
                "desc": GESTURE_DESCRIPTIONS.get(lbl, "")
            }
        else:
            # Single-handed sign
            lms = res.hand_landmarks[0] if num_found >= 1 else None
            if lms is None:
                # Fallback to dataset for single hand
                cls_dir = DATASET_DIR / lbl
                if cls_dir.exists():
                    for fn in sorted(os.listdir(cls_dir)):
                        sample_im = cv2.imread(str(cls_dir / fn))
                        if sample_im is None:
                            continue
                        r2 = detect_hands_in_image(detector, sample_im)
                        if len(r2.hand_landmarks) >= 1:
                            lms = r2.hand_landmarks[0]
                            break

            norm_hand = normalize_hand_landmarks(lms) if lms else []

            isl_templates[lbl] = {
                "twoHanded": False,
                "left": None,
                "right": norm_hand,
                "wristL": [0.0, 0.0],
                "wristR": [0.0, 0.0],
                "desc": GESTURE_DESCRIPTIONS.get(lbl, "")
            }

    # REST and SPACE poses
    neutral_rest_hand = [[0.0, 0.0, 0.0]]
    for i in range(1, 21):
        finger_idx = (i - 1) // 4
        joint_idx = ((i - 1) % 4) + 1
        x_off = round((finger_idx - 2) * 0.18, 3)
        y_off = round(-0.35 * joint_idx, 3)
        neutral_rest_hand.append([x_off, y_off, 0.0])

    isl_templates['REST'] = {
        "twoHanded": False,
        "left": None,
        "right": neutral_rest_hand,
        "wristL": [0.0, 0.0],
        "wristR": [0.0, 0.0],
        "desc": GESTURE_DESCRIPTIONS['REST']
    }

    isl_templates['SPACE'] = {
        "twoHanded": False,
        "left": None,
        "right": neutral_rest_hand,
        "wristL": [0.0, 0.0],
        "wristR": [0.0, 0.0],
        "desc": GESTURE_DESCRIPTIONS['SPACE']
    }

    # Read existing sign_templates.json to preserve ASL
    if TEMPLATES_JSON.exists():
        with open(TEMPLATES_JSON, "r", encoding="utf-8") as f:
            full_data = json.load(f)
    else:
        full_data = {"asl": {}}

    full_data["isl"] = isl_templates

    print(f"[*] Writing updated templates to: {TEMPLATES_JSON}")
    with open(TEMPLATES_JSON, "w", encoding="utf-8") as f:
        json.dump(full_data, f)

    two_h_count = sum(1 for v in isl_templates.values() if v.get("twoHanded"))
    single_h_count = sum(1 for v in isl_templates.values() if not v.get("twoHanded"))
    print(f"[+] Done! ISL templates saved: {two_h_count} two-handed, {single_h_count} single-handed. Total ISL: {len(isl_templates)}")


if __name__ == "__main__":
    build_templates()
