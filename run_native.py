"""
SignLens Native — Real-Time Sign Language Translator & Diagnostic Studio

Features:
- Dual Hand Support (Automatic Left/Right handedness detection & geometric normalization)
- Short Word Protection (Preserves valid short words like 'HI', 'NO', 'OK', 'YES', 'ME', 'MY', 'GO' without over-correction)
- Robust Invariant Feature Extraction: 15 Finger Joint Angles + 63 Normalized & Rotation-Aligned Coordinates
- Gesture Release Detection (Supports double letters like 'HELLO', 'PLEASE')
- Real-Time Diagnostic HUD with Handedness & Top-3 Candidates
"""

import os
import cv2
import time
import pickle
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from word_buffer_engine import WordBufferStateMachine, TwoTierSpellCorrector

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
ASL_MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_landmark_model.pkl")
ISL_MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_landmark_model.pkl")

# Live Thresholds
MIN_DETECTION_CONF = 0.25   # Lowered detection confidence to reliably detect 2 hands simultaneously
CONF_THRESHOLD = 0.40       # Letter acceptance threshold
STABLE_FRAMES_NEEDED = 5    # Consecutive frames required to confirm letter
RELEASE_FRAMES_NEEDED = 2   # Frames to confirm hand release for double letters
NO_HAND_WORD_TIMEOUT = 2.0  # Seconds to auto-commit word on hand drop

# MediaPipe Hand Skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm base
]


def calculate_angle_3d(a, b, c):
    """Calculates angle in degrees at vertex b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle)) / 180.0


class LandmarkSignTranslator:
    def __init__(self):
        # 1. Initialize MediaPipe Hand Detector (Supports up to 2 hands for ISL and dual-hand gestures)
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=MIN_DETECTION_CONF,
            min_hand_presence_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_DETECTION_CONF
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 2. Word Buffer State Machine & Corrector
        self.corrector = TwoTierSpellCorrector()
        self.state_machine = WordBufferStateMachine(
            corrector=self.corrector,
            stable_frames_needed=STABLE_FRAMES_NEEDED,
            release_frames_needed=RELEASE_FRAMES_NEEDED
        )

        # 3. Model Registry
        self.models = {}
        self.classes = {}
        self.encoders = {}
        self.model_configs = {}
        self.current_mode = "asl"
        self._load_models()

        # 4. Pipeline State
        self.last_hand_seen_time = time.time()
        self.no_hand_active = True
        self.debug_mode = True

    def _load_models(self):
        """Loads available ASL and ISL landmark classifiers."""
        for mode, path in [("asl", ASL_MODEL_PATH), ("isl", ISL_MODEL_PATH)]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.models[mode] = data["model"]
                    self.classes[mode] = [str(c) for c in data["classes"]]
                    self.encoders[mode] = data.get("label_encoder", None)
                    self.model_configs[mode] = {
                        "num_features": data.get("num_features", data["model"].n_features_in_),
                        "dual_hand": data.get("dual_hand", data.get("num_features", 78) > 78)
                    }
                    print(f"[+] Loaded {mode.upper()} landmark classifier ({len(data['classes'])} classes, {self.model_configs[mode]['num_features']} features, dual={self.model_configs[mode]['dual_hand']}) from {path}")
            else:
                print(f"[!] Warning: {mode.upper()} model not found at {path}")

        if "asl" not in self.models and "isl" in self.models:
            self.current_mode = "isl"

    def switch_mode(self, mode: str):
        """Switches between ASL and ISL modes."""
        if mode in self.models:
            self.current_mode = mode
            self.state_machine.clear_all()
            print(f"[+] Switched mode to: {mode.upper()}")
            return True
        print(f"[!] Cannot switch to {mode.upper()} (model not loaded).")
        return False

    def _extract_single_hand_vector(self, landmarks, w, h, handedness, apply_handedness_mirror=True):
        """Extracts 78-dim normalized and rotation-aligned feature vector for one hand."""
        pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)

        # Mirror Left hand to match Right hand dataset distribution (Only in single-hand ASL mode)
        if apply_handedness_mirror and handedness == "Left":
            pts[:, 0] = pts[0, 0] - (pts[:, 0] - pts[0, 0])

        wrist = pts[0]
        pts_centered = pts - wrist

        middle_mcp = pts_centered[9]
        scale = np.linalg.norm(middle_mcp[:2])
        if scale < 1e-4:
            scale = 1.0
        pts_scaled = pts_centered / scale

        # 2D Palm vertical alignment
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

    def extract_landmarks(self, raw_unflipped_frame, display_w, display_h):
        """
        Extracts landmarks for up to 2 hands:
        - For ASL: Extracts 78-dim vector from the dominant hand.
        - For ISL: Extracts 156-dim dual-hand vector (Left + Right hands).
        - Returns feature vector, list of display landmark sets, and handedness info string.
        """
        h, w = raw_unflipped_frame.shape[:2]
        frame_rgb = cv2.cvtColor(raw_unflipped_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            return None, [], "None"

        all_pixel_points = []
        hand_info_list = []
        left_feat = np.zeros(78, dtype=np.float32)
        right_feat = np.zeros(78, dtype=np.float32)
        dominant_single_feat = None

        is_dual_mode = self.model_configs.get(self.current_mode, {}).get("dual_hand", False)

        hands_data = []

        for idx, landmarks in enumerate(results.hand_landmarks):
            h_name = "Right"
            score = 1.0
            if results.handedness and idx < len(results.handedness):
                h_name = results.handedness[idx][0].category_name
                score = results.handedness[idx][0].score

            # Drawing coordinates for flipped screen
            pts_display = [(int((1.0 - lm.x) * display_w), int(lm.y * display_h)) for lm in landmarks]
            all_pixel_points.append(pts_display)
            hand_info_list.append(f"{h_name} ({score*100:.0f}%)")

            # Extract 78-dim vector (In ISL mode, keep absolute coordinates without right-hand mirror)
            single_vec = self._extract_single_hand_vector(landmarks, w, h, h_name, apply_handedness_mirror=(not is_dual_mode))
            wrist_x = landmarks[0].x
            hands_data.append((wrist_x, single_vec))

        if is_dual_mode:
            # Sort detected hands strictly by horizontal position (Left to Right)
            hands_data.sort(key=lambda x: x[0])
            slot1 = hands_data[0][1]
            slot2 = hands_data[1][1] if len(hands_data) > 1 else np.zeros(78, dtype=np.float32)
            features = np.concatenate([slot1, slot2]).reshape(1, -1)
        else:
            # Single hand 78-dim
            features = hands_data[0][1].reshape(1, -1)

        handedness_info = " + ".join(hand_info_list)
        return features, all_pixel_points, handedness_info

    def predict(self, features):
        """Predicts class, confidence and top-3 candidates."""
        if self.current_mode not in self.models or features is None:
            return None, 0.0, []

        clf = self.models[self.current_mode]
        probs = clf.predict_proba(features)[0]
        top_idx = int(np.argmax(probs))
        
        le = self.encoders.get(self.current_mode, None)
        if le is not None:
            top_label = str(le.inverse_transform([top_idx])[0])
            classes_list = [str(c) for c in le.classes_]
        else:
            top_label = str(clf.classes_[top_idx])
            classes_list = [str(c) for c in clf.classes_]

        top_conf = float(probs[top_idx])

        # Top 3 predictions
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [(classes_list[i], float(probs[i])) for i in top3_indices]

        return top_label, top_conf, top3

    def process_frame(self, raw_unflipped_frame, display_w, display_h):
        """Processes frame through landmark extraction, state machine, and diagnostics."""
        now = time.time()
        features, pixel_points, handedness_info = self.extract_landmarks(raw_unflipped_frame, display_w, display_h)

        raw_pred = "NO HAND"
        raw_conf = 0.0
        top3 = []

        if features is not None:
            self.last_hand_seen_time = now
            self.no_hand_active = False

            label, conf, top3 = self.predict(features)
            raw_pred = label
            raw_conf = conf

            # Feed prediction into word-buffer state machine
            sm_status = self.state_machine.feed_frame(raw_pred, raw_conf, min_conf=CONF_THRESHOLD)
        else:
            sm_status = self.state_machine.feed_frame("NO HAND", 0.0, min_conf=CONF_THRESHOLD)

            # Auto-commit word after timeout
            if (not self.no_hand_active) and (now - self.last_hand_seen_time >= NO_HAND_WORD_TIMEOUT):
                if self.state_machine.current_word:
                    self.state_machine.commit_word()
                self.no_hand_active = True
                sm_status = self.state_machine._get_status()

        return {
            "hand_present": features is not None,
            "landmarks": pixel_points,
            "handedness": handedness_info if handedness_info else "None",
            "raw_pred": raw_pred,
            "raw_conf": raw_conf,
            "top3": top3,
            "hold_progress": sm_status["hold_progress"],
            "word_buffer": sm_status["word_buffer"],
            "sentence": sm_status["sentence"],
            "suggested_word": sm_status["suggested_word"],
            "suggestions": sm_status["suggestions"],
            "released": sm_status["released"],
            "debug_mode": self.debug_mode
        }

    def commit_word(self):
        self.state_machine.commit_word()

    def backspace(self):
        self.state_machine.backspace()

    def clear_all(self):
        self.state_machine.clear_all()


# ─── UI HUD Drawing ────────────────────────────────────────────────────────────
def draw_hud(frame, state, mode_name, fps):
    h, w = frame.shape[:2]

    # 1. ALWAYS Draw all detected hand skeletons (Supports 1 or 2 hands)
    if state["landmarks"]:
        for pts in state["landmarks"]:
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[p1_idx], pts[p2_idx], (0, 212, 255), 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                if i in [4, 8, 12, 16, 20]:
                    color = (255, 107, 157)
                    r = 6
                elif i == 0:
                    color = (255, 255, 255)
                    r = 7
                else:
                    color = (34, 214, 122)
                    r = 5
                cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, r + 2, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Top Header Banner
    cv2.rectangle(frame, (0, 0), (w, 80), (14, 14, 26), -1)
    cv2.line(frame, (0, 80), (w, 80), (45, 45, 75), 2)

    # Title & Controls
    cv2.putText(frame, f"SignLens | {mode_name.upper()} Mode", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f} | [1] ASL  [2] ISL  [d] Debug  [b] Del  [s] Space  [c] Clear  [q] Quit", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 210), 1, cv2.LINE_AA)

    # Hand Presence & Handedness Badge (Top Right)
    hand_badge = f"{state['handedness']} Hand" if state["hand_present"] else "NO HAND"
    badge_col = (34, 214, 122) if state["hand_present"] else (100, 100, 120)
    cv2.rectangle(frame, (w - 220, 15), (w - 20, 48), (25, 25, 45), -1)
    cv2.rectangle(frame, (w - 220, 15), (w - 20, 48), badge_col, 1)
    cv2.putText(frame, hand_badge, (w - 205, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col, 1, cv2.LINE_AA)

    # 3. Live Sign Card (Left Overlay)
    cv2.rectangle(frame, (20, 100), (280, 240), (18, 18, 35), -1)
    cv2.rectangle(frame, (20, 100), (280, 240), (60, 60, 100), 1)

    raw_pred = state["raw_pred"]
    raw_conf = state["raw_conf"] * 100

    cv2.putText(frame, "LIVE SIGN", (35, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 132, 255), 1, cv2.LINE_AA)

    if state["hand_present"]:
        pred_col = (34, 214, 122) if raw_conf >= (CONF_THRESHOLD * 100) else (0, 200, 255)
        cv2.putText(frame, raw_pred, (35, 175),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, pred_col, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Confidence: {raw_conf:.1f}%", (35, 202),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        # Hold Progress Bar
        bar_w = int(220 * state["hold_progress"])
        cv2.rectangle(frame, (35, 215), (255, 225), (35, 35, 60), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (35, 215), (35 + bar_w, 225), (34, 214, 122), -1)
    else:
        cv2.putText(frame, "—", (35, 175),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (100, 100, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, "Waiting for hand...", (35, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 160), 1, cv2.LINE_AA)

    # 4. Diagnostic Top-3 Box (Right Overlay when Debug Mode is ON)
    if state.get("debug_mode") and state["hand_present"] and state["top3"]:
        cv2.rectangle(frame, (w - 260, 100), (w - 20, 210), (18, 18, 35), -1)
        cv2.rectangle(frame, (w - 260, 100), (w - 20, 210), (60, 60, 100), 1)
        cv2.putText(frame, f"TOP PREDICTIONS ({state['handedness']})", (w - 250, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 212, 255), 1, cv2.LINE_AA)
        for idx, (label, prob) in enumerate(state["top3"]):
            y_pos = 150 + idx * 22
            col = (34, 214, 122) if idx == 0 and prob >= CONF_THRESHOLD else (200, 200, 200)
            cv2.putText(frame, f"{idx+1}. Letter {label}: {prob*100:.1f}%", (w - 245, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 5. Bottom Word & Sentence Dashboard
    cv2.rectangle(frame, (0, h - 120), (w, h), (14, 14, 26), -1)
    cv2.line(frame, (0, h - 120), (w, h - 120), (45, 45, 75), 2)

    # Building Word Bar
    word = state["word_buffer"] if state["word_buffer"] else "(Signing...)"
    word_col = (0, 212, 255) if state["word_buffer"] else (120, 120, 150)
    cv2.putText(frame, "BUILDING WORD:", (25, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, word, (165, h - 82),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, word_col, 2, cv2.LINE_AA)

    # Spellcheck suggestions
    if state["suggested_word"] and state["suggested_word"] != state["word_buffer"]:
        cv2.putText(frame, f"[Suggest: {state['suggested_word']}]", (350, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 214, 122), 1, cv2.LINE_AA)
    elif state["suggestions"]:
        sugg_str = ", ".join(state["suggestions"][:2])
        cv2.putText(frame, f"[Did you mean: {sugg_str}]", (350, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 50), 1, cv2.LINE_AA)

    # Full Sentence Display Tape
    sentence_txt = state["sentence"] if state["sentence"] else "(Empty sentence)"
    sent_col = (255, 255, 255) if state["sentence"] else (100, 100, 120)
    cv2.putText(frame, "SENTENCE:", (25, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 132, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, sentence_txt, (120, h - 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, sent_col, 2, cv2.LINE_AA)


def main():
    print("=" * 70)
    print("  SignLens — Real-Time Sign Language Translator & Diagnostic Studio")
    print("=" * 70)
    print("Controls:")
    print("  [1] Switch to ASL Mode")
    print("  [2] Switch to ISL Mode")
    print("  [d] Toggle Live Debug HUD (Handedness & Top-3 Predictions)")
    print("  [b] Backspace (Delete last letter/word)")
    print("  [s] Add Space / Commit current word")
    print("  [c] Clear everything")
    print("  [q] or [ESC] Quit")
    print("=" * 70)

    translator = LandmarkSignTranslator()

    # Initialize Camera
    cap = None
    for backend, name in [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_ANY, "Default")]:
        temp_cap = cv2.VideoCapture(0, backend)
        if temp_cap.isOpened():
            ret, test_frame = temp_cap.read()
            if ret and test_frame is not None:
                cap = temp_cap
                print(f"[+] Camera initialized with {name} backend")
                break
            temp_cap.release()

    if cap is None:
        print("[ERROR] Could not initialize webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = 30.0
    prev_time = time.time()

    while True:
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            continue

        h, w = raw_frame.shape[:2]

        # FPS calculation
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        # 1. Process landmarks on RAW UNMIRRORED frame with Left/Right normalization
        state = translator.process_frame(raw_frame, w, h)

        # 2. Mirror display frame for natural selfie view
        display_frame = cv2.flip(raw_frame, 1)

        # 3. Draw Overlay
        draw_hud(display_frame, state, translator.current_mode, fps)

        cv2.imshow("SignLens — Real-Time Sign Language Translator", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            translator.switch_mode("asl")
        elif key == ord('2'):
            translator.switch_mode("isl")
        elif key == ord('d'):
            translator.debug_mode = not translator.debug_mode
            print(f"[+] Debug HUD: {'ON' if translator.debug_mode else 'OFF'}")
        elif key == ord('b') or key == 8:
            translator.backspace()
        elif key == ord('s') or key == 32:
            translator.commit_word()
        elif key == ord('c'):
            translator.clear_all()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
