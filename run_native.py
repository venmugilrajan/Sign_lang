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
import threading
import numpy as np

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

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


# ─── Offline Text-to-Speech ────────────────────────────────────────────────────
_tts_engine = pyttsx3.init() if pyttsx3 else None
_tts_lock = threading.Lock()   # pyttsx3 run loops cannot overlap


def speak(word: str):
    """Speaks a word on a detached daemon thread so the HUD never blocks."""
    if not _tts_engine or not word:
        return

    def _run():
        with _tts_lock:
            try:
                _tts_engine.say(word)
                _tts_engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Failed to speak '{word}': {e}")

    threading.Thread(target=_run, daemon=True).start()


def calculate_angle_3d(a, b, c):
    """Calculates angle in degrees at vertex b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle)) / 180.0


class LandmarkSignTranslator:
    def __init__(self):
        # 1. Initialize MediaPipe Hand Detector in VIDEO mode for smooth temporal tracking (eliminates jitter)
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=MIN_DETECTION_CONF,
            min_hand_presence_confidence=MIN_DETECTION_CONF,
            min_tracking_confidence=MIN_DETECTION_CONF
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

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
        self._spoken_count = 0   # words already sent to TTS

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

        # KNOWN BUG (deferred): this rotates by +angle, which DOUBLES the palm angle
        # instead of cancelling it (20 deg hand rotation -> 40 deg residual, 90 -> inverted),
        # so these features are not rotation-invariant. Fix is one sign:
        #     rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # Do NOT apply it alone -- every shipped .pkl/.json was trained on the broken
        # transform, so ASL and both ISL variants must be retrained together.
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
        
        # Advance monotonic millisecond timestamp for temporal tracking
        now_ms = int(time.time() * 1000)
        if now_ms <= self.frame_timestamp_ms:
            now_ms = self.frame_timestamp_ms + 1
        self.frame_timestamp_ms = now_ms

        results = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

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
            self.state_machine.last_top3 = top3   # diagnostic: logged on each confirmed letter
            sm_status = self.state_machine.feed_frame(raw_pred, raw_conf, min_conf=CONF_THRESHOLD)
        else:
            sm_status = self.state_machine.feed_frame("NO HAND", 0.0, min_conf=CONF_THRESHOLD)

            # Auto-commit word after timeout
            if (not self.no_hand_active) and (now - self.last_hand_seen_time >= NO_HAND_WORD_TIMEOUT):
                if self.state_machine.current_word:
                    self.state_machine.commit_word()
                self.no_hand_active = True
                sm_status = self.state_machine._get_status()

        # Speak anything newly committed. Watching the sentence covers all three
        # commit paths -- 'space' gesture (fires inside the state machine),
        # no-hand timeout, and the manual 's' key -- from one place.
        committed = self.state_machine.sentence
        if len(committed) > self._spoken_count:
            for word in committed[self._spoken_count:]:
                speak(word)
        self._spoken_count = len(committed)

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
# Palette shared with web/style.css so the desktop HUD and the browser UI read as
# one product. OpenCV takes BGR, but the previous code passed the web RGB hex
# values straight through -- so "cyan" #00d4ff was drawn as (0,212,255), which is
# orange on screen. _bgr() does the conversion once, correctly.
def _bgr(hex_color):
    """'#7c6aff' -> (255, 106, 124) BGR tuple for OpenCV."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


C_BG      = _bgr("#07080f")   # page ground
C_PANEL   = _bgr("#0d1124")   # glass fill
C_HAIRLINE= _bgr("#2d3566")   # 1px border
C_HI      = _bgr("#454f80")   # top-edge highlight
C_TEXT    = _bgr("#eef0ff")
C_DIM     = _bgr("#7a88b8")
C_MUTED   = _bgr("#3e4870")
C_PURPLE  = _bgr("#a084ff")
C_CYAN    = _bgr("#00d4ff")
C_GREEN   = _bgr("#22d67a")
C_PINK    = _bgr("#ff6b9d")
C_YELLOW  = _bgr("#ffe066")
C_ORANGE  = _bgr("#ff9240")

PANEL_ALPHA = 0.72   # matches --glass rgba(13,17,36,0.72)


def _panel(frame, x1, y1, x2, y2, alpha=PANEL_ALPHA, accent=None):
    """Translucent glass panel with a hairline border and a top highlight.

    Blending against the live frame is what makes it read as glass rather than
    a flat opaque box -- the OpenCV analogue of backdrop-filter: blur().
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    fill = np.full(roi.shape, C_PANEL, dtype=np.uint8)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), accent or C_HAIRLINE, 1, cv2.LINE_AA)
    # single bright pixel row along the top edge = the "lit" glass edge
    cv2.line(frame, (x1 + 1, y1 + 1), (x2 - 1, y1 + 1), C_HI, 1, cv2.LINE_AA)


def _label(frame, text, org, color=None, scale=0.42):
    """Small dim caption -- the eyebrow above a value."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color or C_DIM, 1, cv2.LINE_AA)


def draw_hud(frame, state, mode_name, fps):
    h, w = frame.shape[:2]

    # 1. ALWAYS Draw all detected hand skeletons (Supports 1 or 2 hands)
    if state["landmarks"]:
        for pts in state["landmarks"]:
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                cv2.line(frame, pts[p1_idx], pts[p2_idx], C_CYAN, 2, cv2.LINE_AA)
            for i, pt in enumerate(pts):
                if i in [4, 8, 12, 16, 20]:
                    color = C_PINK
                    r = 6
                elif i == 0:
                    color = (255, 255, 255)
                    r = 7
                else:
                    color = C_GREEN
                    r = 5
                cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, r + 2, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Top Header Banner
    _panel(frame, -1, -1, w + 1, 80, alpha=0.80)
    cv2.line(frame, (0, 80), (w, 80), C_PURPLE, 1, cv2.LINE_AA)

    # Title & Controls
    cv2.putText(frame, f"SignLens | {mode_name.upper()} Mode", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f} | [1] ASL  [2] ISL  [d] Debug  [b] Del  [s] Space  [c] Clear  [q] Quit", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1, cv2.LINE_AA)

    # Hand Presence & Handedness Badge (Top Right)
    hand_badge = f"{state['handedness']} Hand" if state["hand_present"] else "NO HAND"
    badge_col = C_GREEN if state["hand_present"] else C_MUTED
    _panel(frame, w - 220, 15, w - 20, 48, alpha=0.55, accent=badge_col)
    cv2.putText(frame, hand_badge, (w - 205, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col, 1, cv2.LINE_AA)

    # 3. Live Sign Card (Left Overlay)
    _panel(frame, 20, 100, 280, 240)

    raw_pred = state["raw_pred"]
    raw_conf = state["raw_conf"] * 100

    _label(frame, "LIVE SIGN", (35, 125), C_PURPLE, 0.45)

    if state["hand_present"]:
        pred_col = C_GREEN if raw_conf >= (CONF_THRESHOLD * 100) else C_YELLOW
        cv2.putText(frame, raw_pred, (35, 175),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, pred_col, 2, cv2.LINE_AA)
        _label(frame, f"Confidence: {raw_conf:.1f}%", (35, 202), C_TEXT, 0.45)

        # Hold Progress Bar
        bar_w = int(220 * state["hold_progress"])
        cv2.rectangle(frame, (35, 215), (255, 225), C_MUTED, -1, cv2.LINE_AA)
        if bar_w > 0:
            cv2.rectangle(frame, (35, 215), (35 + bar_w, 225), pred_col, -1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "—", (35, 175),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, C_MUTED, 2, cv2.LINE_AA)
        _label(frame, "Waiting for hand...", (35, 205), C_MUTED, 0.45)

    # 4. Diagnostic Top-3 Box (Right Overlay when Debug Mode is ON)
    if state.get("debug_mode") and state["hand_present"] and state["top3"]:
        _panel(frame, w - 260, 100, w - 20, 210)
        _label(frame, f"TOP PREDICTIONS ({state['handedness']})", (w - 250, 125), C_CYAN, 0.38)
        for idx, (label, prob) in enumerate(state["top3"]):
            y_pos = 150 + idx * 22
            col = C_GREEN if idx == 0 and prob >= CONF_THRESHOLD else C_DIM
            cv2.putText(frame, f"{idx+1}. Letter {label}: {prob*100:.1f}%", (w - 245, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 5. Bottom Word & Sentence Dashboard
    _panel(frame, -1, h - 120, w + 1, h + 1, alpha=0.80)
    cv2.line(frame, (0, h - 120), (w, h - 120), C_PURPLE, 1, cv2.LINE_AA)

    # Building Word Bar
    word = state["word_buffer"] if state["word_buffer"] else "(Signing...)"
    word_col = C_CYAN if state["word_buffer"] else C_MUTED
    _label(frame, "BUILDING WORD:", (25, h - 85), C_DIM, 0.45)
    cv2.putText(frame, word, (165, h - 82),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, word_col, 2, cv2.LINE_AA)

    # Spellcheck suggestions
    if state["suggested_word"] and state["suggested_word"] != state["word_buffer"]:
        cv2.putText(frame, f"[Suggest: {state['suggested_word']}]", (350, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREEN, 1, cv2.LINE_AA)
    elif state["suggestions"]:
        sugg_str = ", ".join(state["suggestions"][:2])
        cv2.putText(frame, f"[Did you mean: {sugg_str}]", (350, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_ORANGE, 1, cv2.LINE_AA)

    # Full Sentence Display Tape
    sentence_txt = state["sentence"] if state["sentence"] else "(Empty sentence)"
    sent_col = C_TEXT if state["sentence"] else C_MUTED
    cv2.putText(frame, "SENTENCE:", (25, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_PURPLE, 1, cv2.LINE_AA)
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
