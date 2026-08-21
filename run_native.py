"""
SignLens Native — Real-Time Sign Language Translator & Auto-Framing Engine

Features:
- MediaPipe HandLandmarker (21 3D landmarks, normalized & scale invariant)
- Fast Landmark MLP Classifier (robust against lighting, skin tone, background)
- No-Hand State Detection (prevents false predictions when hand is absent)
- Stability Filter: Confirms a letter only after N consecutive stable frames
- Letter Buffer & Duplicate Protection: Prevents continuous duplicate spamming
- Word Boundary Detection:
    * Predicts 'space' gesture OR triggers 2.0s no-hand timeout
- Letter Deletion: Supports 'del' gesture + 'b' backspace key
- Integrated Spell Checking: Suggests & auto-corrects completed words (pyspellchecker)
- Live HUD: Letter confidence, hold progress, current word buffer, and sentence tape
- Seamless Mode Switching: Press '1' for ASL, '2' for ISL
"""

import os
import cv2
import time
import pickle
import numpy as np
from spellchecker import SpellChecker

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
ASL_MODEL_PATH = os.path.join(BASE_DIR, "models", "asl_landmark_model.pkl")
ISL_MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_landmark_model.pkl")

# Word-framing tuning
CONF_THRESHOLD = 0.65       # Minimum prediction confidence (0.0 to 1.0)
STABLE_FRAMES_NEEDED = 8    # Consecutive frames of same letter required to capture
NO_HAND_WORD_TIMEOUT = 2.0  # Seconds of no hand before auto-committing current word
CAPTURE_COOLDOWN = 1.0      # Seconds before accepting another letter (unless gesture changes)

# MediaPipe Hand Skeleton connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm base
]


class LandmarkSignTranslator:
    def __init__(self):
        # 1. Initialize MediaPipe Hand Detector
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 2. Initialize SpellChecker
        self.spell = SpellChecker()

        # 3. Load Model Registry
        self.models = {}
        self.classes = {}
        self.encoders = {}
        self.current_mode = "asl"
        self._load_models()

        # 4. Pipeline State
        self.word_buffer = ""
        self.sentence = []
        self.last_confirmed_letter = ""
        self.candidate_letter = ""
        self.candidate_count = 0
        self.last_capture_time = 0
        self.last_hand_seen_time = time.time()
        self.no_hand_active = True

    def _load_models(self):
        """Loads available ASL and ISL landmark classifiers."""
        for mode, path in [("asl", ASL_MODEL_PATH), ("isl", ISL_MODEL_PATH)]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.models[mode] = data["model"]
                    self.classes[mode] = [str(c) for c in data["classes"]]
                    self.encoders[mode] = data.get("label_encoder", None)
                    print(f"[+] Loaded {mode.upper()} landmark classifier ({len(data['classes'])} classes) from {path}")
            else:
                print(f"[!] Warning: {mode.upper()} model not found at {path}")

        if "asl" not in self.models and "isl" in self.models:
            self.current_mode = "isl"

    def switch_mode(self, mode: str):
        """Switches between ASL and ISL modes."""
        if mode in self.models:
            self.current_mode = mode
            self.candidate_letter = ""
            self.candidate_count = 0
            print(f"[+] Switched mode to: {mode.upper()}")
            return True
        print(f"[!] Cannot switch to {mode.upper()} (model not loaded).")
        return False

    def extract_landmarks(self, frame_bgr):
        """Runs MediaPipe and extracts 63 normalized coordinates relative to wrist."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            return None, None

        landmarks = results.hand_landmarks[0]

        # Centering and scale normalization
        wrist = landmarks[0]
        scale = np.sqrt((wrist.x - landmarks[9].x) ** 2 + (wrist.y - landmarks[9].y) ** 2)
        if scale < 1e-4:
            scale = 1.0

        features = []
        pixel_points = []
        for lm in landmarks:
            features.extend([
                (lm.x - wrist.x) / scale,
                (lm.y - wrist.y) / scale,
                (lm.z - wrist.z) / scale,
            ])
            pixel_points.append((int(lm.x * w), int(lm.y * h)))

        return np.array(features, dtype=np.float32).reshape(1, -1), pixel_points

    def predict(self, features):
        """Predicts class and confidence from extracted landmark vector."""
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

    def commit_word(self):
        """Commits current word buffer into the sentence with auto-spellchecking."""
        raw_word = self.word_buffer.strip()
        if not raw_word:
            return

        # Apply spellchecker
        corrected = self.spell.correction(raw_word.lower())
        final_word = corrected.upper() if corrected else raw_word

        self.sentence.append(final_word)
        self.word_buffer = ""
        self.candidate_letter = ""
        self.candidate_count = 0
        self.last_confirmed_letter = ""

    def process_frame(self, frame):
        """
        Processes a single video frame through the full prediction & auto-framing pipeline.
        """
        now = time.time()
        features, pixel_points = self.extract_landmarks(frame)

        raw_pred = "NO HAND"
        raw_conf = 0.0
        top3 = []

        if features is not None:
            self.last_hand_seen_time = now
            self.no_hand_active = False

            label, conf, top3 = self.predict(features)
            raw_pred = label
            raw_conf = conf

            # Check confidence threshold
            if conf >= CONF_THRESHOLD:
                # Stability filter
                if label == self.candidate_letter:
                    self.candidate_count += 1
                else:
                    self.candidate_letter = label
                    self.candidate_count = 1

                # Confirm letter if held stable
                if self.candidate_count >= STABLE_FRAMES_NEEDED:
                    # Prevent spamming identical letter while held
                    can_capture = (
                        (label != self.last_confirmed_letter) or
                        ((now - self.last_capture_time) > CAPTURE_COOLDOWN)
                    )

                    if can_capture:
                        self._handle_confirmed_letter(label)
                        self.last_confirmed_letter = label
                        self.last_capture_time = now
                        self.candidate_count = 0
            else:
                self.candidate_count = max(0, self.candidate_count - 1)

        else:
            # No hand detected
            self.candidate_count = 0
            self.candidate_letter = ""
            self.last_confirmed_letter = ""

            # Check no-hand word boundary timeout
            if (not self.no_hand_active) and (now - self.last_hand_seen_time >= NO_HAND_WORD_TIMEOUT):
                if self.word_buffer:
                    self.commit_word()
                self.no_hand_active = True

        suggested = ""
        if self.word_buffer:
            corr = self.spell.correction(self.word_buffer.lower())
            suggested = corr.upper() if corr else self.word_buffer

        return {
            "hand_present": features is not None,
            "landmarks": pixel_points,
            "raw_pred": raw_pred,
            "raw_conf": raw_conf,
            "top3": top3,
            "hold_progress": min(1.0, self.candidate_count / float(STABLE_FRAMES_NEEDED)),
            "word_buffer": self.word_buffer,
            "sentence": " ".join(self.sentence),
            "suggested_word": suggested
        }

    def _handle_confirmed_letter(self, letter: str):
        """Dispatches confirmed gesture to buffer modification or word boundaries."""
        l_lower = letter.lower()
        if l_lower == "space" or l_lower == "_":
            self.commit_word()
        elif l_lower == "del" or l_lower == "delete":
            if self.word_buffer:
                self.word_buffer = self.word_buffer[:-1]
        elif l_lower == "nothing":
            pass  # Background/neutral gesture
        else:
            self.word_buffer += letter.upper()

    def backspace(self):
        """Manual backspace key handler."""
        if self.word_buffer:
            self.word_buffer = self.word_buffer[:-1]
        elif self.sentence:
            self.sentence.pop()

    def clear_all(self):
        """Clears word buffer and entire sentence."""
        self.word_buffer = ""
        self.sentence = []
        self.candidate_count = 0
        self.candidate_letter = ""
        self.last_confirmed_letter = ""


# ─── UI HUD Drawing ────────────────────────────────────────────────────────────
def draw_hud(frame, state, mode_name, fps):
    h, w = frame.shape[:2]

    # 1. Draw hand skeletal connections if present
    if state["landmarks"]:
        pts = state["landmarks"]
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[p1_idx], pts[p2_idx], (0, 212, 255), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            color = (255, 107, 157) if i in [4, 8, 12, 16, 20] else (34, 214, 122)
            cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Top Header Banner
    cv2.rectangle(frame, (0, 0), (w, 80), (14, 14, 26), -1)
    cv2.line(frame, (0, 80), (w, 80), (45, 45, 75), 2)

    # Title & Mode
    cv2.putText(frame, f"SignLens | {mode_name.upper()} Mode", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f} | [1] ASL  [2] ISL  [b] Del  [c] Clear  [q] Quit", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 200), 1, cv2.LINE_AA)

    # Hand Presence Badge (Top Right)
    hand_badge = "HAND DETECTED" if state["hand_present"] else "NO HAND"
    badge_col = (34, 214, 122) if state["hand_present"] else (100, 100, 120)
    cv2.rectangle(frame, (w - 180, 15), (w - 20, 48), (25, 25, 45), -1)
    cv2.rectangle(frame, (w - 180, 15), (w - 20, 48), badge_col, 1)
    cv2.putText(frame, hand_badge, (w - 165, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col, 1, cv2.LINE_AA)

    # 3. Real-Time Detection Card (Left Overlay)
    cv2.rectangle(frame, (20, 100), (280, 240), (18, 18, 35), -1)
    cv2.rectangle(frame, (20, 100), (280, 240), (60, 60, 100), 1)

    raw_pred = state["raw_pred"]
    raw_conf = state["raw_conf"] * 100

    cv2.putText(frame, "LIVE SIGN", (35, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 132, 255), 1, cv2.LINE_AA)

    if state["hand_present"]:
        pred_col = (34, 214, 122) if raw_conf >= 65 else (0, 200, 255)
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

    # 4. Bottom Word & Sentence Dashboard
    cv2.rectangle(frame, (0, h - 120), (w, h), (14, 14, 26), -1)
    cv2.line(frame, (0, h - 120), (w, h - 120), (45, 45, 75), 2)

    # Building Word Bar
    word = state["word_buffer"] if state["word_buffer"] else "(Signing...)"
    word_col = (0, 212, 255) if state["word_buffer"] else (120, 120, 150)
    cv2.putText(frame, "BUILDING WORD:", (25, h - 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, word, (165, h - 82),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, word_col, 2, cv2.LINE_AA)

    # Spellcheck suggestion tag
    if state["suggested_word"] and state["suggested_word"] != state["word_buffer"]:
        cv2.putText(frame, f"[Spellcheck: {state['suggested_word']}]", (350, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 214, 122), 1, cv2.LINE_AA)

    # Full Sentence Display Tape
    sentence_txt = state["sentence"] if state["sentence"] else "(Empty sentence)"
    sent_col = (255, 255, 255) if state["sentence"] else (100, 100, 120)
    cv2.putText(frame, "SENTENCE:", (25, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 132, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, sentence_txt, (120, h - 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, sent_col, 2, cv2.LINE_AA)


def main():
    print("=" * 70)
    print("  SignLens — Real-Time Sign Language Translator (MediaPipe Edition)")
    print("=" * 70)
    print("Controls:")
    print("  [1] Switch to ASL Mode")
    print("  [2] Switch to ISL Mode")
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
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Flip horizontally for natural selfie/mirror view
        frame = cv2.flip(frame, 1)

        # FPS calculation
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, (curr_time - prev_time)))
        prev_time = curr_time

        # Run pipeline
        state = translator.process_frame(frame)

        # Draw Overlay
        draw_hud(frame, state, translator.current_mode, fps)

        cv2.imshow("SignLens — Real-Time Sign Language Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            translator.switch_mode("asl")
        elif key == ord('2'):
            translator.switch_mode("isl")
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
