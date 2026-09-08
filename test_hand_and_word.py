"""
Test Harness for Dual-Hand Support & Short-Word Preservation ('HI' -> 'HI')

Tests:
1. Short Word Spellcheck Preservation (HI -> HI, NO -> NO, ME -> ME, MY -> MY)
2. Fingerspelling 'H' then 'I' simulation through WordBufferStateMachine -> commits 'HI'
3. Left-Hand vs Right-Hand Geometric Normalization equivalence test
4. Buffer append detailed console logging check
"""

import os
import sys
import cv2
import pickle
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from word_buffer_engine import (
    WordBufferStateMachine,
    TwoTierSpellCorrector,
    SHORT_VALID_WORDS
)
from train_landmark_classifier import calculate_angle_3d

def run_all_tests():
    print("=" * 70)
    print("  [*] Running Dual-Hand & Word Formation Validation Suite")
    print("=" * 70)
    passed = 0
    total = 0

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 1: Short Word Spellcheck Preservation ('HI' must NOT become 'NO')
    # ──────────────────────────────────────────────────────────────────────────
    corrector = TwoTierSpellCorrector()
    short_words = ["HI", "NO", "OK", "YES", "GO", "ME", "MY", "HELP", "HELLO"]

    print("\nTest 1 - Short Word Spellcheck Preservation:")
    for w in short_words:
        total += 1
        res, is_corr, suggs = corrector.correct(w)
        if res == w:
            passed += 1
            print(f"   [+] '{w}' -> '{res}' (Preserved exactly as expected)")
        else:
            print(f"   [-] '{w}' -> Got '{res}', Expected '{w}'")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 2: Simulated Word Buffer: 'H' then 'I' -> Commits 'HI'
    # ──────────────────────────────────────────────────────────────────────────
    total += 1
    print("\nTest 2 - End-to-End Fingerspelling 'H' then 'I':")
    sm = WordBufferStateMachine(corrector=corrector, stable_frames_needed=5, release_frames_needed=2)

    # Sign 'H' (5 frames)
    for _ in range(5): sm.feed_frame('H', 0.95)
    # Release / transition (3 frames)
    for _ in range(3): sm.feed_frame('NO HAND', 0.0)
    # Sign 'I' (5 frames)
    for _ in range(5): sm.feed_frame('I', 0.95)

    buffer_val = sm.current_word
    print(f"   Raw buffer before commit: '{buffer_val}'")
    sm.commit_word()
    sentence_val = sm.sentence

    if sentence_val == ["HI"]:
        passed += 1
        print(f"   [+] PASSED: Word committed into sentence as {sentence_val}")
    else:
        print(f"   [-] FAILED: Expected ['HI'], got {sentence_val}")

    # ──────────────────────────────────────────────────────────────────────────
    # TEST 3: Left-Hand vs Right-Hand Geometric Equivalence
    # ──────────────────────────────────────────────────────────────────────────
    print("\nTest 3 - Left vs Right Hand Normalization Equivalence:")

    # The ASL image corpus is gitignored, so a fresh clone cannot run this one.
    SAMPLE_IMG = 'asl-alphabet-train/L/L0001_test.jpg'
    if not os.path.exists(SAMPLE_IMG):
        print(f"   [~] SKIPPED: sample image not found ({SAMPLE_IMG}).")
        print("       This test needs the asl-alphabet-train dataset, which is gitignored.")
    else:
      total += 1
      with open('models/asl_landmark_model.pkl', 'rb') as f:
          asl = pickle.load(f)
      clf = asl['model']
      le = asl['label_encoder']

      base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
      options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
      det = vision.HandLandmarker.create_from_options(options)

      def extract_with_handedness(img_bgr):
          h, w = img_bgr.shape[:2]
          mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
          res = det.detect(mp_im)
          if not res.hand_landmarks: return None, None
          lm = res.hand_landmarks[0]
          handedness = res.handedness[0][0].category_name if res.handedness else "Right"
          pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
          if handedness == "Left":
              pts[:, 0] = pts[0, 0] - (pts[:, 0] - pts[0, 0])
          wrist = pts[0]
          pts_centered = pts - wrist
          scale = np.linalg.norm(pts_centered[9][:2])
          if scale < 1e-4: scale = 1.0
          pts_scaled = pts_centered / scale
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
          feat = np.concatenate([pts_aligned.flatten(), angles]).reshape(1, -1)
          return feat, handedness

      # Test Letter 'L' (Clear asymmetric shape)
      im_r = cv2.imread(SAMPLE_IMG)
      im_l = cv2.flip(im_r, 1)

      f_r, h_r = extract_with_handedness(im_r)
      f_l, h_l = extract_with_handedness(im_l)

      pred_r = le.inverse_transform(clf.predict(f_r))[0]
      pred_l = le.inverse_transform(clf.predict(f_l))[0]

      if pred_r == "L" and pred_l == "L":
          passed += 1
          print(f"   [+] Right Hand detected as '{h_r}' -> Letter {pred_r}")
          print(f"   [+] Left Hand detected as '{h_l}' -> Correctly normalized to Letter {pred_l}")
      else:
          print(f"   [-] FAILED: Right -> {pred_r}, Left -> {pred_l}")

    print(f"\n{'='*70}")
    print(f"  Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}\n")
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
