# 🤟 SignLens — Real-Time Sign Language Recognition & Auto-Framing Engine

A production-grade, real-time American Sign Language (ASL) and Indian Sign Language (ISL) translator using **MediaPipe 3D Hand Landmarks**, Neural Network Classifiers, and an **End-to-End Letter-to-Word Auto-Framing & Spellchecking Pipeline**.

---

## ✨ Features
- **MediaPipe Landmark Classification**: Uses 21 normalized 3D hand keypoints invariant to lighting, skin tone, camera distance, and cluttered backgrounds.
- **High Real-World Accuracy**:
  - **ASL Model**: **94.55%** held-out test accuracy across 39 classes.
  - **ISL Model**: **97.90%** held-out test accuracy across 35 classes.
- **Letter-to-Word Auto-Framing**:
  - **Stability Filter**: Eliminates flickering by requiring 8 consecutive stable frames above threshold.
  - **Duplicate Spam Prevention**: Disallows continuous single-gesture spamming.
  - **Word Boundary Detection**: Auto-commits word on `space` gesture or after a 2.0s no-hand timeout.
  - **Spell Correction**: Integrated with `pyspellchecker` to fix minor letter mistakes into valid English vocabulary.
- **Dual Interfaces**:
  1. **Native OpenCV Desktop HUD**: Live skeletal rendering, hold gauge, word buffer, and sentence tape.
  2. **Browser Web Application**: Zero-dependency client-side neural forward pass + Flask backend.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash 
pip install -r requirements.txt
```

### 2. Run Real-Time Desktop App
```bash
python run_native.py
```
**Controls:**
- `1` : Switch to ASL mode
- `2` : Switch to ISL mode
- `b` : Backspace / Delete last letter
- `s` : Space / Commit current word
- `c` : Clear all
- `q` / `ESC` : Quit

---

### 3. Run Web App
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

### 4. Train Landmark Models
```bash
python train_landmark_classifier.py
```
Extracts coordinates, caches features, performs a 70/15/15 stratified Train/Val/Test split, evaluates performance, and outputs confusion matrices.
