# 🤟 SignLens — Real-Time Sign Language Recognition & Auto-Framing Engine

A production-grade, real-time American Sign Language (ASL) and Indian Sign Language (ISL) translator using **MediaPipe 3D Hand Landmarks**, Neural Network Classifiers, and an **End-to-End Letter-to-Word Auto-Framing & Spellchecking Pipeline**.

---

## ✨ Features
- **MediaPipe Landmark Classification**: Uses 21 normalized 3D hand keypoints invariant to lighting, skin tone, camera distance, and cluttered backgrounds.
- **High Real-World Accuracy**:
  - **ASL Model**: **94.55%** held-out test accuracy across 39 classes.
  - **ISL Model (landmark MLP)**: **97.30%** held-out test accuracy across 35 classes,
    measured with a *frame-block* split (see caveat below). This is the trustworthy
    reported figure for ISL.

> **Accuracy caveat — read before quoting these numbers.**
> The ISL dataset contains ~1200 *sequential video frames* per class from a single
> recording session (one signer, one background, one lighting setup). Splitting those
> frames randomly puts near-identical neighbours on both sides of the split and inflates
> accuracy badly: the landmark model scored **99.94%** that way versus **97.30%** under a
> correct contiguous-block split, and the pixel CNN scored **100.0%**.
>
> Block splitting removes *adjacent-frame* leakage but cannot remove *session* leakage,
> because there is only one session per class. Measured: a held-out frame sits 2.7–4.9
> (mean absolute pixel delta) from its nearest same-class training frame but 25–34 from
> any other class, and a maximal 600-frame temporal halving only widens that to 4.1–10.6.
>
> Consequently the **pixel CNN's ISL accuracy (`models/isl_model.pth`) is unvalidated** —
> it can exploit background and lighting shortcuts, so its number is an upper bound, not a
> measured generalization result. The **landmark model's 97.30%** is the figure to quote,
> since its 156-D geometric features discard background and lighting. A genuine
> generalization number requires a second recording session with a different
> signer/background, which this dataset does not contain.
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
