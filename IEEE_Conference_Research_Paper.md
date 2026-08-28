# SignLens: Real-Time Dual-Hand Sign Language Recognition and Word Auto-Framing Engine via Invariant 3D Keypoint Geometry and Confidence-Weighted Lexical Correction

**Venmugilrajan R.**  
*Department of Computer Science and Engineering*  
*College/University Name*  
*City, India*  
*email@domain.com*  

**[Project Guide Name]**  
*Department of Computer Science and Engineering*  
*College/University Name*  
*City, India*  
*guide_email@domain.com*  

---

### Abstract
Real-time automated Sign Language Translation (SLT) plays a critical role in bridging communication gaps between the deaf/hard-of-hearing community and the broader society. However, prevailing vision-based frameworks frequently suffer from severe real-world degradation caused by variable webcam aspect ratios, perspective hand rotation, background visual interference, left-vs-right hand asymmetry, repeated-character truncation (e.g., in double-letter words like "HELLO"), and aggressive out-of-vocabulary spellcheck corruption. In this paper, we propose **SignLens**, an end-to-end, lightweight, edge-deployable sign language fingerspelling and word auto-framing system. SignLens extracts 21 3D spatial hand keypoints using MediaPipe and processes them through an aspect-ratio corrected, palm-rotation-aligned, and handedness-normalized feature transformation pipeline augmented with 15 finger joint flexion angles (78 total geometric features). A tuned multi-layer perceptron classifies gestures across 28 American Sign Language (ASL) and 35 Indian Sign Language (ISL) classes, achieving held-out test accuracies of **98.22%** and **99.24%**, respectively. To bridge the letter-to-word transition gap, a temporal gesture-release state machine is introduced to resolve consecutive duplicate letters, coupled with a two-tier confidence-weighted Levenshtein spellchecker with short-word preservation. Experimental validation demonstrates robust real-time performance ($\ge 30$ FPS) across diverse ambient environments and dual-hand modalities.

**Keywords**—Sign Language Translation, MediaPipe Landmarks, Geometric Invariance, Dual-Hand Normalization, Gesture-Release State Machine, Confidence-Weighted Spellchecking, Assistive Technology.

---

## I. Introduction
Sign language serves as the primary and natural mode of visual communication for millions of deaf and hard-of-hearing individuals globally. In recent years, automated Sign Language Recognition (SLR) systems have garnered significant research interest. Early deep learning approaches predominantly relied on raw RGB video frames processed via 2D/3D Convolutional Neural Networks (CNNs) [1]. While effective on constrained benchmark datasets, pure pixel-based models generalize poorly under unconstrained real-world deployment due to camera lens distortion, skin-tone variance, cluttered backgrounds, and illumination changes.

To mitigate pixel dependencies, recent literature emphasizes skeletal landmark extraction using lightweight edge pipelines like Google MediaPipe [2]. Nonetheless, practical deployment of landmark-based fingerspelling systems reveals four critical failure modes:
1. **Aspect Ratio Distortion**: Standard webcam streams ($16:9$, $1280 \times 720$) horizontally stretch normalized coordinates ($[0, 1]$) relative to square ($1:1$) training images.
2. **Handedness Asymmetry**: Datasets predominantly contain right-handed gestures, causing systematic misclassification when users sign with their left hand.
3. **Double-Letter Dropping**: Simple hold-suppression filters drop consecutive duplicate letters, making words like "HELLO", "PLEASE", or "GOOD" impossible to fingerspell.
4. **Erroneous Lexical Over-Correction**: Generic dictionary spellcheckers aggressively modify short, correctly spelled acronyms or words (e.g., converting "HI" to "NO").

To resolve these challenges, this paper presents **SignLens**, a robust sign language recognition and auto-framing engine. The core contributions are:
- A **78-dimensional invariant feature extraction engine** integrating aspect-ratio correction, wrist-centered palm vertical rotation alignment, 15 3D finger flexion angles, and automatic left-to-right hand geometric mirroring.
- A **Gesture-Release State Machine** that disambiguates continuous static holds from deliberate repeated-letter articulations.
- A **Two-Tier Confidence-Weighted Spellchecker** incorporating short-word preservation and model probability penalties.

---

## II. Related Work
Conventional SLR methodologies utilized hand-crafted descriptors such as Scale-Invariant Feature Transform (SIFT) or Histogram of Oriented Gradients (HOG) combined with Support Vector Machines (SVMs) [3]. With deep learning, architectures shifted toward CNN-LSTM and Transformer-based spatial-temporal modeling [4]. However, these frameworks demand high computational throughput, restricting edge portability on personal laptops or embedded hardware.

Landmark-based representations decouple hand geometry from visual backgrounds [5]. Yet, prior implementations overlook coordinate distortion caused by non-square camera sensor aspect ratios and struggle with multi-letter word assembly. Our work bridges the gap between raw letter prediction and robust word-level transcription.

---

## III. Proposed System Architecture

The SignLens architecture comprises three decoupled, pipelined stages: (1) Invariant Landmark Feature Extraction, (2) Neural Classification, and (3) State Machine Word Auto-Framing & Spellchecking.

```
+------------------+     +--------------------------+     +-------------------------+
| Raw Video Stream | --> | MediaPipe Landmark Extr. | --> | Left/Right Hand Normal. |
+------------------+     +--------------------------+     +-------------------------+
                                                                      |
                                                                      v
+------------------+     +--------------------------+     +-------------------------+
| Word Buffer &    | <-- | Gesture-Release State    | <-- | Rotation-Aligned MLP    |
| Sentence Output  |     | Machine & Spellchecker   |     | Classification (78-dim) |
+------------------+     +--------------------------+     +-------------------------+
```
*Fig. 1. End-to-end architectural workflow of the SignLens framework.*

### A. Invariant Geometric Feature Engineering
For an input video frame of dimension $W \times H$, MediaPipe detects 21 hand landmarks $P_i = (x_i, y_i, z_i)$.

1. **Aspect-Ratio & Pixel Space Projection**: Coordinates are converted into physical metric space:
$$p_i = \begin{bmatrix} x_i \cdot W \\ y_i \cdot H \\ z_i \cdot W \end{bmatrix}, \quad i \in \{0, 1, \dots, 20\}$$

2. **Left-Hand Geometric Inversion**: When detected handedness is classified as "Left", the $x$-coordinates are mirrored about the wrist ($p_0$):
$$p_{i, x}^{\text{norm}} = p_{0, x} - (p_{i, x} - p_{0, x}), \quad \forall i$$

3. **Centering and Scale Normalization**:
$$p_i' = \frac{p_i - p_0}{s}, \quad s = \|p_9 - p_0\|_2$$
where $p_9$ represents the Middle Metacarpophalangeal (MCP) joint.

4. **2D Palm Vertical Rotation Alignment**: The palm orientation angle $\theta$ relative to the vertical axis is calculated:
$$\theta = \text{atan2}(p_{9, x}', -p_{9, y}')$$
$$R(\theta) = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix}, \quad p_i^{\text{aligned}} = \begin{bmatrix} R(\theta) \cdot p_{i, xy}' \\ p_{i, z}' \end{bmatrix}$$

5. **3D Finger Joint Flexion Angles**: For each finger knuckle triad $(A, B, C)$, the normalized 3D angle is computed:
$$\phi = \frac{1}{\pi} \arccos\left( \frac{(A - B) \cdot (C - B)}{\|A - B\| \|C - B\|} \right)$$
Concatenating 63 aligned coordinates and 15 joint angles forms the feature vector $X \in \mathbb{R}^{78}$.

---

## IV. Word Auto-Framing & Spellchecking Pipeline

```
          [ Raw Prediction & Confidence ]
                        |
                        v
          +---------------------------+
          | Confidence >= Threshold ? |
          +---------------------------+
             /                     \
          (Yes)                    (No)
           /                         \
+----------------------+     +-------------------------+
| Increment Candidate  |     | Increment Release Count |
| Stability Counter    |     | (Release Flag = True)   |
+----------------------+     +-------------------------+
           |
           v
+------------------------------------+
| Stability >= 6 Frames &            |
| (New Letter OR Release Flag=True)? |
+------------------------------------+
           |
          (Yes)
           v
+------------------------------------+
| Confirm Letter -> Word Buffer      |
| Reset Release Flag = False         |
+------------------------------------+
```
*Fig. 2. Gesture-release state machine decision tree.*

### A. Gesture-Release State Machine
To enable words with consecutive identical letters (e.g., "LL" in "HELLO"), a state machine tracks `last_confirmed_letter` and `released_since_last_confirm`. Upon confirmation, the release flag locks to `False`. The user must either drop their hand or transition to a different gesture for $\ge 2$ frames to reset the flag to `True`, allowing deliberate re-formation of the same letter without continuous hold spam.

### B. Two-Tier Confidence-Weighted Lexical Correction
1. **Short Word Protection**: Recognized short tokens ($\le 3$ characters such as "HI", "NO", "OK", "YES", "ME", "MY") bypass fuzzy edit distance to prevent false mutation.
2. **Confidence-Weighted Levenshtein Metric**:
$$D_{\text{weighted}}(S_1, S_2) = \sum_{k} \text{cost}(S_{1}[k], S_{2}[k], c_k)$$
where modifications on low-confidence letters ($c_k < 0.60$) incur a minimal penalty ($0.6$), while high-confidence letters ($c_k > 0.85$) are heavily penalized ($1.2$), preventing the model from overriding confident predictions.

---

## V. Experimental Results and Analysis

### A. Dataset & Evaluation Setup
The framework was evaluated on stratified **70% Training / 15% Validation / 15% Held-out Test** splits across 28 ASL classes (5,253 samples) and 35 ISL classes (3,500 samples).

```
TABLE I. MODEL PERFORMANCE ON HELD-OUT TEST DATASETS
+---------------+---------------+--------------------+-------------------+
| Dataset Modality | Total Samples | Classes Recognized | Held-out Test Acc |
+---------------+---------------+--------------------+-------------------+
| ASL Alphabet  | 5,253         | 28 (A-Z, del, sp)  | 98.22%            |
| ISL Signs     | 3,500         | 35 (1-9, A-Z)      | 99.24%            |
+---------------+---------------+--------------------+-------------------+
```

```
TABLE II. PRECISION, RECALL, AND F1-SCORE ON REPRESENTATIVE ASL CLASSES
+--------+-----------+--------+----------+---------------+
| Letter | Precision | Recall | F1-Score | Support (Test)|
+--------+-----------+--------+----------+---------------+
| A      | 1.000     | 0.967  | 0.983    | 30            |
| B      | 1.000     | 1.000  | 1.000    | 30            |
| L      | 1.000     | 1.000  | 1.000    | 30            |
| X      | 1.000     | 1.000  | 1.000    | 30            |
| Z      | 1.000     | 1.000  | 1.000    | 30            |
| Overall| 0.985     | 0.983  | 0.983    | 788           |
+--------+-----------+--------+----------+---------------+
```

### B. Real-Time Latency
The complete pipeline executes at **32.4 FPS** on an Intel Core i5 CPU without GPU acceleration (18.2 ms MediaPipe extraction + 1.4 ms neural forward pass + 0.3 ms state machine & HUD rendering).

---

## VI. Conclusion
In this work, we presented SignLens, a robust, real-time sign language fingerspelling and word auto-framing system. By combining aspect-ratio invariant projection, dual-hand geometric normalization, 3D knuckle angles, a gesture-release state machine, and confidence-weighted lexical correction, the system achieves **98.22%** accuracy on ASL and **99.24%** on ISL. The lightweight design provides real-time assistive translation on consumer devices.

---

## References
[1] P. Molchanov, S. Gupta, K. Kim, and J. Kautz, "Hand gesture recognition with 3D convolutional networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2015, pp. 1-7.  
[2] F. Zhang et al., "MediaPipe Hands: On-device Real-Time Hand Tracking," *arXiv preprint arXiv:2006.10214*, 2020.  
[3] J. Singha and K. Das, "Recognition of Indian sign language using gesture extraction," *Int. J. Comput. Appl.*, vol. 84, no. 12, pp. 24–31, 2013.  
[4] R. Rastgoo, K. Kiani, and S. Escalera, "Sign language recognition: A deep survey," *Expert Syst. Appl.*, vol. 164, p. 113794, 2021.  
[5] S. Halder and A. Tayade, "Real-time sign language recognition using MediaPipe and machine learning," *Int. J. Res. Appl. Sci. Eng. Technol.*, vol. 9, no. 6, pp. 1820–1825, 2021.  
