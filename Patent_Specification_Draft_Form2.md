# FORM 2
THE PATENTS ACT, 1970
(39 of 1970)
&
THE PATENTS RULES, 2003

# PROVISIONAL SPECIFICATION
(See section 10 and rule 13)

---

### 1. TITLE OF THE INVENTION
**A SYSTEM AND METHOD FOR REAL-TIME DUAL-HANDED SIGN LANGUAGE RECOGNITION AND AUTOMATED WORD-FRAMING USING ROTATION-INVARIANT KEYPOINT TRANSFORMATION**

---

### 2. APPLICANT(S)
1. **Name:** Venmugilrajan R.  
   **Nationality:** Indian  
   **Address:** [Department of Computer Science and Engineering, College Name, City, Tamil Nadu, India]  

2. **Name:** [Project Guide / Supervisor Name]  
   **Nationality:** Indian  
   **Address:** [Department of Computer Science and Engineering, College Name, City, Tamil Nadu, India]  

---

### 3. PREAMBLE TO THE DESCRIPTION
The following specification describes the invention:

---

### 4. FIELD OF THE INVENTION
The present invention relates generally to human-computer interaction, computer vision, and assistive communication technologies. More particularly, the invention relates to an automated, edge-deployable system and method for real-time dual-handed sign language recognition, aspect-ratio-invariant geometric landmark normalization, temporal gesture-release state management, and confidence-weighted lexical word-framing.

---

### 5. BACKGROUND OF THE INVENTION AND PRIOR ART
Sign language is the primary communication medium for individuals who are deaf or hard-of-hearing. Vision-based sign language recognition (SLR) systems have evolved from compute-intensive RGB convolutional neural networks (CNNs) to lightweight 3D skeletal keypoint extractors such as MediaPipe.

However, existing landmark-based translation systems suffer from several critical technical limitations in real-world deployment:
1. **Aspect-Ratio Coordinate Distortion**: Standard webcam streams (e.g., $16:9$, $1280 \times 720$) project normalized coordinate spaces ($[0, 1]$) with non-uniform horizontal scaling compared to square ($1:1$) training distributions.
2. **Handedness Asymmetry**: Prior systems train exclusively on right-hand datasets, leading to high error rates when users sign with their left hand due to opposing lateral knuckle geometries.
3. **Double-Letter Truncation**: Simple stability or hold filters treat repeated identical letters as duplicate noise, dropping the second letter in words such as "HELLO", "PLEASE", or "GOOD".
4. **Lexical Over-Correction**: Generic dictionary algorithms corrupt valid short words (e.g., mutating "HI" into "NO").

There exists an urgent need for an integrated system that achieves rotation and aspect-ratio invariance, dual-hand equivalence, and reliable letter-to-word auto-framing without cloud latency.

---

### 6. OBJECTS OF THE INVENTION
The principal object of the present invention is to provide an aspect-ratio-invariant and rotation-aligned 3D landmark processing pipeline for sign language translation.

Another object of the invention is to provide an automated handedness-detection and geometric inversion module that projects left-hand landmarks into the right-hand feature domain.

Yet another object of the invention is to provide a temporal gesture-release state machine that reliably captures consecutive duplicate letters without spamming.

A further object of the invention is to provide a two-tier, confidence-weighted lexical correction engine with short-word protection.

---

### 7. SUMMARY OF THE INVENTION
The present invention provides a novel system and method for real-time sign language recognition. The system comprises:
- An **Optical Capture Module** capturing live video streams.
- A **Landmark Transformation Processor** converting 21 3D hand keypoints into physical metric space, applying left-to-right geometric mirroring based on detected handedness, scaling by wrist-to-middle knuckle distance, aligning the vertical palm rotation axis, and generating a 78-dimensional invariant feature vector (including 15 knuckle joint flexion angles).
- A **Neural Classification Engine** mapping the feature vector to discrete alphanumeric sign classes.
- A **Gesture-Release State Machine** tracking release transition flags between successive confirmation cycles.
- A **Two-Tier Lexical Engine** filtering words via protected vocabulary lists and confidence-weighted Levenshtein distance penalties.

---

### 8. BRIEF DESCRIPTION OF THE DRAWINGS
- **FIG. 1** illustrates the overall system block diagram and hardware-software architecture.
- **FIG. 2** depicts the geometric transformation pipeline (metric conversion, handedness inversion, and rotation alignment).
- **FIG. 3** shows the flowchart of the temporal gesture-release state machine.
- **FIG. 4** shows the two-tier confidence-weighted lexical correction workflow.

---

### 9. DETAILED DESCRIPTION OF THE INVENTION

#### A. Geometric Invariance and Handedness Normalization Pipeline
1. The video stream captures frames of resolution $W \times H$. MediaPipe extracts 21 spatial landmarks $(x_i, y_i, z_i)$.
2. Physical metric projection: $p_i = [x_i \cdot W, y_i \cdot H, z_i \cdot W]^T$.
3. When the hand detector classifies handedness as "Left", coordinate inversion is performed relative to the wrist $p_0$:
   $$p_{i, x}^{\text{norm}} = p_{0, x} - (p_{i, x} - p_{0, x})$$
4. Palm scale normalization: $p_i' = (p_i - p_0) / \|p_9 - p_0\|_2$.
5. Vertical alignment rotation: Angle $\theta = \text{atan2}(p_{9, x}', -p_{9, y}')$ is calculated, rotating all points by matrix $R(\theta)$.
6. Flexion joint angles: 15 knuckle triad angles $\phi = \frac{1}{\pi} \arccos(\dots)$ are computed across all five digits.
7. Concatenation forms the 78-dimensional descriptor $X \in \mathbb{R}^{78}$.

#### B. Gesture-Release State Machine Architecture
To distinguish deliberate repeated letters from prolonged static holds:
- Upon letter confirmation, `released_since_last_confirm` is set to `False`.
- The state machine requires the hand to transition to `NO HAND` or an alternate gesture for $\ge 2$ consecutive frames before resetting `released_since_last_confirm = True`.
- A candidate letter matching `last_confirmed_letter` is only confirmed if `released_since_last_confirm == True`.

#### C. Two-Tier Confidence-Weighted Spellchecker
- **Tier 0**: Words with length $\le 3$ matching protected vocabulary are passed as-is (0 edit distance).
- **Tier 1**: Lexical candidate searching applies confidence-weighted penalties: modifications on letters with prediction confidence $c_k > 0.85$ are penalized heavily ($1.2$), whereas low confidence letters ($c_k < 0.60$) are penalized lightly ($0.6$).

---

### 10. CLAIMS (Provisional Outline)
We claim:
1. A computer-implemented system for real-time sign language recognition comprising an image sensor, a landmark transformation processor, a neural classifier, and a temporal state machine.
2. The system of claim 1, wherein the landmark processor dynamically mirrors left-hand landmarks into right-hand geometric coordinates upon left-handedness classification.
3. The system of claim 1, wherein palm rotation alignment rotates the wrist-to-middle metacarpophalangeal vector along a fixed vertical axis.
4. The system of claim 1, wherein the temporal state machine utilizes a gesture-release flag to confirm consecutive identical letters while blocking continuous hold spam.
5. The system of claim 1, wherein the lexical engine applies neural prediction confidence weights to edit-distance penalties and protects short-word tokens from modification.

---

Dated this 29th day of August, 2026.

**Signature of Applicants:**  
1. *(Venmugilrajan R.)*  
2. *([Project Guide / Supervisor Name])*  
