import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_form2():
    doc = Document()
    
    # Page setup - Standard Indian Patent Office margins (1 inch / 25.4mm)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        section.page_width = Inches(8.27)  # A4
        section.page_height = Inches(11.69)
        
    # Default styling
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    def add_p(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, italic=False, size=12, space_after=6):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size)
        return p

    def add_title_header():
        add_p("FORM 2", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=14, space_after=2)
        add_p("THE PATENTS ACT 1970 (39 of 1970)", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=2)
        add_p("&", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=11, space_after=2)
        add_p("THE PATENTS (AMENDMENT) RULES, 2006", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=2)
        add_p("COMPLETE SPECIFICATION", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=2)
        add_p("(See section 10 and rule 13)", WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=11, space_after=14)

    add_title_header()

    # 1. TITLE
    add_p("1. TITLE OF THE INVENTION:", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("A REAL-TIME VISION-BASED SIGN LANGUAGE RECOGNITION SYSTEM USING GEOMETRIC LANDMARKS AND MULTI-LAYER PERCEPTRON", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=14)

    # 2. APPLICANTS
    add_p("2. APPLICANTS:", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("Bannari Amman Institute of Technology, Sathyamangalam, Erode, Tamil Nadu, India – 638401.", WD_ALIGN_PARAGRAPH.LEFT, size=12, space_after=14)

    # 3. PREAMBLE
    add_p("3. PREAMBLE TO THE DESCRIPTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("COMPLETE", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("The following specification particularly describes the invention and the manner in which it is to be performed.", WD_ALIGN_PARAGRAPH.LEFT, italic=True, size=12, space_after=18)

    # 4. DESCRIPTION
    add_p("4. DESCRIPTION", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=12)

    add_p("FIELD OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0001] The present invention relates to computer vision, assistive human-computer interaction (HCI), and deep learning systems. More particularly, the invention relates to a real-time, client-side vision-based sign language recognition system configured to recognize both single-hand American Sign Language (ASL) and dual-hand Indian Sign Language (ISL) gestures using kinematic geometric landmarks and multi-layer perceptron neural networks executing directly on consumer-grade computing hardware.")

    add_p("BACKGROUND OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0002] Deaf and hard-of-hearing communities rely heavily on sign languages such as Indian Sign Language (ISL) and American Sign Language (ASL) for daily communication. However, the lack of widespread fluency among the non-signing general public creates severe communication barriers in healthcare, education, workplace, and public service environments.")
    add_p("[0003] Conventional vision-based sign language recognition solutions primarily depend on heavy end-to-end 2D or 3D Convolutional Neural Networks (CNNs) processing raw RGB pixel video streams. These pixel-based approaches suffer from high computational complexity, high latency, high memory footprints, and heavy dependence on background illumination, skin tone variance, and clothing colors.")
    add_p("[0004] Furthermore, such compute-intensive CNN models often cannot execute in real-time on edge devices or client-side web browsers without offloading video feeds to remote cloud servers. Cloud-dependent video streaming introduces privacy risks, network latency, and operational failure in low-connectivity or offline regions.")
    add_p("[0005] In addition, Indian Sign Language (ISL) poses unique technical difficulties because 20 of its 26 alphabet characters require dual-hand coordination, where hand-to-hand occlusions, variable camera distances, and handedness dominance variations may cause existing landmark classification models to fail.")
    add_p("[0006] Accordingly, there exists a critical need for an intelligent, privacy-preserving sign language translation system that normalizes dual-hand geometric landmarks, extracts kinematic flexion angles, and executes multi-dialect gesture classification at real-time speeds with low latency directly on client hardware without transmitting video feeds to external servers.")

    add_p("OBJECTIVES OF THE PRESENT INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0007] The primary objective of the present invention is to provide a real-time, vision-based sign language translation system capable of executing full inference directly on client-side consumer hardware and edge CPU environments without requiring graphics processing units.")
    add_p("[0008] Another objective is to provide a geometric feature normalization framework that establishes distance-invariance, translation-invariance, and rotation alignment establishing canonical orientation for robust single-hand and dual-hand tracking.")
    add_p("[0009] Another objective is to compute fifteen inter-joint finger flexion angles as kinematic descriptors that decouple hand shape understanding from camera perspective distortion.")
    add_p("[0010] Another objective is to provide multi-dialect translation supporting both 28 single-hand ASL classes and 35 dual-hand ISL classes within a unified lightweight neural framework comprising single-hand and dual-hand classification pathways.")
    add_p("[0011] Another objective is to provide a temporal word buffer engine with streak confirmation and cooldown thresholds that reduce flicker, gesture transition noise, and repetition artifacts.")
    add_p("[0012] Another objective is to integrate an interactive 3D skeletal avatar synthesizer that converts input text into animated, dialect-accurate sign sequences with synchronized human reference previews.")
    add_p("[0013] Another objective is to eliminate video transmission to cloud servers, thereby preserving visual privacy and enabling low-latency on-device processing.")

    add_p("SUMMARY OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0014] The present invention provides a real-time vision-based sign language recognition system comprising an optical sensor interface, a 3D skeletal tracking module, a geometric feature normalization module, a multi-layer perceptron (MLP) classification engine, a temporal word-formation engine, and an interactive bidirectional avatar rendering engine.")
    add_p("[0015] The optical sensor captures continuous video frames of a user performing sign gestures at a frame rate of at least 30 frames per second.")
    add_p("[0016] The 3D skeletal tracking module extracts 21 spatial landmarks (x, y, z) per detected hand, capturing wrist and finger joint landmarks across anatomical articulations.")
    add_p("[0017] The geometric feature normalization module centers coordinates around the wrist joint, scales landmark vectors by palm span dimensions to achieve distance invariance, applies rotation alignment along the canonical palm axis, and computes 15 inter-joint finger flexion angles across adjacent phalangeal segments.")
    add_p("[0018] For single-hand ASL gestures, a 78-dimensional feature vector is generated and classified via an MLP network (256 → 128 → 64 neurons). For dual-hand ISL gestures, a 156-dimensional dual-hand feature vector is generated and classified via an MLP network (512 → 256 → 128 neurons).")
    add_p("[0019] The word buffer engine evaluates consecutive classification outputs against a streak threshold (N consecutive identical frames) and a post-capture cooldown timer (ranging from 500 to 1000 milliseconds, configured at 600 ms), automatically aggregating confirmed letters into words and sentences with automated Text-to-Speech (TTS) audio synthesis.")
    add_p("[0020] A reverse translation module converts typed or spoken text into forward animated 3D avatar gestures, synchronizing interpolated kinematic joint motions with human visual reference cards.")

    add_p("BRIEF DESCRIPTION OF THE ACCOMPANYING DRAWINGS", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0021] Figure 1 illustrates a flow diagram representing the real-time sign language recognition, landmark normalization, neural classification, and word aggregation process.")
    add_p("[0022] Figure 2 illustrates a block diagram of the overall system architecture, including optical input, 3D skeletal tracking, geometric feature engineering, dual-dialect MLP engines, temporal stabilization buffer, and bidirectional avatar synthesizer.")

    add_p("DETAILED DESCRIPTION OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0023] The sign language recognition process commences with initialization of an optical video capture stream (100). The camera sensor continuously acquires video frames (101) containing one or two hands of a signing user at resolutions ranging from 640x480 to 1920x1080 pixels.")
    add_p("[0024] The acquired video frames are processed locally (102) by a lightweight 3D hand landmarker pipeline, such as MediaPipe Tasks Vision. The landmarker pipeline identifies the presence of hands, categorizes handedness (Left Hand, Right Hand), and extracts 21 three-dimensional landmark points per hand corresponding to anatomical wrist and finger joints.")
    add_p("[0025] The raw pixel coordinates (x·W, y·H, z·W) are transmitted to a geometric normalization module (103). To ensure translation invariance, the wrist coordinate (x0, y0, z0) is subtracted from all 21 landmark locations, shifting the local coordinate origin directly to the signer's wrist.")
    add_p("[0026] To ensure scale and distance invariance, the Euclidean distance, computed in the image plane (x, y), between the wrist landmark and the middle finger metacarpophalangeal joint landmark (L0 to L9) is determined as the base palm length D_palm. All 21 centered landmark coordinates are subsequently normalized by dividing by D_palm, making feature representation substantially consistent whether the user is seated near or far from the optical sensor. Furthermore, a 2D planar rotation alignment is applied by determining the angular orientation of the vector connecting landmark L0 to landmark L9 with respect to the vertical 12 o'clock axis (0, -1) using an arctangent function, and rotating all centered 2D planar coordinates by an orthogonal rotation matrix, thereby establishing canonical vertical palm alignment across varying hand roll angles.")
    add_p("[0027] Simultaneously, fifteen (15) inter-joint finger flexion angles θ_k (where k = 1 to 15) are extracted (104). The fifteen angles correspond to three anatomical flexion triplets per digit across all five digits defined by landmark indices: for the thumb, triplets [0, 1, 2], [1, 2, 3], and [2, 3, 4]; for the index finger, triplets [0, 5, 6], [5, 6, 7], and [6, 7, 8]; for the middle finger, triplets [0, 9, 10], [9, 10, 11], and [10, 11, 12]; for the ring finger, triplets [0, 13, 14], [13, 14, 15], and [14, 15, 16]; and for the little finger, triplets [0, 17, 18], [17, 18, 19], and [18, 19, 20], wherein index 0 designates the wrist landmark and each triplet specifies an intermediate joint vertex b formed by adjacent vectors ba = a - b and bc = c - b. Each flexion angle θ_k is calculated according to: θ_k = (1/π) * arccos((ba · bc) / (||ba|| * ||bc|| + ε)), wherein ε is a predetermined small positive constant (1e-6) preventing division by zero, and (1/π) normalizes the angular value into a continuous dimensionless range of [0, 1]. These normalized angular descriptors provide reduced sensitivity to camera tilt and perspective projection, as well as signer hand dimension variances.")
    add_p("[0028] For American Sign Language (ASL) mode, the 63 normalized coordinates (21 x 3) and 15 flexion angles are concatenated to form a 78-dimensional feature vector (105). A handedness mirroring step is selectively applied to left-hand detections to map coordinates into right-handed canonical space.")
    add_p("[0029] For Indian Sign Language (ISL) mode, coordinates from a left-most hand and a right-most hand in the image are extracted. Detected hands are spatially sorted along the horizontal image axis, and each hand is independently normalized into a 78-dimensional sub-vector. When only a single hand is present in single-hand ISL signs, comprising the six alphabet letters 'C', 'I', 'L', 'O', 'U', 'V' and the nine numerical digits '1' through '9', the absent hand is padded with a predetermined padding vector, such as a zero vector or a neutral rest-state landmark vector (106), producing a uniform 156-dimensional dual-hand feature vector for consistent neural processing across all 35 ISL classes.")
    add_p("[0030] The normalized feature vector is supplied to an on-device Multi-Layer Perceptron (MLP) neural classifier (107). For ASL, the MLP comprises dense layers with dimensions 78 → 256 → 128 → 64 → 28 output classes, comprising 26 alphabet letters (A through Z) and two control gestures representing space and delete, with ReLU activations and Softmax probability distribution. For ISL, the MLP comprises dense layers with dimensions 156 → 512 → 256 → 128 → 35 output classes. In an embodiment evaluated on held-out test sets, each of the single-hand and dual-hand classifiers achieved a test accuracy exceeding 98%. In an embodiment, the neural network weights may be stored in quantized integer representations. The neural weights are optimized to execute forward passes in under 5 milliseconds on standard client CPU threads without requiring graphics accelerator hardware, achieving a substantial technical reduction in computational power and memory footprint compared to pixel-based convolutional neural networks.")
    add_p("[0031] The classification output yields a predicted character C_t and confidence score P_t (108). To avoid erroneous classifications during dynamic hand transitions, a temporal streak verification engine evaluates whether C_t remains unchanged for N consecutive frames (with N predetermined as 5) with P_t >= 0.70 (109).")
    add_p("[0032] Once the streak threshold is satisfied, the character is locked and transferred into an active word buffer (110), triggering an instantaneous post-capture cooldown timer (configured at 600 ms, with an operable range from 500 ms to 1000 ms) that inhibits repeated duplicate character capture unless a hand release or transition is detected.")
    add_p("[0033] Dropping or removing hands from the active optical capture frame for a predetermined duration automatically commits the active word buffer into a completed sentence string (111). The aggregated sentence is routed to an audio synthesis module for automatic text-to-speech voicing and clipboard sharing.")
    add_p("[0034] In bidirectional practice mode, arbitrary user text or optional spoken voice input is parsed into individual grapheme tokens (112). Where speech input is optionally employed, speech-to-text conversion may be executed via a client platform speech interface, and the aforementioned on-device visual privacy safeguards apply specifically to the optical video frames and hand landmark data. A 3D skeletal avatar engine executes step-by-step cubic interpolation across stored kinematic joint poses, rendering smooth continuous hand, arm, and facial gestures on a 2D/3D canvas (113).")
    add_p("[0035] Synchronized alongside the animated avatar, a visual sign preview reference card is dynamically populated in the graphical interface, presenting high-resolution photographic demonstrations and explicit hand-count rules (1-Hand vs 2-Hand) corresponding to the active sign step.")
    add_p("[0036] The overall hardware architecture (201) comprises an optical video capture module (202), an edge processing CPU unit (203), a memory storage unit (204), a display output screen (205), and an audio loudspeaker synthesizer (206).")
    add_p("[0037] The edge processing CPU unit (203) executes the local landmark extraction pipeline (207), the geometric normalization and angle extraction engine (208), the dual-dialect MLP inference engine (209), the temporal streak aggregation module (210), and the bidirectional skeletal avatar synthesizer (211).")
    add_p("[0038] All optical video frames and landmark calculations are processed exclusively within local volatile memory (204) of the edge processor, and raw video frames are discarded immediately after landmark vector computation. No video streams or biometric facial frames are transmitted to external cloud servers, preserving visual privacy and enabling operation without network connectivity once the models are loaded.")
    add_p("[0039] The system provides full responsive execution across desktop computer displays, tablet terminals, and narrow mobile touchscreens using dynamic viewport constraints, CSS container sizing, and responsive canvas scaling.")

    # 5. CLAIMS
    doc.add_page_break()
    add_p("5. CLAIMS", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=12)
    add_p("We claim:", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=8)

    add_p("1. A real-time vision-based sign language recognition system comprising:\n"
          "   an optical sensor module (202) configured to acquire continuous video frames of a user;\n"
          "   an edge processing unit (203) operatively coupled to the optical sensor (202);\n"
          "   a display module (205) and an audio synthesizer (206) coupled to the edge processing unit (203);\n"
          "   the edge processing unit (203) configured to execute:\n"
          "     a 3D skeletal landmark extraction module (207) configured to extract 21 three-dimensional spatial coordinates representing wrist and finger joint landmarks for each detected hand;\n"
          "     a geometric normalization engine (208) configured to translate the spatial coordinates to a wrist-centered local origin, scale the centered coordinates by a Euclidean distance computed between wrist and middle metacarpophalangeal joints, and compute fifteen inter-joint finger flexion angles comprising three angular triplets per digit across five digits of each detected hand;\n"
          "     a multi-layer perceptron (MLP) neural classifier (209) configured to operate in a single-hand mode and a dual-hand mode, receive a feature vector comprising the normalized coordinates and flexion angles, and output predicted sign characters and confidence scores; and\n"
          "     a temporal word buffer engine (210) configured to aggregate confirmed sign characters into complete sentences using consecutive streak verification and post-capture cooldown suppression.")

    add_p("2. A method for real-time sign language recognition, the method executed on an edge processing unit (203) coupled to an optical sensor (202), comprising:\n"
          "   capturing optical video frames of a user via the optical sensor (202);\n"
          "   extracting 21 three-dimensional spatial coordinates for each detected hand in local volatile memory (204);\n"
          "   translating the spatial coordinates to a wrist-centered local origin;\n"
          "   scaling the centered coordinates by a Euclidean distance computed between wrist and middle metacarpophalangeal joints to achieve distance invariance;\n"
          "   calculating fifteen inter-joint finger flexion angles using adjacent joint vector dot products across three phalangeal joint triplets per digit for five digits;\n"
          "   concatenating the scaled coordinates and fifteen flexion angles into an invariant feature vector;\n"
          "   executing forward neural inference using a multi-layer perceptron (MLP) neural classifier (209) on the edge processing unit (203) without transmitting video data to external servers;\n"
          "   verifying prediction stability across an N-frame consecutive streak threshold; and\n"
          "   appending confirmed characters into a word buffer (210).")

    add_p("3. The system as claimed in claim 1, wherein the geometric normalization engine (208) computes each of the fifteen finger flexion angles θ as:\n"
          "   θ = (1/π) * arccos((ba · bc) / (||ba|| * ||bc|| + ε))\n"
          "   across fifteen anatomical triplets defined by landmark indices: [0, 1, 2], [1, 2, 3], [2, 3, 4] for the thumb; [0, 5, 6], [5, 6, 7], [6, 7, 8] for the index finger; [0, 9, 10], [9, 10, 11], [10, 11, 12] for the middle finger; [0, 13, 14], [13, 14, 15], [14, 15, 16] for the ring finger; and [0, 17, 18], [17, 18, 19], [18, 19, 20] for the little finger, wherein index 0 designates the wrist landmark, wherein ba and bc are vectors from the vertex joint b to the adjacent joints a and c respectively, and ε is a small positive constant preventing division by zero, to provide scale-normalized kinematic descriptors with reduced sensitivity to perspective projection in a dimensionless range of [0, 1].")

    add_p("4. The system as claimed in claim 1, wherein the geometric normalization engine (208) applies a 2D planar rotation alignment by determining an angle of a vector connecting wrist landmark 0 to middle metacarpophalangeal landmark 9 relative to a vertical axis, and rotating centered 2D planar coordinates by an orthogonal rotation matrix to establish canonical orientation.")

    add_p("5. The system as claimed in claim 1, wherein the MLP neural classifier (209) comprises a 78-dimensional input architecture for single-hand American Sign Language (ASL) classifying 28 output classes comprising 26 alphabet letters and two control gestures, and a 156-dimensional input architecture for dual-hand Indian Sign Language (ISL).")

    add_p("6. The system as claimed in claim 5, wherein the dual-hand ISL input architecture receives horizontally sorted, independently normalized 78-dimensional feature vectors corresponding to a left-most hand and a right-most hand in the image, and wherein the ISL neural classifier evaluates 35 distinct classes comprising 20 dual-hand alphabet gestures, 6 single-hand alphabet gestures, and 9 numerical gestures, single-hand gestures being padded with a predetermined padding vector.")

    add_p("7. The system as claimed in claim 1, wherein the temporal word buffer engine (210) enforces a post-capture cooldown period ranging from 500 to 1000 milliseconds to inhibit duplicate character capture and commits active words into sentences upon detecting the absence or drop of hand landmarks for a predetermined duration.")

    add_p("8. The system as claimed in claim 1, further comprising a bidirectional avatar synthesizer (211) configured to parse input text into letter tokens and interpolate stored 3D skeletal joint poses via cubic easing interpolation to visually demonstrate correct signing movements on the display module (205).")

    add_p("9. The system as claimed in claim 8, wherein the bidirectional avatar synthesizer (211) dynamically renders a human gesture photographic reference card displaying explicit required hand counts synchronized with active sign progression.")

    add_p("10. The method as claimed in claim 2, wherein the optical video frames are processed exclusively within local volatile memory (204) of the edge processing unit (203) and discarded after landmark computation without transmitting video streams or facial biometric images to external networks.")

    # 6. DATE AND SIGNATURE
    add_p("6. DATE AND SIGNATURE", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=14)
    
    table = doc.add_table(rows=1, cols=2)
    table.autofit = False
    table.columns[0].width = Inches(4.0)
    table.columns[1].width = Inches(2.5)
    
    cell_left = table.cell(0, 0)
    p_left = cell_left.paragraphs[0]
    p_left.paragraph_format.space_after = Pt(2)
    run_l = p_left.add_run("Dated this 05th day of October, 2026")
    run_l.font.name = 'Times New Roman'
    run_l.font.size = Pt(11)
    
    cell_right = table.cell(0, 1)
    p_right = cell_right.paragraphs[0]
    p_right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_right.paragraph_format.space_after = Pt(2)
    run_r = p_right.add_run("Vinoth Kumar J\nPatent Agent (IN/PA 6111)\nAgent for the Applicant")
    run_r.bold = True
    run_r.font.name = 'Times New Roman'
    run_r.font.size = Pt(11)

    # 7. ABSTRACT
    doc.add_page_break()
    add_p("7. ABSTRACT OF THE INVENTION", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=12)
    add_p("A REAL-TIME VISION-BASED SIGN LANGUAGE RECOGNITION SYSTEM USING GEOMETRIC LANDMARKS AND MULTI-LAYER PERCEPTRON", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=14)
    add_p("A real-time vision-based sign language recognition system and method are disclosed for client-side translation of American Sign Language (ASL) and Indian Sign Language (ISL). The system comprises an optical sensor module (202), an edge processing unit (203), a display module (205), and an audio synthesizer (206). The edge processing unit executes a 3D skeletal landmark extraction module (207) identifying 21 joints per hand, a geometric normalization engine (208) translating coordinates to a wrist-centered origin, scaling coordinates by palm length, and deriving 15 finger flexion angles. A Multi-Layer Perceptron (MLP) neural engine (209) operates in single-hand and dual-hand modes to classify gestures at real-time speeds on client hardware. A temporal word buffer engine (210) confirms predictions through consecutive streak verification, and a bidirectional avatar synthesizer (211) demonstrates sign sequences. Biometric visual privacy is preserved through local video processing without cloud transmission.", WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=12)

    output_docx = r"r:\Projects\Sign Language\PATENT\Form 2_our_project.docx"
    doc.save(output_docx)
    print(f"Saved DOCX to: {output_docx}")

if __name__ == '__main__':
    create_form2()
