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
    add_p("AN EMBEDDED EDGE-AI APPARATUS AND SYSTEM FOR REAL-TIME BIDIRECTIONAL SIGN LANGUAGE TRANSLATION USING KINEMATIC GEOMETRIC LANDMARKS AND NEURAL INFERENCE", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=14)

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
    add_p("[0001] The present invention relates generally to embedded computing systems, assistive human-computer interaction (HCI) devices, and edge-computing apparatuses. More particularly, the invention relates to a standalone, portable edge-AI apparatus and integrated hardware-software system configured for real-time bidirectional translation between vocalized speech, visual sign language gestures (comprising single-hand American Sign Language and dual-hand Indian Sign Language), and animated 3D sign demonstrations using local geometric kinematic landmark extraction and quantized neural classifiers executing entirely on on-board host processor hardware.")

    add_p("BACKGROUND OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0002] Deaf and hard-of-hearing communities rely heavily on visual sign languages such as Indian Sign Language (ISL) and American Sign Language (ASL) for daily social interaction. However, the lack of widespread fluency among the non-signing general public creates severe communication barriers in healthcare facilities, classrooms, government counters, and commercial environments.")
    add_p("[0003] Conventional computational sign language translation attempts typically rely on compute-heavy 2D or 3D Convolutional Neural Networks (CNNs) that continuously stream full raw RGB video frames to external remote cloud servers. Such cloud-centric architectures suffer from fatal drawbacks in practical assistive deployment: they incur excessive network latency, suffer total operational breakdown in rural or low-connectivity zones, demand continuous high-bandwidth internet connectivity, and present severe visual biometric privacy risks by transmitting identifiable human face and hand footage across public networks.")
    add_p("[0004] Furthermore, existing desktop software applications fail to provide a dedicated, physical, self-contained interactive device that can be placed on a counter or carried by a user. They lack specialized acoustic-optical transducer integration capable of capturing both directional vocal audio from a hearing interlocutor and optical sign gestures from a deaf user in real-time within a single portable physical housing.")
    add_p("[0005] Additionally, Indian Sign Language (ISL) presents unique physical and algorithmic challenges because 20 of its 26 alphabet characters require dual-hand coordination, where hand-to-hand occlusions, distance variations from the sensor, and orientation angles severely degrade naive pixel-based classifiers.")
    add_p("[0006] Accordingly, there exists a critical need for an integrated, portable edge-AI apparatus and physical translation system that combines dedicated optical sensors, directional acoustic capture, an on-board host processor executing privacy-preserving geometric feature normalization and quantized neural classification, an interactive visual display for 3D avatar demonstration, and an acoustic synthesizer within a self-powered physical unit.")

    add_p("OBJECTIVES OF THE PRESENT INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0007] The primary objective of the present invention is to provide a standalone, portable edge-AI hardware apparatus and system capable of executing complete real-time bidirectional sign language translation entirely on local embedded computing hardware without requiring external cloud servers or remote network processing.")
    add_p("[0008] Another objective is to integrate dedicated optical sensing, directional acoustic capture, an interactive visual touch display, an acoustic speech synthesizer, and an autonomous power subsystem within a single compact physical enclosure.")
    add_p("[0009] Another objective is to provide a geometric feature normalization engine on the local edge processor that establishes scale-invariance, distance-invariance, and planar rotation alignment for dual-hand and single-hand skeletal tracking.")
    add_p("[0010] Another objective is to extract fifteen anatomical inter-joint finger flexion angles as kinematic descriptors that decouple hand shape understanding from camera perspective distortion and hand sizing variations.")
    add_p("[0011] Another objective is to execute quantized, lightweight Multi-Layer Perceptron (MLP) neural networks locally in under 5 milliseconds on an embedded CPU/SoC, supporting 28 single-hand ASL classes and 35 dual-hand ISL classes.")
    add_p("[0012] Another objective is to provide a temporal consensus word buffer engine with streak confirmation and cooldown suppression to prevent gesture transition flickering and double-character repetition.")
    add_p("[0013] Another objective is to provide bidirectional communication by synthesizing real-time 3D humanoid avatar sign demonstrations on the interactive display in response to incoming vocal speech or text input.")
    add_p("[0014] Another objective is to enforce visual privacy and zero biometric leakage by processing optical video frames exclusively in volatile memory on the embedded processor and instantly discarding frames after skeletal landmark extraction.")

    add_p("SUMMARY OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0015] The present invention provides an embedded edge-AI apparatus and system for real-time bidirectional sign language translation. The physical apparatus comprises an optical vision sensor (202), a directional acoustic capture unit (204), an embedded host processor (206) having local volatile memory and non-volatile flash storage, an interactive visual display subsystem (220), an acoustic synthesizer (222), and an autonomous portable power subsystem (224).")
    add_p("[0016] In a sign-to-speech operating mode, the optical vision sensor captures continuous video frames of a signing user at 30 to 60 frames per second. The embedded host processor executes a local vision pipeline (208) extracting 21 3D spatial skeletal landmarks per detected hand. A spatial kinematics unit (212) shifts coordinates to a wrist-centered local origin, scales them by palm metric length, computes planar rotation alignment, and derives fifteen inter-joint finger flexion angles across all digits.")
    add_p("[0017] A quantized edge neural classifier (216) processes a 78-dimensional feature vector for single-hand gestures and a 156-dimensional feature vector for dual-hand gestures. A temporal word buffer engine (218) applies consecutive streak consensus verification and cooldown suppression to assemble confirmed character tokens into words and complete sentence strings. The committed sentence string is vocalized audibly via the acoustic synthesizer (222) and displayed on the interactive visual display (220).")
    add_p("[0018] In a speech-to-sign operating mode, the directional acoustic capture unit (204) captures vocal speech from a hearing user. An acoustic pipeline (210) extracts text phonemes on the host processor, and a sign synthesizer engine (214) calculates step-by-step 3D bone transform keyframes, driving an interactive 3D humanoid avatar demonstrating the corresponding sign gestures on the visual display (220).")

    add_p("BRIEF DESCRIPTION OF THE ACCOMPANYING DRAWINGS", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0019] Figure 1 illustrates a flow diagram representing the real-time sign language recognition, kinematic landmark normalization, neural classification, and word aggregation process.")
    add_p("[0020] Figure 2 illustrates a hardware and functional block diagram of the standalone edge-AI sign language translation apparatus (200), depicting the physical interconnects between the optical vision sensor (202), acoustic capture unit (204), embedded host processor (206), visual display subsystem (220), acoustic synthesizer (222), and portable power subsystem (224).")

    add_p("DETAILED DESCRIPTION OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0021] Referring to Figure 2, the standalone edge-AI translation apparatus (200) is embodied as a compact, self-contained physical hardware unit comprising an optical vision sensor (202), a directional acoustic capture unit (204), an embedded host processor (206), an interactive visual display subsystem (220), an acoustic synthesizer (222), and a portable power subsystem (224) interconnected via high-speed internal buses.")
    add_p("[0022] The optical vision sensor (202) comprises a wide-angle monocular optical camera module (e.g., wide field-of-view lens operating at 30 to 60 frames per second at resolutions such as 1280x720 or 1920x1080 pixels) operatively coupled with an active auxiliary LED illumination array. The illumination array provides uniform, shadow-free illumination across the signing user's hands, eliminating recognition degradation caused by ambient room shadows or dim lighting conditions. The camera stream is routed to the embedded host processor (206) via an internal MIPI-CSI (Mobile Industry Processor Interface Camera Serial Interface) video bus.")
    add_p("[0023] The directional acoustic capture unit (204) comprises an array of MEMS (Micro-Electro-Mechanical Systems) directional microphones coupled with an active hardware noise cancellation filter. The acoustic capture unit receives spoken vocal audio from a hearing interlocutor, digitizes the signal, and routes it to the host processor via an I2S (Inter-IC Sound) digital audio bus.")
    add_p("[0024] The embedded host processor (206) comprises an edge System-on-Chip (SoC) or Single Board Computer (SBC) (such as a multi-core ARM Cortex processor with integrated neural tensor accelerators, e.g., Raspberry Pi 4/5 or NVIDIA Jetson architecture), local volatile memory (LPDDR4/LPDDR5 RAM), and non-volatile flash storage containing pre-compiled quantized neural model weights.")
    add_p("[0025] Referring to the functional pipelines executed within the embedded host processor (206), the vision pipeline (208) receives raw frames directly in local volatile memory. The vision pipeline localizes hand boundaries and computes 21 three-dimensional skeletal coordinates (x, y, z) per hand representing anatomical joints (wrist, carpals, metacarpals, and phalanges).")
    add_p("[0026] The spatial kinematics unit (212) translates raw spatial coordinates to a wrist-centered local origin (x0, y0, z0) to achieve spatial translation invariance. To establish scale and distance invariance, the Euclidean distance D_palm between wrist landmark 0 and middle finger metacarpophalangeal landmark 9 is computed in the image plane, and all 21 centered coordinates are divided by D_palm. Furthermore, planar rotation alignment is performed by calculating the angle between vector (L0 -> L9) and the canonical vertical axis, and rotating all 2D planar coordinates by an orthogonal rotation matrix, thereby establishing canonical vertical palm alignment across arbitrary wrist tilt.")
    add_p("[0027] Concurrently, fifteen (15) inter-joint finger flexion angles θ_k (k = 1 to 15) are computed across adjacent phalangeal triplets across all five digits. For each anatomical triplet (a, b, c) having intermediate joint vertex b, adjacent unit vectors ba = a - b and bc = c - b are formed. Each flexion angle θ_k is calculated as: θ_k = (1/π) * arccos((ba · bc) / (||ba|| * ||bc|| + ε)), where ε is a positive stabilization constant (1e-6) preventing division by zero. The resulting scalar values are bounded strictly within [0, 1], yielding kinematic descriptors that are fundamentally invariant to user hand size and camera perspective.")
    add_p("[0028] In American Sign Language (ASL) mode, the 63 normalized coordinates and 15 flexion angles are concatenated to form a 78-dimensional feature vector. Left-hand detections are mirrored across the sagittal axis into canonical right-handed space.")
    add_p("[0029] In Indian Sign Language (ISL) mode, coordinates from dual hands are extracted, horizontally sorted, and independently normalized into two 78-dimensional vectors. When a single-hand ISL sign (such as single-hand alphabet signs 'C', 'I', 'L', 'O', 'U', 'V' or numerals '1' to '9') is performed, the absent secondary hand is populated with a predetermined zero-padded vector, producing a uniform 156-dimensional dual-hand vector.")
    add_p("[0030] The edge neural classifier (216) comprises quantized Multi-Layer Perceptron (MLP) architectures executing locally on the host processor. For ASL, the network comprises layer dimensions 78 → 256 → 128 → 64 → 28 classes (26 alphabet letters plus space and delete control classes). For ISL, the network comprises layer dimensions 156 → 512 → 256 → 128 → 35 classes. The forward inference pass executes in under 5 milliseconds on standard CPU cores with greater than 98% test accuracy, consuming less than 5 Watts of operational electrical power.")
    add_p("[0031] The temporal word buffer (218) receives predicted character tokens and confidence scores. A streak consensus filter requires a character prediction to remain stable for N consecutive frames (e.g., N = 5 frames with confidence >= 0.70) before confirming the character. Upon confirmation, a post-capture cooldown timer (operable between 500 ms and 1000 ms, configured at 600 ms) suppresses duplicate character registrations. Dropping hands from the camera frame for a predetermined duration automatically commits the buffered word into a completed sentence string.")
    add_p("[0032] The committed sentence string is routed to the acoustic synthesizer (222), which comprises an on-board audio codec, a Class-D digital audio amplifier, and an internal miniature acoustic speaker, instantly vocalizing the translated signs into spoken speech for non-signing observers.")
    add_p("[0033] Simultaneously, the committed text is rendered on the visual display subsystem (220), which comprises a high-resolution capacitive touch panel (e.g., 5-inch or 7-inch LCD/OLED HUD) mounted on the apparatus housing.")
    add_p("[0034] In bidirectional speech-to-sign mode, vocal speech captured by the acoustic capture unit (204) is transcribed by the acoustic pipeline (210) into text phonemes. The sign synthesizer engine (214) computes 3D bone orientation quaternions and retargets them to an interactive 3D humanoid avatar rendered at 60 frames per second on the visual display subsystem (220), visually demonstrating the sign language gesture to the deaf user.")
    add_p("[0035] The portable power subsystem (224) comprises a rechargeable Lithium-Ion (Li-Ion) battery pack, an integrated Power Management Integrated Circuit (PMIC) with step-down DC-DC buck regulators providing clean 5V and 3.3V DC power rails, and a USB Power Delivery (USB-PD) charging port enabling continuous portable operation.")
    add_p("[0036] Crucially, all video frames captured by the optical vision sensor (202) are held strictly in local volatile RAM (204) and are purged immediately after 3D joint coordinate calculation. No raw video or facial biometric data is ever stored on disk or transmitted across external networks, ensuring complete visual biometric privacy.")

    # 5. CLAIMS
    doc.add_page_break()
    add_p("5. CLAIMS", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=12)
    add_p("We claim:", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=8)

    add_p("1. A standalone edge-AI apparatus (200) for real-time bidirectional sign language translation, the apparatus comprising:\n"
          "   a physical portable housing enclosing:\n"
          "     an optical vision sensor (202) configured to acquire continuous optical video frames of a signing user;\n"
          "     a directional acoustic capture unit (204) comprising a microphone array configured to capture spoken vocal audio;\n"
          "     an embedded host processor (206) operatively coupled to the optical vision sensor (202) and acoustic capture unit (204), the host processor comprising local volatile memory and non-volatile flash storage;\n"
          "     an interactive visual display subsystem (220) coupled to the host processor (206);\n"
          "     an acoustic synthesizer (222) comprising an audio codec and an acoustic speaker coupled to the host processor (206); and\n"
          "     a portable power subsystem (224) comprising a rechargeable battery and a power management circuit providing regulated power to the apparatus;\n"
          "   wherein the embedded host processor (206) is configured to execute:\n"
          "     a vision pipeline (208) configured to extract 21 three-dimensional skeletal joint landmarks per detected hand from the video frames in local volatile memory;\n"
          "     a spatial kinematics unit (212) configured to translate the skeletal landmarks to a wrist-centered origin, scale centered coordinates by a Euclidean palm distance, and compute fifteen inter-joint finger flexion angles across five digits;\n"
          "     an edge neural classifier (216) configured to receive a feature vector comprising the scaled coordinates and flexion angles and output predicted character tokens;\n"
          "     a temporal word buffer (218) configured to aggregate confirmed character tokens into words and complete sentence strings using streak consensus verification; and\n"
          "     wherein the acoustic synthesizer (222) converts the complete sentence strings into vocalized audible speech, and wherein the visual display subsystem (220) displays synthesized text subtitles and animated 3D avatar sign demonstrations.")

    add_p("2. The apparatus as claimed in claim 1, wherein the optical vision sensor (202) comprises a wide-angle monocular optical camera coupled with an active auxiliary LED illumination array configured to provide uniform illumination across the signing user's hands, and wherein the vision sensor is connected to the host processor (206) via an internal MIPI-CSI video interface bus.")

    add_p("3. The apparatus as claimed in claim 1, wherein the spatial kinematics unit (212) computes each of the fifteen finger flexion angles θ_k as:\n"
          "   θ_k = (1/π) * arccos((ba · bc) / (||ba|| * ||bc|| + ε))\n"
          "   across fifteen anatomical triplets defined by landmark indices: [0, 1, 2], [1, 2, 3], [2, 3, 4] for the thumb; [0, 5, 6], [5, 6, 7], [6, 7, 8] for the index finger; [0, 9, 10], [9, 10, 11], [10, 11, 12] for the middle finger; [0, 13, 14], [13, 14, 15], [14, 15, 16] for the ring finger; and [0, 17, 18], [17, 18, 19], [18, 19, 20] for the little finger, wherein index 0 designates the wrist landmark, wherein ba and bc are vectors from the intermediate vertex joint b to adjacent joints a and c respectively, and ε is a predetermined positive constant preventing division by zero, yielding perspective-invariant descriptors bounded strictly within [0, 1].")

    add_p("4. The apparatus as claimed in claim 1, wherein the spatial kinematics unit (212) applies a planar rotation alignment by calculating the angular orientation of a vector connecting wrist landmark 0 to middle finger metacarpophalangeal landmark 9 relative to a canonical vertical axis, and rotating centered 2D planar coordinates by an orthogonal rotation matrix to establish canonical palm orientation.")

    add_p("5. The apparatus as claimed in claim 1, wherein the edge neural classifier (216) comprises a quantized Multi-Layer Perceptron (MLP) architecture comprising a 78-dimensional input layer for single-hand American Sign Language (ASL) classifying 28 output classes, and a 156-dimensional input layer for dual-hand Indian Sign Language (ISL) classifying 35 output classes.")

    add_p("6. The apparatus as claimed in claim 5, wherein the 156-dimensional ISL input receives horizontally sorted, independently normalized 78-dimensional sub-vectors corresponding to a left-most hand and a right-most hand, and wherein single-hand gestures within the ISL mode are padded with a predetermined padding vector to maintain uniform 156-dimensional tensor dimensions.")

    add_p("7. The apparatus as claimed in claim 1, wherein the temporal word buffer (218) enforces an N-frame consecutive identical prediction streak threshold and a post-capture cooldown timer ranging from 500 to 1000 milliseconds to inhibit duplicate character capture, and automatically commits buffered words into completed sentence strings upon detecting the absence of hands from the vision sensor for a predetermined duration.")

    add_p("8. The apparatus as claimed in claim 1, further comprising a bidirectional sign synthesizer engine (214) executed on the host processor (206), the sign synthesizer engine configured to parse spoken vocal audio received from the acoustic capture unit (204) into text phonemes and compute 3D bone orientation quaternions driving an interactive 3D humanoid avatar demonstrating corresponding sign language gestures on the visual display subsystem (220).")

    add_p("9. The apparatus as claimed in claim 1, wherein the optical video frames acquired by the optical vision sensor (202) are processed exclusively within local volatile memory of the embedded host processor (206) and are purged immediately after 3D skeletal landmark extraction without storing raw frames on permanent disk or transmitting video data to external networks.")

    add_p("10. A method for real-time bidirectional sign language translation executed on the apparatus of claim 1, the method comprising:\n"
          "   capturing optical video frames of a user via an optical vision sensor (202);\n"
          "   extracting 21 3D spatial skeletal joint landmarks per detected hand in local volatile memory of an embedded host processor (206);\n"
          "   translating the skeletal landmarks to a wrist-centered origin and scaling centered coordinates by a Euclidean palm distance;\n"
          "   calculating fifteen inter-joint finger flexion angles across adjacent phalangeal triplets;\n"
          "   executing forward neural inference on an edge neural classifier (216) within the host processor (206) to output character tokens;\n"
          "   verifying prediction stability across a consecutive streak threshold and aggregating confirmed characters into words and sentences via a temporal word buffer (218); and\n"
          "   vocalizing committed sentences into audible speech via an acoustic synthesizer (222) and rendering 3D avatar sign demonstrations on a visual display subsystem (220).")

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
    add_p("AN EMBEDDED EDGE-AI APPARATUS AND SYSTEM FOR REAL-TIME BIDIRECTIONAL SIGN LANGUAGE TRANSLATION USING KINEMATIC GEOMETRIC LANDMARKS AND NEURAL INFERENCE", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=14)
    add_p("An embedded edge-AI apparatus (200) and system are disclosed for real-time bidirectional sign language translation between sign gestures, vocalized speech, and visual 3D demonstrations. The physical apparatus comprises a portable housing enclosing an optical vision sensor (202), a directional acoustic capture unit (204), an embedded host processor (206), an interactive visual display subsystem (220), an acoustic synthesizer (222), and an autonomous portable power subsystem (224). The embedded host processor executes a vision pipeline (208) extracting 21 3D skeletal landmarks per hand, a spatial kinematics unit (212) performing wrist centering, palm scaling, planar rotation alignment, and calculating fifteen inter-joint finger flexion angles. A quantized edge neural classifier (216) processes single-hand (ASL: 78-D) and dual-hand (ISL: 156-D) feature vectors in under 5 milliseconds. A temporal word buffer (218) enforces streak consensus verification and cooldown suppression to assemble committed sentences vocalized via the acoustic synthesizer (222). In speech-to-sign mode, vocal speech is transcribed and converted by a sign synthesizer (214) into 3D humanoid avatar demonstrations on the visual display (220). Video frames are processed in volatile memory without cloud transmission, preserving visual privacy.", WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=12)

    output_docx = r"r:\Projects\Sign Language\PATENT\Form 2_our_project.docx"
    doc.save(output_docx)
    print(f"Saved Form 2 DOCX to: {output_docx}")

if __name__ == '__main__':
    create_form2()
