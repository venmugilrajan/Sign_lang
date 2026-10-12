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
    add_p("AN EMBEDDED EDGE-AI KIOSK APPARATUS AND WIRELESS SENSOR GLOVE SYSTEM FOR REAL-TIME BIDIRECTIONAL SIGN LANGUAGE TRANSLATION USING MULTI-MODAL SENSOR FUSION AND NEURAL INFERENCE", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=14)

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
    add_p("[0001] The present invention relates generally to assistive communication devices, embedded computing systems, human-computer interaction (HCI) terminals, and wearable kinematic telemetry interfaces. More particularly, the invention relates to an integrated, standalone edge-AI kiosk apparatus operatively paired with a wireless ergonomic sensor glove, providing a multi-modal hardware-software system for real-time bidirectional translation between vocalized speech, visual sign language gestures (comprising single-hand American Sign Language and dual-hand Indian Sign Language), and animated 3D sign demonstrations using local geometric kinematic landmark extraction, glove telemetry fusion, and quantized neural inference.")

    add_p("BACKGROUND OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0002] Visual sign languages, such as Indian Sign Language (ISL) and American Sign Language (ASL), serve as the principal communication medium for millions of deaf and hard-of-hearing individuals worldwide. However, the overwhelming majority of the hearing population lacks fluency in sign language, creating persistent and debilitating communication barriers in critical environments such as healthcare clinics, banking and postal counters, educational institutions, government administrative centers, and commercial retail terminals.")
    add_p("[0003] Existing computational attempts to bridge this communication divide predominantly fall into two unsatisfactory extremes: purely vision-based software applications, or bulky proprietary data gloves tethered to desktop workstations.")
    add_p("[0004] Purely vision-based approaches rely on monocular RGB webcams or video cameras running deep 2D/3D Convolutional Neural Networks (CNNs). In real-world assistive situations, camera-only systems suffer severe operational vulnerabilities: (i) Optical occlusion: when fingers cross, overlap, or the hand turns away from the optical axis (especially prevalent in complex dual-hand ISL gestures where 20 out of 26 letters require two-hand interaction), camera landmarks fail or jitter wildly; (ii) Environmental illumination: ambient shadows, backlighting, or poor room lighting cause catastrophic tracking failures; (iii) Cloud dependency: many conventional systems stream raw video frames to remote cloud servers, causing unacceptable network latencies (exceeding 500-1500 ms), complete communication blackouts in areas with poor internet connectivity, and severe visual biometric privacy violations by transmitting identifiable human face footage over public networks.")
    add_p("[0005] Conversely, standalone data gloves previously reported in the art are often rigid, tethered by cumbersome wiring harnesses, lack integrated speech synthesis or visual interactive feedback, and cannot perform bidirectional dialogue (failing to convert hearing speech back into sign language for the deaf user). Furthermore, generic off-the-shelf monitors or computer stands cannot provide acoustic isolation, integrated lighting, anti-glare visual privacy hoods, or dedicated edge acceleration required for instantaneous conversational interaction.")
    add_p("[0006] Accordingly, there exists an acute, unfulfilled need for an integrated, standalone Edge-AI Kiosk apparatus paired with a lightweight wireless sensor glove that solves optical occlusion through hardware multi-modal sensor fusion, preserves visual privacy by executing all neural inference locally on edge computing hardware, provides acoustic and visual isolation via a dedicated canopy hood, and provides true real-time bidirectional translation between vocal speech and sign language.")

    add_p("OBJECTIVES OF THE PRESENT INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0007] The primary objective of the present invention is to provide a standalone, counter-mountable edge-AI kiosk apparatus and paired wireless sensor glove system capable of executing real-time bidirectional sign language translation entirely on local embedded hardware without requiring external cloud servers or internet connectivity.")
    add_p("[0008] Another objective of the present invention is to provide an anti-glare, acoustic-isolating privacy canopy hood enclosing a forward-facing optical depth sensor, active dual LED diffusers, and directional beamforming MEMS microphones within a self-contained physical housing.")
    add_p("[0009] Another objective of the present invention is to provide an ergonomic wearable sensor glove incorporating 5-channel dorsal polymer flexible bend sensors, a 6-DoF inertial measurement unit (IMU), an ultra-low-latency 2.4 GHz RF/BLE transceiver, and a tactile haptic feedback actuator.")
    add_p("[0010] Another objective of the present invention is to implement a multi-modal sensor fusion engine that seamlessly fuses optical 3D skeletal landmarks with continuous glove bend resistance and inertial orientation, eliminating classification failure caused by hand-to-hand occlusions and extreme camera angles.")
    add_p("[0011] Another objective of the present invention is to provide a geometric normalization engine that translates skeletal coordinates to a wrist origin, scales them by palm metric length, computes planar rotation alignment, and derives fifteen inter-joint finger flexion angles that are strictly scale, distance, and perspective invariant.")
    add_p("[0012] Another objective of the present invention is to execute quantized Multi-Layer Perceptron (MLP) neural networks locally in under 5 milliseconds on an embedded host SoC, supporting 28 single-hand ASL classes and 35 dual-hand ISL classes.")
    add_p("[0013] Another objective of the present invention is to provide a temporal word buffer engine with consecutive streak consensus and cooldown suppression to assemble stabilized character tokens into committed words and sentences.")
    add_p("[0014] Another objective of the present invention is to provide immediate tactile haptic feedback on the user's wrist when a sign token is committed, allowing the signing user to maintain uninterrupted eye contact with the interlocutor.")
    add_p("[0015] Another objective of the present invention is to provide bidirectional speech-to-sign translation by transcribing vocal speech into text phonemes and driving a real-time 3D humanoid avatar demonstrating corresponding sign language gestures at 60 frames per second on an angled touch display.")
    add_p("[0016] Another objective of the present invention is to guarantee zero visual biometric leakage by processing optical video frames strictly in volatile memory (RAM) and immediately discarding frames after landmark extraction.")

    add_p("SUMMARY OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0017] The present invention provides an integrated embedded edge-AI kiosk apparatus and wireless sensor glove system for real-time bidirectional sign language translation. The physical system comprises two operatively interconnected hardware units: (A) a standalone counter-mountable Edge-AI Kiosk Unit (202), and (B) a paired Wearable Wireless Sensor Glove Unit (240) communicating via a low-latency 2.4 GHz wireless telemetry link (235).")
    add_p("[0018] The Edge-AI Kiosk Unit (202) comprises a rigid enclosure having a countertop support base (304), an anti-glare privacy canopy hood (306), an optical vision sensor and dual auxiliary LED diffusers (204, 308) integrated within the upper hood fascia, a directional beamforming MEMS microphone array (206, 312), an embedded host processor SoC (208), an angled high-resolution capacitive touch display HUD (224, 310), an acoustic synthesizer and Class-D loudspeaker (226, 314), an ultra-low latency wireless transceiver (228), and a portable power subsystem (230).")
    add_p("[0019] The Wearable Wireless Sensor Glove Unit (240, 320) comprises a breathable flexible glove body (322), a 5-channel flexible bend sensor array (242, 322) embedded along the dorsal phalanges of each digit, a 6-DoF Inertial Measurement Unit (IMU) (244), a low-power glove microcontroller (MCU) with a 12-bit ADC (246), a glove wireless transceiver (248), a micro-haptic vibration actuator (250, 328), a multi-color status LED (326), and a rechargeable Li-Po battery pack (252).")
    add_p("[0020] In a sign-to-speech operating mode, the kiosk optical sensor captures video frames of the signing user while the wireless glove continuously transmits finger flex telemetry and wrist inertial orientation. The embedded host processor extracts 21 3D skeletal landmarks per hand in local volatile memory. A spatial kinematics unit (214) computes wrist centering, palm metric scaling, planar rotation alignment, and fifteen inter-joint finger flexion angles. A multi-modal fusion engine (218) correlates optical landmarks with glove bend resistance to compensate for finger occlusions. A quantized dual Multi-Layer Perceptron (MLP) neural classifier (222) outputs predicted character tokens. A temporal word buffer (220) applies streak consensus verification and cooldown suppression to form words and committed sentences. Upon word commitment, the acoustic synthesizer (226) vocalizes the sentence audibly for the hearing interlocutor, and an RF commit packet triggers a tactile haptic pulse in the glove's haptic motor (250).")
    add_p("[0021] In a speech-to-sign operating mode, the directional microphone array (206) captures vocal speech from the hearing user. An acoustic pipeline (212) transcribes the speech into text phonemes. A sign synthesizer (216) computes 3D skeletal bone transform keyframes, driving an interactive 3D humanoid avatar rendered at 60 FPS on the angled touch HUD (224), visually demonstrating the sign gestures to the deaf user.")

    add_p("BRIEF DESCRIPTION OF THE ACCOMPANYING DRAWINGS", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0022] Figure 1 illustrates an end-to-end flowchart depicting the multi-modal bidirectional sign language translation process comprising input acquisition, kinematic normalization, multi-modal sensor fusion, quantized neural inference, streak consensus filtering, and bidirectional output synthesis.")
    add_p("[0023] Figure 2 illustrates a detailed hardware and functional block diagram of the standalone edge-AI kiosk unit (202) operatively coupled via a bidirectional wireless link (235) with the wearable wireless sensor glove unit (240).")
    add_p("[0024] Figure 3 illustrates an isometric perspective schematic view of the physical translation system (300), showing the counter kiosk enclosure (302) with privacy canopy hood (306), optical sensor (308), angled touch display (310), acoustic transducers, docking cradle (316), and the paired wearable sensor glove (320) with 5-channel flex sensors (322) and wrist electronics pod (324).")
    add_p("[0025] Figure 4 illustrates a diagram of the kinematic skeletal hand landmark topology (402), 15 inter-joint flexion angles theta_k (406), palm metric scale D_palm (404), and multi-modal feature vector synthesis (408).")

    add_p("DETAILED DESCRIPTION OF THE INVENTION", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=4)
    add_p("[0026] Referring to Figures 2 and 3, the present invention provides a comprehensive multi-modal system (200, 300) for real-time bidirectional sign language translation. The physical hardware comprises a standalone Edge-AI Kiosk Unit (202, 302) and an operatively paired Wearable Wireless Sensor Glove Unit (240, 320) communicatively coupled via an ultra-low-latency 2.4 GHz wireless link (235, 330).")
    add_p("[0027] The Edge-AI Kiosk Unit (202, 302) is housed in a robust, counter-mountable structural enclosure comprising a wide desktop support base (304) and an upright housing capped by an anti-glare privacy canopy hood (306). The privacy hood extends outward over the upper and lateral perimeters of the kiosk, providing optical shielding against overhead ceiling glare and side reflections, while simultaneously providing visual privacy for user interaction at public service counters.")
    add_p("[0028] Mounted within the downward-angled upper fascia of the canopy hood (306) is a forward-facing optical vision sensor (204, 308) flanked by dual active LED illumination diffusers. The optical vision sensor comprises a wide-angle monocular optical camera module (operating at 30 to 60 frames per second at 720p or 1080p resolution). The integrated dual LED diffusers cast uniform, shadow-free illumination across the signing interaction volume (332), ensuring invariant hand visibility regardless of ambient room dimness or backlight shadows. The optical sensor transmits raw digital video directly to the internal host processor (208) via a high-speed MIPI-CSI (Mobile Industry Processor Interface Camera Serial Interface) video bus.")
    add_p("[0029] Integrated along the fascia of the kiosk is a directional acoustic capture unit (206, 312) comprising a multi-microphone array of Micro-Electro-Mechanical Systems (MEMS) directional microphones coupled with active hardware beamforming noise cancellation circuitry. The microphone array captures acoustic speech from an interlocutor facing the kiosk, filtering out background ambient babble noise, and streams digitized audio to the host processor via an I2S digital audio bus.")
    add_p("[0030] Positioned in an ergonomically tilted recess of the kiosk housing is an angled visual display subsystem (224, 310) comprising a high-resolution capacitive touch panel (e.g., 7.0-inch or 10.1-inch IPS LCD panel). The angled orientation allows comfortable viewing by a seated or standing user while remaining shielded from bystander observation by the canopy hood (306). Below the display, a high-fidelity acoustic synthesizer and loudspeaker (226, 314) comprising an internal audio codec and Class-D amplifier is positioned behind frontal speaker apertures.")
    add_p("[0031] The internal computing platform of the kiosk comprises an embedded host processor (208), such as a multi-core ARM Cortex System-on-Chip (SoC) equipped with dedicated neural tensor processing units, high-speed LPDDR4/LPDDR5 volatile memory (RAM), non-volatile flash storage, and an autonomous power management subsystem (230) comprising rechargeable Lithium-Ion batteries, DC-DC buck converters, and a USB Power Delivery (USB-PD) charging interface. A magnetic docking cradle (316) on the lateral housing of the kiosk incorporates inductive charging coils for safely stowing and recharging the sensor glove when not in use.")
    add_p("[0032] Referring to Figures 2 and 3, the Wearable Wireless Sensor Glove Unit (240, 320) comprises an ergonomic glove body (322) fabricated from a breathable, elastic textile substrate. Embedded along the dorsal aspect of each finger (thumb, index, middle, ring, little) is a 5-channel flexible bend sensor array (242, 322) comprising polymer thick-film flexible resistive strain gauges. As each digit flexes or extends, the corresponding sensor undergoes mechanical curvature, modulating its electrical resistance linearly as a function of bending angle.")
    add_p("[0033] Secured to the dorsal wrist cuff of the glove is a low-profile, contoured electronics pod (324) enclosing a low-power microcontroller unit (MCU) with a multi-channel 12-bit Analog-to-Digital Converter (ADC) (246), a 6-Degrees-of-Freedom (6-DoF) Inertial Measurement Unit (IMU) (244) comprising a 3-axis accelerometer and 3-axis gyroscope, an ultra-low latency 2.4 GHz RF / Bluetooth Low Energy (BLE 5.2) transceiver (248), a micro-haptic vibration actuator (250, 328), a multi-color LED link indicator (326), and a rechargeable miniature Li-Po battery pack (252). The MCU samples the 5 flex sensor channels and 6-DoF IMU at a rate of 100 Hz, digitizes the telemetry into a compact packet, and wirelessly broadcasts it to the kiosk wireless transceiver (228) with latency under 8 milliseconds.")
    add_p("[0034] Referring to Figures 1 and 4, the multi-modal sign-to-speech translation pipeline operates as follows. Optical video frames captured by the optical sensor (204) are loaded exclusively into volatile RAM of the host processor (208). A vision pipeline (210) detects hand boundaries and extracts 21 three-dimensional skeletal joint coordinates (x_i, y_i, z_i) (i = 0 to 20) per hand, corresponding to the wrist, carpals, metacarpals, and phalanges (402).")
    add_p("[0035] The spatial kinematics unit (214) performs rigorous geometric normalization: First, all coordinates are translated to a wrist-centered local origin (x_0, y_0, z_0), achieving spatial translation invariance: x'_i = x_i - x_0, y'_i = y_i - y_0, z'_i = z_i - z_0. Second, to eliminate variations caused by user distance from the camera or individual hand size, the image-plane Euclidean metric distance D_palm (404) between wrist landmark 0 and middle finger metacarpophalangeal landmark 9 is computed: D_palm = sqrt((x_9 - x_0)^2 + (y_9 - y_0)^2). All 21 centered coordinates are divided by D_palm, yielding scale-invariant coordinates (410). Third, planar rotation alignment is performed by computing the inclination angle alpha of vector (L0 -> L9) relative to the canonical vertical axis, and rotating all 2D coordinates by an orthogonal rotation matrix R(-alpha), enforcing canonical vertical alignment across arbitrary wrist roll.")
    add_p("[0036] Concurrently, fifteen (15) inter-joint finger flexion angles theta_k (k = 1 to 15) (406, 412) are derived across adjacent phalangeal triplets across all five digits. For each anatomical triplet (a, b, c) having intermediate joint vertex b, adjacent unit vectors ba = a - b and bc = c - b are formed. Each flexion angle theta_k is calculated as:\n"
          "   theta_k = (1/pi) * arccos((ba · bc) / (||ba|| * ||bc|| + epsilon))\n"
          "where epsilon is a positive stabilization constant (1e-6) preventing numerical division by zero. The scalar values of theta_k are strictly bounded in [0, 1] and provide pure kinematic descriptors that are intrinsically immune to camera perspective distortion and hand sizing variations.")
    add_p("[0037] Referring to Figure 4, the multi-modal sensor fusion engine (218) executes occlusion compensation (416). When the signing user performs signs where fingers overlap or hide behind the palm (e.g., ASL letters 'E', 'M', 'N', 'S', or ISL dual-hand crossings), optical tracking landmarks for the occluded digits degrade in confidence. The fusion engine continuously correlates the optical flexion angles theta_k with the glove's calibrated 5-channel bend resistance vector S_flex = [s_thumb, s_index, s_middle, s_ring, s_little] and 6-DoF IMU orientation vector (414). When optical landmark confidence drops below a threshold (0.60), the fusion engine substitutes the physical strain readings from the glove, guaranteeing continuous 100% gesture recognition without jitter or frame dropping.")
    add_p("[0038] For American Sign Language (ASL), the 63 normalized coordinates and 15 flexion angles (optionally augmented with glove telemetry) form a 78-dimensional feature vector. Left-hand gestures are mirrored across the sagittal axis into canonical right-handed space. For Indian Sign Language (ISL), dual hands are extracted, horizontally sorted (left-most and right-most), and normalized into two 78-dimensional sub-vectors. When single-hand ISL signs are performed, the absent secondary hand is populated with a predefined zero-padded vector, producing a uniform 156-dimensional dual-hand tensor.")
    add_p("[0039] The quantized dual Multi-Layer Perceptron (MLP) neural classifier (222) comprises an ASL network (78 -> 256 -> 128 -> 64 -> 28 classes) and an ISL network (156 -> 512 -> 256 -> 128 -> 35 classes). The models execute forward inference in under 5 milliseconds on the host processor SoC, achieving over 98% classification accuracy while consuming under 5 Watts of total system power.")
    add_p("[0040] The temporal word buffer engine (220) receives predicted class tokens. A streak consensus filter requires a predicted character token to remain consistent for N consecutive frames (e.g., N = 5 frames with confidence >= 0.70) before confirming the character. Upon confirmation, a post-capture cooldown timer (configured at 600 ms) suppresses duplicate character registrations while transitioning between hand shapes. Dropping hands from the tracking zone for a predetermined duration commits the buffered word into a completed sentence string.")
    add_p("[0041] Upon sentence commitment, two simultaneous physical events occur: (1) The sentence string is routed to the acoustic synthesizer (226), which immediately converts the text into vocalized speech through the Class-D loudspeaker (314), speaking the sentence aloud to the hearing interlocutor; and (2) The host wireless transceiver (228) transmits an RF confirmation packet to the glove, causing the glove's haptic motor (250, 328) to vibrate in a distinct tactile pulse. This haptic confirmation provides instantaneous physical feedback to the deaf user that their sentence has been spoken, eliminating the need to look away from their conversational partner.")
    add_p("[0042] In bidirectional speech-to-sign mode, spoken vocal audio from the hearing user is captured by the directional microphone array (206, 312). The acoustic pipeline (212) performs speech-to-text transcription. The sign synthesizer engine (216) maps text tokens into anatomical 3D bone orientation quaternions. These quaternions drive a photorealistic 3D humanoid avatar rendered at 60 frames per second on the angled touch HUD (224, 310), visually enacting the sign language translation directly in front of the deaf user.")
    add_p("[0043] Crucially, all video frames captured by the optical sensor (204) are held strictly in local volatile RAM and are instantly overwritten after landmark extraction. No raw video, facial imagery, or biometric footage is ever recorded on non-volatile media or transmitted over any external network, guaranteeing absolute biometric privacy.")

    # 5. CLAIMS
    doc.add_page_break()
    add_p("5. CLAIMS", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13, space_after=12)
    add_p("We claim:", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=8)

    add_p("1. A standalone edge-AI kiosk apparatus and wireless sensor glove system (200) for real-time bidirectional sign language translation, the system comprising:\n"
          "   A) a counter-mountable kiosk unit (202) comprising a rigid housing enclosing:\n"
          "      an anti-glare privacy canopy hood (306) projecting over an interactive volume;\n"
          "      an optical vision sensor (204) and an active auxiliary illumination array (308) integrated within the canopy hood;\n"
          "      a directional acoustic capture unit (206) comprising a beamforming microphone array;\n"
          "      an embedded host processor (208) comprising local volatile memory and non-volatile flash storage;\n"
          "      an angled visual touch display subsystem (224) mounted in the housing under the canopy hood;\n"
          "      an acoustic synthesizer (226) coupled to an audio loudspeaker;\n"
          "      a kiosk wireless transceiver (228); and\n"
          "      a power subsystem (230) providing regulated operating power;\n"
          "   B) a wearable wireless sensor glove unit (240) communicatively coupled to the kiosk wireless transceiver (228) via a low-latency wireless link (235), the glove unit comprising:\n"
          "      a flexible glove body (322) having a 5-channel flexible bend sensor array (242) disposed along dorsal phalanges of five digits;\n"
          "      a 6-Degrees-of-Freedom (6-DoF) inertial measurement unit (244);\n"
          "      a glove microcontroller (246) and a glove wireless transceiver (248); and\n"
          "      a tactile haptic feedback actuator (250);\n"
          "   wherein the embedded host processor (208) is configured to execute:\n"
          "      a vision pipeline (210) extracting 21 3D spatial skeletal joint landmarks per detected hand from video frames in volatile memory;\n"
          "      a spatial kinematics unit (214) translating skeletal landmarks to a wrist-centered origin, scaling centered coordinates by a Euclidean palm distance, and deriving fifteen inter-joint finger flexion angles across five digits;\n"
          "      a multi-modal fusion engine (218) fusing the skeletal landmarks with glove bend resistance telemetry received from the wireless sensor glove unit (240) to compensate for optical finger occlusions;\n"
          "      an edge neural classifier (222) processing the fused feature vector to output predicted character tokens;\n"
          "      a temporal word buffer (220) aggregating confirmed character tokens into committed sentences using streak consensus verification; and\n"
          "      wherein the acoustic synthesizer (226) converts the committed sentences into vocalized audible speech, and wherein the kiosk wireless transceiver (228) transmits a confirmation signal triggering a tactile vibration pulse on the glove haptic feedback actuator (250).")

    add_p("2. The system as claimed in claim 1, wherein the canopy hood (306) of the kiosk unit encloses the optical vision sensor (204) and dual LED illumination diffusers (308) angled downward to cast shadow-free illumination across the user interaction volume while shielding the visual touch display (224) and optical sensor from overhead ambient glare, and wherein the optical vision sensor is connected to the host processor (208) via an internal MIPI-CSI video interface bus.")

    add_p("3. The system as claimed in claim 1, wherein the 5-channel flexible bend sensor array (242) of the sensor glove comprises polymer thick-film flexible resistive strain gauges whose electrical resistance varies monotonically with finger flexion, sampled by a 12-bit analog-to-digital converter on the glove microcontroller (246) at a sampling rate of at least 100 Hz.")

    add_p("4. The system as claimed in claim 1, wherein the multi-modal fusion engine (218) dynamically evaluates landmark tracking confidence for each digit, and replaces optically derived joint coordinates with calibrated physical bend resistance readings from the corresponding channel of the flexible bend sensor array (242) upon optical landmark confidence falling below a predetermined confidence threshold during finger-crossing or hand-occlusion states.")

    add_p("5. The system as claimed in claim 1, wherein the spatial kinematics unit (214) computes each of the fifteen finger flexion angles theta_k as:\n"
          "   theta_k = (1/pi) * arccos((ba · bc) / (||ba|| * ||bc|| + epsilon))\n"
          "   across fifteen anatomical triplets defined by landmark indices: [0, 1, 2], [1, 2, 3], [2, 3, 4] for the thumb; [0, 5, 6], [5, 6, 7], [6, 7, 8] for the index finger; [0, 9, 10], [9, 10, 11], [10, 11, 12] for the middle finger; [0, 13, 14], [13, 14, 15], [14, 15, 16] for the ring finger; and [0, 17, 18], [17, 18, 19], [18, 19, 20] for the little finger, wherein index 0 designates the wrist landmark, wherein ba and bc are vectors from intermediate vertex joint b to adjacent joints a and c respectively, and epsilon is a predetermined positive constant preventing division by zero, yielding perspective-invariant descriptors bounded strictly within [0, 1].")

    add_p("6. The system as claimed in claim 1, wherein the spatial kinematics unit (214) applies a planar rotation alignment by calculating the angular orientation of a vector connecting wrist landmark 0 to middle finger metacarpophalangeal landmark 9 relative to a canonical vertical axis, and rotating centered 2D planar coordinates by an orthogonal rotation matrix to establish canonical palm orientation.")

    add_p("7. The system as claimed in claim 1, wherein the edge neural classifier (222) comprises quantized Multi-Layer Perceptron (MLP) architectures executed locally on the host processor SoC, comprising a 78-dimensional input layer for single-hand American Sign Language (ASL) classifying 28 output classes, and a 156-dimensional input layer for dual-hand Indian Sign Language (ISL) classifying 35 output classes, wherein single-hand gestures within the ISL mode are padded with a predetermined zero-padding vector to maintain uniform tensor dimensions.")

    add_p("8. The system as claimed in claim 1, wherein the temporal word buffer (220) enforces an N-frame consecutive identical prediction streak threshold and a post-capture cooldown timer ranging from 500 to 1000 milliseconds to inhibit duplicate character registrations, and automatically commits buffered words into completed sentence strings upon detecting the absence of hands from the interaction volume for a predetermined duration.")

    add_p("9. The system as claimed in claim 1, further comprising a bidirectional sign synthesizer engine (216) executed on the host processor (208), the sign synthesizer engine configured to parse spoken vocal audio received from the directional acoustic capture unit (206) into text phonemes and compute 3D bone orientation quaternions driving an interactive 3D humanoid avatar demonstrating corresponding sign language gestures rendered at 60 frames per second on the visual touch display subsystem (224).")

    add_p("10. A method for real-time bidirectional sign language translation executed on the system of claim 1, the method comprising:\n"
          "   acquiring continuous optical video frames of a user via an optical vision sensor (204) housed in a kiosk canopy hood (306) concurrently with receiving wireless finger bend and inertial telemetry from a paired sensor glove (240);\n"
          "   extracting 21 3D spatial skeletal joint landmarks per detected hand in local volatile memory of an embedded host processor (208);\n"
          "   translating the skeletal landmarks to a wrist-centered origin, scaling coordinates by a Euclidean palm distance, and calculating fifteen inter-joint finger flexion angles;\n"
          "   fusing the skeletal landmarks with the glove bend telemetry to resolve optical occlusions via a multi-modal fusion engine (218);\n"
          "   executing forward neural inference on a quantized edge neural classifier (222) within the host processor (208) to output predicted character tokens;\n"
          "   verifying prediction stability across a consecutive streak threshold and assembling confirmed characters into words and sentences via a temporal word buffer (220);\n"
          "   vocalizing committed sentences into audible speech via an acoustic synthesizer (226);\n"
          "   transmitting an RF trigger to actuate a tactile haptic pulse on the sensor glove (240); and\n"
          "   transcribing incoming spoken vocal speech and rendering interactive 3D humanoid avatar sign demonstrations on an angled touch display (224).")

    # 6. DATE AND SIGNATURE
    add_p("6. DATE AND SIGNATURE", WD_ALIGN_PARAGRAPH.LEFT, bold=True, size=12, space_after=14)
    
    table = doc.add_table(rows=1, cols=2)
    table.autofit = False
    table.columns[0].width = Inches(4.0)
    table.columns[1].width = Inches(2.5)
    
    cell_left = table.cell(0, 0)
    p_left = cell_left.paragraphs[0]
    p_left.paragraph_format.space_after = Pt(2)
    run_l = p_left.add_run("Dated this 11th day of October, 2026")
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
    add_p("AN EMBEDDED EDGE-AI KIOSK APPARATUS AND WIRELESS SENSOR GLOVE SYSTEM FOR REAL-TIME BIDIRECTIONAL SIGN LANGUAGE TRANSLATION USING MULTI-MODAL SENSOR FUSION AND NEURAL INFERENCE", WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12, space_after=14)
    add_p("An embedded edge-AI kiosk apparatus and wireless sensor glove system (200) are disclosed for real-time bidirectional sign language translation. The system comprises a counter-mountable kiosk unit (202) having an anti-glare privacy canopy hood (306), an optical vision sensor and dual LED diffusers (204, 308), a directional MEMS microphone array (206), an embedded host processor SoC (208), an angled touch display HUD (224), an acoustic synthesizer (226), and a low-latency wireless transceiver (228), operatively paired with a wearable wireless sensor glove (240) having a 5-channel flexible bend sensor array (242), a 6-DoF IMU (244), and a tactile haptic actuator (250). The host processor executes a vision pipeline (210) extracting 21 3D skeletal landmarks per hand, a spatial kinematics unit (214) performing wrist centering, palm scaling, and deriving fifteen inter-joint finger flexion angles, and a multi-modal fusion engine (218) fusing optical landmarks with glove bend resistance to compensate for optical occlusions. A quantized dual MLP neural classifier (222) performs inference in under 5 ms for single-hand (ASL: 78-D) and dual-hand (ISL: 156-D) gestures. A temporal word buffer (220) enforces streak consensus and cooldown suppression to assemble committed sentences vocalized via the acoustic synthesizer (226) while triggering a tactile confirmation vibration on the glove (250). In speech-to-sign mode, spoken audio is transcribed and animated via a 3D humanoid avatar on the touch HUD (224). All video frames are processed in volatile memory without cloud transmission, preserving absolute visual privacy.", WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, space_after=12)

    output_docx = r"r:\Projects\Sign Language\PATENT\Form 2_our_project.docx"
    doc.save(output_docx)
    print(f"Saved Form 2 DOCX to: {output_docx}")

if __name__ == '__main__':
    create_form2()
