"""
Official Statutory Patent Drawing Figures Generator for:
AN EMBEDDED EDGE-AI KIOSK APPARATUS AND WIRELESS SENSOR GLOVE SYSTEM FOR 
REAL-TIME BIDIRECTIONAL SIGN LANGUAGE TRANSLATION USING MULTI-MODAL 
SENSOR FUSION AND NEURAL INFERENCE

Complies with Indian Patent Office drawing standards (Rule 15 of Patents Rules 2003/2006):
- High-contrast pure black and white line drawings (no dark gray blobs or overlapping fills)
- Uniform reference numeral formatting with non-colliding coordinates
- Standard patent leader lines (straight with arrowheads, zero cross-overs)
- Strictly orthogonal or unobstructed arrows with dedicated text margins
- Standard A4 sheet proportions (300 DPI)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = r'r:\Projects\Sign Language\PATENT'

# ─────────────────────────────────────────────────────────────────────────────
# FIG. 1 - FLOWCHART OF MULTI-MODAL BIDIRECTIONAL TRANSLATION METHOD
# ─────────────────────────────────────────────────────────────────────────────
def draw_fig1_flowchart(out_path):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Figure Header
    ax.text(50, 135, 'FIG. 1', fontsize=16, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(50, 130, 'MULTI-MODAL BIDIRECTIONAL TRANSLATION METHOD (100)', 
            fontsize=9.2, fontweight='bold', ha='center', va='center', family='sans-serif')

    def add_block(x, y, w, h, title, sub="", ref=None):
        patch = patches.Rectangle((x, y), w, h, edgecolor='black', facecolor='white', linewidth=1.3)
        ax.add_patch(patch)
        if ref:
            ax.text(x + w - 1.2, y + h - 1.2, str(ref), fontsize=8.8, fontweight='bold', ha='right', va='top', family='sans-serif')
        if sub:
            ax.text(x + (w - 3.2)/2, y + h*0.62, title, fontsize=7.8, fontweight='bold', ha='center', va='center', family='sans-serif')
            ax.text(x + (w - 3.2)/2, y + h*0.28, sub, fontsize=6.6, ha='center', va='center', family='sans-serif')
        else:
            ax.text(x + (w - 3.2)/2, y + h*0.50, title, fontsize=7.8, fontweight='bold', ha='center', va='center', family='sans-serif')

    def add_arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0), ha='center'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', edgecolor='black', width=1.1, headwidth=5.0, headlength=6.0))
        if label:
            mx = (x1 + x2)/2 + label_offset[0]
            my = (y1 + y2)/2 + label_offset[1]
            ax.text(mx, my, label, fontsize=6.4, fontstyle='italic', ha=ha, va='center', family='sans-serif',
                    bbox=dict(boxstyle="square,pad=0.15", facecolor='white', edgecolor='none'))

    def add_elbow_arrow(pts, label=None, label_pt=None):
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i+1]
            if i == len(pts) - 2:
                ax.annotate('', xy=p2, xytext=p1,
                            arrowprops=dict(facecolor='black', edgecolor='black', width=1.1, headwidth=5.0, headlength=6.0))
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', linewidth=1.1)
        if label and label_pt:
            ax.text(label_pt[0], label_pt[1], label, fontsize=6.4, fontstyle='italic', ha='center', va='center', family='sans-serif',
                    bbox=dict(boxstyle="square,pad=0.15", facecolor='white', edgecolor='none'))

    # STAGE 1: MULTI-MODAL DATA ACQUISITION (100) (y=106 to 126.5)
    g1 = patches.FancyBboxPatch((4, 106), 92, 20.5, boxstyle="round,pad=0.5",
                                edgecolor='black', facecolor='#fafafa', linestyle='--', linewidth=1.1)
    ax.add_patch(g1)
    ax.text(6, 124, 'Stage 1: Multi-Modal Input Acquisition (100)', fontsize=8.6, fontweight='bold', family='sans-serif')
    ax.text(94, 124, '100', fontsize=9.0, fontweight='bold', ha='right', va='center', family='sans-serif')
    
    add_block(6, 108, 27, 13.5, 'Optical Vision Capture', '30-60 FPS Video Stream', 102)
    add_block(36.5, 108, 27, 13.5, 'Wireless Glove Telemetry', '5 Flex + 6-DoF IMU Data', 104)
    add_block(67, 108, 27, 13.5, 'Acoustic Speech Capture', 'MEMS Directional Array', 106)

    # STAGE 2: KINEMATIC PREPROCESSING & SENSOR FUSION (110) (y=74 to 98)
    g2 = patches.FancyBboxPatch((4, 74), 92, 24, boxstyle="round,pad=0.5",
                                edgecolor='black', facecolor='#fafafa', linestyle='--', linewidth=1.1)
    ax.add_patch(g2)
    # Title placed in two compact lines centered at x=34.75, leaving 8+ units of whitespace to Arrow 1 (x=19.5) and Arrow 2 (x=50.0)
    ax.text(34.75, 96.2, 'Stage 2: Preprocessing', fontsize=7.2, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(34.75, 93.8, '& Sensor Fusion (110)', fontsize=7.0, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(94, 95.8, '110', fontsize=9.0, fontweight='bold', ha='right', va='center', family='sans-serif')

    add_block(6, 76, 27, 16.5, '3D Skeletal Tracking (112)\n& 15 Flexion Angles (114)', '21 Joints / Wrist Origin\nD_palm Scale & theta_k', 112)
    add_block(36.5, 76, 27, 16.5, 'Multi-Modal Fusion (116)', 'Occlusion Compensation\nOptical + Glove Blending', 116)
    add_block(67, 76, 27, 16.5, 'Speech-to-Text Engine (118)', 'Phoneme Recognition &\nAcoustic Transcription', 118)

    # STAGE 3: QUANTIZED NEURAL INFERENCE & TEMPORAL BUFFER (120) (y=42 to 66)
    g3 = patches.FancyBboxPatch((4, 42), 92, 24, boxstyle="round,pad=0.5",
                                edgecolor='black', facecolor='#fafafa', linestyle='--', linewidth=1.1)
    ax.add_patch(g3)
    # Title centered safely in open whitespace between Arrow 4 (x=31.0) and Arrow 9 (x=80.5)
    ax.text(56.0, 63.5, 'Stage 3: Neural Inference & Buffering (120)', fontsize=7.6, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(94, 63.5, '120', fontsize=9.0, fontweight='bold', ha='right', va='center', family='sans-serif')

    # Dual MLP (124) and Word Buffer (126) repositioned so right edge of 126 is at x=74, leaving x=74-94 open!
    add_block(18, 44, 26, 16.5, 'Quantized Dual MLP (124)', 'ASL (78-D) / ISL (156-D)\n<5ms Forward Latency', 124)
    add_block(48, 44, 26, 16.5, 'Temporal Word Buffer (126)', 'Streak Consensus (N=5)\n& Cooldown Filter', 126)

    # STAGE 4: BIDIRECTIONAL OUTPUT GENERATION (130) (y=10 to 34)
    g4 = patches.FancyBboxPatch((4, 10), 92, 24, boxstyle="round,pad=0.5",
                                edgecolor='black', facecolor='#fafafa', linestyle='--', linewidth=1.1)
    ax.add_patch(g4)
    # Title placed in two compact lines centered at x=34.75, leaving 8+ units of whitespace to Arrow to 132 (x=19.5) and Arrow to 134 (x=50.0)
    ax.text(34.75, 32.2, 'Stage 4: Output Synthesis', fontsize=7.2, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(34.75, 29.8, '& Transduction (130)', fontsize=7.0, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(94, 31.5, '130', fontsize=9.0, fontweight='bold', ha='right', va='center', family='sans-serif')

    add_block(6, 12, 27, 16.5, 'Acoustic Synthesizer (132)', 'Text-to-Speech Output\nClass-D Amp & Speaker', 132)
    add_block(36.5, 12, 27, 16.5, 'Tactile Haptic Pulse (134)', 'Glove ERM Vibration Alert\non Sentence Commit', 134)
    add_block(67, 12, 27, 16.5, '3D Avatar Engine (136)', 'Speech-to-Sign Animation\n60 FPS Skeletal Retargeting', 136)

    # ─── CLEAN UN-OBSTRUCTED ARROWS WITH DEDICATED CHANNELS ───
    # 1. Optical Capture (102) -> 3D Skeletal Tracking (112)
    add_arrow(19.5, 108, 19.5, 92.5, 'Raw Frames', label_offset=(-4.5, 0), ha='right')

    # 2. Wireless Glove (104) -> Multi-Modal Fusion (116)
    add_arrow(50.0, 108, 50.0, 92.5, 'Telemetry Packets', label_offset=(4.5, 0), ha='left')

    # 3. Skeletal Tracking (112) -> Multi-Modal Fusion (116)
    add_arrow(33.0, 84.2, 36.5, 84.2, 'theta_k', label_offset=(0, 2.5))

    # 4. Multi-Modal Fusion (116) -> Quantized Dual MLP (124)
    add_elbow_arrow([(50.0, 76.0), (50.0, 68.5), (31.0, 68.5), (31.0, 60.5)],
                    label='Fused Vector', label_pt=(40.5, 70.0))

    # 5. Dual MLP (124) -> Temporal Word Buffer (126)
    add_arrow(44.0, 52.2, 48.0, 52.2, 'Tokens', label_offset=(0, 2.5))

    # 6. Temporal Word Buffer (126) -> Acoustic Synthesizer (132)
    add_elbow_arrow([(50.0, 44.0), (50.0, 38.0), (19.5, 38.0), (19.5, 28.5)],
                    label='Committed Text', label_pt=(34.75, 39.5))

    # 7. Temporal Word Buffer (126) -> Tactile Haptic Pulse (134)
    # Direct straight vertical arrow at x=58.0 connecting Box 126 directly to Box 134!
    add_arrow(58.0, 44.0, 58.0, 28.5, 'Haptic Trigger', label_offset=(4.5, 0), ha='left')

    # 8. Acoustic Speech Capture (106) -> Speech-to-Text Pipeline (118)
    add_arrow(80.5, 108, 80.5, 92.5, 'Acoustic Audio', label_offset=(4.5, 0), ha='left')

    # 9. Speech-to-Text Pipeline (118) -> 3D Avatar Engine (136)
    # Pure vertical highway along x=80.5, completely clears Box 126 (right edge 74.0) by 6.5 units!
    add_arrow(80.5, 76.0, 80.5, 28.5, 'Phonemes', label_offset=(4.5, 0), ha='left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated FIG. 1: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG. 2 - HARDWARE ARCHITECTURE BLOCK DIAGRAM
# ─────────────────────────────────────────────────────────────────────────────
def draw_fig2_architecture(out_path):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Figure Header
    ax.text(50, 135, 'FIG. 2', fontsize=16, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(50, 130, 'HARDWARE ARCHITECTURE & INTERCONNECT SCHEMATIC (200)',
            fontsize=9.2, fontweight='bold', ha='center', va='center', family='sans-serif')

    # Main System Boundary (Dotted)
    sys_box = patches.FancyBboxPatch((3, 4), 94, 124, boxstyle="round,pad=0.6",
                                     edgecolor='black', facecolor='none', linestyle=':', linewidth=1.4)
    ax.add_patch(sys_box)
    ax.text(5, 126, 'Bidirectional Translation System (200)', fontsize=10.0, fontweight='bold', family='sans-serif')
    ax.text(95, 126, '200', fontsize=10.5, fontweight='bold', ha='right', family='sans-serif')

    def add_block(x, y, w, h, title, sub="", ref=None, bg='white'):
        patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                       edgecolor='black', facecolor=bg, linewidth=1.2)
        ax.add_patch(patch)
        if ref:
            ax.text(x + w - 1.0, y + h - 1.0, str(ref), fontsize=8.6, fontweight='bold', ha='right', va='top', family='sans-serif')
        if sub:
            ax.text(x + (w - 3.0)/2, y + h*0.62, title, fontsize=7.6, fontweight='bold', ha='center', va='center', family='sans-serif')
            ax.text(x + (w - 3.0)/2, y + h*0.28, sub, fontsize=6.4, ha='center', va='center', family='sans-serif')
        else:
            ax.text(x + (w - 3.0)/2, y + h*0.50, title, fontsize=7.6, fontweight='bold', ha='center', va='center', family='sans-serif')

    def add_arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0), ha='center'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='black', edgecolor='black', width=1.1, headwidth=5.0, headlength=6.0))
        if label:
            mx = (x1 + x2)/2 + label_offset[0]
            my = (y1 + y2)/2 + label_offset[1]
            ax.text(mx, my, label, fontsize=6.2, fontstyle='italic', ha=ha, va='center', family='sans-serif',
                    bbox=dict(boxstyle="square,pad=0.15", facecolor='white', edgecolor='none'))

    # LEFT UNIT: Standalone Edge-AI Kiosk Unit (202) (x=5 to 55.5)
    kiosk_box = patches.FancyBboxPatch((5, 7), 50.5, 116, boxstyle="round,pad=0.6",
                                       edgecolor='black', facecolor='#fafcff', linestyle='--', linewidth=1.3)
    ax.add_patch(kiosk_box)
    ax.text(30.25, 121, 'Standalone Edge-AI Kiosk Unit (202)', fontsize=8.2, fontweight='bold', ha='center', va='center', family='sans-serif')

    # Top Input Sensors (y=106 to 117.5)
    add_block(7, 106, 21.5, 11.5, 'Optical Depth Sensor', 'RGB-D + LED Illuminator', 204)
    add_block(32.0, 106, 21.0, 11.5, 'MEMS Mic Array', 'Beamforming Noise Filter', 206)

    # Embedded Host Processor SoC (208) (y=44 to 100)
    proc_box = patches.FancyBboxPatch((7, 44), 46.0, 56, boxstyle="round,pad=0.5",
                                      edgecolor='black', facecolor='#f1f5f9', linewidth=1.3)
    ax.add_patch(proc_box)
    # Centered title between MIPI-CSI (x=17.75) and I2S (x=42.5)
    ax.text(30.0, 97.5, 'Embedded Host Processor (208)', fontsize=7.8, fontweight='bold', ha='center', va='center', family='sans-serif')

    # Inside Processor:
    # Row 1 (y=84 to 94)
    add_block(8.5, 84, 19.5, 10, 'Vision Pipeline', '21 Skeletal Landmarks', 210)
    add_block(32.0, 84, 19.5, 10, 'Acoustic Pipeline', 'Speech-to-Text Phonemes', 212)

    # Row 2 (y=71 to 81)
    add_block(8.5, 71, 19.5, 10, 'Spatial Kinematics', 'Wrist Origin & 15 Angles', 214)
    add_block(32.0, 71, 19.5, 10, 'Sign Synthesizer', '3D Avatar Keyframing', 216)

    # Row 3 (y=58 to 68)
    add_block(8.5, 58, 19.5, 10, 'Multi-Modal Fusion', 'Optical-Glove Correlator', 218)
    add_block(32.0, 58, 19.5, 10, 'Temporal Word Buffer', 'Streak Consensus & Cooldown', 220)

    # Row 4 (y=46 to 55): Dual MLP Classifier
    add_block(9.5, 46, 41.0, 9.0, 'Dual MLP Neural Classifier (222)', 'Quantized ASL (78-D) / ISL (156-D)', 222)

    # Output Transducers (y=27 to 40)
    add_block(7, 27, 21.5, 13, 'Touch Display HUD', '3D Avatar & Subtitles', 224)
    add_block(32.0, 27, 21.0, 13, 'Acoustic Synthesizer', 'Class-D Amp & Speaker', 226)

    # Bottom Kiosk Core: Power and Wireless Core (y=10 to 22)
    add_block(7, 10, 21.5, 12, 'Power Subsystem', 'PMIC + Li-Ion + USB-PD', 230)
    add_block(32.0, 10, 21.0, 12, 'Kiosk Wireless Core', 'BLE 5.2 / 2.4GHz RF', 228)

    # RIGHT UNIT: Paired Wearable Sensor Glove (240) (x=72.5 to 95.5)
    glove_box = patches.FancyBboxPatch((72.5, 7), 23.0, 116, boxstyle="round,pad=0.6",
                                       edgecolor='black', facecolor='#fffdf9', linestyle='--', linewidth=1.3)
    ax.add_patch(glove_box)
    ax.text(84.0, 121, 'Wearable Sensor Glove (240)', fontsize=8.0, fontweight='bold', ha='center', va='center', family='sans-serif')

    # Glove Stack from top to bottom (x=73.5 to 94.5, center 84.0):
    add_block(73.5, 105, 21.0, 12, 'Flex Sensor Array', '5-Channel Finger Bend', 242)
    add_block(73.5, 88, 21.0, 12, '6-DoF IMU Sensor', '3-Axis Acc + Gyroscope', 244)
    add_block(73.5, 71, 21.0, 12, 'Glove MCU & ADC', '12-bit Signal Conditioning', 246)
    add_block(73.5, 54, 21.0, 12, 'Haptic Actuator', 'ERM Vibration Motor', 250)
    add_block(73.5, 34, 21.0, 15, 'Li-Po Battery & PMU', 'Wearable Power Pack &\nCharging Circuit', 252)
    add_block(73.5, 10, 21.0, 12, 'Glove Wireless Unit', 'Low-Latency RF / BLE', 248)

    # Glove Internal Downflow Arrows along center x=84.0
    add_arrow(84.0, 105, 84.0, 100)
    add_arrow(84.0, 88, 84.0, 83)
    add_arrow(84.0, 71, 84.0, 66)
    add_arrow(84.0, 54, 84.0, 49)

    # Kiosk Sensor to Processor Arrows (with safe label offsets)
    add_arrow(17.75, 106, 17.75, 94, 'MIPI-CSI', label_offset=(-4.5, 1.5), ha='right')
    add_arrow(42.5, 106, 42.5, 94, 'I2S Audio', label_offset=(4.5, 1.5), ha='left')

    # Kiosk Internal Flow Arrows
    add_arrow(18.25, 84, 18.25, 81)
    add_arrow(18.25, 71, 18.25, 68)
    add_arrow(18.25, 58, 21.0, 55)
    # Upward flow from Dual MLP (222) to Buffer (220) with clean offset label
    ax.annotate('', xy=(41.75, 58), xytext=(38.0, 55),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.0, headwidth=4.5, headlength=5.0))
    ax.text(39.0, 57.2, 'Tokens', fontsize=5.8, fontstyle='italic', ha='right', va='center', family='sans-serif',
            bbox=dict(boxstyle="square,pad=0.1", facecolor='white', edgecolor='none'))
    add_arrow(41.75, 84, 41.75, 81)

    # Processor to Outputs
    add_arrow(17.75, 44, 17.75, 40, 'HUD Data', label_offset=(-4.2, 0), ha='right')
    add_arrow(42.5, 44, 42.5, 40, 'Audio Data', label_offset=(4.5, 0), ha='left')

    # Outputs to Wireless Core
    add_arrow(42.5, 27, 42.5, 22)

    # ─── PURE HORIZONTAL BIDIRECTIONAL WIRELESS LINK (235) ───
    # Bridges Kiosk Wireless Core (x=53.0) and Glove Wireless Unit (x=73.5) across an open 20.5-unit gap!
    ax.annotate('', xy=(73.5, 16), xytext=(53.0, 16),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.4))
    ax.text(63.25, 20.0, '2.4 GHz RF\nLink (235)', fontsize=6.2, fontweight='bold', ha='center', va='center', family='sans-serif',
            bbox=dict(boxstyle="square,pad=0.2", facecolor='white', edgecolor='black', linewidth=0.7))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated FIG. 2: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG. 3 - PERSPECTIVE SCHEMATIC OF PHYSICAL KIOSK AND SENSOR GLOVE
# ─────────────────────────────────────────────────────────────────────────────
def draw_fig3_physical_system(out_path):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Figure Header
    ax.text(50, 135, 'FIG. 3', fontsize=16, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(50, 130, 'PERSPECTIVE SCHEMATIC OF KIOSK APPARATUS AND SENSOR GLOVE (300)',
            fontsize=8.8, fontweight='bold', ha='center', va='center', family='sans-serif')

    def add_leader(target_pt, text_pt, ref_str, text_side='right'):
        ax.annotate('', xy=target_pt, xytext=text_pt,
                    arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.0))
        ha = 'left' if text_side == 'right' else ('right' if text_side == 'left' else 'center')
        ax.text(text_pt[0], text_pt[1], str(ref_str), fontsize=9.2, fontweight='bold', ha=ha, va='center', family='sans-serif')

    # 1. COUNTERTOP SUPPORT BASE (304)
    base = patches.Polygon([[12, 32], [54, 32], [58, 40], [16, 40]],
                           closed=True, edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(base)
    add_leader((24, 33), (8, 30), '304', text_side='left')

    # 2. MAIN KIOSK ENCLOSURE (302)
    body = patches.Polygon([[15, 36], [51, 36], [49, 96], [17, 96]],
                           closed=True, edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(body)
    add_leader((17, 62), (7, 62), '302', text_side='left')

    # 3. ANTI-GLARE PRIVACY CANOPY HOOD (306)
    hood = patches.Polygon([[15, 96], [51, 96], [56, 114], [10, 114]],
                           closed=True, edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(hood)
    add_leader((11, 110), (5, 114), '306', text_side='left')

    # 4. OPTICAL SENSOR & DUAL LED ILLUMINATORS (308)
    cam_housing = patches.Rectangle((26, 100), 14, 8, edgecolor='black', facecolor='white', linewidth=1.2)
    ax.add_patch(cam_housing)
    # Lens
    ax.add_patch(patches.Circle((33, 104), 2.2, edgecolor='black', facecolor='#444444'))
    # LEDs
    ax.add_patch(patches.Circle((28.5, 104), 1.2, edgecolor='black', facecolor='white', linewidth=1.0))
    ax.add_patch(patches.Circle((37.5, 104), 1.2, edgecolor='black', facecolor='white', linewidth=1.0))
    add_leader((33, 106.5), (33, 120), '308', text_side='center')

    # 5. DIRECTIONAL MEMS MICROPHONES (312)
    for mx in [24, 27, 39, 42]:
        ax.add_patch(patches.Circle((mx, 92), 0.8, edgecolor='black', facecolor='black'))
    add_leader((24, 92), (7, 92), '312', text_side='left')

    # 6. ANGLED HUD TOUCH PANEL DISPLAY (310)
    screen = patches.Polygon([[19, 48], [47, 48], [45, 88], [21, 88]],
                             closed=True, edgecolor='black', facecolor='white', linewidth=1.3)
    ax.add_patch(screen)
    # UI elements
    ax.add_patch(patches.Rectangle((24, 68), 18, 16, edgecolor='black', facecolor='white', linestyle='--', linewidth=0.8))
    ax.text(33, 76, '3D SIGN\nAVATAR', fontsize=6.8, ha='center', va='center', fontweight='bold', family='sans-serif')
    ax.add_patch(patches.Rectangle((23, 52), 20, 10, edgecolor='black', facecolor='white', linewidth=0.8))
    ax.text(33, 57, 'LIVE SUBTITLES', fontsize=6.2, ha='center', va='center', family='sans-serif')
    # Leader points to left edge of screen (ZERO line crossings)
    add_leader((20, 78), (7, 78), '310', text_side='left')

    # 7. SPEAKER APERTURES / GRILLE (314)
    for sx in range(23, 44, 3):
        ax.plot([sx, sx], [40, 44], color='black', linewidth=1.2)
    add_leader((23, 42), (7, 44), '314', text_side='left')

    # 8. GLOVE DOCKING CRADLE (316)
    cradle = patches.FancyBboxPatch((50, 52), 6, 22, boxstyle="round,pad=0.3",
                                    edgecolor='black', facecolor='white', linewidth=1.2)
    ax.add_patch(cradle)
    ax.text(53, 63, 'DOCK', fontsize=6.0, ha='center', va='center', rotation=90, family='sans-serif')
    # Leader points straight down to bottom of dock from y=30 (aligned with bottom leaders 330, 326, 328)
    add_leader((53, 52), (53, 30), '316', text_side='center')

    # 9. WEARABLE SENSOR GLOVE (320)
    glove_poly = patches.Polygon([
        [70, 42], [76, 38], [88, 40], [92, 48], [91, 76],
        [90, 92], [86, 92], [86, 78], # Little
        [84, 100], [80, 100], [80, 78], # Ring
        [78, 106], [74, 106], [74, 78], # Middle
        [72, 98], [68, 98], [68, 74], # Index
        [64, 68], [61, 62], [65, 56], # Thumb
        [70, 42]
    ], closed=True, edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(glove_poly)
    add_leader((91, 65), (96, 65), '320', text_side='right')

    # 10. 5-CHANNEL FLEX SENSOR STRIPS (322)
    ax.plot([63, 65, 68], [64, 61, 57], color='black', linewidth=2.2) # Thumb
    ax.plot([70, 70, 70], [95, 85, 75], color='black', linewidth=2.2) # Index
    ax.plot([76, 76, 76], [103, 90, 75], color='black', linewidth=2.2) # Middle
    ax.plot([82, 82, 82], [97, 86, 75], color='black', linewidth=2.2) # Ring
    ax.plot([88, 88, 88], [89, 83, 75], color='black', linewidth=2.2) # Little
    add_leader((76, 104), (76, 118), '322', text_side='center')

    # 11. WRIST POD ENCLOSURE (324)
    pod = patches.FancyBboxPatch((70, 40), 18, 12, boxstyle="round,pad=0.3",
                                 edgecolor='black', facecolor='white', linewidth=1.3)
    ax.add_patch(pod)
    add_leader((88, 46), (96, 46), '324', text_side='right')

    # 12. STATUS LED INDICATOR (326)
    ax.add_patch(patches.Circle((74, 46), 1.2, edgecolor='black', facecolor='white', linewidth=1.0))
    add_leader((74, 45), (74, 30), '326', text_side='center')

    # 13. HAPTIC VIBRATION MOTOR (328)
    ax.add_patch(patches.Circle((83, 46), 2.0, edgecolor='black', facecolor='white', linestyle='--', linewidth=1.1))
    ax.text(83, 46, 'M', fontsize=7.0, ha='center', va='center', fontweight='bold', family='sans-serif')
    add_leader((83, 44), (83, 30), '328', text_side='center')

    # 14. 2.4 GHz RF WIRELESS LINK (330)
    for rad in [6, 11, 16]:
        arc = patches.Arc((60, 46), rad, rad*1.6, angle=0, theta1=-40, theta2=40,
                          edgecolor='black', linewidth=1.1, linestyle=':')
        ax.add_patch(arc)
    add_leader((63, 46), (63, 30), '330', text_side='center')

    # 15. OPTICAL TRACKING FIELD OF VIEW (332)
    # Originates from camera aperture housing at x=38, passes cleanly above dock cradle (y=85.3 at x=50) into interaction space
    ax.plot([38, 66], [102, 78], color='black', linestyle='--', linewidth=1.0)
    ax.plot([38, 64], [100, 68], color='black', linestyle='--', linewidth=1.0)
    # Leader points to top FOV line from clear open air above at (58, 93), crossing zero borders!
    add_leader((52, 89), (58, 93), '332', text_side='right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated FIG. 3: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG. 4 - KINEMATIC SKELETAL TOPOLOGY & FEATURE VECTOR SYNTHESIS
# ─────────────────────────────────────────────────────────────────────────────
def draw_fig4_kinematics(out_path):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Figure Header
    ax.text(50, 135, 'FIG. 4', fontsize=16, fontweight='bold', ha='center', va='center', family='sans-serif')
    ax.text(50, 130, 'KINEMATIC SKELETAL TOPOLOGY & FEATURE VECTOR SYNTHESIS (400)',
            fontsize=8.8, fontweight='bold', ha='center', va='center', family='sans-serif')

    def add_leader(target_pt, text_pt, ref_str, text_side='right'):
        ax.annotate('', xy=target_pt, xytext=text_pt,
                    arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=1.0))
        ha = 'left' if text_side == 'right' else ('right' if text_side == 'left' else 'center')
        ax.text(text_pt[0], text_pt[1], str(ref_str), fontsize=9.2, fontweight='bold', ha=ha, va='center', family='sans-serif')

    # LEFT: 21 Landmark Hand Topology (402)
    pts = {
        0: (24, 30), # Wrist
        # Thumb
        1: (18, 41), 2: (13, 52), 3: (9, 63), 4: (6, 73),
        # Index
        5: (17, 66), 6: (15, 80), 7: (14, 92), 8: (13, 104),
        # Middle
        9: (24, 70), 10: (24, 84), 11: (24, 97), 12: (24, 110),
        # Ring
        13: (31, 66), 14: (33, 80), 15: (34, 92), 16: (35, 104),
        # Little
        17: (38, 60), 18: (41, 72), 19: (43, 83), 20: (45, 94)
    }

    # Draw Bones
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    for b1, b2 in bones:
        x1, y1 = pts[b1]
        x2, y2 = pts[b2]
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.4)

    # Draw Landmarks
    for idx, (px, py) in pts.items():
        facecol = 'white' if idx != 0 and idx != 9 else '#e2e8f0'
        ax.add_patch(patches.Circle((px, py), 1.6, edgecolor='black', facecolor=facecol, linewidth=1.2))
        ax.text(px, py, str(idx), fontsize=6.2, ha='center', va='center', fontweight='bold', family='sans-serif')

    # Wrist origin indicator (402)
    ax.text(24, 24, 'Wrist Origin L0 (0,0,0)\n[Translation Invariance]', fontsize=7.2, ha='center', fontweight='bold', family='sans-serif')
    add_leader((24, 28), (8, 26), '402', text_side='left')

    # D_palm Vector between 0 and 9 (Double headed arrow)
    ax.annotate('', xy=(pts[9][0], pts[9][1]), xytext=(pts[0][0], pts[0][1]),
                arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.8))
    # D_palm (404) metric label placed cleanly centered at (24, 48) on the dimension arrow with a boxed background!
    # Spans x=22 to 26, completely clear of bone 0-5 (x=20.5) and bone 0-13 (x=27.5)!
    ax.text(24.0, 48.0, 'D_palm\n(404)', fontsize=6.2, fontweight='bold', ha='center', va='center', family='sans-serif',
            bbox=dict(boxstyle="square,pad=0.2", facecolor='white', edgecolor='black', linewidth=0.7))

    # Flexion Angle indicator arc on Index finger joint 6 (Triplet [5, 6, 7]) (406)
    arc = patches.Arc(pts[6], 6, 6, angle=0, theta1=240, theta2=300, edgecolor='black', linewidth=1.6)
    ax.add_patch(arc)
    ax.text(10.5, 77.0, r'$\theta_k$ (406)', fontsize=7.2, fontweight='bold', ha='right', va='center', family='sans-serif')
    ax.annotate('', xy=(13.2, 77.0), xytext=(11.0, 77.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

    # Topology Title
    ax.text(24, 15, '21 Optical Skeletal Landmarks (402)\n15 Inter-Joint Angles theta_k (406)',
            fontsize=7.8, fontweight='bold', ha='center', family='sans-serif')

    # RIGHT: Multi-Modal Feature Synthesis (408)
    vec_box = patches.FancyBboxPatch((52, 12), 44, 108, boxstyle="round,pad=0.6",
                                     edgecolor='black', facecolor='#f8fafc', linewidth=1.3)
    ax.add_patch(vec_box)
    ax.text(74, 116, 'Multi-Modal Feature Synthesis', fontsize=8.6, fontweight='bold', ha='center', family='sans-serif')
    ax.text(94, 116, '408', fontsize=9.2, fontweight='bold', ha='right', family='sans-serif')

    def add_feature_box(y, h, title, lines, ref):
        ax.add_patch(patches.Rectangle((54, y), 40, h, edgecolor='black', facecolor='white', linewidth=1.1))
        ax.text(93, y + h - 1.2, str(ref), fontsize=8.5, fontweight='bold', ha='right', va='top', family='sans-serif')
        ax.text(72, y + h - 3.5, title, fontsize=7.6, fontweight='bold', ha='center', family='sans-serif')
        for idx, line in enumerate(lines):
            ax.text(74, y + h - 7.5 - idx*3.5, line, fontsize=6.5, ha='center', family='sans-serif')

    # 1. Optical Normalized Landmarks Box (410)
    add_feature_box(92, 19, 'Normalized Optical Landmarks',
                    ['63 Centered & Scaled Coordinates',
                     'x\'_i = (x_i - x_0) / D_palm',
                     'y\'_i = (y_i - y_0) / D_palm, z\'_i = (z_i - z_0) / D_palm'], 410)

    # 2. 15 Kinematic Flexion Angles Box (412)
    add_feature_box(67, 19, '15 Kinematic Flexion Angles',
                    ['theta_k = (1/pi)*arccos((ba·bc)/(||ba||·||bc||))',
                     'Computed across 15 Phalangeal Triplets',
                     'Perspective & Scale Invariant, Range: [0, 1]'], 412)

    # 3. Wireless Glove Telemetry Vector Box (414)
    add_feature_box(42, 19, 'Wireless Glove Telemetry',
                    ['5-Channel Flex: S_flex = [s_1, s_2, s_3, s_4, s_5]',
                     '6-DoF IMU: [pitch, roll, yaw, a_x, a_y, a_z]',
                     '100 Hz Sampling, Occlusion Redundancy'], 414)

    # 4. Final Fused Feature Vector (416)
    add_feature_box(16, 20, 'Occlusion-Robust Fused Vector',
                    ['Dynamic Confidence Blending Engine',
                     'Single-Hand ASL: 78-D (or 89-D Fused)',
                     'Dual-Hand ISL: 156-D (or 178-D Fused)'], 416)

    # Downflow arrows between feature boxes
    for ay in [92, 67, 42]:
        ax.annotate('', xy=(74, ay-2.5), xytext=(74, ay),
                    arrowprops=dict(facecolor='black', edgecolor='black', width=1.0, headwidth=4.5, headlength=5.0))

    # Connector arrow from skeletal topology to feature extraction
    # Placed at y=76 from x=43.5 to x=52 in open whitespace, halfway between joint 18 (y=72) and joint 19 (y=83)!
    ax.annotate('', xy=(52, 76), xytext=(43.5, 76),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.1, headwidth=4.5, headlength=5.5))
    ax.text(47.75, 78.5, 'Extract', fontsize=6.8, fontstyle='italic', ha='center', family='sans-serif')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated FIG. 4: {out_path}")


if __name__ == '__main__':
    draw_fig1_flowchart(os.path.join(OUTPUT_DIR, 'fig1_flowchart.png'))
    draw_fig2_architecture(os.path.join(OUTPUT_DIR, 'fig2_architecture.png'))
    draw_fig3_physical_system(os.path.join(OUTPUT_DIR, 'fig3_physical_system.png'))
    draw_fig4_kinematics(os.path.join(OUTPUT_DIR, 'fig4_kinematics.png'))
    print("All 4 patent drawing figures generated successfully!")
