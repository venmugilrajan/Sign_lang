import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_hardware_architecture(output_path):
    fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis('off')

    # Figure Label
    ax.text(50, 134, 'FIG. 2', fontsize=16, fontweight='bold', ha='center', va='center', family='sans-serif')

    # Main Apparatus Outer Box (Dashed)
    apparatus_box = patches.FancyBboxPatch(
        (4, 6), 92, 123,
        boxstyle="round,pad=1.2",
        edgecolor='black',
        facecolor='none',
        linestyle='--',
        linewidth=2
    )
    ax.add_patch(apparatus_box)
    ax.text(7, 126.5, 'Standalone Edge-AI Translation Apparatus (200)', fontsize=11, fontweight='bold', family='sans-serif')
    ax.text(93, 126.5, '200', fontsize=12, fontweight='bold', family='sans-serif')

    def add_block(x, y, w, h, title, subtitle, ref_num):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6",
            edgecolor='black',
            facecolor='white',
            linewidth=1.4
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, fontsize=9.2, fontweight='bold', ha='center', va='center', family='sans-serif')
            ax.text(x + w/2, y + h*0.32, subtitle, fontsize=7.8, ha='center', va='center', family='sans-serif')
        else:
            ax.text(x + w/2, y + h*0.5, title, fontsize=9.2, fontweight='bold', ha='center', va='center', family='sans-serif')
        ax.text(x + w + 1.2, y + h - 1, str(ref_num), fontsize=10.5, fontweight='bold', ha='left', va='top', family='sans-serif')

    def add_arrow(x1, y1, x2, y2, label=None, label_pos=(0, 0)):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor='black', edgecolor='black', width=1.4, headwidth=5.5, headlength=6.5)
        )
        if label:
            mx = (x1 + x2) / 2 + label_pos[0]
            my = (y1 + y2) / 2 + label_pos[1]
            ax.text(mx, my, label, fontsize=7.2, fontstyle='italic', ha='center', va='center', family='sans-serif',
                    bbox=dict(boxstyle="square,pad=0.2", facecolor='white', edgecolor='none'))

    # TOP INPUT TRANSDUCERS (Row 1)
    add_block(8, 110, 38, 12, 'Optical Vision Sensor (202)', 'Wide-Angle Camera + LED Illuminator', 202)
    add_block(54, 110, 38, 12, 'Acoustic Capture Unit (204)', 'MEMS Microphone Array with Noise Filter', 204)

    # CENTRAL EDGE EMBEDDED COMPUTING PLATFORM
    edge_box = patches.FancyBboxPatch(
        (6, 44), 88, 56,
        boxstyle="round,pad=0.8",
        edgecolor='black',
        facecolor='#f8fafc',
        linewidth=1.6
    )
    ax.add_patch(edge_box)
    ax.text(9, 97, 'Edge-AI Embedded Host Processor (206)', fontsize=10.5, fontweight='bold', family='sans-serif')
    ax.text(91, 97, '206', fontsize=11, fontweight='bold', family='sans-serif')

    # Left Column (Vision-to-Speech Processing)
    add_block(9, 81, 38, 11, 'Vision Pipeline (208)', '3D Joint Localization (21 Landmarks)', 208)
    add_block(9, 64, 38, 11, 'Spatial Kinematics Unit (212)', '15 Flexion Angles + Normalization', 212)
    add_block(9, 47, 38, 11, 'Edge Neural Classifier (216)', 'Quantized Dual MLP (ASL / ISL)', 216)

    # Right Column (Speech-to-Sign & Temporal Aggregation)
    add_block(53, 81, 38, 11, 'Acoustic Pipeline (210)', 'Speech-to-Text Phoneme Engine', 210)
    add_block(53, 64, 38, 11, 'Sign Synthesizer Engine (214)', '3D Humanoid Joint Retargeting', 214)
    add_block(53, 47, 38, 11, 'Temporal Word Buffer (218)', 'Consensus Voting & Cooldown Engine', 218)

    # OUTPUT TRANSDUCERS (Row 4)
    add_block(8, 24, 38, 14, 'Visual Display Subsystem (220)', 'High-Res Capacitive HUD Panel\n(3D Avatar & Live Subtitles)', 220)
    add_block(54, 24, 38, 14, 'Acoustic Synthesizer (222)', 'Audio Codec, Class-D Amp & Speaker\n(Real-Time Vocalized Speech)', 222)

    # POWER MODULE (Row 5 - Bottom)
    add_block(18, 9, 64, 9.5, 'Portable Power Subsystem (224)', 'Rechargeable Li-Ion Battery Pack + PMIC + USB-PD', 224)

    # ARROWS & BUSES
    # 1. Inputs to Edge Processor
    add_arrow(27, 110, 27, 92, 'MIPI-CSI Video Stream', (0, 0))
    add_arrow(73, 110, 73, 92, 'I2S Digital Audio', (0, 0))

    # 2. Vision Pipeline Downflow
    add_arrow(28, 81, 28, 75)
    add_arrow(28, 64, 28, 58)

    # 3. Vision Classifier to Temporal Word Buffer
    add_arrow(47, 52.5, 53, 52.5, 'Tokens', (0, 2))

    # 4. Word Buffer to Acoustic Synthesizer (Speech Output)
    add_arrow(72, 47, 73, 38, 'Committed Text', (0, 0))

    # 5. Acoustic Pipeline to Sign Synthesizer
    add_arrow(72, 81, 72, 75)

    # 6. Sign Synthesizer to Visual Display (Avatar Render)
    # Route via dedicated path
    ax.annotate(
        '', xy=(28, 38), xytext=(53, 68),
        arrowprops=dict(facecolor='black', edgecolor='black', width=1.4, headwidth=5.5, headlength=6.5)
    )
    ax.text(39, 58, '3D Joint Pose Keyframes', fontsize=7.2, fontstyle='italic', ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.2", facecolor='white', edgecolor='none'))

    # 7. Power Bus Indication
    ax.annotate(
        '', xy=(50, 44), xytext=(50, 18.5),
        arrowprops=dict(facecolor='black', edgecolor='black', width=1.2, headwidth=4.5, headlength=5)
    )
    ax.text(50, 21, 'Regulated 5V / 3.3V DC Power Bus', fontsize=7.0, fontstyle='italic', ha='center', va='center',
            bbox=dict(boxstyle="square,pad=0.2", facecolor='white', edgecolor='none'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated clean hardware diagram: {output_path}")

if __name__ == '__main__':
    draw_hardware_architecture(r'r:\Projects\Sign Language\PATENT\fig2_architecture.png')
