import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def setup_canvas():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis('off')
    return fig, ax

def draw_front_view(path):
    fig, ax = setup_canvas()

    # Outer Stand / Base (Beveled pedestal)
    base = patches.FancyBboxPatch((28, 6), 44, 7, boxstyle="round,pad=0.5",
                                  facecolor='#e2e8f0', edgecolor='#1e293b', linewidth=2)
    ax.add_patch(base)
    # Anti-slip foot pads
    ax.add_patch(patches.Rectangle((32, 4), 6, 2, facecolor='#475569', edgecolor='#1e293b', linewidth=1.2))
    ax.add_patch(patches.Rectangle((62, 4), 6, 2, facecolor='#475569', edgecolor='#1e293b', linewidth=1.2))

    # Support Neck / Stem
    neck = patches.Rectangle((46, 13), 8, 8, facecolor='#cbd5e1', edgecolor='#1e293b', linewidth=1.8)
    ax.add_patch(neck)

    # Main Device Enclosure Body
    main_body = patches.FancyBboxPatch((18, 21), 64, 48, boxstyle="round,pad=1.2",
                                       facecolor='#f8fafc', edgecolor='#0f172a', linewidth=2.2)
    ax.add_patch(main_body)

    # Screen Bezel
    screen_bezel = patches.FancyBboxPatch((22, 25), 56, 38, boxstyle="round,pad=0.6",
                                          facecolor='#0f172a', edgecolor='#334155', linewidth=1.5)
    ax.add_patch(screen_bezel)

    # Active Touch Display Area
    screen = patches.Rectangle((24, 27), 52, 34, facecolor='#1e293b', edgecolor='#475569', linewidth=1)
    ax.add_patch(screen)

    # Avatar & HUD representation on screen
    # Avatar Head outline
    ax.add_patch(patches.Ellipse((50, 47), 8, 10, facecolor='#e2e8f0', edgecolor='#94a3b8', linewidth=1))
    # Avatar Torso
    ax.add_patch(patches.Polygon([[44, 38], [56, 38], [58, 28], [42, 28]], facecolor='#cbd5e1', edgecolor='#94a3b8', linewidth=1))
    # Subtitle Pill at bottom of screen
    ax.add_patch(patches.FancyBboxPatch((28, 29), 44, 4, boxstyle="round,pad=0.3",
                                        facecolor='#334155', edgecolor='#64748b', linewidth=0.8))
    ax.text(50, 31, 'TRANSLATED SUBTITLE DISPLAY', color='#93c5fd', fontsize=6.5, fontweight='bold', ha='center', va='center')

    # Top Sensor Pod (Integrated Optical Camera + LED Array + Mics)
    sensor_pod = patches.FancyBboxPatch((36, 67), 28, 6.5, boxstyle="round,pad=0.5",
                                        facecolor='#e2e8f0', edgecolor='#0f172a', linewidth=1.8)
    ax.add_patch(sensor_pod)

    # Wide-Angle Camera Lens (Center)
    ax.add_patch(patches.Circle((50, 70.2), 2.2, facecolor='#0f172a', edgecolor='#38bdf8', linewidth=1.2))
    ax.add_patch(patches.Circle((50, 70.2), 0.9, facecolor='#0284c7'))

    # Ring LED Illuminators (Left and Right of Lens)
    ax.add_patch(patches.Circle((45, 70.2), 0.9, facecolor='#fef08a', edgecolor='#ca8a04', linewidth=0.8))
    ax.add_patch(patches.Circle((55, 70.2), 0.9, facecolor='#fef08a', edgecolor='#ca8a04', linewidth=0.8))

    # MEMS Microphone Pinhole Ports
    ax.add_patch(patches.Circle((40, 70.2), 0.5, facecolor='#0f172a'))
    ax.add_patch(patches.Circle((60, 70.2), 0.5, facecolor='#0f172a'))

    # Front Bottom Speaker Array Slots
    for sx in np.linspace(35, 65, 11):
        ax.add_patch(patches.FancyBboxPatch((sx-0.5, 22.5), 1, 1.2, boxstyle="round,pad=0.1",
                                            facecolor='#94a3b8', edgecolor='#64748b', linewidth=0.5))

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def draw_back_view(path):
    fig, ax = setup_canvas()

    # Base & Feet
    base = patches.FancyBboxPatch((28, 6), 44, 7, boxstyle="round,pad=0.5",
                                  facecolor='#e2e8f0', edgecolor='#1e293b', linewidth=2)
    ax.add_patch(base)
    ax.add_patch(patches.Rectangle((32, 4), 6, 2, facecolor='#475569', edgecolor='#1e293b', linewidth=1.2))
    ax.add_patch(patches.Rectangle((62, 4), 6, 2, facecolor='#475569', edgecolor='#1e293b', linewidth=1.2))

    # Neck
    neck = patches.Rectangle((46, 13), 8, 8, facecolor='#cbd5e1', edgecolor='#1e293b', linewidth=1.8)
    ax.add_patch(neck)

    # Main Enclosure Body (Rear Shell)
    main_body = patches.FancyBboxPatch((18, 21), 64, 48, boxstyle="round,pad=1.2",
                                       facecolor='#f1f5f9', edgecolor='#0f172a', linewidth=2.2)
    ax.add_patch(main_body)

    # Rear Cooling Ventilation Grills (Horizontal Heat Exhaust Slits)
    for vy in np.linspace(42, 58, 7):
        ax.add_patch(patches.FancyBboxPatch((30, vy), 40, 1.2, boxstyle="round,pad=0.2",
                                            facecolor='#cbd5e1', edgecolor='#64748b', linewidth=0.8))

    # Rear I/O Panel Recess
    io_panel = patches.FancyBboxPatch((34, 26), 32, 10, boxstyle="round,pad=0.4",
                                      facecolor='#e2e8f0', edgecolor='#475569', linewidth=1.2)
    ax.add_patch(io_panel)

    # USB-C Power/Data Port
    ax.add_patch(patches.FancyBboxPatch((38, 29.5), 5, 2.5, boxstyle="round,pad=0.3",
                                        facecolor='#0f172a', edgecolor='#64748b', linewidth=0.8))
    # Power Button
    ax.add_patch(patches.Circle((50, 30.7), 2.0, facecolor='#dc2626', edgecolor='#991b1b', linewidth=1.0))
    # Reset/Config Tactile Switch
    ax.add_patch(patches.Circle((60, 30.7), 1.2, facecolor='#475569', edgecolor='#334155', linewidth=0.8))

    # Top Sensor Pod (Back View)
    sensor_pod = patches.FancyBboxPatch((36, 67), 28, 6.5, boxstyle="round,pad=0.5",
                                        facecolor='#e2e8f0', edgecolor='#0f172a', linewidth=1.8)
    ax.add_patch(sensor_pod)
    # Tilt Angle Adjustment Hinge
    ax.add_patch(patches.FancyBboxPatch((44, 66.5), 12, 2.5, boxstyle="round,pad=0.2",
                                        facecolor='#94a3b8', edgecolor='#334155', linewidth=1.0))

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def draw_side_view(path, is_left=True):
    fig, ax = setup_canvas()

    # Pedestal Base Profile (Slanted profile)
    bx = 32 if is_left else 36
    base = patches.Polygon([[bx, 6], [bx+32, 6], [bx+28, 13], [bx+4, 13]],
                           facecolor='#e2e8f0', edgecolor='#1e293b', linewidth=2)
    ax.add_patch(base)
    # Anti-slip foot
    ax.add_patch(patches.Rectangle((bx+4, 4), 6, 2, facecolor='#475569', edgecolor='#1e293b', linewidth=1.2))
    ax.add_patch(patches.Rectangle((bx+22, 4), 6, 2, facecolor='#475569', edgecolor='#1e293b', linewidth=1.2))

    # Angled Support Neck
    neck = patches.Polygon([[bx+14, 13], [bx+22, 13], [bx+20, 23], [bx+12, 23]],
                           facecolor='#cbd5e1', edgecolor='#1e293b', linewidth=1.8)
    ax.add_patch(neck)

    # Main Body Profile (15-degree Ergonomic Viewing Tilt)
    # Front is facing left for left view, right for right view
    if is_left:
        body_poly = patches.Polygon([[38, 22], [54, 24], [48, 68], [32, 66]],
                                    facecolor='#f8fafc', edgecolor='#0f172a', linewidth=2.2)
        ax.add_patch(body_poly)
        # Front screen profile
        screen_line = patches.Polygon([[37, 26], [39, 26.5], [33.5, 63], [31.5, 62.5]],
                                      facecolor='#0f172a', edgecolor='#334155', linewidth=1)
        ax.add_patch(screen_line)
        # Top Sensor Pod (Forward tilted)
        pod = patches.Polygon([[30, 65], [44, 67], [42, 73], [28, 71]],
                              facecolor='#e2e8f0', edgecolor='#0f172a', linewidth=1.8)
        ax.add_patch(pod)
        # Camera Lens protrusion
        ax.add_patch(patches.Polygon([[27, 69.5], [29, 69.8], [28.5, 71.8], [26.5, 71.5]],
                                     facecolor='#0284c7', edgecolor='#0f172a', linewidth=1))
    else:
        body_poly = patches.Polygon([[46, 24], [62, 22], [68, 66], [52, 68]],
                                    facecolor='#f8fafc', edgecolor='#0f172a', linewidth=2.2)
        ax.add_patch(body_poly)
        # Front screen profile
        screen_line = patches.Polygon([[61, 26.5], [63, 26], [68.5, 62.5], [66.5, 63]],
                                      facecolor='#0f172a', edgecolor='#334155', linewidth=1)
        ax.add_patch(screen_line)
        # Top Sensor Pod (Forward tilted)
        pod = patches.Polygon([[56, 67], [70, 65], [72, 71], [58, 73]],
                              facecolor='#e2e8f0', edgecolor='#0f172a', linewidth=1.8)
        ax.add_patch(pod)
        # Camera Lens protrusion
        ax.add_patch(patches.Polygon([[71, 69.8], [73, 69.5], [73.5, 71.5], [71.5, 71.8]],
                                     facecolor='#0284c7', edgecolor='#0f172a', linewidth=1))

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def draw_top_view(path):
    fig, ax = setup_canvas()

    # Pedestal base outline visible underneath
    base_shadow = patches.FancyBboxPatch((28, 14), 44, 20, boxstyle="round,pad=1.0",
                                         facecolor='#e2e8f0', edgecolor='#94a3b8', linestyle=':', linewidth=1.5)
    ax.add_patch(base_shadow)

    # Main Housing Enclosure Top Profile
    body_top = patches.FancyBboxPatch((20, 24), 60, 24, boxstyle="round,pad=1.2",
                                      facecolor='#f8fafc', edgecolor='#0f172a', linewidth=2.2)
    ax.add_patch(body_top)

    # Beveled front edge
    ax.add_patch(patches.FancyBboxPatch((22, 25), 56, 2, boxstyle="round,pad=0.2",
                                        facecolor='#0f172a', edgecolor='#334155', linewidth=0.8))

    # Top Sensor Pod Housing
    sensor_top = patches.FancyBboxPatch((36, 38), 28, 14, boxstyle="round,pad=0.8",
                                        facecolor='#e2e8f0', edgecolor='#0f172a', linewidth=1.8)
    ax.add_patch(sensor_top)

    # Wide-Angle Lens Cap Outline
    ax.add_patch(patches.Ellipse((50, 48), 5.5, 4.5, facecolor='#0f172a', edgecolor='#38bdf8', linewidth=1.2))
    ax.add_patch(patches.Ellipse((50, 48), 2.5, 2.0, facecolor='#0284c7'))

    # Dual MEMS Mic Holes
    ax.add_patch(patches.Circle((42, 45), 0.8, facecolor='#0f172a'))
    ax.add_patch(patches.Circle((58, 45), 0.8, facecolor='#0f172a'))

    # Dual LED Diffusers
    ax.add_patch(patches.Circle((45, 48), 1.0, facecolor='#fef08a', edgecolor='#ca8a04', linewidth=0.8))
    ax.add_patch(patches.Circle((55, 48), 1.0, facecolor='#fef08a', edgecolor='#ca8a04', linewidth=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def draw_bottom_view(path):
    fig, ax = setup_canvas()

    # Pedestal Base Plate
    base = patches.FancyBboxPatch((24, 18), 52, 44, boxstyle="round,pad=1.5",
                                  facecolor='#f1f5f9', edgecolor='#0f172a', linewidth=2.2)
    ax.add_patch(base)

    # Four Anti-Skid Elastomeric Rubber Feet
    ax.add_patch(patches.Circle((32, 26), 3.2, facecolor='#334155', edgecolor='#0f172a', linewidth=1.2))
    ax.add_patch(patches.Circle((68, 26), 3.2, facecolor='#334155', edgecolor='#0f172a', linewidth=1.2))
    ax.add_patch(patches.Circle((32, 54), 3.2, facecolor='#334155', edgecolor='#0f172a', linewidth=1.2))
    ax.add_patch(patches.Circle((68, 54), 3.2, facecolor='#334155', edgecolor='#0f172a', linewidth=1.2))

    # Base Battery Access Cover Door
    door = patches.FancyBboxPatch((34, 32), 32, 16, boxstyle="round,pad=0.5",
                                  facecolor='#e2e8f0', edgecolor='#64748b', linewidth=1.2)
    ax.add_patch(door)
    # Latch Slot
    ax.add_patch(patches.FancyBboxPatch((46, 33.5), 8, 2, boxstyle="round,pad=0.2",
                                        facecolor='#475569', edgecolor='#334155', linewidth=0.8))

    # Bottom Ventilation Ports
    for vx in np.linspace(38, 62, 5):
        ax.add_patch(patches.Rectangle((vx, 51), 1.5, 4, facecolor='#94a3b8'))

    # Statutory Label Area (Specification Tag)
    ax.add_patch(patches.Rectangle((37, 21), 26, 7, facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def draw_perspective_view(path):
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1.2, 1.0, 1.1])
    ax.axis('off')

    # Draw 3D Box Wireframe / Shaded Voxels representing the apparatus
    # Base Pedestal
    X = np.array([[-3, 3], [-3, 3]])
    Y = np.array([[-3, -3], [3, 3]])
    Z_base = np.zeros_like(X)
    ax.plot_surface(X, Y, Z_base, color='#cbd5e1', edgecolors='#334155', alpha=0.8)

    # Enclosure Body Mesh
    # Bottom plate
    ax.plot_surface(X*1.2, Y*0.8, Z_base + 1.5, color='#94a3b8', edgecolors='#1e293b', alpha=0.9)
    # Front Screen surface (tilted)
    Xs = np.array([[-3.4, 3.4], [-3.0, 3.0]])
    Ys = np.array([[-1.8, -1.8], [-1.0, -1.0]])
    Zs = np.array([[2.0, 2.0], [7.0, 7.0]])
    ax.plot_surface(Xs, Ys, Zs, color='#0f172a', edgecolors='#38bdf8', linewidth=1.5, alpha=0.95)

    # Active display inset
    Xsi = np.array([[-3.0, 3.0], [-2.7, 2.7]])
    Ysi = np.array([[-1.7, -1.7], [-1.0, -1.0]])
    Zsi = np.array([[2.4, 2.4], [6.6, 6.6]])
    ax.plot_surface(Xsi, Ysi, Zsi, color='#1e293b', edgecolors='#60a5fa', linewidth=0.8, alpha=0.95)

    # Back plate (tilted)
    Xb = np.array([[-3.4, 3.4], [-3.0, 3.0]])
    Yb = np.array([[1.8, 1.8], [1.0, 1.0]])
    Zb = np.array([[2.0, 2.0], [7.0, 7.0]])
    ax.plot_surface(Xb, Yb, Zb, color='#f1f5f9', edgecolors='#334155', alpha=0.85)

    # Top Pod
    Xp = np.array([[-1.6, 1.6], [-1.4, 1.4]])
    Yp = np.array([[-1.2, -1.2], [0.6, 0.6]])
    Zp = np.array([[7.0, 7.0], [8.2, 8.2]])
    ax.plot_surface(Xp, Yp, Zp, color='#e2e8f0', edgecolors='#0f172a', linewidth=1.2, alpha=0.9)

    # Camera Lens element
    ax.scatter([0], [-1.3], [7.6], color='#0284c7', s=120, edgecolors='#38bdf8', depthshade=False)

    ax.view_init(elev=24, azim=-55)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == '__main__':
    out_dir = r'r:\Projects\Sign Language\DESIGN_REGISTRATION'
    os.makedirs(out_dir, exist_ok=True)
    draw_front_view(os.path.join(out_dir, 'fig1_front.png'))
    draw_back_view(os.path.join(out_dir, 'fig2_back.png'))
    draw_side_view(os.path.join(out_dir, 'fig3_left.png'), is_left=True)
    draw_side_view(os.path.join(out_dir, 'fig4_right.png'), is_left=False)
    draw_top_view(os.path.join(out_dir, 'fig5_top.png'))
    draw_bottom_view(os.path.join(out_dir, 'fig6_bottom.png'))
    draw_perspective_view(os.path.join(out_dir, 'fig7_perspective.png'))
    print('All 7 design views generated successfully.')
