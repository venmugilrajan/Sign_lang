"""
Generate 7 statutory design registration views for:
EDGE-AI BIDIRECTIONAL SIGN LANGUAGE TRANSLATION KIOSK WITH WIRELESS SENSOR GLOVE

Hardware System:
  A) COUNTER KIOSK UNIT:
     - Compact box enclosure (like a small ATM terminal)
     - Angled privacy hood on top that shields the screen from ambient glare
       and provides visual privacy for the deaf user
     - Integrated depth camera (Intel RealSense style) recessed into the
       top of the privacy hood, looking down at the user's hands
     - 10" angled touchscreen display (tilted ~15° for ergonomic viewing)
     - Front-facing directional speaker grille below screen
     - Dual MEMS microphone array on the hood edges
     - Side panel: USB-C port, 3.5mm audio jack, power LED indicator
     - Rear panel: power inlet (IEC C14), Ethernet RJ45, ventilation grille,
       Kensington lock slot
     - Internal: Coral Edge TPU, Raspberry Pi CM4, battery backup module
     - Base: heavy anti-tip weighted base with cable management channel
       and anti-slip rubber feet

  B) WIRELESS SENSOR GLOVE (shown alongside kiosk in perspective view):
     - Lightweight fabric glove with 5 resistive flex sensors sewn along
       each finger dorsum
     - 9-axis IMU (accelerometer + gyroscope + magnetometer) on wrist
       dorsum PCB module
     - Bluetooth Low Energy (BLE) transmitter on wrist strap
     - Small LED status indicator (green=connected, red=low battery)
     - Rechargeable LiPo cell in wrist band
     - Micro-USB charging port on wrist strap edge
     - Adjustable velcro wrist closure

Views: Front, Back, Left, Right, Top, Bottom, Perspective
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Arc, FancyArrowPatch
import numpy as np
import os


# ---------- colour palette ----------
BODY      = '#e8ecf0'   # main enclosure
BODY_DK   = '#c8cfd8'   # body shading
HOOD      = '#d0d6de'   # privacy hood
HOOD_DK   = '#b0b8c4'
SCREEN_BZ = '#1a1e28'   # screen bezel
SCREEN    = '#0d1117'   # screen active area
ACCENT    = '#38bdf8'   # UI accent (sky-blue)
ACCENT2   = '#22d3ee'   # secondary accent
GRILLE    = '#8b95a5'   # speaker/vent grille
PORT_COL  = '#4b5563'   # ports & jacks
EDGE      = '#0f172a'   # outlines
FOOT      = '#374151'   # rubber feet
LED_GRN   = '#4ade80'   
LED_RED   = '#f87171'
GLOVE_COL = '#a8b8cc'   # glove fabric
SENSOR_COL= '#f59e0b'   # flex sensor strip
PCB_COL   = '#059669'   # wrist PCB
STRAP_COL = '#6b7280'   # wrist strap


def setup_canvas(figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    return fig, ax


# ========================================================================
#  FRONT VIEW
# ========================================================================
def draw_front_view(path):
    fig, ax = setup_canvas()

    # --- Base / weighted foot ---
    ax.add_patch(FancyBboxPatch((25, 3), 50, 5, boxstyle="round,pad=0.6",
        facecolor=FOOT, edgecolor=EDGE, linewidth=2))
    # anti-slip pads
    for bx in [30, 68]:
        ax.add_patch(patches.Rectangle((bx, 2), 5, 1.5, facecolor='#1f2937',
            edgecolor=EDGE, linewidth=0.8, zorder=3))
    # cable channel notch (center)
    ax.add_patch(patches.Rectangle((47, 2), 6, 1, facecolor='#1f2937',
        edgecolor=EDGE, linewidth=0.6))

    # --- Main body enclosure ---
    ax.add_patch(FancyBboxPatch((28, 8), 44, 42, boxstyle="round,pad=1",
        facecolor=BODY, edgecolor=EDGE, linewidth=2.2))

    # --- Screen bezel ---
    ax.add_patch(FancyBboxPatch((31, 12), 38, 28, boxstyle="round,pad=0.5",
        facecolor=SCREEN_BZ, edgecolor='#334155', linewidth=1.5))
    # Active screen
    ax.add_patch(patches.Rectangle((33, 14), 34, 24, facecolor=SCREEN,
        edgecolor='#475569', linewidth=0.8))

    # On-screen content: avatar silhouette + subtitle bar
    ax.add_patch(patches.Ellipse((50, 30), 7, 9, facecolor='#e2e8f0',
        edgecolor='#94a3b8', linewidth=0.8, alpha=0.7))
    ax.add_patch(patches.Polygon([[45, 22], [55, 22], [57, 15], [43, 15]],
        facecolor='#cbd5e1', edgecolor='#94a3b8', linewidth=0.6, alpha=0.7))
    ax.add_patch(FancyBboxPatch((35, 15), 30, 3.5, boxstyle="round,pad=0.3",
        facecolor='#1e293b', edgecolor='#475569', linewidth=0.5, alpha=0.85))
    ax.text(50, 16.8, 'SUBTITLE DISPLAY', color=ACCENT, fontsize=5,
        fontweight='bold', ha='center', va='center')

    # --- Speaker grille (below screen) ---
    for sx in np.linspace(37, 63, 15):
        ax.add_patch(FancyBboxPatch((sx-0.4, 9.5), 0.8, 1.8,
            boxstyle="round,pad=0.08", facecolor=GRILLE,
            edgecolor='#64748b', linewidth=0.4))

    # --- Privacy hood ---
    hood_pts = np.array([
        [26, 50], [74, 50],   # base of hood (sits on top of enclosure)
        [72, 64],             # top-right
        [28, 64],             # top-left
    ])
    ax.add_patch(patches.Polygon(hood_pts, closed=True, facecolor=HOOD,
        edgecolor=EDGE, linewidth=2))
    # Hood inner shadow
    ax.add_patch(patches.Polygon(
        [[28, 50], [72, 50], [71, 56], [29, 56]],
        closed=True, facecolor=HOOD_DK, edgecolor='none', alpha=0.5))

    # Depth camera recessed in hood underside (center)
    ax.add_patch(FancyBboxPatch((45, 51), 10, 4, boxstyle="round,pad=0.4",
        facecolor='#1e293b', edgecolor=EDGE, linewidth=1.2))
    # Camera lens
    ax.add_patch(patches.Circle((50, 53), 1.4, facecolor=EDGE,
        edgecolor=ACCENT, linewidth=1))
    ax.add_patch(patches.Circle((50, 53), 0.6, facecolor='#0284c7'))
    # IR emitters (depth camera)
    ax.add_patch(patches.Circle((47, 53), 0.5, facecolor='#7c3aed',
        edgecolor='#a78bfa', linewidth=0.5))
    ax.add_patch(patches.Circle((53, 53), 0.5, facecolor='#7c3aed',
        edgecolor='#a78bfa', linewidth=0.5))

    # MEMS microphones on hood edges
    ax.add_patch(patches.Circle((29, 57), 0.6, facecolor=EDGE))
    ax.add_patch(patches.Circle((71, 57), 0.6, facecolor=EDGE))

    # Hood front label
    ax.text(50, 61, 'SIGNSPEAK', color='#64748b', fontsize=5,
        fontweight='bold', ha='center', va='center',
        fontstyle='italic', alpha=0.6)

    # Status LED on body (below hood, right side)
    ax.add_patch(patches.Circle((69, 48), 0.7, facecolor=LED_GRN,
        edgecolor='#166534', linewidth=0.6))

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================
#  BACK VIEW
# ========================================================================
def draw_back_view(path):
    fig, ax = setup_canvas()

    # Base
    ax.add_patch(FancyBboxPatch((25, 3), 50, 5, boxstyle="round,pad=0.6",
        facecolor=FOOT, edgecolor=EDGE, linewidth=2))
    for bx in [30, 68]:
        ax.add_patch(patches.Rectangle((bx, 2), 5, 1.5, facecolor='#1f2937',
            edgecolor=EDGE, linewidth=0.8))

    # Main body (rear shell)
    ax.add_patch(FancyBboxPatch((28, 8), 44, 42, boxstyle="round,pad=1",
        facecolor=BODY_DK, edgecolor=EDGE, linewidth=2.2))

    # Ventilation grille (large, center-rear)
    for vy in np.linspace(20, 40, 10):
        ax.add_patch(FancyBboxPatch((35, vy), 30, 1, boxstyle="round,pad=0.15",
            facecolor='#b0b8c4', edgecolor='#64748b', linewidth=0.6))

    # I/O bay (recessed panel, lower-right)
    ax.add_patch(FancyBboxPatch((56, 10), 12, 8, boxstyle="round,pad=0.3",
        facecolor='#94a3b8', edgecolor=EDGE, linewidth=1.2))
    # Power inlet (IEC C14)
    ax.add_patch(patches.Rectangle((58, 14.5), 4, 2.5, facecolor=EDGE,
        edgecolor='#475569', linewidth=0.8))
    ax.text(60, 15.7, 'AC', color='white', fontsize=3.5, ha='center',
        va='center', fontweight='bold')
    # Ethernet RJ45
    ax.add_patch(patches.Rectangle((63, 14.5), 3.5, 2, facecolor=PORT_COL,
        edgecolor=EDGE, linewidth=0.8))
    ax.text(64.75, 15.5, 'LAN', color='white', fontsize=2.5, ha='center',
        va='center')
    # Kensington lock slot
    ax.add_patch(patches.Ellipse((60, 11.5), 2.5, 1, facecolor=EDGE))

    # Product label sticker
    ax.add_patch(patches.Rectangle((34, 10), 18, 6, facecolor='white',
        edgecolor='#94a3b8', linewidth=0.6))
    ax.text(43, 14.5, 'SIGNSPEAK', fontsize=4, ha='center', va='center',
        fontweight='bold', color='#334155')
    ax.text(43, 13, 'Model: SL-K100', fontsize=2.5, ha='center',
        va='center', color='#64748b')
    ax.text(43, 11.8, 'Edge TPU · CM4 · BLE 5.0', fontsize=2,
        ha='center', va='center', color='#94a3b8')

    # Privacy hood (rear view — solid panel)
    hood_pts = np.array([
        [26, 50], [74, 50], [72, 64], [28, 64]
    ])
    ax.add_patch(patches.Polygon(hood_pts, closed=True, facecolor=HOOD_DK,
        edgecolor=EDGE, linewidth=2))

    # Rear ventilation slits on hood
    for hy in np.linspace(53, 61, 4):
        ax.add_patch(FancyBboxPatch((40, hy), 20, 0.8,
            boxstyle="round,pad=0.1", facecolor='#94a3b8',
            edgecolor='#64748b', linewidth=0.4))

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================
#  SIDE VIEW (LEFT / RIGHT)
# ========================================================================
def draw_side_view(path, is_left=True):
    fig, ax = setup_canvas()

    # Base (side profile — narrow rectangle)
    ax.add_patch(FancyBboxPatch((30, 3), 40, 5, boxstyle="round,pad=0.5",
        facecolor=FOOT, edgecolor=EDGE, linewidth=2))
    ax.add_patch(patches.Rectangle((33, 2), 5, 1.5, facecolor='#1f2937',
        edgecolor=EDGE, linewidth=0.8))
    ax.add_patch(patches.Rectangle((62, 2), 5, 1.5, facecolor='#1f2937',
        edgecolor=EDGE, linewidth=0.8))

    # Body (side profile — taller rectangle, slightly narrower depth)
    ax.add_patch(FancyBboxPatch((35, 8), 30, 42, boxstyle="round,pad=0.8",
        facecolor=BODY if is_left else BODY_DK, edgecolor=EDGE, linewidth=2.2))

    # Screen edge visible on front side
    if is_left:
        ax.add_patch(patches.Rectangle((35.5, 14), 1.5, 26,
            facecolor=SCREEN_BZ, edgecolor='#334155', linewidth=0.8))
    else:
        ax.add_patch(patches.Rectangle((63, 14), 1.5, 26,
            facecolor=SCREEN_BZ, edgecolor='#334155', linewidth=0.8))

    # Side ports (on right side view)
    if not is_left:
        # USB-C port
        ax.add_patch(patches.Rectangle((63.5, 35), 2, 1, facecolor=PORT_COL,
            edgecolor=EDGE, linewidth=0.8))
        ax.text(64.5, 35.5, 'USB-C', color='white', fontsize=2,
            ha='center', va='center')
        # 3.5mm audio jack
        ax.add_patch(patches.Circle((64.5, 32), 0.6, facecolor=EDGE))
        # Volume rocker
        ax.add_patch(FancyBboxPatch((63.5, 27), 1.8, 4,
            boxstyle="round,pad=0.2", facecolor='#94a3b8',
            edgecolor=EDGE, linewidth=0.8))
    else:
        # SD card slot
        ax.add_patch(patches.Rectangle((35, 30), 1.5, 2.5,
            facecolor=PORT_COL, edgecolor=EDGE, linewidth=0.6))
        ax.text(35.8, 31.2, 'SD', color='white', fontsize=2,
            ha='center', va='center')

    # Privacy hood (side profile — trapezoidal)
    hood_pts = np.array([
        [33, 50], [67, 50],
        [63, 64], [37, 64],
    ])
    ax.add_patch(patches.Polygon(hood_pts, closed=True, facecolor=HOOD,
        edgecolor=EDGE, linewidth=2))

    # Show the hood overhang shadow
    ax.add_patch(patches.Polygon(
        [[34, 50], [66, 50], [65, 53], [35, 53]],
        closed=True, facecolor=HOOD_DK, edgecolor='none', alpha=0.4))

    # Power LED
    ax.add_patch(patches.Circle((50, 48), 0.6, facecolor=LED_GRN,
        edgecolor='#166534', linewidth=0.5))

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================
#  TOP VIEW
# ========================================================================
def draw_top_view(path):
    fig, ax = setup_canvas()

    # Hood outer shell (top-down view — trapezoid shape)
    hood_outer = np.array([
        [20, 20], [80, 20],   # rear edge
        [78, 60],             # front-right
        [22, 60],             # front-left
    ])
    ax.add_patch(patches.Polygon(hood_outer, closed=True, facecolor=HOOD,
        edgecolor=EDGE, linewidth=2.2))

    # Hood ventilation slits on top surface
    for vy in np.linspace(25, 45, 8):
        ax.add_patch(FancyBboxPatch((35, vy), 30, 0.8,
            boxstyle="round,pad=0.1", facecolor='#94a3b8',
            edgecolor='#7c8594', linewidth=0.5))

    # Camera module visible through top opening
    ax.add_patch(FancyBboxPatch((44, 52), 12, 5, boxstyle="round,pad=0.4",
        facecolor='#1e293b', edgecolor=EDGE, linewidth=1.2))
    # Depth camera lens
    ax.add_patch(patches.Circle((50, 54.5), 1.5, facecolor=EDGE,
        edgecolor=ACCENT, linewidth=1))
    ax.add_patch(patches.Circle((50, 54.5), 0.6, facecolor='#0284c7'))
    # IR emitters
    ax.add_patch(patches.Circle((47, 54.5), 0.5, facecolor='#7c3aed',
        edgecolor='#a78bfa', linewidth=0.4))
    ax.add_patch(patches.Circle((53, 54.5), 0.5, facecolor='#7c3aed',
        edgecolor='#a78bfa', linewidth=0.4))

    # MEMS mic positions (on hood edge)
    ax.add_patch(patches.Circle((24, 55), 0.7, facecolor=EDGE))
    ax.add_patch(patches.Circle((76, 55), 0.7, facecolor=EDGE))

    # Brand mark
    ax.text(50, 35, 'SIGNSPEAK', color='#94a3b8', fontsize=6,
        fontweight='bold', ha='center', va='center', alpha=0.4,
        fontstyle='italic')

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================
#  BOTTOM VIEW
# ========================================================================
def draw_bottom_view(path):
    fig, ax = setup_canvas()

    # Base footprint (bottom-up view)
    ax.add_patch(FancyBboxPatch((20, 15), 60, 50, boxstyle="round,pad=1",
        facecolor=FOOT, edgecolor=EDGE, linewidth=2.2))

    # Anti-slip rubber pads (4 corners)
    for (px, py) in [(25, 19), (25, 57), (71, 19), (71, 57)]:
        ax.add_patch(FancyBboxPatch((px, py), 6, 5, boxstyle="round,pad=0.3",
            facecolor='#1f2937', edgecolor=EDGE, linewidth=1))

    # Cable management channel (center groove)
    ax.add_patch(FancyBboxPatch((45, 17), 10, 46, boxstyle="round,pad=0.3",
        facecolor='#4b5563', edgecolor=EDGE, linewidth=0.8))

    # Weight ballast recess (center)
    ax.add_patch(patches.Circle((50, 40), 8, facecolor='#374151',
        edgecolor='#1f2937', linewidth=1.5))
    ax.text(50, 40, 'BALLAST', color='#6b7280', fontsize=4,
        ha='center', va='center', fontweight='bold')

    # Screw holes
    for (sx, sy) in [(30, 25), (30, 55), (70, 25), (70, 55)]:
        ax.add_patch(patches.Circle((sx, sy), 0.8, facecolor='#1f2937',
            edgecolor=EDGE, linewidth=0.6))

    # Certification recess
    ax.add_patch(patches.Rectangle((33, 22), 14, 5, facecolor='#4b5563',
        edgecolor='#374151', linewidth=0.5))
    ax.text(40, 24.5, 'CE  FCC  BIS', color='#9ca3af', fontsize=3,
        ha='center', va='center')

    # Serial number area
    ax.add_patch(patches.Rectangle((55, 22), 12, 3, facecolor='white',
        edgecolor='#94a3b8', linewidth=0.4))
    ax.text(61, 23.5, 'S/N: XXXXXX', color='#6b7280', fontsize=2.5,
        ha='center', va='center')

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================
#  PERSPECTIVE VIEW (Kiosk + Glove shown together)
# ========================================================================
def draw_perspective_view(path):
    fig = plt.figure(figsize=(12, 9), dpi=300)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-6, 10)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 12)
    ax.axis('off')

    # ---- KIOSK BASE ----
    bx = np.array([[-3, 3], [-3, 3]])
    by = np.array([[-2, -2], [2, 2]])
    bz_bot = np.array([[0, 0], [0, 0]])
    bz_top = np.array([[0.8, 0.8], [0.8, 0.8]])
    ax.plot_surface(bx, by, bz_bot, color=FOOT, edgecolors=EDGE,
        linewidth=0.8, alpha=0.9)
    ax.plot_surface(bx, by, bz_top, color=FOOT, edgecolors=EDGE,
        linewidth=0.8, alpha=0.9)
    # Base sides
    for fixed_y in [-2, 2]:
        sx = np.array([[-3, 3], [-3, 3]])
        sy = np.array([[fixed_y, fixed_y], [fixed_y, fixed_y]])
        sz = np.array([[0, 0], [0.8, 0.8]])
        ax.plot_surface(sx, sy, sz, color=FOOT, edgecolors=EDGE,
            linewidth=0.5, alpha=0.85)

    # ---- KIOSK BODY ----
    body_h = 6.5
    body_x = np.array([[-2.5, 2.5], [-2.5, 2.5]])
    body_y_front = np.array([[-1.5, -1.5], [-1.5, -1.5]])
    body_y_back = np.array([[1.5, 1.5], [1.5, 1.5]])
    body_z = np.array([[0.8, 0.8], [0.8 + body_h, 0.8 + body_h]])

    # Front face
    ax.plot_surface(body_x, body_y_front, body_z, color=BODY,
        edgecolors=EDGE, linewidth=0.8, alpha=0.95)
    # Back face
    ax.plot_surface(body_x, body_y_back, body_z, color=BODY_DK,
        edgecolors=EDGE, linewidth=0.8, alpha=0.85)
    # Left face
    lx = np.array([[-2.5, -2.5], [-2.5, -2.5]])
    ly = np.array([[-1.5, 1.5], [-1.5, 1.5]])
    ax.plot_surface(lx, ly, body_z, color=BODY_DK, edgecolors=EDGE,
        linewidth=0.5, alpha=0.85)
    # Right face
    rx = np.array([[2.5, 2.5], [2.5, 2.5]])
    ax.plot_surface(rx, ly, body_z, color=BODY, edgecolors=EDGE,
        linewidth=0.5, alpha=0.9)
    # Top face
    tx = np.array([[-2.5, 2.5], [-2.5, 2.5]])
    ty = np.array([[-1.5, -1.5], [1.5, 1.5]])
    tz = np.array([[0.8+body_h, 0.8+body_h], [0.8+body_h, 0.8+body_h]])
    ax.plot_surface(tx, ty, tz, color=BODY, edgecolors=EDGE,
        linewidth=0.5, alpha=0.9)

    # ---- SCREEN (on front face) ----
    scr_x = np.array([[-2, 2], [-2, 2]])
    scr_y = np.array([[-1.52, -1.52], [-1.52, -1.52]])
    scr_z = np.array([[1.5, 1.5], [5.5, 5.5]])
    ax.plot_surface(scr_x, scr_y, scr_z, color=SCREEN,
        edgecolors='#475569', linewidth=0.6, alpha=0.95)

    # ---- PRIVACY HOOD ----
    hood_base_z = 0.8 + body_h
    hood_top_z = hood_base_z + 2.5
    # Front panel of hood (angled outward)
    hf_x = np.array([[-2.8, 2.8], [-3.2, 3.2]])
    hf_y = np.array([[-1.8, -1.8], [-2.5, -2.5]])
    hf_z = np.array([[hood_base_z, hood_base_z], [hood_top_z, hood_top_z]])
    ax.plot_surface(hf_x, hf_y, hf_z, color=HOOD, edgecolors=EDGE,
        linewidth=0.8, alpha=0.9)
    # Back panel
    hb_x = np.array([[-2.8, 2.8], [-3.0, 3.0]])
    hb_y = np.array([[1.8, 1.8], [1.8, 1.8]])
    hb_z = np.array([[hood_base_z, hood_base_z], [hood_top_z, hood_top_z]])
    ax.plot_surface(hb_x, hb_y, hb_z, color=HOOD_DK, edgecolors=EDGE,
        linewidth=0.6, alpha=0.85)
    # Left side of hood
    hl_x = np.array([[-2.8, -2.8], [-3.2, -3.0]])
    hl_y = np.array([[-1.8, 1.8], [-2.5, 1.8]])
    hl_z = np.array([[hood_base_z, hood_base_z], [hood_top_z, hood_top_z]])
    ax.plot_surface(hl_x, hl_y, hl_z, color=HOOD_DK, edgecolors=EDGE,
        linewidth=0.5, alpha=0.85)
    # Right side of hood
    hr_x = np.array([[2.8, 2.8], [3.2, 3.0]])
    hr_y = np.array([[-1.8, 1.8], [-2.5, 1.8]])
    hr_z = np.array([[hood_base_z, hood_base_z], [hood_top_z, hood_top_z]])
    ax.plot_surface(hr_x, hr_y, hr_z, color=HOOD, edgecolors=EDGE,
        linewidth=0.5, alpha=0.9)
    # Top of hood
    ht_x = np.array([[-3.2, 3.2], [-3.0, 3.0]])
    ht_y = np.array([[-2.5, -2.5], [1.8, 1.8]])
    ht_z = np.array([[hood_top_z, hood_top_z], [hood_top_z, hood_top_z]])
    ax.plot_surface(ht_x, ht_y, ht_z, color=HOOD, edgecolors=EDGE,
        linewidth=0.5, alpha=0.9)

    # Camera (dot on hood underside)
    ax.scatter([0], [-2.0], [hood_base_z + 0.3], color='#0284c7', s=60,
        edgecolors=ACCENT, depthshade=False, zorder=5)

    # ---- WIRELESS SENSOR GLOVE (placed beside kiosk) ----
    # Glove palm (flat shape)
    gx = 5  # offset to the right
    gz = 1.5

    # Palm base
    palm_x = np.array([[gx, gx+2.5], [gx, gx+2.5]])
    palm_y = np.array([[-1, -1], [1, 1]])
    palm_z = np.array([[gz, gz], [gz, gz]])
    ax.plot_surface(palm_x, palm_y, palm_z, color=GLOVE_COL,
        edgecolors='#6b7280', linewidth=0.8, alpha=0.9)

    # Fingers (5 strips extending from palm)
    finger_starts = np.linspace(-0.8, 0.8, 5)
    for fi, fy in enumerate(finger_starts):
        flen = 1.8 if fi != 0 else 1.2  # thumb shorter
        fx_start = gx + 2.5
        # Finger strip
        finger_x = [fx_start, fx_start + flen]
        finger_y = [fy, fy + (0.1 if fi < 2 else -0.1 if fi > 2 else 0)]
        finger_z = [gz, gz + 0.1]
        ax.plot(finger_x, finger_y, finger_z, color=GLOVE_COL,
            linewidth=4, solid_capstyle='round')
        # Flex sensor on finger (orange strip)
        ax.plot(finger_x, finger_y, [gz+0.05, gz+0.15],
            color=SENSOR_COL, linewidth=2, solid_capstyle='round')

    # Wrist strap with PCB module
    wrist_x = np.array([[gx-0.5, gx+0.5], [gx-0.5, gx+0.5]])
    wrist_y = np.array([[-1.2, -1.2], [1.2, 1.2]])
    wrist_z = np.array([[gz, gz], [gz, gz]])
    ax.plot_surface(wrist_x, wrist_y, wrist_z, color=STRAP_COL,
        edgecolors='#4b5563', linewidth=0.6, alpha=0.85)

    # IMU PCB on wrist
    pcb_x = np.array([[gx-0.3, gx+0.3], [gx-0.3, gx+0.3]])
    pcb_y = np.array([[-0.4, -0.4], [0.4, 0.4]])
    pcb_z = np.array([[gz+0.05, gz+0.05], [gz+0.05, gz+0.05]])
    ax.plot_surface(pcb_x, pcb_y, pcb_z, color=PCB_COL,
        edgecolors='#047857', linewidth=0.6, alpha=0.95)

    # BLE antenna + LED
    ax.scatter([gx], [0], [gz+0.15], color=ACCENT, s=20,
        edgecolors='#0369a1', depthshade=False)
    ax.scatter([gx+0.3], [0.3], [gz+0.12], color=LED_GRN, s=12,
        edgecolors='#166534', depthshade=False)

    ax.view_init(elev=22, azim=-50)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================
#  MAIN
# ========================================================================
if __name__ == '__main__':
    out_dir = r'r:\Projects\Sign Language\DESIGN_REGISTRATION'
    os.makedirs(out_dir, exist_ok=True)

    draw_front_view(os.path.join(out_dir, 'fig1_front.png'))
    print('1/7  Front View - done')

    draw_back_view(os.path.join(out_dir, 'fig2_back.png'))
    print('2/7  Back View - done')

    draw_side_view(os.path.join(out_dir, 'fig3_left.png'), is_left=True)
    print('3/7  Left View - done')

    draw_side_view(os.path.join(out_dir, 'fig4_right.png'), is_left=False)
    print('4/7  Right View - done')

    draw_top_view(os.path.join(out_dir, 'fig5_top.png'))
    print('5/7  Top View - done')

    draw_bottom_view(os.path.join(out_dir, 'fig6_bottom.png'))
    print('6/7  Bottom View - done')

    draw_perspective_view(os.path.join(out_dir, 'fig7_perspective.png'))
    print('7/7  Perspective View - done')

    print('All 7 design views generated for KIOSK + SENSOR GLOVE system.')
