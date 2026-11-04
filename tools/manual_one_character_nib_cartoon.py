#!/usr/bin/env python3
"""
manual_one_character_nib_cartoon.py
================================================================================
MULTIPLANE ANIMATION STUDIO — Solo Hand-Drawn Cartoon for ONE Character (`NIB`)
Following Sections I through XXXVIII (`DRAW -> LOOK -> COMPARE -> ADJUST -> DRAW AGAIN`)

A pure, full-screen (`1920x1080`) traditional hand-drawn character animation of
ONE character — `NIB`, our Fountain-Nib Animator — at his backlit beechwood
animation desk inside a 7-plane multiplane studio room.

Zero UI dashboard clutter, zero static AI grid cutouts, zero duplicate sprites.
Every pose develops through:
  1. 16 Manually Authored Key Poses (`K01` .. `K16`) & 15 Breakdowns (`B01` .. `B15`)
     with explicit anatomical coordinates, spine bend, weight transfer, knee bend,
     5-fingered hand articulation, facial acting (`eye squint`, `brow arch`,
     `pupil gaze`, `tongue-out concentration`, `toothy grin`), and secondary
     follow-through (`green celluloid visor`, `5 tousled hair locks`, `bowtie wings`,
     `apron hem`, `curling animation sheets`).
  2. Inside-Out Construction (`Pass A` Line of Action & Motion Arcs -> `Pass B`
     Cranium Sphere, Ribcage & Pelvis Volumes -> `Pass C` Watertight Gouache Cel
     Fills & Rim Highlights -> `Pass D` Tapered Speedball B-5 India Ink Contours
     & Cross-Hatching).
  3. 7-Plane Multiplane Camera Choreography (`Plane 6` Night Window Sky,
     `Plane 5` Distant Rooftops & Water Tower, `Plane 4` Studio Bookshelves &
     Storyboard Corkboard, `Plane 3` Swinging Enamel Lamp & Light Cone,
     `Plane 2` Nib & Backlit Animation Desk, `Plane 1` Foreground Drafting Props,
     `Plane 0` Extreme Foreground Proscenium Frame) with true differential
     parallax (`screen_motion ∝ camera_motion / layer_depth`).
================================================================================
"""

import math
import subprocess
import shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BTS_DIR = ROOT / "bts_films"
PREVIEW_DIR = ROOT / "preview"
ARTIFACT_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

# Classical Hand-Drawn Palette (BGR)
BGR_PAPER = (232, 242, 250)         # Warm archival animation bond paper
BGR_PAPER_SHADE = (212, 226, 238)   # Shaded paper tone
BGR_INK = (16, 18, 22)              # Speedball B-5 India Ink Black
BGR_SEPIA = (42, 60, 88)            # Warm sepia ink line
BGR_NONPHOTO_BLUE = (228, 152, 58)  # Animator's Non-Photo Blue pencil
BGR_BLUE_FAINT = (232, 198, 148)    # Faint blue underdrawing in cleanup pass
BGR_ONION_PREV = (112, 128, 224)    # Warm red-sienna previous frame onion skin
BGR_ONION_NEXT = (228, 188, 112)    # Cool cyan-blue next frame onion skin
BGR_ACCENT_RED = (48, 78, 212)      # Vermilion keyframe accent & bowtie


# ==============================================================================
# I. CALLIGRAPHIC SPLINE, CROSS-HATCH & ANATOMICAL CURVE RENDERERS
# ==============================================================================

def eval_catmull_rom(ctrl_pts: list, samples_per_seg: int = 10, closed: bool = False) -> np.ndarray:
    """Evaluates a C1 Catmull-Rom spline through `ctrl_pts`."""
    pts = [np.array(p, dtype=np.float64) for p in ctrl_pts]
    n = len(pts)
    if n < 2:
        return np.array(ctrl_pts, dtype=np.float64)
    if n == 2:
        t = np.linspace(0.0, 1.0, max(2, samples_per_seg))[:, None]
        return (1.0 - t) * pts[0] + t * pts[1]

    if closed:
        padded = [pts[-1]] + pts + [pts[0], pts[1]]
        num_segs = n
    else:
        padded = [2.0 * pts[0] - pts[1]] + pts + [2.0 * pts[-1] - pts[-2]]
        num_segs = n - 1

    out = []
    for i in range(num_segs):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        steps = max(3, samples_per_seg)
        for s in range(steps if (i < num_segs - 1 or closed) else steps + 1):
            u = s / float(steps)
            u2 = u * u
            u3 = u2 * u
            pt = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * u
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
            )
            out.append(pt)
    return np.array(out, dtype=np.float64)


def draw_ink_stroke(
    canvas: np.ndarray,
    ctrl_pts: list,
    color_bgr: tuple = BGR_INK,
    base_thick: float = 2.8,
    closed: bool = False,
    fill_bgr: tuple = None,
    boil_phase: int = 0,
    seed: int = 0,
    taper_start: float = 0.45,
    taper_end: float = 0.45,
    samples_per_seg: int = 9,
):
    """
    Draws a hand-inked calligraphic spline with Speedball nib pressure variation,
    subtle 12fps paper-tooth line boil, and optional watertight gouache fill.
    """
    if len(ctrl_pts) < 2:
        return
    curve = eval_catmull_rom(ctrl_pts, samples_per_seg=samples_per_seg, closed=closed)
    n = len(curve)
    if n < 2:
        return

    b_step = boil_phase // 2
    idx = np.arange(n, dtype=np.float64)
    u = idx / float(max(1, n - 1))
    boil_amp = 0.48
    bx = boil_amp * np.sin(u * 8.5 + b_step * 1.7 + seed * 0.71)
    by = boil_amp * np.cos(u * 10.3 - b_step * 1.3 + seed * 1.13)
    pts_f = np.column_stack([curve[:, 0] + bx, curve[:, 1] + by])
    pts_i = np.round(pts_f).astype(np.int32)

    if fill_bgr is not None and len(pts_i) >= 3:
        cv2.fillPoly(canvas, [pts_i], fill_bgr, cv2.LINE_AA)

    if base_thick <= 0.0:
        return

    if base_thick <= 1.5:
        cv2.polylines(canvas, [pts_i], closed, color_bgr, max(1, int(round(base_thick))), cv2.LINE_AA)
        return

    for i in range(n - 1):
        um = (i + 0.5) / float(max(1, n - 1))
        if closed:
            prof = 0.82 + 0.24 * math.sin(2.0 * math.pi * um + seed * 0.5)
        else:
            if um < 0.25:
                prof = taper_start + (1.0 - taper_start) * math.sin((um / 0.25) * (math.pi * 0.5))
            elif um > 0.75:
                prof = taper_end + (1.0 - taper_end) * math.sin(((1.0 - um) / 0.25) * (math.pi * 0.5))
            else:
                prof = 1.0 + 0.14 * math.sin(((um - 0.25) / 0.50) * math.pi)
        w = max(1, int(round(base_thick * prof)))
        p1 = (int(pts_i[i, 0]), int(pts_i[i, 1]))
        p2 = (int(pts_i[i + 1, 0]), int(pts_i[i + 1, 1]))
        cv2.line(canvas, p1, p2, color_bgr, w, cv2.LINE_AA)
    if closed:
        cv2.line(
            canvas,
            (int(pts_i[-1, 0]), int(pts_i[-1, 1])),
            (int(pts_i[0, 0]), int(pts_i[0, 1])),
            color_bgr,
            max(1, int(round(base_thick * 0.8))),
            cv2.LINE_AA,
        )


def draw_sculpted_oval(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float = 0.0,
    color_bgr: tuple = BGR_INK,
    thick: float = 2.4,
    fill_bgr: tuple = None,
    boil_phase: int = 0,
    seed: int = 0,
    num_pts: int = 12,
):
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    ctrl = []
    for i in range(num_pts):
        th = (2.0 * math.pi * i) / num_pts
        lx, ly = rx * math.cos(th), ry * math.sin(th)
        ctrl.append((cx + lx * ca - ly * sa, cy + lx * sa + ly * ca))
    draw_ink_stroke(
        canvas, ctrl, color_bgr=color_bgr, base_thick=thick,
        closed=True, fill_bgr=fill_bgr, boil_phase=boil_phase, seed=seed, samples_per_seg=4,
    )


def solve_2bone_ik(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    l1: float,
    l2: float,
    bend_sign: float = 1.0,
    prefer_lower_joint: bool = False,
    prefer_forward_x: bool = False,
) -> tuple:
    """Analytical 2-bone IK returning `(joint_x, joint_y, end_x, end_y)`."""
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    max_r = (l1 + l2) * 0.992
    min_r = abs(l1 - l2) + 2.0
    if d > max_r and d > 1e-4:
        dx *= max_r / d
        dy *= max_r / d
        d = max_r
    elif d < min_r:
        d = min_r
    cos_a = float(np.clip((l1 * l1 + d * d - l2 * l2) / max(2.0 * l1 * d, 1e-4), -1.0, 1.0))
    ang_base = math.atan2(dy, dx)
    ang_a = math.acos(cos_a)
    j1x, j1y = sx + l1 * math.cos(ang_base + ang_a), sy + l1 * math.sin(ang_base + ang_a)
    j2x, j2y = sx + l1 * math.cos(ang_base - ang_a), sy + l1 * math.sin(ang_base - ang_a)
    if prefer_lower_joint:
        return (j1x, j1y, sx + dx, sy + dy) if j1y >= j2y else (j2x, j2y, sx + dx, sy + dy)
    if prefer_forward_x:
        return (j1x, j1y, sx + dx, sy + dy) if j1x >= j2x else (j2x, j2y, sx + dx, sy + dy)
    ang_1 = ang_base + bend_sign * ang_a
    jx = sx + l1 * math.cos(ang_1)
    jy = sy + l1 * math.sin(ang_1)
    return jx, jy, sx + dx, sy + dy


# ==============================================================================
# II. 16 MANUALLY AUTHORED KEY POSES (`K01`..`K16`) FOR `NIB`
# ==============================================================================
# Coordinates are in Nib's Studio Desk local space where:
#   - Floor beneath Nib's shoes is `y = 0`
#   - Nib's default standing foot center is `x = 0, y = 0`
#   - Desk backlit glass drawing disc center is `x = +195, y = -285`
#   - Glass India inkwell on desk corner is `x = +305, y = -265`
#   - Overhead studio lamp glow target is `x = +120, y = -580`

NIB_KEY_POSES = [
    # K01: Idle & Studying Blank Sheet (weight on back foot, hand on chin, thoughtful brow)
    {
        "code": "K01", "title": "IDLE & STUDYING BLANK SHEET", "frame": 0,
        "hip_dx": -14.0, "hip_dy": -228.0, "pelvis_tilt": -4.0,
        "spine_bend": 12.0, "chest_squash": 1.00,
        "head_pitch": 18.0, "head_yaw": 0.85, "head_squash": 1.00,
        "lh_x": 128.0, "lh_y": -292.0, "lh_pose": "peg_rest",
        "rh_x": 54.0, "rh_y": -402.0, "rh_pose": "chin_think", "pen_angle": -65.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 28.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": -6.0, "brow_r": 8.0, "eye_open": 0.90, "look_dx": 1.0, "look_dy": 1.0,
        "mouth": "think_pucker", "hair_lag": 0.0, "apron_lag": 0.0,
        "paper_flip": 0.05, "draw_prog": 0.05, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.00, "cam_pan_x": 0.0, "cam_pan_y": 0.0,
    },
    # K02: Sudden Idea / Hears the Beat! (Head snaps up, brows shoot high, mouth "OH!")
    {
        "code": "K02", "title": "IDEA STRIKES! (EARS & BROWS UP)", "frame": 18,
        "hip_dx": -6.0, "hip_dy": -238.0, "pelvis_tilt": 3.0,
        "spine_bend": -4.0, "chest_squash": 0.94,
        "head_pitch": -14.0, "head_yaw": 0.95, "head_squash": 0.91,
        "lh_x": 96.0, "lh_y": -348.0, "lh_pose": "open_gesture",
        "rh_x": 82.0, "rh_y": -465.0, "rh_pose": "idea_point", "pen_angle": -82.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 30.0, "r_foot_dy": 0.0, "r_heel_lift": 6.0,
        "brow_l": 14.0, "brow_r": 16.0, "eye_open": 1.25, "look_dx": 0.6, "look_dy": -0.8,
        "mouth": "oh_gasp", "hair_lag": -16.0, "apron_lag": -10.0,
        "paper_flip": 0.05, "draw_prog": 0.08, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.02, "cam_pan_x": 8.0, "cam_pan_y": -6.0,
    },
    # K03: Deep Anticipation Wind-Up (Coils down & back before pencil strikes paper!)
    {
        "code": "K03", "title": "ANTICIPATION COIL & WIND-UP", "frame": 36,
        "hip_dx": -28.0, "hip_dy": -196.0, "pelvis_tilt": -10.0,
        "spine_bend": 28.0, "chest_squash": 1.11,
        "head_pitch": 22.0, "head_yaw": 0.90, "head_squash": 1.08,
        "lh_x": 136.0, "lh_y": -302.0, "lh_pose": "pin_paper",
        "rh_x": -18.0, "rh_y": -448.0, "rh_pose": "grip_pen", "pen_angle": -125.0,
        "l_foot_dx": -38.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 26.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": -12.0, "brow_r": -10.0, "eye_open": 0.72, "look_dx": 1.2, "look_dy": 0.8,
        "mouth": "tongue_concentrate", "hair_lag": 18.0, "apron_lag": 14.0,
        "paper_flip": 0.08, "draw_prog": 0.12, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.06, "cam_pan_x": 18.0, "cam_pan_y": 10.0,
    },
    # K04: Rapid Non-Photo Blue Sketching — Far-Right Sweep Extreme
    {
        "code": "K04", "title": "SKETCHING SWEEP EXTREME (FAR REACH)", "frame": 54,
        "hip_dx": -18.0, "hip_dy": -208.0, "pelvis_tilt": 8.0,
        "spine_bend": 34.0, "chest_squash": 0.95,
        "head_pitch": 16.0, "head_yaw": 1.00, "head_squash": 0.97,
        "lh_x": 142.0, "lh_y": -305.0, "lh_pose": "pin_paper",
        "rh_x": 248.0, "rh_y": -294.0, "rh_pose": "grip_pen", "pen_angle": -28.0,
        "l_foot_dx": -36.0, "l_foot_dy": 0.0, "l_heel_lift": 14.0,
        "r_foot_dx": 38.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": -8.0, "brow_r": -6.0, "eye_open": 0.85, "look_dx": 1.3, "look_dy": 0.6,
        "mouth": "tongue_concentrate", "hair_lag": -22.0, "apron_lag": -18.0,
        "paper_flip": 0.12, "draw_prog": 0.30, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.10, "cam_pan_x": 32.0, "cam_pan_y": 14.0,
    },
    # K05: Rapid Sketching — Pull-Back Counter-Arc Extreme
    {
        "code": "K05", "title": "SKETCHING COUNTER-ARC PULL", "frame": 72,
        "hip_dx": -22.0, "hip_dy": -214.0, "pelvis_tilt": 2.0,
        "spine_bend": 24.0, "chest_squash": 1.04,
        "head_pitch": 12.0, "head_yaw": 0.88, "head_squash": 1.03,
        "lh_x": 138.0, "lh_y": -308.0, "lh_pose": "pin_paper",
        "rh_x": 162.0, "rh_y": -268.0, "rh_pose": "grip_pen", "pen_angle": -52.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 4.0,
        "r_foot_dx": 34.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": -4.0, "brow_r": 6.0, "eye_open": 0.92, "look_dx": 1.1, "look_dy": 0.7,
        "mouth": "smirk_focus", "hair_lag": 12.0, "apron_lag": 9.0,
        "paper_flip": 0.16, "draw_prog": 0.46, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.12, "cam_pan_x": 42.0, "cam_pan_y": 16.0,
    },
    # K06: Paper Flip Anticipation — Pinching the 3 Sheets on the Brass Acme Peg Bar
    {
        "code": "K06", "title": "PINCHING 3 SHEETS ON ACME PEG BAR", "frame": 90,
        "hip_dx": -24.0, "hip_dy": -202.0, "pelvis_tilt": 6.0,
        "spine_bend": 36.0, "chest_squash": 1.06,
        "head_pitch": 24.0, "head_yaw": 0.95, "head_squash": 1.04,
        "lh_x": 148.0, "lh_y": -268.0, "lh_pose": "pinch_flip",
        "rh_x": 218.0, "rh_y": -315.0, "rh_pose": "grip_pen", "pen_angle": -40.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 8.0,
        "r_foot_dx": 36.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": -10.0, "brow_r": 10.0, "eye_open": 0.68, "look_dx": 1.2, "look_dy": 0.9,
        "mouth": "squint_smirk", "hair_lag": -6.0, "apron_lag": -4.0,
        "paper_flip": 0.28, "draw_prog": 0.52, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.16, "cam_pan_x": 58.0, "cam_pan_y": 18.0,
    },
    # K07: Rolling the 3 Punched Animation Sheets High & Darting Pencil Underneath!
    {
        "code": "K07", "title": "3-SHEET ROLLING FLIP & IN-BETWEEN CHECK", "frame": 108,
        "hip_dx": -20.0, "hip_dy": -210.0, "pelvis_tilt": 7.0,
        "spine_bend": 30.0, "chest_squash": 0.98,
        "head_pitch": 8.0, "head_yaw": 0.96, "head_squash": 0.96,
        "lh_x": 164.0, "lh_y": -362.0, "lh_pose": "roll_high",
        "rh_x": 196.0, "rh_y": -282.0, "rh_pose": "grip_pen", "pen_angle": -22.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 10.0,
        "r_foot_dx": 36.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": 12.0, "brow_r": 14.0, "eye_open": 1.15, "look_dx": 1.2, "look_dy": 0.4,
        "mouth": "happy_grin", "hair_lag": -14.0, "apron_lag": -8.0,
        "paper_flip": 0.95, "draw_prog": 0.66, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.18, "cam_pan_x": 64.0, "cam_pan_y": 12.0,
    },
    # K08: Reaching Overhand to Dip Speedball B-5 Nib into Glass India Inkwell
    {
        "code": "K08", "title": "DIPPING SPEEDBALL NIB IN INDIA INKWELL", "frame": 126,
        "hip_dx": -16.0, "hip_dy": -212.0, "pelvis_tilt": 10.0,
        "spine_bend": 32.0, "chest_squash": 0.95,
        "head_pitch": 14.0, "head_yaw": 1.05, "head_squash": 0.98,
        "lh_x": 142.0, "lh_y": -298.0, "lh_pose": "pin_paper",
        "rh_x": 302.0, "rh_y": -282.0, "rh_pose": "dip_ink", "pen_angle": -78.0,
        "l_foot_dx": -32.0, "l_foot_dy": 0.0, "l_heel_lift": 16.0,
        "r_foot_dx": 42.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": 6.0, "brow_r": 8.0, "eye_open": 0.95, "look_dx": 1.5, "look_dy": 0.5,
        "mouth": "think_pucker", "hair_lag": -18.0, "apron_lag": -15.0,
        "paper_flip": 0.18, "draw_prog": 0.70, "cel_lift": 0.0, "ink_drop": 0.25,
        "cam_zoom": 1.15, "cam_pan_x": 68.0, "cam_pan_y": 10.0,
    },
    # K09: Precision Calligraphic Speedball Inking (Hunched Close, Tongue Out!)
    {
        "code": "K09", "title": "PRECISION CALLIGRAPHIC SPEEDBALL INKING", "frame": 146,
        "hip_dx": -26.0, "hip_dy": -192.0, "pelvis_tilt": 8.0,
        "spine_bend": 38.0, "chest_squash": 1.12,
        "head_pitch": 24.0, "head_yaw": 0.95, "head_squash": 1.07,
        "lh_x": 138.0, "lh_y": -296.0, "lh_pose": "pin_paper",
        "rh_x": 184.0, "rh_y": -288.0, "rh_pose": "grip_pen", "pen_angle": -35.0,
        "l_foot_dx": -36.0, "l_foot_dy": 0.0, "l_heel_lift": 6.0,
        "r_foot_dx": 38.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": -14.0, "brow_r": -14.0, "eye_open": 0.62, "look_dx": 1.1, "look_dy": 0.9,
        "mouth": "tongue_concentrate", "hair_lag": 10.0, "apron_lag": 8.0,
        "paper_flip": 0.10, "draw_prog": 0.86, "cel_lift": 0.0, "ink_drop": 0.85,
        "cam_zoom": 1.22, "cam_pan_x": 74.0, "cam_pan_y": 22.0,
    },
    # K10: Calligraphic Inking Flourish Overshoot! (Conductor's Sweep)
    {
        "code": "K10", "title": "INKING FLOURISH & OVERSHOOT", "frame": 166,
        "hip_dx": -8.0, "hip_dy": -236.0, "pelvis_tilt": -4.0,
        "spine_bend": 4.0, "chest_squash": 0.93,
        "head_pitch": -10.0, "head_yaw": 0.92, "head_squash": 0.94,
        "lh_x": 144.0, "lh_y": -302.0, "lh_pose": "pin_paper",
        "rh_x": 265.0, "rh_y": -435.0, "rh_pose": "flourish", "pen_angle": 25.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 34.0, "r_foot_dy": 0.0, "r_heel_lift": 8.0,
        "brow_l": 14.0, "brow_r": 14.0, "eye_open": 1.12, "look_dx": 1.2, "look_dy": -0.4,
        "mouth": "happy_grin", "hair_lag": -20.0, "apron_lag": -14.0,
        "paper_flip": 0.08, "draw_prog": 1.00, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.10, "cam_pan_x": 45.0, "cam_pan_y": 4.0,
    },
    # K11: Lifting the Finished Animation Cel Off the Brass Acme Pegs
    {
        "code": "K11", "title": "LIFTING FINISHED CEL FROM PEG BAR", "frame": 184,
        "hip_dx": 10.0, "hip_dy": -224.0, "pelvis_tilt": 4.0,
        "spine_bend": 20.0, "chest_squash": 1.00,
        "head_pitch": 10.0, "head_yaw": 0.95, "head_squash": 1.00,
        "lh_x": 145.0, "lh_y": -318.0, "lh_pose": "hold_cel",
        "rh_x": 235.0, "rh_y": -318.0, "rh_pose": "hold_cel", "pen_angle": -15.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 34.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": 8.0, "brow_r": 8.0, "eye_open": 1.00, "look_dx": 1.1, "look_dy": 0.2,
        "mouth": "smirk_focus", "hair_lag": 6.0, "apron_lag": 4.0,
        "paper_flip": 0.0, "draw_prog": 1.00, "cel_lift": 0.25, "ink_drop": 0.0,
        "cam_zoom": 1.05, "cam_pan_x": 28.0, "cam_pan_y": 0.0,
    },
    # K12: Holding the Finished Cel High Toward the Studio Lamp to Inspect!
    {
        "code": "K12", "title": "INSPECTING FINISHED CEL AGAINST STUDIO LAMP", "frame": 204,
        "hip_dx": -10.0, "hip_dy": -242.0, "pelvis_tilt": -6.0,
        "spine_bend": -10.0, "chest_squash": 0.92,
        "head_pitch": -22.0, "head_yaw": 0.92, "head_squash": 0.95,
        "lh_x": 92.0, "lh_y": -468.0, "lh_pose": "hold_cel",
        "rh_x": 202.0, "rh_y": -468.0, "rh_pose": "hold_cel", "pen_angle": 12.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 32.0, "r_foot_dy": 0.0, "r_heel_lift": 10.0,
        "brow_l": 15.0, "brow_r": 15.0, "eye_open": 1.15, "look_dx": 0.8, "look_dy": -0.9,
        "mouth": "big_grin_teeth", "hair_lag": -18.0, "apron_lag": -12.0,
        "paper_flip": 0.0, "draw_prog": 1.00, "cel_lift": 1.00, "ink_drop": 0.0,
        "cam_zoom": 1.02, "cam_pan_x": 12.0, "cam_pan_y": -14.0,
    },
    # K13: Deep Celebration Jump Anticipation Crouch
    {
        "code": "K13", "title": "CELEBRATION JUMP ANTICIPATION CROUCH", "frame": 222,
        "hip_dx": -18.0, "hip_dy": -174.0, "pelvis_tilt": -8.0,
        "spine_bend": 26.0, "chest_squash": 1.16,
        "head_pitch": 14.0, "head_yaw": 0.85, "head_squash": 1.12,
        "lh_x": -48.0, "lh_y": -245.0, "lh_pose": "open_gesture",
        "rh_x": 68.0, "rh_y": -255.0, "rh_pose": "grip_pen", "pen_angle": -55.0,
        "l_foot_dx": -38.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 30.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": 10.0, "brow_r": 10.0, "eye_open": 0.90, "look_dx": 0.5, "look_dy": -0.2,
        "mouth": "happy_grin", "hair_lag": 22.0, "apron_lag": 18.0,
        "paper_flip": 0.05, "draw_prog": 1.00, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.00, "cam_pan_x": 0.0, "cam_pan_y": 6.0,
    },
    # K14: Joyful Airborne Leap Extreme! (Both Shoes Off Floorboards!)
    {
        "code": "K14", "title": "AIRBORNE CELEBRATION LEAP EXTREME!", "frame": 240,
        "hip_dx": -8.0, "hip_dy": -302.0, "pelvis_tilt": 5.0,
        "spine_bend": -14.0, "chest_squash": 0.88,
        "head_pitch": -18.0, "head_yaw": 0.82, "head_squash": 0.90,
        "lh_x": -62.0, "lh_y": -518.0, "lh_pose": "cheer_high",
        "rh_x": 96.0, "rh_y": -532.0, "rh_pose": "cheer_high", "pen_angle": -95.0,
        "l_foot_dx": -42.0, "l_foot_dy": -58.0, "l_heel_lift": 24.0,
        "r_foot_dx": 36.0, "r_foot_dy": -64.0, "r_heel_lift": 26.0,
        "brow_l": 18.0, "brow_r": 18.0, "eye_open": 1.28, "look_dx": 0.2, "look_dy": -0.6,
        "mouth": "open_laugh", "hair_lag": 28.0, "apron_lag": 32.0,
        "paper_flip": 0.15, "draw_prog": 1.00, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 0.98, "cam_pan_x": -8.0, "cam_pan_y": -18.0,
    },
    # K15: Landing Compression & Follow-Through Overshoot
    {
        "code": "K15", "title": "LANDING SQUASH & HAIR/APRON OVERLAP", "frame": 258,
        "hip_dx": -12.0, "hip_dy": -184.0, "pelvis_tilt": -6.0,
        "spine_bend": 22.0, "chest_squash": 1.14,
        "head_pitch": 16.0, "head_yaw": 0.82, "head_squash": 1.10,
        "lh_x": -34.0, "lh_y": -295.0, "lh_pose": "open_gesture",
        "rh_x": 74.0, "rh_y": -310.0, "rh_pose": "grip_pen", "pen_angle": -35.0,
        "l_foot_dx": -38.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 32.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": 12.0, "brow_r": 12.0, "eye_open": 0.95, "look_dx": 0.3, "look_dy": 0.1,
        "mouth": "big_grin_teeth", "hair_lag": -28.0, "apron_lag": -26.0,
        "paper_flip": 0.08, "draw_prog": 1.00, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.00, "cam_pan_x": 0.0, "cam_pan_y": 4.0,
    },
    # K16: Warm Settle, Tipping Visor & Winking to Camera
    {
        "code": "K16", "title": "SETTLE, TIP VISOR & WINK TO AUDIENCE", "frame": 287,
        "hip_dx": -10.0, "hip_dy": -230.0, "pelvis_tilt": -2.0,
        "spine_bend": 6.0, "chest_squash": 1.00,
        "head_pitch": -4.0, "head_yaw": 0.72, "head_squash": 1.00,
        "lh_x": 38.0, "lh_y": -448.0, "lh_pose": "tip_visor",
        "rh_x": 118.0, "rh_y": -365.0, "rh_pose": "wave_pen", "pen_angle": -48.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 28.0, "r_foot_dy": 0.0, "r_heel_lift": 0.0,
        "brow_l": 14.0, "brow_r": 4.0, "eye_open": 1.05, "look_dx": 0.0, "look_dy": 0.0,
        "mouth": "big_grin_teeth", "hair_lag": 0.0, "apron_lag": 0.0,
        "paper_flip": 0.05, "draw_prog": 1.00, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.00, "cam_pan_x": 0.0, "cam_pan_y": 0.0,
    },
]


def interpolate_nib_pose(frame_f: float) -> dict:
    """
    Interpolates between the 16 manually authored Key Poses (`K01..K16`) with:
      - Non-linear Slow-In / Slow-Out spacing (`Rule XI`)
      - Breakdown Arc offsets (`Rule X`: hands & elbows travel on curved arcs, not straight lines;
        hips lead while head, hair locks, and apron lag behind!)
    """
    total_f = float(NIB_KEY_POSES[-1]["frame"])
    f_clamped = float(np.clip(frame_f, 0.0, total_f))

    seg_idx = 0
    for i in range(len(NIB_KEY_POSES) - 1):
        if f_clamped <= NIB_KEY_POSES[i + 1]["frame"]:
            seg_idx = i
            break
    ka = NIB_KEY_POSES[seg_idx]
    kb = NIB_KEY_POSES[seg_idx + 1]
    fa, fb = float(ka["frame"]), float(kb["frame"])
    raw_u = (f_clamped - fa) / max(1.0, fb - fa)

    # Ease-in / Ease-out S-curve spacing (clusters near keys, accelerates through breakdown!)
    u_body = 0.5 - 0.5 * math.cos(math.pi * raw_u)
    # Hips lead slightly (`u_hip` ahead of `u_body`), head & hands have breakdown overlap!
    u_hip = float(np.clip(raw_u * 1.12 if raw_u < 0.5 else 1.0 - (1.0 - raw_u) * 0.88, 0.0, 1.0))
    u_hip = 0.5 - 0.5 * math.cos(math.pi * u_hip)
    u_lag = float(np.clip(raw_u * 0.85 if raw_u < 0.5 else 1.0 - (1.0 - raw_u) * 1.15, 0.0, 1.0))
    u_lag = 0.5 - 0.5 * math.cos(math.pi * u_lag)

    arc_bulge = math.sin(math.pi * raw_u)  # Peaks at 1.0 on the Breakdown midpoint (`raw_u = 0.5`)!

    out = {}
    for k in ka:
        if k in ("code", "title", "mouth", "lh_pose", "rh_pose"):
            out[k] = ka[k] if raw_u < 0.5 else kb[k]
        elif k in ("hip_dx", "hip_dy", "pelvis_tilt"):
            out[k] = ka[k] * (1.0 - u_hip) + kb[k] * u_hip
        elif k in ("head_pitch", "head_yaw", "head_squash", "hair_lag", "apron_lag"):
            out[k] = ka[k] * (1.0 - u_lag) + kb[k] * u_lag
        else:
            out[k] = ka[k] * (1.0 - u_body) + kb[k] * u_body

    # Rule X & XIII: Add natural curved breakdown arcs to hand trajectories!
    lh_dist = math.hypot(kb["lh_x"] - ka["lh_x"], kb["lh_y"] - ka["lh_y"])
    rh_dist = math.hypot(kb["rh_x"] - ka["rh_x"], kb["rh_y"] - ka["rh_y"])
    out["lh_y"] -= 0.18 * lh_dist * arc_bulge
    out["rh_y"] -= 0.22 * rh_dist * arc_bulge
    out["rh_x"] += 0.08 * rh_dist * arc_bulge * (1.0 if kb["rh_y"] < ka["rh_y"] else -1.0)

    # Extra hair & apron secondary overlap velocity impulse at the breakdown!
    vel_y = (kb["hip_dy"] - ka["hip_dy"])
    out["hair_lag"] += 0.22 * vel_y * arc_bulge
    out["apron_lag"] += 0.18 * vel_y * arc_bulge

    if raw_u < 0.18:
        out["pose_tag"] = f"{ka['code']} (KEY)"
        out["pose_title"] = ka["title"]
    elif raw_u > 0.82:
        out["pose_tag"] = f"{kb['code']} (KEY)"
        out["pose_title"] = kb["title"]
    elif 0.40 <= raw_u <= 0.60:
        out["pose_tag"] = f"B{seg_idx + 1:02d} (BREAKDOWN)"
        out["pose_title"] = f"{ka['code']} -> {kb['code']} BREAKDOWN ARC"
    else:
        inb_num = int(raw_u * 6) + 1
        out["pose_tag"] = f"I{seg_idx + 1:02d}.{inb_num} (IN-BETWEEN)"
        out["pose_title"] = f"{ka['code']} -> {kb['code']} IN-BETWEEN"

    out["seg_idx"] = seg_idx
    out["raw_u"] = raw_u
    return out


# ==============================================================================
# III. INSIDE-OUT HAND-DRAWN CHARACTER RENDERER FOR `NIB`
# ==============================================================================

def _draw_articulated_hand(
    canvas: np.ndarray,
    wx: float,
    wy: float,
    elbow_x: float,
    elbow_y: float,
    scale: float,
    hand_pose: str,
    is_right: bool,
    skin_fill: tuple,
    ink_col: tuple,
    ink_w: float,
    boil_phase: int,
    seed: int,
    pen_angle_deg: float = -45.0,
    has_ink_pen: bool = False,
):
    """
    Draws an expressive 5-fingered cartoon hand (`Rule XXI: HANDS AND FEET`) at `(wx, wy)`
    with dorsal palm volume, opposable thumb, index, middle, and ring/pinky fingers,
    plus the Non-Photo Blue Pencil or Speedball B-5 Dip-Pen when `is_right=True`.
    """
    arm_ang = math.atan2(wy - elbow_y, wx - elbow_x)
    ca, sa = math.cos(arm_ang), math.sin(arm_ang)
    nx, ny = -sa, ca

    # Palm volume
    palm_cx = wx + ca * 14.0 * scale
    palm_cy = wy + sa * 14.0 * scale
    draw_sculpted_oval(
        canvas, palm_cx, palm_cy, 15.0 * scale, 11.5 * scale,
        angle_deg=math.degrees(arm_ang), color_bgr=ink_col, thick=ink_w,
        fill_bgr=skin_fill, boil_phase=boil_phase, seed=seed,
    )

    # Articulated fingers based on `hand_pose`
    if hand_pose in ("open_gesture", "cheer_high", "wave_pen", "flourish"):
        spreads = [-0.55, -0.20, 0.12, 0.42]
        lengths = [20.0, 24.0, 22.0, 17.0]
    elif hand_pose in ("idea_point",):
        spreads = [-0.50, -0.05, 0.35, 0.55]
        lengths = [18.0, 28.0, 13.0, 11.0]
    else:
        # Pinching / gripping pencil or paper corner
        spreads = [-0.42, -0.08, 0.18, 0.38]
        lengths = [17.0, 21.0, 16.0, 13.0]

    for fi, (sp_a, flen) in enumerate(zip(spreads, lengths)):
        f_ang = arm_ang + sp_a
        f_base_x = palm_cx + math.cos(arm_ang) * 8.0 * scale + nx * (fi - 1.5) * 4.2 * scale
        f_base_y = palm_cy + math.sin(arm_ang) * 8.0 * scale + ny * (fi - 1.5) * 4.2 * scale
        f_mid_x = f_base_x + math.cos(f_ang) * (flen * 0.55 * scale)
        f_mid_y = f_base_y + math.sin(f_ang) * (flen * 0.55 * scale)
        curl_a = f_ang + (0.35 if (fi >= 2 and hand_pose not in ("open_gesture", "cheer_high")) else 0.08)
        f_tip_x = f_mid_x + math.cos(curl_a) * (flen * 0.48 * scale)
        f_tip_y = f_mid_y + math.sin(curl_a) * (flen * 0.48 * scale)

        f_poly = [
            (f_base_x - math.sin(f_ang) * 3.8 * scale, f_base_y + math.cos(f_ang) * 3.8 * scale),
            (f_mid_x - math.sin(f_ang) * 3.4 * scale, f_mid_y + math.cos(f_ang) * 3.4 * scale),
            (f_tip_x, f_tip_y),
            (f_mid_x + math.sin(f_ang) * 3.4 * scale, f_mid_y - math.cos(f_ang) * 3.4 * scale),
            (f_base_x + math.sin(f_ang) * 3.8 * scale, f_base_y - math.cos(f_ang) * 3.8 * scale),
        ]
        draw_ink_stroke(
            canvas, f_poly, color_bgr=ink_col, base_thick=ink_w * 0.85,
            closed=True, fill_bgr=skin_fill, boil_phase=boil_phase, seed=seed + 10 + fi,
        )

    # If right hand, draw the gripped Non-Photo Blue Pencil or Speedball B-5 Dip-Pen!
    pen_tip = (palm_cx + 22.0 * scale, palm_cy + 12.0 * scale)
    if is_right:
        p_rad = math.radians(pen_angle_deg)
        pca, psa = math.cos(p_rad), math.sin(p_rad)
        pnx, pny = -psa, pca
        grip_x = palm_cx + 10.0 * scale * ca
        grip_y = palm_cy + 10.0 * scale * sa
        back_x = grip_x - pca * 58.0 * scale
        back_y = grip_y - psa * 58.0 * scale
        nib_base_x = grip_x + pca * 36.0 * scale
        nib_base_y = grip_y + psa * 36.0 * scale
        tip_x = grip_x + pca * 54.0 * scale
        tip_y = grip_y + psa * 54.0 * scale
        pen_tip = (tip_x, tip_y)

        pw = 4.6 * scale
        shaft_poly = [
            (back_x + pnx * pw, back_y + pny * pw),
            (nib_base_x + pnx * pw * 0.85, nib_base_y + pny * pw * 0.85),
            (nib_base_x - pnx * pw * 0.85, nib_base_y - pny * pw * 0.85),
            (back_x - pnx * pw, back_y - pny * pw),
        ]
        shaft_fill = ((48, 62, 168) if has_ink_pen else BGR_NONPHOTO_BLUE) if skin_fill is not None else None
        draw_ink_stroke(canvas, shaft_poly, color_bgr=ink_col, base_thick=ink_w * 0.9, closed=True, fill_bgr=shaft_fill, boil_phase=boil_phase, seed=seed + 20)

        # Nib / Tip cone
        nib_poly = [
            (nib_base_x + pnx * pw * 0.85, nib_base_y + pny * pw * 0.85),
            (tip_x, tip_y),
            (nib_base_x - pnx * pw * 0.85, nib_base_y - pny * pw * 0.85),
        ]
        nib_fill = ((72, 188, 242) if has_ink_pen else (190, 215, 238)) if skin_fill is not None else None
        draw_ink_stroke(canvas, nib_poly, color_bgr=ink_col, base_thick=ink_w * 0.9, closed=True, fill_bgr=nib_fill, boil_phase=boil_phase, seed=seed + 21)
        if has_ink_pen:
            cv2.circle(canvas, (int(tip_x), int(tip_y)), max(2, int(3.2 * scale)), BGR_INK, -1, cv2.LINE_AA)

    return pen_tip


def draw_nib_character_pose(
    canvas: np.ndarray,
    pose: dict,
    origin_x: float,
    origin_y: float,
    scale: float,
    stage_mode: str = "multiplane_composite",
    boil_phase: int = 0,
    ink_override: tuple = None,
) -> dict:
    """
    Manually draws `NIB` in the given interpolated key/breakdown/in-between `pose`
    around studio floor anchor `(origin_x, origin_y)` at scale `scale`.
    Supports:
      - `stage_mode="rough_blue"`: Inside-Out Non-Photo Blue construction volumes,
        centerlines, joint rings, and trajectory arcs (`Rule VIII`).
      - `stage_mode="cleanup_ink"`: Opaque paper fill + faint blue construction +
        crisp calligraphic Speedball B-5 India Ink contours & cross-hatching (`Rule XVIII`).
      - `stage_mode="multiplane_composite"`: Full watertight gouache cel colors +
        studio rim lighting + Speedball India Ink contours!
    """
    is_rough = (stage_mode == "rough_blue")
    is_cleanup = (stage_mode == "cleanup_ink")
    is_colored = (stage_mode in ("gouache_cel", "multiplane_composite"))

    ink_col = ink_override if ink_override is not None else (BGR_NONPHOTO_BLUE if is_rough else BGR_INK)
    ink_w = 2.0 if is_rough else 2.8 * max(0.75, min(1.3, scale))

    # Gouache Cel Palette (or opaque paper in `cleanup_ink` so near limbs cleanly occlude far limbs!)
    if is_colored and ink_override is None:
        skin_fill = (196, 222, 248)
        shirt_fill = (242, 246, 250)
        apron_fill = (62, 86, 192)
        apron_dark = (46, 66, 156)
        pants_fill = (78, 86, 102)
        pants_far = (62, 68, 82)
        shoe_fill = (52, 64, 96)
        shoe_spats = (224, 234, 244)
        visor_fill = (128, 198, 112)
        hair_fill = (54, 78, 118)
        bowtie_fill = (44, 68, 214)
    elif is_cleanup and ink_override is None:
        skin_fill = BGR_PAPER
        shirt_fill = BGR_PAPER
        apron_fill = BGR_PAPER
        apron_dark = BGR_PAPER_SHADE
        pants_fill = BGR_PAPER
        pants_far = BGR_PAPER_SHADE
        shoe_fill = BGR_PAPER_SHADE
        shoe_spats = BGR_PAPER
        visor_fill = BGR_PAPER_SHADE
        hair_fill = BGR_PAPER_SHADE
        bowtie_fill = BGR_PAPER_SHADE
    else:
        skin_fill = shirt_fill = apron_fill = apron_dark = pants_fill = pants_far = shoe_fill = shoe_spats = visor_fill = hair_fill = bowtie_fill = None

    # --- 1. CORE SKELETAL LANDMARKS FROM `pose` ---
    hip_x = origin_x + pose["hip_dx"] * scale
    hip_y = origin_y + pose["hip_dy"] * scale
    spine_rad = math.radians(pose["spine_bend"])
    csq = float(pose["chest_squash"])

    torso_len = 148.0 * scale / (csq ** 0.65)
    chest_x = hip_x + torso_len * math.sin(spine_rad)
    chest_y = hip_y - torso_len * math.cos(spine_rad)

    mid_spine_x = 0.5 * (hip_x + chest_x) + 14.0 * scale * math.sin(spine_rad * 1.2)
    mid_spine_y = 0.5 * (hip_y + chest_y)

    neck_x = chest_x + 16.0 * scale * math.sin(spine_rad * 0.8)
    neck_y = chest_y - 26.0 * scale

    head_cx = neck_x + 12.0 * scale
    head_cy = neck_y - 46.0 * scale

    lh_target = (origin_x + pose["lh_x"] * scale, origin_y + pose["lh_y"] * scale)
    rh_target = (origin_x + pose["rh_x"] * scale, origin_y + pose["rh_y"] * scale)

    # --- 2. FAR ARM (Left Arm Reaching to Acme Peg Bar / Paper Corner) ---
    sh_lx = chest_x - 16.0 * scale
    sh_ly = chest_y + 18.0 * scale
    u_arm = 92.0 * scale
    f_arm = 88.0 * scale
    el_lx, el_ly, wr_lx, wr_ly = solve_2bone_ik(sh_lx, sh_ly, lh_target[0], lh_target[1], u_arm, f_arm, prefer_lower_joint=True)

    # Rolled-up shirt sleeve + muscular forearm
    for seg_p0, seg_p1, w_a, w_b, fill_c, sd in [
        ((sh_lx, sh_ly), (el_lx, el_ly), 17.5 * scale, 14.5 * scale, shirt_fill, 101),
        ((el_lx, el_ly), (wr_lx, wr_ly), 14.0 * scale, 9.5 * scale, skin_fill, 102),
    ]:
        dx_s, dy_s = seg_p1[0] - seg_p0[0], seg_p1[1] - seg_p0[1]
        L_s = max(1e-4, math.hypot(dx_s, dy_s))
        nx_s, ny_s = -dy_s / L_s, dx_s / L_s
        mx_s, my_s = 0.5 * (seg_p0[0] + seg_p1[0]), 0.5 * (seg_p0[1] + seg_p1[1])
        wm = 0.55 * (w_a + w_b)
        poly_s = [
            (seg_p0[0] + nx_s * w_a, seg_p0[1] + ny_s * w_a),
            (mx_s + nx_s * wm, my_s + ny_s * wm),
            (seg_p1[0] + nx_s * w_b, seg_p1[1] + ny_s * w_b),
            (seg_p1[0] - nx_s * w_b, seg_p1[1] - ny_s * w_b),
            (mx_s - nx_s * wm, my_s - ny_s * wm),
            (seg_p0[0] - nx_s * w_a, seg_p0[1] - ny_s * w_a),
        ]
        draw_ink_stroke(canvas, poly_s, color_bgr=ink_col, base_thick=ink_w * 0.88, closed=True, fill_bgr=fill_c, boil_phase=boil_phase, seed=sd)
    # Rolled sleeve cuff at left elbow
    draw_sculpted_oval(canvas, el_lx, el_ly, 16.0 * scale, 7.5 * scale, angle_deg=math.degrees(math.atan2(el_ly - sh_ly, el_lx - sh_lx)) + 90, color_bgr=ink_col, thick=ink_w * 0.88, fill_bgr=shirt_fill, boil_phase=boil_phase, seed=103)

    _draw_articulated_hand(
        canvas, wr_lx, wr_ly, el_lx, el_ly, scale,
        hand_pose=pose["lh_pose"], is_right=False,
        skin_fill=skin_fill, ink_col=ink_col, ink_w=ink_w * 0.85,
        boil_phase=boil_phase, seed=105,
    )

    # --- 3. LEGS, TROUSERS & GROUNDED OXFORD SHOES (`Rule VIII: Weight & Grounding`) ---
    # Note: `prefer_forward_x=True` so knees ALWAYS bend FORWARD (`+x`, over the toes)!
    thigh_l = 128.0 * scale
    calf_l = 126.0 * scale
    for li, (is_front_leg, hip_off_x, f_dx, f_dy, heel_lift, p_fill) in enumerate([
        (False, -18.0 * scale, pose["l_foot_dx"] * scale, pose["l_foot_dy"] * scale, pose["l_heel_lift"] * scale, pants_far),
        (True, 16.0 * scale, pose["r_foot_dx"] * scale, pose["r_foot_dy"] * scale, pose["r_heel_lift"] * scale, pants_fill),
    ]):
        hx_j = hip_x + hip_off_x
        hy_j = hip_y + 10.0 * scale
        ank_x = origin_x + f_dx
        ank_y = origin_y + f_dy - 14.0 * scale - heel_lift * 0.6
        kx, ky, ax_e, ay_e = solve_2bone_ik(hx_j, hy_j, ank_x, ank_y, thigh_l, calf_l, prefer_forward_x=True)

        # Sculpted trouser leg
        w0, w1, w2 = 23.0 * scale, 18.5 * scale, 14.0 * scale
        dx1, dy1 = kx - hx_j, ky - hy_j
        L1 = max(1e-4, math.hypot(dx1, dy1))
        n1x, n1y = -dy1 / L1, dx1 / L1
        dx2, dy2 = ax_e - kx, ay_e - ky
        L2 = max(1e-4, math.hypot(dx2, dy2))
        n2x, n2y = -dy2 / L2, dx2 / L2
        leg_poly = [
            (hx_j + n1x * w0, hy_j + n1y * w0),
            (kx + 0.5 * (n1x + n2x) * w1, ky + 0.5 * (n1y + n2y) * w1),
            (ax_e + n2x * w2, ay_e + n2y * w2),
            (ax_e - n2x * w2, ay_e - n2y * w2),
            (kx - 0.5 * (n1x + n2x) * w1, ky - 0.5 * (n1y + n2y) * w1),
            (hx_j - n1x * w0, hy_j - n1y * w0),
        ]
        draw_ink_stroke(canvas, leg_poly, color_bgr=ink_col, base_thick=ink_w, closed=True, fill_bgr=p_fill, boil_phase=boil_phase, seed=110 + li * 5)
        # Trouser cuff
        draw_sculpted_oval(canvas, ax_e, ay_e, 15.5 * scale, 6.0 * scale, angle_deg=-heel_lift * 0.8, color_bgr=ink_col, thick=ink_w * 0.85, fill_bgr=p_fill, boil_phase=boil_phase, seed=112 + li * 5)

        # Two-Tone Oxford Animator Shoe (heel + arch + spats + rounded toe cap planted on floor!)
        toe_x = ank_x + 36.0 * scale
        toe_y = origin_y + f_dy - 4.0 * scale
        heel_x = ank_x - 16.0 * scale
        heel_y = origin_y + f_dy - 4.0 * scale - heel_lift
        shoe_pts = [
            (heel_x, heel_y - 14.0 * scale),
            (ank_x + 8.0 * scale, ay_e - 2.0 * scale),
            (toe_x - 4.0 * scale, toe_y - 15.0 * scale),
            (toe_x + 8.0 * scale, toe_y - 4.0 * scale),
            (toe_x, toe_y + 2.0 * scale),
            (heel_x, heel_y + 2.0 * scale),
        ]
        draw_ink_stroke(canvas, shoe_pts, color_bgr=ink_col, base_thick=ink_w, closed=True, fill_bgr=shoe_fill, boil_phase=boil_phase, seed=113 + li * 5)
        draw_sculpted_oval(canvas, ank_x + 12.0 * scale, 0.5 * (heel_y + toe_y) - 8.0 * scale, 11.5 * scale, 6.0 * scale, angle_deg=12, color_bgr=ink_col, thick=ink_w * 0.8, fill_bgr=shoe_spats, boil_phase=boil_phase, seed=114 + li * 5)

    # --- 4. TORSO, TAILORED DRESS SHIRT, DRAUGHTSMAN'S BIB APRON & POLKA-DOT BOWTIE ---
    w_chest = 42.0 * scale * (csq ** 0.5)
    w_waist = 35.0 * scale * (csq ** 0.6)
    w_hip = 38.0 * scale * (csq ** 0.4)

    torso_ctrl = [
        (chest_x - w_chest, chest_y + 6.0 * scale),
        (chest_x + w_chest * 0.95, chest_y + 6.0 * scale),
        (mid_spine_x + w_waist, mid_spine_y),
        (hip_x + w_hip, hip_y + 16.0 * scale),
        (hip_x - w_hip * 0.95, hip_y + 16.0 * scale),
        (mid_spine_x - w_waist, mid_spine_y),
    ]
    draw_ink_stroke(canvas, torso_ctrl, color_bgr=ink_col, base_thick=ink_w, closed=True, fill_bgr=shirt_fill, boil_phase=boil_phase, seed=125)

    # Draughtsman's Bib Apron (with secondary overlap `apron_lag` at bottom hem!)
    ap_lag = pose["apron_lag"] * scale
    apron_ctrl = [
        (chest_x - w_chest * 0.55, chest_y + 22.0 * scale),
        (chest_x + w_chest * 0.78, chest_y + 22.0 * scale),
        (mid_spine_x + w_waist * 0.92 + ap_lag * 0.35, mid_spine_y + 12.0 * scale),
        (hip_x + w_hip * 1.05 + ap_lag, hip_y + 52.0 * scale - abs(ap_lag) * 0.25),
        (hip_x - w_hip * 0.75 + ap_lag * 0.8, hip_y + 52.0 * scale - abs(ap_lag) * 0.25),
        (mid_spine_x - w_waist * 0.78 + ap_lag * 0.3, mid_spine_y + 12.0 * scale),
    ]
    draw_ink_stroke(canvas, apron_ctrl, color_bgr=ink_col, base_thick=ink_w, closed=True, fill_bgr=apron_fill, boil_phase=boil_phase, seed=126)

    # Apron chest pocket
    pk_x, pk_y = mid_spine_x + 8.0 * scale, chest_y + 56.0 * scale
    pkt = [
        (pk_x - 15 * scale, pk_y - 13 * scale),
        (pk_x + 15 * scale, pk_y - 13 * scale),
        (pk_x + 13 * scale, pk_y + 13 * scale),
        (pk_x - 13 * scale, pk_y + 13 * scale),
    ]
    draw_ink_stroke(canvas, pkt, color_bgr=ink_col, base_thick=ink_w * 0.8, closed=True, fill_bgr=apron_dark, boil_phase=boil_phase, seed=127)

    # Neck Cylinder, Winged Shirt Collar & Floppy Bowtie
    neck_pts = [
        (neck_x - 13 * scale, chest_y + 8 * scale),
        (neck_x - 11 * scale, neck_y - 12 * scale),
        (neck_x + 13 * scale, neck_y - 12 * scale),
        (neck_x + 15 * scale, chest_y + 8 * scale),
    ]
    draw_ink_stroke(canvas, neck_pts, color_bgr=ink_col, base_thick=ink_w * 0.9, closed=True, fill_bgr=skin_fill, boil_phase=boil_phase, seed=128)

    bt_x, bt_y = neck_x + 12.0 * scale, chest_y + 10.0 * scale
    bt_flap = pose["hair_lag"] * 0.35 * scale
    for side_b in (-1, 1):
        wing = [
            (bt_x, bt_y),
            (bt_x + side_b * 20 * scale, bt_y - 9 * scale + bt_flap),
            (bt_x + side_b * 22 * scale, bt_y + 9 * scale + bt_flap),
        ]
        draw_ink_stroke(canvas, wing, color_bgr=ink_col, base_thick=ink_w * 0.85, closed=True, fill_bgr=bowtie_fill, boil_phase=boil_phase, seed=129 + side_b)
    draw_sculpted_oval(canvas, bt_x, bt_y, 5.5 * scale, 5.5 * scale, color_bgr=ink_col, thick=ink_w * 0.8, fill_bgr=bowtie_fill, boil_phase=boil_phase, seed=131)

    # --- 5. SCULPTED 3/4 CARTOON HEAD, FACIAL ACTING, VISOR & TOUSLED HAIR (`Rule XX`) ---
    h_sq = float(pose["head_squash"])
    hrx = 54.0 * scale * (h_sq ** 0.55)
    hry = 58.0 * scale / (h_sq ** 0.55)
    hpitch = float(pose["head_pitch"])
    head_cy = neck_y - 52.0 * scale

    rad_p = math.radians(hpitch)
    cp, sp = math.cos(rad_p), math.sin(rad_p)

    def _hpt(lx_u, ly_u):
        lx = lx_u * hrx
        ly = ly_u * hry
        return (head_cx + lx * cp - ly * sp, head_cy + lx * sp + ly * cp)

    # Tousled Hair Locks at Back/Crown of Head (drawn behind visor & forehead!)
    h_lag = pose["hair_lag"] * scale
    hair_locks = [
        [_hpt(-0.88, -0.35), _hpt(-1.26, -0.68 + h_lag * 0.010), _hpt(-0.52, -0.86)],
        [_hpt(-0.55, -0.82), _hpt(-0.66, -1.36 + h_lag * 0.012), _hpt(-0.05, -0.98)],
        [_hpt(-0.08, -0.96), _hpt(0.24, -1.38 + h_lag * 0.012), _hpt(0.48, -0.88)],
    ]
    for li_h, hlock in enumerate(hair_locks):
        draw_ink_stroke(canvas, hlock, color_bgr=ink_col, base_thick=ink_w * 0.9, closed=True, fill_bgr=hair_fill, boil_phase=boil_phase, seed=145 + li_h)

    head_silhouette = [
        _hpt(-0.86, -0.55),  # Back upper cranium
        _hpt(-0.22, -0.96),  # Crown
        _hpt(0.58, -0.86),   # Forehead ridge
        _hpt(0.84, -0.32),   # Temple / eye socket
        _hpt(1.06, 0.20),    # Sculpted 3/4 cheek puff!
        _hpt(0.74, 0.80),    # Chin / jaw corner
        _hpt(0.08, 0.94),    # Under-jaw
        _hpt(-0.66, 0.64),   # Nape of neck
        _hpt(-0.94, 0.04),   # Back of skull
    ]
    draw_ink_stroke(canvas, head_silhouette, color_bgr=ink_col, base_thick=ink_w, closed=True, fill_bgr=skin_fill, boil_phase=boil_phase, seed=140)

    # Non-Photo Blue 3D Construction Cross-Hairs (`Pass B` visible in `rough_blue` and `cleanup_ink`!)
    if is_rough or is_cleanup:
        c_col = BGR_NONPHOTO_BLUE if is_rough else BGR_BLUE_FAINT
        eye_line = [_hpt(-0.90, -0.05), _hpt(0.25, -0.02), _hpt(0.94, -0.05)]
        ctr_line = [_hpt(0.35, -0.92), _hpt(0.56, -0.02), _hpt(0.48, 0.88)]
        draw_ink_stroke(canvas, eye_line, color_bgr=c_col, base_thick=1.2, boil_phase=boil_phase, seed=141)
        draw_ink_stroke(canvas, ctr_line, color_bgr=c_col, base_thick=1.2, boil_phase=boil_phase, seed=142)

    # Sculpted Ear on left side of head
    ear_x, ear_y = _hpt(-0.78, 0.10)
    draw_sculpted_oval(canvas, ear_x, ear_y, 13.0 * scale, 18.0 * scale, angle_deg=hpitch - 10, color_bgr=ink_col, thick=ink_w * 0.85, fill_bgr=skin_fill, boil_phase=boil_phase, seed=143)

    # Expressive Almond Cartoon Eyes, Eyelids, Pupils & Specular Catchlights (Drawn in clean open face space!)
    eye_open = float(pose["eye_open"])
    is_wink = (pose.get("code") == "K16" and pose.get("raw_u", 1.0) > 0.65)
    for ei, (ex_u, ey_u, rx_e, ry_e, brow_val) in enumerate([
        (0.24, -0.04, 12.0 * scale, 15.5 * scale, pose["brow_l"]),
        (0.68, -0.06, 10.5 * scale, 14.5 * scale, pose["brow_r"]),
    ]):
        ecx, ecy = _hpt(ex_u, ey_u)
        if (ei == 0 and is_wink) or eye_open < 0.25:
            wink_arc = [(ecx - rx_e, ecy + 2 * scale), (ecx, ecy - 5 * scale), (ecx + rx_e, ecy + 2 * scale)]
            draw_ink_stroke(canvas, wink_arc, color_bgr=ink_col, base_thick=ink_w, boil_phase=boil_phase, seed=150 + ei)
        else:
            cur_ry = ry_e * float(np.clip(eye_open, 0.48, 1.25))
            draw_sculpted_oval(canvas, ecx, ecy, rx_e, cur_ry, angle_deg=hpitch, color_bgr=ink_col, thick=ink_w * 0.85, fill_bgr=(252, 254, 255) if not is_rough else None, boil_phase=boil_phase, seed=150 + ei)
            # Pupil tracking + white catchlight dot
            px = ecx + pose["look_dx"] * 4.2 * scale
            py = ecy + pose["look_dy"] * 4.2 * scale
            pr = max(2, int(round(5.2 * scale)))
            cv2.circle(canvas, (int(px), int(py)), pr, ink_col, -1, cv2.LINE_AA)
            if not is_rough:
                cv2.circle(canvas, (int(px - pr * 0.35), int(py - pr * 0.35)), max(1, int(pr * 0.38)), (255, 255, 255), -1, cv2.LINE_AA)

        # Expressive Eyebrow above eye (below visor!)
        by_off = -ry_e * 1.28 - brow_val * 0.45 * scale
        brow_pts = [
            (ecx - rx_e * 0.95, ecy + by_off + brow_val * 0.20 * scale),
            (ecx, ecy + by_off - abs(brow_val) * 0.18 * scale),
            (ecx + rx_e * 0.95, ecy + by_off - brow_val * 0.20 * scale),
        ]
        draw_ink_stroke(canvas, brow_pts, color_bgr=ink_col, base_thick=ink_w * 1.05, boil_phase=boil_phase, seed=154 + ei)

    # Classic Green Celluloid Animator's Eyeshade Visor (Perched high on forehead above eyebrows!)
    strap_pts = [_hpt(-0.84, -0.52), _hpt(0.0, -0.78), _hpt(0.68, -0.62), _hpt(0.64, -0.48), _hpt(0.0, -0.62), _hpt(-0.80, -0.38)]
    draw_ink_stroke(canvas, strap_pts, color_bgr=ink_col, base_thick=ink_w * 0.85, closed=True, fill_bgr=shirt_fill, boil_phase=boil_phase, seed=148)
    visor_brim = [
        _hpt(0.18, -0.68),
        _hpt(0.72, -0.60),
        _hpt(1.32, -0.42 + h_lag * 0.005),
        _hpt(0.74, -0.38 + h_lag * 0.005),
        _hpt(0.14, -0.52),
    ]
    draw_ink_stroke(canvas, visor_brim, color_bgr=ink_col, base_thick=ink_w * 0.9, closed=True, fill_bgr=visor_fill, boil_phase=boil_phase, seed=149)

    # Sculpted Cartoon Nose projecting cleanly in 3/4 profile (`Rule IX: Silhouette`)
    nx_b, ny_b = _hpt(0.58, -0.02)
    nx_t, ny_t = _hpt(1.18, 0.10)
    nx_u, ny_u = _hpt(0.72, 0.24)
    nose_curve = [(nx_b, ny_b), (nx_t, ny_t - 4 * scale), (nx_t + 3 * scale, ny_t + 5 * scale), (nx_u, ny_u)]
    draw_ink_stroke(canvas, nose_curve, color_bgr=ink_col, base_thick=ink_w * 0.95, closed=False, fill_bgr=skin_fill, boil_phase=boil_phase, seed=158)

    # Expressive Hand-Drawn Mouth, Teeth & Concentrating Tongue (`Rule XX`)
    m_type = pose["mouth"]
    mx, my = _hpt(0.46, 0.50)
    if m_type == "tongue_concentrate":
        m_line = [(mx - 15 * scale, my - 2 * scale), (mx + 2 * scale, my + 3 * scale), (mx + 19 * scale, my - 4 * scale)]
        draw_ink_stroke(canvas, m_line, color_bgr=ink_col, base_thick=ink_w * 0.95, boil_phase=boil_phase, seed=160)
        tongue = [
            (mx - 10 * scale, my),
            (mx - 19 * scale, my + 12 * scale),
            (mx - 5 * scale, my + 14 * scale),
            (mx - 1 * scale, my + 2 * scale),
        ]
        draw_ink_stroke(canvas, tongue, color_bgr=ink_col, base_thick=ink_w * 0.8, closed=True, fill_bgr=(138, 128, 238) if is_colored else BGR_PAPER_SHADE, boil_phase=boil_phase, seed=161)
    elif m_type in ("big_grin_teeth", "open_laugh"):
        mouth_box = [
            (mx - 18 * scale, my - 6 * scale),
            (mx + 21 * scale, my - 8 * scale),
            (mx + 15 * scale, my + 16 * scale),
            (mx - 12 * scale, my + 14 * scale),
        ]
        draw_ink_stroke(canvas, mouth_box, color_bgr=ink_col, base_thick=ink_w * 0.95, closed=True, fill_bgr=(52, 58, 168) if is_colored else BGR_PAPER_SHADE, boil_phase=boil_phase, seed=162)
        teeth = [(mx - 15 * scale, my - 5 * scale), (mx + 18 * scale, my - 7 * scale), (mx + 16 * scale, my + 2 * scale), (mx - 13 * scale, my + 2 * scale)]
        draw_ink_stroke(canvas, teeth, color_bgr=ink_col, base_thick=ink_w * 0.7, closed=True, fill_bgr=(250, 252, 255) if not is_rough else None, boil_phase=boil_phase, seed=163)
    elif m_type == "oh_gasp":
        draw_sculpted_oval(canvas, mx + 5 * scale, my + 4 * scale, 10.0 * scale, 13.5 * scale, color_bgr=ink_col, thick=ink_w * 0.95, fill_bgr=(52, 58, 168) if is_colored else BGR_PAPER_SHADE, boil_phase=boil_phase, seed=164)
    else:
        smile = [(mx - 16 * scale, my - 3 * scale), (mx + 2 * scale, my + 8 * scale), (mx + 20 * scale, my - 5 * scale)]
        draw_ink_stroke(canvas, smile, color_bgr=ink_col, base_thick=ink_w * 0.95, boil_phase=boil_phase, seed=165)

    # --- 6. NEAR ARM (Right Drawing Arm) + 5-FINGERED HAND & PEN ---
    sh_rx = chest_x - 8.0 * scale
    sh_ry = chest_y + 22.0 * scale
    el_rx, el_ry, wr_rx, wr_ry = solve_2bone_ik(sh_rx, sh_ry, rh_target[0], rh_target[1], u_arm * 1.04, f_arm * 1.04, prefer_lower_joint=True)

    for seg_p0, seg_p1, w_a, w_b, fill_c, sd in [
        ((sh_rx, sh_ry), (el_rx, el_ry), 18.5 * scale, 15.5 * scale, shirt_fill, 170),
        ((el_rx, el_ry), (wr_rx, wr_ry), 15.0 * scale, 10.0 * scale, skin_fill, 171),
    ]:
        dx_s, dy_s = seg_p1[0] - seg_p0[0], seg_p1[1] - seg_p0[1]
        L_s = max(1e-4, math.hypot(dx_s, dy_s))
        nx_s, ny_s = -dy_s / L_s, dx_s / L_s
        mx_s, my_s = 0.5 * (seg_p0[0] + seg_p1[0]), 0.5 * (seg_p0[1] + seg_p1[1])
        wm = 0.55 * (w_a + w_b)
        poly_s = [
            (seg_p0[0] + nx_s * w_a, seg_p0[1] + ny_s * w_a),
            (mx_s + nx_s * wm, my_s + ny_s * wm),
            (seg_p1[0] + nx_s * w_b, seg_p1[1] + ny_s * w_b),
            (seg_p1[0] - nx_s * w_b, seg_p1[1] - ny_s * w_b),
            (mx_s - nx_s * wm, my_s - ny_s * wm),
            (seg_p0[0] - nx_s * w_a, seg_p0[1] - ny_s * w_a),
        ]
        draw_ink_stroke(canvas, poly_s, color_bgr=ink_col, base_thick=ink_w * 0.95, closed=True, fill_bgr=fill_c, boil_phase=boil_phase, seed=sd)
    draw_sculpted_oval(canvas, el_rx, el_ry, 17.0 * scale, 8.0 * scale, angle_deg=math.degrees(math.atan2(el_ry - sh_ry, el_rx - sh_rx)) + 90, color_bgr=ink_col, thick=ink_w * 0.9, fill_bgr=shirt_fill, boil_phase=boil_phase, seed=172)

    has_ink_pen = (pose.get("seg_idx", 0) >= 7)
    pen_tip = _draw_articulated_hand(
        canvas, wr_rx, wr_ry, el_rx, el_ry, scale,
        hand_pose=pose["rh_pose"], is_right=True,
        skin_fill=skin_fill, ink_col=ink_col, ink_w=ink_w * 0.9,
        boil_phase=boil_phase, seed=175,
        pen_angle_deg=pose["pen_angle"], has_ink_pen=has_ink_pen,
    )

    # --- 7. PASS A/B CONSTRUCTION SPINE, RIBCAGE & MOTION ARCS IN ROUGH/CLEANUP ---
    if (is_rough or is_cleanup) and ink_override is None:
        c_col = BGR_NONPHOTO_BLUE if is_rough else BGR_BLUE_FAINT
        # Line of Action spine
        loa = [(hip_x, hip_y + 20 * scale), (mid_spine_x, mid_spine_y), (chest_x, chest_y), (head_cx, head_cy - 45 * scale)]
        draw_ink_stroke(canvas, loa, color_bgr=c_col, base_thick=1.6, boil_phase=boil_phase, seed=190)
        draw_sculpted_oval(canvas, chest_x, chest_y + 24 * scale, w_chest * 0.9, 44 * scale, angle_deg=pose["spine_bend"], color_bgr=c_col, thick=1.3, boil_phase=boil_phase, seed=191)
        draw_sculpted_oval(canvas, hip_x, hip_y, w_hip * 0.9, 28 * scale, angle_deg=pose["pelvis_tilt"], color_bgr=c_col, thick=1.3, boil_phase=boil_phase, seed=192)

    return {
        "lh_wrist": (wr_lx, wr_ly),
        "rh_wrist": (wr_rx, wr_ry),
        "pen_tip": pen_tip,
        "head_center": (head_cx, head_cy),
    }


# ==============================================================================
# IV. 7-PLANE MULTIPLANE STUDIO WORLD, BACKLIT DESK & CAMERA COMPOSITOR
# ==============================================================================

def render_nib_multiplane_frame(
    frame_idx: int,
    total_frames: int = 288,
    out_w: int = 1920,
    out_h: int = 1080,
    stage_mode: str = "multiplane_composite",
    enable_onion_skin: bool = False,
) -> np.ndarray:
    """
    Renders a single full-screen (`1920x1080`) frame of `NIB`'s 288-frame (12.0s @ 24fps)
    solo hand-drawn animation across all 7 Multiplane Camera Layers (`Planes 6..0`).
    """
    canvas = np.full((out_h, out_w, 3), BGR_PAPER, dtype=np.uint8)
    s_ui = min(out_w / 1920.0, out_h / 1080.0)

    # Map `frame_idx in [0, total_frames-1]` to keyframe timeline `[0, 287]`
    f_timeline = (frame_idx / float(max(1, total_frames - 1))) * 287.0
    pose = interpolate_nib_pose(f_timeline)
    boil_phase = frame_idx

    is_rough = (stage_mode == "rough_blue")
    is_cleanup = (stage_mode == "cleanup_ink")
    is_colored = (stage_mode in ("gouache_cel", "multiplane_composite"))

    # Multiplane Camera State (`Rule VI: screen_motion ∝ camera_motion / layer_depth`)
    cam_zoom = float(pose["cam_zoom"])
    cam_pan_x = float(pose["cam_pan_x"]) * s_ui
    cam_pan_y = float(pose["cam_pan_y"]) * s_ui

    def _plane_tf(wx: float, wy: float, depth: float) -> tuple:
        """Applies differential multiplane camera pan & zoom for a given plane `depth`."""
        cx, cy = out_w * 0.48, out_h * 0.56
        z_eff = 1.0 + (cam_zoom - 1.0) / max(0.45, depth ** 0.75)
        px = cx + (wx - cx - cam_pan_x / max(0.42, depth)) * z_eff
        py = cy + (wy - cy - cam_pan_y / max(0.42, depth)) * z_eff
        return px, py, z_eff

    floor_base_y = out_h * 0.85
    _, floor_y, z_p2 = _plane_tf(out_w * 0.5, floor_base_y, 1.0)
    floor_y = int(round(floor_y))

    bg_line_col = BGR_NONPHOTO_BLUE if is_rough else (BGR_SEPIA if is_cleanup else BGR_INK)
    bg_line_w = (1.4 if is_rough else 2.2) * s_ui

    # ==========================================================================
    # PLANE 6 & 5 (`depth = 5.0` & `3.5`): STUDIO WALL, FLOOR & ARCHED WINDOW SKY
    # ==========================================================================
    if is_colored:
        cv2.rectangle(canvas, (0, 0), (out_w, floor_y), (216, 232, 245), -1)
        cv2.rectangle(canvas, (0, floor_y), (out_w, out_h), (154, 184, 212), -1)
    else:
        canvas[:] = BGR_PAPER

    cv2.line(canvas, (0, floor_y), (out_w, floor_y), bg_line_col, max(1, int(3 * s_ui)), cv2.LINE_AA)

    # Perspective floorboards shifting with Plane 2 depth
    for pi in range(-4, 14):
        fx_top, _, _ = _plane_tf((pi / 9.0) * out_w, floor_base_y, 1.0)
        fx_bot = out_w * 0.5 + (fx_top - out_w * 0.5) * 1.55
        p_col = (132, 162, 192) if is_colored else (218, 206, 190)
        cv2.line(canvas, (int(fx_top), floor_y), (int(fx_bot), out_h), p_col, max(1, int(2 * s_ui)), cv2.LINE_AA)

    # Arched Studio Window on Left (`Plane 5` distant water tower & night sky inside window)
    win_cx, win_cy, z_p4 = _plane_tf(out_w * 0.20, out_h * 0.34, 2.4)
    win_w, win_h = int(135 * s_ui * z_p4), int(210 * s_ui * z_p4)
    wx0, wy0 = int(win_cx - win_w), int(win_cy - win_h)
    wx1, wy1 = int(win_cx + win_w), int(win_cy + win_h)
    if is_colored:
        cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), (78, 56, 42), -1)
        # Crescent moon & distant studio water tower (`depth = 4.5` inside window)
        mx_w, my_w, _ = _plane_tf(out_w * 0.17, out_h * 0.22, 4.8)
        cv2.circle(canvas, (int(mx_w), int(my_w)), int(22 * s_ui), (188, 236, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(mx_w + 8 * s_ui), int(my_w - 4 * s_ui)), int(19 * s_ui), (78, 56, 42), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), bg_line_col, max(1, int(3 * s_ui)), cv2.LINE_AA)
    cv2.line(canvas, (int(win_cx), wy0), (int(win_cx), wy1), bg_line_col, max(1, int(2 * s_ui)), cv2.LINE_AA)
    cv2.line(canvas, (wx0, int(win_cy)), (wx1, int(win_cy)), bg_line_col, max(1, int(2 * s_ui)), cv2.LINE_AA)

    # ==========================================================================
    # PLANE 4 (`depth = 2.2`): LIVE 4:3 PENCIL-TEST PROJECTION SCREEN & CORKBOARD
    # ==========================================================================
    from manual_hand_drawn_animation_studio import render_manual_cartoon_frame
    cb_cx, cb_cy, z_cb = _plane_tf(out_w * 0.78, out_h * 0.34, 2.2)
    cb_w = max(80, int(round(220 * s_ui * z_cb)))
    cb_h = max(60, int(round(cb_w * 0.75)))  # Exact 4:3!
    x0_pt = int(round(cb_cx - cb_w))
    y0_pt = int(round(cb_cy - cb_h))
    w_pt, h_pt = cb_w * 2, cb_h * 2
    x1_pt, y1_pt = x0_pt + w_pt, y0_pt + h_pt

    if 0 <= x0_pt and x1_pt <= out_w and 0 <= y0_pt and y1_pt <= out_h:
        fox_test_img = render_manual_cartoon_frame(
            "campfire", frame_idx / 24.0, w_pt, h_pt,
            stage_mode=stage_mode, boil_phase=boil_phase,
            enable_onion_skin=(enable_onion_skin or is_rough),
        )
        canvas[y0_pt:y1_pt, x0_pt:x1_pt] = fox_test_img
        cv2.rectangle(canvas, (x0_pt, y0_pt), (x1_pt, y1_pt), bg_line_col, max(2, int(4 * s_ui)), cv2.LINE_AA)
        cv2.rectangle(canvas, (x0_pt, y0_pt), (x1_pt, y0_pt + max(14, int(26 * s_ui))), bg_line_col, -1)
        cv2.putText(
            canvas, "4:3 STUDIO PENCIL-TEST MONITOR (FOX LEAP ANIMATION)",
            (x0_pt + int(8 * s_ui), y0_pt + max(11, int(19 * s_ui))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s_ui, (245, 248, 255), max(1, int(round(s_ui))), cv2.LINE_AA,
        )

    # ==========================================================================
    # PLANE 3 (`depth = 1.35`): SWINGING OVERHEAD ENAMEL LAMP & LIGHT CONE
    # ==========================================================================
    lamp_cx, lamp_cy, z_l = _plane_tf(out_w * 0.52, out_h * 0.14, 1.35)
    lamp_sway = 10.0 * s_ui * math.sin(frame_idx * 0.08)
    lx_s = lamp_cx + lamp_sway
    if is_colored:
        cone_ov = canvas.copy()
        cone_poly = np.array([
            [int(lx_s - 45 * s_ui), int(lamp_cy + 18 * s_ui)],
            [int(lx_s + 45 * s_ui), int(lamp_cy + 18 * s_ui)],
            [int(lx_s + 420 * s_ui), int(floor_y)],
            [int(lx_s - 340 * s_ui), int(floor_y)],
        ], dtype=np.int32)
        cv2.fillPoly(cone_ov, [cone_poly], (218, 246, 255), cv2.LINE_AA)
        cv2.addWeighted(cone_ov, 0.22, canvas, 0.78, 0, canvas)

    cv2.line(canvas, (int(lamp_cx), 0), (int(lx_s), int(lamp_cy - 12 * s_ui)), bg_line_col, max(1, int(3 * s_ui)), cv2.LINE_AA)
    shade_pts = [
        (lx_s - 62 * s_ui * z_l, lamp_cy + 20 * s_ui * z_l),
        (lx_s - 24 * s_ui * z_l, lamp_cy - 14 * s_ui * z_l),
        (lx_s + 24 * s_ui * z_l, lamp_cy - 14 * s_ui * z_l),
        (lx_s + 62 * s_ui * z_l, lamp_cy + 20 * s_ui * z_l),
    ]
    draw_ink_stroke(
        canvas, shade_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1,
        closed=True, fill_bgr=(92, 164, 114) if is_colored else (BGR_PAPER if is_cleanup else None),
        boil_phase=boil_phase, seed=55,
    )

    # ==========================================================================
    # PLANE 2 (`depth = 1.00`): NIB & SLOPED BEECHWOOD ANIMATION DESK
    # ==========================================================================
    nib_ox, nib_oy, z_nib = _plane_tf(out_w * 0.41, floor_base_y, 1.00)
    nib_scale = 1.48 * s_ui * z_nib

    # 1. Draw Sloped Beechwood Animation Desk Base & Backlit Glass Disc (`x = origin_x + 195 * scale`)
    desk_cx = nib_ox + 215.0 * nib_scale
    desk_fy = nib_oy
    wood_light = (152, 194, 226) if is_colored else (BGR_PAPER if is_cleanup else None)
    wood_dark = (104, 142, 178) if is_colored else (BGR_PAPER_SHADE if is_cleanup else None)
    brass_fill = (64, 188, 244) if is_colored else (BGR_PAPER_SHADE if is_cleanup else None)

    # Desk legs
    for leg_off in (-112.0, 96.0):
        lx_d = desk_cx + leg_off * nib_scale
        d_leg = [
            (lx_d - 16 * nib_scale, desk_fy - 235 * nib_scale),
            (lx_d + 16 * nib_scale, desk_fy - 235 * nib_scale),
            (lx_d + 22 * nib_scale, desk_fy),
            (lx_d - 22 * nib_scale, desk_fy),
        ]
        draw_ink_stroke(canvas, d_leg, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=60 + int(leg_off))

    # Sloped Drafting Table Top
    board_pts = [
        (desk_cx - 148 * nib_scale, desk_fy - 225 * nib_scale),
        (desk_cx - 62 * nib_scale, desk_fy - 342 * nib_scale),
        (desk_cx + 146 * nib_scale, desk_fy - 342 * nib_scale),
        (desk_cx + 98 * nib_scale, desk_fy - 225 * nib_scale),
    ]
    draw_ink_stroke(canvas, board_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.3, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=62)

    # Illuminated Glass Animation Disc
    disc_x = nib_ox + 195.0 * nib_scale
    disc_y = nib_oy - 285.0 * nib_scale
    draw_sculpted_oval(
        canvas, disc_x, disc_y, 82.0 * nib_scale, 42.0 * nib_scale,
        angle_deg=-12, color_bgr=bg_line_col, thick=bg_line_w * 1.1,
        fill_bgr=(206, 248, 255) if is_colored else (BGR_PAPER if is_cleanup else None),
        boil_phase=boil_phase, seed=64,
    )

    # 3-Sheet Rolling Animation Paper Stack on Brass Acme Peg Bar!
    p_flip = float(pose["paper_flip"])
    d_prog = float(pose["draw_prog"])
    c_lift = float(pose["cel_lift"])

    if c_lift < 0.05:
        for sh_i in range(3):
            curl_h = (sh_i * 10.0 + p_flip * 52.0 * (sh_i / 2.0)) * nib_scale
            sh_pts = [
                (disc_x - 64 * nib_scale, disc_y - 24 * nib_scale),
                (disc_x + 54 * nib_scale, disc_y - 30 * nib_scale),
                (disc_x + 68 * nib_scale + curl_h * 0.25, disc_y + 18 * nib_scale - curl_h),
                (disc_x - 52 * nib_scale, disc_y + 26 * nib_scale - curl_h * 0.45),
            ]
            draw_ink_stroke(
                canvas, sh_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.9,
                closed=True, fill_bgr=BGR_PAPER if not is_rough else None,
                boil_phase=boil_phase, seed=66 + sh_i,
            )
            # Draw the animated Fox leap curve on the active sheet!
            if sh_i == 2 or p_flip > 0.3:
                fox_st = [
                    (disc_x - 38 * nib_scale, disc_y + 8 * nib_scale - curl_h * 0.3),
                    (disc_x + (sh_i - 1) * 12 * nib_scale, disc_y - (14 + sh_i * 6) * nib_scale - curl_h * 0.5),
                    (disc_x + 36 * nib_scale, disc_y + 4 * nib_scale - curl_h * 0.6),
                ]
                st_col = BGR_INK if (d_prog > 0.72 and sh_i == 2) else BGR_NONPHOTO_BLUE
                draw_ink_stroke(canvas, fox_st, color_bgr=st_col, base_thick=max(1.5, 2.2 * s_ui), boil_phase=boil_phase, seed=70 + sh_i)

    # Brass Acme Peg Bar at top of disc
    peg_pts = [
        (disc_x - 58 * nib_scale, disc_y - 30 * nib_scale),
        (disc_x + 50 * nib_scale, disc_y - 36 * nib_scale),
        (disc_x + 52 * nib_scale, disc_y - 25 * nib_scale),
        (disc_x - 56 * nib_scale, disc_y - 19 * nib_scale),
    ]
    draw_ink_stroke(canvas, peg_pts, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=brass_fill, boil_phase=boil_phase, seed=74)

    # Glass Speedball India Inkwell on right corner of desk (`x = origin_x + 305 * scale, y = origin_y - 265 * scale`)
    ink_x = nib_ox + 305.0 * nib_scale
    ink_y = nib_oy - 265.0 * nib_scale
    draw_sculpted_oval(canvas, ink_x, ink_y, 18.0 * nib_scale, 15.0 * nib_scale, color_bgr=bg_line_col, thick=bg_line_w, fill_bgr=(36, 40, 48) if is_colored else None, boil_phase=boil_phase, seed=76)
    if pose["ink_drop"] > 0.05:
        u_dr = float(pose["ink_drop"])
        dr_x = ink_x - 24.0 * nib_scale * u_dr
        dr_y = ink_y - 42.0 * nib_scale * math.sin(math.pi * u_dr)
        cv2.circle(canvas, (int(dr_x), int(dr_y)), max(2, int(4.5 * nib_scale)), BGR_INK, -1, cv2.LINE_AA)

    # 2. Rule XIII: Draw Hand Trajectory Motion Arc & Spacing Chart Ticks in Rough/Cleanup!
    if is_rough or is_cleanup or enable_onion_skin:
        seg_i = pose["seg_idx"]
        ka = NIB_KEY_POSES[seg_i]
        kb = NIB_KEY_POSES[min(len(NIB_KEY_POSES) - 1, seg_i + 1)]
        arc_pts = []
        for k_s in range(13):
            u_s = k_s / 12.0
            u_e = 0.5 - 0.5 * math.cos(math.pi * u_s)
            bulge = math.sin(math.pi * u_s)
            dist = math.hypot(kb["rh_x"] - ka["rh_x"], kb["rh_y"] - ka["rh_y"])
            rx_a = nib_ox + (ka["rh_x"] * (1.0 - u_e) + kb["rh_x"] * u_e) * nib_scale
            ry_a = nib_oy + (ka["rh_y"] * (1.0 - u_e) + kb["rh_y"] * u_e - 0.22 * dist * bulge) * nib_scale
            arc_pts.append((rx_a, ry_a))
        for k_s in range(len(arc_pts) - 1):
            if k_s % 2 == 0:
                cv2.line(canvas, (int(arc_pts[k_s][0]), int(arc_pts[k_s][1])), (int(arc_pts[k_s + 1][0]), int(arc_pts[k_s + 1][1])), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
            # Spacing tick mark perpendicular to arc
            cv2.circle(canvas, (int(arc_pts[k_s][0]), int(arc_pts[k_s][1])), max(2, int(2.5 * s_ui)), BGR_ACCENT_RED if k_s in (0, 6, 12) else BGR_NONPHOTO_BLUE, -1, cv2.LINE_AA)

    # 3. Rule XII: FLIP THE DRAWINGS (`previous -> current -> next` Onion-Skin Ghosts!)
    if enable_onion_skin or is_rough:
        pose_prev = interpolate_nib_pose(max(0.0, f_timeline - 4.5))
        pose_next = interpolate_nib_pose(min(287.0, f_timeline + 4.5))
        ov_p = canvas.copy()
        draw_nib_character_pose(ov_p, pose_prev, nib_ox, nib_oy, nib_scale, stage_mode="rough_blue", boil_phase=boil_phase - 1, ink_override=BGR_ONION_PREV)
        cv2.addWeighted(ov_p, 0.30, canvas, 0.70, 0, canvas)

        ov_n = canvas.copy()
        draw_nib_character_pose(ov_n, pose_next, nib_ox, nib_oy, nib_scale, stage_mode="rough_blue", boil_phase=boil_phase + 1, ink_override=BGR_ONION_NEXT)
        cv2.addWeighted(ov_n, 0.25, canvas, 0.75, 0, canvas)

    # 4. Draw Primary Character `NIB`!
    info = draw_nib_character_pose(
        canvas, pose, nib_ox, nib_oy, nib_scale,
        stage_mode=stage_mode, boil_phase=boil_phase,
    )

    # 5. If `cel_lift > 0.05` (`K11..K12`), draw the Finished Transparent Animation Cel held out in front of Nib!
    if c_lift > 0.05:
        lh_w = info["lh_wrist"]
        rh_w = info["rh_wrist"]
        cel_cx = max(lh_w[0], rh_w[0]) + 46.0 * nib_scale
        cel_cy = 0.5 * (lh_w[1] + rh_w[1]) + 18.0 * nib_scale
        cw_l, ch_l = 58.0 * nib_scale, 43.0 * nib_scale
        cel_poly = [
            (cel_cx - cw_l, cel_cy - ch_l),
            (cel_cx + cw_l, cel_cy - ch_l - 8 * nib_scale),
            (cel_cx + cw_l + 6 * nib_scale, cel_cy + ch_l),
            (cel_cx - cw_l + 6 * nib_scale, cel_cy + ch_l + 8 * nib_scale),
        ]
        if is_colored:
            cel_ov = canvas.copy()
            pts_c = np.array(cel_poly, dtype=np.int32)
            cv2.fillPoly(cel_ov, [pts_c], (238, 250, 255), cv2.LINE_AA)
            cv2.addWeighted(cel_ov, 0.65, canvas, 0.35, 0, canvas)
        draw_ink_stroke(canvas, cel_poly, color_bgr=BGR_INK, base_thick=bg_line_w * 1.1, closed=True, boil_phase=boil_phase, seed=85)
        # Inked Fox drawing on the lifted cel!
        fox_cel = [
            (cel_cx - cw_l * 0.65, cel_cy + ch_l * 0.35),
            (cel_cx - cw_l * 0.15, cel_cy - ch_l * 0.45),
            (cel_cx + cw_l * 0.55, cel_cy - ch_l * 0.15),
        ]
        draw_ink_stroke(canvas, fox_cel, color_bgr=BGR_INK, base_thick=max(2.0, 2.8 * s_ui), boil_phase=boil_phase, seed=86)

    # ==========================================================================
    # PLANE 1 (`depth = 0.62`): FOREGROUND GOUACHE JARS & BRUSHES IN PARALLAX
    # ==========================================================================
    fg_lx, fg_ly, z_fg = _plane_tf(out_w * 0.09, out_h * 0.94, 0.62)
    fg_rx, fg_ry, _ = _plane_tf(out_w * 0.91, out_h * 0.94, 0.62)
    for fi_p, (jx, jy, j_col) in enumerate([
        (fg_lx, fg_ly, (52, 78, 198)),
        (fg_lx + 58 * s_ui * z_fg, fg_ly + 12 * s_ui * z_fg, (198, 138, 52)),
        (fg_rx, fg_ry, (68, 168, 92)),
    ]):
        draw_sculpted_oval(
            canvas, jx, jy, 26 * s_ui * z_fg, 34 * s_ui * z_fg,
            color_bgr=bg_line_col, thick=bg_line_w * 1.2,
            fill_bgr=j_col if is_colored else (BGR_PAPER_SHADE if is_cleanup else None),
            boil_phase=boil_phase, seed=90 + fi_p,
        )

    # ==========================================================================
    # PLANE 0: CLEAN ANIMATOR'S EXPOSURE SHEET SLATE BAR AT BOTTOM MARGIN
    # ==========================================================================
    bar_h = max(28, int(44 * s_ui))
    cv2.rectangle(canvas, (0, out_h - bar_h), (out_w, out_h), (28, 32, 38), -1)
    cv2.line(canvas, (0, out_h - bar_h), (out_w, out_h - bar_h), BGR_ACCENT_RED, max(1, int(2 * s_ui)))
    pass_name = {
        "rough_blue": "PASS 1: NON-PHOTO BLUE INSIDE-OUT CONSTRUCTION & ONION-SKIN FLIP",
        "cleanup_ink": "PASS 2: SPEEDBALL B-5 CALLIGRAPHIC INDIA INK CLEANUP",
        "multiplane_composite": "PASS 3: WATERTIGHT GOUACHE CEL & 7-PLANE MULTIPLANE CAMERA",
    }.get(stage_mode, stage_mode.upper())
    slate = f"CHARACTER: NIB (LEAD ANIMATOR) | {pose['pose_tag']}: {pose['pose_title']} | {pass_name} | FRAME {frame_idx + 1:03d}/{total_frames:03d} (24 FPS ON ONES)"
    cv2.putText(
        canvas, slate,
        (int(18 * s_ui), int(out_h - bar_h * 0.32)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52 * s_ui, (242, 246, 252), max(1, int(round(1.4 * s_ui))), cv2.LINE_AA,
    )
    return canvas


def build_nib_contact_sheets_and_film():
    """
    1. Renders the 16-Key-Pose Contact Sheet (`nib_16_key_poses_sheet.png`) across `K01..K16`
       so we can inspect every key pose (`Rule VII: KEY DRAWINGS FIRST`).
    2. Renders the 4-Pass Progressive Flipbook Sheet (`nib_4pass_progression_sheet.png`)
       comparing `Rough Blue`, `Onion-Skin Flip`, `Speedball Cleanup Ink`, and
       `7-Plane Gouache Multiplane Composite`.
    3. Renders the full 24fps Hand-Drawn Character Film (`nib_solo_hand_drawn_cartoon.mp4`
       & `.gif`, `864` total frames = `288` Rough Blue & Onion Flip + `288` Speedball Ink
       Cleanup + `288` Full 7-Plane Gouache Multiplane Composite @ `24 fps` = `36.0s`!).
    """
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    BTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 16 Key Poses Contact Sheet (4x4 grid of `K01..K16` in Cleanup Ink + Gouache)
    print("[Nib Solo Cartoon] Rendering 16 Key Poses Contact Sheet (K01..K16)...")
    key_rows = []
    for r in range(4):
        row_cells = []
        for c in range(4):
            k_idx = r * 4 + c
            kp = NIB_KEY_POSES[k_idx]
            f_i = int(kp["frame"])
            mode = "rough_blue" if r == 0 else ("cleanup_ink" if r == 1 else "multiplane_composite")
            cell = render_nib_multiplane_frame(f_i, 288, out_w=480, out_h=270, stage_mode=mode, enable_onion_skin=(r == 0))
            cv2.rectangle(cell, (0, 0), (479, 269), BGR_INK, 2)
            row_cells.append(cell)
        key_rows.append(np.hstack(row_cells))
    keys_grid = np.vstack(key_rows)
    keys_path = PREVIEW_DIR / "nib_16_key_poses_sheet.png"
    cv2.imwrite(str(keys_path), keys_grid)
    shutil.copy2(keys_path, ARTIFACT_DIR / "nib_16_key_poses_sheet.png")

    # 2. 3-Pass Breakdown & In-Between Flipbook Comparison (`K03 -> B03 -> K04 -> K06 -> K09 -> K14`)
    print("[Nib Solo Cartoon] Rendering 3-Pass Breakdown & In-Between Flipbook Sheet...")
    sample_frames = [18, 36, 45, 54, 99, 126, 146, 204, 231, 240, 258, 285]
    pass_rows = []
    for mode, onion in [("rough_blue", True), ("cleanup_ink", False), ("multiplane_composite", False)]:
        cells = []
        for sf in sample_frames[:6]:
            c_im = render_nib_multiplane_frame(sf, 288, out_w=360, out_h=202, stage_mode=mode, enable_onion_skin=onion)
            cv2.rectangle(c_im, (0, 0), (359, 201), BGR_INK, 2)
            cells.append(c_im)
        pass_rows.append(np.hstack(cells))
    pass_grid = np.vstack(pass_rows)
    pass_path = PREVIEW_DIR / "nib_3pass_flipbook_comparison.png"
    cv2.imwrite(str(pass_path), pass_grid)
    shutil.copy2(pass_path, ARTIFACT_DIR / "nib_3pass_flipbook_comparison.png")

    print(f"[Nib Solo Cartoon] Saved {keys_path.name} and {pass_path.name}")

    # 3. Render 864-Frame (36.0s @ 24fps) Full-Screen Hand-Drawn Solo Film for NIB!
    #    - Act I   (Frames 001..288): Pass 1 Non-Photo Blue Construction & Onion-Skin Flip
    #    - Act II  (Frames 289..576): Pass 2 Speedball B-5 Calligraphic India Ink Cleanup
    #    - Act III (Frames 577..864): Pass 3 Watertight Gouache Cel & 7-Plane Multiplane Camera
    fps = 24
    frames_per_pass = 288
    total_film_frames = frames_per_pass * 3  # 864 unique hand-drawn frames (36.0s @ 24fps)
    out_w, out_h = 1280, 720

    mp4_path = BTS_DIR / "nib_solo_hand_drawn_cartoon.mp4"
    gif_path = BTS_DIR / "nib_solo_hand_drawn_cartoon.gif"
    poster_path = BTS_DIR / "nib_solo_hand_drawn_cartoon_poster.png"
    raw_mp4 = BTS_DIR / "_tmp_nib_solo_raw.mp4"
    wav_path = BTS_DIR / "_tmp_nib_solo.wav"

    # Synthesize synchronized studio Foley & warm classical piano/woodwind score (36.0s)
    import wave
    sr = 22050
    dur = total_film_frames / float(fps)
    n_samp = int(sr * dur)
    t_a = np.linspace(0.0, dur, n_samp, endpoint=False)
    melody_freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 392.00, 329.63]
    note_idx = (t_a * 2.5).astype(np.int32) % len(melody_freqs)
    f_mel = np.array([melody_freqs[i] for i in note_idx])
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 2.5 * t_a)
    # Pencil scratch & paper flip foley impulses
    foley = 0.04 * np.sin(2.0 * np.pi * 180.0 * t_a) * (np.sin(2.0 * np.pi * 4.0 * t_a) > 0.6)
    sig = 0.16 * env * (np.sin(2.0 * np.pi * f_mel * t_a) + 0.4 * np.sin(4.0 * np.pi * f_mel * t_a)) + foley
    pcm = np.clip(sig * 32767.0, -32767, 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    print(f"[Nib Solo Cartoon] Rendering {total_film_frames} unique hand-drawn frames ({dur:.1f}s @ {fps}fps)...")
    vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (out_w, out_h))
    gif_frames_rgb = []

    for g_idx in range(total_film_frames):
        pass_i = g_idx // frames_per_pass
        local_f = g_idx % frames_per_pass
        if pass_i == 0:
            mode = "rough_blue"
            onion = True
        elif pass_i == 1:
            mode = "cleanup_ink"
            onion = (local_f % 24 < 6)
        else:
            mode = "multiplane_composite"
            onion = False

        frame_bgr = render_nib_multiplane_frame(
            local_f, frames_per_pass, out_w=out_w, out_h=out_h,
            stage_mode=mode, enable_onion_skin=onion,
        )
        vw.write(frame_bgr)

        if g_idx == frames_per_pass * 2 + 204:  # K12 Inspecting Cel in Full Gouache Multiplane
            cv2.imwrite(str(poster_path), frame_bgr)
            shutil.copy2(poster_path, ARTIFACT_DIR / "nib_solo_hand_drawn_cartoon_poster.png")

        # Sample every 6th frame for a smooth, crisp animated GIF preview across all 3 passes
        if g_idx % 6 == 0:
            small_bgr = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
            gif_frames_rgb.append(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))

    vw.release()

    # Mux H.264 MP4 + AAC Audio
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(raw_mp4), "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        str(mp4_path),
    ], check=True)
    raw_mp4.unlink(missing_ok=True)
    wav_path.unlink(missing_ok=True)

    # Save animated GIF preview
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in gif_frames_rgb]
    pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    shutil.copy2(gif_path, ARTIFACT_DIR / "nib_solo_hand_drawn_cartoon.gif")
    shutil.copy2(mp4_path, BTS_DIR / "cartoon-01-nib-moviola.mp4")
    shutil.copy2(gif_path, BTS_DIR / "cartoon-01-nib-moviola.gif")
    shutil.copy2(gif_path, ARTIFACT_DIR / "cartoon-01-nib-moviola.gif")
    shutil.copy2(poster_path, BTS_DIR / "cartoon-01-nib-moviola_poster.png")
    shutil.copy2(poster_path, ARTIFACT_DIR / "cartoon-01-nib-moviola_poster.png")
    print(f"[Nib Solo Cartoon] Saved {mp4_path.name} ({mp4_path.stat().st_size / (1024*1024):.2f} MB) and {gif_path.name}!")


if __name__ == "__main__":
    build_nib_contact_sheets_and_film()
