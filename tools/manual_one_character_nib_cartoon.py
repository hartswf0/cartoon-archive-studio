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


def eval_sharp_polygon(ctrl_pts: list, samples_per_seg: int = 8, closed: bool = True) -> np.ndarray:
    """Subdivides polygon edges linearly while keeping every corner vertex sharp."""
    pts = [np.array(p, dtype=np.float64) for p in ctrl_pts]
    n = len(pts)
    if n < 2:
        return np.array(ctrl_pts, dtype=np.float64)
    num_segs = n if closed else (n - 1)
    steps = max(2, samples_per_seg)
    out = []
    for i in range(num_segs):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        end_s = steps if (i < num_segs - 1 or closed) else (steps + 1)
        for s in range(end_s):
            u = s / float(steps)
            out.append((1.0 - u) * p0 + u * p1)
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
    sharp_corners: bool = None,
):
    """
    Draws a hand-inked calligraphic spline or sharp-cornered architectural polygon
    with Speedball nib pressure variation, subtle 12fps paper-tooth line boil,
    and optional watertight gouache fill.
    """
    if len(ctrl_pts) < 2:
        return
    use_sharp = sharp_corners if sharp_corners is not None else (closed and len(ctrl_pts) <= 6)
    if use_sharp:
        curve = eval_sharp_polygon(ctrl_pts, samples_per_seg=samples_per_seg, closed=closed)
    else:
        curve = eval_catmull_rom(ctrl_pts, samples_per_seg=samples_per_seg, closed=closed)
    n = len(curve)
    if n < 2:
        return

    b_step = boil_phase // 2
    idx = np.arange(n, dtype=np.float64)
    u = idx / float(max(1, n - 1))
    boil_amp = 0.36 if use_sharp else 0.48
    # Pin vertices when `use_sharp=True` so 3D perspective corners remain razor-sharp!
    if use_sharp:
        seg_u = (idx % max(2, samples_per_seg)) / float(max(2, samples_per_seg))
        pin_env = np.sin(np.pi * seg_u)
    else:
        pin_env = 1.0
    bx = boil_amp * pin_env * np.sin(u * 8.5 + b_step * 1.7 + seed * 0.71)
    by = boil_amp * pin_env * np.cos(u * 10.3 - b_step * 1.3 + seed * 1.13)
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
    # K10: Blowing Dry Fresh India Ink on the Desk (`nib_inking_keys.jpg` Panel 6: Puffed Cheeks & Air Cloud!)
    {
        "code": "K10", "title": "BLOWING DRY FRESH INDIA INK ON DESK", "frame": 166,
        "hip_dx": -14.0, "hip_dy": -224.0, "pelvis_tilt": 4.0,
        "spine_bend": 18.0, "chest_squash": 1.02,
        "head_pitch": 12.0, "head_yaw": 0.94, "head_squash": 1.02,
        "lh_x": 144.0, "lh_y": -302.0, "lh_pose": "pin_paper",
        "rh_x": 245.0, "rh_y": -395.0, "rh_pose": "flourish", "pen_angle": 18.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 34.0, "r_foot_dy": 0.0, "r_heel_lift": 6.0,
        "brow_l": 10.0, "brow_r": 10.0, "eye_open": 1.05, "look_dx": 1.1, "look_dy": 0.4,
        "mouth": "blow_puff", "hair_lag": -14.0, "apron_lag": -10.0,
        "paper_flip": 0.14, "draw_prog": 1.00, "cel_lift": 0.0, "ink_drop": 0.0,
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
    # K12: Holding the Finished Cel High Toward the Studio Lamp to Inspect (`nib_inking_keys.jpg` Panel 5)!
    {
        "code": "K12", "title": "INSPECTING FINISHED CEL AGAINST STUDIO LAMP", "frame": 204,
        "hip_dx": -10.0, "hip_dy": -242.0, "pelvis_tilt": -6.0,
        "spine_bend": -10.0, "chest_squash": 0.92,
        "head_pitch": -16.0, "head_yaw": 0.92, "head_squash": 0.95,
        "lh_x": 92.0, "lh_y": -458.0, "lh_pose": "hold_cel",
        "rh_x": 202.0, "rh_y": -458.0, "rh_pose": "hold_cel", "pen_angle": 12.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 32.0, "r_foot_dy": 0.0, "r_heel_lift": 10.0,
        "brow_l": 15.0, "brow_r": 15.0, "eye_open": 1.12, "look_dx": 0.8, "look_dy": -0.6,
        "mouth": "confident_smirk", "hair_lag": -18.0, "apron_lag": -12.0,
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
    has_ink_pen: bool = True,
):
    """
    Draws an expressive cartoon hand (`Rule XXI`) at `(wx, wy)` matching `nib_inking_keys.jpg`
    and `agent_nib_keys.jpg`, plus Nib's iconic **Fountain-Nib Dip Pen** (with diamond nib head,
    central slit line, and circular breather hole `o`) when `is_right=True`.
    """
    arm_ang = math.atan2(wy - elbow_y, wx - elbow_x)
    ca, sa = math.cos(arm_ang), math.sin(arm_ang)
    nx, ny = -sa, ca
    w_i = max(2, int(round(ink_w)))

    palm_cx = wx + ca * 13.0 * scale
    palm_cy = wy + sa * 13.0 * scale

    # If right hand, draw the back shaft of the Fountain-Nib Dip Pen first (behind fingers),
    # and the diamond Fountain-Nib front assembly emerging cleanly between thumb & index finger!
    pen_tip = (palm_cx + 22.0 * scale, palm_cy + 12.0 * scale)
    if is_right:
        p_rad = math.radians(pen_angle_deg)
        pca, psa = math.cos(p_rad), math.sin(p_rad)
        pnx, pny = -psa, pca
        grip_x = palm_cx + 8.0 * scale * ca
        grip_y = palm_cy + 8.0 * scale * sa
        back_x = grip_x - pca * 56.0 * scale
        back_y = grip_y - psa * 56.0 * scale
        collar_x = grip_x + pca * 24.0 * scale
        collar_y = grip_y + psa * 24.0 * scale
        nib_mid_x = grip_x + pca * 42.0 * scale
        nib_mid_y = grip_y + psa * 42.0 * scale
        tip_x = grip_x + pca * 66.0 * scale
        tip_y = grip_y + psa * 66.0 * scale
        pen_tip = (tip_x, tip_y)

        pw = 5.2 * scale
        shaft_poly = np.array([
            [int(back_x + pnx * pw * 0.75), int(back_y + pny * pw * 0.75)],
            [int(collar_x + pnx * pw), int(collar_y + pny * pw)],
            [int(collar_x - pnx * pw), int(collar_y - pny * pw)],
            [int(back_x - pnx * pw * 0.75), int(back_y - pny * pw * 0.75)],
        ], dtype=np.int32)
        if skin_fill is not None:
            s_fill = (248, 250, 252) if skin_fill == (255, 255, 255) else ((62, 82, 184) if has_ink_pen else BGR_NONPHOTO_BLUE)
            cv2.fillPoly(canvas, [shaft_poly], s_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [shaft_poly], True, ink_col, w_i, cv2.LINE_AA)

    # Articulated Cartoon Fingers & Palm (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`)
    if hand_pose in ("open_gesture", "cheer_high", "wave_pen", "flourish"):
        spreads = [-0.62, -0.22, 0.12, 0.45]
        lengths = [18.0, 22.0, 21.0, 17.0]
    elif hand_pose in ("idea_point",):
        spreads = [-0.55, -0.05, 0.32, 0.55]
        lengths = [16.0, 26.0, 13.0, 11.0]
    else:
        spreads = [-0.45, -0.10, 0.18, 0.40]
        lengths = [16.0, 19.0, 15.0, 13.0]

    # Draw palm oval
    p_deg = math.degrees(arm_ang)
    if skin_fill is not None:
        cv2.ellipse(canvas, (int(palm_cx), int(palm_cy)), (int(15.5 * scale), int(12.5 * scale)), p_deg, 0, 360, skin_fill, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(palm_cx), int(palm_cy)), (int(15.5 * scale), int(12.5 * scale)), p_deg, 0, 360, ink_col, w_i, cv2.LINE_AA)

    # Draw rounded cartoon fingers
    for fi, (sp_a, flen) in enumerate(zip(spreads, lengths)):
        f_ang = arm_ang + sp_a
        fcx = palm_cx + math.cos(f_ang) * (flen * 0.58 * scale) + nx * (fi - 1.5) * 3.8 * scale
        fcy = palm_cy + math.sin(f_ang) * (flen * 0.58 * scale) + ny * (fi - 1.5) * 3.8 * scale
        frx = int(max(3, flen * 0.48 * scale))
        fry = int(max(3, 5.2 * scale))
        f_deg = math.degrees(f_ang)
        if skin_fill is not None:
            cv2.ellipse(canvas, (int(fcx), int(fcy)), (frx, fry), f_deg, 0, 360, skin_fill, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(fcx), int(fcy)), (frx, fry), f_deg, 0, 360, ink_col, w_i, cv2.LINE_AA)

    # Re-cover center of palm so finger bases merge seamlessly into the hand!
    if skin_fill is not None:
        cv2.ellipse(canvas, (int(palm_cx), int(palm_cy)), (int(13.0 * scale), int(10.0 * scale)), p_deg, 0, 360, skin_fill, -1, cv2.LINE_AA)

    # Draw the iconic Diamond-Shaped Fountain-Pen Nib Head (`agent_nib_keys.jpg` Q1-Q4)!
    if is_right:
        p_rad = math.radians(pen_angle_deg)
        pca, psa = math.cos(p_rad), math.sin(p_rad)
        pnx, pny = -psa, pca
        grip_x = palm_cx + 8.0 * scale * ca
        grip_y = palm_cy + 8.0 * scale * sa
        collar_x = grip_x + pca * 22.0 * scale
        collar_y = grip_y + psa * 22.0 * scale
        nib_mid_x = grip_x + pca * 42.0 * scale
        nib_mid_y = grip_y + psa * 42.0 * scale
        tip_x, tip_y = pen_tip
        nw = 9.5 * scale

        # Ferrule collar ring
        ferrule = np.array([
            [int(collar_x - pca * 4 * scale + pnx * 6 * scale), int(collar_y - psa * 4 * scale + pny * 6 * scale)],
            [int(collar_x + pca * 4 * scale + pnx * 6 * scale), int(collar_y + psa * 4 * scale + pny * 6 * scale)],
            [int(collar_x + pca * 4 * scale - pnx * 6 * scale), int(collar_y + psa * 4 * scale - pny * 6 * scale)],
            [int(collar_x - pca * 4 * scale - pnx * 6 * scale), int(collar_y - psa * 4 * scale - pny * 6 * scale)],
        ], dtype=np.int32)
        if skin_fill is not None:
            cv2.fillPoly(canvas, [ferrule], (255, 255, 255) if skin_fill == (255, 255, 255) else (190, 215, 238), cv2.LINE_AA)
        cv2.polylines(canvas, [ferrule], True, ink_col, w_i, cv2.LINE_AA)

        # Diamond Fountain-Nib Head
        nib_poly = np.array([
            [int(collar_x + pca * 4 * scale + pnx * 4.5 * scale), int(collar_y + psa * 4 * scale + pny * 4.5 * scale)],
            [int(nib_mid_x + pnx * nw), int(nib_mid_y + pny * nw)],
            [int(tip_x), int(tip_y)],
            [int(nib_mid_x - pnx * nw), int(nib_mid_y - pny * nw)],
            [int(collar_x + pca * 4 * scale - pnx * 4.5 * scale), int(collar_y + psa * 4 * scale - pny * 4.5 * scale)],
        ], dtype=np.int32)
        if skin_fill is not None:
            n_fill = (255, 255, 255) if skin_fill == (255, 255, 255) else (196, 232, 252)
            cv2.fillPoly(canvas, [nib_poly], n_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [nib_poly], True, ink_col, w_i, cv2.LINE_AA)

        # Nib central slit line + circular breather hole `o`!
        hole_x = int(nib_mid_x - pca * 2.0 * scale)
        hole_y = int(nib_mid_y - psa * 2.0 * scale)
        cv2.line(canvas, (int(tip_x), int(tip_y)), (hole_x, hole_y), ink_col, max(2, w_i - 1), cv2.LINE_AA)
        cv2.circle(canvas, (hole_x, hole_y), max(2, int(3.2 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (hole_x, hole_y), max(2, int(3.2 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)

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
    around studio floor anchor `(origin_x, origin_y)` at scale `scale`, matching the
    exact canonical character design and bold line quality of:
      - `assets/studio_doodles/nib_inking_keys.jpg`
      - `assets/studio_doodles/agent_nib_keys.jpg`
      - `assets/studio_doodles/nib_moviola_keys.jpg`
      - `assets/studio_doodles/nib_cameracrank_keys.jpg`
      - `assets/studio_doodles/studio_crew_celebration_keys.jpg`
    """
    is_rough = (stage_mode == "rough_blue")
    is_cleanup = (stage_mode == "cleanup_ink")
    is_colored = (stage_mode in ("gouache_cel", "multiplane_composite"))

    ink_col = ink_override if ink_override is not None else (BGR_NONPHOTO_BLUE if is_rough else BGR_INK)
    ink_w = max(2.0, (2.2 if is_rough else 3.8) * max(0.65, min(1.45, scale)))
    w_i = max(2, int(round(ink_w)))

    # Canonical Palette:
    # - In `cleanup_ink` (`Pass 2`): Pure crisp white fills `(255, 255, 255)` with bold black contours,
    #   matching the exact look of `nib_inking_keys.jpg` and `agent_nib_keys.jpg`!
    # - In `multiplane_composite` (`Pass 3`): Warm vintage cel gouache fills!
    if is_colored and ink_override is None:
        skin_fill = (232, 243, 252)
        shirt_fill = (248, 250, 254)
        apron_fill = (88, 114, 206)
        apron_strap_fill = (72, 96, 184)
        pants_fill = (84, 94, 112)
        pants_far = (68, 76, 92)
        shoe_fill = (56, 68, 98)
        crown_fill = (214, 238, 252)
    elif is_cleanup and ink_override is None:
        skin_fill = (255, 255, 255)
        shirt_fill = (255, 255, 255)
        apron_fill = (255, 255, 255)
        apron_strap_fill = (255, 255, 255)
        pants_fill = (255, 255, 255)
        pants_far = (244, 246, 248)
        shoe_fill = (244, 246, 248)
        crown_fill = (255, 255, 255)
    else:
        skin_fill = shirt_fill = apron_fill = apron_strap_fill = pants_fill = pants_far = shoe_fill = crown_fill = None

    # --- 1. CLASSIC 3-HEAD-TALL CARTOON PROPORTIONS (`studio_crew_celebration_keys.jpg` & `nib_inking_keys.jpg`) ---
    # Map the 16 key pose coordinates into compact, sturdy 3-head-tall cartoon proportions!
    hip_x = origin_x + pose["hip_dx"] * 0.75 * scale
    hip_y = origin_y + (pose["hip_dy"] * 0.56) * scale
    spine_rad = math.radians(pose["spine_bend"] * 0.72)
    csq = float(pose["chest_squash"])

    torso_len = 116.0 * scale / (csq ** 0.55)
    chest_x = hip_x + torso_len * math.sin(spine_rad)
    chest_y = hip_y - torso_len * math.cos(spine_rad)

    mid_spine_x = 0.5 * (hip_x + chest_x) + 8.0 * scale * math.sin(spine_rad)
    mid_spine_y = 0.5 * (hip_y + chest_y)

    neck_x = chest_x + 8.0 * scale * math.sin(spine_rad * 0.7)
    neck_y = chest_y - 10.0 * scale

    # Large, appealing round cartoon cranium (`rx = 66 * scale, ry = 60 * scale`) sitting right above the V-collar!
    h_sq = float(pose["head_squash"])
    hrx = 66.0 * scale * (h_sq ** 0.45)
    hry = 60.0 * scale / (h_sq ** 0.45)
    head_cx = neck_x + 6.0 * scale
    head_cy = neck_y - 48.0 * scale

    # Map hand targets proportionally to the desk & body
    lh_target = (origin_x + pose["lh_x"] * 0.85 * scale, origin_y + (pose["lh_y"] * 0.68) * scale)
    rh_target = (origin_x + pose["rh_x"] * 0.85 * scale, origin_y + (pose["rh_y"] * 0.68) * scale)

    # Helper to draw a smooth cartoon limb segment + thick rolled elbow cuff
    def _draw_cartoon_arm(sh_pt, el_pt, wr_pt, w_up, w_low):
        for p0, p1, wa, wb, f_col in [
            (sh_pt, el_pt, w_up, w_up * 0.86, shirt_fill),
            (el_pt, wr_pt, w_low, w_low * 0.72, skin_fill),
        ]:
            dx_s, dy_s = p1[0] - p0[0], p1[1] - p0[1]
            L_s = max(1e-4, math.hypot(dx_s, dy_s))
            nx_s, ny_s = -dy_s / L_s, dx_s / L_s
            mx_s, my_s = 0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])
            wm = 0.54 * (wa + wb)
            poly_s = np.array([
                [int(p0[0] + nx_s * wa), int(p0[1] + ny_s * wa)],
                [int(mx_s + nx_s * wm), int(my_s + ny_s * wm)],
                [int(p1[0] + nx_s * wb), int(p1[1] + ny_s * wb)],
                [int(p1[0] - nx_s * wb), int(p1[1] - ny_s * wb)],
                [int(mx_s - nx_s * wm), int(my_s - ny_s * wm)],
                [int(p0[0] - nx_s * wa), int(p0[1] - ny_s * wa)],
            ], dtype=np.int32)
            if f_col is not None:
                cv2.fillPoly(canvas, [poly_s], f_col, cv2.LINE_AA)
            cv2.polylines(canvas, [poly_s], True, ink_col, w_i, cv2.LINE_AA)

        # Thick, puffy rolled-up sleeve cuff at elbow (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`)!
        cuff_deg = math.degrees(math.atan2(el_pt[1] - sh_pt[1], el_pt[0] - sh_pt[0])) + 90.0
        crx, cry = int(w_up * 1.18), int(9.5 * scale)
        if shirt_fill is not None:
            cv2.ellipse(canvas, (int(el_pt[0]), int(el_pt[1])), (crx, cry), cuff_deg, 0, 360, shirt_fill, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(el_pt[0]), int(el_pt[1])), (crx, cry), cuff_deg, 0, 360, ink_col, w_i, cv2.LINE_AA)
        # Inner cuff roll fold line
        cv2.ellipse(canvas, (int(el_pt[0]), int(el_pt[1])), (int(crx * 0.72), int(cry * 0.45)), cuff_deg, 15, 165, ink_col, max(1, w_i - 1), cv2.LINE_AA)

    # --- 2. FAR ARM (Left Arm Reaching to Acme Peg Bar / Paper Corner) ---
    sh_lx = chest_x - 24.0 * scale
    sh_ly = chest_y + 18.0 * scale
    u_arm = 78.0 * scale
    f_arm = 74.0 * scale
    el_lx, el_ly, wr_lx, wr_ly = solve_2bone_ik(sh_lx, sh_ly, lh_target[0], lh_target[1], u_arm, f_arm, prefer_lower_joint=True)
    _draw_cartoon_arm((sh_lx, sh_ly), (el_lx, el_ly), (wr_lx, wr_ly), 19.0 * scale, 15.0 * scale)
    _draw_articulated_hand(
        canvas, wr_lx, wr_ly, el_lx, el_ly, scale,
        hand_pose=pose["lh_pose"], is_right=False,
        skin_fill=skin_fill, ink_col=ink_col, ink_w=ink_w,
        boil_phase=boil_phase, seed=105,
    )

    # --- 3. STURDY CARTOON LEGS, CUFFED TROUSERS & ROUNDED ANIMATOR SHOES (`studio_crew_celebration_keys.jpg`) ---
    thigh_l = 72.0 * scale
    calf_l = 70.0 * scale
    for li, (hip_off_x, f_dx, f_dy, heel_lift, p_fill) in enumerate([
        (-22.0 * scale, pose["l_foot_dx"] * 0.9 * scale, pose["l_foot_dy"] * 0.65 * scale, pose["l_heel_lift"] * 0.6 * scale, pants_far),
        (20.0 * scale, pose["r_foot_dx"] * 0.9 * scale, pose["r_foot_dy"] * 0.65 * scale, pose["r_heel_lift"] * 0.6 * scale, pants_fill),
    ]):
        hx_j = hip_x + hip_off_x
        hy_j = hip_y + 8.0 * scale
        ank_x = origin_x + f_dx
        ank_y = origin_y + f_dy - 14.0 * scale - heel_lift * 0.5
        kx, ky, ax_e, ay_e = solve_2bone_ik(hx_j, hy_j, ank_x, ank_y, thigh_l, calf_l, prefer_forward_x=True)

        w0, w1, w2 = 22.0 * scale, 19.0 * scale, 16.5 * scale
        dx1, dy1 = kx - hx_j, ky - hy_j
        L1 = max(1e-4, math.hypot(dx1, dy1))
        n1x, n1y = -dy1 / L1, dx1 / L1
        dx2, dy2 = ax_e - kx, ay_e - ky
        L2 = max(1e-4, math.hypot(dx2, dy2))
        n2x, n2y = -dy2 / L2, dx2 / L2
        leg_poly = np.array([
            [int(hx_j + n1x * w0), int(hy_j + n1y * w0)],
            [int(kx + 0.5 * (n1x + n2x) * w1), int(ky + 0.5 * (n1y + n2y) * w1)],
            [int(ax_e + n2x * w2), int(ay_e + n2y * w2)],
            [int(ax_e - n2x * w2), int(ay_e - n2y * w2)],
            [int(kx - 0.5 * (n1x + n2x) * w1), int(ky - 0.5 * (n1y + n2y) * w1)],
            [int(hx_j - n1x * w0), int(hy_j - n1y * w0)],
        ], dtype=np.int32)
        if p_fill is not None:
            cv2.fillPoly(canvas, [leg_poly], p_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [leg_poly], True, ink_col, w_i, cv2.LINE_AA)

        # Rounded Cartoon Animator Shoe + Trouser Cuff (`studio_crew_celebration_keys.jpg` Q3/Q4)
        shoe_cx = int(ank_x + 12.0 * scale)
        shoe_cy = int(origin_y + f_dy - 9.0 * scale - heel_lift * 0.3)
        if shoe_fill is not None:
            cv2.ellipse(canvas, (shoe_cx, shoe_cy), (int(28 * scale), int(12.5 * scale)), -heel_lift * 0.5, 0, 360, shoe_fill, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (shoe_cx, shoe_cy), (int(28 * scale), int(12.5 * scale)), -heel_lift * 0.5, 0, 360, ink_col, w_i, cv2.LINE_AA)

        cuff_rect = np.array([
            [int(ax_e - 18 * scale), int(ay_e - 8 * scale)],
            [int(ax_e + 18 * scale), int(ay_e - 8 * scale)],
            [int(ax_e + 18 * scale), int(ay_e + 3 * scale)],
            [int(ax_e - 18 * scale), int(ay_e + 3 * scale)],
        ], dtype=np.int32)
        if p_fill is not None:
            cv2.fillPoly(canvas, [cuff_rect], p_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [cuff_rect], True, ink_col, w_i, cv2.LINE_AA)

    # --- 4. TORSO, SHIRT, CANONICAL BIB APRON & POINTED V-COLLAR (`\/\/`) ---
    w_chest = 48.0 * scale * (csq ** 0.45)
    w_waist = 44.0 * scale * (csq ** 0.50)
    w_hip = 46.0 * scale * (csq ** 0.40)

    torso_poly = np.array([
        [int(chest_x - w_chest), int(chest_y + 4.0 * scale)],
        [int(chest_x + w_chest), int(chest_y + 4.0 * scale)],
        [int(mid_spine_x + w_waist), int(mid_spine_y)],
        [int(hip_x + w_hip), int(hip_y + 14.0 * scale)],
        [int(hip_x - w_hip), int(hip_y + 14.0 * scale)],
        [int(mid_spine_x - w_waist), int(mid_spine_y)],
    ], dtype=np.int32)
    if shirt_fill is not None:
        cv2.fillPoly(canvas, [torso_poly], shirt_fill, cv2.LINE_AA)
    cv2.polylines(canvas, [torso_poly], True, ink_col, w_i, cv2.LINE_AA)

    # Canonical Animator's Bib Apron (`nib_inking_keys.jpg`, `agent_nib_keys.jpg`, `studio_crew_celebration_keys.jpg`)
    ap_lag = pose["apron_lag"] * 0.7 * scale
    bib_top_y = chest_y + 34.0 * scale
    apron_skirt = np.array([
        [int(chest_x - w_chest * 0.62), int(bib_top_y)],
        [int(chest_x + w_chest * 0.68), int(bib_top_y)],
        [int(mid_spine_x + w_waist * 0.90 + ap_lag * 0.35), int(mid_spine_y + 16.0 * scale)],
        [int(hip_x + w_hip * 1.02 + ap_lag), int(hip_y + 52.0 * scale)],
        [int(hip_x - w_hip * 0.88 + ap_lag * 0.8), int(hip_y + 52.0 * scale)],
        [int(mid_spine_x - w_waist * 0.82 + ap_lag * 0.3), int(mid_spine_y + 16.0 * scale)],
    ], dtype=np.int32)
    if apron_fill is not None:
        cv2.fillPoly(canvas, [apron_skirt], apron_fill, cv2.LINE_AA)
    cv2.polylines(canvas, [apron_skirt], True, ink_col, w_i, cv2.LINE_AA)

    # Two Apron Shoulder Straps & Square-U Chest Bib Top (`nib_inking_keys.jpg` Q1-Q4)!
    for side_s, x_top_f, x_bot_f in [(-1, -0.58, -0.46), (1, 0.62, 0.50)]:
        strap_poly = np.array([
            [int(chest_x + x_top_f * w_chest - 6 * scale), int(chest_y + 4 * scale)],
            [int(chest_x + x_top_f * w_chest + 6 * scale), int(chest_y + 4 * scale)],
            [int(chest_x + x_bot_f * w_chest + 6 * scale), int(bib_top_y)],
            [int(chest_x + x_bot_f * w_chest - 6 * scale), int(bib_top_y)],
        ], dtype=np.int32)
        if apron_strap_fill is not None:
            cv2.fillPoly(canvas, [strap_poly], apron_strap_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [strap_poly], True, ink_col, w_i, cv2.LINE_AA)

    # Crisp Pointed V-Neck Shirt Collar (`\/\/` two triangular collar flaps meeting at central V)!
    col_cx = chest_x + 4.0 * scale
    col_cy = chest_y + 6.0 * scale
    for side_c in (-1, 1):
        flap = np.array([
            [int(col_cx), int(col_cy + 14 * scale)],
            [int(col_cx + side_c * 14 * scale), int(col_cy + 24 * scale)],
            [int(col_cx + side_c * 24 * scale), int(col_cy + 4 * scale)],
            [int(col_cx + side_c * 14 * scale), int(col_cy - 4 * scale)],
        ], dtype=np.int32)
        if shirt_fill is not None:
            cv2.fillPoly(canvas, [flap], shirt_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [flap], True, ink_col, w_i, cv2.LINE_AA)

    # --- 5. CANONICAL ROUND CARTOON CRANIUM, 3-POINTED FOUNTAIN-NIB CROWN HAT & FACIAL ACTING ---
    hpitch = float(pose["head_pitch"]) * 0.45
    rad_p = math.radians(hpitch)
    cp, sp = math.cos(rad_p), math.sin(rad_p)

    def _hpt(lx_u: float, ly_u: float) -> tuple:
        lx = lx_u * hrx
        ly = ly_u * hry
        return (head_cx + lx * cp - ly * sp, head_cy + lx * sp + ly * cp)

    # Smooth Round-Oval Cartoon Head Cranium (`nib_inking_keys.jpg`, `agent_nib_keys.jpg`)
    if skin_fill is not None:
        cv2.ellipse(canvas, (int(head_cx), int(head_cy)), (int(hrx), int(hry)), hpitch, 0, 360, skin_fill, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(head_cx), int(head_cy)), (int(hrx), int(hry)), hpitch, 0, 360, ink_col, w_i, cv2.LINE_AA)

    # Non-Photo Blue 3D Construction Cross-Hairs ONLY in `rough_blue` (`Pass 1`) so `Pass 2` & `Pass 3` have 100% clean faces!
    if is_rough:
        eye_line = np.array([_hpt(-0.92, -0.02), _hpt(0.0, 0.02), _hpt(0.92, -0.02)], dtype=np.int32)
        ctr_line = np.array([_hpt(0.14, -0.92), _hpt(0.18, 0.0), _hpt(0.14, 0.92)], dtype=np.int32)
        cv2.polylines(canvas, [eye_line], False, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [ctr_line], False, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    # Canonical 3-Pointed Fountain-Pen Nib Crown Hat (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`)!
    # Features:
    #   - Curved headband rim across the forehead (`y = -0.44 * hry`)
    #   - Sharp Left Peak & Right Peak
    #   - Tall Center Fountain-Pen Nib Peak with vertical slit line + circular breather hole `o`!
    crown_lag = float(pose["hair_lag"]) * 0.006
    crown_pts = np.array([
        _hpt(-0.92, -0.42),
        _hpt(-1.06 + crown_lag * 0.5, -1.24),
        _hpt(-0.44 + crown_lag * 0.3, -0.84),
        _hpt(0.02 + crown_lag, -1.94),
        _hpt(0.48 + crown_lag * 0.3, -0.84),
        _hpt(1.04 + crown_lag * 0.5, -1.22),
        _hpt(0.92, -0.42),
        _hpt(0.45, -0.35),
        _hpt(0.0, -0.32),
        _hpt(-0.45, -0.35),
    ], dtype=np.int32)
    if crown_fill is not None:
        cv2.fillPoly(canvas, [crown_pts], crown_fill, cv2.LINE_AA)
    cv2.polylines(canvas, [crown_pts], True, ink_col, w_i, cv2.LINE_AA)

    # Fountain-Nib Crown Slit & Circular Breather Hole `o`
    apex_pt = _hpt(0.02 + crown_lag, -1.94)
    hole_pt = _hpt(0.01 + crown_lag * 0.35, -0.88)
    cv2.line(canvas, (int(apex_pt[0]), int(apex_pt[1])), (int(hole_pt[0]), int(hole_pt[1])), ink_col, w_i, cv2.LINE_AA)
    hole_r = max(4, int(round(0.115 * hrx)))
    if crown_fill is not None:
        cv2.circle(canvas, (int(hole_pt[0]), int(hole_pt[1])), hole_r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(hole_pt[0]), int(hole_pt[1])), hole_r, ink_col, w_i, cv2.LINE_AA)

    # --- 6. CANONICAL FACIAL ACTING (`● ●` Solid Black Oval Eyes, Expressive Brows, C-Nose, 5 Mouths) ---
    eye_open = float(pose["eye_open"])
    is_wink = (pose.get("code") == "K16" and pose.get("raw_u", 1.0) > 0.65)
    is_happy_arch = (pose.get("code") == "K14" and 0.25 <= pose.get("raw_u", 0.5) <= 0.85)
    face_shift_u = 0.14  # Slight 3/4 rightward gaze toward the animation desk

    for ei, (ex_u, ey_u, brow_val) in enumerate([
        (-0.24 + face_shift_u, -0.01, pose["brow_l"]),
        (0.28 + face_shift_u, -0.01, pose["brow_r"]),
    ]):
        ecx, ecy = _hpt(ex_u + pose["look_dx"] * 0.04, ey_u + pose["look_dy"] * 0.03)
        erx = max(3, int(round(0.105 * hrx)))
        ery = max(4, int(round(0.165 * hry * float(np.clip(eye_open, 0.55, 1.20)))))

        if ei == 0 and is_wink:
            # Winking chevron `>` (`nib_moviola_keys.jpg` Q3)
            w_pts = np.array([
                [int(ecx - erx * 1.2), int(ecy - ery * 0.55)],
                [int(ecx + erx * 1.1), int(ecy)],
                [int(ecx - erx * 1.2), int(ecy + ery * 0.55)],
            ], dtype=np.int32)
            cv2.polylines(canvas, [w_pts], False, ink_col, w_i, cv2.LINE_AA)
        elif is_happy_arch:
            # Joyful arched crescent eyes `^ ^` (`nib_cameracrank_keys.jpg` Q4)
            cv2.ellipse(canvas, (int(ecx), int(ecy + ery * 0.3)), (int(erx * 1.25), int(ery * 0.85)), hpitch, 195, 345, ink_col, w_i, cv2.LINE_AA)
        else:
            # Canonical solid-black vertical oval cartoon eye (`● ●`)!
            cv2.ellipse(canvas, (int(ecx), int(ecy)), (erx, ery), hpitch, 0, 360, ink_col, -1, cv2.LINE_AA)

        # Expressive Eyebrow above each eye (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`)
        by_c = ecy - ery * 1.35 - brow_val * 0.35 * scale
        tilt_b = -brow_val * 0.35 * scale * (-1 if ei == 0 else 1)
        brow_pts = np.array([
            [int(ecx - erx * 1.35), int(by_c - tilt_b)],
            [int(ecx), int(by_c - abs(brow_val) * 0.22 * scale - 3.0 * scale)],
            [int(ecx + erx * 1.35), int(by_c + tilt_b)],
        ], dtype=np.int32)
        cv2.polylines(canvas, [brow_pts], False, ink_col, w_i, cv2.LINE_AA)

    # Cute Rounded C/U-Shaped Cartoon Button Nose (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`)
    ncx, ncy = _hpt(0.06 + face_shift_u, 0.16)
    cv2.ellipse(
        canvas, (int(ncx), int(ncy)),
        (max(3, int(round(0.115 * hrx))), max(3, int(round(0.085 * hry)))),
        hpitch + 10, -35, 205, ink_col, w_i, cv2.LINE_AA,
    )

    # Canonical Mouths from `nib_inking_keys.jpg` Q1-Q4 & `agent_nib_keys.jpg` Q1-Q4!
    m_type = pose["mouth"]
    mx, my = _hpt(0.02 + face_shift_u, 0.44)
    if m_type == "think_pucker":
        # `nib_inking_keys.jpg` Q1: Determined slight downward frown arch `︵`
        cv2.ellipse(canvas, (int(mx), int(my + 6 * scale)), (int(16 * scale), int(7 * scale)), hpitch, 200, 340, ink_col, w_i, cv2.LINE_AA)
    elif m_type == "tongue_concentrate":
        # `nib_inking_keys.jpg` Q2: Concentrated smile curve + cute rounded tongue sticking out the right corner!
        smile_pts = np.array([
            [int(mx - 16 * scale), int(my + 2 * scale)],
            [int(mx + 2 * scale), int(my + 7 * scale)],
            [int(mx + 20 * scale), int(my - 4 * scale)],
        ], dtype=np.int32)
        cv2.polylines(canvas, [smile_pts], False, ink_col, w_i, cv2.LINE_AA)
        # Side tongue loop at `(mx + 16 * scale, my - 2 * scale)`
        tcx, tcy = int(mx + 18 * scale), int(my - 2 * scale)
        t_fill = (178, 168, 248) if is_colored else ((255, 255, 255) if is_cleanup else None)
        if t_fill is not None:
            cv2.ellipse(canvas, (tcx, tcy), (int(10 * scale), int(7.5 * scale)), -20, 0, 360, t_fill, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (tcx, tcy), (int(10 * scale), int(7.5 * scale)), -20, 0, 360, ink_col, w_i, cv2.LINE_AA)
        cv2.line(canvas, (int(tcx - 4 * scale), int(tcy + 1 * scale)), (int(tcx + 5 * scale), int(tcy - 2 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)
    elif m_type == "blow_puff" or pose.get("code") == "K10":
        # `nib_inking_keys.jpg` Panel 6 (`6. BLOW DRY`): Puffed cheeks `) (` blowing air stream + fluffy cartoon puff cloud `💨`!
        cv2.ellipse(canvas, (int(mx - 10 * scale), int(my)), (int(6 * scale), int(9 * scale)), 0, -70, 70, ink_col, w_i, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(mx + 6 * scale), int(my)), (int(5 * scale), int(6.5 * scale)), 0, 0, 360, ink_col, -1, cv2.LINE_AA)
        # Sweeping air lines + fluffy cloud `💨` blowing across the freshly inked paper on the desk!
        for ay_off in (-3.5, 0.0, 3.8):
            air_pts = np.array([
                [int(mx + 16 * scale), int(my + ay_off * scale)],
                [int(mx + 46 * scale), int(my + (ay_off + 4.0) * scale)],
                [int(mx + 78 * scale), int(my + (ay_off + 8.0) * scale)],
            ], dtype=np.int32)
            cv2.polylines(canvas, [air_pts], False, ink_col, max(2, w_i - 1), cv2.LINE_AA)
        cloud_cx, cloud_cy = int(mx + 98 * scale), int(my + 8 * scale)
        for cl_dx, cl_dy, cl_r in [(0, 0, 13), (11, -7, 11), (20, 2, 12), (9, 9, 10)]:
            if not is_rough:
                cv2.circle(canvas, (cloud_cx + int(cl_dx * scale), cloud_cy + int(cl_dy * scale)), int(cl_r * scale), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cloud_cx + int(cl_dx * scale), cloud_cy + int(cl_dy * scale)), int(cl_r * scale), ink_col, max(2, w_i - 1), cv2.LINE_AA)
        if not is_rough:
            cv2.circle(canvas, (cloud_cx + int(10 * scale), cloud_cy), int(12 * scale), (255, 255, 255), -1, cv2.LINE_AA)
    elif m_type in ("big_grin_teeth", "open_laugh", "happy_grin", "oh_gasp"):
        # `agent_nib_keys.jpg` Q4 & `studio_crew_celebration_keys.jpg` Q2-Q4: Wide happy D-shaped open laugh mouth with bottom tongue!
        mw, mh = int(20 * scale), int(14 * scale)
        cv2.ellipse(canvas, (int(mx), int(my - 2 * scale)), (mw, mh), hpitch, 0, 180, ink_col, -1, cv2.LINE_AA)
        if not is_rough:
            t_col = (168, 158, 245) if is_colored else (255, 255, 255)
            cv2.ellipse(canvas, (int(mx), int(my + 6 * scale)), (int(mw * 0.65), int(mh * 0.42)), hpitch, 180, 360, t_col, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(mx), int(my - 2 * scale)), (mw, mh), hpitch, 0, 180, ink_col, w_i, cv2.LINE_AA)
        cv2.line(canvas, (int(mx - mw), int(my - 2 * scale)), (int(mx + mw), int(my - 2 * scale)), ink_col, w_i, cv2.LINE_AA)
    else:
        # `nib_inking_keys.jpg` Q3 & `agent_nib_keys.jpg` Q1-Q3: Confident upward smirk with corner tick!
        cv2.ellipse(canvas, (int(mx), int(my - 4 * scale)), (int(18 * scale), int(9 * scale)), hpitch - 6, 15, 165, ink_col, w_i, cv2.LINE_AA)
        cv2.line(canvas, (int(mx - 17 * scale), int(my - 5 * scale)), (int(mx - 20 * scale), int(my - 1 * scale)), ink_col, max(2, w_i - 1), cv2.LINE_AA)

    # --- 7. NEAR ARM (Right Drawing Arm) + MULTI-ARM SPEED ECHOES (`nib_inking_keys.jpg` Q3) + FOUNTAIN-NIB PEN ---
    sh_rx = chest_x - 6.0 * scale
    sh_ry = chest_y + 20.0 * scale
    el_rx, el_ry, wr_rx, wr_ry = solve_2bone_ik(sh_rx, sh_ry, rh_target[0], rh_target[1], u_arm * 1.04, f_arm * 1.04, prefer_lower_joint=True)

    # In rapid sketching/inking (`K04`, `K05`, `K09`), draw classical comic multi-arm motion-blur echoes & speed arcs `(( ))` (`nib_inking_keys.jpg` Q3)!
    if pose.get("seg_idx") in (3, 4, 8):
        for echo_off_y, echo_ang in [(-38.0 * scale, -55.0), (26.0 * scale, -18.0)]:
            e_tx, e_ty = rh_target[0] - 18.0 * scale, rh_target[1] + echo_off_y
            e_elx, e_ely, e_wrx, e_wry = solve_2bone_ik(sh_rx, sh_ry, e_tx, e_ty, u_arm, f_arm, prefer_lower_joint=True)
            # Speed arc lines `(( ))` behind moving arm
            arc_cx, arc_cy = int(0.5 * (e_wrx + wr_rx) + 24 * scale), int(0.5 * (e_wry + wr_ry))
            cv2.ellipse(canvas, (arc_cx, arc_cy), (int(18 * scale), int(34 * scale)), 15, -55, 55, ink_col, max(1, w_i - 2), cv2.LINE_AA)
            cv2.ellipse(canvas, (arc_cx + int(8 * scale), arc_cy), (int(18 * scale), int(30 * scale)), 15, -45, 45, ink_col, max(1, w_i - 2), cv2.LINE_AA)

    _draw_cartoon_arm((sh_rx, sh_ry), (el_rx, el_ry), (wr_rx, wr_ry), 20.5 * scale, 16.0 * scale)

    has_ink_pen = (pose.get("seg_idx", 0) >= 6)
    pen_tip = _draw_articulated_hand(
        canvas, wr_rx, wr_ry, el_rx, el_ry, scale,
        hand_pose=pose["rh_pose"], is_right=True,
        skin_fill=skin_fill, ink_col=ink_col, ink_w=ink_w,
        boil_phase=boil_phase, seed=175,
        pen_angle_deg=pose["pen_angle"], has_ink_pen=has_ink_pen,
    )

    # --- 8. PASS A/B CONSTRUCTION SPINE & RIBCAGE ONLY IN `rough_blue` (`Pass 1`) ---
    if is_rough and ink_override is None:
        loa = np.array([(hip_x, hip_y + 15 * scale), (mid_spine_x, mid_spine_y), (chest_x, chest_y), (head_cx, head_cy - 35 * scale)], dtype=np.int32)
        cv2.polylines(canvas, [loa], False, BGR_NONPHOTO_BLUE, 2, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(chest_x), int(chest_y + 22 * scale)), (int(w_chest * 0.85), int(36 * scale)), pose["spine_bend"] * 0.7, 0, 360, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hip_x), int(hip_y)), (int(w_hip * 0.85), int(24 * scale)), pose["pelvis_tilt"], 0, 360, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

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

    bg_line_col = BGR_NONPHOTO_BLUE if is_rough else BGR_INK
    bg_line_w = (1.5 if is_rough else 2.4) * s_ui
    paper_fill = BGR_PAPER if not is_rough else None
    shade_fill = BGR_PAPER_SHADE if not is_rough else None

    # ==========================================================================
    # PLANE 6 & 5 (`depth = 4.8` & `2.4`): STUDIO ARCHITECTURE, ROOF TRUSSES,
    # INDUSTRIAL MULTI-PANE FACTORY WINDOW & CAST-IRON RADIATOR
    # (Matches `multiplane_plane5_distant_bg_keys.jpg`)
    # ==========================================================================
    if is_colored:
        cv2.rectangle(canvas, (0, 0), (out_w, floor_y), (218, 234, 246), -1)
        cv2.rectangle(canvas, (0, floor_y), (out_w, out_h), (156, 186, 214), -1)
    else:
        canvas[:] = BGR_PAPER

    # Baseboard molding & floor line
    baseboard_y = max(0, floor_y - int(14 * s_ui))
    if is_colored:
        cv2.rectangle(canvas, (0, baseboard_y), (out_w, floor_y), (138, 168, 196), -1)
    cv2.line(canvas, (0, baseboard_y), (out_w, baseboard_y), bg_line_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)
    cv2.line(canvas, (0, floor_y), (out_w, floor_y), bg_line_col, max(2, int(3.0 * s_ui)), cv2.LINE_AA)

    # Perspective wooden floorboards & cross-joints shifting with Plane 2 depth
    for pi in range(-4, 15):
        fx_top, _, _ = _plane_tf((pi / 9.0) * out_w, floor_base_y, 1.0)
        fx_bot = out_w * 0.5 + (fx_top - out_w * 0.5) * 1.58
        p_col = (128, 156, 186) if is_colored else (BGR_NONPHOTO_BLUE if is_rough else (175, 168, 158))
        cv2.line(canvas, (int(fx_top), floor_y), (int(fx_bot), out_h), p_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)

    # Upper studio structural roof truss beam (`Plane 5`)
    _, truss_y, z_tr = _plane_tf(out_w * 0.5, out_h * 0.07, 3.2)
    truss_h = int(22 * s_ui * z_tr)
    if is_colored:
        cv2.rectangle(canvas, (0, max(0, int(truss_y - truss_h))), (out_w, int(truss_y)), (178, 192, 206), -1)
    cv2.line(canvas, (0, int(truss_y)), (out_w, int(truss_y)), bg_line_col, max(1, int(2.2 * s_ui)), cv2.LINE_AA)
    cv2.line(canvas, (0, max(0, int(truss_y - truss_h))), (out_w, max(0, int(truss_y - truss_h))), bg_line_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)
    for ti in range(8):
        tx0, _, _ = _plane_tf((ti / 7.0) * out_w, out_h * 0.07, 3.2)
        tx1, _, _ = _plane_tf(((ti + 0.5) / 7.0) * out_w, out_h * 0.07, 3.2)
        cv2.line(canvas, (int(tx0), 0), (int(tx1), int(truss_y)), bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA)
        cv2.line(canvas, (int(tx1), int(truss_y)), (int(tx0 + (tx1 - tx0) * 2), 0), bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA)

    # Industrial Multi-Pane Factory Window on Left (`4x5` panes + sill + Cast-Iron Radiator below)
    win_cx, win_cy, z_p5 = _plane_tf(out_w * 0.10, out_h * 0.35, 2.6)
    win_w, win_h = int(98 * s_ui * z_p5), int(168 * s_ui * z_p5)
    wx0, wy0 = int(win_cx - win_w), int(win_cy - win_h)
    wx1, wy1 = int(win_cx + win_w), int(win_cy + win_h)
    if is_colored:
        cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), (78, 58, 44), -1)
        # Crescent moon & distant water tower silhouette (`depth = 4.8` inside window)
        mx_w, my_w, _ = _plane_tf(out_w * 0.08, out_h * 0.23, 4.8)
        cv2.circle(canvas, (int(mx_w), int(my_w)), int(18 * s_ui), (192, 238, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(mx_w + 6 * s_ui), int(my_w - 4 * s_ui)), int(15 * s_ui), (78, 58, 44), -1, cv2.LINE_AA)
    elif is_cleanup:
        cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), BGR_PAPER, -1)
        # Diagonal glass reflection hatch marks inside window panes
        for r_i in range(3):
            rx_gl = wx0 + int((0.22 + r_i * 0.28) * (wx1 - wx0))
            ry_gl = wy0 + int((0.18 + r_i * 0.22) * (wy1 - wy0))
            cv2.line(canvas, (rx_gl, ry_gl + int(22 * s_ui)), (rx_gl + int(16 * s_ui), ry_gl), bg_line_col, max(1, int(1.4 * s_ui)), cv2.LINE_AA)

    # Outer double window casing & 4x5 industrial muntin grid
    cv2.rectangle(canvas, (wx0 - int(6 * s_ui), wy0 - int(6 * s_ui)), (wx1 + int(6 * s_ui), wy1 + int(6 * s_ui)), bg_line_col, max(2, int(2.8 * s_ui)), cv2.LINE_AA)
    cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), bg_line_col, max(1, int(2.0 * s_ui)), cv2.LINE_AA)
    for gx_i in range(1, 4):
        gx_p = wx0 + int((gx_i / 4.0) * (wx1 - wx0))
        cv2.line(canvas, (gx_p, wy0), (gx_p, wy1), bg_line_col, max(1, int(1.6 * s_ui)), cv2.LINE_AA)
    for gy_i in range(1, 5):
        gy_p = wy0 + int((gy_i / 5.0) * (wy1 - wy0))
        cv2.line(canvas, (wx0, gy_p), (wx1, gy_p), bg_line_col, max(1, int(1.6 * s_ui)), cv2.LINE_AA)
    # Protruding window sill
    sill_pts = [
        (wx0 - 12 * s_ui, wy1 + 4 * s_ui),
        (wx1 + 12 * s_ui, wy1 + 4 * s_ui),
        (wx1 + 12 * s_ui, wy1 + 18 * s_ui),
        (wx0 - 12 * s_ui, wy1 + 18 * s_ui),
    ]
    draw_ink_stroke(canvas, sill_pts, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=shade_fill, boil_phase=boil_phase, seed=41)

    # Cast-Iron Ribbed Steam Radiator under Window (`multiplane_plane5_distant_bg_keys.jpg`)
    rad_y0 = wy1 + int(30 * s_ui * z_p5)
    rad_y1 = min(baseboard_y - int(4 * s_ui), wy1 + int(134 * s_ui * z_p5))
    if rad_y1 > rad_y0 + 20:
        for r_fin in range(8):
            fx_r = wx0 + int(14 * s_ui) + int((r_fin / 7.0) * (wx1 - wx0 - 28 * s_ui))
            fin_w = max(4, int(8 * s_ui * z_p5))
            fin_pts = [
                (fx_r - fin_w, rad_y0),
                (fx_r + fin_w, rad_y0),
                (fx_r + fin_w, rad_y1),
                (fx_r - fin_w, rad_y1),
            ]
            draw_ink_stroke(
                canvas, fin_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.85,
                closed=True, fill_bgr=(162, 174, 186) if is_colored else shade_fill,
                boil_phase=boil_phase, seed=42 + r_fin,
            )

    # ==========================================================================
    # PLANE 4 (`depth = 2.1`): STORYBOARD CORKBOARD, 35MM FILM-CAN CREDENZA,
    # AND LIVE 4:3 STUDIO PENCIL-TEST PROJECTION SCREEN
    # (Matches `multiplane_plane4_midground_arch_keys.jpg` and keeps Nib's crown
    #  silhouette 100% clear per `Rule IX: DRAW FOR SILHOUETTE`!)
    # ==========================================================================
    sb_cx, sb_cy, z_sb = _plane_tf(out_w * 0.255, out_h * 0.29, 2.1)
    sb_w, sb_h = int(92 * s_ui * z_sb), int(64 * s_ui * z_sb)
    sb_pts = [
        (sb_cx - sb_w, sb_cy - sb_h),
        (sb_cx + sb_w, sb_cy - sb_h),
        (sb_cx + sb_w, sb_cy + sb_h),
        (sb_cx - sb_w, sb_cy + sb_h),
    ]
    draw_ink_stroke(
        canvas, sb_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1,
        closed=True, fill_bgr=(142, 178, 212) if is_colored else paper_fill,
        boil_phase=boil_phase, seed=48,
    )
    # Inner corkboard border & 6 pinned 4:3 storyboard cards
    cv2.rectangle(
        canvas,
        (int(sb_cx - sb_w + 7 * s_ui), int(sb_cy - sb_h + 7 * s_ui)),
        (int(sb_cx + sb_w - 7 * s_ui), int(sb_cy + sb_h - 7 * s_ui)),
        bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA,
    )
    for sc_r in range(2):
        for sc_c in range(3):
            cx_card = sb_cx - sb_w * 0.56 + sc_c * (sb_w * 0.56)
            cy_card = sb_cy - sb_h * 0.40 + sc_r * (sb_h * 0.80)
            cw_c, ch_c = int(21 * s_ui * z_sb), int(15 * s_ui * z_sb)
            cv2.rectangle(
                canvas, (int(cx_card - cw_c), int(cy_card - ch_c)),
                (int(cx_card + cw_c), int(cy_card + ch_c)),
                (248, 250, 252) if not is_rough else BGR_PAPER, -1,
            )
            cv2.rectangle(
                canvas, (int(cx_card - cw_c), int(cy_card - ch_c)),
                (int(cx_card + cw_c), int(cy_card + ch_c)),
                bg_line_col, max(1, int(1.4 * s_ui)), cv2.LINE_AA,
            )
            cv2.circle(canvas, (int(cx_card), int(cy_card - ch_c + 3 * s_ui)), max(2, int(2.2 * s_ui)), BGR_ACCENT_RED if is_colored else bg_line_col, -1, cv2.LINE_AA)
            cv2.line(canvas, (int(cx_card - cw_c * 0.55), int(cy_card + 2 * s_ui)), (int(cx_card + cw_c * 0.55), int(cy_card - 2 * s_ui)), bg_line_col, max(1, int(1.2 * s_ui)), cv2.LINE_AA)

    # Wooden Studio Credenza / Shelving Unit below Corkboard stacked with 35mm Film Cans (`multiplane_plane4_midground_arch_keys.jpg`)
    cred_top_y = int(floor_y - 135 * s_ui * z_sb)
    cred_bot_y = baseboard_y
    if cred_bot_y > cred_top_y + 20:
        cred_pts = [
            (sb_cx - sb_w * 0.95, cred_top_y),
            (sb_cx + sb_w * 0.95, cred_top_y),
            (sb_cx + sb_w * 0.95, cred_bot_y),
            (sb_cx - sb_w * 0.95, cred_bot_y),
        ]
        draw_ink_stroke(
            canvas, cred_pts, color_bgr=bg_line_col, base_thick=bg_line_w,
            closed=True, fill_bgr=(132, 168, 202) if is_colored else shade_fill,
            boil_phase=boil_phase, seed=49,
        )
        mid_shelf_y = (cred_top_y + cred_bot_y) // 2
        cv2.line(canvas, (int(sb_cx - sb_w * 0.92), mid_shelf_y), (int(sb_cx + sb_w * 0.92), mid_shelf_y), bg_line_col, max(1, int(2.0 * s_ui)), cv2.LINE_AA)
        # Stacked horizontal 35mm film cans on upper shelf & vertical dope-sheet binders on lower shelf
        for fc_i in range(3):
            fc_y = mid_shelf_y - int((8 + fc_i * 11) * s_ui * z_sb)
            cv2.rectangle(
                canvas,
                (int(sb_cx - sb_w * 0.72), fc_y - int(4 * s_ui)),
                (int(sb_cx - sb_w * 0.12), fc_y + int(4 * s_ui)),
                bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA,
            )

    # Live 4:3 Studio Pencil-Test Monitor on Right Midground (`out_w * 0.78`)
    from manual_hand_drawn_animation_studio import render_manual_cartoon_frame
    cb_cx, cb_cy, z_cb = _plane_tf(out_w * 0.78, out_h * 0.34, 2.1)
    cb_w = max(80, int(round(215 * s_ui * z_cb)))
    cb_h = max(60, int(round(cb_w * 0.75)))  # Exact 4:3!
    x0_pt = int(round(cb_cx - cb_w))
    y0_pt = int(round(cb_cy - cb_h))
    w_pt, h_pt = cb_w * 2, cb_h * 2
    x1_pt, y1_pt = x0_pt + w_pt, y0_pt + h_pt

    # Clip cleanly to canvas bounds so the 4:3 monitor NEVER disappears during camera zoom/pan!
    cx0, cy0 = max(0, x0_pt), max(0, y0_pt)
    cx1, cy1 = min(out_w, x1_pt), min(out_h, y1_pt)
    if cx1 > cx0 + 20 and cy1 > cy0 + 20:
        fox_test_img = render_manual_cartoon_frame(
            "campfire", frame_idx / 24.0, w_pt, h_pt,
            stage_mode=stage_mode, boil_phase=boil_phase,
            enable_onion_skin=(enable_onion_skin or is_rough),
        )
        canvas[cy0:cy1, cx0:cx1] = fox_test_img[cy0 - y0_pt:cy1 - y0_pt, cx0 - x0_pt:cx1 - x0_pt]
        cv2.rectangle(canvas, (x0_pt - int(5 * s_ui), y0_pt - int(5 * s_ui)), (x1_pt + int(5 * s_ui), y1_pt + int(5 * s_ui)), bg_line_col, max(2, int(3.2 * s_ui)), cv2.LINE_AA)
        cv2.rectangle(canvas, (x0_pt, y0_pt), (x1_pt, y1_pt), bg_line_col, max(2, int(2.2 * s_ui)), cv2.LINE_AA)
        cv2.rectangle(canvas, (x0_pt, y0_pt), (x1_pt, y0_pt + max(14, int(25 * s_ui))), bg_line_col, -1)
        cv2.putText(
            canvas, "4:3 STUDIO PENCIL-TEST MONITOR (FOX LEAP ANIMATION)",
            (x0_pt + int(8 * s_ui), y0_pt + max(11, int(18 * s_ui))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.37 * s_ui, (245, 248, 255), max(1, int(round(s_ui))), cv2.LINE_AA,
        )

    # ==========================================================================
    # PLANE 2 (`depth = 1.00`): NIB & 3D PERSPECTIVE TILTED BEECHWOOD ANIMATION DESK
    # (Matches `apparatus_lightbox_anim_keys.jpg` & `nib_inking_keys.jpg`!)
    # ==========================================================================
    nib_ox, nib_oy, z_nib = _plane_tf(out_w * 0.41, floor_base_y, 1.00)
    nib_scale = 1.56 * s_ui * z_nib

    desk_cx = nib_ox + 215.0 * nib_scale
    desk_fy = nib_oy
    wood_light = (158, 198, 230) if is_colored else paper_fill
    wood_side = (124, 162, 196) if is_colored else shade_fill
    wood_dark = (98, 134, 168) if is_colored else shade_fill
    brass_fill = (68, 192, 246) if is_colored else shade_fill

    # 1. A-Frame Wooden Trestle Legs, Splayed Floor Feet, Crossbar & Brass Tilt Quadrant
    for leg_off in (-118.0, 108.0):
        lx_d = desk_cx + leg_off * nib_scale
        d_leg = [
            (lx_d - 15 * nib_scale, desk_fy - 245 * nib_scale),
            (lx_d + 15 * nib_scale, desk_fy - 245 * nib_scale),
            (lx_d + 20 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d - 20 * nib_scale, desk_fy - 14 * nib_scale),
        ]
        draw_ink_stroke(canvas, d_leg, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=60 + int(leg_off))
        # Splayed wooden floor foot pad
        foot_pad = [
            (lx_d - 34 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d + 34 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d + 38 * nib_scale, desk_fy),
            (lx_d - 38 * nib_scale, desk_fy),
        ]
        draw_ink_stroke(canvas, foot_pad, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=61 + int(leg_off))

    # Horizontal stretcher beam between legs
    stretcher_pts = [
        (desk_cx - 110 * nib_scale, desk_fy - 98 * nib_scale),
        (desk_cx + 102 * nib_scale, desk_fy - 98 * nib_scale),
        (desk_cx + 102 * nib_scale, desk_fy - 78 * nib_scale),
        (desk_cx - 110 * nib_scale, desk_fy - 78 * nib_scale),
    ]
    draw_ink_stroke(canvas, stretcher_pts, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=62)

    # Curved tilt-adjust quadrant arc & wingnut under drafting board (`apparatus_lightbox_anim_keys.jpg`)
    quad_pts = [
        (desk_cx - 72 * nib_scale, desk_fy - 220 * nib_scale),
        (desk_cx - 38 * nib_scale, desk_fy - 178 * nib_scale),
        (desk_cx + 14 * nib_scale, desk_fy - 196 * nib_scale),
    ]
    draw_ink_stroke(canvas, quad_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1, closed=False, boil_phase=boil_phase, seed=63)
    cv2.circle(canvas, (int(desk_cx - 42 * nib_scale), int(desk_fy - 188 * nib_scale)), max(3, int(6 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)

    # 2. 3D Beveled Rectangular Drafting Board Tilted Toward Nib (`apparatus_lightbox_anim_keys.jpg`)
    # Top face of the tilted drafting board:
    b_fl = (desk_cx - 165 * nib_scale, desk_fy - 222 * nib_scale)  # Front-left (closest to Nib)
    b_bl = (desk_cx - 75 * nib_scale, desk_fy - 348 * nib_scale)   # Back-left (higher up)
    b_br = (desk_cx + 162 * nib_scale, desk_fy - 336 * nib_scale)  # Back-right
    b_fr = (desk_cx + 112 * nib_scale, desk_fy - 210 * nib_scale)  # Front-right
    board_top = [b_fl, b_bl, b_br, b_fr]
    # 3D Front & Right Bevel Thickness of the wooden drafting board:
    lip_h = 22.0 * nib_scale
    board_front_lip = [
        b_fl, b_fr,
        (b_fr[0], b_fr[1] + lip_h),
        (b_fl[0], b_fl[1] + lip_h),
    ]
    board_right_lip = [
        b_fr, b_br,
        (b_br[0], b_br[1] + lip_h * 0.85),
        (b_fr[0], b_fr[1] + lip_h),
    ]
    draw_ink_stroke(canvas, board_front_lip, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=64)
    draw_ink_stroke(canvas, board_right_lip, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=65)
    draw_ink_stroke(canvas, board_top, color_bgr=bg_line_col, base_thick=bg_line_w * 1.35, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=66)

    # Recessed Rectangular Lightbox Frame inside the Tilted Drafting Board
    lb_pts = [
        (desk_cx - 122 * nib_scale, desk_fy - 234 * nib_scale),
        (desk_cx - 56 * nib_scale, desk_fy - 332 * nib_scale),
        (desk_cx + 98 * nib_scale, desk_fy - 324 * nib_scale),
        (desk_cx + 52 * nib_scale, desk_fy - 224 * nib_scale),
    ]
    draw_ink_stroke(canvas, lb_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=67)

    # 3. Illuminated Milk-Glass Animation Disc (`disc_x = nib_ox + 195 * scale, disc_y = nib_oy - 282 * scale`)
    disc_x = nib_ox + 195.0 * nib_scale
    disc_y = nib_oy - 282.0 * nib_scale
    if is_colored:
        glow_ov = canvas.copy()
        cv2.ellipse(glow_ov, (int(disc_x), int(disc_y)), (int(96 * nib_scale), int(54 * nib_scale)), -10, 0, 360, (212, 250, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_ov, 0.38, canvas, 0.62, 0, canvas)

    draw_sculpted_oval(
        canvas, disc_x, disc_y, 84.0 * nib_scale, 45.0 * nib_scale,
        angle_deg=-10, color_bgr=bg_line_col, thick=bg_line_w * 1.2,
        fill_bgr=(216, 250, 255) if is_colored else paper_fill,
        boil_phase=boil_phase, seed=68,
    )
    # Diagonal milk-glass sheen lines (`//`) inside disc (`apparatus_lightbox_anim_keys.jpg`)
    for sh_k in (-1, 0, 1):
        gx_s = disc_x + sh_k * 26.0 * nib_scale
        gy_s = disc_y + sh_k * 4.0 * nib_scale
        cv2.line(
            canvas,
            (int(gx_s - 14 * nib_scale), int(gy_s + 14 * nib_scale)),
            (int(gx_s + 14 * nib_scale), int(gy_s - 14 * nib_scale)),
            bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA,
        )

    # 4. 3-Sheet Rolling Animation Paper Stack on 4-Peg Acme Registration Bar!
    p_flip = float(pose["paper_flip"])
    d_prog = float(pose["draw_prog"])
    c_lift = float(pose["cel_lift"])

    if c_lift < 0.05:
        for sh_i in range(3):
            curl_h = (sh_i * 9.0 + p_flip * 56.0 * (sh_i / 2.0)) * nib_scale
            sh_pts = [
                (disc_x - 66 * nib_scale, disc_y - 24 * nib_scale),
                (disc_x + 56 * nib_scale, disc_y - 30 * nib_scale),
                (disc_x + 64 * nib_scale + curl_h * 0.22, disc_y + 12 * nib_scale - curl_h * 0.65),
                (disc_x + 70 * nib_scale + curl_h * 0.32, disc_y + 22 * nib_scale - curl_h),
                (disc_x + 4 * nib_scale, disc_y + 26 * nib_scale - curl_h * 0.72),
                (disc_x - 54 * nib_scale, disc_y + 28 * nib_scale - curl_h * 0.42),
            ]
            draw_ink_stroke(
                canvas, sh_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95,
                closed=True, fill_bgr=BGR_PAPER if not is_rough else None,
                boil_phase=boil_phase, seed=70 + sh_i,
            )
            # Draw the animated Fox leap curve & cross-hatching on the active sheet (`closeup_hand_drawing_keys.jpg`)!
            if sh_i == 2 or p_flip > 0.25:
                fox_st = [
                    (disc_x - 40 * nib_scale, disc_y + 10 * nib_scale - curl_h * 0.3),
                    (disc_x - 10 * nib_scale, disc_y - (12 + sh_i * 6) * nib_scale - curl_h * 0.48),
                    (disc_x + 22 * nib_scale, disc_y - (10 + sh_i * 4) * nib_scale - curl_h * 0.55),
                    (disc_x + 40 * nib_scale, disc_y + 4 * nib_scale - curl_h * 0.62),
                ]
                st_col = BGR_INK if (d_prog > 0.65 and sh_i == 2 and not is_rough) else BGR_NONPHOTO_BLUE
                draw_ink_stroke(canvas, fox_st, color_bgr=st_col, base_thick=max(1.6, 2.4 * s_ui), boil_phase=boil_phase, seed=73 + sh_i)
                # Expressive fox head & ears on the top sheet as `draw_prog` advances!
                if sh_i == 2 and d_prog > 0.25:
                    fx_hx = disc_x + 18 * nib_scale
                    fx_hy = disc_y - 14 * nib_scale - curl_h * 0.52
                    cv2.circle(canvas, (int(fx_hx), int(fx_hy)), max(3, int(9 * nib_scale)), st_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)
                    # Pointed fox ear
                    ear_p = [
                        (fx_hx - 4 * nib_scale, fx_hy - 7 * nib_scale),
                        (fx_hx + 2 * nib_scale, fx_hy - 19 * nib_scale),
                        (fx_hx + 7 * nib_scale, fx_hy - 6 * nib_scale),
                    ]
                    draw_ink_stroke(canvas, ear_p, color_bgr=st_col, base_thick=max(1.4, 1.8 * s_ui), boil_phase=boil_phase, seed=76)

        # Classic hand-drawn double-arc paper-flip motion lines `(( ))` when `p_flip > 0.25` (`apparatus_lightbox_anim_keys.jpg`)
        if p_flip > 0.25:
            for m_arc in range(2):
                ax_m = disc_x + (78 + m_arc * 10) * nib_scale
                ay_m = disc_y - (12 + p_flip * 26) * nib_scale
                arc_m_pts = [
                    (ax_m - 6 * nib_scale, ay_m - 18 * nib_scale),
                    (ax_m + 6 * nib_scale, ay_m),
                    (ax_m - 4 * nib_scale, ay_m + 18 * nib_scale),
                ]
                draw_ink_stroke(canvas, arc_m_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.8, boil_phase=boil_phase, seed=77 + m_arc)

    # 5. Raised 4-Peg Brass Acme Registration Bar (`o | | o`) bolted along upper disc edge
    peg_pts = [
        (disc_x - 62 * nib_scale, disc_y - 29 * nib_scale),
        (disc_x + 54 * nib_scale, disc_y - 36 * nib_scale),
        (disc_x + 56 * nib_scale, disc_y - 24 * nib_scale),
        (disc_x - 60 * nib_scale, disc_y - 17 * nib_scale),
    ]
    draw_ink_stroke(canvas, peg_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.05, closed=True, fill_bgr=brass_fill, boil_phase=boil_phase, seed=78)
    # 4 upright registration pegs (`apparatus_lightbox_anim_keys.jpg`)
    for p_u in (-0.75, -0.25, 0.25, 0.75):
        px_peg = disc_x - 4 * nib_scale + p_u * 54.0 * nib_scale
        py_peg = disc_y - 26.5 * nib_scale - p_u * 3.5 * nib_scale
        cv2.circle(canvas, (int(px_peg), int(py_peg)), max(2, int(3.5 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)
        cv2.line(canvas, (int(px_peg), int(py_peg)), (int(px_peg), int(py_peg - 8 * nib_scale)), bg_line_col, max(1, int(2.2 * s_ui)), cv2.LINE_AA)

    # 6. Upper-Left Desk Corner Prop: Ceramic Mug of Sharpened Pencils (`apparatus_lightbox_anim_keys.jpg`)
    mug_x = desk_cx - 48.0 * nib_scale
    mug_y = desk_fy - 355.0 * nib_scale
    mug_pts = [
        (mug_x - 16 * nib_scale, mug_y - 22 * nib_scale),
        (mug_x + 16 * nib_scale, mug_y - 22 * nib_scale),
        (mug_x + 14 * nib_scale, mug_y + 16 * nib_scale),
        (mug_x - 14 * nib_scale, mug_y + 16 * nib_scale),
    ]
    draw_ink_stroke(canvas, mug_pts, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=79)
    # 4 sharpened pencils fanning out of the mug
    for p_ang in (-22, -7, 8, 24):
        rad_p = math.radians(p_ang - 90)
        p_tx = mug_x + math.cos(rad_p) * 44.0 * nib_scale
        p_ty = (mug_y - 18 * nib_scale) + math.sin(rad_p) * 44.0 * nib_scale
        cv2.line(canvas, (int(mug_x + p_ang * 0.35 * nib_scale), int(mug_y - 22 * nib_scale)), (int(p_tx), int(p_ty)), bg_line_col, max(1, int(2.4 * s_ui)), cv2.LINE_AA)
        cv2.circle(canvas, (int(p_tx), int(p_ty)), max(1, int(2.0 * s_ui)), BGR_ACCENT_RED if (p_ang < 0 and is_colored) else bg_line_col, -1, cv2.LINE_AA)

    # 7. Right Desk Corner Prop: Faceted Glass `INDIA INK` Bottle & Open Cap (`nib_inking_keys.jpg` & `apparatus_lightbox_anim_keys.jpg`)
    # Positioned at `ink_x = nib_ox + 302 * scale, ink_y = nib_oy - 262 * scale` so Nib's dip at `K07` (`rh_x=292, rh_y=-258`) lands right in the bottle!
    ink_x = nib_ox + 302.0 * nib_scale
    ink_y = nib_oy - 262.0 * nib_scale
    # Bottle neck & rim
    neck_pts = [
        (ink_x - 9 * nib_scale, ink_y - 24 * nib_scale),
        (ink_x + 9 * nib_scale, ink_y - 24 * nib_scale),
        (ink_x + 11 * nib_scale, ink_y - 12 * nib_scale),
        (ink_x - 11 * nib_scale, ink_y - 12 * nib_scale),
    ]
    # Faceted square glass bottle body
    bot_pts = [
        (ink_x - 20 * nib_scale, ink_y - 12 * nib_scale),
        (ink_x + 20 * nib_scale, ink_y - 12 * nib_scale),
        (ink_x + 22 * nib_scale, ink_y + 20 * nib_scale),
        (ink_x - 22 * nib_scale, ink_y + 20 * nib_scale),
    ]
    draw_ink_stroke(canvas, bot_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=80)
    # Dark India ink reservoir inside lower 65% of glass bottle
    ink_res = np.array([
        [int(ink_x - 17 * nib_scale), int(ink_y - 2 * nib_scale)],
        [int(ink_x + 17 * nib_scale), int(ink_y - 2 * nib_scale)],
        [int(ink_x + 19 * nib_scale), int(ink_y + 18 * nib_scale)],
        [int(ink_x - 19 * nib_scale), int(ink_y + 18 * nib_scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [ink_res], BGR_NONPHOTO_BLUE if is_rough else BGR_INK, cv2.LINE_AA)
    # White bottle front label reading `INK`
    cv2.rectangle(
        canvas,
        (int(ink_x - 13 * nib_scale), int(ink_y + 1 * nib_scale)),
        (int(ink_x + 13 * nib_scale), int(ink_y + 13 * nib_scale)),
        BGR_PAPER, -1,
    )
    cv2.putText(
        canvas, "INK",
        (int(ink_x - 11 * nib_scale), int(ink_y + 10 * nib_scale)),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.25, 0.30 * nib_scale), bg_line_col, max(1, int(round(s_ui))), cv2.LINE_AA,
    )
    draw_ink_stroke(canvas, neck_pts, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=81)
    # Small ribbed screw cap resting beside the bottle (`apparatus_lightbox_anim_keys.jpg`)
    draw_sculpted_oval(canvas, ink_x + 32 * nib_scale, ink_y + 14 * nib_scale, 8 * nib_scale, 5 * nib_scale, color_bgr=bg_line_col, thick=bg_line_w * 0.9, fill_bgr=bg_line_col, boil_phase=boil_phase, seed=82)

    # Flying India ink droplet & concentric ripple rings during `K07..K08` (`ink_drop > 0.05`)
    if pose["ink_drop"] > 0.05:
        u_dr = float(pose["ink_drop"])
        dr_x = ink_x - 24.0 * nib_scale * u_dr
        dr_y = (ink_y - 24 * nib_scale) - 38.0 * nib_scale * math.sin(math.pi * u_dr)
        cv2.circle(canvas, (int(dr_x), int(dr_y)), max(2, int(4.5 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(dr_x + 7 * nib_scale), int(dr_y - 6 * nib_scale)), max(1, int(2.5 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)

    # 8. Articulated Double-Strut Swing-Arm Architect Lamp Clamped to Desk (`apparatus_lightbox_anim_keys.jpg`)
    clamp_x = desk_cx + 95.0 * nib_scale
    clamp_y = desk_fy - 340.0 * nib_scale
    lamp_sway = 6.0 * nib_scale * math.sin(frame_idx * 0.08)
    elbow_x = clamp_x + 42.0 * nib_scale + lamp_sway * 0.4
    elbow_y = clamp_y - 62.0 * nib_scale
    head_lx = clamp_x - 42.0 * nib_scale + lamp_sway
    head_ly = clamp_y - 96.0 * nib_scale

    # Warm translucent light cone in `is_colored`
    if is_colored:
        cone_ov = canvas.copy()
        cone_poly = np.array([
            [int(head_lx - 28 * nib_scale), int(head_ly + 18 * nib_scale)],
            [int(head_lx + 28 * nib_scale), int(head_ly + 18 * nib_scale)],
            [int(disc_x + 135 * nib_scale), int(disc_y + 55 * nib_scale)],
            [int(disc_x - 135 * nib_scale), int(disc_y + 55 * nib_scale)],
        ], dtype=np.int32)
        cv2.fillPoly(cone_ov, [cone_poly], (218, 248, 255), cv2.LINE_AA)
        cv2.addWeighted(cone_ov, 0.22, canvas, 0.78, 0, canvas)

    # Double-strut lower arm & upper arm
    for st_off in (-4.0, 4.0):
        cv2.line(canvas, (int(clamp_x + st_off * nib_scale), int(clamp_y)), (int(elbow_x + st_off * nib_scale), int(elbow_y)), bg_line_col, max(1, int(2.0 * s_ui)), cv2.LINE_AA)
        cv2.line(canvas, (int(elbow_x), int(elbow_y + st_off * nib_scale)), (int(head_lx), int(head_ly + st_off * nib_scale)), bg_line_col, max(1, int(2.0 * s_ui)), cv2.LINE_AA)
    cv2.circle(canvas, (int(clamp_x), int(clamp_y)), max(3, int(6 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(elbow_x), int(elbow_y)), max(3, int(7 * nib_scale)), brass_fill if is_colored else BGR_PAPER, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(elbow_x), int(elbow_y)), max(3, int(7 * nib_scale)), bg_line_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)

    # Domed architect lamp bell shade angled toward the animation disc
    shade_pts = [
        (head_lx - 42 * nib_scale, head_ly + 22 * nib_scale),
        (head_lx - 18 * nib_scale, head_ly - 12 * nib_scale),
        (head_lx + 18 * nib_scale, head_ly - 12 * nib_scale),
        (head_lx + 42 * nib_scale, head_ly + 22 * nib_scale),
    ]
    draw_ink_stroke(
        canvas, shade_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15,
        closed=True, fill_bgr=(92, 168, 118) if is_colored else paper_fill,
        boil_phase=boil_phase, seed=55,
    )
    # Glowing bulb & classic radiating light-ray ticks (`\ | /`) under lamp shade (`apparatus_lightbox_anim_keys.jpg`)
    cv2.ellipse(canvas, (int(head_lx), int(head_ly + 22 * nib_scale)), (int(14 * nib_scale), int(7 * nib_scale)), 0, 0, 180, (190, 245, 255) if is_colored else BGR_PAPER, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(head_lx), int(head_ly + 22 * nib_scale)), (int(14 * nib_scale), int(7 * nib_scale)), 0, 0, 180, bg_line_col, max(1, int(1.6 * s_ui)), cv2.LINE_AA)
    for r_ang in (-32, -16, 0, 16, 32):
        rad_l = math.radians(90 + r_ang)
        rx0_l = head_lx + math.cos(rad_l) * 26.0 * nib_scale
        ry0_l = head_ly + 22 * nib_scale + math.sin(rad_l) * 18.0 * nib_scale
        rx1_l = head_lx + math.cos(rad_l) * 42.0 * nib_scale
        ry1_l = head_ly + 22 * nib_scale + math.sin(rad_l) * 32.0 * nib_scale
        cv2.line(canvas, (int(rx0_l), int(ry0_l)), (int(rx1_l), int(ry1_l)), bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA)

    # 9. Rule XIII: Draw Hand Trajectory Motion Arc & Spacing Chart Ticks in Rough/Onion-Skin!
    if is_rough or enable_onion_skin:
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
            cv2.circle(canvas, (int(arc_pts[k_s][0]), int(arc_pts[k_s][1])), max(2, int(2.5 * s_ui)), BGR_ACCENT_RED if k_s in (0, 6, 12) else BGR_NONPHOTO_BLUE, -1, cv2.LINE_AA)

    # 10. Rule XII: FLIP THE DRAWINGS (`previous -> current -> next` Onion-Skin Ghosts!)
    if enable_onion_skin or is_rough:
        pose_prev = interpolate_nib_pose(max(0.0, f_timeline - 4.5))
        pose_next = interpolate_nib_pose(min(287.0, f_timeline + 4.5))
        ov_p = canvas.copy()
        draw_nib_character_pose(ov_p, pose_prev, nib_ox, nib_oy, nib_scale, stage_mode="rough_blue", boil_phase=boil_phase - 1, ink_override=BGR_ONION_PREV)
        cv2.addWeighted(ov_p, 0.30, canvas, 0.70, 0, canvas)

        ov_n = canvas.copy()
        draw_nib_character_pose(ov_n, pose_next, nib_ox, nib_oy, nib_scale, stage_mode="rough_blue", boil_phase=boil_phase + 1, ink_override=BGR_ONION_NEXT)
        cv2.addWeighted(ov_n, 0.25, canvas, 0.75, 0, canvas)

    # 11. Draw Primary Character `NIB`!
    info = draw_nib_character_pose(
        canvas, pose, nib_ox, nib_oy, nib_scale,
        stage_mode=stage_mode, boil_phase=boil_phase,
    )

    # 12. If `cel_lift > 0.05` (`K11..K12`), draw the Finished Transparent Animation Cel held out in front of Nib!
    if c_lift > 0.05:
        lh_w = info["lh_wrist"]
        rh_w = info["rh_wrist"]
        cel_cx = max(lh_w[0], rh_w[0]) + 46.0 * nib_scale
        cel_cy = 0.5 * (lh_w[1] + rh_w[1]) + 14.0 * nib_scale
        cw_l, ch_l = 62.0 * nib_scale, 46.0 * nib_scale
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
            cv2.addWeighted(cel_ov, 0.72, canvas, 0.28, 0, canvas)
        draw_ink_stroke(
            canvas, cel_poly, color_bgr=BGR_INK, base_thick=bg_line_w * 1.15,
            closed=True, fill_bgr=BGR_PAPER if is_cleanup else None,
            boil_phase=boil_phase, seed=85, sharp_corners=True,
        )
        # 3 Acme registration peg holes along the top edge of the lifted cel (`o | o`)
        for ph_u in (-0.55, 0.0, 0.55):
            ph_x = cel_cx + ph_u * cw_l
            ph_y = cel_cy - ch_l + 6 * nib_scale - ph_u * 4 * nib_scale
            cv2.circle(canvas, (int(ph_x), int(ph_y)), max(2, int(3.0 * nib_scale)), BGR_INK, max(1, int(1.5 * s_ui)), cv2.LINE_AA)
        # Hand-inked Leaping Fox character on the lifted cel (arched spine, bushy tail, head & pointed ears)!
        fox_cel_body = [
            (cel_cx - cw_l * 0.45, cel_cy + ch_l * 0.18),
            (cel_cx - cw_l * 0.05, cel_cy - ch_l * 0.35),
            (cel_cx + cw_l * 0.42, cel_cy - ch_l * 0.12),
            (cel_cx + cw_l * 0.05, cel_cy + ch_l * 0.05),
        ]
        fox_fill = (68, 128, 232) if is_colored else None
        draw_ink_stroke(canvas, fox_cel_body, color_bgr=BGR_INK, base_thick=max(1.8, 2.5 * s_ui), closed=True, fill_bgr=fox_fill, boil_phase=boil_phase, seed=86, sharp_corners=False)
        # Bushy tail & pointed ears on lifted cel fox
        fox_tail = [
            (cel_cx - cw_l * 0.45, cel_cy + ch_l * 0.18),
            (cel_cx - cw_l * 0.72, cel_cy - ch_l * 0.05),
            (cel_cx - cw_l * 0.55, cel_cy + ch_l * 0.36),
        ]
        draw_ink_stroke(canvas, fox_tail, color_bgr=BGR_INK, base_thick=max(1.6, 2.2 * s_ui), closed=True, fill_bgr=fox_fill, boil_phase=boil_phase, seed=87, sharp_corners=False)
        fx_hx = int(cel_cx + cw_l * 0.42)
        fx_hy = int(cel_cy - ch_l * 0.18)
        if fox_fill is not None:
            cv2.circle(canvas, (fx_hx, fx_hy), max(3, int(9 * nib_scale)), fox_fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, (fx_hx, fx_hy), max(3, int(9 * nib_scale)), BGR_INK, max(1, int(2.0 * s_ui)), cv2.LINE_AA)
        cv2.line(canvas, (fx_hx - int(3 * nib_scale), fx_hy - int(8 * nib_scale)), (fx_hx + int(2 * nib_scale), fx_hy - int(18 * nib_scale)), BGR_INK, max(1, int(2.0 * s_ui)), cv2.LINE_AA)
        cv2.line(canvas, (fx_hx + int(2 * nib_scale), fx_hy - int(18 * nib_scale)), (fx_hx + int(6 * nib_scale), fx_hy - int(7 * nib_scale)), BGR_INK, max(1, int(2.0 * s_ui)), cv2.LINE_AA)

    # ==========================================================================
    # PLANE 1 (`depth = 0.62`): FOREGROUND 35MM FILM CANS, UNSPOOLING FILMSTRIP,
    # AND ARTIST'S GOUACHE PALETTE & JARS IN PARALLAX
    # (Matches `multiplane_plane1_foreground_props_keys.jpg`)
    # ==========================================================================
    fg_lx, fg_ly, z_fg = _plane_tf(out_w * 0.09, out_h * 0.92, 0.62)
    fg_rx, fg_ry, _ = _plane_tf(out_w * 0.90, out_h * 0.92, 0.62)

    # Left Foreground: Stacked 35mm metal film cans & curling sprocketed filmstrip
    for can_i in range(2):
        cy_can = fg_ly - can_i * 22.0 * s_ui * z_fg
        cx_can = fg_lx + can_i * 8.0 * s_ui * z_fg
        can_body = [
            (cx_can - 48 * s_ui * z_fg, cy_can - 10 * s_ui * z_fg),
            (cx_can + 48 * s_ui * z_fg, cy_can - 10 * s_ui * z_fg),
            (cx_can + 48 * s_ui * z_fg, cy_can + 10 * s_ui * z_fg),
            (cx_can - 48 * s_ui * z_fg, cy_can + 10 * s_ui * z_fg),
        ]
        draw_ink_stroke(canvas, can_body, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1, closed=True, fill_bgr=(168, 178, 188) if is_colored else shade_fill, boil_phase=boil_phase, seed=88 + can_i)
        draw_sculpted_oval(canvas, cx_can, cy_can - 10 * s_ui * z_fg, 48 * s_ui * z_fg, 12 * s_ui * z_fg, color_bgr=bg_line_col, thick=bg_line_w, fill_bgr=(198, 206, 214) if is_colored else paper_fill, boil_phase=boil_phase, seed=90 + can_i)

    # Unspooling 35mm filmstrip ribbon curling to the right of the film cans
    strip_pts = [
        (fg_lx + 45 * s_ui * z_fg, fg_ly - 6 * s_ui * z_fg),
        (fg_lx + 92 * s_ui * z_fg, fg_ly - 24 * s_ui * z_fg),
        (fg_lx + 138 * s_ui * z_fg, fg_ly - 4 * s_ui * z_fg),
        (fg_lx + 178 * s_ui * z_fg, fg_ly - 18 * s_ui * z_fg),
    ]
    draw_ink_stroke(canvas, strip_pts, color_bgr=bg_line_col, base_thick=max(2.0, 9.0 * s_ui * z_fg), boil_phase=boil_phase, seed=92)

    # Right Foreground: Kidney-shaped gouache palette + paint jars with brushes
    draw_sculpted_oval(
        canvas, fg_rx - 52 * s_ui * z_fg, fg_ry + 4 * s_ui * z_fg,
        54 * s_ui * z_fg, 22 * s_ui * z_fg, angle_deg=-8,
        color_bgr=bg_line_col, thick=bg_line_w * 1.15,
        fill_bgr=(148, 188, 222) if is_colored else paper_fill,
        boil_phase=boil_phase, seed=93,
    )
    for fi_p, (jx, jy, j_col) in enumerate([
        (fg_rx + 18 * s_ui * z_fg, fg_ry - 8 * s_ui * z_fg, (52, 78, 198)),
        (fg_rx + 68 * s_ui * z_fg, fg_ry + 4 * s_ui * z_fg, (68, 168, 92)),
    ]):
        draw_sculpted_oval(
            canvas, jx, jy, 22 * s_ui * z_fg, 30 * s_ui * z_fg,
            color_bgr=bg_line_col, thick=bg_line_w * 1.15,
            fill_bgr=j_col if is_colored else shade_fill,
            boil_phase=boil_phase, seed=94 + fi_p,
        )
        # Paintbrush handle sticking out of jar
        cv2.line(canvas, (int(jx - 4 * s_ui), int(jy - 24 * s_ui * z_fg)), (int(jx + (14 if fi_p == 0 else -14) * s_ui * z_fg), int(jy - 64 * s_ui * z_fg)), bg_line_col, max(2, int(3.0 * s_ui)), cv2.LINE_AA)

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


def _render_closeup_flipbook_cell(
    frame_idx: int,
    total_frames: int,
    cell_w: int,
    cell_h: int,
    stage_mode: str,
    enable_onion_skin: bool,
) -> np.ndarray:
    """
    Renders a high-resolution `1280x720` frame and frames a hero medium-closeup
    around Nib, his 3-pointed crown hat, his 3D backlit drafting desk, the India ink
    bottle, and the 4:3 pencil-test monitor (`nib_inking_keys.jpg` framing!),
    with a crisp hand-lettered keyframe badge in the top-left corner.
    """
    hi_w, hi_h = 1280, 720
    full_im = render_nib_multiplane_frame(
        frame_idx, total_frames, out_w=hi_w, out_h=hi_h,
        stage_mode=stage_mode, enable_onion_skin=enable_onion_skin,
    )
    # Crop the hero studio region (`x: 20%..94%`, `y: 4%..93%`) so Nib, his lamp & his desk fill the panel!
    x0, x1 = int(hi_w * 0.20), int(hi_w * 0.94)
    y0, y1 = int(hi_h * 0.04), int(hi_h * 0.93)
    crop = full_im[y0:y1, x0:x1]
    cell = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

    # Stamp a crisp hand-lettered keyframe badge in the top-left corner (`nib_inking_keys.jpg` style)
    f_timeline = (frame_idx / float(max(1, total_frames - 1))) * 287.0
    pose = interpolate_nib_pose(f_timeline)
    badge_txt = f"{pose['pose_tag']}  {pose['pose_title'][:26]}"
    font_s = max(0.35, cell_w / 1080.0)
    font_t = max(1, int(round(cell_w / 440.0)))
    (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, font_s, font_t)
    bh = max(20, th + 10)
    bw = min(cell_w - 12, tw + 16)
    cv2.rectangle(cell, (6, 6), (6 + bw, 6 + bh), BGR_PAPER, -1)
    cv2.rectangle(cell, (6, 6), (6 + bw, 6 + bh), BGR_NONPHOTO_BLUE if stage_mode == "rough_blue" else BGR_INK, 2)
    cv2.putText(
        cell, badge_txt,
        (13, 6 + bh - 6),
        cv2.FONT_HERSHEY_SIMPLEX, font_s,
        BGR_NONPHOTO_BLUE if stage_mode == "rough_blue" else BGR_INK,
        font_t, cv2.LINE_AA,
    )
    cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), BGR_INK, 2)
    return cell


def build_nib_contact_sheets_and_film():
    """
    1. Renders the 16-Key-Pose Contact Sheet (`nib_16_key_poses_sheet.png`) across `K01..K16`
       so we can inspect every key pose (`Rule VII: KEY DRAWINGS FIRST`).
    2. Renders the 3-Pass Progressive Flipbook Sheet (`nib_3pass_flipbook_comparison.png`)
       comparing `Rough Blue & Onion-Skin Flip`, `Speedball Cleanup Ink`, and
       `7-Plane Gouache Multiplane Composite`.
    3. Renders the full 24fps Hand-Drawn Character Film (`nib_solo_hand_drawn_cartoon.mp4`
       & `.gif`, `864` total frames = `288` Rough Blue & Onion Flip + `288` Speedball Ink
       Cleanup + `288` Full 7-Plane Gouache Multiplane Composite @ `24 fps` = `36.0s`!).
    """
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    BTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 16 Key Poses Contact Sheet (4x4 grid of `K01..K16` across Rough Blue, Cleanup Ink, and Gouache)
    print("[Nib Solo Cartoon] Rendering 16 Key Poses Contact Sheet (K01..K16)...")
    key_rows = []
    for r in range(4):
        row_cells = []
        for c in range(4):
            k_idx = r * 4 + c
            kp = NIB_KEY_POSES[k_idx]
            f_i = int(kp["frame"])
            mode = "rough_blue" if r == 0 else ("cleanup_ink" if r == 1 else "multiplane_composite")
            cell = _render_closeup_flipbook_cell(f_i, 288, cell_w=480, cell_h=270, stage_mode=mode, enable_onion_skin=(r == 0))
            row_cells.append(cell)
        key_rows.append(np.hstack(row_cells))
    keys_grid = np.vstack(key_rows)
    keys_path = PREVIEW_DIR / "nib_16_key_poses_sheet.png"
    cv2.imwrite(str(keys_path), keys_grid)
    shutil.copy2(keys_path, ARTIFACT_DIR / "nib_16_key_poses_sheet.png")

    # 2. 3-Pass Breakdown & In-Between Flipbook Comparison (`K03 -> B03 -> K04 -> K07 -> K09 -> K10`)
    print("[Nib Solo Cartoon] Rendering 3-Pass Breakdown & In-Between Flipbook Sheet...")
    sample_frames = [36, 45, 54, 126, 146, 166]
    pass_rows = []
    for mode, onion in [("rough_blue", True), ("cleanup_ink", False), ("multiplane_composite", False)]:
        cells = []
        for sf in sample_frames:
            c_im = _render_closeup_flipbook_cell(sf, 288, cell_w=380, cell_h=214, stage_mode=mode, enable_onion_skin=onion)
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
