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


def _capsule_poly(p0: tuple, p1: tuple, r0: float, r1: float, cap_steps: int = 14) -> np.ndarray:
    """
    Constructs a smooth, watertight 2D tapered capsule polygon connecting circle
    `(p0, r0)` and circle `(p1, r1)` with semicircular end-caps.
    Used for Dual-Pass Casing (`Pass 1: r + ink_w` in `ink_col`, `Pass 2: r` in `fill_col`)
    so multi-segment limbs, cuffs, palms, and fingers fuse with ZERO joint slop!
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        ang_arr = np.linspace(0.0, 2.0 * math.pi, cap_steps * 2, endpoint=False)
        rm = max(1.0, 0.5 * (r0 + r1))
        return np.round(np.column_stack([x0 + rm * np.cos(ang_arr), y0 + rm * np.sin(ang_arr)])).astype(np.int32)

    base_ang = math.atan2(dy, dx)
    pts = []
    # Semicircle around p1 from `base_ang - pi/2` to `base_ang + pi/2`
    for i in range(cap_steps + 1):
        a = base_ang - 0.5 * math.pi + (math.pi * i) / cap_steps
        pts.append((x1 + r1 * math.cos(a), y1 + r1 * math.sin(a)))
    # Semicircle around p0 from `base_ang + pi/2` to `base_ang + 3*pi/2`
    for i in range(cap_steps + 1):
        a = base_ang + 0.5 * math.pi + (math.pi * i) / cap_steps
        pts.append((x0 + r0 * math.cos(a), y0 + r0 * math.sin(a)))
    return np.round(np.array(pts, dtype=np.float64)).astype(np.int32)


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
    Draws a hand-inked spline or sharp-cornered architectural polygon with
    clean anti-aliased contours (zero stair-step thickness lumps!) and optional
    watertight gouache fill.
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
    boil_amp = 0.18 if use_sharp else 0.32
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

    w_uniform = max(1, int(round(base_thick)))
    if closed or use_sharp or base_thick <= 2.2:
        cv2.polylines(canvas, [pts_i], closed, color_bgr, w_uniform, cv2.LINE_AA)
        return

    # Open calligraphic stroke: construct a smooth sub-pixel tapered ribbon polygon
    left_pts = []
    right_pts = []
    for i in range(n):
        um = i / float(max(1, n - 1))
        if um < 0.22:
            prof = taper_start + (1.0 - taper_start) * math.sin((um / 0.22) * (math.pi * 0.5))
        elif um > 0.78:
            prof = taper_end + (1.0 - taper_end) * math.sin(((1.0 - um) / 0.22) * (math.pi * 0.5))
        else:
            prof = 1.0 + 0.12 * math.sin(((um - 0.22) / 0.56) * math.pi)
        hw = 0.5 * base_thick * prof
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        tx = pts_f[i1, 0] - pts_f[i0, 0]
        ty = pts_f[i1, 1] - pts_f[i0, 1]
        tl = max(1e-4, math.hypot(tx, ty))
        nx, ny = -ty / tl, tx / tl
        left_pts.append((pts_f[i, 0] + nx * hw, pts_f[i, 1] + ny * hw))
        right_pts.append((pts_f[i, 0] - nx * hw, pts_f[i, 1] - ny * hw))
    ribbon = np.round(np.array(left_pts + right_pts[::-1], dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(canvas, [ribbon], color_bgr, cv2.LINE_AA)
    cv2.polylines(canvas, [pts_i], False, color_bgr, max(1, int(round(base_thick * 0.75))), cv2.LINE_AA)


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
    num_pts: int = 24,
):
    irx, iry = max(1, int(round(rx))), max(1, int(round(ry)))
    icx, icy = int(round(cx)), int(round(cy))
    if fill_bgr is not None:
        cv2.ellipse(canvas, (icx, icy), (irx, iry), float(angle_deg), 0, 360, fill_bgr, -1, cv2.LINE_AA)
    if thick > 0:
        cv2.ellipse(canvas, (icx, icy), (irx, iry), float(angle_deg), 0, 360, color_bgr, max(1, int(round(thick))), cv2.LINE_AA)


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
        "rh_x": 106.0, "rh_y": -382.0, "rh_pose": "chin_think", "pen_angle": -65.0,
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
        "lh_x": -64.0, "lh_y": -348.0, "lh_pose": "open_gesture",
        "rh_x": 116.0, "rh_y": -445.0, "rh_pose": "idea_point", "pen_angle": -82.0,
        "l_foot_dx": -34.0, "l_foot_dy": 0.0, "l_heel_lift": 0.0,
        "r_foot_dx": 30.0, "r_foot_dy": 0.0, "r_heel_lift": 6.0,
        "brow_l": 14.0, "brow_r": 16.0, "eye_open": 1.25, "look_dx": 0.6, "look_dy": -0.8,
        "mouth": "oh_gasp", "hair_lag": -16.0, "apron_lag": -10.0,
        "paper_flip": 0.05, "draw_prog": 0.08, "cel_lift": 0.0, "ink_drop": 0.0,
        "cam_zoom": 1.02, "cam_pan_x": 8.0, "cam_pan_y": -6.0,
    },
    # K03: Deep Anticipation Wind-Up (Coils down & winds pen high to the right before striking paper!)
    {
        "code": "K03", "title": "ANTICIPATION COIL & WIND-UP", "frame": 36,
        "hip_dx": -28.0, "hip_dy": -196.0, "pelvis_tilt": -10.0,
        "spine_bend": 28.0, "chest_squash": 1.11,
        "head_pitch": 22.0, "head_yaw": 0.90, "head_squash": 1.08,
        "lh_x": 136.0, "lh_y": -302.0, "lh_pose": "pin_paper",
        "rh_x": 128.0, "rh_y": -428.0, "rh_pose": "grip_pen", "pen_angle": -68.0,
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
        "lh_x": -58.0, "lh_y": -285.0, "lh_pose": "open_gesture",
        "rh_x": 114.0, "rh_y": -255.0, "rh_pose": "grip_pen", "pen_angle": -55.0,
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
        "lh_x": -82.0, "lh_y": -495.0, "lh_pose": "cheer_high",
        "rh_x": 118.0, "rh_y": -512.0, "rh_pose": "cheer_high", "pen_angle": -95.0,
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
        "lh_x": -56.0, "lh_y": -295.0, "lh_pose": "open_gesture",
        "rh_x": 112.0, "rh_y": -310.0, "rh_pose": "grip_pen", "pen_angle": -35.0,
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
        "lh_x": -64.0, "lh_y": -385.0, "lh_pose": "tip_visor",
        "rh_x": 122.0, "rh_y": -365.0, "rh_pose": "wave_pen", "pen_angle": -48.0,
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
# III. WATERTIGHT DUAL-PASS CASING CHARACTER RENDERER FOR `NIB`
#      (Zero Joint Slop: Seamless Shoulders, Puffy Rolled Cuffs, Unified Hands,
#       Continuous Trouser Tubes, Sculpted Flat-Soled Shoes & Pointed W-Collar)
# ==============================================================================

def _draw_seamless_arm_and_hand(
    canvas: np.ndarray,
    sh_pt: tuple,
    el_pt: tuple,
    wr_pt: tuple,
    w_up: float,
    w_low: float,
    scale: float,
    hand_pose: str,
    is_right: bool,
    is_near_shoulder: bool,
    shirt_fill: tuple,
    skin_fill: tuple,
    ink_col: tuple,
    ink_w: float,
    pen_angle_deg: float = -45.0,
    has_ink_pen: bool = True,
) -> tuple:
    """
    Draws a 100% seamless cartoon arm (`Upper Sleeve -> Puffy Rolled Elbow Cuff ->
    Bare Forearm -> Sculpted 4-Fingered Hand + Diamond Fountain-Nib Dip Pen`) using
    Dual-Pass Capsule Union (`Pass 1: Outer Ink Casing -> Pass 2: Opaque Interior Fill
    -> Pass 3: Interior Overlap Creases`) so there are ZERO intersecting joint rings
    at the shoulder, elbow, wrist, palm, or knuckles (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`).
    """
    w_i = max(2, int(round(ink_w)))
    f_ang = math.atan2(wr_pt[1] - el_pt[1], wr_pt[0] - el_pt[0])
    fca, fsa = math.cos(f_ang), math.sin(f_ang)
    fnx, fny = -fsa, fca

    u_ang = math.atan2(el_pt[1] - sh_pt[1], el_pt[0] - sh_pt[0])
    uca, usa = math.cos(u_ang), math.sin(u_ang)
    unx, uny = -usa, uca

    palm_cx = wr_pt[0] + fca * 12.5 * scale
    palm_cy = wr_pt[1] + fsa * 12.5 * scale
    pen_tip = (palm_cx + 22.0 * scale, palm_cy + 12.0 * scale)

    # Rough Pass 1 (`shirt_fill is None`): clean non-photo blue construction skeleton + volume tubes
    if shirt_fill is None or skin_fill is None:
        cv2.polylines(canvas, [_capsule_poly(sh_pt, el_pt, w_up, w_up * 0.88)], True, ink_col, max(1, w_i - 1), cv2.LINE_AA)
        cv2.polylines(canvas, [_capsule_poly(el_pt, (palm_cx, palm_cy), w_low, w_low * 0.78)], True, ink_col, max(1, w_i - 1), cv2.LINE_AA)
        cv2.circle(canvas, (int(sh_pt[0]), int(sh_pt[1])), max(2, int(4 * scale)), ink_col, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(el_pt[0]), int(el_pt[1])), max(2, int(5 * scale)), ink_col, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(palm_cx), int(palm_cy)), max(3, int(11 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)
        if is_right:
            p_rad = math.radians(pen_angle_deg)
            pca, psa = math.cos(p_rad), math.sin(p_rad)
            grip_x, grip_y = palm_cx + 7.0 * scale * fca, palm_cy + 7.0 * scale * fsa
            tip_x, tip_y = grip_x + pca * 64.0 * scale, grip_y + psa * 64.0 * scale
            pen_tip = (tip_x, tip_y)
            cv2.line(canvas, (int(grip_x - pca * 48 * scale), int(grip_y - psa * 48 * scale)), (int(tip_x), int(tip_y)), ink_col, max(2, w_i), cv2.LINE_AA)
        return pen_tip

    # ==========================================================================
    # 1. UPPER SLEEVE TUBE (`sh_pt -> el_pt`)
    # ==========================================================================
    up_outer = _capsule_poly(sh_pt, el_pt, w_up + w_i, w_up * 0.90 + w_i)
    up_inner = _capsule_poly(sh_pt, el_pt, w_up, w_up * 0.90)
    cv2.fillPoly(canvas, [up_outer], ink_col, cv2.LINE_AA)
    cv2.fillPoly(canvas, [up_inner], shirt_fill, cv2.LINE_AA)

    # Interior-only white shirt shoulder bridge (radius < `w_up`, shifted toward chest so
    # the back/top/bottom outer black shoulder contour of `up_outer` stays 100% intact!)
    if is_near_shoulder:
        bridge_pt = (sh_pt[0] + 6.5 * scale, sh_pt[1] - 3.5 * scale)
        cv2.circle(canvas, (int(bridge_pt[0]), int(bridge_pt[1])), int(w_up * 0.76), shirt_fill, -1, cv2.LINE_AA)
        # Canonical Armpit Crease Line (`/` in `nib_inking_keys.jpg` Q1-Q4) where inner sleeve meets left apron strap
        ax0 = int(sh_pt[0] + w_up * 0.55)
        ay0 = int(sh_pt[1] + w_up * 0.85)
        cv2.line(canvas, (ax0, ay0), (int(ax0 + 5.0 * scale), int(ay0 - 11.0 * scale)), ink_col, max(2, w_i - 1), cv2.LINE_AA)

    # ==========================================================================
    # 2. PROXIMAL FOREARM TUBE (`el_pt -> wr_pt`)
    #    Drawn BEFORE the Puffy Rolled-Up Sleeve Cuff so the cuff overlaps the
    #    forearm root cleanly with zero semicircular cap inside the cuff!
    # ==========================================================================
    r_wrist = w_low * 0.74
    cv2.fillPoly(canvas, [_capsule_poly(el_pt, wr_pt, w_low + w_i, r_wrist + w_i)], ink_col, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(el_pt, wr_pt, w_low, r_wrist)], skin_fill, cv2.LINE_AA)

    # ==========================================================================
    # 3. PUFFY ROLLED-UP SLEEVE CUFF AT ELBOW & CLOTH FOLD TICKS (`nib_inking_keys.jpg` Q1-Q4)
    #    Wraps cleanly over the `sh_pt -> el_pt -> wr_pt` elbow joint!
    # ==========================================================================
    # 2 short curved cloth-bunching fold ticks inside the elbow bend just above the rolled cuff (`nib_inking_keys.jpg`)
    fold_cx = el_pt[0] - uca * 10.0 * scale
    fold_cy = el_pt[1] - usa * 10.0 * scale
    cv2.line(canvas, (int(fold_cx - unx * 3.0 * scale), int(fold_cy - uny * 3.0 * scale)), (int(fold_cx + unx * 6.5 * scale - uca * 3.0 * scale), int(fold_cy + uny * 6.5 * scale - usa * 3.0 * scale)), ink_col, max(2, w_i - 1), cv2.LINE_AA)
    cv2.line(canvas, (int(fold_cx - uca * 5.0 * scale), int(fold_cy - usa * 5.0 * scale)), (int(fold_cx + unx * 5.0 * scale - uca * 7.5 * scale), int(fold_cy + uny * 5.0 * scale - usa * 7.5 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)

    cuff_cx = el_pt[0] - uca * 1.5 * scale + fca * 1.5 * scale
    cuff_cy = el_pt[1] - usa * 1.5 * scale + fsa * 1.5 * scale
    cuff_hw = w_up * 0.90
    c0 = (cuff_cx - unx * cuff_hw, cuff_cy - uny * cuff_hw)
    c1 = (cuff_cx + unx * cuff_hw, cuff_cy + uny * cuff_hw)
    r_cuff = 6.5 * scale
    cv2.fillPoly(canvas, [_capsule_poly(c0, c1, r_cuff + w_i, r_cuff + w_i)], ink_col, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(c0, c1, r_cuff, r_cuff)], shirt_fill, cv2.LINE_AA)

    # ==========================================================================
    # 4. BACK SHAFT OF FOUNTAIN-NIB DIP PEN (Behind Palm/Fingers when `is_right`)
    # ==========================================================================
    if is_right:
        p_rad = math.radians(pen_angle_deg)
        pca, psa = math.cos(p_rad), math.sin(p_rad)
        pnx, pny = -psa, pca
        grip_x = palm_cx + 8.0 * scale * fca
        grip_y = palm_cy + 8.0 * scale * fsa
        back_x = grip_x - pca * 54.0 * scale
        back_y = grip_y - psa * 54.0 * scale
        collar_x = grip_x + pca * 22.0 * scale
        collar_y = grip_y + psa * 22.0 * scale
        tip_x = grip_x + pca * 64.0 * scale
        tip_y = grip_y + psa * 64.0 * scale
        pen_tip = (tip_x, tip_y)

        pw = 5.0 * scale
        s_fill = (255, 255, 255) if skin_fill == (255, 255, 255) else ((62, 82, 184) if has_ink_pen else BGR_NONPHOTO_BLUE)
        cv2.fillPoly(canvas, [_capsule_poly((back_x, back_y), (collar_x, collar_y), pw * 0.78 + w_i, pw + w_i)], ink_col, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly((back_x, back_y), (collar_x, collar_y), pw * 0.78, pw)], s_fill, cv2.LINE_AA)

    # ==========================================================================
    # 5. SEAMLESS WRIST + PALM + 3 FINGERS + THUMB (Dual-Pass Union!)
    # ==========================================================================
    mid_forearm = (0.48 * el_pt[0] + 0.52 * wr_pt[0], 0.48 * el_pt[1] + 0.52 * wr_pt[1])
    r_mid = 0.48 * w_low + 0.52 * r_wrist
    rx_palm, ry_palm = 14.5 * scale, 12.0 * scale
    p_deg = math.degrees(f_ang)

    if hand_pose in ("open_gesture", "cheer_high", "wave_pen", "flourish"):
        f_defs = [
            (-0.24, 21.0, -5.5),
            (0.06, 22.0, 0.0),
            (0.34, 19.0, 5.5),
        ]
        th_ang, th_len, th_off = -0.78, 17.0, -8.5
    elif hand_pose in ("idea_point",):
        f_defs = [
            (-0.08, 25.0, -4.5),
            (0.22, 14.0, 1.0),
            (0.42, 12.0, 5.5),
        ]
        th_ang, th_len, th_off = -0.65, 15.0, -7.5
    else:
        f_defs = [
            (-0.12, 19.5, -4.5),
            (0.14, 17.5, 0.5),
            (0.36, 15.0, 5.0),
        ]
        th_ang, th_len, th_off = -0.58, 16.5, -7.5

    finger_capsules = []
    for sp_a, flen, n_off in f_defs:
        fa = f_ang + sp_a
        fb_x = palm_cx + math.cos(fa) * 5.0 * scale + fnx * n_off * scale
        fb_y = palm_cy + math.sin(fa) * 5.0 * scale + fny * n_off * scale
        ft_x = fb_x + math.cos(fa) * (flen * 0.68 * scale)
        ft_y = fb_y + math.sin(fa) * (flen * 0.68 * scale)
        finger_capsules.append(((fb_x, fb_y), (ft_x, ft_y), 5.0 * scale, 4.4 * scale))

    tha = f_ang + th_ang
    th_bx = palm_cx + math.cos(tha) * 3.0 * scale + fnx * th_off * scale
    th_by = palm_cy + math.sin(tha) * 3.0 * scale + fny * th_off * scale
    th_tx = th_bx + math.cos(tha) * (th_len * 0.68 * scale)
    th_ty = th_by + math.sin(tha) * (th_len * 0.68 * scale)
    thumb_cap = ((th_bx, th_by), (th_tx, th_ty), 5.2 * scale, 4.5 * scale)

    # PASS 1 (Outer Ink Casing for Palm + Fingers + Thumb)
    cv2.ellipse(canvas, (int(palm_cx), int(palm_cy)), (int(rx_palm + w_i), int(ry_palm + w_i)), p_deg, 0, 360, ink_col, -1, cv2.LINE_AA)
    for fb, ft, rf0, rf1 in finger_capsules:
        cv2.fillPoly(canvas, [_capsule_poly(fb, ft, rf0 + w_i, rf1 + w_i)], ink_col, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2] + w_i, thumb_cap[3] + w_i)], ink_col, cv2.LINE_AA)

    # PASS 2 (Opaque Interior Skin Fill — Fuses Distal Forearm, Wrist, Palm & Fingers with ZERO internal seams!)
    cv2.fillPoly(canvas, [_capsule_poly(mid_forearm, wr_pt, r_mid, r_wrist)], skin_fill, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(palm_cx), int(palm_cy)), (int(rx_palm), int(ry_palm)), p_deg, 0, 360, skin_fill, -1, cv2.LINE_AA)
    for fb, ft, rf0, rf1 in finger_capsules:
        cv2.fillPoly(canvas, [_capsule_poly(fb, ft, rf0, rf1)], skin_fill, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2], thumb_cap[3])], skin_fill, cv2.LINE_AA)

    # PASS 3 (Deliberate Finger Cleft Crease Lines + Front Fountain-Nib Dip Pen + Overlapping Thumb Pinch!)
    for ci in range(len(finger_capsules) - 1):
        f0_b, f0_t, _, _ = finger_capsules[ci]
        f1_b, f1_t, _, _ = finger_capsules[ci + 1]
        cleft_x = 0.5 * (f0_t[0] + f1_t[0])
        cleft_y = 0.5 * (f0_t[1] + f1_t[1])
        base_x = 0.5 * (f0_b[0] + f1_b[0]) + fca * 3.0 * scale
        base_y = 0.5 * (f0_b[1] + f1_b[1]) + fsa * 3.0 * scale
        cv2.line(canvas, (int(base_x), int(base_y)), (int(0.55 * base_x + 0.45 * cleft_x), int(0.55 * base_y + 0.45 * cleft_y)), ink_col, max(2, w_i - 1), cv2.LINE_AA)

    if is_right:
        p_rad = math.radians(pen_angle_deg)
        pca, psa = math.cos(p_rad), math.sin(p_rad)
        pnx, pny = -psa, pca
        grip_x = palm_cx + 8.0 * scale * fca
        grip_y = palm_cy + 8.0 * scale * fsa
        collar_x = grip_x + pca * 20.0 * scale
        collar_y = grip_y + psa * 20.0 * scale
        nib_mid_x = grip_x + pca * 40.0 * scale
        nib_mid_y = grip_y + psa * 40.0 * scale
        tip_x, tip_y = pen_tip
        nw = 9.2 * scale

        # Front barrel segment from grip to ferrule collar
        pw = 5.0 * scale
        s_fill = (255, 255, 255) if skin_fill == (255, 255, 255) else ((62, 82, 184) if has_ink_pen else BGR_NONPHOTO_BLUE)
        cv2.fillPoly(canvas, [_capsule_poly((grip_x, grip_y), (collar_x, collar_y), pw + w_i, pw + w_i)], ink_col, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly((grip_x, grip_y), (collar_x, collar_y), pw, pw)], s_fill, cv2.LINE_AA)

        # Diamond Fountain-Nib Head (`agent_nib_keys.jpg` & `nib_inking_keys.jpg`)
        nib_poly = np.array([
            [int(collar_x + pnx * 5.0 * scale), int(collar_y + pny * 5.0 * scale)],
            [int(nib_mid_x + pnx * nw), int(nib_mid_y + pny * nw)],
            [int(tip_x), int(tip_y)],
            [int(nib_mid_x - pnx * nw), int(nib_mid_y - pny * nw)],
            [int(collar_x - pnx * 5.0 * scale), int(collar_y - pny * 5.0 * scale)],
        ], dtype=np.int32)
        n_fill = (255, 255, 255) if skin_fill == (255, 255, 255) else (196, 232, 252)
        cv2.fillPoly(canvas, [nib_poly], n_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [nib_poly], True, ink_col, w_i, cv2.LINE_AA)

        # Nib central slit line + circular breather hole `o`!
        hole_x = int(nib_mid_x - pca * 2.0 * scale)
        hole_y = int(nib_mid_y - psa * 2.0 * scale)
        cv2.line(canvas, (int(tip_x), int(tip_y)), (hole_x, hole_y), ink_col, max(2, w_i - 1), cv2.LINE_AA)
        cv2.circle(canvas, (hole_x, hole_y), max(2, int(3.2 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (hole_x, hole_y), max(2, int(3.2 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)

        # Re-draw Thumb Pinch ON TOP of pen grip so thumb wraps around the pen barrel (`closeup_hand_drawing_keys.jpg`)!
        cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2] + w_i, thumb_cap[3] + w_i)], ink_col, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2], thumb_cap[3])], skin_fill, cv2.LINE_AA)

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
    exact canonical character design, watertight joints, and bold line quality of:
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
    ink_w = max(2.2, (2.2 if is_rough else 4.0) * max(0.65, min(1.45, scale)))
    w_i = max(2, int(round(ink_w)))

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

    # --- 1. CLASSIC 3-HEAD-TALL CARTOON PROPORTIONS (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`) ---
    hip_x = origin_x + pose["hip_dx"] * 0.75 * scale
    hip_y = origin_y + (pose["hip_dy"] * 0.56) * scale
    spine_rad = math.radians(pose["spine_bend"] * 0.72)
    csq = float(pose["chest_squash"])

    torso_len = 114.0 * scale / (csq ** 0.55)
    chest_x = hip_x + torso_len * math.sin(spine_rad)
    chest_y = hip_y - torso_len * math.cos(spine_rad)

    mid_spine_x = 0.5 * (hip_x + chest_x) + 6.0 * scale * math.sin(spine_rad)
    mid_spine_y = 0.5 * (hip_y + chest_y)

    neck_x = chest_x + 6.0 * scale * math.sin(spine_rad * 0.7)
    neck_y = chest_y - 8.0 * scale

    h_sq = float(pose["head_squash"])
    hrx = 66.0 * scale * (h_sq ** 0.45)
    hry = 59.0 * scale / (h_sq ** 0.45)
    head_cx = neck_x + 6.0 * scale
    head_cy = neck_y - 46.0 * scale

    lh_target = (origin_x + pose["lh_x"] * 0.85 * scale, origin_y + (pose["lh_y"] * 0.68) * scale)
    rh_target = (origin_x + pose["rh_x"] * 0.85 * scale, origin_y + (pose["rh_y"] * 0.68) * scale)

    # --- 2. LEFT-ON-SCREEN ARM (Attached at Left Outer Shoulder `chest_x - w_chest * 0.72`,
    #        bending outward-left with forearm below the Square-U bib, matching `nib_inking_keys.jpg` Q2 & `agent_nib_keys.jpg` Q3!) ---
    w_chest = 47.0 * scale * (csq ** 0.45)
    w_waist = 45.0 * scale * (csq ** 0.50)
    w_hip = 48.0 * scale * (csq ** 0.40)
    bib_top_y = chest_y + 34.0 * scale

    sh_lx = chest_x - w_chest * 0.72
    sh_ly = chest_y + 18.0 * scale
    u_arm = 76.0 * scale
    f_arm = 74.0 * scale
    lh_adj_x, lh_adj_y = lh_target
    if lh_adj_x > chest_x - w_chest * 0.45:
        lh_adj_y = max(lh_adj_y, bib_top_y + 26.0 * scale)
        lh_adj_x = min(lh_adj_x, chest_x + w_chest * 0.35)
    el_lx, el_ly, wr_lx, wr_ly = solve_2bone_ik(sh_lx, sh_ly, lh_adj_x, lh_adj_y, 44.0 * scale, 44.0 * scale, prefer_lower_joint=True)
    # Keep left elbow bending outward to the left of the left apron strap and below the bib!
    el_lx = min(el_lx, sh_lx - 14.0 * scale)
    el_ly = max(el_ly, sh_ly + 28.0 * scale)

    # --- 3. SEAMLESS CURVED TROUSER LEGS & SCULPTED FLAT-SOLED ANIMATOR SHOES ---
    # Hip joints are anchored to `0.5 * (hip_x + mid_spine_x)` with 3-head-tall limb lengths
    # (`52 * scale`) so the hip spheres stay 100% inside the flared apron skirt at all spine bends!
    thigh_l = 52.0 * scale
    calf_l = 52.0 * scale
    hip_anchor_x = 0.5 * (hip_x + mid_spine_x)
    for li, (hip_off_x, f_dx, f_dy, heel_lift, p_fill) in enumerate([
        (-8.0 * scale, pose["l_foot_dx"] * 0.9 * scale, pose["l_foot_dy"] * 0.65 * scale, pose["l_heel_lift"] * 0.6 * scale, pants_far),
        (14.0 * scale, pose["r_foot_dx"] * 0.9 * scale, pose["r_foot_dy"] * 0.65 * scale, pose["r_heel_lift"] * 0.6 * scale, pants_fill),
    ]):
        hx_j = hip_anchor_x + hip_off_x
        hy_j = hip_y + 16.0 * scale
        ank_x = origin_x + f_dx
        ank_y = origin_y + f_dy - 14.0 * scale - heel_lift * 0.5
        kx, ky, ax_e, ay_e = solve_2bone_ik(hx_j, hy_j, ank_x, ank_y, thigh_l, calf_l, prefer_forward_x=True)

        # 3A. Sculpted Flat-Soled Animator Shoe (drawn first, under the trouser cuff!)
        sole_y = int(origin_y + f_dy - heel_lift * 0.25)
        heel_x = int(ax_e - 18.0 * scale)
        toe_x = int(ax_e + 36.0 * scale)
        instep_y = int(sole_y - 20.0 * scale)
        shoe_pts = np.array([
            [heel_x, sole_y - int(4 * scale)],
            [heel_x + int(2 * scale), instep_y + int(3 * scale)],
            [int(ax_e + 10 * scale), instep_y],
            [toe_x - int(6 * scale), instep_y + int(4 * scale)],
            [toe_x, sole_y - int(6 * scale)],
            [toe_x - int(2 * scale), sole_y],
            [heel_x + int(2 * scale), sole_y],
        ], dtype=np.int32)
        if shoe_fill is not None:
            cv2.fillPoly(canvas, [shoe_pts], shoe_fill, cv2.LINE_AA)
            cv2.polylines(canvas, [shoe_pts], True, ink_col, w_i, cv2.LINE_AA)
            # Crisp sole welt line parallel to bottom of shoe
            cv2.line(canvas, (heel_x + int(2 * scale), sole_y - int(4 * scale)), (toe_x - int(2 * scale), sole_y - int(4 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [shoe_pts], True, ink_col, max(1, w_i - 1), cv2.LINE_AA)

        # 3B. Dual-Pass Capsule Union for Thigh + Calf (100% seamless knee joint!)
        r_th, r_kn, r_an = 20.0 * scale, 17.5 * scale, 15.0 * scale
        if p_fill is not None:
            # Pass 1: Outer Ink Casing
            cv2.fillPoly(canvas, [_capsule_poly((hx_j, hy_j), (kx, ky), r_th + w_i, r_kn + w_i)], ink_col, cv2.LINE_AA)
            cv2.fillPoly(canvas, [_capsule_poly((kx, ky), (ax_e, ay_e), r_kn + w_i, r_an + w_i)], ink_col, cv2.LINE_AA)
            # Pass 2: Opaque Trouser Fill (erases internal knee seam!)
            cv2.fillPoly(canvas, [_capsule_poly((hx_j, hy_j), (kx, ky), r_th, r_kn)], p_fill, cv2.LINE_AA)
            cv2.fillPoly(canvas, [_capsule_poly((kx, ky), (ax_e, ay_e), r_kn, r_an)], p_fill, cv2.LINE_AA)
            # Pass 3: Neat Cuffed Trouser Hem at Ankle (`cuff_p0 -> cuff_p1`)
            c_p0 = (ax_e - 15.5 * scale, ay_e - 2.0 * scale)
            c_p1 = (ax_e + 15.5 * scale, ay_e - 2.0 * scale)
            cv2.fillPoly(canvas, [_capsule_poly(c_p0, c_p1, 5.0 * scale + w_i, 5.0 * scale + w_i)], ink_col, cv2.LINE_AA)
            cv2.fillPoly(canvas, [_capsule_poly(c_p0, c_p1, 5.0 * scale, 5.0 * scale)], p_fill, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [_capsule_poly((hx_j, hy_j), (kx, ky), r_th, r_kn)], True, ink_col, max(1, w_i - 1), cv2.LINE_AA)
            cv2.polylines(canvas, [_capsule_poly((kx, ky), (ax_e, ay_e), r_kn, r_an)], True, ink_col, max(1, w_i - 1), cv2.LINE_AA)

    # --- 4. UNIFIED TORSO, SHIRT SHOULDERS, BIB APRON, WAIST TIE & POINTED `W` COLLAR (`nib_inking_keys.jpg` Q1-Q4) ---
    ap_lag = pose["apron_lag"] * 0.65 * scale
    bib_top_y = chest_y + 34.0 * scale
    apron_bot_y = hip_y + 46.0 * scale

    # 4A. Upper Shirt Chest & Back Torso Dome (with visible left shirt flank & waist-tie strap as in `nib_inking_keys.jpg` Q2!)
    shirt_dome = np.array([
        [int(chest_x - w_chest * 0.90), int(chest_y + 18.0 * scale)],
        [int(chest_x - w_chest * 0.55), int(chest_y + 2.0 * scale)],
        [int(chest_x + w_chest * 0.58), int(chest_y + 2.0 * scale)],
        [int(chest_x + w_chest * 0.88), int(chest_y + 18.0 * scale)],
        [int(mid_spine_x + w_waist * 0.88), int(mid_spine_y + 20.0 * scale)],
        [int(mid_spine_x - w_waist * 0.94), int(mid_spine_y + 24.0 * scale)],
    ], dtype=np.int32)
    if shirt_fill is not None:
        cv2.fillPoly(canvas, [shirt_dome], shirt_fill, cv2.LINE_AA)
    cv2.polylines(canvas, [shirt_dome], True, ink_col, w_i, cv2.LINE_AA)

    # 4B. Animator's Bib Apron Skirt & Square-U Bib (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`)
    apron_body = np.array([
        [int(chest_x - w_chest * 0.52), int(bib_top_y)],
        [int(chest_x + w_chest * 0.62), int(bib_top_y)],
        [int(mid_spine_x + w_waist * 0.90 + ap_lag * 0.35), int(mid_spine_y + 14.0 * scale)],
        [int(hip_x + w_hip * 1.00 + ap_lag), int(apron_bot_y)],
        [int(hip_x - w_hip * 0.88 + ap_lag * 0.8), int(apron_bot_y)],
        [int(mid_spine_x - w_waist * 0.80 + ap_lag * 0.3), int(mid_spine_y + 14.0 * scale)],
    ], dtype=np.int32)
    if apron_fill is not None:
        cv2.fillPoly(canvas, [apron_body], apron_fill, cv2.LINE_AA)
    cv2.polylines(canvas, [apron_body], True, ink_col, w_i, cv2.LINE_AA)

    # Horizontal Apron Waist-Tie Band wrapping around the left flank (`nib_inking_keys.jpg` Q2 & `studio_crew_celebration_keys.jpg`)
    wt_y = int(mid_spine_y + 15.0 * scale)
    wt_x0 = int(mid_spine_x - w_waist * 0.93)
    wt_x1 = int(mid_spine_x - w_waist * 0.78 + ap_lag * 0.3)
    if apron_strap_fill is not None and wt_x1 > wt_x0:
        cv2.rectangle(canvas, (wt_x0, wt_y - int(4 * scale)), (wt_x1, wt_y + int(4 * scale)), apron_strap_fill, -1)
        cv2.rectangle(canvas, (wt_x0, wt_y - int(4 * scale)), (wt_x1, wt_y + int(4 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)

    # Subtle hand-drawn cloth drape ticks inside the lower apron skirt (`studio_crew_celebration_keys.jpg` Q1-Q2)
    cv2.line(
        canvas,
        (int(hip_x - w_hip * 0.32 + ap_lag * 0.5), int(apron_bot_y - 16.0 * scale)),
        (int(hip_x - w_hip * 0.12 + ap_lag * 0.6), int(apron_bot_y - 24.0 * scale)),
        ink_col, max(1, w_i - 1), cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (int(hip_x + w_hip * 0.18 + ap_lag * 0.6), int(apron_bot_y - 14.0 * scale)),
        (int(hip_x + w_hip * 0.38 + ap_lag * 0.7), int(apron_bot_y - 20.0 * scale)),
        ink_col, max(1, w_i - 1), cv2.LINE_AA,
    )

    # Two Vertical Apron Shoulder Straps framing the Pointed `W` Shirt Collar!
    for x_top_f, x_bot_f in [(-0.52, -0.46), (0.58, 0.54)]:
        strap_poly = np.array([
            [int(chest_x + x_top_f * w_chest - 5.5 * scale), int(chest_y + 3 * scale)],
            [int(chest_x + x_top_f * w_chest + 5.5 * scale), int(chest_y + 3 * scale)],
            [int(chest_x + x_bot_f * w_chest + 5.5 * scale), int(bib_top_y + 2 * scale)],
            [int(chest_x + x_bot_f * w_chest - 5.5 * scale), int(bib_top_y + 2 * scale)],
        ], dtype=np.int32)
        if apron_strap_fill is not None:
            cv2.fillPoly(canvas, [strap_poly], apron_strap_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [strap_poly], True, ink_col, w_i, cv2.LINE_AA)
    # Re-stroke horizontal Square-U Bib Top line so strap-to-bib join is seamless
    if apron_fill is not None:
        cv2.rectangle(
            canvas,
            (int(chest_x - w_chest * 0.48), int(bib_top_y + 1 * scale)),
            (int(chest_x + w_chest * 0.56), int(bib_top_y + 10 * scale)),
            apron_fill, -1,
        )
    cv2.line(
        canvas,
        (int(chest_x - w_chest * 0.52), int(bib_top_y)),
        (int(chest_x + w_chest * 0.62), int(bib_top_y)),
        ink_col, w_i, cv2.LINE_AA,
    )

    # 4C. Crisp Pointed `W` Shirt Collar (`\/\/` tucked cleanly between the apron straps right below the chin!)
    col_cx = chest_x + 4.0 * scale
    col_cy = chest_y + 4.0 * scale
    w_collar_pts = np.array([
        [int(col_cx - 21 * scale), int(col_cy)],
        [int(col_cx - 12 * scale), int(col_cy + 22 * scale)],
        [int(col_cx), int(col_cy + 10 * scale)],
        [int(col_cx + 12 * scale), int(col_cy + 22 * scale)],
        [int(col_cx + 21 * scale), int(col_cy)],
    ], dtype=np.int32)
    if shirt_fill is not None:
        cv2.fillPoly(canvas, [w_collar_pts], shirt_fill, cv2.LINE_AA)
    cv2.polylines(canvas, [w_collar_pts], False, ink_col, w_i, cv2.LINE_AA)
    # Inner V-neck throat lines
    cv2.line(canvas, (int(col_cx - 10 * scale), int(col_cy - 2 * scale)), (int(col_cx), int(col_cy + 10 * scale)), ink_col, w_i, cv2.LINE_AA)
    cv2.line(canvas, (int(col_cx + 10 * scale), int(col_cy - 2 * scale)), (int(col_cx), int(col_cy + 10 * scale)), ink_col, w_i, cv2.LINE_AA)

    # --- 5. CANONICAL ROUND CARTOON CRANIUM & SMOOTH-RIMMED 3-POINTED FOUNTAIN-NIB CROWN HAT ---
    hpitch = float(pose["head_pitch"]) * 0.45
    rad_p = math.radians(hpitch)
    cp, sp = math.cos(rad_p), math.sin(rad_p)

    def _hpt(lx_u: float, ly_u: float) -> tuple:
        lx = lx_u * hrx
        ly = ly_u * hry
        return (head_cx + lx * cp - ly * sp, head_cy + lx * sp + ly * cp)

    # Sculpted Cartoon C-Ear on Left Side of Head (`nib_inking_keys.jpg` Q2 & `studio_crew_celebration_keys.jpg`)
    ear_cx, ear_cy = _hpt(-0.92, 0.08)
    ear_rx, ear_ry = max(4, int(11.0 * scale)), max(5, int(13.5 * scale))
    if skin_fill is not None:
        cv2.ellipse(canvas, (int(ear_cx), int(ear_cy)), (ear_rx, ear_ry), hpitch, 0, 360, skin_fill, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(ear_cx), int(ear_cy)), (ear_rx, ear_ry), hpitch, 70, 290, ink_col, w_i, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(ear_cx + 2.0 * scale), int(ear_cy)), (max(2, int(ear_rx * 0.48)), max(2, int(ear_ry * 0.48))), hpitch, 95, 265, ink_col, max(1, w_i - 1), cv2.LINE_AA)

    if skin_fill is not None:
        cv2.ellipse(canvas, (int(head_cx), int(head_cy)), (int(hrx), int(hry)), hpitch, -28, 208, skin_fill, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(head_cx), int(head_cy)), (int(hrx), int(hry)), hpitch, -28, 208, ink_col, w_i, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (int(head_cx), int(head_cy)), (int(hrx), int(hry)), hpitch, 0, 360, ink_col, w_i, cv2.LINE_AA)

    if is_rough:
        eye_line = np.array([_hpt(-0.92, -0.02), _hpt(0.0, 0.02), _hpt(0.92, -0.02)], dtype=np.int32)
        ctr_line = np.array([_hpt(0.14, -0.92), _hpt(0.18, 0.0), _hpt(0.14, 0.92)], dtype=np.int32)
        cv2.polylines(canvas, [eye_line], False, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [ctr_line], False, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    # Canonical 3-Pointed Fountain-Pen Nib Crown Hat with Smoothly Arched Headband Rim (`nib_inking_keys.jpg` & `agent_nib_keys.jpg`)
    crown_lag = float(pose["hair_lag"]) * 0.005
    crown_peaks = [
        _hpt(-0.95, -0.38),
        _hpt(-1.06 + crown_lag * 0.5, -1.24),
        _hpt(-0.45 + crown_lag * 0.3, -0.82),
        _hpt(0.02 + crown_lag, -1.92),
        _hpt(0.47 + crown_lag * 0.3, -0.82),
        _hpt(1.04 + crown_lag * 0.5, -1.22),
        _hpt(0.95, -0.38),
    ]
    # 11-point smooth arched headband curve from right temple `(0.95, -0.38)` across forehead `(0.0, -0.50)` to left temple `(-0.95, -0.38)`
    rim_curve = []
    for ri in range(1, 11):
        u_r = ri / 11.0
        rx_u = 0.95 * (1.0 - 2.0 * u_r)
        ry_u = -0.38 - 0.12 * math.sin(math.pi * u_r)
        rim_curve.append(_hpt(rx_u, ry_u))
    crown_poly = np.round(np.array(crown_peaks + rim_curve, dtype=np.float64)).astype(np.int32)
    if crown_fill is not None:
        cv2.fillPoly(canvas, [crown_poly], crown_fill, cv2.LINE_AA)
    cv2.polylines(canvas, [crown_poly], True, ink_col, w_i, cv2.LINE_AA)

    # Fountain-Nib Crown Slit & Circular Breather Hole `o`
    apex_pt = _hpt(0.02 + crown_lag, -1.92)
    hole_pt = _hpt(0.01 + crown_lag * 0.35, -0.90)
    cv2.line(canvas, (int(apex_pt[0]), int(apex_pt[1])), (int(hole_pt[0]), int(hole_pt[1])), ink_col, w_i, cv2.LINE_AA)
    hole_r = max(4, int(round(0.115 * hrx)))
    if crown_fill is not None:
        cv2.circle(canvas, (int(hole_pt[0]), int(hole_pt[1])), hole_r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(hole_pt[0]), int(hole_pt[1])), hole_r, ink_col, w_i, cv2.LINE_AA)

    # --- 6. CANONICAL FACIAL ACTING (`● ●` Solid Black Oval Eyes, Expressive Brows, C-Nose, 5 Mouths) ---
    eye_open = float(pose["eye_open"])
    is_wink = (pose.get("code") == "K16" and pose.get("raw_u", 1.0) > 0.65)
    is_happy_arch = (pose.get("code") == "K14" and 0.25 <= pose.get("raw_u", 0.5) <= 0.85)
    face_shift_u = 0.14

    for ei, (ex_u, ey_u, brow_val) in enumerate([
        (-0.24 + face_shift_u, -0.01, pose["brow_l"]),
        (0.28 + face_shift_u, -0.01, pose["brow_r"]),
    ]):
        ecx, ecy = _hpt(ex_u + pose["look_dx"] * 0.04, ey_u + pose["look_dy"] * 0.03)
        erx = max(3, int(round(0.105 * hrx)))
        ery = max(4, int(round(0.165 * hry * float(np.clip(eye_open, 0.55, 1.20)))))

        if ei == 0 and is_wink:
            w_pts = np.array([
                [int(ecx - erx * 1.2), int(ecy - ery * 0.55)],
                [int(ecx + erx * 1.1), int(ecy)],
                [int(ecx - erx * 1.2), int(ecy + ery * 0.55)],
            ], dtype=np.int32)
            cv2.polylines(canvas, [w_pts], False, ink_col, w_i, cv2.LINE_AA)
        elif is_happy_arch:
            cv2.ellipse(canvas, (int(ecx), int(ecy + ery * 0.3)), (int(erx * 1.25), int(ery * 0.85)), hpitch, 195, 345, ink_col, w_i, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (int(ecx), int(ecy)), (erx, ery), hpitch, 0, 360, ink_col, -1, cv2.LINE_AA)

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

    m_type = pose["mouth"]
    mx, my = _hpt(0.02 + face_shift_u, 0.44)
    if m_type == "think_pucker":
        cv2.ellipse(canvas, (int(mx), int(my + 6 * scale)), (int(16 * scale), int(7 * scale)), hpitch, 200, 340, ink_col, w_i, cv2.LINE_AA)
    elif m_type == "tongue_concentrate":
        smile_pts = np.array([
            [int(mx - 16 * scale), int(my + 2 * scale)],
            [int(mx + 2 * scale), int(my + 7 * scale)],
            [int(mx + 20 * scale), int(my - 4 * scale)],
        ], dtype=np.int32)
        cv2.polylines(canvas, [smile_pts], False, ink_col, w_i, cv2.LINE_AA)
        tcx, tcy = int(mx + 18 * scale), int(my - 2 * scale)
        t_fill = (178, 168, 248) if is_colored else ((255, 255, 255) if is_cleanup else None)
        if t_fill is not None:
            cv2.ellipse(canvas, (tcx, tcy), (int(10 * scale), int(7.5 * scale)), -20, 0, 360, t_fill, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (tcx, tcy), (int(10 * scale), int(7.5 * scale)), -20, 0, 360, ink_col, w_i, cv2.LINE_AA)
        cv2.line(canvas, (int(tcx - 4 * scale), int(tcy + 1 * scale)), (int(tcx + 5 * scale), int(tcy - 2 * scale)), ink_col, max(1, w_i - 1), cv2.LINE_AA)
    elif m_type == "blow_puff" or pose.get("code") == "K10":
        cv2.ellipse(canvas, (int(mx - 10 * scale), int(my)), (int(6 * scale), int(9 * scale)), 0, -70, 70, ink_col, w_i, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(mx + 6 * scale), int(my)), (int(5 * scale), int(6.5 * scale)), 0, 0, 360, ink_col, -1, cv2.LINE_AA)
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
        mw, mh = int(20 * scale), int(14 * scale)
        cv2.ellipse(canvas, (int(mx), int(my - 2 * scale)), (mw, mh), hpitch, 0, 180, ink_col, -1, cv2.LINE_AA)
        if not is_rough:
            t_col = (168, 158, 245) if is_colored else (255, 255, 255)
            cv2.ellipse(canvas, (int(mx), int(my + 6 * scale)), (int(mw * 0.65), int(mh * 0.42)), hpitch, 180, 360, t_col, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(mx), int(my - 2 * scale)), (mw, mh), hpitch, 0, 180, ink_col, w_i, cv2.LINE_AA)
        cv2.line(canvas, (int(mx - mw), int(my - 2 * scale)), (int(mx + mw), int(my - 2 * scale)), ink_col, w_i, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (int(mx), int(my - 4 * scale)), (int(18 * scale), int(9 * scale)), hpitch - 6, 15, 165, ink_col, w_i, cv2.LINE_AA)
        cv2.line(canvas, (int(mx - 17 * scale), int(my - 5 * scale)), (int(mx - 20 * scale), int(my - 1 * scale)), ink_col, max(2, w_i - 1), cv2.LINE_AA)

    # --- 7A. DRAW LEFT-ON-SCREEN ARM (`sh_lx = chest_x - w_chest * 0.72` -> `el_lx` -> `wr_lx` below bib) ---
    _draw_seamless_arm_and_hand(
        canvas, (sh_lx, sh_ly), (el_lx, el_ly), (wr_lx, wr_ly),
        w_up=17.5 * scale, w_low=14.0 * scale, scale=scale,
        hand_pose=pose["lh_pose"], is_right=False, is_near_shoulder=True,
        shirt_fill=shirt_fill, skin_fill=skin_fill, ink_col=ink_col, ink_w=ink_w,
    )

    # --- 7B. DRAW RIGHT-ON-SCREEN DRAWING ARM (Attached at Right Outer Shoulder `chest_x + w_chest * 0.70`,
    #         outside the right apron strap so the `\/\/` collar, `|___|` Square-U bib, and both `| |` apron
    #         straps stay 100% VISIBLE AND UNOCCLUDED, matching `nib_inking_keys.jpg` Q1-Q4 & `agent_nib_keys.jpg`!) ---
    sh_rx = chest_x + w_chest * 0.70
    sh_ry = chest_y + 18.0 * scale
    el_rx, el_ry, wr_rx, wr_ry = solve_2bone_ik(sh_rx, sh_ry, rh_target[0], rh_target[1], u_arm * 0.92, f_arm * 0.92, prefer_lower_joint=True)
    # Keep right-on-screen elbow cleanly to the right of the right apron strap (`sh_rx + 10 * scale`)!
    el_rx = max(el_rx, sh_rx + 10.0 * scale)

    if pose.get("seg_idx") in (3, 4, 8):
        for echo_off_y, echo_ang in [(-38.0 * scale, -55.0), (26.0 * scale, -18.0)]:
            e_tx, e_ty = rh_target[0] - 18.0 * scale, rh_target[1] + echo_off_y
            e_elx, e_ely, e_wrx, e_wry = solve_2bone_ik(sh_rx, sh_ry, e_tx, e_ty, u_arm * 0.92, f_arm * 0.92, prefer_lower_joint=True)
            arc_cx, arc_cy = int(0.5 * (e_wrx + wr_rx) + 24 * scale), int(0.5 * (e_wry + wr_ry))
            cv2.ellipse(canvas, (arc_cx, arc_cy), (int(18 * scale), int(34 * scale)), 15, -55, 55, ink_col, max(1, w_i - 2), cv2.LINE_AA)
            cv2.ellipse(canvas, (arc_cx + int(8 * scale), arc_cy), (int(18 * scale), int(30 * scale)), 15, -45, 45, ink_col, max(1, w_i - 2), cv2.LINE_AA)

    has_ink_pen = (pose.get("seg_idx", 0) >= 6)
    pen_tip = _draw_seamless_arm_and_hand(
        canvas, (sh_rx, sh_ry), (el_rx, el_ry), (wr_rx, wr_ry),
        w_up=18.5 * scale, w_low=14.8 * scale, scale=scale,
        hand_pose=pose["rh_pose"], is_right=True, is_near_shoulder=False,
        shirt_fill=shirt_fill, skin_fill=skin_fill, ink_col=ink_col, ink_w=ink_w,
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
        # Subtle diagonal glass reflection hatch marks (`///`) inside select panes (`multiplane_plane5_distant_bg_keys.jpg`)
        for r_i, (px_u, py_u) in enumerate([(0.14, 0.12), (0.64, 0.32), (0.38, 0.54), (0.66, 0.72)]):
            rx_gl = wx0 + int(px_u * (wx1 - wx0))
            ry_gl = wy0 + int(py_u * (wy1 - wy0))
            for hk in range(2):
                off_h = hk * int(6 * s_ui)
                cv2.line(canvas, (rx_gl + off_h, ry_gl + int(18 * s_ui)), (rx_gl + off_h + int(14 * s_ui), ry_gl + int(2 * s_ui)), (128, 112, 96), max(1, int(1.3 * s_ui)), cv2.LINE_AA)
    elif is_cleanup:
        cv2.rectangle(canvas, (wx0, wy0), (wx1, wy1), BGR_PAPER, -1)
        # Diagonal glass reflection hatch marks (`///`) inside window panes (`multiplane_plane5_distant_bg_keys.jpg`)
        for r_i, (px_u, py_u) in enumerate([(0.14, 0.12), (0.64, 0.32), (0.38, 0.54), (0.66, 0.72)]):
            rx_gl = wx0 + int(px_u * (wx1 - wx0))
            ry_gl = wy0 + int(py_u * (wy1 - wy0))
            for hk in range(3):
                off_h = hk * int(5 * s_ui)
                cv2.line(canvas, (rx_gl + off_h, ry_gl + int(18 * s_ui)), (rx_gl + off_h + int(14 * s_ui), ry_gl + int(2 * s_ui)), bg_line_col, max(1, int(1.3 * s_ui)), cv2.LINE_AA)

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

    # Cast-Iron Ribbed Steam Radiator under Window (`multiplane_plane5_distant_bg_keys.jpg` Q2)
    rad_y0 = wy1 + int(30 * s_ui * z_p5)
    rad_y1 = min(baseboard_y - int(4 * s_ui), wy1 + int(134 * s_ui * z_p5))
    if rad_y1 > rad_y0 + 20:
        rad_x0 = wx0 + int(14 * s_ui)
        rad_x1 = wx1 - int(14 * s_ui)
        # Upper & lower horizontal connecting steam manifold rods across all 8 fins
        for ry_rod in (rad_y0 + int(10 * s_ui), rad_y1 - int(10 * s_ui)):
            cv2.line(canvas, (rad_x0 - int(10 * s_ui), ry_rod), (rad_x1 + int(10 * s_ui), ry_rod), bg_line_col, max(2, int(3.2 * s_ui)), cv2.LINE_AA)
        for r_fin in range(8):
            fx_r = rad_x0 + int((r_fin / 7.0) * (rad_x1 - rad_x0))
            fin_w = max(4, int(7.5 * s_ui * z_p5))
            fin_pts = [
                (fx_r - fin_w, rad_y0 + fin_w),
                (fx_r - fin_w * 0.5, rad_y0),
                (fx_r + fin_w * 0.5, rad_y0),
                (fx_r + fin_w, rad_y0 + fin_w),
                (fx_r + fin_w, rad_y1),
                (fx_r - fin_w, rad_y1),
            ]
            draw_ink_stroke(
                canvas, fin_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.85,
                closed=True, fill_bgr=(162, 174, 186) if is_colored else shade_fill,
                boil_phase=boil_phase, seed=42 + r_fin,
            )
            # Center rib highlight line down each radiator column
            cv2.line(canvas, (fx_r, rad_y0 + int(6 * s_ui)), (fx_r, rad_y1 - int(6 * s_ui)), bg_line_col, max(1, int(1.1 * s_ui)), cv2.LINE_AA)
        # 90-degree bent steam pipe & bolted flange couplings (`||`) on right & left of radiator (`multiplane_plane5_distant_bg_keys.jpg` Q2)
        pipe_rx = rad_x1 + int(18 * s_ui)
        pipe_ry = rad_y0 + int(16 * s_ui)
        cv2.line(canvas, (rad_x1 + int(6 * s_ui), pipe_ry), (pipe_rx, pipe_ry), bg_line_col, max(2, int(3.8 * s_ui)), cv2.LINE_AA)
        cv2.line(canvas, (pipe_rx, pipe_ry), (pipe_rx, floor_y), bg_line_col, max(2, int(3.8 * s_ui)), cv2.LINE_AA)
        # Bolted horizontal flange coupling (`=`) & brass steam valve wheel
        for fl_y in (pipe_ry + int(16 * s_ui), pipe_ry + int(20 * s_ui)):
            cv2.line(canvas, (pipe_rx - int(6 * s_ui), fl_y), (pipe_rx + int(6 * s_ui), fl_y), bg_line_col, max(1, int(2.0 * s_ui)), cv2.LINE_AA)
        cv2.ellipse(canvas, (pipe_rx, pipe_ry - int(9 * s_ui)), (max(3, int(7 * s_ui)), max(2, int(3 * s_ui))), 0, 0, 360, bg_line_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)
        cv2.line(canvas, (pipe_rx, pipe_ry - int(9 * s_ui)), (pipe_rx, pipe_ry), bg_line_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)

    # ==========================================================================
    # PLANE 4 (`depth = 2.1`): BENTWOOD COAT RACK, STORYBOARD CORKBOARD,
    # 35MM FILM-CAN CREDENZA, AND LIVE 4:3 STUDIO PENCIL-TEST PROJECTION SCREEN
    # (Matches `multiplane_plane4_midground_arch_keys.jpg` and keeps Nib's crown
    #  silhouette 100% clear per `Rule IX: DRAW FOR SILHOUETTE`!)
    # ==========================================================================
    sb_cx, sb_cy, z_sb = _plane_tf(out_w * 0.255, out_h * 0.29, 2.1)
    sb_w, sb_h = int(92 * s_ui * z_sb), int(64 * s_ui * z_sb)

    # Bentwood Studio Coat Rack / Hat Tree (`multiplane_plane4_midground_arch_keys.jpg` Q1)
    rack_x = sb_cx - sb_w * 1.18
    rack_top_y = sb_cy - sb_h * 0.88
    rack_bot_y = floor_y - int(4 * s_ui)
    if rack_x > wx1 + int(14 * s_ui):
        # Vertical turned wood post
        cv2.line(canvas, (int(rack_x), int(rack_top_y)), (int(rack_x), int(rack_bot_y)), bg_line_col, max(2, int(4.2 * s_ui * z_sb)), cv2.LINE_AA)
        cv2.circle(canvas, (int(rack_x), int(rack_top_y - 4 * s_ui)), max(2, int(4.5 * s_ui * z_sb)), bg_line_col, -1, cv2.LINE_AA)
        # 4 S-curved bentwood coat hooks at top crown
        for hk_sign in (-1.0, 1.0):
            hk_pts = [
                (rack_x, rack_top_y + 14 * s_ui * z_sb),
                (rack_x + hk_sign * 16 * s_ui * z_sb, rack_top_y + 4 * s_ui * z_sb),
                (rack_x + hk_sign * 22 * s_ui * z_sb, rack_top_y - 8 * s_ui * z_sb),
                (rack_x + hk_sign * 15 * s_ui * z_sb, rack_top_y - 14 * s_ui * z_sb),
            ]
            draw_ink_stroke(canvas, hk_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.9, boil_phase=boil_phase, seed=46 + int(hk_sign + 2))
            hk_low = [
                (rack_x, rack_top_y + 26 * s_ui * z_sb),
                (rack_x + hk_sign * 12 * s_ui * z_sb, rack_top_y + 20 * s_ui * z_sb),
                (rack_x + hk_sign * 14 * s_ui * z_sb, rack_top_y + 12 * s_ui * z_sb),
            ]
            draw_ink_stroke(canvas, hk_low, color_bgr=bg_line_col, base_thick=bg_line_w * 0.8, boil_phase=boil_phase, seed=146 + int(hk_sign + 2))
        # Tripod curved base legs
        for leg_s in (-1.0, 0.0, 1.0):
            leg_pts = [
                (rack_x, rack_bot_y - 38 * s_ui * z_sb),
                (rack_x + leg_s * 14 * s_ui * z_sb, rack_bot_y - 18 * s_ui * z_sb),
                (rack_x + leg_s * 22 * s_ui * z_sb, rack_bot_y),
            ]
            draw_ink_stroke(canvas, leg_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95, boil_phase=boil_phase, seed=47 + int(leg_s + 2))

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
    # Inner corkboard border & stippled cork texture dots (`· · ·` in `multiplane_plane4_midground_arch_keys.jpg` Q1)
    cv2.rectangle(
        canvas,
        (int(sb_cx - sb_w + 7 * s_ui), int(sb_cy - sb_h + 7 * s_ui)),
        (int(sb_cx + sb_w - 7 * s_ui), int(sb_cy + sb_h - 7 * s_ui)),
        bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA,
    )
    for st_x, st_y in [(-0.82, -0.78), (-0.28, -0.82), (0.30, -0.78), (0.82, -0.76), (-0.84, 0.02), (0.84, 0.04), (-0.30, 0.82), (0.28, 0.80)]:
        cv2.circle(canvas, (int(sb_cx + st_x * sb_w), int(sb_cy + st_y * sb_h)), max(1, int(1.2 * s_ui)), bg_line_col, -1, cv2.LINE_AA)

    for sc_r in range(2):
        for sc_c in range(3):
            cx_card = sb_cx - sb_w * 0.56 + sc_c * (sb_w * 0.56)
            cy_card = sb_cy - sb_h * 0.40 + sc_r * (sb_h * 0.80)
            cw_c, ch_c = int(21 * s_ui * z_sb), int(15 * s_ui * z_sb)
            tilt_y = int(2.5 * s_ui) if (sc_r == 1 and sc_c == 1) else 0
            card_poly = np.array([
                [int(cx_card - cw_c), int(cy_card - ch_c - tilt_y)],
                [int(cx_card + cw_c), int(cy_card - ch_c + tilt_y)],
                [int(cx_card + cw_c), int(cy_card + ch_c + tilt_y)],
                [int(cx_card - cw_c), int(cy_card + ch_c - tilt_y)],
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [card_poly], (248, 250, 252) if not is_rough else BGR_PAPER, cv2.LINE_AA)
            cv2.polylines(canvas, [card_poly], True, bg_line_col, max(1, int(1.4 * s_ui)), cv2.LINE_AA)
            # Push-pin at top center
            cv2.circle(canvas, (int(cx_card), int(cy_card - ch_c + 3 * s_ui)), max(2, int(2.2 * s_ui)), BGR_ACCENT_RED if is_colored else bg_line_col, -1, cv2.LINE_AA)
            # Storyboard thumbnail sketch in upper half + 2 horizontal caption lines in lower half (`multiplane_plane4_midground_arch_keys.jpg` Q1)
            cv2.line(canvas, (int(cx_card - cw_c * 0.55), int(cy_card - 2 * s_ui)), (int(cx_card + cw_c * 0.55), int(cy_card - 5 * s_ui)), bg_line_col, max(1, int(1.2 * s_ui)), cv2.LINE_AA)
            cv2.line(canvas, (int(cx_card - cw_c * 0.65), int(cy_card + ch_c * 0.35)), (int(cx_card + cw_c * 0.65), int(cy_card + ch_c * 0.35)), bg_line_col, max(1, int(1.0 * s_ui)), cv2.LINE_AA)
            cv2.line(canvas, (int(cx_card - cw_c * 0.65), int(cy_card + ch_c * 0.65)), (int(cx_card + cw_c * 0.45), int(cy_card + ch_c * 0.65)), bg_line_col, max(1, int(1.0 * s_ui)), cv2.LINE_AA)

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
        # Left upper shelf: flat horizontal stacked 35mm film cans
        for fc_i in range(3):
            fc_y = mid_shelf_y - int((7 + fc_i * 10) * s_ui * z_sb)
            cv2.rectangle(
                canvas,
                (int(sb_cx - sb_w * 0.78), fc_y - int(4 * s_ui)),
                (int(sb_cx - sb_w * 0.18), fc_y + int(4 * s_ui)),
                bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA,
            )
        # Right upper shelf: upright vertical row of round 35mm film cans `( ) ) ) )` (`multiplane_plane4_midground_arch_keys.jpg` Q1)
        for rc_i in range(5):
            rc_x = int(sb_cx + (0.08 + rc_i * 0.14) * sb_w)
            rc_y = mid_shelf_y - int(16 * s_ui * z_sb)
            cv2.ellipse(
                canvas, (rc_x, rc_y),
                (max(3, int(8 * s_ui * z_sb)), max(5, int(15 * s_ui * z_sb))),
                0, 0, 360, (186, 196, 206) if is_colored else paper_fill, -1, cv2.LINE_AA,
            )
            cv2.ellipse(
                canvas, (rc_x, rc_y),
                (max(3, int(8 * s_ui * z_sb)), max(5, int(15 * s_ui * z_sb))),
                0, 0, 360, bg_line_col, max(1, int(1.4 * s_ui)), cv2.LINE_AA,
            )
        # Lower shelf: upright dope-sheet binders (`| | |`)
        for bd_i in range(6):
            bd_x0 = int(sb_cx - sb_w * 0.78 + bd_i * 11 * s_ui * z_sb)
            cv2.rectangle(
                canvas,
                (bd_x0, mid_shelf_y + int(8 * s_ui * z_sb)),
                (bd_x0 + int(9 * s_ui * z_sb), cred_bot_y - int(4 * s_ui)),
                bg_line_col, max(1, int(1.3 * s_ui)), cv2.LINE_AA,
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

    # 1. 3D Beveled A-Frame Wooden Trestle Legs, Splayed Floor Runners, Crossbar & Tilt Mechanism
    for leg_off in (-118.0, 108.0):
        lx_d = desk_cx + leg_off * nib_scale
        # 3D right bevel of vertical trestle post
        d_leg_side = [
            (lx_d + 12 * nib_scale, desk_fy - 245 * nib_scale),
            (lx_d + 22 * nib_scale, desk_fy - 240 * nib_scale),
            (lx_d + 26 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d + 16 * nib_scale, desk_fy - 14 * nib_scale),
        ]
        draw_ink_stroke(canvas, d_leg_side, color_bgr=bg_line_col, base_thick=bg_line_w * 1.05, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=59 + int(leg_off), sharp_corners=True)
        # Front face of vertical trestle post
        d_leg = [
            (lx_d - 14 * nib_scale, desk_fy - 245 * nib_scale),
            (lx_d + 12 * nib_scale, desk_fy - 245 * nib_scale),
            (lx_d + 16 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d - 18 * nib_scale, desk_fy - 14 * nib_scale),
        ]
        draw_ink_stroke(canvas, d_leg, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=60 + int(leg_off), sharp_corners=True)
        # Delicate vertical wood-grain ink lines along trestle leg (`multiplane_plane4_midground_arch_keys.jpg` Q2)
        for wg_x in (-5.0, 3.0):
            cv2.line(
                canvas,
                (int(lx_d + wg_x * nib_scale), int(desk_fy - 218 * nib_scale)),
                (int(lx_d + (wg_x - 1.5) * nib_scale), int(desk_fy - 34 * nib_scale)),
                bg_line_col, max(1, int(1.1 * s_ui)), cv2.LINE_AA,
            )
        # 3D Splayed wooden floor runner pad (`apparatus_lightbox_anim_keys.jpg`)
        foot_side = [
            (lx_d + 34 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d + 44 * nib_scale, desk_fy - 10 * nib_scale),
            (lx_d + 46 * nib_scale, desk_fy),
            (lx_d + 36 * nib_scale, desk_fy),
        ]
        draw_ink_stroke(canvas, foot_side, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=58 + int(leg_off), sharp_corners=True)
        foot_pad = [
            (lx_d - 34 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d + 34 * nib_scale, desk_fy - 14 * nib_scale),
            (lx_d + 36 * nib_scale, desk_fy),
            (lx_d - 38 * nib_scale, desk_fy),
        ]
        draw_ink_stroke(canvas, foot_pad, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=61 + int(leg_off), sharp_corners=True)

    # Horizontal stretcher beam between legs with recessed wood-grain groove
    stretcher_pts = [
        (desk_cx - 104 * nib_scale, desk_fy - 98 * nib_scale),
        (desk_cx + 94 * nib_scale, desk_fy - 98 * nib_scale),
        (desk_cx + 94 * nib_scale, desk_fy - 78 * nib_scale),
        (desk_cx - 104 * nib_scale, desk_fy - 78 * nib_scale),
    ]
    draw_ink_stroke(canvas, stretcher_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=62, sharp_corners=True)
    cv2.line(canvas, (int(desk_cx - 98 * nib_scale), int(desk_fy - 88 * nib_scale)), (int(desk_cx + 88 * nib_scale), int(desk_fy - 88 * nib_scale)), bg_line_col, max(1, int(1.4 * s_ui)), cv2.LINE_AA)

    # Central tilt-quadrant bracket & hand-wheel lock knob under drafting board (`apparatus_lightbox_anim_keys.jpg`)
    quad_pts = [
        (desk_cx - 68 * nib_scale, desk_fy - 220 * nib_scale),
        (desk_cx - 34 * nib_scale, desk_fy - 178 * nib_scale),
        (desk_cx + 18 * nib_scale, desk_fy - 196 * nib_scale),
    ]
    draw_ink_stroke(canvas, quad_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=False, boil_phase=boil_phase, seed=63)
    cv2.circle(canvas, (int(desk_cx - 38 * nib_scale), int(desk_fy - 188 * nib_scale)), max(3, int(7 * nib_scale)), brass_fill if is_colored else BGR_PAPER, -1, cv2.LINE_AA)
    cv2.circle(canvas, (int(desk_cx - 38 * nib_scale), int(desk_fy - 188 * nib_scale)), max(3, int(7 * nib_scale)), bg_line_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)

    # Right-Hand Horizontal Side-Shelf Bracket & Diagonal Strut for Ink Bottle & Pencil Mug (`apparatus_lightbox_anim_keys.jpg`)
    sh_x0 = desk_cx + 76.0 * nib_scale
    sh_x1 = desk_cx + 158.0 * nib_scale
    sh_y = desk_fy - 242.0 * nib_scale
    # Diagonal wooden support strut under side shelf
    strut_pts = [
        (desk_cx + 108 * nib_scale, sh_y + 10 * nib_scale),
        (sh_x1 - 12 * nib_scale, sh_y + 10 * nib_scale),
        (desk_cx + 116 * nib_scale, sh_y + 52 * nib_scale),
        (desk_cx + 108 * nib_scale, sh_y + 52 * nib_scale),
    ]
    draw_ink_stroke(canvas, strut_pts, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=160, sharp_corners=True)
    # 3D horizontal shelf top & front edge
    shelf_top = [
        (sh_x0, sh_y),
        (sh_x0 + 18 * nib_scale, sh_y - 16 * nib_scale),
        (sh_x1 + 14 * nib_scale, sh_y - 16 * nib_scale),
        (sh_x1, sh_y),
    ]
    shelf_front = [
        (sh_x0, sh_y),
        (sh_x1, sh_y),
        (sh_x1, sh_y + 11 * nib_scale),
        (sh_x0, sh_y + 11 * nib_scale),
    ]
    draw_ink_stroke(canvas, shelf_top, color_bgr=bg_line_col, base_thick=bg_line_w * 1.05, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=161, sharp_corners=True)
    draw_ink_stroke(canvas, shelf_front, color_bgr=bg_line_col, base_thick=bg_line_w * 1.1, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=162, sharp_corners=True)

    # 2. 3D Beveled Rectangular Drafting Board Tilted Toward Nib (`apparatus_lightbox_anim_keys.jpg`)
    # Top face of the tilted drafting board:
    b_fl = (desk_cx - 165 * nib_scale, desk_fy - 222 * nib_scale)  # Front-left (closest to Nib)
    b_bl = (desk_cx - 75 * nib_scale, desk_fy - 348 * nib_scale)   # Back-left (higher up)
    b_br = (desk_cx + 158 * nib_scale, desk_fy - 336 * nib_scale)  # Back-right
    b_fr = (desk_cx + 106 * nib_scale, desk_fy - 210 * nib_scale)  # Front-right
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
    draw_ink_stroke(canvas, board_front_lip, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=64, sharp_corners=True)
    draw_ink_stroke(canvas, board_right_lip, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=65, sharp_corners=True)
    draw_ink_stroke(canvas, board_top, color_bgr=bg_line_col, base_thick=bg_line_w * 1.35, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=66, sharp_corners=True)

    # Recessed Rectangular Lightbox Frame inside the Tilted Drafting Board
    lb_pts = [
        (desk_cx - 122 * nib_scale, desk_fy - 234 * nib_scale),
        (desk_cx - 56 * nib_scale, desk_fy - 332 * nib_scale),
        (desk_cx + 96 * nib_scale, desk_fy - 324 * nib_scale),
        (desk_cx + 50 * nib_scale, desk_fy - 224 * nib_scale),
    ]
    draw_ink_stroke(canvas, lb_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=67, sharp_corners=True)

    # Upper-Board-Edge Raised 3D 4-Peg Acme Registration Bar (`[ 0  0  0  0 ]` along top of drafting board, `apparatus_lightbox_anim_keys.jpg` Q1-Q4)
    ub_top = [
        (desk_cx - 44 * nib_scale, desk_fy - 344 * nib_scale),
        (desk_cx + 88 * nib_scale, desk_fy - 337 * nib_scale),
        (desk_cx + 86 * nib_scale, desk_fy - 328 * nib_scale),
        (desk_cx - 46 * nib_scale, desk_fy - 335 * nib_scale),
    ]
    draw_ink_stroke(canvas, ub_top, color_bgr=bg_line_col, base_thick=bg_line_w * 1.0, closed=True, fill_bgr=brass_fill, boil_phase=boil_phase, seed=167, sharp_corners=True)
    for up_u in (0.14, 0.38, 0.62, 0.86):
        up_x = (desk_cx - 45 * nib_scale) + up_u * 132.0 * nib_scale
        up_y = (desk_fy - 339 * nib_scale) + up_u * 7.0 * nib_scale
        cv2.line(canvas, (int(up_x), int(up_y)), (int(up_x), int(up_y - 8 * nib_scale)), bg_line_col, max(2, int(2.4 * s_ui)), cv2.LINE_AA)
        cv2.circle(canvas, (int(up_x), int(up_y - 8 * nib_scale)), max(1, int(1.6 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)

    # 3. Illuminated Milk-Glass Animation Disc (`disc_x = nib_ox + 195 * scale, disc_y = nib_oy - 282 * scale`)
    disc_x = nib_ox + 195.0 * nib_scale
    disc_y = nib_oy - 282.0 * nib_scale
    if is_colored:
        glow_ov = canvas.copy()
        cv2.ellipse(glow_ov, (int(disc_x), int(disc_y)), (int(96 * nib_scale), int(54 * nib_scale)), -10, 0, 360, (212, 250, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow_ov, 0.38, canvas, 0.62, 0, canvas)

    # Outer metal rotating ring + inner milk-glass disc
    draw_sculpted_oval(
        canvas, disc_x, disc_y, 88.0 * nib_scale, 48.0 * nib_scale,
        angle_deg=-10, color_bgr=bg_line_col, thick=bg_line_w * 1.1,
        fill_bgr=shade_fill if not is_colored else (188, 212, 228),
        boil_phase=boil_phase, seed=168,
    )
    draw_sculpted_oval(
        canvas, disc_x, disc_y, 80.0 * nib_scale, 42.0 * nib_scale,
        angle_deg=-10, color_bgr=bg_line_col, thick=bg_line_w * 1.15,
        fill_bgr=(216, 250, 255) if is_colored else paper_fill,
        boil_phase=boil_phase, seed=68,
    )
    # Diagonal milk-glass sheen lines (`//`) inside disc (`apparatus_lightbox_anim_keys.jpg`)
    for sh_k in (-1, 0, 1):
        gx_s = disc_x + sh_k * 24.0 * nib_scale
        gy_s = disc_y + sh_k * 4.0 * nib_scale
        cv2.line(
            canvas,
            (int(gx_s - 13 * nib_scale), int(gy_s + 13 * nib_scale)),
            (int(gx_s + 13 * nib_scale), int(gy_s - 13 * nib_scale)),
            bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA,
        )

    # 4. 3-Sheet Rolling Animation Paper Stack on 4-Peg Acme Registration Bar!
    p_flip = float(pose["paper_flip"])
    d_prog = float(pose["draw_prog"])
    c_lift = float(pose["cel_lift"])

    def _draw_sparkle_star(cx_st: float, cy_st: float, r_st: float, col_st: tuple):
        """Draws a crisp 4-pointed hand-drawn cartoon sparkle star (`✦`) from `apparatus_lightbox_anim_keys.jpg` Q4."""
        r_in = r_st * 0.28
        st_pts = [
            (cx_st, cy_st - r_st),
            (cx_st + r_in, cy_st - r_in),
            (cx_st + r_st, cy_st),
            (cx_st + r_in, cy_st + r_in),
            (cx_st, cy_st + r_st),
            (cx_st - r_in, cy_st + r_in),
            (cx_st - r_st, cy_st),
            (cx_st - r_in, cy_st - r_in),
        ]
        pts_arr = np.array([[int(round(px)), int(round(py))] for px, py in st_pts], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_arr], (190, 245, 255) if is_colored else BGR_PAPER, cv2.LINE_AA)
        cv2.polylines(canvas, [pts_arr], True, col_st, max(1, int(round(1.6 * s_ui))), cv2.LINE_AA)

    if c_lift < 0.05:
        for sh_i in range(3):
            curl_h = (sh_i * 8.5 + p_flip * 56.0 * (sh_i / 2.0)) * nib_scale
            sh_pts = [
                (disc_x - 64 * nib_scale, disc_y - 24 * nib_scale),
                (disc_x + 54 * nib_scale, disc_y - 30 * nib_scale),
                (disc_x + 62 * nib_scale + curl_h * 0.22, disc_y + 12 * nib_scale - curl_h * 0.65),
                (disc_x + 68 * nib_scale + curl_h * 0.32, disc_y + 22 * nib_scale - curl_h),
                (disc_x + 4 * nib_scale, disc_y + 26 * nib_scale - curl_h * 0.72),
                (disc_x - 52 * nib_scale, disc_y + 28 * nib_scale - curl_h * 0.42),
            ]
            draw_ink_stroke(
                canvas, sh_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95,
                closed=True, fill_bgr=BGR_PAPER if not is_rough else None,
                boil_phase=boil_phase, seed=70 + sh_i, sharp_corners=True,
            )
            # Draw the animated Fox leap curve & cross-hatching on the active sheet (`closeup_hand_drawing_keys.jpg` & `nib_inking_keys.jpg` Q3)!
            if sh_i == 2 or p_flip > 0.25:
                fox_st = [
                    (disc_x - 38 * nib_scale, disc_y + 10 * nib_scale - curl_h * 0.3),
                    (disc_x - 10 * nib_scale, disc_y - (12 + sh_i * 5) * nib_scale - curl_h * 0.48),
                    (disc_x + 20 * nib_scale, disc_y - (10 + sh_i * 4) * nib_scale - curl_h * 0.55),
                    (disc_x + 38 * nib_scale, disc_y + 4 * nib_scale - curl_h * 0.62),
                ]
                st_col = BGR_INK if (d_prog > 0.65 and sh_i == 2 and not is_rough) else BGR_NONPHOTO_BLUE
                draw_ink_stroke(canvas, fox_st, color_bgr=st_col, base_thick=max(1.6, 2.4 * s_ui), boil_phase=boil_phase, seed=73 + sh_i)
                # Expressive fox head, pointed ear & fine cross-hatching ribs on top sheet as `draw_prog` advances!
                if sh_i == 2 and d_prog > 0.25:
                    fx_hx = disc_x + 18 * nib_scale
                    fx_hy = disc_y - 14 * nib_scale - curl_h * 0.52
                    cv2.circle(canvas, (int(fx_hx), int(fx_hy)), max(3, int(8.5 * nib_scale)), st_col, max(1, int(1.8 * s_ui)), cv2.LINE_AA)
                    ear_p = [
                        (fx_hx - 4 * nib_scale, fx_hy - 7 * nib_scale),
                        (fx_hx + 2 * nib_scale, fx_hy - 18 * nib_scale),
                        (fx_hx + 7 * nib_scale, fx_hy - 6 * nib_scale),
                    ]
                    draw_ink_stroke(canvas, ear_p, color_bgr=st_col, base_thick=max(1.4, 1.8 * s_ui), boil_phase=boil_phase, seed=76)
                    # Cross-hatching ticks along fox's torso (`nib_inking_keys.jpg` Q3)
                    if d_prog > 0.45:
                        for h_i in range(4):
                            hx_t = disc_x - (16 - h_i * 8) * nib_scale
                            hy_t = disc_y - (8 + h_i * 1.5) * nib_scale - curl_h * 0.45
                            cv2.line(canvas, (int(hx_t - 4 * nib_scale), int(hy_t + 5 * nib_scale)), (int(hx_t + 3 * nib_scale), int(hy_t - 4 * nib_scale)), st_col, max(1, int(1.3 * s_ui)), cv2.LINE_AA)
                    # Sparkle stars around finished inked sheet (`d_prog > 0.85`, `apparatus_lightbox_anim_keys.jpg` Q4)
                    if d_prog > 0.85:
                        _draw_sparkle_star(disc_x - 46 * nib_scale, disc_y - 8 * nib_scale, 7.5 * nib_scale, bg_line_col)
                        _draw_sparkle_star(disc_x + 48 * nib_scale, disc_y - 16 * nib_scale, 6.5 * nib_scale, bg_line_col)

        # Classic hand-drawn double-arc paper-flip motion lines `(( ))` when `p_flip > 0.25` (`apparatus_lightbox_anim_keys.jpg`)
        if p_flip > 0.25:
            for m_arc in range(2):
                ax_m = disc_x + (76 + m_arc * 10) * nib_scale
                ay_m = disc_y - (12 + p_flip * 26) * nib_scale
                arc_m_pts = [
                    (ax_m - 6 * nib_scale, ay_m - 18 * nib_scale),
                    (ax_m + 6 * nib_scale, ay_m),
                    (ax_m - 4 * nib_scale, ay_m + 18 * nib_scale),
                ]
                draw_ink_stroke(canvas, arc_m_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 0.8, boil_phase=boil_phase, seed=77 + m_arc)

    # 5. Raised 3D 4-Peg Brass Acme Registration Bar (`o | | | | o`) bolted along upper disc edge (`apparatus_lightbox_anim_keys.jpg`)
    peg_front = [
        (disc_x - 60 * nib_scale, disc_y - 18 * nib_scale),
        (disc_x + 56 * nib_scale, disc_y - 25 * nib_scale),
        (disc_x + 56 * nib_scale, disc_y - 19 * nib_scale),
        (disc_x - 60 * nib_scale, disc_y - 12 * nib_scale),
    ]
    peg_top = [
        (disc_x - 62 * nib_scale, disc_y - 29 * nib_scale),
        (disc_x + 54 * nib_scale, disc_y - 36 * nib_scale),
        (disc_x + 56 * nib_scale, disc_y - 25 * nib_scale),
        (disc_x - 60 * nib_scale, disc_y - 18 * nib_scale),
    ]
    draw_ink_stroke(canvas, peg_front, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=178, sharp_corners=True)
    draw_ink_stroke(canvas, peg_top, color_bgr=bg_line_col, base_thick=bg_line_w * 1.05, closed=True, fill_bgr=brass_fill, boil_phase=boil_phase, seed=78, sharp_corners=True)
    # 4 upright cylindrical registration pegs (`apparatus_lightbox_anim_keys.jpg`)
    for p_u in (-0.72, -0.24, 0.24, 0.72):
        px_peg = disc_x - 3 * nib_scale + p_u * 54.0 * nib_scale
        py_peg = disc_y - 25.5 * nib_scale - p_u * 3.5 * nib_scale
        cv2.ellipse(canvas, (int(px_peg), int(py_peg)), (max(2, int(4.0 * nib_scale)), max(1, int(2.2 * nib_scale))), -4, 0, 360, bg_line_col, -1, cv2.LINE_AA)
        cv2.line(canvas, (int(px_peg), int(py_peg)), (int(px_peg), int(py_peg - 9 * nib_scale)), bg_line_col, max(2, int(2.6 * s_ui)), cv2.LINE_AA)
        cv2.circle(canvas, (int(px_peg), int(py_peg - 9 * nib_scale)), max(1, int(1.6 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)

    # 6. Right-Edge Desk Props on the Level Side-Shelf Bracket (`apparatus_lightbox_anim_keys.jpg`):
    #    6a. Ceramic Mug of Sharpened Pencils (upper shelf corner)
    mug_x = desk_cx + 124.0 * nib_scale
    mug_y = desk_fy - 322.0 * nib_scale
    mug_pts = [
        (mug_x - 15 * nib_scale, mug_y - 20 * nib_scale),
        (mug_x + 15 * nib_scale, mug_y - 20 * nib_scale),
        (mug_x + 13 * nib_scale, mug_y + 15 * nib_scale),
        (mug_x - 13 * nib_scale, mug_y + 15 * nib_scale),
    ]
    draw_ink_stroke(canvas, mug_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.05, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=79, sharp_corners=True)
    draw_sculpted_oval(canvas, mug_x, mug_y - 20 * nib_scale, 15 * nib_scale, 4.5 * nib_scale, color_bgr=bg_line_col, thick=bg_line_w * 0.95, fill_bgr=shade_fill, boil_phase=boil_phase, seed=179)
    # 4 sharpened pencils fanning out of the mug
    for p_ang in (-20, -6, 8, 22):
        rad_p = math.radians(p_ang - 90)
        p_tx = mug_x + math.cos(rad_p) * 38.0 * nib_scale
        p_ty = (mug_y - 18 * nib_scale) + math.sin(rad_p) * 38.0 * nib_scale
        cv2.line(canvas, (int(mug_x + p_ang * 0.32 * nib_scale), int(mug_y - 20 * nib_scale)), (int(p_tx), int(p_ty)), bg_line_col, max(1, int(2.2 * s_ui)), cv2.LINE_AA)
        cv2.circle(canvas, (int(p_tx), int(p_ty)), max(1, int(1.8 * s_ui)), BGR_ACCENT_RED if (p_ang < 0 and is_colored) else bg_line_col, -1, cv2.LINE_AA)

    #    6b. Faceted Glass `INDIA INK` Bottle & Open Cap resting squarely on the Horizontal Side-Shelf (`nib_inking_keys.jpg` & `apparatus_lightbox_anim_keys.jpg`)
    #    Positioned at `ink_x = nib_ox + 302 * scale, ink_y = nib_oy - 262 * scale` so Nib's dip at `K07` (`rh_x=292, rh_y=-258`) lands right in the bottle!
    ink_x = nib_ox + 302.0 * nib_scale
    ink_y = nib_oy - 262.0 * nib_scale
    neck_pts = [
        (ink_x - 9 * nib_scale, ink_y - 24 * nib_scale),
        (ink_x + 9 * nib_scale, ink_y - 24 * nib_scale),
        (ink_x + 11 * nib_scale, ink_y - 12 * nib_scale),
        (ink_x - 11 * nib_scale, ink_y - 12 * nib_scale),
    ]
    bot_pts = [
        (ink_x - 20 * nib_scale, ink_y - 12 * nib_scale),
        (ink_x + 20 * nib_scale, ink_y - 12 * nib_scale),
        (ink_x + 22 * nib_scale, ink_y + 20 * nib_scale),
        (ink_x - 22 * nib_scale, ink_y + 20 * nib_scale),
    ]
    draw_ink_stroke(canvas, bot_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=80, sharp_corners=True)
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
    draw_ink_stroke(canvas, neck_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.05, closed=True, fill_bgr=paper_fill, boil_phase=boil_phase, seed=81, sharp_corners=True)
    draw_sculpted_oval(canvas, ink_x, ink_y - 24 * nib_scale, 9.5 * nib_scale, 3.2 * nib_scale, color_bgr=bg_line_col, thick=bg_line_w * 0.95, fill_bgr=BGR_INK, boil_phase=boil_phase, seed=181)
    # Small ribbed screw cap resting beside the bottle on the side-shelf (`apparatus_lightbox_anim_keys.jpg`)
    draw_sculpted_oval(canvas, ink_x + 31 * nib_scale, ink_y + 15 * nib_scale, 8 * nib_scale, 5 * nib_scale, color_bgr=bg_line_col, thick=bg_line_w * 0.95, fill_bgr=bg_line_col, boil_phase=boil_phase, seed=82)

    # Flying India ink droplet & concentric ripple rings during `K07..K08` (`ink_drop > 0.05`)
    if pose["ink_drop"] > 0.05:
        u_dr = float(pose["ink_drop"])
        dr_x = ink_x - 24.0 * nib_scale * u_dr
        dr_y = (ink_y - 24 * nib_scale) - 38.0 * nib_scale * math.sin(math.pi * u_dr)
        cv2.circle(canvas, (int(dr_x), int(dr_y)), max(2, int(4.5 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(dr_x + 7 * nib_scale), int(dr_y - 6 * nib_scale)), max(1, int(2.5 * nib_scale)), bg_line_col, -1, cv2.LINE_AA)

    # 8. Articulated Double-Strut Swing-Arm Architect Lamp Clamped to Upper-Left of Desk (`apparatus_lightbox_anim_keys.jpg`)
    #    Mounted at `desk_cx - 58 * scale` so it arches cleanly over the animation disc and NEVER overlaps the 4:3 monitor!
    clamp_x = desk_cx - 58.0 * nib_scale
    clamp_y = desk_fy - 344.0 * nib_scale
    lamp_sway = 4.0 * nib_scale * math.sin(frame_idx * 0.08)
    elbow_x = clamp_x - 24.0 * nib_scale + lamp_sway * 0.35
    elbow_y = clamp_y - 62.0 * nib_scale
    head_lx = clamp_x + 56.0 * nib_scale + lamp_sway
    head_ly = clamp_y - 92.0 * nib_scale

    # Warm translucent light cone in `is_colored`
    if is_colored:
        cone_ov = canvas.copy()
        cone_poly = np.array([
            [int(head_lx - 26 * nib_scale), int(head_ly + 20 * nib_scale)],
            [int(head_lx + 26 * nib_scale), int(head_ly + 20 * nib_scale)],
            [int(disc_x + 125 * nib_scale), int(disc_y + 52 * nib_scale)],
            [int(disc_x - 125 * nib_scale), int(disc_y + 52 * nib_scale)],
        ], dtype=np.int32)
        cv2.fillPoly(cone_ov, [cone_poly], (218, 248, 255), cv2.LINE_AA)
        cv2.addWeighted(cone_ov, 0.22, canvas, 0.78, 0, canvas)

    # C-Clamp base block with mechanical T-screw (`T`) on upper edge of drafting board (`apparatus_lightbox_anim_keys.jpg` Q1-Q4)
    clamp_box = [
        (clamp_x - 10 * nib_scale, clamp_y - 7 * nib_scale),
        (clamp_x + 10 * nib_scale, clamp_y - 7 * nib_scale),
        (clamp_x + 12 * nib_scale, clamp_y + 11 * nib_scale),
        (clamp_x - 12 * nib_scale, clamp_y + 11 * nib_scale),
    ]
    draw_ink_stroke(canvas, clamp_box, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=185, sharp_corners=True)
    # Threaded vertical screw & horizontal T-bar handle under C-clamp
    cv2.line(canvas, (int(clamp_x - 2 * nib_scale), int(clamp_y + 11 * nib_scale)), (int(clamp_x - 2 * nib_scale), int(clamp_y + 23 * nib_scale)), bg_line_col, max(1, int(2.2 * s_ui)), cv2.LINE_AA)
    cv2.line(canvas, (int(clamp_x - 9 * nib_scale), int(clamp_y + 23 * nib_scale)), (int(clamp_x + 5 * nib_scale), int(clamp_y + 23 * nib_scale)), bg_line_col, max(2, int(2.6 * s_ui)), cv2.LINE_AA)

    # Parallel Double-Strut Lower Arm (`clamp -> elbow`) & Upper Arm (`elbow -> lamp neck`)
    neck_lx = head_lx - 8.0 * nib_scale
    neck_ly = head_ly - 16.0 * nib_scale
    for st_off in (-4.2, 4.2):
        cv2.line(canvas, (int(clamp_x + st_off * nib_scale), int(clamp_y - 4 * nib_scale)), (int(elbow_x + st_off * nib_scale), int(elbow_y)), bg_line_col, max(1, int(2.2 * s_ui)), cv2.LINE_AA)
        cv2.line(canvas, (int(elbow_x), int(elbow_y + st_off * nib_scale)), (int(neck_lx), int(neck_ly + st_off * nib_scale)), bg_line_col, max(1, int(2.2 * s_ui)), cv2.LINE_AA)

    # Circular friction-hinge joints at base, elbow & lamp neck (`o` in `apparatus_lightbox_anim_keys.jpg`)
    for jx_l, jy_l, jr_l in [
        (clamp_x, clamp_y - 4 * nib_scale, 5.5 * nib_scale),
        (elbow_x, elbow_y, 7.5 * nib_scale),
        (neck_lx, neck_ly, 5.5 * nib_scale),
    ]:
        cv2.circle(canvas, (int(jx_l), int(jy_l)), max(3, int(jr_l)), brass_fill if is_colored else BGR_PAPER, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(jx_l), int(jy_l)), max(3, int(jr_l)), bg_line_col, max(1, int(1.9 * s_ui)), cv2.LINE_AA)
        cv2.circle(canvas, (int(jx_l), int(jy_l)), max(1, int(jr_l * 0.35)), bg_line_col, -1, cv2.LINE_AA)

    # Stepped Architect Lamp Cap + Flared Bell Shade angled down-right toward the animation disc (`apparatus_lightbox_anim_keys.jpg`)
    cap_pts = [
        (head_lx - 13 * nib_scale, head_ly - 20 * nib_scale),
        (head_lx + 11 * nib_scale, head_ly - 22 * nib_scale),
        (head_lx + 16 * nib_scale, head_ly - 11 * nib_scale),
        (head_lx - 16 * nib_scale, head_ly - 9 * nib_scale),
    ]
    draw_ink_stroke(canvas, cap_pts, color_bgr=bg_line_col, base_thick=bg_line_w, closed=True, fill_bgr=brass_fill if is_colored else shade_fill, boil_phase=boil_phase, seed=186, sharp_corners=True)
    shade_pts = [
        (head_lx - 38 * nib_scale, head_ly + 20 * nib_scale),
        (head_lx - 18 * nib_scale, head_ly - 10 * nib_scale),
        (head_lx + 18 * nib_scale, head_ly - 12 * nib_scale),
        (head_lx + 42 * nib_scale, head_ly + 16 * nib_scale),
    ]
    draw_ink_stroke(
        canvas, shade_pts, color_bgr=bg_line_col, base_thick=bg_line_w * 1.15,
        closed=True, fill_bgr=(92, 168, 118) if is_colored else paper_fill,
        boil_phase=boil_phase, seed=55, sharp_corners=True,
    )
    # Glowing bulb & classic radiating light-ray ticks (`\ | /`) under lamp shade (`apparatus_lightbox_anim_keys.jpg`)
    cv2.ellipse(canvas, (int(head_lx + 2 * nib_scale), int(head_ly + 18 * nib_scale)), (int(14 * nib_scale), int(7 * nib_scale)), -4, 0, 180, (190, 245, 255) if is_colored else BGR_PAPER, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(head_lx + 2 * nib_scale), int(head_ly + 18 * nib_scale)), (int(14 * nib_scale), int(7 * nib_scale)), -4, 0, 180, bg_line_col, max(1, int(1.6 * s_ui)), cv2.LINE_AA)
    for r_ang in (-28, -14, 0, 14, 28):
        rad_l = math.radians(84 + r_ang)
        rx0_l = head_lx + math.cos(rad_l) * 24.0 * nib_scale
        ry0_l = head_ly + 20 * nib_scale + math.sin(rad_l) * 16.0 * nib_scale
        rx1_l = head_lx + math.cos(rad_l) * 38.0 * nib_scale
        ry1_l = head_ly + 20 * nib_scale + math.sin(rad_l) * 29.0 * nib_scale
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
        # Hand-inked Leaping Fox character on the lifted cel (Dual-Pass Union so head, ears, torso & tail have ZERO internal circles)!
        fox_fill = (68, 128, 232) if is_colored else (BGR_PAPER if is_cleanup else None)
        fx_hx = int(cel_cx + cw_l * 0.36)
        fx_hy = int(cel_cy - ch_l * 0.16)
        r_fh = max(4, int(round(9.5 * nib_scale)))
        w_fx_i = max(2, int(round(2.4 * s_ui)))
        ear_poly = np.array([
            [fx_hx - int(3 * nib_scale), fx_hy - int(6 * nib_scale)],
            [fx_hx + int(2 * nib_scale), fx_hy - int(18 * nib_scale)],
            [fx_hx + int(7 * nib_scale), fx_hy - int(5 * nib_scale)],
        ], dtype=np.int32)
        snout_cap = _capsule_poly((fx_hx, fx_hy), (fx_hx + 13 * nib_scale, fx_hy - 2 * nib_scale), r_fh * 0.85, r_fh * 0.42)
        body_cap = _capsule_poly((cel_cx - cw_l * 0.35, cel_cy + ch_l * 0.14), (fx_hx - 2 * nib_scale, fx_hy + 2 * nib_scale), 7.5 * nib_scale, 8.5 * nib_scale)
        tail_cap = _capsule_poly((cel_cx - cw_l * 0.35, cel_cy + ch_l * 0.14), (cel_cx - cw_l * 0.68, cel_cy + ch_l * 0.02), 5.5 * nib_scale, 7.5 * nib_scale)
        for p_cap in (tail_cap, body_cap, snout_cap):
            cv2.polylines(canvas, [p_cap], True, BGR_INK, w_fx_i * 2, cv2.LINE_AA)
        cv2.polylines(canvas, [ear_poly], True, BGR_INK, w_fx_i * 2, cv2.LINE_AA)
        if fox_fill is not None:
            for p_cap in (tail_cap, body_cap, snout_cap):
                cv2.fillPoly(canvas, [p_cap], fox_fill, cv2.LINE_AA)
            cv2.fillPoly(canvas, [ear_poly], fox_fill, cv2.LINE_AA)
        cv2.circle(canvas, (fx_hx + int(4 * nib_scale), fx_hy - int(2 * nib_scale)), max(1, int(2.0 * nib_scale)), BGR_INK, -1, cv2.LINE_AA)
        cv2.circle(canvas, (fx_hx + int(13 * nib_scale), fx_hy - int(2 * nib_scale)), max(1, int(2.2 * nib_scale)), BGR_INK, -1, cv2.LINE_AA)
        # 4-pointed sparkle stars (`✦`) twinkling around the finished cel (`apparatus_lightbox_anim_keys.jpg` Q4)!
        _draw_sparkle_star(cel_cx - cw_l - 14 * nib_scale, cel_cy - ch_l + 8 * nib_scale, 9.0 * nib_scale, bg_line_col)
        _draw_sparkle_star(cel_cx + cw_l + 16 * nib_scale, cel_cy - ch_l * 0.45, 8.0 * nib_scale, bg_line_col)
        _draw_sparkle_star(cel_cx + cw_l + 12 * nib_scale, cel_cy + ch_l * 0.65, 6.5 * nib_scale, bg_line_col)

    # ==========================================================================
    # PLANE 1 (`depth = 0.62`): FOREGROUND 35MM FILM CANS, SPOKED FILM REEL,
    # SPROCKETED FILMSTRIP, GUILLOTINE SPLICER, KIDNEY PALETTE & GOUACHE JARS
    # (Matches `multiplane_plane1_foreground_props_keys.jpg` Q1 & Q3!)
    # ==========================================================================
    fg_lx, fg_ly, z_fg = _plane_tf(out_w * 0.09, out_h * 0.92, 0.62)
    fg_rx, fg_ry, _ = _plane_tf(out_w * 0.90, out_h * 0.92, 0.62)

    # Left Foreground (`multiplane_plane1_foreground_props_keys.jpg` Q1):
    # 1. Stacked cylindrical 35mm metal film cans with vertical rib lines
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
        for rb_u in (-0.65, -0.25, 0.25, 0.65):
            rx_rb = int(cx_can + rb_u * 44 * s_ui * z_fg)
            cv2.line(canvas, (rx_rb, int(cy_can - 6 * s_ui * z_fg)), (rx_rb, int(cy_can + 9 * s_ui * z_fg)), bg_line_col, max(1, int(1.2 * s_ui)), cv2.LINE_AA)

    # 2. Spoked 35mm Metal Film Reel resting atop the film cans (6 radial spokes + circular cutout holes `o`)
    reel_cx = fg_lx + 10 * s_ui * z_fg
    reel_cy = fg_ly - 42 * s_ui * z_fg
    draw_sculpted_oval(canvas, reel_cx, reel_cy, 44 * s_ui * z_fg, 14 * s_ui * z_fg, color_bgr=bg_line_col, thick=bg_line_w * 1.15, fill_bgr=(182, 194, 206) if is_colored else paper_fill, boil_phase=boil_phase, seed=190)
    draw_sculpted_oval(canvas, reel_cx, reel_cy, 10 * s_ui * z_fg, 3.5 * s_ui * z_fg, color_bgr=bg_line_col, thick=bg_line_w * 0.95, fill_bgr=bg_line_col, boil_phase=boil_phase, seed=191)
    for sp_i in range(6):
        ang_sp = sp_i * (math.pi / 3.0)
        sx_1 = reel_cx + math.cos(ang_sp) * 40 * s_ui * z_fg
        sy_1 = reel_cy + math.sin(ang_sp) * 12.5 * s_ui * z_fg
        cv2.line(canvas, (int(reel_cx), int(reel_cy)), (int(sx_1), int(sy_1)), bg_line_col, max(1, int(1.5 * s_ui)), cv2.LINE_AA)
        hx_c = reel_cx + math.cos(ang_sp + math.pi / 6.0) * 24 * s_ui * z_fg
        hy_c = reel_cy + math.sin(ang_sp + math.pi / 6.0) * 7.5 * s_ui * z_fg
        cv2.ellipse(canvas, (int(hx_c), int(hy_c)), (max(2, int(5 * s_ui * z_fg)), max(1, int(2 * s_ui * z_fg))), 0, 0, 360, bg_line_col, max(1, int(1.3 * s_ui)), cv2.LINE_AA)

    # 3. Unspooling S-coiled 35mm filmstrip ribbon with double rails, frame lines & sprocket dots (`· · · ·`)
    strip_top = [
        (fg_lx + 42 * s_ui * z_fg, fg_ly - 18 * s_ui * z_fg),
        (fg_lx + 92 * s_ui * z_fg, fg_ly - 32 * s_ui * z_fg),
        (fg_lx + 138 * s_ui * z_fg, fg_ly - 12 * s_ui * z_fg),
        (fg_lx + 184 * s_ui * z_fg, fg_ly - 24 * s_ui * z_fg),
    ]
    strip_bot = [(px, py + 13 * s_ui * z_fg) for px, py in strip_top]
    film_poly = strip_top + list(reversed(strip_bot))
    draw_ink_stroke(canvas, film_poly, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95, closed=True, fill_bgr=(82, 92, 104) if is_colored else shade_fill, boil_phase=boil_phase, seed=92)
    for sf_i in range(9):
        u_f = sf_i / 8.0
        idx_f = min(2, int(u_f * 3))
        t_f = u_f * 3 - idx_f
        fx_s = strip_top[idx_f][0] * (1.0 - t_f) + strip_top[idx_f + 1][0] * t_f
        fy_s = strip_top[idx_f][1] * (1.0 - t_f) + strip_top[idx_f + 1][1] * t_f
        cv2.circle(canvas, (int(fx_s), int(fy_s + 2.5 * s_ui * z_fg)), max(1, int(1.3 * s_ui)), BGR_PAPER, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(fx_s), int(fy_s + 10.5 * s_ui * z_fg)), max(1, int(1.3 * s_ui)), BGR_PAPER, -1, cv2.LINE_AA)
        if sf_i % 2 == 1:
            cv2.line(canvas, (int(fx_s), int(fy_s + 2 * s_ui * z_fg)), (int(fx_s), int(fy_s + 11 * s_ui * z_fg)), bg_line_col, max(1, int(1.3 * s_ui)), cv2.LINE_AA)

    # 4. Guillotine 35mm Film Splicer block with hinged diagonal cutter handle (`multiplane_plane1_foreground_props_keys.jpg` Q1)
    sp_x = fg_lx + 96 * s_ui * z_fg
    sp_y = fg_ly + 4 * s_ui * z_fg
    sp_box = [
        (sp_x - 22 * s_ui * z_fg, sp_y - 9 * s_ui * z_fg),
        (sp_x + 22 * s_ui * z_fg, sp_y - 9 * s_ui * z_fg),
        (sp_x + 25 * s_ui * z_fg, sp_y + 6 * s_ui * z_fg),
        (sp_x - 25 * s_ui * z_fg, sp_y + 6 * s_ui * z_fg),
    ]
    draw_ink_stroke(canvas, sp_box, color_bgr=bg_line_col, base_thick=bg_line_w * 0.95, closed=True, fill_bgr=(152, 164, 176) if is_colored else shade_fill, boil_phase=boil_phase, seed=192, sharp_corners=True)
    cv2.line(canvas, (int(sp_x - 14 * s_ui * z_fg), int(sp_y - 9 * s_ui * z_fg)), (int(sp_x + 18 * s_ui * z_fg), int(sp_y - 28 * s_ui * z_fg)), bg_line_col, max(2, int(2.8 * s_ui)), cv2.LINE_AA)

    # Right Foreground (`multiplane_plane1_foreground_props_keys.jpg` Q3):
    # 1. Kidney-Bean Shaped Artist's Palette with Thumbhole & 6 Wavy Gouache Blobs + 3 Tapered Sable Brushes
    pal_cx = fg_rx - 54 * s_ui * z_fg
    pal_cy = fg_ry + 2 * s_ui * z_fg
    draw_sculpted_oval(
        canvas, pal_cx, pal_cy,
        58 * s_ui * z_fg, 24 * s_ui * z_fg, angle_deg=-8,
        color_bgr=bg_line_col, thick=bg_line_w * 1.15,
        fill_bgr=(148, 188, 222) if is_colored else paper_fill,
        boil_phase=boil_phase, seed=93,
    )
    # Oval thumbhole cutout on right side of kidney palette
    draw_sculpted_oval(
        canvas, pal_cx + 32 * s_ui * z_fg, pal_cy + 4 * s_ui * z_fg,
        9 * s_ui * z_fg, 5 * s_ui * z_fg, angle_deg=-8,
        color_bgr=bg_line_col, thick=bg_line_w * 0.95,
        fill_bgr=(28, 32, 38) if is_colored else shade_fill,
        boil_phase=boil_phase, seed=193,
    )
    # 6 gouache paint blobs around perimeter of kidney palette
    blob_specs = [
        (-38, -4, (62, 74, 214)),
        (-24, -12, (68, 188, 242)),
        (-4, -14, (84, 178, 96)),
        (16, -12, (198, 128, 58)),
        (-30, 9, (168, 82, 168)),
        (-8, 11, (238, 244, 248)),
    ]
    for b_idx, (bx_o, by_o, b_col) in enumerate(blob_specs):
        draw_sculpted_oval(
            canvas, pal_cx + bx_o * s_ui * z_fg, pal_cy + by_o * s_ui * z_fg,
            7.5 * s_ui * z_fg, 4.2 * s_ui * z_fg,
            color_bgr=bg_line_col, thick=bg_line_w * 0.8,
            fill_bgr=b_col if is_colored else shade_fill,
            boil_phase=boil_phase, seed=200 + b_idx,
        )
    # 3 tapered sable paintbrushes resting across the palette (`multiplane_plane1_foreground_props_keys.jpg` Q3)
    for br_i in range(3):
        bx0_p = pal_cx - (48 - br_i * 8) * s_ui * z_fg
        by0_p = pal_cy + (16 - br_i * 5) * s_ui * z_fg
        bx1_p = pal_cx + (24 + br_i * 10) * s_ui * z_fg
        by1_p = pal_cy - (16 - br_i * 4) * s_ui * z_fg
        cv2.line(canvas, (int(bx0_p), int(by0_p)), (int(bx1_p), int(by1_p)), bg_line_col, max(1, int(2.2 * s_ui)), cv2.LINE_AA)
        cv2.circle(canvas, (int(bx1_p), int(by1_p)), max(2, int(2.8 * s_ui * z_fg)), BGR_ACCENT_RED if is_colored else bg_line_col, -1, cv2.LINE_AA)

    # 2. Ribbed-Cap Gouache Paint Jars with Front Labels + Glass Water Tumbler with Meniscus (`multiplane_plane1_foreground_props_keys.jpg` Q3)
    for fi_p, (jx, jy, j_col) in enumerate([
        (fg_rx + 18 * s_ui * z_fg, fg_ry - 8 * s_ui * z_fg, (52, 78, 198)),
        (fg_rx + 68 * s_ui * z_fg, fg_ry + 4 * s_ui * z_fg, (68, 168, 92)),
    ]):
        draw_sculpted_oval(
            canvas, jx, jy, 20 * s_ui * z_fg, 26 * s_ui * z_fg,
            color_bgr=bg_line_col, thick=bg_line_w * 1.15,
            fill_bgr=j_col if is_colored else shade_fill,
            boil_phase=boil_phase, seed=94 + fi_p,
        )
        # White front jar label
        cv2.rectangle(
            canvas,
            (int(jx - 12 * s_ui * z_fg), int(jy - 6 * s_ui * z_fg)),
            (int(jx + 12 * s_ui * z_fg), int(jy + 9 * s_ui * z_fg)),
            BGR_PAPER, -1,
        )
        cv2.rectangle(
            canvas,
            (int(jx - 12 * s_ui * z_fg), int(jy - 6 * s_ui * z_fg)),
            (int(jx + 12 * s_ui * z_fg), int(jy + 9 * s_ui * z_fg)),
            bg_line_col, max(1, int(1.2 * s_ui)), cv2.LINE_AA,
        )
        # Ribbed screw lid / rim at top of jar + paintbrush handle sticking out
        draw_sculpted_oval(
            canvas, jx, jy - 22 * s_ui * z_fg, 15 * s_ui * z_fg, 5.5 * s_ui * z_fg,
            color_bgr=bg_line_col, thick=bg_line_w * 0.95,
            fill_bgr=BGR_PAPER if not is_rough else None,
            boil_phase=boil_phase, seed=194 + fi_p,
        )
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
    badge_txt = f"{pose['pose_tag']}  {pose['pose_title'][:42]}"
    font_s = max(0.32, cell_w / 1180.0)
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
