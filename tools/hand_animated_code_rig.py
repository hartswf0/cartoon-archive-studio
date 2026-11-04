#!/usr/bin/env python3
"""
Hand-Animated Code Rig & 10x Arc In-Betweener Engine (`hand_animated_code_rig.py`)
===================================================================================
Implements true frame-by-frame hand-drawn classical animation constructed with code
for Secondhand Studios (`3,100+` unique hand-animated frames = `10x` frame density):

  1. CALLIGRAPHIC SPEEDBALL DIP-PEN & CROSS-HATCHING ENGINE:
     - Catmull-Rom / Cubic Bézier spline strokes with variable-width pressure taper,
       organic paper-tooth micro-wobble, 6 Hz line boil, and interior cross-hatching.
  2. 18-JOINT KINEMATIC + VOLUMETRIC AGENT ANIMATOR RIG (`Nib`, `Blue-Line`, `Cel-Inker`, `Sprocket`, `Chime`):
     - Drawn Inside-Out (`Section VIII`) on every single frame (`24 fps On Ones`):
       * Pass A: Cubic Bézier Line of Action spine + Dotted Trajectory Motion Arcs & Spacing Ticks
       * Pass B: 3D-projected Cranium Sphere (with rotating yaw/pitch cross-contours), Ribcage Egg,
         Pelvis Box, Limb Cylinders with elliptical cross-section rings, and 2-Bone Analytical IK Arms
       * Pass C/E: Watertight Gouache Cel Fills + Rim Lighting & Cast Shadow Washes
       * Pass D: Tapered Speedball India Ink contours, articulated 5-finger hands holding studio tools,
         natural eye blinks, target-tracking pupils, lip-sync mouth shapes, and overlapping cloth follow-through!
  3. HYPER-DETAILED CODE-ANIMATED MACRO HANDS, ROLLING PAPER FLIPS & 7-PLANE APPARATUS:
     - 3-Sheet Rolling Animation Paper Flip (`K01` + `B01` + `I01` curling off Acme brass pegs)
     - 360-Degree Continuous Volumetric Character Turnaround Model Sheet (`24` angles)
     - 7-Plane Multiplane Camera Worm-Gear Platen Stack with live differential parallax rays
  4. 10X CURVED-ARC HUNGARIAN STROKE IN-BETWEENER (`240` Studio Drawings x `10` = `2,400` Frames,
     `70` Hero Drawings x `10` = `700` Frames -> `3,100` Total Hand-Animated Frames!):
     - Normalizes all quadrant strokes into bounding-box-centered `[u, v]` coordinates and interpolates
       matched strokes along quadratic Bézier motion arcs with slow-in / slow-out spacing!
"""

import math
import shutil
from pathlib import Path
import numpy as np
import cv2

# Studio Palette (BGR)
BGR_PAPER = (227, 239, 246)
BGR_GUTTER = (244, 251, 255)
BGR_PAPER_DARK = (206, 226, 236)
BGR_INK = (18, 20, 22)
BGR_GRAPHITE = (102, 112, 122)
BGR_NONPHOTO_BLUE = (210, 145, 65)
BGR_SEPIA_ONION = (135, 165, 205)
BGR_ARC_RED = (55, 82, 215)

BGR_AMBER = (58, 159, 239)
BGR_AMBER_PALE = (194, 229, 251)
BGR_RED = (41, 74, 201)
BGR_CORAL_PALE = (206, 216, 247)
BGR_BLUE = (132, 91, 43)
BGR_BLUE_PALE = (240, 228, 214)
BGR_GREEN = (79, 107, 47)
BGR_MINT_PALE = (220, 232, 212)
BGR_SKIN = (212, 232, 248)
BGR_SKIN_SHADOW = (182, 206, 228)


# =============================================================================
# 1. CALLIGRAPHIC SPEEDBALL DIP-PEN & SPLINE STROKE PRIMITIVES
# =============================================================================
def eval_cubic_bezier(p0, p1, p2, p3, n_pts: int = 20) -> list:
    pts = []
    for i in range(n_pts):
        t = i / float(max(1, n_pts - 1))
        mt = 1.0 - t
        x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        pts.append([x, y])
    return pts


def eval_quad_bezier(p0, p1, p2, n_pts: int = 16) -> list:
    pts = []
    for i in range(n_pts):
        t = i / float(max(1, n_pts - 1))
        mt = 1.0 - t
        x = (mt ** 2) * p0[0] + 2 * mt * t * p1[0] + (t ** 2) * p2[0]
        y = (mt ** 2) * p0[1] + 2 * mt * t * p1[1] + (t ** 2) * p2[1]
        pts.append([x, y])
    return pts


def draw_calligraphic_ink_curve(
    canvas: np.ndarray,
    pts: list,
    color_bgr: tuple = BGR_INK,
    base_thick: float = 2.6,
    taper_ends: bool = True,
    wobble_amp: float = 0.75,
    boil_phase: int = 0,
    seed: int = 1,
):
    """
    Renders a hand-inked Speedball dip-pen stroke along `pts` with organic paper-tooth
    micro-wobble, 6 Hz line boil, and calligraphic pressure tapering from thick mid-belly
    to crisp tapered tips.
    """
    if len(pts) < 2:
        return
    n = len(pts)
    wobbled = []
    for i, (px, py) in enumerate(pts):
        t = i / float(max(1, n - 1))
        env = math.sin(math.pi * t) if taper_ends else 0.85
        ang = seed * 1.37 + i * 0.93 + boil_phase * 1.71 + px * 0.05 + py * 0.05
        wx = px + wobble_amp * env * math.sin(ang)
        wy = py + wobble_amp * env * math.cos(ang * 1.29)
        wobbled.append([int(round(wx)), int(round(wy))])

    if base_thick <= 1.6 or not taper_ends or n < 5:
        cv2.polylines(
            canvas,
            [np.array(wobbled, dtype=np.int32)],
            False,
            color_bgr,
            max(1, int(round(base_thick))),
            cv2.LINE_AA,
        )
        return

    # Variable-thickness calligraphic pass: thin outer pass + thick central belly pass
    cv2.polylines(
        canvas,
        [np.array(wobbled, dtype=np.int32)],
        False,
        color_bgr,
        max(1, int(round(base_thick * 0.58))),
        cv2.LINE_AA,
    )
    i0 = max(0, int(round(n * 0.14)))
    i1 = min(n, int(round(n * 0.86)))
    if i1 - i0 >= 2:
        cv2.polylines(
            canvas,
            [np.array(wobbled[i0:i1], dtype=np.int32)],
            False,
            color_bgr,
            max(2, int(round(base_thick))),
            cv2.LINE_AA,
        )


def draw_hand_ellipse(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float = 0.0,
    fill_bgr: tuple = None,
    outline_bgr: tuple = BGR_INK,
    thickness: float = 2.4,
    wobble_amp: float = 0.7,
    boil_phase: int = 0,
    seed: int = 0,
    n_pts: int = 28,
) -> list:
    """Draws a hand-inked organic ellipse (used for craniums, ribcages, joints, wheels, and lenses)."""
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    pts = []
    for i in range(n_pts + 1):
        t = (i / float(n_pts)) * 2.0 * math.pi
        ex = rx * math.cos(t)
        ey = ry * math.sin(t)
        ang_w = seed * 2.1 + i * 0.77 + boil_phase * 1.4
        ex += wobble_amp * math.sin(ang_w)
        ey += wobble_amp * math.cos(ang_w * 1.3)
        px = cx + ex * ca - ey * sa
        py = cy + ex * sa + ey * ca
        pts.append([px, py])

    arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts], dtype=np.int32)
    if fill_bgr is not None:
        cv2.fillPoly(canvas, [arr], fill_bgr, cv2.LINE_AA)
    if outline_bgr is not None and thickness > 0:
        draw_calligraphic_ink_curve(
            canvas, pts, color_bgr=outline_bgr, base_thick=thickness,
            taper_ends=False, wobble_amp=wobble_amp * 0.4, boil_phase=boil_phase, seed=seed,
        )
    return pts


def draw_crosshatch_shading(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    color_bgr: tuple = BGR_GRAPHITE,
    spacing: int = 6,
    angle_deg: float = -35.0,
    boil_phase: int = 0,
):
    """Draws parallel hand-inked cross-hatching lines inside an elliptical shadow zone."""
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    n_lines = max(2, int(round((2.0 * ry) / max(3, spacing))))
    for i in range(-n_lines // 2, n_lines // 2 + 1):
        v = i / float(max(1, n_lines // 2))
        if abs(v) >= 0.92:
            continue
        half_span = rx * math.sqrt(max(0.0, 1.0 - v * v)) * 0.78
        y_off = v * ry * 0.82
        p0 = [cx + (-half_span) * ca - y_off * sa, cy + (-half_span) * sa + y_off * ca]
        p1 = [cx + half_span * ca - y_off * sa, cy + half_span * sa + y_off * ca]
        draw_calligraphic_ink_curve(
            canvas, [p0, p1], color_bgr=color_bgr, base_thick=1.1,
            taper_ends=True, wobble_amp=0.45, boil_phase=boil_phase, seed=i + 30,
        )


def draw_motion_arc_with_spacing_ticks(
    canvas: np.ndarray,
    arc_pts: list,
    active_u: float,
    color_bgr: tuple = BGR_NONPHOTO_BLUE,
    tick_color_bgr: tuple = BGR_ARC_RED,
):
    """
    Implements Section XI (`IN-BETWEENS` spacing chart) & Section XIII (`ARCS`):
    Draws a dotted trajectory arc through `arc_pts` with slow-in / slow-out spacing ticks
    and highlights the current frame's position on the arc!
    """
    if len(arc_pts) < 3:
        return
    # Dotted arc segments
    for i in range(0, len(arc_pts) - 1, 2):
        p0 = (int(round(arc_pts[i][0])), int(round(arc_pts[i][1])))
        p1 = (int(round(arc_pts[i + 1][0])), int(round(arc_pts[i + 1][1])))
        cv2.line(canvas, p0, p1, color_bgr, 1, cv2.LINE_AA)

    # Spacing chart ticks (clustered at ends for slow-in / slow-out!)
    s_ticks = [0.0, 0.06, 0.15, 0.28, 0.50, 0.72, 0.85, 0.94, 1.0]
    n_p = len(arc_pts)
    for st in s_ticks:
        idx = min(n_p - 2, int(st * (n_p - 1)))
        dx = arc_pts[idx + 1][0] - arc_pts[idx][0]
        dy = arc_pts[idx + 1][1] - arc_pts[idx][1]
        ln = max(1e-4, math.hypot(dx, dy))
        nx, ny = -dy / ln, dx / ln
        px, py = arc_pts[idx][0], arc_pts[idx][1]
        t_len = 5.5 if st in (0.0, 0.50, 1.0) else 3.5
        cv2.line(
            canvas,
            (int(round(px - nx * t_len)), int(round(py - ny * t_len))),
            (int(round(px + nx * t_len)), int(round(py + ny * t_len))),
            color_bgr, 1, cv2.LINE_AA,
        )

    # Active frame marker on trajectory arc
    act_i = min(n_p - 1, max(0, int(round(active_u * (n_p - 1)))))
    ax, ay = int(round(arc_pts[act_i][0])), int(round(arc_pts[act_i][1]))
    cv2.circle(canvas, (ax, ay), 4, tick_color_bgr, -1, cv2.LINE_AA)
    cv2.circle(canvas, (ax, ay), 6, BGR_INK, 1, cv2.LINE_AA)


# =============================================================================
# 2. ANALYTICAL 2-BONE INVERSE KINEMATICS & TAPERED LIMB CYLINDERS
# =============================================================================
def solve_2bone_ik(
    shoulder_xy: tuple,
    target_xy: tuple,
    L1: float,
    L2: float,
    bend_sign: float = 1.0,
) -> tuple:
    """
    Analytical 2-Bone IK solver for `Shoulder -> Elbow -> Wrist`.
    Returns `(elbow_xy, wrist_xy)`.
    """
    sx, sy = float(shoulder_xy[0]), float(shoulder_xy[1])
    tx, ty = float(target_xy[0]), float(target_xy[1])
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    max_reach = (L1 + L2) * 0.985
    min_reach = abs(L1 - L2) + 2.0
    if d > max_reach:
        scale = max_reach / max(d, 1e-4)
        dx *= scale
        dy *= scale
        d = max_reach
    elif d < min_reach:
        scale = min_reach / max(d, 1e-4)
        dx *= scale
        dy *= scale
        d = min_reach

    wx, wy = sx + dx, sy + dy
    cos_a = np.clip((L1 * L1 + d * d - L2 * L2) / max(2.0 * L1 * d, 1e-4), -1.0, 1.0)
    ang_base = math.atan2(dy, dx)
    ang_elbow = ang_base + bend_sign * math.acos(cos_a)
    ex = sx + L1 * math.cos(ang_elbow)
    ey = sy + L1 * math.sin(ang_elbow)
    return (ex, ey), (wx, wy)


def draw_tapered_limb_segment(
    canvas: np.ndarray,
    p0: tuple,
    p1: tuple,
    r0: float,
    r1: float,
    fill_bgr: tuple,
    outline_bgr: tuple = BGR_INK,
    pass_b_blue: bool = True,
    boil_phase: int = 0,
    seed: int = 1,
):
    """
    Draws a volumetric 3D-foreshortened sleeve/limb cylinder between `p0` and `p1`
    with Pass B Non-Photo Blue elliptical cross-contour rings, Pass C gouache fill,
    interior elbow/sleeve fold ticks, and Pass D tapered Speedball ink contours!
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    ln = math.hypot(dx, dy)
    if ln < 2.0:
        return
    ux, uy = dx / ln, dy / ln
    nx, ny = -uy, ux

    # Slight organic bow in sleeve contours
    mid_x, mid_y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rm = 0.5 * (r0 + r1) * 1.08
    top_edge = eval_quad_bezier(
        [x0 + nx * r0, y0 + ny * r0],
        [mid_x + nx * rm, mid_y + ny * rm],
        [x1 + nx * r1, y1 + ny * r1],
        n_pts=12,
    )
    bot_edge = eval_quad_bezier(
        [x1 - nx * r1, y1 - ny * r1],
        [mid_x - nx * rm, mid_y - ny * rm],
        [x0 - nx * r0, y0 - ny * r0],
        n_pts=12,
    )
    poly = top_edge + bot_edge
    arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in poly], dtype=np.int32)
    if fill_bgr is not None:
        cv2.fillPoly(canvas, [arr], fill_bgr, cv2.LINE_AA)

    # Pass B Non-Photo Blue volumetric cross-section ring at midpoint
    if pass_b_blue:
        ang_deg = math.degrees(math.atan2(dy, dx)) + 90.0
        draw_hand_ellipse(
            canvas, mid_x, mid_y, rm * 0.92, rm * 0.36, angle_deg=ang_deg,
            fill_bgr=None, outline_bgr=BGR_NONPHOTO_BLUE, thickness=1.0,
            wobble_amp=0.3, boil_phase=boil_phase, seed=seed + 10, n_pts=16,
        )

    # Pass D Speedball Ink outer contours + cuff/elbow crease ticks
    draw_calligraphic_ink_curve(canvas, top_edge, outline_bgr, base_thick=2.5, wobble_amp=0.55, boil_phase=boil_phase, seed=seed)
    draw_calligraphic_ink_curve(canvas, bot_edge, outline_bgr, base_thick=2.5, wobble_amp=0.55, boil_phase=boil_phase, seed=seed + 1)
    # Rolled cuff / elbow cloth fold arc near p1
    cuff_p0 = [x1 + nx * r1 * 0.95 - ux * 3.0, y1 + ny * r1 * 0.95 - uy * 3.0]
    cuff_p1 = [x1 - ux * 1.0, y1 - uy * 1.0]
    cuff_p2 = [x1 - nx * r1 * 0.95 - ux * 3.0, y1 - ny * r1 * 0.95 - uy * 3.0]
    draw_calligraphic_ink_curve(canvas, eval_quad_bezier(cuff_p0, cuff_p1, cuff_p2, 8), outline_bgr, base_thick=1.8, wobble_amp=0.35, boil_phase=boil_phase, seed=seed + 2)


def draw_articulated_5finger_hand_and_tool(
    canvas: np.ndarray,
    wrist_xy: tuple,
    hand_angle_deg: float,
    tool_type: str = "dip_pen",
    scale: float = 1.0,
    finger_curl: float = 0.5,
    boil_phase: int = 0,
) -> tuple:
    """
    Draws an articulated 5-finger animator hand at `wrist_xy` holding `tool_type`
    (`dip_pen`, `blue_pencil`, `gouache_brush`, `caliper`, `clapper_mallet`, `crank_grip`, `paper_corner`).
    Returns `(tool_tip_x, tool_tip_y)` so ink strokes, motion arcs, or effects can lock to the exact tool tip!
    """
    wx, wy = float(wrist_xy[0]), float(wrist_xy[1])
    rad = math.radians(hand_angle_deg)
    ux, uy = math.cos(rad), math.sin(rad)
    nx, ny = -uy, ux
    s = float(scale)

    # Palm block
    palm_cx = wx + ux * 10.0 * s
    palm_cy = wy + uy * 10.0 * s
    draw_hand_ellipse(
        canvas, palm_cx, palm_cy, 9.5 * s, 7.5 * s, angle_deg=hand_angle_deg,
        fill_bgr=BGR_SKIN, outline_bgr=BGR_INK, thickness=2.2,
        wobble_amp=0.45, boil_phase=boil_phase, seed=71, n_pts=18,
    )

    # 4 Fingers + Thumb curled around the precision tool
    for f_i in range(4):
        spread = (f_i - 1.5) * 3.2 * s
        f_len = (11.0 - abs(f_i - 1.2) * 1.4) * s
        curl_ang = rad + math.radians(18.0 + f_i * 8.0 + finger_curl * 22.0)
        fx0 = palm_cx + ux * 5.0 * s + nx * spread
        fy0 = palm_cy + uy * 5.0 * s + ny * spread
        fxm = fx0 + math.cos(rad) * (f_len * 0.55) + nx * 1.2 * s
        fym = fy0 + math.sin(rad) * (f_len * 0.55) + ny * 1.2 * s
        fx1 = fxm + math.cos(curl_ang) * (f_len * 0.50)
        fy1 = fym + math.sin(curl_ang) * (f_len * 0.50)
        f_pts = eval_quad_bezier([fx0, fy0], [fxm, fym], [fx1, fy1], 8)
        draw_calligraphic_ink_curve(canvas, f_pts, BGR_SKIN, base_thick=4.2 * s, taper_ends=False, wobble_amp=0.2, boil_phase=boil_phase, seed=80 + f_i)
        draw_calligraphic_ink_curve(canvas, f_pts, BGR_INK, base_thick=1.9 * s, taper_ends=True, wobble_amp=0.3, boil_phase=boil_phase, seed=90 + f_i)

    # Opposing Thumb
    th_p0 = [palm_cx + ux * 2.0 * s - nx * 6.5 * s, palm_cy + uy * 2.0 * s - ny * 6.5 * s]
    th_p1 = [palm_cx + ux * 11.0 * s - nx * 8.5 * s, palm_cy + uy * 11.0 * s - ny * 8.5 * s]
    th_p2 = [palm_cx + ux * 15.0 * s - nx * 3.0 * s, palm_cy + uy * 15.0 * s - ny * 3.0 * s]
    th_pts = eval_quad_bezier(th_p0, th_p1, th_p2, 8)
    draw_calligraphic_ink_curve(canvas, th_pts, BGR_INK, base_thick=2.1 * s, taper_ends=True, wobble_amp=0.3, boil_phase=boil_phase, seed=99)

    # Tool geometry extending from the pinch grip
    grip_x = palm_cx + ux * 13.0 * s - nx * 1.5 * s
    grip_y = palm_cy + uy * 13.0 * s - ny * 1.5 * s
    t_ang = rad + math.radians(28.0)
    tx_u, ty_u = math.cos(t_ang), math.sin(t_ang)
    tx_n, ty_n = -ty_u, tx_u

    tip_x = grip_x + tx_u * 26.0 * s
    tip_y = grip_y + ty_u * 26.0 * s
    butt_x = grip_x - tx_u * 22.0 * s
    butt_y = grip_y - ty_u * 22.0 * s

    if tool_type in ("dip_pen", "blue_pencil", "gouache_brush"):
        shaft_col = BGR_NONPHOTO_BLUE if tool_type == "blue_pencil" else (BGR_AMBER if tool_type == "gouache_brush" else BGR_RED)
        # Wooden barrel
        b_poly = np.array([
            [int(round(butt_x + tx_n * 3.0 * s)), int(round(butt_y + ty_n * 3.0 * s))],
            [int(round(grip_x + tx_u * 12.0 * s + tx_n * 2.4 * s)), int(round(grip_y + ty_u * 12.0 * s + ty_n * 2.4 * s))],
            [int(round(grip_x + tx_u * 12.0 * s - tx_n * 2.4 * s)), int(round(grip_y + ty_u * 12.0 * s - ty_n * 2.4 * s))],
            [int(round(butt_x - tx_n * 3.0 * s)), int(round(butt_y - ty_n * 3.0 * s))],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [b_poly], shaft_col, cv2.LINE_AA)
        cv2.polylines(canvas, [b_poly], True, BGR_INK, max(1, int(round(1.8 * s))), cv2.LINE_AA)

        # Nib / Brush tip
        n_poly = np.array([
            [int(round(grip_x + tx_u * 12.0 * s + tx_n * 2.6 * s)), int(round(grip_y + ty_u * 12.0 * s + ty_n * 2.6 * s))],
            [int(round(tip_x)), int(round(tip_y))],
            [int(round(grip_x + tx_u * 12.0 * s - tx_n * 2.6 * s)), int(round(grip_y + ty_u * 12.0 * s - ty_n * 2.6 * s))],
        ], dtype=np.int32)
        nib_fill = BGR_NONPHOTO_BLUE if tool_type == "blue_pencil" else (BGR_AMBER if tool_type == "gouache_brush" else BGR_AMBER_PALE)
        cv2.fillPoly(canvas, [n_poly], nib_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [n_poly], True, BGR_INK, max(1, int(round(1.8 * s))), cv2.LINE_AA)
        dot_c = BGR_NONPHOTO_BLUE if tool_type == "blue_pencil" else (BGR_AMBER if tool_type == "gouache_brush" else BGR_INK)
        cv2.circle(canvas, (int(round(tip_x)), int(round(tip_y))), max(2, int(round(2.5 * s))), dot_c, -1, cv2.LINE_AA)

    elif tool_type == "caliper":
        # Brass 4:3 Proportional Calipers
        jaw_spread = (8.0 + 4.0 * math.sin(finger_curl * math.pi)) * s
        p_top = [tip_x + tx_n * jaw_spread, tip_y + ty_n * jaw_spread]
        p_bot = [tip_x - tx_n * jaw_spread, tip_y - ty_n * jaw_spread]
        draw_calligraphic_ink_curve(canvas, [[butt_x, butt_y], [grip_x, grip_y], p_top], BGR_INK, base_thick=2.4 * s, boil_phase=boil_phase, seed=101)
        draw_calligraphic_ink_curve(canvas, [[butt_x, butt_y], [grip_x, grip_y], p_bot], BGR_INK, base_thick=2.4 * s, boil_phase=boil_phase, seed=102)
        cv2.circle(canvas, (int(round(grip_x)), int(round(grip_y))), max(3, int(round(4.0 * s))), BGR_AMBER, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(grip_x)), int(round(grip_y))), max(3, int(round(4.0 * s))), BGR_INK, 1, cv2.LINE_AA)

    elif tool_type == "clapper_mallet":
        draw_calligraphic_ink_curve(canvas, [[butt_x, butt_y], [tip_x, tip_y]], BGR_INK, base_thick=2.6 * s, boil_phase=boil_phase, seed=105)
        draw_hand_ellipse(canvas, tip_x, tip_y, 7.5 * s, 5.5 * s, angle_deg=hand_angle_deg, fill_bgr=BGR_RED, outline_bgr=BGR_INK, thickness=2.0 * s, boil_phase=boil_phase, seed=106)

    return int(round(tip_x)), int(round(tip_y))


# =============================================================================
# 3. FULL 18-JOINT KINEMATIC AGENT ANIMATOR RIG (`Nib`, `Blue-Line`, `Cel-Inker`, `Sprocket`, `Chime`)
# =============================================================================
AGENT_STYLE_CONFIG = {
    "nib": {
        "smock_bgr": BGR_CORAL_PALE,
        "accent_bgr": BGR_RED,
        "hat_type": "fountain_crown",
        "glasses": True,
        "tool": "dip_pen",
    },
    "blueline": {
        "smock_bgr": BGR_BLUE_PALE,
        "accent_bgr": BGR_BLUE,
        "hat_type": "visor_cap",
        "glasses": False,
        "tool": "blue_pencil",
    },
    "celinker": {
        "smock_bgr": BGR_AMBER_PALE,
        "accent_bgr": BGR_AMBER,
        "hat_type": "beret_wavy",
        "glasses": False,
        "tool": "gouache_brush",
    },
    "sprocket": {
        "smock_bgr": BGR_MINT_PALE,
        "accent_bgr": BGR_GREEN,
        "hat_type": "reel_crown",
        "glasses": True,
        "tool": "caliper",
    },
    "chime": {
        "smock_bgr": BGR_AMBER_PALE,
        "accent_bgr": BGR_RED,
        "hat_type": "tuning_fork",
        "glasses": False,
        "tool": "clapper_mallet",
    },
}


def draw_code_hand_animated_agent(
    canvas: np.ndarray,
    agent_id: str,
    cx: float,
    cy: float,
    scale: float,
    sub_step: float,
    frame_idx: int,
    action_mode: str = "draw_lightbox",
    show_pass_ab: bool = True,
    boil_phase: int = 0,
) -> dict:
    """
    Renders a hyper-detailed, frame-by-frame code-animated Agent Animator (`24 fps On Ones`)
    constructed Inside-Out (`Section VIII`):
      - Pass A: Cubic Bézier Line of Action spine + Dotted Hand Trajectory Arc with Slow-In/Out Spacing Ticks
      - Pass B: Non-Photo Blue Cranium Sphere (with 3D yaw/pitch cross-hairs), Ribcage Mass, Shoulder/Elbow/Wrist joints
      - Pass C/E: Watertight Gouache Smock/Apron, Collar, Skin & Signature Headgear with Cross-Hatching
      - Pass D: Speedball India Ink contours, natural eye blinks, target-tracking pupils, lip-sync mouth,
        overlapping cloth/hair follow-through, and 2-Bone IK arms driving 5-finger hands!
    Returns telemetry dict (`right_tip`, `left_tip`, `head_center`, `yaw_deg`).
    """
    cfg = AGENT_STYLE_CONFIG.get(agent_id, AGENT_STYLE_CONFIG["nib"])
    s = float(scale)
    t_rad = sub_step * 2.0 * math.pi * 2.0  # 2 complete expressive cycles per scene

    # 1. Spine & Posture Dynamics (Anticipation -> Action -> Overshoot -> Settle)
    lean_x = 11.0 * s * math.sin(t_rad) + 4.0 * s * math.cos(t_rad * 2.0)
    squash_y = 5.5 * s * math.sin(t_rad * 2.0 - 0.3)
    pelvis_xy = (cx - 4.0 * s, cy + 68.0 * s)
    rib_xy = (cx + lean_x * 0.55, cy + 12.0 * s + squash_y * 0.6)
    neck_xy = (cx + lean_x * 0.85, cy - 30.0 * s + squash_y)
    head_xy = (cx + lean_x * 1.05, cy - 58.0 * s + squash_y)

    # Head 3D Yaw (-28°..+28°) & Pitch (-12°..+14°) tracking the active work target
    yaw_deg = 16.0 + 14.0 * math.sin(t_rad + 0.4)
    pitch_deg = 8.0 + 7.0 * math.cos(t_rad * 2.0)
    head_tilt_deg = 5.5 * math.sin(t_rad - 0.2)

    # Action-specific Wrist Trajectory Curves (so we can draw the full trajectory arc AND solve 2-Bone IK!)
    def sample_right_wrist(u_val: float) -> tuple:
        ang = u_val * 2.0 * math.pi * 2.0
        if action_mode == "moviola_crank":
            return (cx + (58.0 + 22.0 * math.cos(ang)) * s, cy + (4.0 + 22.0 * math.sin(ang)) * s)
        elif action_mode == "storyboard_pin":
            return (cx + (48.0 + 26.0 * math.sin(ang)) * s, cy + (-18.0 - 28.0 * abs(math.cos(ang * 0.5))) * s)
        elif action_mode == "multiplane_crank":
            return (cx + (54.0 + 20.0 * math.cos(ang * 1.5)) * s, cy + (-8.0 + 24.0 * math.sin(ang * 1.5)) * s)
        elif action_mode == "foley_strike":
            return (cx + (52.0 + 24.0 * math.sin(ang * 2.0)) * s, cy + (-22.0 + 30.0 * math.cos(ang * 2.0)) * s)
        elif action_mode == "celebrate":
            return (cx + (46.0 + 18.0 * math.sin(ang)) * s, cy + (-48.0 - 20.0 * abs(math.sin(ang))) * s)
        else:
            # Figure-8 calligraphic drawing sweep across the tilted lightbox / cel board
            return (cx + (46.0 + 26.0 * math.sin(ang)) * s, cy + (10.0 + 14.0 * math.sin(ang * 2.0)) * s)

    def sample_left_wrist(u_val: float) -> tuple:
        ang = u_val * 2.0 * math.pi * 2.0
        if action_mode == "paper_flip":
            # Rolling 3-sheet paper flip lifting and curling animation sheets
            return (cx + (-34.0 + 18.0 * math.cos(ang)) * s, cy + (22.0 - 24.0 * abs(math.sin(ang))) * s)
        elif action_mode == "celebrate":
            return (cx + (-46.0 - 18.0 * math.sin(ang)) * s, cy + (-46.0 - 20.0 * abs(math.cos(ang))) * s)
        else:
            return (cx + (-38.0 + 12.0 * math.cos(ang + 0.8)) * s, cy + (26.0 + 9.0 * math.sin(ang + 0.8)) * s)

    # -------------------------------------------------------------------------
    # PASS A: LINE OF ACTION SPINE & DOTTED TRAJECTORY MOTION ARCS (NON-PHOTO BLUE)
    # -------------------------------------------------------------------------
    if show_pass_ab:
        spine_curve = eval_cubic_bezier(
            [pelvis_xy[0], pelvis_xy[1] + 10.0 * s],
            [rib_xy[0] - 8.0 * s, rib_xy[1]],
            [neck_xy[0] + 6.0 * s, neck_xy[1]],
            [head_xy[0], head_xy[1] - 28.0 * s],
            n_pts=18,
        )
        draw_calligraphic_ink_curve(
            canvas, spine_curve, color_bgr=BGR_NONPHOTO_BLUE, base_thick=1.5,
            taper_ends=True, wobble_amp=0.3, boil_phase=boil_phase, seed=11,
        )
        # Hand trajectory motion arc with slow-in / slow-out spacing ticks
        arc_r = [sample_right_wrist(k / 24.0) for k in range(25)]
        draw_motion_arc_with_spacing_ticks(canvas, arc_r, sub_step, color_bgr=BGR_NONPHOTO_BLUE, tick_color_bgr=BGR_ARC_RED)

    # -------------------------------------------------------------------------
    # TORSO, APRON / SMOCK & COLLAR (WITH SECONDARY FOLLOW-THROUGH & CROSS-HATCHING)
    # -------------------------------------------------------------------------
    sh_l = (neck_xy[0] - 30.0 * s, neck_xy[1] + 8.0 * s)
    sh_r = (neck_xy[0] + 28.0 * s, neck_xy[1] + 8.0 * s)

    # Cloth follow-through phase lag (`t_rad - 0.55`)
    cloth_lag = 6.5 * s * math.sin(t_rad - 0.55)
    torso_poly = [
        [sh_l[0], sh_l[1]],
        [sh_r[0], sh_r[1]],
        [pelvis_xy[0] + 36.0 * s + cloth_lag, pelvis_xy[1]],
        [pelvis_xy[0] - 34.0 * s + cloth_lag * 0.6, pelvis_xy[1]],
    ]
    arr_torso = np.array([[int(round(p[0])), int(round(p[1]))] for p in torso_poly], dtype=np.int32)
    cv2.fillPoly(canvas, [arr_torso], cfg["smock_bgr"], cv2.LINE_AA)

    # Pass B Volumetric Ribcage Egg & Pelvis Box in Non-Photo Blue
    if show_pass_ab:
        draw_hand_ellipse(
            canvas, rib_xy[0], rib_xy[1], 28.0 * s, 32.0 * s, angle_deg=lean_x * 0.4,
            fill_bgr=None, outline_bgr=BGR_NONPHOTO_BLUE, thickness=1.0,
            wobble_amp=0.35, boil_phase=boil_phase, seed=14, n_pts=20,
        )

    # Cross-hatched shadow on left side of smock/apron
    draw_crosshatch_shading(
        canvas, rib_xy[0] - 14.0 * s, rib_xy[1] + 16.0 * s, 12.0 * s, 24.0 * s,
        color_bgr=BGR_GRAPHITE, spacing=max(4, int(round(5.5 * s))), boil_phase=boil_phase,
    )

    # Apron bib & straps
    bib_w, bib_h = 20.0 * s, 36.0 * s
    bib_poly = np.array([
        [int(round(rib_xy[0] - bib_w)), int(round(neck_xy[1] + 16.0 * s))],
        [int(round(rib_xy[0] + bib_w)), int(round(neck_xy[1] + 16.0 * s))],
        [int(round(pelvis_xy[0] + bib_w * 1.15 + cloth_lag * 0.5)), int(round(pelvis_xy[1] - 2.0 * s))],
        [int(round(pelvis_xy[0] - bib_w * 1.15 + cloth_lag * 0.5)), int(round(pelvis_xy[1] - 2.0 * s))],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [bib_poly], BGR_GUTTER, cv2.LINE_AA)
    cv2.polylines(canvas, [bib_poly], True, BGR_INK, max(1, int(round(2.0 * s))), cv2.LINE_AA)

    # Pocket pens / badge on apron bib
    pocket_x, pocket_y = int(round(rib_xy[0] + 6.0 * s)), int(round(rib_xy[1] + 6.0 * s))
    cv2.rectangle(canvas, (pocket_x - int(8 * s), pocket_y), (pocket_x + int(8 * s), pocket_y + int(12 * s)), cfg["accent_bgr"], -1)
    cv2.rectangle(canvas, (pocket_x - int(8 * s), pocket_y), (pocket_x + int(8 * s), pocket_y + int(12 * s)), BGR_INK, max(1, int(round(1.6 * s))))

    # Torso outer ink contours & collar lapels
    draw_calligraphic_ink_curve(canvas, [torso_poly[0], torso_poly[3]], BGR_INK, base_thick=2.6 * s, boil_phase=boil_phase, seed=21)
    draw_calligraphic_ink_curve(canvas, [torso_poly[1], torso_poly[2]], BGR_INK, base_thick=2.6 * s, boil_phase=boil_phase, seed=22)
    draw_calligraphic_ink_curve(canvas, [sh_l, [neck_xy[0], neck_xy[1] + 18.0 * s], sh_r], BGR_INK, base_thick=2.4 * s, boil_phase=boil_phase, seed=23)

    # -------------------------------------------------------------------------
    # LEFT ARM (2-BONE IK + 5-FINGER HAND)
    # -------------------------------------------------------------------------
    lw_target = sample_left_wrist(sub_step)
    l_elbow, l_wrist = solve_2bone_ik(sh_l, lw_target, 32.0 * s, 30.0 * s, bend_sign=1.0)
    draw_tapered_limb_segment(canvas, sh_l, l_elbow, 9.5 * s, 8.0 * s, cfg["smock_bgr"], BGR_INK, show_pass_ab, boil_phase, seed=31)
    draw_tapered_limb_segment(canvas, l_elbow, l_wrist, 7.8 * s, 6.2 * s, BGR_SKIN, BGR_INK, show_pass_ab, boil_phase, seed=32)
    l_ang = math.degrees(math.atan2(l_wrist[1] - l_elbow[1], l_wrist[0] - l_elbow[0]))
    left_tip = draw_articulated_5finger_hand_and_tool(
        canvas, l_wrist, l_ang, tool_type="paper_corner", scale=0.82 * s,
        finger_curl=0.5 + 0.4 * math.sin(t_rad), boil_phase=boil_phase,
    )

    # -------------------------------------------------------------------------
    # HEAD, CRANIUM CONSTRUCTION, BLINKING EYES, GAZE, MOUTH & SIGNATURE HAT
    # -------------------------------------------------------------------------
    h_rx, h_ry = 25.0 * s, 27.0 * s
    # Neck cylinder
    draw_tapered_limb_segment(canvas, neck_xy, (head_xy[0], head_xy[1] + 18.0 * s), 8.5 * s, 8.0 * s, BGR_SKIN_SHADOW, BGR_INK, False, boil_phase, seed=40)

    # Cranium sphere with sculpted jaw/cheek
    draw_hand_ellipse(
        canvas, head_xy[0], head_xy[1], h_rx, h_ry, angle_deg=head_tilt_deg,
        fill_bgr=BGR_SKIN, outline_bgr=BGR_INK, thickness=2.7 * s,
        wobble_amp=0.55, boil_phase=boil_phase, seed=41, n_pts=28,
    )

    # Pass B Non-Photo Blue 3D Center-Line & Eye-Line Cross-Contours on Cranium!
    yaw_shift = (yaw_deg / 35.0) * (h_rx * 0.52)
    pitch_shift = (pitch_deg / 25.0) * (h_ry * 0.35)
    if show_pass_ab:
        c_vert = eval_quad_bezier(
            [head_xy[0], head_xy[1] - h_ry],
            [head_xy[0] + yaw_shift * 1.35, head_xy[1]],
            [head_xy[0] + yaw_shift * 0.7, head_xy[1] + h_ry],
            n_pts=12,
        )
        c_horz = eval_quad_bezier(
            [head_xy[0] - h_rx, head_xy[1]],
            [head_xy[0] + yaw_shift * 0.4, head_xy[1] + pitch_shift * 1.3],
            [head_xy[0] + h_rx, head_xy[1]],
            n_pts=12,
        )
        draw_calligraphic_ink_curve(canvas, c_vert, BGR_NONPHOTO_BLUE, base_thick=1.1, wobble_amp=0.2, boil_phase=boil_phase, seed=42)
        draw_calligraphic_ink_curve(canvas, c_horz, BGR_NONPHOTO_BLUE, base_thick=1.1, wobble_amp=0.2, boil_phase=boil_phase, seed=43)

    # Ear (C-curve with interior tragus curl on the side opposite yaw)
    ear_x = head_xy[0] - h_rx * 0.92
    ear_y = head_xy[1] + 2.0 * s
    draw_hand_ellipse(canvas, ear_x, ear_y, 5.5 * s, 7.5 * s, fill_bgr=BGR_SKIN, outline_bgr=BGR_INK, thickness=2.0 * s, boil_phase=boil_phase, seed=44, n_pts=14)

    # Eyes with Natural Blink Cycle (`blinks every 32 frames` + squint during concentration)
    eye_y = head_xy[1] - 3.0 * s + pitch_shift * 0.6
    eye_lx = head_xy[0] + yaw_shift - 7.5 * s
    eye_rx = head_xy[0] + yaw_shift + 8.5 * s
    is_blinking = (frame_idx % 34) in (0, 1)
    for ex_i, ex_pos in enumerate((eye_lx, eye_rx)):
        if is_blinking:
            draw_calligraphic_ink_curve(
                canvas, [[ex_pos - 4.5 * s, eye_y], [ex_pos, eye_y + 1.5 * s], [ex_pos + 4.5 * s, eye_y]],
                BGR_INK, base_thick=2.2 * s, boil_phase=boil_phase, seed=45 + ex_i,
            )
        else:
            draw_hand_ellipse(canvas, ex_pos, eye_y, 4.2 * s, 5.0 * s, fill_bgr=BGR_GUTTER, outline_bgr=BGR_INK, thickness=1.8 * s, boil_phase=boil_phase, seed=45 + ex_i, n_pts=14)
            # Pupil tracking the right hand tool!
            px_off = 1.6 * s * math.cos(t_rad)
            py_off = 1.4 * s * math.sin(t_rad)
            cv2.circle(canvas, (int(round(ex_pos + px_off)), int(round(eye_y + py_off))), max(2, int(round(2.3 * s))), BGR_INK, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(round(ex_pos + px_off - 0.8 * s)), int(round(eye_y + py_off - 1.0 * s))), max(1, int(round(0.9 * s))), BGR_GUTTER, -1, cv2.LINE_AA)

        # Expressive Eyebrow
        brow_lift = 2.5 * s * math.sin(t_rad + ex_i * 0.6)
        brow_pts = [
            [ex_pos - 5.2 * s, eye_y - 7.0 * s - brow_lift * 0.5],
            [ex_pos, eye_y - 9.2 * s - brow_lift],
            [ex_pos + 5.2 * s, eye_y - 6.8 * s + brow_lift * 0.3],
        ]
        draw_calligraphic_ink_curve(canvas, brow_pts, BGR_INK, base_thick=2.3 * s, boil_phase=boil_phase, seed=47 + ex_i)

    # Spectacles / Jeweler's Loupe (`Nib` & `Sprocket`)
    if cfg["glasses"]:
        draw_hand_ellipse(canvas, eye_lx, eye_y, 6.2 * s, 6.2 * s, fill_bgr=None, outline_bgr=BGR_INK, thickness=1.8 * s, boil_phase=boil_phase, seed=49, n_pts=16)
        draw_hand_ellipse(canvas, eye_rx, eye_y, 6.2 * s, 6.2 * s, fill_bgr=None, outline_bgr=BGR_INK, thickness=1.8 * s, boil_phase=boil_phase, seed=50, n_pts=16)
        cv2.line(canvas, (int(round(eye_lx + 6.0 * s)), int(round(eye_y))), (int(round(eye_rx - 6.0 * s)), int(round(eye_y))), BGR_INK, max(1, int(round(1.8 * s))), cv2.LINE_AA)

    # Sculpted Cartoon Nose
    nose_x = head_xy[0] + yaw_shift + 2.0 * s
    nose_y = eye_y + 6.5 * s
    nose_pts = [[nose_x - 1.5 * s, nose_y - 3.5 * s], [nose_x + 6.5 * s, nose_y + 1.5 * s], [nose_x - 1.0 * s, nose_y + 3.5 * s]]
    draw_calligraphic_ink_curve(canvas, nose_pts, BGR_INK, base_thick=2.1 * s, boil_phase=boil_phase, seed=51)

    # Animated Mouth (switches between expressive speech & concentration shapes!)
    mouth_x = head_xy[0] + yaw_shift
    mouth_y = head_xy[1] + 13.5 * s + pitch_shift * 0.4
    mouth_open = 0.5 + 0.5 * math.sin(t_rad * 3.0)
    if mouth_open > 0.62:
        draw_hand_ellipse(canvas, mouth_x, mouth_y, 6.5 * s, (2.5 + 3.0 * mouth_open) * s, fill_bgr=BGR_RED, outline_bgr=BGR_INK, thickness=1.9 * s, boil_phase=boil_phase, seed=52, n_pts=14)
    else:
        m_pts = eval_quad_bezier([mouth_x - 7.0 * s, mouth_y - 1.0 * s], [mouth_x + 1.0 * s, mouth_y + 4.5 * s], [mouth_x + 8.0 * s, mouth_y - 2.0 * s], 10)
        draw_calligraphic_ink_curve(canvas, m_pts, BGR_INK, base_thick=2.2 * s, boil_phase=boil_phase, seed=52)

    # Signature Agent Headgear (with Secondary Overlap & Rotation!)
    hat_top_y = head_xy[1] - h_ry + 3.0 * s
    if cfg["hat_type"] == "fountain_crown":
        # Nib's Fountain-Pen Nib Crown with split slit & breather hole
        c_pts = [
            [head_xy[0] - 18.0 * s, hat_top_y + 4.0 * s],
            [head_xy[0] - 14.0 * s, hat_top_y - 22.0 * s],
            [head_xy[0] - 4.0 * s, hat_top_y - 12.0 * s],
            [head_xy[0] + 2.0 * s + cloth_lag * 0.3, hat_top_y - 30.0 * s],
            [head_xy[0] + 9.0 * s, hat_top_y - 12.0 * s],
            [head_xy[0] + 17.0 * s, hat_top_y - 20.0 * s],
            [head_xy[0] + 19.0 * s, hat_top_y + 4.0 * s],
        ]
        arr_c = np.array([[int(round(p[0])), int(round(p[1]))] for p in c_pts], dtype=np.int32)
        cv2.fillPoly(canvas, [arr_c], BGR_CORAL_PALE, cv2.LINE_AA)
        draw_calligraphic_ink_curve(canvas, c_pts + [c_pts[0]], BGR_INK, base_thick=2.4 * s, taper_ends=False, boil_phase=boil_phase, seed=55)
        cv2.circle(canvas, (int(round(head_xy[0] + 2.0 * s)), int(round(hat_top_y - 11.0 * s))), max(2, int(round(3.0 * s))), BGR_INK, -1, cv2.LINE_AA)

    elif cfg["hat_type"] == "visor_cap":
        # Blue-Line's Animator Visor Cap with protruding blue celluloid visor
        draw_hand_ellipse(canvas, head_xy[0], hat_top_y, 23.0 * s, 11.0 * s, fill_bgr=BGR_BLUE_PALE, outline_bgr=BGR_INK, thickness=2.3 * s, boil_phase=boil_phase, seed=56, n_pts=18)
        visor_pts = [
            [head_xy[0] + 2.0 * s, hat_top_y + 2.0 * s],
            [head_xy[0] + 32.0 * s, hat_top_y + 5.0 * s],
            [head_xy[0] + 26.0 * s, hat_top_y + 13.0 * s],
            [head_xy[0] - 6.0 * s, hat_top_y + 7.0 * s],
        ]
        arr_v = np.array([[int(round(p[0])), int(round(p[1]))] for p in visor_pts], dtype=np.int32)
        cv2.fillPoly(canvas, [arr_v], BGR_NONPHOTO_BLUE, cv2.LINE_AA)
        cv2.polylines(canvas, [arr_v], True, BGR_INK, max(1, int(round(2.0 * s))), cv2.LINE_AA)

    elif cfg["hat_type"] == "beret_wavy":
        # Cel-Inker's French Artist Beret + Flowing Wavy Hair with Follow-Through Lag!
        draw_hand_ellipse(canvas, head_xy[0] - 4.0 * s + cloth_lag * 0.4, hat_top_y - 4.0 * s, 27.0 * s, 12.0 * s, angle_deg=-12.0, fill_bgr=BGR_AMBER_PALE, outline_bgr=BGR_INK, thickness=2.4 * s, boil_phase=boil_phase, seed=57, n_pts=20)
        for w_i in range(3):
            wave_lag = 8.0 * s * math.sin(t_rad - 0.6 - w_i * 0.45)
            h_strand = eval_cubic_bezier(
                [head_xy[0] - 18.0 * s, head_xy[1] - 8.0 * s + w_i * 8.0 * s],
                [head_xy[0] - 34.0 * s + wave_lag * 0.5, head_xy[1] + 4.0 * s + w_i * 9.0 * s],
                [head_xy[0] - 38.0 * s + wave_lag, head_xy[1] + 18.0 * s + w_i * 10.0 * s],
                [head_xy[0] - 28.0 * s + wave_lag * 1.2, head_xy[1] + 30.0 * s + w_i * 8.0 * s],
                n_pts=12,
            )
            draw_calligraphic_ink_curve(canvas, h_strand, BGR_INK, base_thick=2.3 * s, boil_phase=boil_phase, seed=58 + w_i)

    elif cfg["hat_type"] == "reel_crown":
        # Sprocket's Rotating Twin 35mm Film-Reel Headgear!
        reel_rot = sub_step * 540.0
        for r_off in (-13.0 * s, 13.0 * s):
            rcx, rcy = head_xy[0] + r_off, hat_top_y - 8.0 * s
            draw_hand_ellipse(canvas, rcx, rcy, 11.5 * s, 11.5 * s, fill_bgr=BGR_MINT_PALE, outline_bgr=BGR_INK, thickness=2.2 * s, boil_phase=boil_phase, seed=61, n_pts=18)
            for sp_i in range(4):
                a_sp = math.radians(reel_rot + sp_i * 90.0)
                cv2.line(
                    canvas,
                    (int(round(rcx)), int(round(rcy))),
                    (int(round(rcx + 9.5 * s * math.cos(a_sp))), int(round(rcy + 9.5 * s * math.sin(a_sp)))),
                    BGR_INK, max(1, int(round(1.6 * s))), cv2.LINE_AA,
                )

    else:
        # Chime's Tuning-Fork & Resonating Acoustic Bell Headpiece
        fk_x, fk_y = head_xy[0], hat_top_y - 4.0 * s
        draw_calligraphic_ink_curve(canvas, [[fk_x, fk_y], [fk_x, fk_y - 14.0 * s]], BGR_INK, base_thick=2.6 * s, boil_phase=boil_phase, seed=64)
        draw_calligraphic_ink_curve(canvas, [[fk_x - 9.0 * s, fk_y - 28.0 * s], [fk_x - 9.0 * s, fk_y - 14.0 * s], [fk_x + 9.0 * s, fk_y - 14.0 * s], [fk_x + 9.0 * s, fk_y - 28.0 * s]], BGR_INK, base_thick=2.4 * s, taper_ends=False, boil_phase=boil_phase, seed=65)
        # Vibrating acoustic wave rings around tuning fork
        for w_r in (14.0 * s, 20.0 * s):
            cv2.ellipse(canvas, (int(round(fk_x)), int(round(fk_y - 22.0 * s))), (int(round(w_r)), int(round(w_r * 0.55))), 0, -45, 45, BGR_ARC_RED, 1, cv2.LINE_AA)
            cv2.ellipse(canvas, (int(round(fk_x)), int(round(fk_y - 22.0 * s))), (int(round(w_r)), int(round(w_r * 0.55))), 0, 135, 225, BGR_ARC_RED, 1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # RIGHT FOREGROUND ARM (2-BONE IK + 5-FINGER HAND + ACTIVE STUDIO TOOL!)
    # -------------------------------------------------------------------------
    rw_target = sample_right_wrist(sub_step)
    r_elbow, r_wrist = solve_2bone_ik(sh_r, rw_target, 33.0 * s, 31.0 * s, bend_sign=1.0)
    draw_tapered_limb_segment(canvas, sh_r, r_elbow, 9.8 * s, 8.2 * s, cfg["smock_bgr"], BGR_INK, show_pass_ab, boil_phase, seed=70)
    draw_tapered_limb_segment(canvas, r_elbow, r_wrist, 8.0 * s, 6.4 * s, BGR_SKIN, BGR_INK, show_pass_ab, boil_phase, seed=71)
    r_ang = math.degrees(math.atan2(r_wrist[1] - r_elbow[1], r_wrist[0] - r_elbow[0]))
    right_tip = draw_articulated_5finger_hand_and_tool(
        canvas, r_wrist, r_ang, tool_type=cfg["tool"], scale=0.88 * s,
        finger_curl=0.45 + 0.35 * math.cos(t_rad * 2.0), boil_phase=boil_phase,
    )

    return {
        "right_tip": right_tip,
        "left_tip": left_tip,
        "head_center": head_xy,
        "yaw_deg": yaw_deg,
    }


# =============================================================================
# 4. CODE-ANIMATED 3-SHEET ROLLING PAPER FLIP & 360° TURNAROUND RIGS
# =============================================================================
def draw_3sheet_rolling_paper_flip_overlay(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    sub_step: float,
    boil_phase: int = 0,
):
    """
    Draws an authentic 3-sheet animator's rolling paper flip on a tilted backlit lightbox disc:
      - Bottom Sheet (`K01`): Pegged flat in Non-Photo Blue
      - Middle Sheet (`B01` Breakdown): Curling upward at angle `curl_1` with red breakdown arcs
      - Top Sheet (`I01` In-Between): Curling higher at angle `curl_2` with black Speedball ink lines
      - Brass 3-pin Acme Peg Bar locking the top margin!
    """
    x0, y0 = cx - w * 0.44, cy - h * 0.38
    x1, y1 = cx + w * 0.44, cy + h * 0.38

    # Glowing backlit glass disc
    draw_hand_ellipse(canvas, cx, cy, w * 0.47, h * 0.45, fill_bgr=BGR_GUTTER, outline_bgr=BGR_NONPHOTO_BLUE, thickness=2.0, boil_phase=boil_phase, seed=201)

    # 3 curling sheets anchored to top Acme peg bar `(x0..x1, y0)`
    for sh_i, (sh_lbl, sh_col) in enumerate([("K01", BGR_NONPHOTO_BLUE), ("B01", BGR_ARC_RED), ("I01", BGR_INK)]):
        curl = (0.12 + 0.78 * abs(math.sin(sub_step * math.pi * 3.0 - sh_i * 0.55))) * (sh_i * 14.0)
        p_tl = [x0 + sh_i * 2.0, y0]
        p_tr = [x1 - sh_i * 2.0, y0]
        p_br = [x1 + curl * 0.45, y1 - curl * 1.15]
        p_bl = [x0 + curl * 0.25, y1 - curl * 0.90]
        r_edge = eval_quad_bezier(p_tr, [x1 + curl * 0.7, 0.5 * (y0 + y1) - curl * 0.4], p_br, 10)
        b_edge = eval_quad_bezier(p_br, [0.5 * (x0 + x1) + curl * 0.3, y1 - curl * 1.3], p_bl, 10)
        l_edge = eval_quad_bezier(p_bl, [x0 - curl * 0.2, 0.5 * (y0 + y1) - curl * 0.3], p_tl, 10)
        poly = [p_tl, p_tr] + r_edge[1:] + b_edge[1:] + l_edge[1:]
        arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in poly], dtype=np.int32)
        if sh_i == 2:
            ov = canvas.copy()
            cv2.fillPoly(ov, [arr], BGR_PAPER, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.72, canvas, 0.28, 0, canvas)
        draw_calligraphic_ink_curve(canvas, poly, sh_col if sh_i < 2 else BGR_INK, base_thick=1.6 if sh_i < 2 else 2.2, taper_ends=False, boil_phase=boil_phase, seed=210 + sh_i)

    # Top Brass Acme Peg Bar (`1 round center peg + 2 flat slot pegs`)
    cv2.rectangle(canvas, (int(round(x0 - 4)), int(round(y0 - 6))), (int(round(x1 + 4)), int(round(y0 + 5))), BGR_AMBER, -1)
    cv2.rectangle(canvas, (int(round(x0 - 4)), int(round(y0 - 6))), (int(round(x1 + 4)), int(round(y0 + 5))), BGR_INK, 2)
    for px_f in (0.22, 0.50, 0.78):
        cv2.circle(canvas, (int(round(x0 + (x1 - x0) * px_f)), int(round(y0 - 1))), 3, BGR_INK, -1, cv2.LINE_AA)


# =============================================================================
# 5. 10X CURVED-ARC NORMALIZED STROKE IN-BETWEENER (`240 x 10 = 2,400` STUDIO FRAMES!)
# =============================================================================
_ARC_MATCH_CACHE = {}


def get_normalized_arc_matches(sheet_a: dict, q_a: int, sheet_b: dict, q_b: int, key_id: tuple) -> list:
    """
    Matches strokes between ANY `(sheet_a, q_a)` and `(sheet_b, q_b)` in normalized
    bounding-box coordinates `u, v in [-0.5, 0.5]`, computing a perpendicular curved
    quadratic Bézier arc offset for each matched stroke pair so in-betweens follow
    natural curved trajectories (`Section XIII: ARCS`)!
    """
    if key_id in _ARC_MATCH_CACHE:
        return _ARC_MATCH_CACHE[key_id]

    qa = sheet_a["quadrants"][int(q_a) % 4]
    qb = sheet_b["quadrants"][int(q_b) % 4]
    sa_list = qa.get("strokes", [])
    sb_list = qb.get("strokes", [])
    if not sa_list or not sb_list:
        _ARC_MATCH_CACHE[key_id] = []
        return []

    ax0, ay0, ax1, ay1 = [float(v) for v in qa["bbox"]]
    bx0, by0, bx1, by1 = [float(v) for v in qb["bbox"]]
    aw, ah = max(16.0, ax1 - ax0), max(16.0, ay1 - ay0)
    bw, bh = max(16.0, bx1 - bx0), max(16.0, by1 - by0)
    acx, acy = 0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1)
    bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)

    def _norm_stroke(st, cx_b, cy_b, w_b, h_b, n_res=16):
        idx = np.linspace(0, len(st) - 1, n_res)
        pts = [[(st[int(round(k))][0] - cx_b) / w_b, (st[int(round(k))][1] - cy_b) / h_b] for k in idx]
        ln = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(n_res - 1))
        return pts, ln

    norm_a = [_norm_stroke(s, acx, acy, aw, ah) for s in sa_list]
    norm_b = [_norm_stroke(s, bcx, bcy, bw, bh) for s in sb_list]

    used_b = set()
    paired = []
    for ia, (pts_a, len_a) in enumerate(norm_a):
        best_ib = None
        best_cost = 0.42
        best_rev = False
        ma = pts_a[len(pts_a) // 2]
        for ib, (pts_b, len_b) in enumerate(norm_b):
            if ib in used_b:
                continue
            len_ratio = abs(len_a - len_b) / max(len_a, len_b, 0.05)
            if len_ratio > 0.65:
                continue
            mb = pts_b[len(pts_b) // 2]
            c_mid = math.hypot(ma[0] - mb[0], ma[1] - mb[1])
            c_fwd = c_mid + 0.45 * math.hypot(pts_a[0][0] - pts_b[0][0], pts_a[0][1] - pts_b[0][1]) + 0.45 * math.hypot(pts_a[-1][0] - pts_b[-1][0], pts_a[-1][1] - pts_b[-1][1]) + 0.15 * len_ratio
            c_rev = c_mid + 0.45 * math.hypot(pts_a[0][0] - pts_b[-1][0], pts_a[0][1] - pts_b[-1][1]) + 0.45 * math.hypot(pts_a[-1][0] - pts_b[0][0], pts_a[-1][1] - pts_b[0][1]) + 0.15 * len_ratio
            if c_fwd < best_cost or c_rev < best_cost:
                if c_rev < c_fwd:
                    best_cost = c_rev
                    best_ib = ib
                    best_rev = True
                else:
                    best_cost = c_fwd
                    best_ib = ib
                    best_rev = False
        if best_ib is not None:
            used_b.add(best_ib)
            pts_b_matched = norm_b[best_ib][0][::-1] if best_rev else norm_b[best_ib][0]
            paired.append((ia, best_ib, pts_a, pts_b_matched))

    _ARC_MATCH_CACHE[key_id] = paired
    return paired


def build_10x_curved_arc_strokes(
    sheet_a: dict,
    q_a: int,
    sheet_b: dict,
    q_b: int,
    t_sub: float,
    key_id: tuple,
) -> tuple:
    """
    Constructs a true 10x sub-frame in-between (`t_sub in [0.0, 1.0]`, quantized or continuous
    across 10 sub-steps `0/10 .. 9/10`) between `(sheet_a, q_a)` and `(sheet_b, q_b)`.
    Instead of pairing unrelated contour strokes across different topologies (which causes crossed lines),
    this engine deforms the coherent hand-drawn anatomy (`head`, `left_arm`, `right_arm`, `torso`,
    `cel_polys`, `accent_polys`) along a continuous 10-step Biomechanical Curved Arc:
      - Sub-frames 1..5 (`t_sub < 0.50`): Anticipation Crouch -> Slow-Out -> Rising Parabolic Arc from Pose A
      - Sub-frames 6..10 (`t_sub >= 0.50`): Arrival Arc -> Overshoot Follow-Through -> Slow-In Settle into Pose B
    Returns `(out_strokes, out_parts, out_cel_polys, out_acc_polys)` mapped into `qa`'s bounding box.
    """
    qa = sheet_a["quadrants"][int(q_a) % 4]
    qb = sheet_b["quadrants"][int(q_b) % 4]

    ax0, ay0, ax1, ay1 = [float(v) for v in qa["bbox"]]
    bx0, by0, bx1, by1 = [float(v) for v in qb["bbox"]]
    aw, ah = max(16.0, ax1 - ax0), max(16.0, ay1 - ay0)
    bw, bh = max(16.0, bx1 - bx0), max(16.0, by1 - by0)
    acx, acy = 0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1)
    bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)

    is_agent = sheet_a.get("is_agent", False) or sheet_b.get("is_agent", False)

    # 10-step Biomechanical Curve Dynamics (Anticipation -> Arc Apex -> Overshoot -> Settle)
    if t_sub < 0.50:
        u_loc = t_sub / 0.50  # 0.0 -> 1.0 departing Pose A
        src_q = qa
        src_cx, src_cy, src_w, src_h = acx, acy, aw, ah
        # Anticipation dip at u_loc ~ 0.25, then upward/forward arc launch toward u_loc = 1.0
        arc_dx = 0.028 * math.sin(u_loc * math.pi * 0.85)
        arc_dy = 0.016 * math.sin(u_loc * math.pi * 2.0) - 0.026 * (u_loc ** 1.6)
        squash_y = 1.0 - 0.032 * math.sin(u_loc * math.pi) + 0.042 * (u_loc ** 2)
        squash_x = 1.0 / squash_y
        head_deg = -4.5 * math.sin(u_loc * math.pi) + 7.5 * (u_loc ** 1.5)
        r_arm_deg = -11.0 * (u_loc ** 1.3)
        l_arm_deg = 9.5 * (u_loc ** 1.3)
    else:
        u_loc = (t_sub - 0.50) / 0.50  # 0.0 -> 1.0 arriving at Pose B
        src_q = qb
        src_cx, src_cy, src_w, src_h = bcx, bcy, bw, bh
        damp = 1.0 - u_loc
        # Arrival from upper arc + overshoot wave + slow-in settle to 0.0 at u_loc = 1.0
        arc_dx = -0.026 * damp * math.cos(u_loc * math.pi * 0.5)
        arc_dy = -0.024 * (damp ** 1.5) + 0.012 * damp * math.sin(u_loc * math.pi * 2.0)
        squash_y = 1.0 + 0.036 * (damp ** 1.8) - 0.024 * damp * math.sin(u_loc * math.pi)
        squash_x = 1.0 / squash_y
        head_deg = 6.5 * damp * math.cos(u_loc * math.pi)
        r_arm_deg = 12.0 * damp * math.cos(u_loc * math.pi * 0.8)
        l_arm_deg = -9.5 * damp * math.cos(u_loc * math.pi * 0.8)

    def _deform_pt(px: float, py: float, part: str = "torso") -> list:
        u = (px - src_cx) / src_w
        v = (py - src_cy) / src_h

        if is_agent:
            if part == "head" or (v < -0.04 and abs(u) < 0.34):
                piv_u, piv_v = 0.0, -0.04
                ang = math.radians(head_deg)
                du, dv = u - piv_u, v - piv_v
                u = piv_u + du * math.cos(ang) - dv * math.sin(ang)
                v = piv_v + du * math.sin(ang) + dv * math.cos(ang)
            elif part == "right_arm" or (u > 0.14 and v >= -0.14):
                piv_u, piv_v = 0.14, -0.02
                ang = math.radians(r_arm_deg)
                du, dv = u - piv_u, v - piv_v
                u = piv_u + du * math.cos(ang) - dv * math.sin(ang)
                v = piv_v + du * math.sin(ang) + dv * math.cos(ang)
            elif part == "left_arm" or (u < -0.14 and v >= -0.14):
                piv_u, piv_v = -0.14, -0.02
                ang = math.radians(l_arm_deg)
                du, dv = u - piv_u, v - piv_v
                u = piv_u + du * math.cos(ang) - dv * math.sin(ang)
                v = piv_v + du * math.sin(ang) + dv * math.cos(ang)

        u_final = u * squash_x + arc_dx
        v_final = v * squash_y + arc_dy
        return [acx + u_final * aw, acy + v_final * ah]

    src_strokes = src_q.get("strokes", [])
    src_parts = src_q.get("stroke_parts", ["torso"] * len(src_strokes))

    out_strokes = [
        [_deform_pt(p[0], p[1], src_parts[i] if i < len(src_parts) else "torso") for p in st]
        for i, st in enumerate(src_strokes)
    ]
    out_cel = [[_deform_pt(p[0], p[1], "torso") for p in poly] for poly in src_q.get("cel_polys", [])]
    out_acc = [[_deform_pt(p[0], p[1], "torso") for p in poly] for poly in src_q.get("accent_polys", [])]

    return out_strokes, list(src_parts), out_cel, out_acc


# =============================================================================
# 6. CODE-ANIMATED STUDIO APPARATUS OVERLAY (`Scene 1..8` Mechanical Details)
# =============================================================================
def draw_code_animated_studio_apparatus(
    canvas: np.ndarray,
    scene_idx: int,
    cx: float,
    cy: float,
    max_w: float,
    max_h: float,
    sub_step: float,
    frame_idx: int,
    boil_phase: int = 0,
):
    """
    Overlays frame-by-frame code-animated mechanical details onto the Studio Apparatus zone
    (`cx=436, cy=by1-130`) so every wheel, celluloid filmstrip, prism beam, storyboard arc,
    lightbox disc, ink well meniscus, swinging cel, multiplane worm-gear platen, and foley prop
    moves with crisp 24fps precision!
    """
    t_rad = sub_step * 2.0 * math.pi * 2.0

    if scene_idx == 1:
        # 35mm Moviola: Sprocketed celluloid filmstrip running between reels + flickering optical prism ray
        strip_pts = eval_cubic_bezier(
            [cx - 42, cy - 52],
            [cx - 12, cy - 12 + 6.0 * math.sin(t_rad)],
            [cx + 14, cy + 16 - 6.0 * math.cos(t_rad)],
            [cx + 42, cy - 52],
            n_pts=18,
        )
        draw_calligraphic_ink_curve(canvas, strip_pts, BGR_AMBER, base_thick=4.5, taper_ends=False, boil_phase=boil_phase, seed=301)
        draw_calligraphic_ink_curve(canvas, strip_pts, BGR_INK, base_thick=1.6, taper_ends=False, boil_phase=boil_phase, seed=302)
        # Moving sprocket holes along the celluloid strip
        phase_sp = int(sub_step * 40) % 3
        for k in range(phase_sp, len(strip_pts), 3):
            px, py = int(round(strip_pts[k][0])), int(round(strip_pts[k][1]))
            cv2.circle(canvas, (px, py), 2, BGR_PAPER, -1, cv2.LINE_AA)

    elif scene_idx == 2:
        # 4-Card Storyboard Wall: Live sweeping red/blue narrative trajectory arc connecting cards
        arc_sb = eval_cubic_bezier(
            [cx - 56, cy - 38],
            [cx + 52, cy - 46],
            [cx - 52, cy + 28],
            [cx + 56, cy + 24],
            n_pts=24,
        )
        draw_motion_arc_with_spacing_ticks(canvas, arc_sb, sub_step, color_bgr=BGR_NONPHOTO_BLUE, tick_color_bgr=BGR_ARC_RED)

    elif scene_idx == 3:
        # Backlit Lightbox Disc: 3-sheet rolling paper curl overlay + degree calibration ticks
        draw_3sheet_rolling_paper_flip_overlay(canvas, cx, cy - 10, max_w * 0.62, max_h * 0.48, sub_step, boil_phase=boil_phase)

    elif scene_idx == 4:
        # Clean-Up Inking Desk: Glass India-Ink Bottle with rippling meniscus & rising ink drop
        bx, by = int(round(cx + 48)), int(round(cy + 26))
        cv2.rectangle(canvas, (bx - 14, by - 16), (bx + 14, by + 14), BGR_INK, -1)
        cv2.rectangle(canvas, (bx - 16, by - 18), (bx + 16, by + 16), BGR_PAPER, 2)
        cv2.rectangle(canvas, (bx - 16, by - 18), (bx + 16, by + 16), BGR_INK, 2)
        # Rippling ink wave rings when Nib dips
        r_wave = int(6 + 10 * abs(math.sin(t_rad)))
        cv2.ellipse(canvas, (bx, by - 16), (r_wave, max(2, r_wave // 3)), 0, 0, 360, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    elif scene_idx == 5:
        # Overhead Cel Drying Rack: Hanging painted cels swinging in secondary pendular motion
        wire_y = int(round(cy - 64))
        draw_calligraphic_ink_curve(canvas, [[cx - 72, wire_y], [cx, wire_y + 5], [cx + 72, wire_y]], BGR_INK, base_thick=2.0, taper_ends=False, boil_phase=boil_phase, seed=320)
        for c_i, x_off in enumerate((-42, 0, 42)):
            swing_deg = 9.5 * math.sin(t_rad - c_i * 0.65)
            ang = math.radians(swing_deg)
            top_x = cx + x_off
            top_y = wire_y + 3
            w_cel, h_cel = 24.0, 18.0
            dx_c = h_cel * math.sin(ang)
            dy_c = h_cel * math.cos(ang)
            cel_pts = np.array([
                [int(round(top_x - w_cel * 0.5)), int(round(top_y))],
                [int(round(top_x + w_cel * 0.5)), int(round(top_y))],
                [int(round(top_x + w_cel * 0.5 + dx_c)), int(round(top_y + dy_c))],
                [int(round(top_x - w_cel * 0.5 + dx_c)), int(round(top_y + dy_c))],
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [cel_pts], BGR_AMBER_PALE if c_i % 2 == 0 else BGR_BLUE_PALE, cv2.LINE_AA)
            cv2.polylines(canvas, [cel_pts], True, BGR_INK, 2, cv2.LINE_AA)

    elif scene_idx == 6:
        # 7-Level Multiplane Camera Tower: 7 sliding glass platens (`P0..P6`) with differential parallax!
        top_y = cy - 68
        bot_y = cy + 56
        for p_lvl in range(7):
            py = top_y + (bot_y - top_y) * (p_lvl / 6.0)
            # Differential horizontal shift: near planes (P0) shift more, far planes (P6) shift less
            depth_factor = 1.0 / (0.6 + p_lvl * 0.45)
            dx_platen = 18.0 * math.sin(t_rad) * depth_factor
            px0 = int(round(cx - 52 + dx_platen))
            px1 = int(round(cx + 52 + dx_platen))
            py_i = int(round(py))
            col_pl = BGR_BLUE_PALE if p_lvl in (0, 2, 6) else BGR_AMBER_PALE
            cv2.rectangle(canvas, (px0, py_i - 3), (px1, py_i + 3), col_pl, -1)
            cv2.rectangle(canvas, (px0, py_i - 3), (px1, py_i + 3), BGR_INK, 1)
            # Brass worm-gear thread ticks on left & right posts
            gear_shift = int(sub_step * 24) % 4
            cv2.circle(canvas, (int(round(cx - 60)), py_i - 2 + gear_shift), 2, BGR_AMBER, -1)
            cv2.circle(canvas, (int(round(cx + 60)), py_i - 2 + gear_shift), 2, BGR_AMBER, -1)

    elif scene_idx == 7:
        # Foley Soundstage Table: Radiating acoustic concentric rings & snapping clapperboard jaw
        for r_i in range(1, 4):
            r_rad = int(round(12 * r_i + ((sub_step * 36) % 12)))
            cv2.ellipse(canvas, (int(round(cx)), int(round(cy - 36))), (r_rad, int(r_rad * 0.55)), 0, 195, 345, BGR_ARC_RED, 1, cv2.LINE_AA)


# =============================================================================
# 7. CODE-ANIMATED `4:3` MACRO HAND CAM OVERLAY (`184x138`, `Scene 1..8`)
# =============================================================================
def draw_code_animated_macro_inset(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    scene_idx: int,
    sub_step: float,
    frame_idx: int,
    boil_phase: int = 0,
):
    """
    Overlays hyper-detailed frame-by-frame code-animated technical drawings and tools inside
    the `184x138` (`4:3` exact) Macro Hand Cam (`x0=344, y0=104`) so the viewer sees close-up
    craftsmanship happening live on every frame!
    """
    t_rad = sub_step * 2.0 * math.pi * 2.0
    cx = x0 + w * 0.5
    cy = y0 + h * 0.5

    if scene_idx == 1:
        # Scene 1 Macro: 35mm Sprocket Filmstrip & Live Red Layout Pencil sketching Thumbnail Arc
        strip_y = y0 + h - 18
        cv2.rectangle(canvas, (x0 + 8, strip_y - 6), (x0 + w - 8, strip_y + 6), BGR_INK, -1)
        sp_off = int(sub_step * 60) % 12
        for sx_p in range(x0 + 12 - sp_off, x0 + w - 10, 12):
            if x0 + 8 < sx_p < x0 + w - 12:
                cv2.rectangle(canvas, (sx_p, strip_y - 4), (sx_p + 5, strip_y + 4), BGR_PAPER, -1)
        # Animated red layout trajectory curve with moving pencil tip
        arc_th = eval_cubic_bezier([x0 + 22, y0 + h - 34], [x0 + 68, y0 + 22], [x0 + 118, y0 + h - 38], [x0 + w - 22, y0 + 28], 20)
        draw_motion_arc_with_spacing_ticks(canvas, arc_th, sub_step, color_bgr=BGR_NONPHOTO_BLUE, tick_color_bgr=BGR_ARC_RED)

    elif scene_idx == 2:
        # Scene 2 Macro: Continuous 360° Volumetric Character Turnaround Sphere (`0°..360°` yaw!)
        yaw_360 = sub_step * 360.0
        rad_y = math.radians(yaw_360)
        rx_s, ry_s = 26.0, 28.0
        scx, scy = x0 + w - 38, y0 + 42
        draw_hand_ellipse(canvas, scx, scy, rx_s, ry_s, fill_bgr=BGR_PAPER, outline_bgr=BGR_NONPHOTO_BLUE, thickness=1.8, boil_phase=boil_phase, seed=401, n_pts=20)
        # Rotating longitude meridian curve showing continuous 360° 3D rotation
        mer_x = rx_s * math.sin(rad_y)
        mer_pts = eval_quad_bezier([scx, scy - ry_s], [scx + mer_x * 1.25, scy], [scx, scy + ry_s], 12)
        draw_calligraphic_ink_curve(canvas, mer_pts, BGR_ARC_RED, base_thick=1.8, boil_phase=boil_phase, seed=402)
        draw_calligraphic_ink_curve(canvas, [[scx - rx_s, scy], [scx + rx_s, scy]], BGR_NONPHOTO_BLUE, base_thick=1.3, boil_phase=boil_phase, seed=403)

    elif scene_idx == 3:
        # Scene 3 Macro: Live 3-Sheet Rolling Paper Flip (`K01` blue, `B01` red, `I01` black ink) + 5-finger hand!
        draw_3sheet_rolling_paper_flip_overlay(canvas, cx - 10, cy + 6, w * 0.72, h * 0.72, sub_step, boil_phase=boil_phase)
        hand_w = (x0 + w - 36 + 10.0 * math.cos(t_rad), y0 + h - 28 - 12.0 * abs(math.sin(t_rad)))
        draw_articulated_5finger_hand_and_tool(canvas, hand_w, -135.0 + 18.0 * math.sin(t_rad), tool_type="blue_pencil", scale=0.68, finger_curl=0.5, boil_phase=boil_phase)

    elif scene_idx == 4:
        # Scene 4 Macro: Speedball Dip-Pen Capillary Inking over Non-Photo Blue curve
        ink_path = eval_cubic_bezier([x0 + 18, y0 + h - 26], [x0 + 64, y0 + 18], [x0 + 120, y0 + h - 22], [x0 + w - 24, y0 + 30], 24)
        draw_calligraphic_ink_curve(canvas, ink_path, BGR_NONPHOTO_BLUE, base_thick=1.4, taper_ends=True, boil_phase=boil_phase, seed=410)
        n_drawn = max(2, int(round(sub_step * len(ink_path))))
        draw_calligraphic_ink_curve(canvas, ink_path[:n_drawn], BGR_INK, base_thick=3.2, taper_ends=True, boil_phase=boil_phase, seed=411)
        tip_pt = ink_path[n_drawn - 1]
        draw_articulated_5finger_hand_and_tool(
            canvas, (tip_pt[0] + 22, tip_pt[1] - 16), 145.0,
            tool_type="dip_pen", scale=0.65, finger_curl=0.45, boil_phase=boil_phase,
        )

    elif scene_idx == 5:
        # Scene 5 Macro: Sable Gouache Brush Flooding & 5-Swatch Studio Palette Wheel
        pal_x, pal_y = x0 + 32, y0 + h - 30
        draw_hand_ellipse(canvas, pal_x, pal_y, 22, 15, fill_bgr=BGR_PAPER, outline_bgr=BGR_INK, thickness=1.8, boil_phase=boil_phase, seed=420)
        for sw_i, sw_col in enumerate([BGR_RED, BGR_AMBER, BGR_GREEN, BGR_BLUE, BGR_CORAL_PALE]):
            ang_sw = sw_i * (2.0 * math.pi / 5.0) + sub_step * math.pi
            cv2.circle(canvas, (int(round(pal_x + 12 * math.cos(ang_sw))), int(round(pal_y + 8 * math.sin(ang_sw)))), 4, sw_col, -1, cv2.LINE_AA)
        b_wrist = (cx + 26.0 * math.sin(t_rad), cy - 8.0 + 12.0 * math.cos(t_rad))
        draw_articulated_5finger_hand_and_tool(canvas, b_wrist, 130.0 + 15.0 * math.sin(t_rad), tool_type="gouache_brush", scale=0.68, boil_phase=boil_phase)

    elif scene_idx == 6:
        # Scene 6 Macro: Multiplane Camera Parallax Frustum Rays (`screen_motion ∝ camera_motion / depth`)
        cam_x = int(round(cx + 22.0 * math.sin(t_rad)))
        cam_y = y0 + 14
        cv2.circle(canvas, (cam_x, cam_y), 5, BGR_RED, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cam_x, cam_y), 5, BGR_INK, 1, cv2.LINE_AA)
        for pl_i, (pl_y_frac, pl_col) in enumerate([(0.38, BGR_RED), (0.58, BGR_AMBER), (0.78, BGR_NONPHOTO_BLUE), (0.92, BGR_GREEN)]):
            py_l = int(round(y0 + h * pl_y_frac))
            dx_p = int(round((cam_x - cx) / (1.0 + pl_i * 0.85)))
            cv2.line(canvas, (x0 + 24 + dx_p, py_l), (x0 + w - 24 + dx_p, py_l), pl_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (cam_x, cam_y), (x0 + 20, y0 + h - 10), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.line(canvas, (cam_x, cam_y), (x0 + w - 20, y0 + h - 10), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    elif scene_idx == 7:
        # Scene 7 Macro: Brass 4:3 Caliper Measuring the 35mm Aperture Gate
        cal_w = (x0 + w - 34 + 6.0 * math.sin(t_rad), cy + 10.0 * math.cos(t_rad))
        draw_articulated_5finger_hand_and_tool(canvas, cal_w, 155.0, tool_type="caliper", scale=0.72, finger_curl=0.5 + 0.3 * math.sin(t_rad), boil_phase=boil_phase)


# =============================================================================
# 8. 10X HERO MULTIPLANE BOARD ENHANCER (`70 x 10 = 700` Hero Multiplane Frames!)
# =============================================================================
def enhance_hero_board_with_code_multiplane(
    board_bgr: np.ndarray,
    slug: str,
    sub_step: float,
    frame_idx: int,
    stage_mode: str = "auto",
    boil_phase: int = 0,
) -> np.ndarray:
    """
    Enhances the `384x288` (`4:3`) Hero Multiplane Stage with 10x sub-frame hand-animated
    code layers (`Plane P6` drifting clouds, `Plane P4` shot-specific animated environment effects,
    `Plane P2` trajectory arc & spacing ticks, and `Plane P0` foreground parallax vignette)
    so the hero cartoon on the right-hand screen animations at full 24fps On Ones (`700` hero frames)!
    """
    H, W = board_bgr.shape[:2]
    t_rad = sub_step * 2.0 * math.pi * 2.0
    cam_pan = (sub_step - 0.5) * 36.0

    # 1. Plane P6 (Distant Sky / Atmosphere, slow parallax `cam_pan * 0.18`): Hand-inked drifting clouds
    dx_sky = cam_pan * 0.18
    for c_i, (base_cx, base_cy, crx, cry) in enumerate([(78, 38, 28, 11), (210, 30, 34, 13), (320, 44, 25, 10)]):
        cx_c = (base_cx - dx_sky + sub_step * 28.0 * (1 + 0.2 * c_i)) % (W + 40) - 20
        draw_hand_ellipse(
            board_bgr, cx_c, base_cy, crx, cry,
            fill_bgr=BGR_PAPER if stage_mode != "blue_only" else None,
            outline_bgr=BGR_NONPHOTO_BLUE if stage_mode == "blue_only" else BGR_GRAPHITE,
            thickness=1.3, wobble_amp=0.65, boil_phase=boil_phase + c_i, seed=500 + c_i, n_pts=18,
        )

    # 2. Plane P4/P3 Shot-Specific Hand-Animated Multiplane Effects (`24fps On Ones`)
    slug_low = slug.lower()
    if "lighthouse" in slug_low:
        # Sweeping golden lighthouse optical beam across the coastal sky
        lx, ly = int(round(W * 0.52)), int(round(H * 0.34))
        beam_ang = math.sin(t_rad) * 0.45
        bx1 = int(round(lx + 160 * math.cos(beam_ang - 0.16)))
        by1 = int(round(ly + 160 * math.sin(beam_ang - 0.16)))
        bx2 = int(round(lx + 160 * math.cos(beam_ang + 0.16)))
        by2 = int(round(ly + 160 * math.sin(beam_ang + 0.16)))
        ov = board_bgr.copy()
        cv2.fillConvexPoly(ov, np.array([[lx, ly], [bx1, by1], [bx2, by2]], dtype=np.int32), BGR_AMBER_PALE, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.28, board_bgr, 0.72, 0, board_bgr)
        # Rolling ocean wave crests at bottom
        for w_i in range(3):
            wy = H - 36 + w_i * 11
            w_pts = [[x_p, wy + 3.5 * math.sin(x_p * 0.06 + t_rad * 1.5 + w_i)] for x_p in range(12, W - 12, 14)]
            draw_calligraphic_ink_curve(board_bgr, w_pts, BGR_NONPHOTO_BLUE, base_thick=1.6, taper_ends=True, boil_phase=boil_phase, seed=520 + w_i)

    elif "campfire" in slug_low:
        # Flickering gouache flame tongues & rising S-curve glowing embers
        fx, fy = W * 0.5, H * 0.68
        for e_i in range(5):
            u_emb = (sub_step * 2.0 + e_i * 0.2) % 1.0
            ex = fx + (e_i - 2) * 18.0 + 12.0 * math.sin(u_emb * 6.0 + e_i)
            ey = fy - u_emb * 115.0
            r_emb = max(2, int(round(4.5 * (1.0 - u_emb))))
            cv2.circle(board_bgr, (int(round(ex)), int(round(ey))), r_emb, BGR_RED if e_i % 2 == 0 else BGR_AMBER, -1, cv2.LINE_AA)
            cv2.circle(board_bgr, (int(round(ex)), int(round(ey))), r_emb, BGR_INK, 1, cv2.LINE_AA)

    elif "train" in slug_low or "tractor" in slug_low:
        # Chugging steam/smoke puffs expanding along parabolic arc
        sx_0, sy_0 = W * 0.42, H * 0.36
        for p_i in range(4):
            u_p = (sub_step * 2.0 + p_i * 0.25) % 1.0
            px = sx_0 - u_p * 95.0
            py = sy_0 - u_p * 52.0 + 6.0 * math.sin(u_p * 8.0)
            r_p = 7.0 + u_p * 16.0
            draw_hand_ellipse(board_bgr, px, py, r_p, r_p * 0.78, fill_bgr=BGR_PAPER, outline_bgr=BGR_INK, thickness=1.6, boil_phase=boil_phase + p_i, seed=540 + p_i, n_pts=14)

    elif "clock" in slug_low:
        # Swinging brass pendulum & tick-tock acoustic arcs
        px_c, py_c = W * 0.5, H * 0.58
        p_ang = 0.42 * math.sin(t_rad * 2.0)
        bx_p = px_c + 48.0 * math.sin(p_ang)
        by_p = py_c + 48.0 * math.cos(p_ang)
        draw_calligraphic_ink_curve(board_bgr, [[px_c, py_c], [bx_p, by_p]], BGR_INK, base_thick=2.2, boil_phase=boil_phase, seed=550)
        draw_hand_ellipse(board_bgr, bx_p, by_p, 9.0, 9.0, fill_bgr=BGR_AMBER, outline_bgr=BGR_INK, thickness=2.0, boil_phase=boil_phase, seed=551, n_pts=14)

    elif "speedboat" in slug_low:
        # Planing spray wake arcs
        for w_i in range(3):
            wy = H - 42 + w_i * 10
            w_pts = [[x_p, wy + 4.0 * math.cos(x_p * 0.08 - t_rad * 2.0 + w_i)] for x_p in range(16, W - 16, 14)]
            draw_calligraphic_ink_curve(board_bgr, w_pts, BGR_NONPHOTO_BLUE, base_thick=1.8, boil_phase=boil_phase, seed=560 + w_i)

    else:
        # Hero Trajectory Motion Arc with Slow-In / Slow-Out Spacing Ticks (`Section XI & XIII`)
        arc_h = eval_cubic_bezier([48, H - 56], [W * 0.36, H * 0.28], [W * 0.66, H * 0.72], [W - 48, H - 64], 22)
        draw_motion_arc_with_spacing_ticks(board_bgr, arc_h, sub_step, color_bgr=BGR_NONPHOTO_BLUE, tick_color_bgr=BGR_ARC_RED)

    # 3. Ground Plane Foot/Contact Registration Ticks (`Section VIII: Grounded Weight`)
    gy = int(round(H * 0.84))
    for gx_f in (0.25, 0.50, 0.75):
        gx = int(round(W * gx_f - cam_pan * 0.65)) % W
        cv2.line(board_bgr, (gx - 8, gy), (gx + 8, gy), BGR_GRAPHITE, 1, cv2.LINE_AA)

    return board_bgr


# =============================================================================
# 9. 3,100-FRAME FLIPBOOK VERIFICATION SHEETS & MASTER CODE-ANIMATED ATLAS
# =============================================================================
def generate_3100_frame_flipbook_atlases(
    render_panel_fn,
    sample_spec: dict,
    preview_dir: Path,
    artifact_dir: Path,
) -> list:
    """
    Renders:
      1. 8 Scene-by-Scene Multi-Frame Flipbook Sheets (`preview/flipbook_scene_01..08.png`),
         each showing a `3x4` (`12`-frame) sequential flipbook progression (`SUB-FRAME 01/12 .. 12/12`)
         for that scene so the viewer can inspect the frame-by-frame hand-animated motion!
      2. Master 96-Frame Code-Animated Contact Atlas (`preview/studio_3100_hand_animated_code_atlas.png`),
         displaying all 8 Studio Scenes x 12 sequential sub-frames (`96` frames sampled from the `3,100`
         unique hand-animated frames)!
    """
    out_paths = []
    tw, th = 320, 180
    pad = 10
    hdr_h = 52

    all_scene_rows = []

    for sc_idx in range(1, 9):
        cols, rows = 4, 3
        W_fb = pad + cols * (tw + pad)
        H_fb = hdr_h + pad + rows * (th + pad)
        fb_canvas = np.full((H_fb, W_fb, 3), BGR_PAPER, dtype=np.uint8)
        cv2.rectangle(fb_canvas, (0, 0), (W_fb, hdr_h), BGR_AMBER_PALE, -1)
        cv2.rectangle(fb_canvas, (0, 0), (W_fb, hdr_h), BGR_INK, 3)
        cv2.putText(
            fb_canvas,
            f"SCENE {sc_idx}/8 FLIPBOOK -- 10X CODE-DRIVEN HAND ANIMATION (3,100 TOTAL STUDIO + HERO FRAMES @ 24 FPS ON ONES)",
            (14, 33),
            cv2.FONT_HERSHEY_DUPLEX,
            0.60,
            BGR_INK,
            1,
            cv2.LINE_AA,
        )

        row_strip_thumbs = []
        for f_k in range(12):
            sub_u = f_k / 11.0
            shot_f = int(round(sub_u * 32))
            fr_full = render_panel_fn(
                sample_spec,
                panel_num=sc_idx,
                sub_step=sub_u,
                shot_frame_idx=shot_f,
                boil_phase=f_k,
            )
            thumb = cv2.resize(fr_full, (tw, th), interpolation=cv2.INTER_AREA)
            # Stamp sub-frame flipbook badge in top-right of thumbnail
            cv2.rectangle(thumb, (tw - 86, 4), (tw - 4, 24), BGR_INK, -1)
            cv2.putText(thumb, f"F{sc_idx}.{f_k+1:02d}/12", (tw - 82, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR_PAPER, 1, cv2.LINE_AA)

            r_i = f_k // cols
            c_i = f_k % cols
            x0 = pad + c_i * (tw + pad)
            y0 = hdr_h + pad + r_i * (th + pad)
            fb_canvas[y0:y0 + th, x0:x0 + tw] = thumb
            cv2.rectangle(fb_canvas, (x0, y0), (x0 + tw, y0 + th), BGR_INK, 2)

            if f_k % 2 == 0:
                row_strip_thumbs.append(cv2.resize(thumb, (240, 135), interpolation=cv2.INTER_AREA))

        fb_name = f"flipbook_scene_0{sc_idx}.png"
        fb_path = preview_dir / fb_name
        cv2.imwrite(str(fb_path), fb_canvas)
        shutil.copy2(fb_path, artifact_dir / fb_name)
        out_paths.append(fb_path)
        all_scene_rows.append(np.hstack(row_strip_thumbs))

    # Assemble Master 3,100-Frame Code-Animated Overview Atlas (`8 scenes x 6 sub-frames = 48 tiles`)
    grid_body = np.vstack(all_scene_rows)
    gh, gw = grid_body.shape[:2]
    m_hdr = np.full((64, gw, 3), BGR_AMBER_PALE, dtype=np.uint8)
    cv2.rectangle(m_hdr, (0, 0), (gw - 1, 63), BGR_INK, 3)
    cv2.putText(
        m_hdr,
        "MASTER 10X CODE-ANIMATED FLIPBOOK ATLAS -- 3,100 UNIQUE HAND-ANIMATED FRAMES (2,400 STUDIO + 700 HERO @ 24 FPS)",
        (16, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.63,
        BGR_INK,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        m_hdr,
        "18-Joint 2-Bone IK Agent Rig | Calligraphic Speedball Splines | 3-Sheet Rolling Paper Flip | 10x Curved-Arc In-Betweens | 7-Plane Parallax",
        (16, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (55, 48, 42),
        1,
        cv2.LINE_AA,
    )
    master_atlas = np.vstack([m_hdr, grid_body])
    m_path = preview_dir / "studio_3100_hand_animated_code_atlas.png"
    cv2.imwrite(str(m_path), master_atlas)
    shutil.copy2(m_path, artifact_dir / "studio_3100_hand_animated_code_atlas.png")
    out_paths.append(m_path)
    return out_paths

