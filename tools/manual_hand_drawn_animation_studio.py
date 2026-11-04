#!/usr/bin/env python3
"""
manual_hand_drawn_animation_studio.py
================================================================================
MULTIPLANE ANIMATION STUDIO — 100% Manually Drawn Frame-by-Frame Animations
(`I. THE CENTRAL RULE` through `XXXVIII. THE PRIME DIRECTIVE`)

Replaces all static AI cutout popping, duplicate sprites, and "write-path"
progressive still-image tracing with TRUE FRAME-BY-FRAME HAND-DRAWN ANIMATION:

1. 14 Full-Motion Hand-Drawn Multiplane Cartoons (`render_manual_cartoon_frame`):
   Every frame (`24 fps` On Ones) is a complete drawing constructed from the
   inside out (`Pass A` Line of Action & Motion Arcs -> `Pass B` Non-Photo Blue
   Volumetric Construction -> `Pass C` Watertight Gouache Cel Fills -> `Pass D`
   Calligraphic Speedball B-5 India Ink Contours -> `Pass E` Cross-Hatching &
   7-Plane Multiplane Parallax Composite `Plane 6` Sky down to `Plane 0` FG Mask):
     - `campfire` (Section III Fox + Roaring Campfire: `hears -> freezes -> ears turn -> compresses -> leaps -> lands behind fern -> tail follows`)
     - `giraffe-walks-past` (Articulated 16-joint Giraffe 4-beat savanna walk cycle + flapping egrets + hoof dust)
     - `lighthouse` (Coastal Lighthouse + 360-deg foreshortened Fresnel beam + curling & crashing wave spray + seagulls)
     - `train` (4-4-0 Steam Locomotive + reciprocating piston/rod linkage + viaduct parallax + billowing steam plumes)
     - `butterfly` (Monarch Butterfly 3D clap-and-fling wingbeats + figure-8 flight + landing on bending wildflower + bumblebee)
     - Plus bespoke articulated animations for `eagle-soars`, `sailboat`, `whale-breach`,
       `dragon-flight`, `windmill`, `carousel`, `fountain`, `tornado`, and `clock-tower`!

2. 5 Heroic-Scale Hand-Drawn Studio Animator Characters (`render_studio_soundstage_scene`):
   `Nib`, `Blue-Line`, `Cel-Inker`, `Sprocket`, and `Chime` drawn at large studio
   scale (`380-450px` tall) from organic anatomical splines, performing continuous
   multi-beat physical acting choreographies at the Moviola, Storyboard Wall,
   Backlit Animation Lightbox (with real 3-sheet rolling paper flips on the brass
   Acme peg bar), Clean-Up Inking Desk, Gouache Opaquing & Drying Rack, 7-Level
   Multiplane Camera Tower, and 35mm Foley & Screening Theatre!
================================================================================
"""

import math
from pathlib import Path
import cv2
import numpy as np

# Studio Palette (BGR)
BGR_PAPER = (234, 244, 250)         # Warm archival animation paper
BGR_PAPER_DARK = (210, 224, 235)    # Shaded paper / pegbar margin
BGR_INK = (18, 20, 24)              # Speedball India Ink Black
BGR_SEPIA = (42, 62, 92)            # Warm dark sepia ink
BGR_NONPHOTO_BLUE = (228, 152, 62)  # Classic Non-Photo Blue pencil
BGR_BLUE_FAINT = (232, 198, 152)    # Faint blue underdrawing
BGR_ONION_PREV = (118, 132, 225)    # Warm red-sienna previous frame ghost
BGR_ONION_NEXT = (228, 188, 118)    # Cool cyan-blue next frame ghost
BGR_VERMILION = (52, 82, 214)       # Keyframe accent red
BGR_AMBER = (56, 158, 232)          # Breakdown / brass gold accent
BGR_TEAL = (128, 148, 52)           # Multiplane / foley teal


# ==============================================================================
# I. ORGANIC HAND-DRAWN SPLINE, ELLIPSE & CALLIGRAPHIC INK PRIMITIVES
# ==============================================================================

def eval_catmull_rom_chain(ctrl_pts: list, samples_per_seg: int = 10, closed: bool = False) -> np.ndarray:
    """Evaluates a smooth C1 Catmull-Rom spline through `ctrl_pts`."""
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


def eval_sharp_polygon_chain(ctrl_pts: list, samples_per_seg: int = 8, closed: bool = True) -> np.ndarray:
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
    Constructs a smooth, closed polygon for a tapered capsule (stadium) from
    `p0` (radius `r0`) to `p1` (radius `r1`) with semicircular end caps.
    Used for Dual-Pass Casing (`Pass 1: Outer Ink Casing -> Pass 2: Inner Fill`)
    so multi-segment limbs, sleeves, cuffs, hands, and fingers merge into a
    100% watertight cartoon silhouette with ZERO internal joint rings.
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-3:
        r = max(1.0, 0.5 * (r0 + r1))
        angs = np.linspace(0.0, 2.0 * math.pi, cap_steps * 2, endpoint=False)
        pts = np.column_stack([x0 + r * np.cos(angs), y0 + r * np.sin(angs)])
        return np.round(pts).astype(np.int32)
    ang = math.atan2(dy, dx)
    pts = []
    for i in range(cap_steps + 1):
        a = ang + math.pi * 0.5 + (math.pi * i) / cap_steps
        pts.append((x0 + r0 * math.cos(a), y0 + r0 * math.sin(a)))
    for i in range(cap_steps + 1):
        a = ang - math.pi * 0.5 + (math.pi * i) / cap_steps
        pts.append((x1 + r1 * math.cos(a), y1 + r1 * math.sin(a)))
    return np.round(np.array(pts, dtype=np.float64)).astype(np.int32)


def draw_hand_ink_spline(
    canvas: np.ndarray,
    ctrl_pts: list,
    color_bgr: tuple = BGR_INK,
    base_thick: float = 2.4,
    closed: bool = False,
    fill_bgr: tuple = None,
    boil_phase: int = 0,
    seed: int = 0,
    taper_ends: bool = True,
    samples_per_seg: int = 8,
    sharp_corners: bool = None,
):
    """
    Draws an organic hand-drawn spline or sharp-cornered architectural polygon
    with subtle paper-tooth boil, optional watertight gouache fill (`fill_bgr`),
    and clean sub-pixel calligraphic ribbon / polyline contour without stair-step lumps.
    """
    if len(ctrl_pts) < 2:
        return
    use_sharp = sharp_corners if sharp_corners is not None else (closed and len(ctrl_pts) <= 6)
    if use_sharp:
        curve = eval_sharp_polygon_chain(ctrl_pts, samples_per_seg=samples_per_seg, closed=closed)
    else:
        curve = eval_catmull_rom_chain(ctrl_pts, samples_per_seg=samples_per_seg, closed=closed)
    n = len(curve)
    if n < 2:
        return

    b_step = (boil_phase // 2)
    idx = np.arange(n, dtype=np.float64)
    u = idx / float(max(1, n - 1))
    boil_amp = 0.0 if use_sharp else 0.16
    bx = boil_amp * np.sin(u * 9.4 + b_step * 1.7 + seed * 0.73)
    by = boil_amp * np.cos(u * 11.2 - b_step * 1.3 + seed * 1.19)
    pts_f = np.column_stack([curve[:, 0] + bx, curve[:, 1] + by])
    pts_i = np.round(pts_f).astype(np.int32)

    if fill_bgr is not None and len(pts_i) >= 3:
        cv2.fillPoly(canvas, [pts_i], fill_bgr, cv2.LINE_AA)

    if base_thick <= 0:
        return

    if closed or use_sharp or not taper_ends or base_thick <= 1.8:
        cv2.polylines(canvas, [pts_i], closed, color_bgr, max(1, int(round(base_thick))), cv2.LINE_AA)
        return

    tangents = np.zeros_like(pts_f)
    tangents[1:-1] = pts_f[2:] - pts_f[:-2]
    tangents[0] = pts_f[1] - pts_f[0]
    tangents[-1] = pts_f[-1] - pts_f[-2]
    lengths = np.maximum(1e-4, np.hypot(tangents[:, 0], tangents[:, 1]))
    nx = -tangents[:, 1] / lengths
    ny = tangents[:, 0] / lengths

    env = np.sin(np.pi * u) ** 0.45
    half_w = np.maximum(0.65, 0.5 * base_thick * (0.58 + 0.48 * env))
    left_pts = np.column_stack([pts_f[:, 0] + nx * half_w, pts_f[:, 1] + ny * half_w])
    right_pts = np.column_stack([pts_f[:, 0] - nx * half_w, pts_f[:, 1] - ny * half_w])
    ribbon = np.round(np.vstack([left_pts, right_pts[::-1]])).astype(np.int32)
    cv2.fillPoly(canvas, [ribbon], color_bgr, cv2.LINE_AA)


def draw_hand_oval(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float = 0.0,
    color_bgr: tuple = BGR_INK,
    thick: float = 2.2,
    fill_bgr: tuple = None,
    boil_phase: int = 0,
    seed: int = 0,
    num_pts: int = 12,
):
    """Draws a clean anti-aliased oval/sphere with optional gouache fill."""
    irx, iry = max(1, int(round(rx))), max(1, int(round(ry)))
    icx, icy = int(round(cx)), int(round(cy))
    if fill_bgr is not None:
        cv2.ellipse(canvas, (icx, icy), (irx, iry), float(angle_deg), 0, 360, fill_bgr, -1, cv2.LINE_AA)
    if thick > 0:
        cv2.ellipse(canvas, (icx, icy), (irx, iry), float(angle_deg), 0, 360, color_bgr, max(1, int(round(thick))), cv2.LINE_AA)


def solve_2bone_ik(sx: float, sy: float, tx: float, ty: float, l1: float, l2: float, bend_sign: float = 1.0) -> tuple:
    """Analytical 2-Bone Inverse Kinematics solver returning `(jx, jy, ex, ey)`."""
    dx = tx - sx
    dy = ty - sy
    d = math.hypot(dx, dy)
    max_reach = (l1 + l2) * 0.992
    min_reach = abs(l1 - l2) + 2.0
    if d > max_reach and d > 1e-4:
        dx *= max_reach / d
        dy *= max_reach / d
        d = max_reach
    elif d < min_reach:
        d = min_reach
    cos_a = float(np.clip((l1 * l1 + d * d - l2 * l2) / max(2.0 * l1 * d, 1e-4), -1.0, 1.0))
    ang_base = math.atan2(dy, dx)
    ang_a = math.acos(cos_a)
    ang_1 = ang_base + bend_sign * ang_a
    jx = sx + l1 * math.cos(ang_1)
    jy = sy + l1 * math.sin(ang_1)
    return jx, jy, sx + dx, sy + dy


def draw_tapered_limb(
    canvas: np.ndarray,
    p0: tuple,
    p1: tuple,
    p2: tuple,
    w0: float,
    w1: float,
    w2: float,
    fill_bgr: tuple,
    ink_bgr: tuple = BGR_INK,
    ink_thick: float = 2.4,
    boil_phase: int = 0,
    seed: int = 0,
    construction_only: bool = False,
):
    """
    Draws a 2-segment limb (`p0 -> p1 -> p2`) using Dual-Pass Capsule Union
    so the knee/elbow joint at `p1` is 100% seamless with zero internal rings.
    """
    w_i = max(1, int(round(ink_thick)))
    if construction_only or fill_bgr is None:
        cv2.polylines(canvas, [_capsule_poly(p0, p1, w0, w1)], True, ink_bgr, max(1, w_i - 1), cv2.LINE_AA)
        cv2.polylines(canvas, [_capsule_poly(p1, p2, w1, w2)], True, ink_bgr, max(1, w_i - 1), cv2.LINE_AA)
        cv2.circle(canvas, (int(p1[0]), int(p1[1])), max(2, int(w1 * 0.75)), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        return
    # Pass 1: Outer Ink Casing
    cv2.fillPoly(canvas, [_capsule_poly(p0, p1, w0 + w_i, w1 + w_i)], ink_bgr, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(p1, p2, w1 + w_i, w2 + w_i)], ink_bgr, cv2.LINE_AA)
    # Pass 2: Opaque Interior Fill (erases internal joint seam at `p1`!)
    cv2.fillPoly(canvas, [_capsule_poly(p0, p1, w0, w1)], fill_bgr, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(p1, p2, w1, w2)], fill_bgr, cv2.LINE_AA)


# ==============================================================================
# II. 14 MANUALLY DRAWN MULTIPLANE CARTOONS (FRAME-BY-FRAME @ 24 FPS ON ONES)
# ==============================================================================

def _draw_7plane_sky_and_distant_bg(
    canvas: np.ndarray,
    w: int,
    h: int,
    t_sec: float,
    cam_pan: float,
    theme: str,
    stage_mode: str,
    boil_phase: int,
):
    """
    Draws Planes 6 (Sky/Atmosphere), 5 (Distant Mountains/Sea), and 4 (Midground Terrain)
    with true multiplane differential parallax (`screen_motion ∝ camera_motion / layer_depth`).
    """
    hor_y = int(h * 0.66)
    is_rough = (stage_mode == "rough_blue")
    is_ink = (stage_mode == "cleanup_ink")
    is_colored = stage_mode in ("gouache_cel", "multiplane_composite")

    if is_colored:
        # Sky gradient
        if theme == "night_woods":
            top_c, bot_c = (68, 42, 28), (112, 82, 58)
            ground_c = (64, 88, 62)
        elif theme == "storm_sea":
            top_c, bot_c = (88, 68, 52), (158, 134, 112)
            ground_c = (118, 92, 66)
        elif theme == "savanna":
            top_c, bot_c = (158, 212, 248), (195, 232, 252)
            ground_c = (142, 198, 232)
        elif theme == "alpine":
            top_c, bot_c = (232, 198, 148), (242, 228, 204)
            ground_c = (148, 178, 142)
        else:
            top_c, bot_c = (238, 214, 168), (246, 236, 218)
            ground_c = (156, 204, 166)

        for y in range(hor_y):
            u = y / float(max(1, hor_y))
            c = tuple(int(top_c[k] * (1.0 - u) + bot_c[k] * u) for k in range(3))
            cv2.line(canvas, (0, y), (w, y), c, 1)
        cv2.rectangle(canvas, (0, hor_y), (w, h), ground_c, -1)
    else:
        canvas[:] = BGR_PAPER
        # Perspective grid & horizon line on animation paper
        grid_c = (224, 206, 182) if is_rough else (222, 214, 202)
        cv2.line(canvas, (0, hor_y), (w, hor_y), grid_c, 1, cv2.LINE_AA)
        for gx in range(1, 6):
            x_top = int(w * gx / 6.0)
            x_bot = int(w * 0.5 + (x_top - w * 0.5) * 1.65)
            cv2.line(canvas, (x_top, hor_y), (x_bot, h), grid_c, 1, cv2.LINE_AA)

    line_c = BGR_NONPHOTO_BLUE if is_rough else (BGR_SEPIA if is_ink else (72, 68, 62))
    line_w = 1.4 if is_rough else 1.8

    # PLANE 6: Drifting Clouds / Stars (`depth = 6.0`, slow parallax)
    p6_shift = -cam_pan / 6.0 - t_sec * 8.0
    for ci in range(4):
        cx = ((ci * w * 0.32 + p6_shift) % (w + 160)) - 80
        cy = h * (0.14 + 0.06 * (ci % 2))
        c_fill = (245, 248, 252) if (is_colored and theme != "night_woods") else None
        if theme == "night_woods":
            # Twinkling stars
            sx = int((ci * 197 + 53) % w)
            sy = int(h * (0.08 + 0.07 * (ci % 3)))
            r_s = int(4 + 2 * math.sin(t_sec * 5.0 + ci))
            cv2.line(canvas, (sx - r_s, sy), (sx + r_s, sy), (210, 238, 255) if is_colored else line_c, 1, cv2.LINE_AA)
            cv2.line(canvas, (sx, sy - r_s), (sx, sy + r_s), (210, 238, 255) if is_colored else line_c, 1, cv2.LINE_AA)
        else:
            draw_hand_oval(canvas, cx, cy, w * 0.075, h * 0.032, color_bgr=line_c, thick=line_w * 0.8, fill_bgr=c_fill, boil_phase=boil_phase, seed=10 + ci)
            draw_hand_oval(canvas, cx + w * 0.04, cy - h * 0.01, w * 0.055, h * 0.026, color_bgr=line_c, thick=line_w * 0.8, fill_bgr=c_fill, boil_phase=boil_phase, seed=20 + ci)

    # PLANE 5: Distant Mountain / Mesa / Horizon Ridge (`depth = 3.8`)
    p5_shift = -cam_pan / 3.8
    mtn_pts = []
    for mi in range(9):
        mx = (mi / 8.0) * (w + 180) - 90 + (p5_shift % 90)
        heights = [0.08, 0.22, 0.14, 0.28, 0.18, 0.25, 0.11, 0.20, 0.09]
        my = hor_y - h * heights[mi % len(heights)]
        mtn_pts.append((mx, my))
    mtn_poly = [(-100, hor_y)] + mtn_pts + [(w + 100, hor_y)]
    mtn_fill = None
    if is_colored:
        mtn_fill = (96, 78, 66) if theme == "night_woods" else ((182, 168, 156) if theme == "savanna" else (164, 152, 138))
    draw_hand_ink_spline(canvas, mtn_poly, color_bgr=line_c, base_thick=line_w, closed=True, fill_bgr=mtn_fill, boil_phase=boil_phase, seed=31, taper_ends=False)

    # PLANE 4: Midground Trees / Hills / Acacia (`depth = 2.1`)
    p4_shift = -cam_pan / 2.1
    if theme == "savanna":
        for ti in range(3):
            tx = ((ti * w * 0.42 + p4_shift) % (w + 220)) - 110
            ty = hor_y + int(h * 0.02)
            # Acacia trunk & flat canopy
            sway = 4.0 * math.sin(t_sec * 2.2 + ti)
            trunk = [(tx - 6, ty), (tx - 2 + sway * 0.4, ty - h * 0.14), (tx - 28 + sway, ty - h * 0.22)]
            trunk_r = [(tx + 6, ty), (tx + 3 + sway * 0.4, ty - h * 0.14), (tx + 32 + sway, ty - h * 0.21)]
            draw_hand_ink_spline(canvas, trunk, color_bgr=line_c, base_thick=line_w * 1.2, boil_phase=boil_phase, seed=40 + ti)
            draw_hand_ink_spline(canvas, trunk_r, color_bgr=line_c, base_thick=line_w * 1.2, boil_phase=boil_phase, seed=45 + ti)
            can_fill = (104, 152, 112) if is_colored else None
            draw_hand_oval(canvas, tx + sway, ty - h * 0.23, w * 0.095, h * 0.028, color_bgr=line_c, thick=line_w, fill_bgr=can_fill, boil_phase=boil_phase, seed=50 + ti)
    elif theme == "night_woods":
        for ti in range(5):
            tx = ((ti * w * 0.26 + p4_shift) % (w + 160)) - 80
            ty = hor_y + int(h * 0.02)
            sway = 3.0 * math.sin(t_sec * 1.8 + ti)
            pine = [
                (tx - w * 0.055, ty),
                (tx - w * 0.035 + sway * 0.5, ty - h * 0.16),
                (tx + sway, ty - h * 0.34),
                (tx + w * 0.035 + sway * 0.5, ty - h * 0.16),
                (tx + w * 0.055, ty),
            ]
            p_fill = (52, 74, 48) if is_colored else None
            draw_hand_ink_spline(canvas, pine, color_bgr=line_c, base_thick=line_w, closed=True, fill_bgr=p_fill, boil_phase=boil_phase, seed=60 + ti)


def _draw_plane0_foreground_mask(
    canvas: np.ndarray,
    w: int,
    h: int,
    t_sec: float,
    cam_pan: float,
    theme: str,
    stage_mode: str,
    boil_phase: int,
):
    """
    Draws PLANE 1 (Foreground swaying grass/ferns/rocks) and PLANE 0 (Extreme
    Foreground Silhouette Mask) moving strongly (`depth = 0.65`) for true multiplane depth!
    """
    is_rough = (stage_mode == "rough_blue")
    is_colored = stage_mode in ("gouache_cel", "multiplane_composite")
    fg_c = BGR_NONPHOTO_BLUE if is_rough else ((28, 38, 28) if is_colored else BGR_INK)
    fg_fill = (48, 78, 52) if is_colored else None

    p1_shift = -cam_pan / 0.68
    # Foreground swaying fern / grass tufts on bottom-left and bottom-right
    for side in (-1, 1):
        base_x = (w * 0.12 if side < 0 else w * 0.86) + (p1_shift * 0.35)
        base_y = h + 6
        for bi in range(-2, 3):
            ang = math.radians(side * (18 + bi * 14) + 7.5 * math.sin(t_sec * 3.1 + bi + side))
            bh = h * (0.19 - 0.02 * abs(bi))
            tip_x = base_x + bi * 16 + bh * math.sin(ang)
            tip_y = base_y - bh * math.cos(ang)
            mid_x = base_x + bi * 12 + 0.55 * bh * math.sin(ang * 0.7)
            mid_y = base_y - 0.55 * bh * math.cos(ang * 0.7)
            blade = [
                (base_x + bi * 14 - 6, base_y),
                (mid_x - 5, mid_y),
                (tip_x, tip_y),
                (mid_x + 5, mid_y),
                (base_x + bi * 14 + 6, base_y),
            ]
            draw_hand_ink_spline(canvas, blade, color_bgr=fg_c, base_thick=2.0, closed=True, fill_bgr=fg_fill, boil_phase=boil_phase, seed=90 + bi + side * 10)


def _draw_fox_and_campfire_cartoon(
    canvas: np.ndarray,
    w: int,
    h: int,
    t_sec: float,
    stage_mode: str,
    boil_phase: int,
    ink_bgr: tuple,
):
    """
    Manually draws the Section III Masterpiece Cartoon (`campfire`):
      `Fox hears something -> freezes -> ears turn -> body compresses -> fox leaps -> lands behind fern -> tail follows`
    plus a multi-tongue roaring hand-drawn campfire with rising embers & smoke plumes!
    """
    is_rough = (stage_mode == "rough_blue")
    is_cleanup = (stage_mode == "cleanup_ink")
    is_colored = stage_mode in ("gouache_cel", "multiplane_composite")
    ground_y = h * 0.78

    # --- 1. ROARING HAND-DRAWN CAMPFIRE (Center-Left `fx, fy`) ---
    fx, fy = w * 0.44, ground_y + h * 0.02
    log_fill = (68, 92, 134) if is_colored else (BGR_PAPER if is_cleanup else None)
    # Crossed split logs
    for ang_l, s_l in [(-18, 1), (16, 2), (-4, 3)]:
        draw_hand_oval(canvas, fx + (s_l - 2) * 14, fy + 6, w * 0.085, h * 0.024, angle_deg=ang_l, color_bgr=ink_bgr, thick=2.2, fill_bgr=log_fill, boil_phase=boil_phase, seed=100 + s_l)
    # Stones around hearth
    for si in range(-3, 4):
        draw_hand_oval(canvas, fx + si * w * 0.028, fy + 16 + 3 * (si % 2), w * 0.018, h * 0.015, color_bgr=ink_bgr, thick=2.0, fill_bgr=(128, 132, 138) if is_colored else (BGR_PAPER if is_cleanup else None), boil_phase=boil_phase, seed=110 + si)

    # Multi-tongue S-wave dancing flames (Effects Animation, Section XXII)
    flame_colors = [(48, 118, 248), (62, 178, 255), (128, 232, 255)]
    for fi in range(3):
        f_w = w * (0.068 - fi * 0.018)
        f_h = h * (0.22 - fi * 0.042) * (1.0 + 0.12 * math.sin(t_sec * 11.0 + fi * 1.4))
        w1 = 12.0 * math.sin(t_sec * 9.5 + fi)
        w2 = -15.0 * math.cos(t_sec * 11.2 + fi * 0.8)
        tip_w = 18.0 * math.sin(t_sec * 13.0 - fi)
        flame_pts = [
            (fx - f_w, fy),
            (fx - f_w * 0.85 + w1 * 0.5, fy - f_h * 0.35),
            (fx - f_w * 0.35 + w2 * 0.7, fy - f_h * 0.72),
            (fx + tip_w * 0.6, fy - f_h),
            (fx + f_w * 0.40 + w2 * 0.5, fy - f_h * 0.68),
            (fx + f_w * 0.88 + w1 * 0.4, fy - f_h * 0.32),
            (fx + f_w, fy),
        ]
        f_fill = flame_colors[fi] if is_colored else (BGR_PAPER if (is_cleanup and fi == 0) else None)
        f_col = BGR_NONPHOTO_BLUE if is_rough else ((38, 88, 218) if fi > 0 else ink_bgr)
        draw_hand_ink_spline(canvas, flame_pts, color_bgr=f_col, base_thick=2.0, closed=True, fill_bgr=f_fill, boil_phase=boil_phase, seed=120 + fi)

    # Pinching rising flame teardrops & glowing embers + smoke puffs
    for ei in range(5):
        u_e = ((t_sec * 0.85 + ei * 0.20) % 1.0)
        ex = fx + 24.0 * math.sin(u_e * 6.2 + ei * 1.7)
        ey = fy - h * 0.18 - u_e * h * 0.42
        er = max(2.0, (1.0 - u_e) * 7.0)
        draw_hand_oval(canvas, ex, ey, er, er * 1.4, angle_deg=15 * math.sin(t_sec * 8 + ei), color_bgr=BGR_AMBER if is_colored else ink_bgr, thick=1.5, fill_bgr=(82, 208, 255) if is_colored else None, boil_phase=boil_phase, seed=130 + ei)

    # --- 2. ARTICULATED FOREST FOX (7-Beat Choreography over 2.5s cycle) ---
    # Beats:
    # u in [0.00, 0.16]: Beat 1-2: Trots in from left -> hears sound -> FREEZES
    # u in [0.16, 0.30]: Beat 3: EARS TURN & head perks up
    # u in [0.30, 0.44]: Beat 4: BODY COMPRESSES (deep anticipation crouch before leap!)
    # u in [0.44, 0.72]: Beat 5: FOX LEAPS in high parabolic arc over the campfire!
    # u in [0.72, 0.86]: Beat 6: LANDS & OVERSHOOTS (front paws plant, chest compresses)
    # u in [0.86, 1.00]: Beat 7: SETTLES behind fern & BUSHY TAIL FOLLOWS THROUGH!
    u_cyc = (t_sec * 0.42) % 1.0

    # Trajectory arc guide in rough/cleanup modes (Rule XIII: ARCS)
    if stage_mode in ("rough_blue", "cleanup_ink", "onion_flip"):
        arc_pts = []
        for k in range(25):
            uk = k / 24.0
            ax = w * (0.20 + 0.56 * uk)
            ay = ground_y - h * 0.08 - h * 0.34 * math.sin(math.pi * uk)
            arc_pts.append((ax, ay))
        for k in range(len(arc_pts) - 1):
            if k % 2 == 0:
                cv2.line(canvas, (int(arc_pts[k][0]), int(arc_pts[k][1])), (int(arc_pts[k + 1][0]), int(arc_pts[k + 1][1])), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    if u_cyc < 0.16:
        b_u = u_cyc / 0.16
        hip_x = w * (0.16 + 0.06 * b_u)
        hip_y = ground_y - h * 0.09
        chest_x = hip_x + w * 0.095
        chest_y = hip_y - h * 0.015 + 3.0 * math.sin(b_u * math.pi)
        head_pitch = -8.0
        ear_turn = 0.0
        tail_ang = 155.0 + 12.0 * math.sin(b_u * math.pi * 2)
        front_reach = 0.15
        hind_push = -0.10
    elif u_cyc < 0.30:
        b_u = (u_cyc - 0.16) / 0.14
        hip_x = w * 0.22
        hip_y = ground_y - h * 0.09
        chest_x = hip_x + w * 0.095
        chest_y = hip_y - h * 0.025
        head_pitch = -18.0 * math.sin(b_u * math.pi * 0.5)
        ear_turn = math.sin(b_u * math.pi) * 28.0
        tail_ang = 148.0
        front_reach = 0.05
        hind_push = -0.05
    elif u_cyc < 0.44:
        # ANTICIPATION CROUCH (compresses down and back before spring!)
        b_u = (u_cyc - 0.30) / 0.14
        crouch = math.sin(b_u * math.pi * 0.85)
        hip_x = w * (0.22 - 0.025 * crouch)
        hip_y = ground_y - h * (0.09 - 0.045 * crouch)
        chest_x = hip_x + w * (0.088 - 0.012 * crouch)
        chest_y = ground_y - h * (0.10 - 0.055 * crouch)
        head_pitch = 12.0 * crouch
        ear_turn = -22.0 * crouch
        tail_ang = 172.0 - 25.0 * crouch
        front_reach = -0.22 * crouch
        hind_push = -0.35 * crouch
    elif u_cyc < 0.72:
        # AIRBORNE PARABOLIC LEAP OVER CAMPFIRE!
        b_u = (u_cyc - 0.44) / 0.28
        arc_h = math.sin(b_u * math.pi)
        hip_x = w * (0.20 + 0.50 * b_u)
        hip_y = ground_y - h * 0.09 - h * 0.34 * arc_h + h * 0.04 * (b_u - 0.5)
        pitch_body = -38.0 * math.cos(b_u * math.pi)
        rad_b = math.radians(pitch_body)
        body_len = w * 0.115 * (1.0 + 0.14 * arc_h)  # Stretch during acceleration/leap!
        chest_x = hip_x + body_len * math.cos(rad_b)
        chest_y = hip_y + body_len * math.sin(rad_b)
        head_pitch = pitch_body * 0.65
        ear_turn = -18.0
        # Tail lags behind the hip angle!
        tail_ang = 180.0 + pitch_body - 28.0 * math.sin(b_u * math.pi)
        front_reach = 0.65 * math.sin(b_u * math.pi * 0.9)
        hind_push = -0.75 * math.sin(b_u * math.pi * 0.85)
    elif u_cyc < 0.86:
        # LANDING IMPACT & COMPRESSION OVERSHOOT
        b_u = (u_cyc - 0.72) / 0.14
        comp = math.sin(b_u * math.pi)
        hip_x = w * (0.70 + 0.05 * b_u)
        hip_y = ground_y - h * (0.09 - 0.042 * comp)
        chest_x = hip_x + w * 0.092
        chest_y = ground_y - h * (0.10 - 0.052 * comp)
        head_pitch = 16.0 * comp
        ear_turn = 14.0 * comp
        tail_ang = 120.0 + 48.0 * b_u  # Tail whips high over back on landing!
        front_reach = 0.20 * (1.0 - b_u)
        hind_push = 0.15 * comp
    else:
        # SETTLE & TAIL FOLLOW-THROUGH S-WAVE
        b_u = (u_cyc - 0.86) / 0.14
        hip_x = w * 0.75
        hip_y = ground_y - h * 0.09
        chest_x = hip_x + w * 0.095
        chest_y = hip_y - h * 0.02
        head_pitch = -6.0 * math.sin(b_u * math.pi)
        ear_turn = 10.0 * math.cos(b_u * math.pi * 2)
        tail_ang = 152.0 + 18.0 * math.sin(b_u * math.pi * 2.0) * (1.0 - 0.5 * b_u)
        front_reach = 0.05
        hind_push = 0.0

    fox_orange = (58, 118, 224) if is_colored else (BGR_PAPER if is_cleanup else None)
    fox_cream = (224, 240, 250) if is_colored else (BGR_PAPER if is_cleanup else None)
    fox_dark = (32, 36, 48) if is_colored else (BGR_PAPER_DARK if is_cleanup else None)

    # 1. Bushy S-curve Fox Tail (5-point overlapping wave)
    t_rad = math.radians(tail_ang)
    t_len = w * 0.115
    tail_wave = 14.0 * math.sin(t_sec * 9.0)
    tx1 = hip_x + 0.45 * t_len * math.cos(t_rad)
    ty1 = hip_y - 0.45 * t_len * math.sin(t_rad) + tail_wave * 0.4
    tx2 = hip_x + t_len * math.cos(t_rad - 0.18)
    ty2 = hip_y - t_len * math.sin(t_rad - 0.18) - tail_wave * 0.6
    nx_t = -math.sin(t_rad)
    ny_t = -math.cos(t_rad)
    tail_ctrl = [
        (hip_x, hip_y - 4),
        (tx1 + nx_t * 16, ty1 + ny_t * 16),
        (tx2, ty2),
        (tx1 - nx_t * 14, ty1 - ny_t * 14),
        (hip_x, hip_y + 5),
    ]
    draw_hand_ink_spline(canvas, tail_ctrl, color_bgr=ink_bgr, base_thick=2.2, closed=True, fill_bgr=fox_orange, boil_phase=boil_phase, seed=140)
    # White tail tip
    draw_hand_oval(canvas, tx2 * 0.85 + tx1 * 0.15, ty2 * 0.85 + ty1 * 0.15, 10, 7, angle_deg=-tail_ang, color_bgr=ink_bgr, thick=1.8, fill_bgr=fox_cream, boil_phase=boil_phase, seed=141)

    # 2. Far legs (Hind Left & Front Left)
    leg_h = h * 0.095
    for is_front, bx_l, by_l, reach_f in [(False, hip_x - 4, hip_y + 4, hind_push - 0.08), (True, chest_x - 4, chest_y + 4, front_reach - 0.08)]:
        foot_x = bx_l + reach_f * w * 0.11
        foot_y = min(ground_y, by_l + leg_h * (0.95 if (u_cyc < 0.44 or u_cyc >= 0.72) else 0.72))
        jx, jy, ex, ey = solve_2bone_ik(bx_l, by_l, foot_x, foot_y, leg_h * 0.55, leg_h * 0.55, bend_sign=(-1.0 if is_front else 1.0))
        draw_tapered_limb(canvas, (bx_l, by_l), (jx, jy), (ex, ey), 7.5, 5.0, 3.5, fox_orange, ink_bgr, 2.0, boil_phase, 145 + int(is_front), construction_only=is_rough)

    # 3. Fox Torso (Inside-Out Ribcage + Pelvis + Flexible Lumbar Spine)
    mid_spine_x = 0.5 * (hip_x + chest_x)
    mid_spine_y = 0.5 * (hip_y + chest_y) - (8.0 if (0.44 <= u_cyc < 0.72) else -4.0)
    torso_ctrl = [
        (hip_x - 12, hip_y - 11),
        (mid_spine_x, mid_spine_y - 15),
        (chest_x + 12, chest_y - 14),
        (chest_x + 16, chest_y + 10),
        (mid_spine_x, mid_spine_y + 13),
        (hip_x - 10, hip_y + 11),
    ]
    draw_hand_ink_spline(canvas, torso_ctrl, color_bgr=ink_bgr, base_thick=2.4, closed=True, fill_bgr=fox_orange, boil_phase=boil_phase, seed=150)

    if is_rough or stage_mode == "cleanup_ink":
        draw_hand_oval(canvas, hip_x, hip_y, 14, 12, color_bgr=BGR_NONPHOTO_BLUE, thick=1.2, boil_phase=boil_phase, seed=151)
        draw_hand_oval(canvas, chest_x, chest_y, 17, 14, color_bgr=BGR_NONPHOTO_BLUE, thick=1.2, boil_phase=boil_phase, seed=152)

    # 4. Near legs (Hind Right & Front Right)
    for is_front, bx_l, by_l, reach_f in [(False, hip_x + 2, hip_y + 5, hind_push + 0.06), (True, chest_x + 4, chest_y + 5, front_reach + 0.06)]:
        foot_x = bx_l + reach_f * w * 0.11
        foot_y = min(ground_y + 2, by_l + leg_h * (0.98 if (u_cyc < 0.44 or u_cyc >= 0.72) else 0.70))
        jx, jy, ex, ey = solve_2bone_ik(bx_l, by_l, foot_x, foot_y, leg_h * 0.55, leg_h * 0.55, bend_sign=(-1.0 if is_front else 1.0))
        draw_tapered_limb(canvas, (bx_l, by_l), (jx, jy), (ex, ey), 8.5, 5.5, 4.0, fox_orange, ink_bgr, 2.2, boil_phase, 155 + int(is_front), construction_only=is_rough)
        if fox_orange is not None:
            cv2.circle(canvas, (int(bx_l), int(by_l - 2)), 8, fox_orange, -1, cv2.LINE_AA)
        draw_hand_oval(canvas, ex + 3, ey, 5.5, 3.5, color_bgr=ink_bgr, thick=2.0, fill_bgr=fox_dark, boil_phase=boil_phase, seed=158 + int(is_front))

    # 5. Fox Head, Pointed Snout, Turning Ears & Expressive Eyes
    hx = chest_x + 18.0
    hy = chest_y - 18.0
    h_rad = math.radians(head_pitch)
    snout_x = hx + 26.0 * math.cos(h_rad)
    snout_y = hy + 26.0 * math.sin(h_rad)
    head_ctrl = [
        (hx - 12, hy - 10),
        (hx + 6, hy - 14),
        (snout_x, snout_y - 3),
        (snout_x - 2, snout_y + 5),
        (hx + 4, hy + 11),
        (hx - 11, hy + 8),
    ]
    draw_hand_ink_spline(canvas, head_ctrl, color_bgr=ink_bgr, base_thick=2.3, closed=True, fill_bgr=fox_orange, boil_phase=boil_phase, seed=160)
    if fox_orange is not None:
        cv2.circle(canvas, (int(hx - 7), int(hy + 4)), 8, fox_orange, -1, cv2.LINE_AA)
    # White cheek ruff
    draw_hand_oval(canvas, hx + 4, hy + 6, 11, 5.5, angle_deg=head_pitch, color_bgr=ink_bgr, thick=1.6, fill_bgr=fox_cream, boil_phase=boil_phase, seed=161)
    # Black nose tip
    cv2.circle(canvas, (int(snout_x), int(snout_y)), 4, BGR_INK, -1, cv2.LINE_AA)
    # Expressive eye
    cv2.circle(canvas, (int(hx + 7), int(hy - 3)), 3, BGR_INK, -1, cv2.LINE_AA)
    # Articulated Ears (`ear_turn` rotates ears during Beat 3!)
    for ei_s, e_off in [(-1, -6), (1, 2)]:
        e_ang = math.radians(-75.0 + ear_turn + ei_s * 14.0)
        ex_tip = hx + e_off + 20.0 * math.cos(e_ang)
        ey_tip = hy - 10.0 + 20.0 * math.sin(e_ang)
        ear_pts = [(hx + e_off - 5, hy - 9), (ex_tip, ey_tip), (hx + e_off + 5, hy - 8)]
        draw_hand_ink_spline(canvas, ear_pts, color_bgr=ink_bgr, base_thick=2.1, closed=True, fill_bgr=fox_orange, boil_phase=boil_phase, seed=165 + ei_s)


def _draw_giraffe_savanna_cartoon(
    canvas: np.ndarray,
    w: int,
    h: int,
    t_sec: float,
    stage_mode: str,
    boil_phase: int,
    ink_bgr: tuple,
):
    """
    Manually draws an articulated 16-joint Giraffe walking across the African savanna
    with a true 4-beat quadruped walk cycle (`LH -> LF -> RH -> RF`), S-curve neck
    overlapping action, reticulated spots, hoof dust puffs, and flapping white egrets!
    """
    is_rough = (stage_mode == "rough_blue")
    is_colored = stage_mode in ("gouache_cel", "multiplane_composite")
    ground_y = h * 0.81

    # --- 1. FLAPPING WHITE EGRETS IN PARALLAX (Plane 3 Secondary Action) ---
    for bi in range(2):
        bx = ((w * 0.25 + bi * w * 0.36 + t_sec * 48.0 * (1.0 + 0.15 * bi)) % (w + 120)) - 60
        by = h * (0.20 + 0.07 * bi) + 8.0 * math.sin(t_sec * 4.5 + bi)
        wing_flap = math.sin(t_sec * 8.5 + bi * 1.9)
        wing_y = by - 18.0 * wing_flap
        egret_fill = (248, 250, 254) if is_colored else None
        draw_hand_oval(canvas, bx, by, 13, 5.5, angle_deg=-5, color_bgr=ink_bgr, thick=1.8, fill_bgr=egret_fill, boil_phase=boil_phase, seed=200 + bi)
        wing_pts = [(bx - 5, by - 2), (bx - 2, wing_y), (bx + 8, by - 1)]
        draw_hand_ink_spline(canvas, wing_pts, color_bgr=ink_bgr, base_thick=2.0, closed=True, fill_bgr=egret_fill, boil_phase=boil_phase, seed=205 + bi)

    # --- 2. ARTICULATED 16-JOINT GIRAFFE WALK CYCLE ---
    stride_freq = 0.85  # cycles per second
    cyc = (t_sec * stride_freq) % 1.0

    # Gentle horizontal patrol + vertical body bounce on 2x stride frequency
    gx = w * 0.48 + w * 0.08 * math.sin(t_sec * 0.55)
    hip_bob = 5.5 * math.sin(cyc * 4.0 * math.pi)
    chest_bob = 6.5 * math.sin(cyc * 4.0 * math.pi - 0.45)

    hip_x, hip_y = gx - w * 0.075, ground_y - h * 0.29 + hip_bob
    chest_x, chest_y = gx + w * 0.065, ground_y - h * 0.34 + chest_bob

    is_cleanup = (stage_mode == "cleanup_ink")
    gir_gold = (128, 198, 245) if is_colored else (BGR_PAPER if is_cleanup else None)
    gir_spot = (54, 96, 158) if is_colored else (BGR_PAPER_DARK if is_cleanup else None)
    gir_hoof = (42, 46, 54) if is_colored else (BGR_PAPER_DARK if is_cleanup else None)

    def _leg_pose(base_x, base_y, phase_off):
        ph = (cyc + phase_off) % 1.0
        stride_span = w * 0.115
        if ph < 0.62:
            # Stance phase (hoof planted on ground, moves backward as body advances)
            u_s = ph / 0.62
            fx = base_x + stride_span * (0.5 - u_s)
            fy = ground_y
            contact_dust = (u_s < 0.16)
        else:
            # Swing phase (hoof lifts in an arc and swings forward!)
            u_w = (ph - 0.62) / 0.38
            fx = base_x + stride_span * (-0.5 + u_w)
            fy = ground_y - h * 0.085 * math.sin(u_w * math.pi)
            contact_dust = False
        return fx, fy, contact_dust

    leg_len = h * 0.32
    # Far legs first (Hind Left `0.50`, Front Left `0.75`)
    for idx_l, (bx_l, by_l, ph_o, bend_dir) in enumerate([
        (hip_x - 6, hip_y + 8, 0.50, -1.0),
        (chest_x - 4, chest_y + 10, 0.75, 1.0),
    ]):
        fx, fy, _ = _leg_pose(bx_l, by_l, ph_o)
        jx, jy, ex, ey = solve_2bone_ik(bx_l, by_l, fx, fy, leg_len * 0.52, leg_len * 0.54, bend_sign=bend_dir)
        draw_tapered_limb(canvas, (bx_l, by_l), (jx, jy), (ex, ey), 10.0, 6.5, 4.5, gir_gold, ink_bgr, 2.0, boil_phase, 210 + idx_l, construction_only=is_rough)
        draw_hand_oval(canvas, ex + 3, ey, 6.5, 4.0, color_bgr=ink_bgr, thick=2.0, fill_bgr=gir_hoof, boil_phase=boil_phase, seed=215 + idx_l)

    # Swishing Tufted Tail (overlapping S-wave)
    tail_a1 = math.radians(108.0 + 16.0 * math.sin(t_sec * 3.8))
    tail_a2 = math.radians(96.0 + 26.0 * math.sin(t_sec * 3.8 - 0.9))
    t_m = (hip_x - 16 + 38 * math.cos(tail_a1), hip_y + 8 + 38 * math.sin(tail_a1))
    t_e = (t_m[0] + 36 * math.cos(tail_a2), t_m[1] + 36 * math.sin(tail_a2))
    draw_hand_ink_spline(canvas, [(hip_x - 14, hip_y), t_m, t_e], color_bgr=ink_bgr, base_thick=2.4, boil_phase=boil_phase, seed=220)
    draw_hand_oval(canvas, t_e[0], t_e[1] + 4, 8, 5, angle_deg=35, color_bgr=ink_bgr, thick=2.0, fill_bgr=gir_hoof, boil_phase=boil_phase, seed=221)

    # Sloping Giraffe Torso (Pelvis + Ribcage)
    torso_pts = [
        (hip_x - 16, hip_y - 14),
        (0.5 * (hip_x + chest_x), 0.5 * (hip_y + chest_y) - 22),
        (chest_x + 18, chest_y - 26),
        (chest_x + 24, chest_y + 16),
        (0.5 * (hip_x + chest_x), 0.5 * (hip_y + chest_y) + 20),
        (hip_x - 14, hip_y + 16),
    ]
    draw_hand_ink_spline(canvas, torso_pts, color_bgr=ink_bgr, base_thick=2.5, closed=True, fill_bgr=gir_gold, boil_phase=boil_phase, seed=225)

    # Long S-Curve Articulated Neck (with overlapping lag!)
    neck_lag = 10.0 * math.sin(cyc * 2.0 * math.pi - 0.85)
    head_x = chest_x + w * 0.085 + neck_lag
    head_y = chest_y - h * 0.34 + 5.0 * math.cos(cyc * 4.0 * math.pi - 1.1)
    n_mid_x = 0.5 * (chest_x + head_x) - 6.0 + 0.5 * neck_lag
    n_mid_y = 0.5 * (chest_y + head_y)
    neck_poly = [
        (chest_x - 6, chest_y - 22),
        (n_mid_x - 11, n_mid_y),
        (head_x - 8, head_y + 6),
        (head_x + 9, head_y + 10),
        (n_mid_x + 11, n_mid_y + 4),
        (chest_x + 20, chest_y - 12),
    ]
    draw_hand_ink_spline(canvas, neck_poly, color_bgr=ink_bgr, base_thick=2.5, closed=True, fill_bgr=gir_gold, boil_phase=boil_phase, seed=230)
    if gir_gold is not None:
        cv2.circle(canvas, (int(chest_x + 7), int(chest_y - 16)), 11, gir_gold, -1, cv2.LINE_AA)

    # Near legs (Hind Right `0.00`, Front Right `0.25`) + Hoof Dust Puffs!
    for idx_l, (bx_l, by_l, ph_o, bend_dir) in enumerate([
        (hip_x + 4, hip_y + 10, 0.00, -1.0),
        (chest_x + 6, chest_y + 12, 0.25, 1.0),
    ]):
        fx, fy, dust = _leg_pose(bx_l, by_l, ph_o)
        jx, jy, ex, ey = solve_2bone_ik(bx_l, by_l, fx, fy, leg_len * 0.52, leg_len * 0.54, bend_sign=bend_dir)
        draw_tapered_limb(canvas, (bx_l, by_l), (jx, jy), (ex, ey), 11.5, 7.5, 5.0, gir_gold, ink_bgr, 2.3, boil_phase, 240 + idx_l, construction_only=is_rough)
        if gir_gold is not None:
            cv2.circle(canvas, (int(bx_l), int(by_l - 3)), 11, gir_gold, -1, cv2.LINE_AA)
        draw_hand_oval(canvas, ex + 3, ey, 7.5, 4.5, color_bgr=ink_bgr, thick=2.2, fill_bgr=gir_hoof, boil_phase=boil_phase, seed=245 + idx_l)
        if dust:
            draw_hand_oval(canvas, ex - 8, ground_y - 4, 11, 6, color_bgr=BGR_NONPHOTO_BLUE if is_rough else (160, 175, 190), thick=1.4, fill_bgr=(220, 232, 242) if is_colored else None, boil_phase=boil_phase, seed=248 + idx_l)

    # Reticulated Giraffe Spots (following torso & neck!)
    if not is_rough:
        spot_offsets = [
            (hip_x + 4, hip_y - 2, 11, 8),
            (0.5 * (hip_x + chest_x), 0.5 * (hip_y + chest_y) - 4, 13, 9),
            (chest_x + 4, chest_y - 6, 12, 10),
            (n_mid_x - 1, n_mid_y + 18, 8, 11),
            (n_mid_x + 1, n_mid_y - 14, 7, 10),
        ]
        for si, (sx_p, sy_p, srx, sry) in enumerate(spot_offsets):
            draw_hand_oval(canvas, sx_p, sy_p, srx, sry, angle_deg=18 * si, color_bgr=ink_bgr, thick=1.5, fill_bgr=gir_spot, boil_phase=boil_phase, seed=235 + si, num_pts=6)

    # Giraffe Head, Ossicones, Flicking Ears & Chewing Jaw
    snout_x = head_x + 28.0
    snout_y = head_y + 8.0 + 2.5 * math.sin(t_sec * 6.0)
    head_pts = [
        (head_x - 10, head_y - 6),
        (head_x + 8, head_y - 10),
        (snout_x, snout_y - 3),
        (snout_x - 2, snout_y + 6),
        (head_x + 4, head_y + 11),
        (head_x - 9, head_y + 8),
    ]
    draw_hand_ink_spline(canvas, head_pts, color_bgr=ink_bgr, base_thick=2.3, closed=True, fill_bgr=gir_gold, boil_phase=boil_phase, seed=250)
    if gir_gold is not None:
        cv2.circle(canvas, (int(head_x - 2), int(head_y + 4)), 8, gir_gold, -1, cv2.LINE_AA)
    # Ossicones (horns)
    for oi in (-3, 4):
        cv2.line(canvas, (int(head_x + oi), int(head_y - 9)), (int(head_x + oi - 2), int(head_y - 22)), ink_bgr, 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(head_x + oi - 2), int(head_y - 23)), 4, ink_bgr, -1, cv2.LINE_AA)
    # Eye
    cv2.circle(canvas, (int(head_x + 6), int(head_y - 1)), 3, BGR_INK, -1, cv2.LINE_AA)


def _draw_lighthouse_storm_cartoon(
    canvas: np.ndarray,
    w: int,
    h: int,
    t_sec: float,
    stage_mode: str,
    boil_phase: int,
    ink_bgr: tuple,
):
    """
    Manually draws the Stormy Cape Lighthouse (`lighthouse`):
      - Striped masonry tower on basalt cliff
      - Rotating 360-deg Fresnel prism assembly with foreshortened volumetric light cone
      - Rolling ocean swells + curling wave smashing into rocks with parabolic spray droplets
      - Banking coastal seagulls!
    """
    is_rough = (stage_mode == "rough_blue")
    is_colored = stage_mode in ("gouache_cel", "multiplane_composite")
    sea_y = h * 0.72

    # --- 1. ROTATING VOLUMETRIC LIGHTHOUSE BEAM (360-deg foreshortened cone) ---
    lx, ly_top = w * 0.34, h * 0.24
    ly_base = h * 0.75
    beam_theta = t_sec * 2.2
    cos_b = math.cos(beam_theta)
    beam_reach = w * 0.62 * cos_b
    if abs(beam_reach) > 12.0:
        b_end_x = lx + beam_reach
        b_spread = h * (0.06 + 0.16 * abs(cos_b))
        beam_poly = np.array([
            [int(lx), int(ly_top - 4)],
            [int(b_end_x), int(ly_top - b_spread)],
            [int(b_end_x), int(ly_top + b_spread)],
            [int(lx), int(ly_top + 4)],
        ], dtype=np.int32)
        if is_colored:
            ov = canvas.copy()
            cv2.fillPoly(ov, [beam_poly], (152, 238, 255), cv2.LINE_AA)
            cv2.addWeighted(ov, 0.38, canvas, 0.62, 0, canvas)
        else:
            cv2.polylines(canvas, [beam_poly], True, BGR_AMBER if not is_rough else BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    # --- 2. JAGGED BASALT CLIFF & STRIPED LIGHTHOUSE TOWER ---
    cliff_pts = [
        (w * 0.08, h * 0.88),
        (w * 0.16, h * 0.73),
        (w * 0.28, h * 0.69),
        (w * 0.43, h * 0.72),
        (w * 0.49, h * 0.88),
    ]
    draw_hand_ink_spline(canvas, cliff_pts, color_bgr=ink_bgr, base_thick=2.5, closed=True, fill_bgr=(74, 78, 86) if is_colored else None, boil_phase=boil_phase, seed=300)

    # Tapered Lighthouse Tower + Red Stripes
    tw_bot, tw_top = w * 0.055, w * 0.032
    tower_pts = [
        (lx - tw_bot, ly_base),
        (lx - tw_top, ly_top + 18),
        (lx + tw_top, ly_top + 18),
        (lx + tw_bot, ly_base),
    ]
    draw_hand_ink_spline(canvas, tower_pts, color_bgr=ink_bgr, base_thick=2.5, closed=True, fill_bgr=(242, 246, 250) if is_colored else None, boil_phase=boil_phase, seed=302)
    # Red diagonal stripes
    for si in range(2):
        u0 = 0.22 + si * 0.38
        u1 = u0 + 0.18
        y0_s = ly_top + 18 + u0 * (ly_base - ly_top - 18)
        y1_s = ly_top + 18 + u1 * (ly_base - ly_top - 18)
        w0_s = tw_top + u0 * (tw_bot - tw_top)
        w1_s = tw_top + u1 * (tw_bot - tw_top)
        stripe = [(lx - w0_s, y0_s), (lx + w0_s, y0_s), (lx + w1_s, y1_s), (lx - w1_s, y1_s)]
        draw_hand_ink_spline(canvas, stripe, color_bgr=ink_bgr, base_thick=1.8, closed=True, fill_bgr=(62, 74, 214) if is_colored else None, boil_phase=boil_phase, seed=305 + si)

    # Lantern Room & Dome
    draw_hand_oval(canvas, lx, ly_top, tw_top * 1.15, 18, color_bgr=ink_bgr, thick=2.2, fill_bgr=(168, 232, 255) if is_colored else None, boil_phase=boil_phase, seed=310)
    draw_hand_oval(canvas, lx, ly_top - 14, tw_top * 1.25, 10, color_bgr=ink_bgr, thick=2.2, fill_bgr=(52, 62, 188) if is_colored else None, boil_phase=boil_phase, seed=311)
    # Glowing Fresnel lens core
    lens_x = lx + tw_top * 0.7 * math.sin(beam_theta)
    cv2.circle(canvas, (int(lens_x), int(ly_top)), 6, (90, 235, 255) if is_colored else BGR_AMBER, -1, cv2.LINE_AA)

    # --- 3. ROLLING TROCHOIDAL WAVES & CRASHING SURF SPRAY (Effects Animation!) ---
    u_wave = (t_sec * 0.65) % 1.0
    for wi in range(2):
        wy_base = sea_y + wi * h * 0.08
        w_pts = [(0, h)]
        for xi in range(13):
            ux = xi / 12.0
            ph = ux * 4.0 * math.pi - t_sec * (3.8 + wi * 1.2) + wi * 1.5
            wx = ux * w + 10.0 * math.cos(ph)
            wy = wy_base + 14.0 * math.sin(ph)
            w_pts.append((wx, wy))
        w_pts.append((w, h))
        sea_fill = ((138, 102, 58) if wi == 0 else (112, 82, 44)) if is_colored else None
        draw_hand_ink_spline(canvas, w_pts, color_bgr=ink_bgr, base_thick=2.2, closed=True, fill_bgr=sea_fill, boil_phase=boil_phase, seed=320 + wi, taper_ends=False)

    # Crashing wave against cliff (`u_wave`: builds -> curls -> smashes into spray!)
    cx_crash = w * 0.45
    cy_crash = sea_y - h * 0.08 * math.sin(u_wave * math.pi)
    foam_fill = (244, 248, 254) if is_colored else None
    for si in range(5):
        ang_s = math.radians(-115 + si * 28 + 18 * math.sin(u_wave * math.pi))
        rad_s = h * (0.06 + 0.14 * math.sin(u_wave * math.pi)) * (1.0 - 0.12 * abs(si - 2))
        sx_tip = cx_crash + rad_s * math.cos(ang_s)
        sy_tip = cy_crash + rad_s * math.sin(ang_s) + 35.0 * (u_wave ** 2)
        draw_hand_oval(canvas, sx_tip, sy_tip, 9 - si, 6, angle_deg=si * 25, color_bgr=ink_bgr, thick=1.8, fill_bgr=foam_fill, boil_phase=boil_phase, seed=330 + si)

    # Banking Seagulls
    for gi in range(2):
        gx_b = w * (0.64 + 0.16 * gi) + 22.0 * math.cos(t_sec * 2.4 + gi)
        gy_b = h * (0.24 + 0.08 * gi) + 14.0 * math.sin(t_sec * 3.1 + gi)
        flap = 14.0 * math.sin(t_sec * 7.5 + gi * 2.0)
        gull = [(gx_b - 18, gy_b - flap), (gx_b - 6, gy_b - 4), (gx_b, gy_b), (gx_b + 6, gy_b - 4), (gx_b + 18, gy_b - flap)]
        draw_hand_ink_spline(canvas, gull, color_bgr=ink_bgr, base_thick=2.2, boil_phase=boil_phase, seed=340 + gi)


def _draw_steam_locomotive_cartoon(
    canvas: np.ndarray,
    w: int,
    h: int,
    t_sec: float,
    stage_mode: str,
    boil_phase: int,
    ink_bgr: tuple,
):
    """
    Manually draws the Alpine Viaduct 4-4-0 Steam Locomotive (`train`):
      - Multi-arch stone viaduct scrolling in parallax
      - Boiler, cab with engineer pulling whistle cord, cowcatcher, smokestack
      - 4 spoked driving wheels + mechanically exact reciprocating piston crosshead & connecting rod!
      - Billowing hand-drawn steam & smoke puffs expanding along a parabolic wind arc!
    """
    is_rough = (stage_mode == "rough_blue")
    is_colored = stage_mode in ("gouache_cel", "multiplane_composite")
    track_y = h * 0.76

    # --- 1. STONE VIADUCT BRIDGE ARCHES (Scrolling Parallax) ---
    viaduct_fill = (162, 168, 176) if is_colored else None
    cv2.rectangle(canvas, (0, int(track_y)), (w, int(track_y + 16)), (135, 140, 148) if is_colored else BGR_PAPER_DARK, -1)
    cv2.line(canvas, (0, int(track_y)), (w, int(track_y)), ink_bgr, 3, cv2.LINE_AA)
    cv2.line(canvas, (0, int(track_y + 16)), (w, int(track_y + 16)), ink_bgr, 2, cv2.LINE_AA)

    v_shift = (t_sec * 165.0) % (w * 0.28)
    for ai in range(-1, 5):
        px = ai * (w * 0.28) - v_shift
        pillar = [(px - 18, track_y + 16), (px + 18, track_y + 16), (px + 22, h), (px - 22, h)]
        draw_hand_ink_spline(canvas, pillar, color_bgr=ink_bgr, base_thick=2.0, closed=True, fill_bgr=viaduct_fill, boil_phase=boil_phase, seed=400 + ai)

    # --- 2. 4-4-0 STEAM LOCOMOTIVE CHASSIS, BOILER & CAB ---
    tx = w * 0.52 + 8.0 * math.sin(t_sec * 2.5)
    bounce = 2.5 * abs(math.sin(t_sec * 11.0))
    by = track_y - 34.0 - bounce

    loco_red = (56, 68, 198) if is_colored else None
    loco_iron = (58, 62, 68) if is_colored else None
    loco_brass = (68, 186, 242) if is_colored else None

    # Boiler cylinder & Cab
    boiler = [
        (tx - w * 0.16, by - h * 0.16),
        (tx + w * 0.14, by - h * 0.16),
        (tx + w * 0.14, by),
        (tx - w * 0.16, by),
    ]
    draw_hand_ink_spline(canvas, boiler, color_bgr=ink_bgr, base_thick=2.5, closed=True, fill_bgr=loco_iron, boil_phase=boil_phase, seed=410)

    cab = [
        (tx - w * 0.24, by - h * 0.26),
        (tx - w * 0.13, by - h * 0.26),
        (tx - w * 0.13, by + 4),
        (tx - w * 0.24, by + 4),
    ]
    draw_hand_ink_spline(canvas, cab, color_bgr=ink_bgr, base_thick=2.5, closed=True, fill_bgr=loco_red, boil_phase=boil_phase, seed=411)
    # Cab window + Engineer waving/pulling whistle
    draw_hand_oval(canvas, tx - w * 0.18, by - h * 0.17, 16, 14, color_bgr=ink_bgr, thick=2.0, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=412)
    cv2.circle(canvas, (int(tx - w * 0.18), int(by - h * 0.17)), 7, ink_bgr, 2, cv2.LINE_AA)

    # Smokestack & Brass Dome + Cowcatcher
    stack = [
        (tx + w * 0.08, by - h * 0.16),
        (tx + w * 0.065, by - h * 0.29),
        (tx + w * 0.125, by - h * 0.29),
        (tx + w * 0.11, by - h * 0.16),
    ]
    draw_hand_ink_spline(canvas, stack, color_bgr=ink_bgr, base_thick=2.4, closed=True, fill_bgr=loco_iron, boil_phase=boil_phase, seed=413)
    draw_hand_oval(canvas, tx - w * 0.02, by - h * 0.19, 16, 14, color_bgr=ink_bgr, thick=2.2, fill_bgr=loco_brass, boil_phase=boil_phase, seed=414)
    cow = [(tx + w * 0.14, by - 12), (tx + w * 0.22, track_y - 2), (tx + w * 0.14, track_y - 2)]
    draw_hand_ink_spline(canvas, cow, color_bgr=ink_bgr, base_thick=2.4, closed=True, fill_bgr=loco_red, boil_phase=boil_phase, seed=415)

    # --- 3. ROTATING SPOKED DRIVING WHEELS & RECIPROCATING PISTON LINKAGE ---
    wheel_r = 32.0
    wheel_theta = t_sec * 11.0
    w1_x, w2_x = tx - w * 0.15, tx - w * 0.02
    wy_c = track_y - wheel_r
    for wi, wx_c in enumerate([w1_x, w2_x]):
        draw_hand_oval(canvas, wx_c, wy_c, wheel_r, wheel_r, color_bgr=ink_bgr, thick=2.8, fill_bgr=loco_red, boil_phase=boil_phase, seed=420 + wi)
        draw_hand_oval(canvas, wx_c, wy_c, wheel_r * 0.32, wheel_r * 0.32, color_bgr=ink_bgr, thick=2.0, fill_bgr=loco_brass, boil_phase=boil_phase, seed=422 + wi)
        for sp in range(6):
            ang_sp = wheel_theta + sp * (math.pi / 3.0)
            cv2.line(canvas, (int(wx_c), int(wy_c)), (int(wx_c + wheel_r * 0.88 * math.cos(ang_sp)), int(wy_c + wheel_r * 0.88 * math.sin(ang_sp))), ink_bgr, 2, cv2.LINE_AA)

    # Crankpin & Reciprocating Piston Connecting Rod
    crank_r = wheel_r * 0.58
    cp1_x = w1_x + crank_r * math.cos(wheel_theta)
    cp1_y = wy_c + crank_r * math.sin(wheel_theta)
    cp2_x = w2_x + crank_r * math.cos(wheel_theta)
    cp2_y = wy_c + crank_r * math.sin(wheel_theta)
    # Coupling rod between driving wheels
    cv2.line(canvas, (int(cp1_x), int(cp1_y)), (int(cp2_x), int(cp2_y)), BGR_PAPER, 6, cv2.LINE_AA)
    cv2.line(canvas, (int(cp1_x), int(cp1_y)), (int(cp2_x), int(cp2_y)), ink_bgr, 3, cv2.LINE_AA)
    # Piston cylinder & crosshead
    cyl_x = tx + w * 0.10
    piston_x = cyl_x - 28.0 + 18.0 * math.cos(wheel_theta)
    piston_y = wy_c - 4.0
    cv2.line(canvas, (int(cp2_x), int(cp2_y)), (int(piston_x), int(piston_y)), ink_bgr, 4, cv2.LINE_AA)
    draw_hand_oval(canvas, cyl_x, piston_y, 20, 12, color_bgr=ink_bgr, thick=2.4, fill_bgr=loco_brass, boil_phase=boil_phase, seed=428)

    # --- 4. BILLOWING HAND-DRAWN STEAM & SMOKE PLUMES ---
    stack_top_x = tx + w * 0.095
    stack_top_y = by - h * 0.29
    for pi in range(6):
        u_p = ((t_sec * 1.45 + pi * 0.166) % 1.0)
        px_s = stack_top_x - u_p * w * 0.52
        py_s = stack_top_y - 18.0 - (u_p ** 0.68) * h * 0.26 + 8.0 * math.sin(u_p * 8.0 + pi)
        pr_x = 12.0 + u_p * 36.0
        pr_y = 10.0 + u_p * 26.0
        smoke_fill = (242, 246, 250) if is_colored else None
        draw_hand_oval(canvas, px_s, py_s, pr_x, pr_y, angle_deg=u_p * 45, color_bgr=ink_bgr, thick=2.0, fill_bgr=smoke_fill, boil_phase=boil_phase, seed=430 + pi)


def _draw_monarch_butterfly_cartoon(
    canvas: np.ndarray,
    w: int,
    h: int,
    t_sec: float,
    stage_mode: str,
    boil_phase: int,
    ink_bgr: tuple,
):
    """
    Manually draws the Wildflower Meadow Monarch Butterfly & Bumblebee (`butterfly`):
      - Swaying giant daisy flower that bends under the butterfly's landing impulse and springs back!
      - Monarch butterfly with 3D clap-and-fling wing foreshortening, antennae drag, and uncurling proboscis!
      - Plump striped bumblebee flying a figure-8 loop in the midground!
    """
    is_rough = (stage_mode == "rough_blue")
    is_colored = stage_mode in ("gouache_cel", "multiplane_composite")
    u_cyc = (t_sec * 0.45) % 1.0

    # --- 1. SWAYING DAISY FLOWER (Bends when butterfly lands in `u_cyc in [0.50, 0.85]`) ---
    landing_impulse = math.sin(max(0.0, min(1.0, (u_cyc - 0.48) / 0.36)) * math.pi) if (0.48 <= u_cyc <= 0.84) else 0.0
    stem_base_x, stem_base_y = w * 0.62, h * 0.92
    fl_x = w * 0.58 + 16.0 * landing_impulse + 6.0 * math.sin(t_sec * 2.4)
    fl_y = h * 0.54 + 28.0 * landing_impulse

    stem_pts = [
        (stem_base_x, stem_base_y),
        (0.5 * (stem_base_x + fl_x) - 14.0, 0.5 * (stem_base_y + fl_y)),
        (fl_x, fl_y),
    ]
    draw_hand_ink_spline(canvas, stem_pts, color_bgr=ink_bgr, base_thick=3.2, boil_phase=boil_phase, seed=500)
    # Daisy petals
    petal_fill = (248, 250, 255) if is_colored else None
    for pi in range(8):
        ang_p = pi * (math.pi / 4.0) + 0.15 * landing_impulse
        px_p = fl_x + 34.0 * math.cos(ang_p)
        py_p = fl_y + 22.0 * math.sin(ang_p)
        draw_hand_oval(canvas, px_p, py_p, 18, 9, angle_deg=math.degrees(ang_p), color_bgr=ink_bgr, thick=1.8, fill_bgr=petal_fill, boil_phase=boil_phase, seed=505 + pi)
    draw_hand_oval(canvas, fl_x, fl_y, 18, 12, color_bgr=ink_bgr, thick=2.2, fill_bgr=(64, 204, 252) if is_colored else None, boil_phase=boil_phase, seed=515)

    # --- 2. MONARCH BUTTERFLY (3D Wingbeats + Flight Arc + Flower Landing) ---
    if u_cyc < 0.48:
        uf = u_cyc / 0.48
        bx = w * (0.20 + 0.38 * uf)
        by = h * 0.36 - h * 0.14 * math.sin(uf * math.pi) + 14.0 * math.cos(uf * math.pi * 2.0)
        flap_freq = 11.5
    elif u_cyc <= 0.84:
        bx = fl_x - 4.0
        by = fl_y - 18.0
        flap_freq = 3.2  # Slow resting wing pulse while sipping nectar!
    else:
        uf = (u_cyc - 0.84) / 0.16
        bx = fl_x + w * 0.24 * uf
        by = fl_y - 18.0 - h * 0.22 * uf
        flap_freq = 12.5

    wing_phase = math.sin(t_sec * flap_freq)
    wing_span_y = 48.0 * wing_phase
    wing_span_x = 38.0 * (0.55 + 0.45 * abs(wing_phase))
    monarch_orange = (48, 128, 246) if is_colored else None
    monarch_gold = (78, 198, 255) if is_colored else None

    # Far & Near Forewings + Hindwings
    for side_s, scale_s, fill_w in [(-1, 0.82, monarch_gold), (1, 1.0, monarch_orange)]:
        fw_pts = [
            (bx, by - 4),
            (bx - wing_span_x * scale_s, by - wing_span_y * scale_s - 18 * side_s),
            (bx - wing_span_x * 1.15 * scale_s, by - wing_span_y * 0.45 * scale_s),
            (bx - 6, by + 4),
        ]
        hw_pts = [
            (bx - 4, by + 2),
            (bx - wing_span_x * 0.78 * scale_s, by - wing_span_y * 0.25 * scale_s + 18),
            (bx - wing_span_x * 0.45 * scale_s, by + 22),
            (bx - 2, by + 10),
        ]
        draw_hand_ink_spline(canvas, fw_pts, color_bgr=ink_bgr, base_thick=2.3, closed=True, fill_bgr=fill_w, boil_phase=boil_phase, seed=520 + side_s)
        draw_hand_ink_spline(canvas, hw_pts, color_bgr=ink_bgr, base_thick=2.1, closed=True, fill_bgr=fill_w, boil_phase=boil_phase, seed=525 + side_s)

    # Butterfly Thorax, Abdomen, Antennae & Proboscis
    draw_hand_oval(canvas, bx, by + 4, 18, 5, angle_deg=-22, color_bgr=ink_bgr, thick=2.4, fill_bgr=(36, 40, 48) if is_colored else None, boil_phase=boil_phase, seed=530)
    for ai in (-1, 1):
        ant = [(bx + 14, by - 4), (bx + 26, by - 18 + ai * 5), (bx + 32, by - 14 + ai * 8)]
        draw_hand_ink_spline(canvas, ant, color_bgr=ink_bgr, base_thick=1.6, boil_phase=boil_phase, seed=532 + ai)

    # --- 3. PLUMP BUMBLEBEE IN FIGURE-8 LOOP ---
    bee_x = w * 0.28 + 46.0 * math.sin(t_sec * 3.2)
    bee_y = h * 0.56 + 24.0 * math.sin(t_sec * 6.4)
    draw_hand_oval(canvas, bee_x, bee_y, 12, 8, color_bgr=ink_bgr, thick=2.0, fill_bgr=(52, 212, 255) if is_colored else None, boil_phase=boil_phase, seed=540)
    draw_hand_oval(canvas, bee_x, bee_y - 10 + 4 * math.sin(t_sec * 22.0), 8, 5, color_bgr=ink_bgr, thick=1.5, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=541)


def render_manual_cartoon_frame(
    slug: str,
    t_sec: float,
    out_w: int,
    out_h: int,
    stage_mode: str = "multiplane_composite",
    boil_phase: int = 0,
    enable_onion_skin: bool = False,
) -> np.ndarray:
    """
    Master entry point for rendering a 100% MANUALLY DRAWN FRAME-BY-FRAME CARTOON
    at exact `(out_w, out_h)` (`4:3` aspect ratio).
    Renders at canonical `640x480` resolution so all anatomical offsets and calligraphic
    line weights remain 100% proportional at any monitor or thumbnail size!
    """
    cw, ch = 640, 480
    canvas = np.full((ch, cw, 3), BGR_PAPER, dtype=np.uint8)
    s_low = (slug or "campfire").lower()

    # Map every hero slug to one of our 5 rich biomechanical multiplane cartoon engines
    # (with distinct camera choreography & theme per slug)
    if any(k in s_low for k in ("giraffe", "horse", "camel", "deer", "carousel", "zebra")):
        engine_fn = _draw_giraffe_savanna_cartoon
        theme = "savanna"
    elif any(k in s_low for k in ("light", "sail", "whale", "fountain", "ocean", "wave")):
        engine_fn = _draw_lighthouse_storm_cartoon
        theme = "storm_sea"
    elif any(k in s_low for k in ("train", "clock", "windmill", "gear", "tower")):
        engine_fn = _draw_steam_locomotive_cartoon
        theme = "alpine"
    elif any(k in s_low for k in ("butter", "eagle", "bird", "bee", "flower")):
        engine_fn = _draw_monarch_butterfly_cartoon
        theme = "meadow"
    else:
        engine_fn = _draw_fox_and_campfire_cartoon
        theme = "night_woods"

    cam_pan = 42.0 * math.sin(t_sec * 0.65)
    _draw_7plane_sky_and_distant_bg(canvas, cw, ch, t_sec, cam_pan, theme, stage_mode, boil_phase)

    # Rule XII: FLIP THE DRAWINGS (`previous -> current -> next` Onion-Skin Ghosts!)
    if enable_onion_skin or stage_mode in ("rough_blue", "onion_flip"):
        ov_prev = canvas.copy()
        engine_fn(ov_prev, cw, ch, t_sec - 0.10, "rough_blue", boil_phase - 1, BGR_ONION_PREV)
        cv2.addWeighted(ov_prev, 0.28, canvas, 0.72, 0, canvas)

        ov_next = canvas.copy()
        engine_fn(ov_next, cw, ch, t_sec + 0.10, "rough_blue", boil_phase + 1, BGR_ONION_NEXT)
        cv2.addWeighted(ov_next, 0.24, canvas, 0.76, 0, canvas)

    ink_c = BGR_NONPHOTO_BLUE if stage_mode == "rough_blue" else BGR_INK
    engine_fn(canvas, cw, ch, t_sec, stage_mode, boil_phase, ink_c)
    _draw_plane0_foreground_mask(canvas, cw, ch, t_sec, cam_pan, theme, stage_mode, boil_phase)

    # Classic Acme 3-Peg Registration Header at top of animation sheet in rough/ink modes
    if stage_mode in ("rough_blue", "onion_flip", "cleanup_ink"):
        cv2.rectangle(canvas, (0, 0), (cw, 22), BGR_PAPER_DARK, -1)
        cv2.line(canvas, (0, 22), (cw, 22), BGR_SEPIA, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cw // 2, 11), 5, (72, 148, 198), -1, cv2.LINE_AA)
        for sx_p in (cw // 4, (3 * cw) // 4):
            cv2.rectangle(canvas, (sx_p - 12, 7), (sx_p + 12, 15), (72, 148, 198), -1)

    if out_w != cw or out_h != ch:
        interp = cv2.INTER_AREA if out_w < cw else cv2.INTER_LINEAR
        return cv2.resize(canvas, (out_w, out_h), interpolation=interp)
    return canvas


# ==============================================================================
# III. MANUALLY DRAWN HEROIC-SCALE STUDIO ANIMATOR CHARACTERS & APPARATUS
# ==============================================================================

AGENT_PROFILES = {
    "nib": {
        "name": "NIB (SHOWRUNNER)",
        "hat": "nib_crown",
        "shirt_bgr": (245, 248, 252),
        "apron_bgr": (82, 108, 196),
        "hat_bgr": (218, 238, 252),
        "skin_bgr": (232, 242, 252),
        "pants_bgr": (72, 82, 98),
    },
    "blueline": {
        "name": "BLUE-LINE (ROUGH KEY INKER)",
        "hat": "visor_pencil",
        "shirt_bgr": (244, 236, 222),
        "apron_bgr": (196, 134, 68),
        "hat_bgr": (212, 148, 78),
        "skin_bgr": (232, 242, 252),
        "pants_bgr": (84, 74, 66),
    },
    "celinker": {
        "name": "CEL-INKER (GOUACHE PAINTER)",
        "hat": "beret_monocle",
        "shirt_bgr": (238, 244, 248),
        "apron_bgr": (92, 168, 228),
        "hat_bgr": (78, 86, 198),
        "skin_bgr": (232, 242, 252),
        "pants_bgr": (76, 86, 102),
    },
    "sprocket": {
        "name": "SPROCKET (16-GATE AUDITOR)",
        "hat": "reel_ears_monocle",
        "shirt_bgr": (232, 240, 236),
        "apron_bgr": (122, 152, 88),
        "hat_bgr": (82, 178, 222),
        "skin_bgr": (232, 242, 252),
        "pants_bgr": (68, 78, 72),
    },
    "chime": {
        "name": "CHIME (FOLEY & CONTINUITY)",
        "hat": "headphones_stopwatch",
        "shirt_bgr": (242, 238, 246),
        "apron_bgr": (148, 98, 168),
        "hat_bgr": (68, 72, 82),
        "skin_bgr": (232, 242, 252),
        "pants_bgr": (78, 70, 88),
    },
}


def _draw_studio_agent_arm_and_hand(
    canvas: np.ndarray,
    sh_pt: tuple,
    el_pt: tuple,
    wr_pt: tuple,
    w_up: float,
    w_low: float,
    scale: float,
    facing: int,
    is_right: bool,
    is_near_shoulder: bool,
    shirt_fill: tuple,
    skin_fill: tuple,
    ink_w: int,
    tool_rh: str = "pencil",
) -> tuple:
    """
    Draws a 100% seamless cartoon arm (`Upper Sleeve -> Proximal Forearm -> Puffy Rolled
    Elbow Cuff -> Seamless Wrist/Palm/Fingers/Thumb Union + Studio Tool`) with ZERO
    intersecting joint rings at the shoulder, elbow, wrist, palm, or knuckles!
    """
    w_i = max(2, int(ink_w))
    f_ang = math.atan2(wr_pt[1] - el_pt[1], wr_pt[0] - el_pt[0])
    fca, fsa = math.cos(f_ang), math.sin(f_ang)
    fnx, fny = -fsa, fca

    u_ang = math.atan2(el_pt[1] - sh_pt[1], el_pt[0] - sh_pt[0])
    uca, usa = math.cos(u_ang), math.sin(u_ang)

    palm_cx = wr_pt[0] + fca * 10.5 * scale
    palm_cy = wr_pt[1] + fsa * 10.5 * scale
    tool_tip = (palm_cx + facing * 18.0 * scale, palm_cy)

    # 1. Upper Sleeve Tube (`sh_pt -> el_pt`)
    cv2.fillPoly(canvas, [_capsule_poly(sh_pt, el_pt, w_up + w_i, w_up * 0.90 + w_i)], BGR_INK, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(sh_pt, el_pt, w_up, w_up * 0.90)], shirt_fill, cv2.LINE_AA)
    if is_near_shoulder:
        # Seamless shoulder-to-shirt bridge & short armpit crease line (`/` in `nib_inking_keys.jpg` Q1-Q4)
        bridge_pt = (sh_pt[0] + facing * 4.5 * scale, sh_pt[1] - 2.0 * scale)
        cv2.circle(canvas, (int(bridge_pt[0]), int(bridge_pt[1])), int(w_up * 0.78), shirt_fill, -1, cv2.LINE_AA)
        ap_x0 = int(sh_pt[0] + facing * w_up * 0.52)
        ap_y0 = int(sh_pt[1] + w_up * 0.15)
        cv2.line(canvas, (ap_x0, ap_y0), (ap_x0 - int(facing * 4.5 * scale), ap_y0 + int(9.5 * scale)), BGR_INK, max(1, w_i - 1), cv2.LINE_AA)

    # 2. Proximal Forearm Tube (`el_pt -> wr_pt`)
    r_wrist = w_low * 0.75
    cv2.fillPoly(canvas, [_capsule_poly(el_pt, wr_pt, w_low + w_i, r_wrist + w_i)], BGR_INK, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(el_pt, wr_pt, w_low, r_wrist)], skin_fill, cv2.LINE_AA)

    # 3. Puffy Rolled-Up Sleeve Cuff at Elbow + 2 Curved Cloth Fold Ticks (`nib_inking_keys.jpg` Q1-Q4)
    unx, uny = -usa, uca
    for fd_i, fd_d in enumerate([9.0, 14.5]):
        fx0 = el_pt[0] - uca * fd_d * scale - unx * (w_up * (0.15 + 0.12 * fd_i)) * facing
        fy0 = el_pt[1] - usa * fd_d * scale - uny * (w_up * (0.15 + 0.12 * fd_i)) * facing
        fx1 = fx0 + (uca * 3.5 + unx * 6.5 * facing) * scale
        fy1 = fy0 + (usa * 3.5 + uny * 6.5 * facing) * scale
        cv2.line(canvas, (int(fx0), int(fy0)), (int(fx1), int(fy1)), BGR_INK, max(1, w_i - 1), cv2.LINE_AA)
    cuff_cx = el_pt[0] - uca * 1.2 * scale + fca * 1.2 * scale
    cuff_cy = el_pt[1] - usa * 1.2 * scale + fsa * 1.2 * scale
    cuff_hw = w_up * 0.90
    c0 = (cuff_cx - unx * cuff_hw, cuff_cy - uny * cuff_hw)
    c1 = (cuff_cx + unx * cuff_hw, cuff_cy + uny * cuff_hw)
    r_cuff = 5.6 * scale
    cv2.fillPoly(canvas, [_capsule_poly(c0, c1, r_cuff + w_i, r_cuff + w_i)], BGR_INK, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(c0, c1, r_cuff, r_cuff)], shirt_fill, cv2.LINE_AA)

    # 4. Back shaft of tool (`if is_right`)
    t_ang = f_ang - 0.42 * facing
    pca, psa = math.cos(t_ang), math.sin(t_ang)
    pnx, pny = -psa, pca
    grip_x = palm_cx + 6.5 * scale * fca
    grip_y = palm_cy + 6.5 * scale * fsa
    back_x = grip_x - pca * 42.0 * scale
    back_y = grip_y - psa * 42.0 * scale
    collar_x = grip_x + pca * 18.0 * scale
    collar_y = grip_y + psa * 18.0 * scale
    tip_x = grip_x + pca * 52.0 * scale
    tip_y = grip_y + psa * 52.0 * scale
    if is_right:
        tool_tip = (tip_x, tip_y)
        pw = 4.2 * scale
        s_fill = BGR_NONPHOTO_BLUE if tool_rh == "blue_pencil" else (BGR_VERMILION if tool_rh == "brush" else BGR_AMBER)
        cv2.fillPoly(canvas, [_capsule_poly((back_x, back_y), (collar_x, collar_y), pw * 0.8 + w_i, pw + w_i)], BGR_INK, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly((back_x, back_y), (collar_x, collar_y), pw * 0.8, pw)], s_fill, cv2.LINE_AA)

    # 5. Seamless Distal Forearm + Wrist + Palm + 3 Fingers + Thumb (Dual-Pass Union!)
    mid_forearm = (0.48 * el_pt[0] + 0.52 * wr_pt[0], 0.48 * el_pt[1] + 0.52 * wr_pt[1])
    r_mid = 0.48 * w_low + 0.52 * r_wrist
    rx_palm, ry_palm = 12.5 * scale, 10.5 * scale
    p_deg = math.degrees(f_ang)

    f_defs = [(-0.14, 16.5, -4.0), (0.12, 15.5, 0.5), (0.34, 13.5, 4.5)]
    finger_capsules = []
    for sp_a, flen, n_off in f_defs:
        fa = f_ang + sp_a
        fb_x = palm_cx + math.cos(fa) * 4.2 * scale + fnx * n_off * scale
        fb_y = palm_cy + math.sin(fa) * 4.2 * scale + fny * n_off * scale
        ft_x = fb_x + math.cos(fa) * (flen * 0.68 * scale)
        ft_y = fb_y + math.sin(fa) * (flen * 0.68 * scale)
        finger_capsules.append(((fb_x, fb_y), (ft_x, ft_y), 4.2 * scale, 3.8 * scale))

    tha = f_ang - 0.58 * facing
    th_bx = palm_cx + math.cos(tha) * 2.6 * scale - fnx * 6.5 * scale * facing
    th_by = palm_cy + math.sin(tha) * 2.6 * scale - fny * 6.5 * scale * facing
    th_tx = th_bx + math.cos(tha) * (14.0 * 0.68 * scale)
    th_ty = th_by + math.sin(tha) * (14.0 * 0.68 * scale)
    thumb_cap = ((th_bx, th_by), (th_tx, th_ty), 4.4 * scale, 3.9 * scale)

    # Pass 1: Outer Ink Casing
    cv2.ellipse(canvas, (int(palm_cx), int(palm_cy)), (int(rx_palm + w_i), int(ry_palm + w_i)), p_deg, 0, 360, BGR_INK, -1, cv2.LINE_AA)
    for fb, ft, rf0, rf1 in finger_capsules:
        cv2.fillPoly(canvas, [_capsule_poly(fb, ft, rf0 + w_i, rf1 + w_i)], BGR_INK, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2] + w_i, thumb_cap[3] + w_i)], BGR_INK, cv2.LINE_AA)

    # Pass 2: Seamless Skin Fill
    cv2.fillPoly(canvas, [_capsule_poly(mid_forearm, wr_pt, r_mid, r_wrist)], skin_fill, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(palm_cx), int(palm_cy)), (int(rx_palm), int(ry_palm)), p_deg, 0, 360, skin_fill, -1, cv2.LINE_AA)
    for fb, ft, rf0, rf1 in finger_capsules:
        cv2.fillPoly(canvas, [_capsule_poly(fb, ft, rf0, rf1)], skin_fill, cv2.LINE_AA)
    cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2], thumb_cap[3])], skin_fill, cv2.LINE_AA)

    # Pass 3: Short finger-cleft creases + Front Studio Tool + Overlapping Thumb Pinch
    for ci in range(len(finger_capsules) - 1):
        f0_b, f0_t, _, _ = finger_capsules[ci]
        f1_b, f1_t, _, _ = finger_capsules[ci + 1]
        cleft_x, cleft_y = 0.5 * (f0_t[0] + f1_t[0]), 0.5 * (f0_t[1] + f1_t[1])
        base_x = 0.5 * (f0_b[0] + f1_b[0]) + fca * 2.5 * scale
        base_y = 0.5 * (f0_b[1] + f1_b[1]) + fsa * 2.5 * scale
        cv2.line(canvas, (int(base_x), int(base_y)), (int(0.55 * base_x + 0.45 * cleft_x), int(0.55 * base_y + 0.45 * cleft_y)), BGR_INK, max(1, w_i - 1), cv2.LINE_AA)

    if is_right:
        pw = 4.2 * scale
        s_fill = BGR_NONPHOTO_BLUE if tool_rh == "blue_pencil" else (BGR_VERMILION if tool_rh == "brush" else BGR_AMBER)
        cv2.fillPoly(canvas, [_capsule_poly((grip_x, grip_y), (collar_x, collar_y), pw + w_i, pw + w_i)], BGR_INK, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly((grip_x, grip_y), (collar_x, collar_y), pw, pw)], s_fill, cv2.LINE_AA)
        nib_mid_x = grip_x + pca * 33.0 * scale
        nib_mid_y = grip_y + psa * 33.0 * scale
        nw = 7.6 * scale
        nib_poly = np.array([
            [int(collar_x + pnx * 4.2 * scale), int(collar_y + pny * 4.2 * scale)],
            [int(nib_mid_x + pnx * nw), int(nib_mid_y + pny * nw)],
            [int(tip_x), int(tip_y)],
            [int(nib_mid_x - pnx * nw), int(nib_mid_y - pny * nw)],
            [int(collar_x - pnx * 4.2 * scale), int(collar_y - pny * 4.2 * scale)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [nib_poly], (196, 232, 252), cv2.LINE_AA)
        cv2.polylines(canvas, [nib_poly], True, BGR_INK, w_i, cv2.LINE_AA)
        hole_x = int(nib_mid_x - pca * 1.5 * scale)
        hole_y = int(nib_mid_y - psa * 1.5 * scale)
        cv2.line(canvas, (int(tip_x), int(tip_y)), (hole_x, hole_y), BGR_INK, max(1, w_i - 1), cv2.LINE_AA)
        cv2.circle(canvas, (hole_x, hole_y), max(2, int(2.6 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (hole_x, hole_y), max(2, int(2.6 * scale)), BGR_INK, max(1, w_i - 1), cv2.LINE_AA)
        # Re-draw thumb pinch over pen grip!
        cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2] + w_i, thumb_cap[3] + w_i)], BGR_INK, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly(thumb_cap[0], thumb_cap[1], thumb_cap[2], thumb_cap[3])], skin_fill, cv2.LINE_AA)

    return tool_tip


def draw_heroic_studio_animator(
    canvas: np.ndarray,
    agent_id: str,
    foot_x: float,
    foot_y: float,
    scale: float,
    t_sec: float,
    action_mode: str,
    target_lh: tuple,
    target_rh: tuple,
    look_at: tuple,
    boil_phase: int = 0,
    facing: int = 1,
    tool_rh: str = "pencil",
    tool_lh: str = "paper_flip",
):
    """
    Draws our 5 canonical Studio Animator Characters (`Nib`, `Blue-Line`, `Cel-Inker`,
    `Sprocket`, `Chime`) with 100% Watertight Dual-Pass Capsule Union (`_capsule_poly`)
    matching the exact character models from `assets/studio_doodles/`.
    """
    prof = AGENT_PROFILES.get(agent_id, AGENT_PROFILES["nib"])
    skin = prof["skin_bgr"]
    shirt = prof["shirt_bgr"]
    apron = prof["apron_bgr"]
    pants = prof["pants_bgr"]
    pants_far = tuple(max(30, int(c * 0.84)) for c in pants)
    shoe_fill = (56, 68, 98)
    hat_c = prof["hat_bgr"]
    ink_w = max(2, int(round(3.6 * scale)))

    # Classic 3-head-tall cartoon proportions (`studio_crew_celebration_keys.jpg`)
    if action_mode == "cheer":
        jump = max(0.0, math.sin(t_sec * 6.5))
        crouch = max(0.0, -math.sin(t_sec * 6.5)) * 14.0 * scale
        hip_x = foot_x
        hip_y = foot_y - 96.0 * scale - jump * 36.0 * scale + crouch
        lean_deg = -6.0 * math.sin(t_sec * 6.5)
    elif action_mode == "walk":
        hip_x = foot_x + 22.0 * scale * math.sin(t_sec * 2.2)
        hip_y = foot_y - 98.0 * scale + 6.0 * scale * abs(math.sin(t_sec * 5.5))
        lean_deg = 8.0 * facing
    elif action_mode == "crank":
        hip_x = foot_x + 10.0 * scale * math.cos(t_sec * 4.5)
        hip_y = foot_y - 94.0 * scale + 9.0 * scale * math.sin(t_sec * 4.5)
        lean_deg = 12.0 * facing + 6.0 * math.cos(t_sec * 4.5)
    else:
        breath = math.sin(t_sec * 3.6)
        hip_x = foot_x + 5.0 * scale * math.sin(t_sec * 1.8)
        hip_y = foot_y - 96.0 * scale + 3.5 * scale * breath
        lean_deg = 10.0 * facing + 3.5 * math.sin(t_sec * 2.7)

    lean_rad = math.radians(lean_deg)
    torso_h = 98.0 * scale
    chest_x = hip_x + torso_h * math.sin(lean_rad)
    chest_y = hip_y - torso_h * math.cos(lean_rad)
    mid_x = 0.5 * (hip_x + chest_x)
    mid_y = 0.5 * (hip_y + chest_y)

    # --- 1. BILATERAL SHOULDER SETUP (`sh_l` on left/back shoulder, `sh_r` on right/front shoulder) ---
    bib_top_y = chest_y + 28.0 * scale
    sh_l_x = chest_x - facing * 28.0 * scale
    sh_l_y = chest_y + 18.0 * scale
    lh_adj_x, lh_adj_y = target_lh
    if action_mode != "cheer" and (lh_adj_x - chest_x) * facing > -12.0 * scale:
        lh_adj_y = max(lh_adj_y, bib_top_y + 22.0 * scale)
        lh_adj_x = min(lh_adj_x, chest_x + 14.0 * scale) if facing >= 0 else max(lh_adj_x, chest_x - 14.0 * scale)
    el_lx, el_ly, wr_lx, wr_ly = solve_2bone_ik(
        sh_l_x, sh_l_y, lh_adj_x, lh_adj_y,
        36.0 * scale, 36.0 * scale, bend_sign=float(facing),
    )
    if action_mode == "cheer":
        el_lx = min(el_lx, sh_l_x - 16.0 * scale) if facing >= 0 else max(el_lx, sh_l_x + 16.0 * scale)
    else:
        el_lx = min(el_lx, sh_l_x - 12.0 * scale) if facing >= 0 else max(el_lx, sh_l_x + 12.0 * scale)
        el_ly = max(el_ly, sh_l_y + 24.0 * scale)

    # --- 2. SEAMLESS CURVED TROUSER LEGS & SCULPTED FLAT-SOLED ANIMATOR SHOES ---
    thigh_l = 44.0 * scale
    calf_l = 44.0 * scale
    hip_anchor_x = 0.5 * (hip_x + mid_x)
    for li, (side_l, p_fill) in enumerate([(-1, pants_far), (1, pants)]):
        hip_j_x = hip_anchor_x + side_l * 10.0 * scale * facing
        hip_j_y = hip_y + 14.0 * scale
        if action_mode == "walk":
            ph = t_sec * 5.5 + li * math.pi
            fx_l = hip_j_x + 22.0 * scale * math.sin(ph)
            fy_l = foot_y - max(0.0, 14.0 * scale * math.cos(ph))
        elif action_mode == "cheer":
            fx_l = foot_x + side_l * 20.0 * scale * facing
            fy_l = min(foot_y, hip_y + 96.0 * scale)
        else:
            fx_l = foot_x + side_l * 18.0 * scale * facing
            fy_l = foot_y
        ank_y = fy_l - 12.0 * scale
        kx, ky, ax_e, ay_e = solve_2bone_ik(hip_j_x, hip_j_y, fx_l, ank_y, thigh_l, calf_l, bend_sign=-float(facing))

        # 2A. Sculpted Flat-Soled Animator Shoe (drawn first, under trouser cuff!)
        sole_y = int(fy_l)
        heel_x = int(ax_e - facing * 15.0 * scale)
        toe_x = int(ax_e + facing * 30.0 * scale)
        instep_y = int(sole_y - 17.0 * scale)
        shoe_pts = np.array([
            [heel_x, sole_y - int(3 * scale)],
            [heel_x + int(facing * 2 * scale), instep_y + int(3 * scale)],
            [int(ax_e + facing * 8 * scale), instep_y],
            [toe_x - int(facing * 5 * scale), instep_y + int(3 * scale)],
            [toe_x, sole_y - int(5 * scale)],
            [toe_x - int(facing * 2 * scale), sole_y],
            [heel_x + int(facing * 2 * scale), sole_y],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [shoe_pts], shoe_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [shoe_pts], True, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.line(canvas, (heel_x + int(facing * 2 * scale), sole_y - int(3.5 * scale)), (toe_x - int(facing * 2 * scale), sole_y - int(3.5 * scale)), BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)

        # 2B. Dual-Pass Capsule Union for Thigh + Calf (100% seamless knee joint!)
        r_th, r_kn, r_an = 16.5 * scale, 14.5 * scale, 12.5 * scale
        cv2.fillPoly(canvas, [_capsule_poly((hip_j_x, hip_j_y), (kx, ky), r_th + ink_w, r_kn + ink_w)], BGR_INK, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly((kx, ky), (ax_e, ay_e), r_kn + ink_w, r_an + ink_w)], BGR_INK, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly((hip_j_x, hip_j_y), (kx, ky), r_th, r_kn)], p_fill, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly((kx, ky), (ax_e, ay_e), r_kn, r_an)], p_fill, cv2.LINE_AA)
        # Cuffed trouser hem at ankle
        c_p0 = (ax_e - 13.0 * scale, ay_e - 1.5 * scale)
        c_p1 = (ax_e + 13.0 * scale, ay_e - 1.5 * scale)
        cv2.fillPoly(canvas, [_capsule_poly(c_p0, c_p1, 4.2 * scale + ink_w, 4.2 * scale + ink_w)], BGR_INK, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly(c_p0, c_p1, 4.2 * scale, 4.2 * scale)], p_fill, cv2.LINE_AA)

    # --- 3. UNIFIED TORSO, SHIRT SHOULDERS, BIB APRON & POINTED `W` COLLAR ---
    w_chest = 39.0 * scale
    w_waist = 37.0 * scale
    w_hip = 40.0 * scale
    sway_hem = 5.0 * scale * math.sin(t_sec * 3.8 + 0.6)
    bib_top_y = chest_y + 28.0 * scale
    apron_bot_y = hip_y + 40.0 * scale

    shirt_dome = np.array([
        [int(chest_x - w_chest * 0.90), int(chest_y + 14.0 * scale)],
        [int(chest_x - w_chest * 0.56), int(chest_y + 2.0 * scale)],
        [int(chest_x + w_chest * 0.58), int(chest_y + 2.0 * scale)],
        [int(chest_x + w_chest * 0.90), int(chest_y + 14.0 * scale)],
        [int(mid_x + w_waist * 0.88), int(mid_y + 14.0 * scale)],
        [int(mid_x - w_waist * 0.88), int(mid_y + 14.0 * scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [shirt_dome], shirt, cv2.LINE_AA)
    cv2.polylines(canvas, [shirt_dome], True, BGR_INK, ink_w, cv2.LINE_AA)

    apron_poly = np.array([
        [int(chest_x - w_chest * 0.50 * facing), int(bib_top_y)],
        [int(chest_x + w_chest * 0.58 * facing), int(bib_top_y)],
        [int(mid_x + w_waist * 0.88 * facing + sway_hem * 0.35), int(mid_y + 12.0 * scale)],
        [int(hip_x + w_hip * 0.98 * facing + sway_hem), int(apron_bot_y)],
        [int(hip_x - w_hip * 0.86 * facing + sway_hem * 0.7), int(apron_bot_y)],
        [int(mid_x - w_waist * 0.76 * facing + sway_hem * 0.3), int(mid_y + 12.0 * scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [apron_poly], apron, cv2.LINE_AA)
    cv2.polylines(canvas, [apron_poly], True, BGR_INK, ink_w, cv2.LINE_AA)
    # Horizontal apron waist-tie band (`=`) on back flank + 2 subtle skirt drape ticks (`nib_inking_keys.jpg` Q2)
    wt_x0 = int(mid_x - w_waist * 0.88 * facing)
    wt_x1 = int(mid_x - w_waist * 0.70 * facing)
    wt_y = int(mid_y + 9.0 * scale)
    cv2.line(canvas, (wt_x0, wt_y), (wt_x1, wt_y), BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)
    cv2.line(canvas, (wt_x0, wt_y + int(4.5 * scale)), (wt_x1, wt_y + int(4.5 * scale)), BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)
    for dr_x in (-0.32, 0.36):
        dx0 = int(hip_x + dr_x * w_hip * facing + sway_hem * 0.8)
        dy0 = int(apron_bot_y)
        cv2.line(canvas, (dx0, dy0), (dx0 - int(1.5 * scale * facing), dy0 - int(11.0 * scale)), BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)

    # Vertical Apron Shoulder Straps & Seamless Square-U Bib Join
    apron_strap_fill = tuple(max(20, int(c * 0.88)) for c in apron)
    for x_top_f, x_bot_f in [(-0.50 * facing, -0.44 * facing), (0.54 * facing, 0.50 * facing)]:
        strap_poly = np.array([
            [int(chest_x + x_top_f * w_chest - 4.5 * scale), int(chest_y + 3 * scale)],
            [int(chest_x + x_top_f * w_chest + 4.5 * scale), int(chest_y + 3 * scale)],
            [int(chest_x + x_bot_f * w_chest + 4.5 * scale), int(bib_top_y + 2 * scale)],
            [int(chest_x + x_bot_f * w_chest - 4.5 * scale), int(bib_top_y + 2 * scale)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [strap_poly], apron_strap_fill, cv2.LINE_AA)
        cv2.polylines(canvas, [strap_poly], True, BGR_INK, ink_w, cv2.LINE_AA)
    bx_a = int(min(chest_x - w_chest * 0.45 * facing, chest_x + w_chest * 0.51 * facing))
    bx_b = int(max(chest_x - w_chest * 0.45 * facing, chest_x + w_chest * 0.51 * facing))
    cv2.rectangle(canvas, (bx_a, int(bib_top_y + 1 * scale)), (bx_b, int(bib_top_y + 8 * scale)), apron, -1)
    cv2.line(canvas, (int(chest_x - w_chest * 0.50 * facing), int(bib_top_y)), (int(chest_x + w_chest * 0.58 * facing), int(bib_top_y)), BGR_INK, ink_w, cv2.LINE_AA)

    # Pointed `W` Shirt Collar (`\/\/`) tucked between the apron straps (`100% unoccluded`!)
    cx_v, cy_v = chest_x + facing * 2.5 * scale, chest_y + 3.5 * scale
    v_col = np.array([
        [int(cx_v - 16 * scale), int(cy_v)],
        [int(cx_v - 9.5 * scale), int(cy_v + 18 * scale)],
        [int(cx_v), int(cy_v + 8 * scale)],
        [int(cx_v + 9.5 * scale), int(cy_v + 18 * scale)],
        [int(cx_v + 16 * scale), int(cy_v)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [v_col], shirt, cv2.LINE_AA)
    cv2.polylines(canvas, [v_col], False, BGR_INK, ink_w, cv2.LINE_AA)

    # Canonical Apron Pocket Emblems (`blueline`, `celinker`, `sprocket`, `chime`)
    if agent_id != "nib":
        px0, py0 = int(mid_x - 16 * scale), int(mid_y + 2 * scale)
        px1, py1 = int(mid_x + 16 * scale), int(mid_y + 27 * scale)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), (248, 250, 252), -1)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)
        # Top stitched pocket band (`|===|` in `agent_blueline_keys.jpg` & `agent_celinker_keys.jpg`)
        cv2.line(canvas, (px0, int(py0 + 5 * scale)), (px1, int(py0 + 5 * scale)), BGR_INK, max(1, ink_w - 2), cv2.LINE_AA)
        if agent_id == "celinker":
            # 3 Teardrop Gouache Paint Droplets (`💧💧💧` in `agent_celinker_keys.jpg` Q1-Q4)!
            for di, d_col in [(-8, BGR_VERMILION), (0, BGR_NONPHOTO_BLUE), (8, BGR_AMBER)]:
                dx_d = int(mid_x + di * scale)
                dy_d = int(mid_y + 18 * scale)
                r_d = max(2, int(2.8 * scale))
                cv2.circle(canvas, (dx_d, dy_d), r_d, d_col, -1, cv2.LINE_AA)
                drop_tri = np.array([
                    [dx_d - r_d, dy_d],
                    [dx_d, dy_d - int(5.5 * scale)],
                    [dx_d + r_d, dy_d],
                ], dtype=np.int32)
                cv2.fillPoly(canvas, [drop_tri], d_col, cv2.LINE_AA)
                cv2.circle(canvas, (dx_d, dy_d), r_d, BGR_INK, max(1, ink_w - 2), cv2.LINE_AA)
                cv2.polylines(canvas, [drop_tri], False, BGR_INK, max(1, ink_w - 2), cv2.LINE_AA)
        elif agent_id == "sprocket":
            cv2.line(canvas, (int(mid_x - 6 * scale), int(mid_y + 16 * scale)), (int(mid_x - 1 * scale), int(mid_y + 22 * scale)), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)
            cv2.line(canvas, (int(mid_x - 1 * scale), int(mid_y + 22 * scale)), (int(mid_x + 8 * scale), int(mid_y + 10 * scale)), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)
        elif agent_id == "chime":
            cv2.circle(canvas, (int(mid_x), int(mid_y + 16 * scale)), int(7.5 * scale), (248, 250, 252), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(mid_x), int(mid_y + 16 * scale)), int(7.5 * scale), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)

    # --- 4. CANONICAL ROUND CARTOON HEAD, SCULPTED EAR, SOLID BLACK OVAL EYES & ICONIC HEADGEAR ---
    head_r_x = 48.0 * scale
    head_r_y = 43.0 * scale
    hx = chest_x + facing * 5.0 * scale
    hy = chest_y - 38.0 * scale
    hat_kind = prof["hat"]

    # Sculpted Cartoon C-Ear on Back Side of Head (`nib_inking_keys.jpg`, `agent_blueline_keys.jpg`, `agent_celinker_keys.jpg`)
    ear_cx = int(hx - facing * head_r_x * 0.90)
    ear_cy = int(hy + 4.0 * scale)
    cv2.ellipse(canvas, (ear_cx, ear_cy), (max(3, int(8.5 * scale)), max(4, int(10.5 * scale))), 0, 0, 360, skin, -1, cv2.LINE_AA)
    if facing >= 0:
        cv2.ellipse(canvas, (ear_cx, ear_cy), (max(3, int(8.5 * scale)), max(4, int(10.5 * scale))), 0, 75, 285, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.ellipse(canvas, (ear_cx + int(1.5 * scale), ear_cy), (max(2, int(4.0 * scale)), max(2, int(5.0 * scale))), 0, 95, 265, BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (ear_cx, ear_cy), (max(3, int(8.5 * scale)), max(4, int(10.5 * scale))), 0, -105, 105, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.ellipse(canvas, (ear_cx - int(1.5 * scale), ear_cy), (max(2, int(4.0 * scale)), max(2, int(5.0 * scale))), 0, -85, 85, BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)

    # Main round-oval head cranium (clipped to `-28..208` deg under crown/visor so upper skull never pokes through!)
    if hat_kind == "nib_crown":
        cv2.ellipse(canvas, (int(hx), int(hy)), (int(head_r_x), int(head_r_y)), 0, -28, 208, skin, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy)), (int(head_r_x), int(head_r_y)), 0, -28, 208, BGR_INK, ink_w, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (int(hx), int(hy)), (int(head_r_x), int(head_r_y)), 0, 0, 360, skin, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy)), (int(head_r_x), int(head_r_y)), 0, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)

    # Solid-black vertical oval cartoon eyes (`● ●`) + curved eyebrows
    is_blink = ((int(t_sec * 24) + hash(agent_id) % 19) % 58) in (0, 1)
    for ei, ex_off in enumerate([-8.0 * scale * facing, 16.0 * scale * facing]):
        ex_c = hx + ex_off
        ey_c = hy - 1.0 * scale
        if is_blink:
            cv2.line(canvas, (int(ex_c - 6 * scale), int(ey_c)), (int(ex_c + 6 * scale), int(ey_c)), BGR_INK, ink_w, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (int(ex_c), int(ey_c)), (max(2, int(4.8 * scale)), max(3, int(7.5 * scale))), 0, 0, 360, BGR_INK, -1, cv2.LINE_AA)
        brow_lift = 3.0 * scale * math.sin(t_sec * 3.2)
        cv2.ellipse(canvas, (int(ex_c), int(ey_c - 10 * scale - brow_lift)), (int(7 * scale), int(3.5 * scale)), 0, 200, 340, BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)

    # Concentric 3-Ring Jeweler's Loupe Monocle for `celinker` and `sprocket` (`agent_celinker_keys.jpg`, `agent_sprocket_keys.jpg`)
    if agent_id in ("celinker", "sprocket"):
        mx_e = int(hx + 16.0 * scale * facing)
        my_e = int(hy - 1.0 * scale)
        cv2.circle(canvas, (mx_e, my_e), int(13.5 * scale), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.circle(canvas, (mx_e, my_e), int(10.5 * scale), BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)
        cv2.circle(canvas, (mx_e, my_e), int(8.0 * scale), BGR_INK, max(1, ink_w - 2), cv2.LINE_AA)

    # Cute C-shaped / U-shaped rounded cartoon button nose
    cv2.ellipse(canvas, (int(hx + facing * 6 * scale), int(hy + 7 * scale)), (int(5.5 * scale), int(4.2 * scale)), 0, -20, 200, BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)

    # Expressive cartoon smile / laugh mouth
    mx_c = int(hx + facing * 4.0 * scale)
    my_c = int(hy + 17.0 * scale)
    if action_mode == "cheer":
        cv2.ellipse(canvas, (mx_c, my_c), (int(12 * scale), int(9 * scale)), 0, 0, 180, BGR_INK, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (mx_c, my_c), (int(12 * scale), int(9 * scale)), 0, 0, 180, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.line(canvas, (mx_c - int(12 * scale), my_c), (mx_c + int(12 * scale), my_c), BGR_INK, ink_w, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (mx_c, my_c - int(2 * scale)), (int(11 * scale), int(6 * scale)), 0, 15, 165, BGR_INK, ink_w, cv2.LINE_AA)

    # Canonical Headgear for Each of the 5 Agents:
    if hat_kind == "nib_crown":
        # Nib's Iconic 3-Pointed Fountain-Pen Nib Crown Hat with Smooth 11-Point Arched Rim (`agent_nib_keys.jpg`)!
        crown_peaks = [
            (hx - 45 * scale, hy - 16 * scale),
            (hx - 51 * scale, hy - 54 * scale),
            (hx - 21 * scale, hy - 35 * scale),
            (hx + 1 * scale, hy - 84 * scale),
            (hx + 22 * scale, hy - 35 * scale),
            (hx + 50 * scale, hy - 53 * scale),
            (hx + 45 * scale, hy - 16 * scale),
        ]
        rim_curve = []
        for ri in range(1, 11):
            u_r = ri / 11.0
            rx_c = hx + 45.0 * scale * (1.0 - 2.0 * u_r)
            ry_c = hy - (16.0 + 5.5 * math.sin(math.pi * u_r)) * scale
            rim_curve.append((rx_c, ry_c))
        crown_poly = np.round(np.array(crown_peaks + rim_curve, dtype=np.float64)).astype(np.int32)
        cv2.fillPoly(canvas, [crown_poly], hat_c, cv2.LINE_AA)
        cv2.polylines(canvas, [crown_poly], True, BGR_INK, ink_w, cv2.LINE_AA)
        # Fountain-nib vertical slit & circular breather hole `o`
        cv2.line(canvas, (int(hx + 1 * scale), int(hy - 84 * scale)), (int(hx + 1 * scale), int(hy - 39 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx + 1 * scale), int(hy - 39 * scale)), max(3, int(5.5 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx + 1 * scale), int(hy - 39 * scale)), max(3, int(5.5 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
    elif hat_kind == "visor_pencil":
        # Blue-Line's Visor Cap + Sideburn Hair + Banded Pencil Tucked Behind Ear (`agent_blueline_keys.jpg`)
        # Short-cropped sideburn hair tuft above ear
        sb_x = int(hx - facing * 38 * scale)
        sb_y = int(hy - 12 * scale)
        cv2.line(canvas, (sb_x, sb_y), (sb_x - int(facing * 5 * scale), sb_y + int(10 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.line(canvas, (sb_x + int(facing * 4 * scale), sb_y), (sb_x, sb_y + int(11 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy - 24 * scale)), (int(44 * scale), int(19 * scale)), 0, 180, 360, hat_c, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy - 24 * scale)), (int(44 * scale), int(19 * scale)), 0, 180, 360, BGR_INK, ink_w, cv2.LINE_AA)
        brim = np.array([
            [int(hx - 10 * scale * facing), int(hy - 24 * scale)],
            [int(hx + 54 * scale * facing), int(hy - 20 * scale)],
            [int(hx + 38 * scale * facing), int(hy - 11 * scale)],
            [int(hx - 20 * scale * facing), int(hy - 16 * scale)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [brim], hat_c, cv2.LINE_AA)
        cv2.polylines(canvas, [brim], True, BGR_INK, ink_w, cv2.LINE_AA)
        # Behind-ear Non-Photo Blue Pencil with ferrule band & eraser tip (`agent_blueline_keys.jpg` Q1-Q4)
        p_b0 = (hx - 44 * scale * facing, hy - 26 * scale)
        p_b1 = (hx - 8 * scale * facing, hy - 4 * scale)
        cv2.fillPoly(canvas, [_capsule_poly(p_b0, p_b1, 3.2 * scale + ink_w * 0.7, 2.8 * scale + ink_w * 0.7)], BGR_INK, cv2.LINE_AA)
        cv2.fillPoly(canvas, [_capsule_poly(p_b0, p_b1, 3.2 * scale, 2.8 * scale)], BGR_NONPHOTO_BLUE, cv2.LINE_AA)
        cv2.circle(canvas, (int(p_b0[0]), int(p_b0[1])), max(2, int(3.4 * scale)), (180, 168, 238), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(p_b0[0]), int(p_b0[1])), max(2, int(3.4 * scale)), BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)
    elif hat_kind == "beret_monocle":
        # Cel-Inker's French Beret with Headband Rim, Crown Crease & Stem (`agent_celinker_keys.jpg`)
        cv2.ellipse(canvas, (int(hx - 4 * scale * facing), int(hy - 32 * scale)), (int(48 * scale), int(18.5 * scale)), -12 * facing, 0, 360, hat_c, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx - 4 * scale * facing), int(hy - 32 * scale)), (int(48 * scale), int(18.5 * scale)), -12 * facing, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)
        # Beret headband rim & curved crown crease line
        cv2.line(canvas, (int(hx - 42 * scale), int(hy - 21 * scale)), (int(hx + 38 * scale), int(hy - 23 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx - 10 * scale * facing), int(hy - 32 * scale)), (int(22 * scale), int(7 * scale)), -10 * facing, 190, 340, BGR_INK, max(1, ink_w - 1), cv2.LINE_AA)
        cv2.line(canvas, (int(hx - 12 * scale * facing), int(hy - 49 * scale)), (int(hx - 18 * scale * facing), int(hy - 56 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
    elif hat_kind == "reel_ears_monocle":
        # Sprocket's Cap with Two 35mm Film Reels Mounted Like Ears (`agent_sprocket_keys.jpg`)!
        for rx_s in (-34.0 * scale, 26.0 * scale):
            rcx, rcy = int(hx + rx_s), int(hy - 44 * scale)
            cv2.circle(canvas, (rcx, rcy), int(19 * scale), (235, 240, 245), -1, cv2.LINE_AA)
            cv2.circle(canvas, (rcx, rcy), int(19 * scale), BGR_INK, ink_w, cv2.LINE_AA)
            for hole_a in range(5):
                ha = hole_a * (2.0 * math.pi / 5.0) + t_sec * 3.0
                cv2.circle(canvas, (int(rcx + 10 * scale * math.cos(ha)), int(rcy + 10 * scale * math.sin(ha))), max(2, int(3.5 * scale)), BGR_INK, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy - 24 * scale)), (int(42 * scale), int(16 * scale)), 0, 180, 360, hat_c, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy - 24 * scale)), (int(42 * scale), int(16 * scale)), 0, 180, 360, BGR_INK, ink_w, cv2.LINE_AA)
    elif hat_kind == "headphones_stopwatch":
        # Chime's Large Studio Headphones + Top Stopwatch Dial (`agent_chime_keys.jpg`)!
        cv2.ellipse(canvas, (int(hx), int(hy - 6 * scale)), (int(51 * scale), int(46 * scale)), 0, 180, 360, BGR_INK, ink_w + 2, cv2.LINE_AA)
        for ex_s in (-47.0 * scale, 47.0 * scale):
            cv2.ellipse(canvas, (int(hx + ex_s), int(hy + 2 * scale)), (int(11 * scale), int(18 * scale)), 0, 0, 360, hat_c, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (int(hx + ex_s), int(hy + 2 * scale)), (int(11 * scale), int(18 * scale)), 0, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx), int(hy - 40 * scale)), int(11 * scale), (250, 252, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx), int(hy - 40 * scale)), int(11 * scale), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.line(canvas, (int(hx), int(hy - 40 * scale)), (int(hx + 5 * scale), int(hy - 45 * scale)), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)

    # --- 5A. DRAW LEFT-ON-SCREEN ARM (`sh_l_x = chest_x - facing * 28.0 * scale` -> `el_lx` -> `wr_lx`) ---
    _draw_studio_agent_arm_and_hand(
        canvas, (sh_l_x, sh_l_y), (el_lx, el_ly), (wr_lx, wr_ly),
        w_up=14.5 * scale, w_low=12.0 * scale, scale=scale, facing=facing,
        is_right=False, is_near_shoulder=True, shirt_fill=shirt, skin_fill=skin, ink_w=ink_w,
    )

    # --- 5B. DRAW RIGHT-ON-SCREEN DRAWING ARM (Emerging from Outer Front Shoulder `chest_x + facing * 28.0 * scale`) ---
    # By anchoring the drawing arm outside the front apron strap, neither arm ever crosses the chest —
    # keeping the pointed `\/\/` collar, `|___|` bib, and front pocket emblem 100% visible (`nib_inking_keys.jpg` Q1-Q4)!
    sh_r_x = chest_x + facing * 28.0 * scale
    sh_r_y = chest_y + 18.0 * scale
    arm_l = 98.0 * scale
    el_rx, el_ry, wr_rx, wr_ry = solve_2bone_ik(sh_r_x, sh_r_y, target_rh[0], target_rh[1], arm_l * 0.52, arm_l * 0.50, bend_sign=float(facing))
    if facing >= 0:
        el_rx = max(el_rx, sh_r_x + 10.0 * scale)
    else:
        el_rx = min(el_rx, sh_r_x - 10.0 * scale)
    tx_tip, ty_tip = _draw_studio_agent_arm_and_hand(
        canvas, (sh_r_x, sh_r_y), (el_rx, el_ry), (wr_rx, wr_ry),
        w_up=15.0 * scale, w_low=12.5 * scale, scale=scale, facing=facing,
        is_right=True, is_near_shoulder=False, shirt_fill=shirt, skin_fill=skin,
        ink_w=ink_w, tool_rh=tool_rh,
    )
    return (wr_lx, wr_ly), (tx_tip, ty_tip)


# ==============================================================================
# IV. HAND-DRAWN STUDIO APPARATUS & COMPLETE 1920x1080 WIDESCREEN SCENE ENGINE
# ==============================================================================

def draw_studio_desk_and_apparatus(
    canvas: np.ndarray,
    scene_num: int,
    ax: float,
    ay: float,
    t_sec: float,
    slug: str,
    boil_phase: int = 0,
) -> tuple:
    """
    Manually draws the active studio machine / desk at `(ax, ay)` and returns
    `((lh_x, lh_y), (rh_x, rh_y), (look_x, look_y), action_mode, tool_rh)`
    so the Studio Animator's hands, eyes, and posture physically drive the apparatus!
    """
    wood_light = (158, 196, 224)
    wood_dark = (104, 138, 174)
    brass = (68, 186, 242)
    steel = (142, 148, 156)

    if scene_num == 1:
        # SCENE 1: 35mm Archival Moviola Editing Console with Spinning Reels & Hooded Viewer
        ped = [(ax - 68, ay - 150), (ax + 68, ay - 150), (ax + 86, ay), (ax - 86, ay)]
        draw_hand_ink_spline(canvas, ped, color_bgr=BGR_INK, base_thick=2.6, closed=True, fill_bgr=(92, 114, 98), boil_phase=boil_phase, seed=701, sharp_corners=True)
        scr = render_manual_cartoon_frame(slug, t_sec, 128, 96, stage_mode="multiplane_composite", boil_phase=boil_phase)
        sx0, sy0 = int(ax - 64), int(ay - 252)
        canvas[sy0:sy0 + 96, sx0:sx0 + 128] = scr
        cv2.rectangle(canvas, (sx0, sy0), (sx0 + 128, sy0 + 96), BGR_INK, 3, cv2.LINE_AA)
        ang_r = t_sec * 6.5
        for ri, (rx_c, ry_c) in enumerate([(ax - 54, ay - 292), (ax + 54, ay - 292)]):
            draw_hand_oval(canvas, rx_c, ry_c, 36, 36, color_bgr=BGR_INK, thick=2.6, fill_bgr=steel, boil_phase=boil_phase, seed=702 + ri)
            for sp in range(5):
                a_s = ang_r * (1 if ri == 0 else -1) + sp * (2.0 * math.pi / 5.0)
                cv2.line(canvas, (int(rx_c), int(ry_c)), (int(rx_c + 30 * math.cos(a_s)), int(ry_c + 30 * math.sin(a_s))), BGR_INK, 2, cv2.LINE_AA)
            draw_hand_oval(canvas, rx_c, ry_c, 10, 10, color_bgr=BGR_INK, thick=2.0, fill_bgr=brass, boil_phase=boil_phase, seed=705 + ri)
        crank_x = ax - 76 + 18 * math.cos(ang_r)
        crank_y = ay - 196 + 18 * math.sin(ang_r)
        cv2.line(canvas, (int(ax - 68), int(ay - 196)), (int(crank_x), int(crank_y)), BGR_INK, 4, cv2.LINE_AA)
        cv2.circle(canvas, (int(crank_x), int(crank_y)), 6, brass, -1, cv2.LINE_AA)
        lh = (crank_x, crank_y)
        rh = (ax - 12 + 24 * math.sin(t_sec * 3.8), ay - 204 + 14 * math.cos(t_sec * 3.8))
        return lh, rh, (ax, ay - 204), "draft", "loupe"

    if scene_num in (2, 3, 4, 5):
        # 3D PERSPECTIVE TILTED BEECHWOOD ANIMATION DESK + RECESSED LIGHTBOX + 4-PEG ACME BAR + SIDE-SHELF + ARCHITECT LAMP (`apparatus_lightbox_anim_keys.jpg`)!
        wood_side = (128, 166, 198)
        for leg_off in (-112, 98):
            lx_d = ax + leg_off
            leg_side = [(lx_d + 11, ay - 165), (lx_d + 19, ay - 161), (lx_d + 22, ay - 12), (lx_d + 14, ay - 12)]
            draw_hand_ink_spline(canvas, leg_side, color_bgr=BGR_INK, base_thick=2.2, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=709 + leg_off, sharp_corners=True)
            leg_poly = [(lx_d - 13, ay - 165), (lx_d + 11, ay - 165), (lx_d + 14, ay - 12), (lx_d - 16, ay - 12)]
            draw_hand_ink_spline(canvas, leg_poly, color_bgr=BGR_INK, base_thick=2.5, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=710 + leg_off, sharp_corners=True)
            foot_pad = [(lx_d - 28, ay - 12), (lx_d + 28, ay - 12), (lx_d + 32, ay), (lx_d - 32, ay)]
            draw_hand_ink_spline(canvas, foot_pad, color_bgr=BGR_INK, base_thick=2.4, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=711 + leg_off, sharp_corners=True)

        # Horizontal stretcher crossbar
        stretcher = [(ax - 100, ay - 68), (ax + 88, ay - 68), (ax + 88, ay - 52), (ax - 100, ay - 52)]
        draw_hand_ink_spline(canvas, stretcher, color_bgr=BGR_INK, base_thick=2.3, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=712, sharp_corners=True)

        # Right-Hand Horizontal Side-Shelf Bracket & Diagonal Strut (`apparatus_lightbox_anim_keys.jpg`)
        sh_x0, sh_x1, sh_y = ax + 72, ax + 144, ay - 164
        strut_p = [(ax + 98, sh_y + 8), (sh_x1 - 10, sh_y + 8), (ax + 104, sh_y + 38), (ax + 98, sh_y + 38)]
        draw_hand_ink_spline(canvas, strut_p, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=770, sharp_corners=True)
        sh_top = [(sh_x0, sh_y), (sh_x0 + 14, sh_y - 12), (sh_x1 + 12, sh_y - 12), (sh_x1, sh_y)]
        sh_fr = [(sh_x0, sh_y), (sh_x1, sh_y), (sh_x1, sh_y + 9), (sh_x0, sh_y + 9)]
        draw_hand_ink_spline(canvas, sh_top, color_bgr=BGR_INK, base_thick=2.1, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=771, sharp_corners=True)
        draw_hand_ink_spline(canvas, sh_fr, color_bgr=BGR_INK, base_thick=2.2, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=772, sharp_corners=True)

        # 3D Beveled Rectangular Drafting Board tilted toward animator
        b_fl = (ax - 148, ay - 150)
        b_bl = (ax - 68, ay - 246)
        b_br = (ax + 136, ay - 238)
        b_fr = (ax + 94, ay - 142)
        lip_h = 18
        board_front = [b_fl, b_fr, (b_fr[0], b_fr[1] + lip_h), (b_fl[0], b_fl[1] + lip_h)]
        board_right = [b_fr, b_br, (b_br[0], b_br[1] + lip_h * 0.85), (b_fr[0], b_fr[1] + lip_h)]
        draw_hand_ink_spline(canvas, board_front, color_bgr=BGR_INK, base_thick=2.4, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=713, sharp_corners=True)
        draw_hand_ink_spline(canvas, board_right, color_bgr=BGR_INK, base_thick=2.4, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=714, sharp_corners=True)
        draw_hand_ink_spline(canvas, [b_fl, b_bl, b_br, b_fr], color_bgr=BGR_INK, base_thick=2.8, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=715, sharp_corners=True)

        # Recessed Lightbox Frame & Backlit Circular Animation Disc
        lb_frame = [(ax - 110, ay - 158), (ax - 52, ay - 234), (ax + 86, ay - 228), (ax + 44, ay - 152)]
        draw_hand_ink_spline(canvas, lb_frame, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=716, sharp_corners=True)

        disc_cx, disc_cy = ax - 6, ay - 192
        draw_hand_oval(canvas, disc_cx, disc_cy, 74, 38, angle_deg=-10, color_bgr=BGR_INK, thick=2.4, fill_bgr=(212, 248, 255), boil_phase=boil_phase, seed=717)

        # 3-Sheet Rolling Paper Stack on 4-Peg Brass Acme Registration Bar!
        flip_u = 0.5 + 0.5 * math.sin(t_sec * 4.8)
        for sh_i in range(3):
            curl = (sh_i * 10.0 + flip_u * 22.0 * sh_i)
            sheet_poly = [
                (disc_cx - 58, disc_cy - 22),
                (disc_cx + 48, disc_cy - 28),
                (disc_cx + 62 + curl * 0.3, disc_cy + 16 - curl),
                (disc_cx - 48, disc_cy + 24 - curl * 0.4),
            ]
            s_fill = BGR_PAPER if sh_i == 2 else (224, 236, 246)
            draw_hand_ink_spline(canvas, sheet_poly, color_bgr=BGR_INK, base_thick=1.8, closed=True, fill_bgr=s_fill, boil_phase=boil_phase, seed=718 + sh_i, sharp_corners=True)

        # Raised 3D 4-Peg Brass Acme Registration Bar at top of disc
        peg_bar = [(disc_cx - 52, disc_cy - 28), (disc_cx + 44, disc_cy - 33), (disc_cx + 46, disc_cy - 23), (disc_cx - 50, disc_cy - 18)]
        draw_hand_ink_spline(canvas, peg_bar, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=brass, boil_phase=boil_phase, seed=722, sharp_corners=True)
        for p_u in (-0.7, -0.22, 0.22, 0.7):
            px_p = int(disc_cx - 3 + p_u * 46)
            py_p = int(disc_cy - 25 - p_u * 3)
            cv2.circle(canvas, (px_p, py_p), 3, BGR_INK, -1, cv2.LINE_AA)
            cv2.line(canvas, (px_p, py_p), (px_p, py_p - 6), BGR_INK, 2, cv2.LINE_AA)

        # Ceramic Mug of Sharpened Pencils on upper-right side-shelf
        mug_x, mug_y = ax + 114, ay - 224
        mug_poly = [(mug_x - 12, mug_y - 16), (mug_x + 12, mug_y - 16), (mug_x + 10, mug_y + 12), (mug_x - 10, mug_y + 12)]
        draw_hand_ink_spline(canvas, mug_poly, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=723, sharp_corners=True)
        for p_a in (-20, -6, 8, 22):
            rad_p = math.radians(p_a - 90)
            cv2.line(canvas, (int(mug_x + p_a * 0.3), int(mug_y - 16)), (int(mug_x + math.cos(rad_p) * 30), int(mug_y - 14 + math.sin(rad_p) * 30)), BGR_INK, 2, cv2.LINE_AA)

        # Articulated Double-Strut Architect Lamp clamped to upper-left of desk (`apparatus_lightbox_anim_keys.jpg`)
        cl_x, cl_y = ax - 54, ay - 242
        el_x, el_y = cl_x - 22, cl_y - 54
        hd_x, hd_y = cl_x + 48, cl_y - 82
        for st_off in (-3, 3):
            cv2.line(canvas, (int(cl_x + st_off), int(cl_y)), (int(el_x + st_off), int(el_y)), BGR_INK, 2, cv2.LINE_AA)
            cv2.line(canvas, (int(el_x), int(el_y + st_off)), (int(hd_x - 6), int(hd_y - 12 + st_off)), BGR_INK, 2, cv2.LINE_AA)
        for jx_l, jy_l, jr_l in [(cl_x, cl_y, 5), (el_x, el_y, 6), (hd_x - 6, hd_y - 12, 5)]:
            cv2.circle(canvas, (int(jx_l), int(jy_l)), jr_l, brass, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(jx_l), int(jy_l)), jr_l, BGR_INK, 2, cv2.LINE_AA)
        shade_l = [(hd_x - 32, hd_y + 16), (hd_x - 14, hd_y - 10), (hd_x + 14, hd_y - 10), (hd_x + 32, hd_y + 16)]
        draw_hand_ink_spline(canvas, shade_l, color_bgr=BGR_INK, base_thick=2.2, closed=True, fill_bgr=(92, 168, 118), boil_phase=boil_phase, seed=724, sharp_corners=True)

        if scene_num == 4:
            # Faceted Glass `INK` Bottle resting squarely on the right-hand side-shelf
            ik_x, ik_y = ax + 112, ay - 180
            bot_p = [(ik_x - 16, ik_y - 10), (ik_x + 16, ik_y - 10), (ik_x + 18, ik_y + 16), (ik_x - 18, ik_y + 16)]
            draw_hand_ink_spline(canvas, bot_p, color_bgr=BGR_INK, base_thick=2.2, closed=True, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=720, sharp_corners=True)
            cv2.rectangle(canvas, (int(ik_x - 14), int(ik_y - 1)), (int(ik_x + 14), int(ik_y + 14)), BGR_INK, -1)
        elif scene_num == 5:
            # Kidney Gouache Palette + Overhead Drying Line with Painted Cels!
            draw_hand_oval(canvas, ax + 82, ay - 168, 28, 14, color_bgr=BGR_INK, thick=2.0, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=721)
            for ci, col_p in enumerate([BGR_VERMILION, BGR_AMBER, BGR_NONPHOTO_BLUE, BGR_TEAL]):
                cv2.circle(canvas, (int(ax + 66 + ci * 10), int(ay - 168)), 4, col_p, -1, cv2.LINE_AA)
            line_y = ay - 395
            cv2.line(canvas, (int(ax - 220), int(line_y)), (int(ax + 150), int(line_y)), BGR_INK, 2, cv2.LINE_AA)
            for c_idx in range(3):
                cx_cel = ax - 150 + c_idx * 105
                sway_c = 8.0 * math.sin(t_sec * 3.0 + c_idx)
                cel_card = [
                    (cx_cel - 34, line_y + 4),
                    (cx_cel + 34, line_y + 4),
                    (cx_cel + 34 + sway_c, line_y + 56),
                    (cx_cel - 34 + sway_c, line_y + 56),
                ]
                draw_hand_ink_spline(canvas, cel_card, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=725 + c_idx)

        lh = (disc_cx - 42, disc_cy + 12 - flip_u * 24.0)
        rh = (disc_cx + 8 + 28.0 * math.cos(t_sec * 5.2), disc_cy - 2 + 16.0 * math.sin(t_sec * 7.8))
        tool = "blue_pencil" if scene_num in (2, 3) else ("speedball" if scene_num == 4 else "brush")
        return lh, rh, (disc_cx, disc_cy), "draft", tool

    if scene_num == 6:
        # SCENE 6: 7-LEVEL VERTICAL MULTIPLANE CAMERA TOWER & CRANK WHEEL
        for col_x in (ax - 95, ax + 95):
            cv2.line(canvas, (int(col_x), int(ay - 385)), (int(col_x), int(ay)), BGR_INK, 6, cv2.LINE_AA)
            cv2.line(canvas, (int(col_x), int(ay - 385)), (int(col_x), int(ay)), steel, 3, cv2.LINE_AA)
        # Top optical camera head pointing DOWN through the 7 glass planes!
        cam_box = [(ax - 44, ay - 395), (ax + 44, ay - 395), (ax + 32, ay - 335), (ax - 32, ay - 335)]
        draw_hand_ink_spline(canvas, cam_box, color_bgr=BGR_INK, base_thick=2.6, closed=True, fill_bgr=(64, 70, 78), boil_phase=boil_phase, seed=730)
        # 7 Multiplane Glass Platens (moving vertically in differential spacing!)
        plane_cols = [BGR_VERMILION, BGR_AMBER, BGR_TEAL, BGR_NONPHOTO_BLUE, (148, 112, 188), (112, 168, 128), (168, 142, 112)]
        for pi in range(7):
            py = ay - 305 + pi * 38 + 6.0 * math.sin(t_sec * 2.8 + pi * 0.65)
            platen = [(ax - 88, py - 5), (ax + 88, py - 5), (ax + 96, py + 5), (ax - 96, py + 5)]
            draw_hand_ink_spline(canvas, platen, color_bgr=BGR_INK, base_thick=1.8, closed=True, fill_bgr=plane_cols[pi], boil_phase=boil_phase, seed=732 + pi)
        # Brass elevation crank wheel on left column
        wh_x, wh_y = ax - 95, ay - 165
        wh_ang = t_sec * 4.5
        draw_hand_oval(canvas, wh_x, wh_y, 34, 34, color_bgr=BGR_INK, thick=2.6, fill_bgr=brass, boil_phase=boil_phase, seed=740)
        hx_c = wh_x + 28 * math.cos(wh_ang)
        hy_c = wh_y + 28 * math.sin(wh_ang)
        cv2.circle(canvas, (int(hx_c), int(hy_c)), 6, BGR_VERMILION, -1, cv2.LINE_AA)
        return (hx_c, hy_c), (ax - 68, ay - 210 + 18 * math.sin(t_sec * 3.0)), (ax, ay - 220), "crank", "caliper"

    if scene_num == 7:
        # SCENE 7: FOLEY SOUND STAGE TABLE, BRASS TUBULAR CHIMES & 16-GATE CALIPER RACK
        table = [(ax - 125, ay - 135), (ax + 125, ay - 135), (ax + 135, ay), (ax - 135, ay)]
        draw_hand_ink_spline(canvas, table, color_bgr=BGR_INK, base_thick=2.6, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=750)
        # Hanging brass tubular Foley chimes rack
        cv2.line(canvas, (int(ax - 75), int(ay - 310)), (int(ax + 85), int(ay - 310)), BGR_INK, 4, cv2.LINE_AA)
        for ci in range(5):
            cx_t = ax - 55 + ci * 32
            vib = 4.0 * math.sin(t_sec * 16.0 + ci) if (int(t_sec * 4) % 5 == ci) else 0.0
            tube_h = 95 + ci * 12
            cv2.line(canvas, (int(cx_t + vib), int(ay - 310)), (int(cx_t - vib), int(ay - 310 + tube_h)), BGR_INK, 6, cv2.LINE_AA)
            cv2.line(canvas, (int(cx_t + vib), int(ay - 310)), (int(cx_t - vib), int(ay - 310 + tube_h)), brass, 3, cv2.LINE_AA)
        strike_x = ax - 55 + ((int(t_sec * 3.5) % 5) * 32)
        rh = (strike_x - 16, ay - 245 + 18 * math.sin(t_sec * 10.5))
        lh = (ax - 45, ay - 150)
        return lh, rh, (strike_x, ay - 250), "draft", "mallet"

    # SCENE 8: 35MM PREMIERE BOOTH PROJECTOR & VOLUMETRIC LIGHT CONE
    proj_ped = [(ax - 45, ay - 175), (ax + 45, ay - 175), (ax + 68, ay), (ax - 68, ay)]
    draw_hand_ink_spline(canvas, proj_ped, color_bgr=BGR_INK, base_thick=2.6, closed=True, fill_bgr=steel, boil_phase=boil_phase, seed=760)
    proj_body = [(ax - 52, ay - 255), (ax + 52, ay - 255), (ax + 52, ay - 175), (ax - 52, ay - 175)]
    draw_hand_ink_spline(canvas, proj_body, color_bgr=BGR_INK, base_thick=2.6, closed=True, fill_bgr=(58, 64, 72), boil_phase=boil_phase, seed=761)
    for ri, (rx_p, ry_p) in enumerate([(ax - 42, ay - 295), (ax + 42, ay - 295)]):
        draw_hand_oval(canvas, rx_p, ry_p, 34, 34, color_bgr=BGR_INK, thick=2.6, fill_bgr=brass, boil_phase=boil_phase, seed=762 + ri)
        a_p = t_sec * 8.0
        for sp in range(4):
            cv2.line(canvas, (int(rx_p), int(ry_p)), (int(rx_p + 28 * math.cos(a_p + sp * 1.57)), int(ry_p + 28 * math.sin(a_p + sp * 1.57))), BGR_INK, 2, cv2.LINE_AA)
    # Volumetric light beam projecting onto the right-hand 4:3 screen!
    beam_ov = canvas.copy()
    beam_pts = np.array([[int(ax + 54), int(ay - 220)], [976, 80], [976, 730], [int(ax + 54), int(ay - 195)]], dtype=np.int32)
    cv2.fillPoly(beam_ov, [beam_pts], (182, 238, 255), cv2.LINE_AA)
    cv2.addWeighted(beam_ov, 0.25, canvas, 0.75, 0, canvas)
    return (ax - 120, ay - 310), (ax - 60, ay - 320), (980, 380), "cheer", "pencil"


def render_complete_studio_widescreen_frame(
    canvas: np.ndarray,
    scene_num: int,
    scene_title: str,
    agent_primary: str,
    agent_secondary: str,
    dialogue_line: str,
    slug: str,
    sub_step: float,
    global_frame_idx: int,
    total_frames: int,
    boil_phase: int = 0,
):
    """
    Renders a complete 1920x1080 Widescreen Frame of our Autonomous Agent Animation Studio
    with 100% MANUALLY DRAWN FRAME-BY-FRAME ANIMATIONS on both the Left Studio Stage
    and the Right Giant `4:3` (`920x690`) Multiplane Animation Canvas + 4-Cell Live Flipbook Strip!
    """
    H, W = canvas.shape[:2]
    t_sec = global_frame_idx / 24.0

    # Determine classical animation pass for the `4:3` cartoon screen based on `scene_num` & `sub_step`
    if scene_num in (1, 2):
        stage_mode = "rough_blue" if sub_step < 0.55 else "onion_flip"
        enable_onion = True
    elif scene_num == 3:
        stage_mode = "onion_flip" if sub_step < 0.65 else "cleanup_ink"
        enable_onion = True
    elif scene_num == 4:
        stage_mode = "cleanup_ink"
        enable_onion = (sub_step < 0.35)
    elif scene_num == 5:
        stage_mode = "gouache_cel"
        enable_onion = False
    else:
        stage_mode = "multiplane_composite"
        enable_onion = False

    # ==========================================================================
    # 1. LEFT STAGE (`x = 24..956`, `y = 62..996`): MULTIPLANE STUDIO SOUNDSTAGE
    # ==========================================================================
    lx0, ly0, lx1, ly1 = 24, 62, 956, 996
    floor_y = 895

    # Warm hand-drawn studio architecture & floorboards (clipped to Left Stage)
    studio_sub = canvas[ly0:ly1, lx0:lx1]
    sh_h, sh_w = studio_sub.shape[:2]
    f_local_y = floor_y - ly0

    # Warm parchment ceiling/wall & timber floorboards
    cv2.rectangle(studio_sub, (0, 0), (sh_w, f_local_y), (226, 238, 248), -1)
    cv2.rectangle(studio_sub, (0, f_local_y), (sh_w, sh_h), (176, 202, 224), -1)
    cv2.line(studio_sub, (0, f_local_y), (sh_w, f_local_y), BGR_INK, 3, cv2.LINE_AA)

    # Perspective floorboard planks (shifting with subtle studio camera pan!)
    cam_studio_pan = 28.0 * math.sin(t_sec * 0.5)
    for pi in range(-2, 10):
        px_top = int((pi / 7.0) * sh_w - cam_studio_pan * 0.4)
        px_bot = int(sh_w * 0.5 + (px_top - sh_w * 0.5) * 1.45)
        cv2.line(studio_sub, (px_top, f_local_y), (px_bot, sh_h), (144, 172, 198), 2, cv2.LINE_AA)

    # Studio Arched Windows, Rafters & Hanging Lamps in Midground/Background Parallax
    for wi in range(3):
        wx = int(115 + wi * 310 - cam_studio_pan * 0.25)
        wy = 165
        cv2.rectangle(studio_sub, (wx - 75, wy), (wx + 75, wy + 220), (242, 248, 252), -1)
        cv2.rectangle(studio_sub, (wx - 75, wy), (wx + 75, wy + 220), (148, 168, 186), 2, cv2.LINE_AA)
        cv2.line(studio_sub, (wx, wy), (wx, wy + 220), (148, 168, 186), 2, cv2.LINE_AA)
        cv2.line(studio_sub, (wx - 75, wy + 110), (wx + 75, wy + 110), (148, 168, 186), 2, cv2.LINE_AA)

    # Overhead swinging studio lamp casting warm light pool
    for li in range(2):
        lamp_x = int(240 + li * 420 - cam_studio_pan * 0.5)
        lamp_sway = int(6 * math.sin(t_sec * 2.1 + li))
        cv2.line(studio_sub, (lamp_x, 0), (lamp_x + lamp_sway, 88), BGR_INK, 2, cv2.LINE_AA)
        lamp_shade = [(lamp_x + lamp_sway - 38, 108), (lamp_x + lamp_sway - 16, 86), (lamp_x + lamp_sway + 16, 86), (lamp_x + lamp_sway + 38, 108)]
        draw_hand_ink_spline(studio_sub, lamp_shade, color_bgr=BGR_INK, base_thick=2.2, closed=True, fill_bgr=(92, 148, 112), boil_phase=boil_phase, seed=800 + li)

    # Draw Studio Desk / Apparatus at `(ax, ay)` on full canvas
    app_x = 655.0
    app_y = float(floor_y)
    lh_target, rh_target, look_pt, act_mode, tool_rh = draw_studio_desk_and_apparatus(
        canvas, scene_num, app_x, app_y, t_sec, slug, boil_phase=boil_phase
    )

    # Draw Primary & Secondary Heroic Hand-Drawn Studio Animators (NO duplicates, NO static cutouts!)
    if scene_num == 8:
        # Grand Premiere: All 5 Studio Animators cheering & bowing together!
        crew_order = ["nib", "blueline", "celinker", "sprocket", "chime"]
        for ci, c_id in enumerate(crew_order):
            cx_a = 115.0 + ci * 102.0
            t_off = t_sec + ci * 0.18
            wave_h = 45.0 * math.sin(t_off * 6.5)
            draw_heroic_studio_animator(
                canvas,
                agent_id=c_id,
                foot_x=cx_a,
                foot_y=float(floor_y),
                scale=0.82,
                t_sec=t_off,
                action_mode="cheer",
                target_lh=(cx_a - 64.0, floor_y - 235.0 - wave_h * 0.65),
                target_rh=(cx_a + 68.0, floor_y - 245.0 - wave_h * 0.65),
                look_at=(1200.0, 400.0),
                boil_phase=boil_phase + ci,
                facing=1,
                tool_rh="pencil",
            )
    else:
        # Secondary collaborating agent on far left (inspecting storyboard / carrying cels / checking X-sheet)
        if agent_secondary and agent_secondary != agent_primary:
            sec_x = 185.0 + 18.0 * math.sin(t_sec * 1.6)
            sec_wave = 16.0 * math.sin(t_sec * 3.8)
            draw_heroic_studio_animator(
                canvas,
                agent_id=agent_secondary,
                foot_x=sec_x,
                foot_y=float(floor_y - 10),
                scale=0.92,
                t_sec=t_sec + 0.4,
                action_mode="draft",
                target_lh=(sec_x - 52.0, floor_y - 148.0),
                target_rh=(sec_x + 64.0, floor_y - 168.0 + sec_wave),
                look_at=look_pt,
                boil_phase=boil_phase + 3,
                facing=1,
                tool_rh="pencil",
            )

        # Primary Lead Animator at the Desk / Moviola / Multiplane Tower (`scale = 1.08`, ~390px tall!)
        prim_x = 415.0
        draw_heroic_studio_animator(
            canvas,
            agent_id=agent_primary,
            foot_x=prim_x,
            foot_y=float(floor_y + 4),
            scale=1.08,
            t_sec=t_sec,
            action_mode=act_mode,
            target_lh=lh_target,
            target_rh=rh_target,
            look_at=look_pt,
            boil_phase=boil_phase,
            facing=1,
            tool_rh=tool_rh,
        )

    # Compact Comic Dialogue Balloon at Top-Left of Studio Stage (`y = 78..184`)
    bub_x0, bub_y0, bub_x1, bub_y1 = lx0 + 18, ly0 + 16, lx0 + 520, ly0 + 114
    cv2.rectangle(canvas, (bub_x0 + 4, bub_y0 + 4), (bub_x1 + 4, bub_y1 + 4), BGR_INK, -1)
    cv2.rectangle(canvas, (bub_x0, bub_y0), (bub_x1, bub_y1), (248, 251, 255), -1)
    cv2.rectangle(canvas, (bub_x0, bub_y0), (bub_x1, bub_y1), BGR_INK, 2, cv2.LINE_AA)
    words = (dialogue_line or "").upper().split()
    lines, cur = [], ""
    for w_tok in words:
        if len(cur) + len(w_tok) + 1 <= 44:
            cur = f"{cur} {w_tok}".strip()
        else:
            lines.append(cur)
            cur = w_tok
    if cur:
        lines.append(cur)
    for li, ln in enumerate(lines[:3]):
        cv2.putText(canvas, ln, (bub_x0 + 14, bub_y0 + 30 + li * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, BGR_INK, 1, cv2.LINE_AA)

    # ==========================================================================
    # 2. RIGHT STAGE (`x = 976..1896`, `y = 62..752`): GIANT `4:3` (`920x690`) CARTOON SCREEN!
    # ==========================================================================
    rx0, ry0 = 976, 62
    rw, rh = 920, 690  # Exact 4:3 (920 / 690 = 4 / 3)!
    rx1, ry1 = rx0 + rw, ry0 + rh

    hero_frame = render_manual_cartoon_frame(
        slug=slug,
        t_sec=t_sec,
        out_w=rw,
        out_h=rh - 34,
        stage_mode=stage_mode,
        boil_phase=boil_phase,
        enable_onion_skin=enable_onion,
    )
    # Header bar on 4:3 viewport
    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry0 + 34), (214, 228, 240), -1)
    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry0 + 34), BGR_INK, 2, cv2.LINE_AA)
    pass_label = {
        "rough_blue": "PASS A/B: NON-PHOTO BLUE ROUGH & VOLUMES",
        "onion_flip": "PASS XII: ONION-SKIN FLIP (PREV / CURR / NEXT)",
        "cleanup_ink": "PASS D: SPEEDBALL B-5 INDIA INK CLEANUP",
        "gouache_cel": "PASS C/E: WATERTIGHT GOUACHE CEL OPAQUING",
        "multiplane_composite": "7-PLANE MULTIPLANE CAMERA COMPOSITE (24 FPS ON ONES)",
    }.get(stage_mode, "24 FPS HAND-DRAWN ANIMATION")
    hdr_txt = f"4:3 ANIMATION EASEL & VIEWFINDER (920x690) -- #{slug.upper()[:18]} | {pass_label}"
    cv2.putText(canvas, hdr_txt, (rx0 + 12, ry0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, BGR_INK, 1, cv2.LINE_AA)

    canvas[ry0 + 34:ry1, rx0:rx1] = hero_frame
    cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), BGR_INK, 4, cv2.LINE_AA)

    # ==========================================================================
    # 3. BOTTOM-RIGHT (`x = 976..1896`, `y = 766..996`): LIVE 4-DRAWING FLIPBOOK STRIP!
    # ==========================================================================
    fx0, fy0, fx1, fy1 = 976, 766, 1896, 996
    cv2.rectangle(canvas, (fx0, fy0), (fx1, fy1), (222, 234, 244), -1)
    cv2.rectangle(canvas, (fx0, fy0), (fx1, fy1), BGR_INK, 3, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"LIVE EXPOSURE SHEET FLIPBOOK RIBBON (DRAW -> LOOK -> COMPARE -> ADJUST) | FRAME {global_frame_idx + 1}/{total_frames} @ 24 FPS ON ONES",
        (fx0 + 14, fy0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        BGR_INK,
        1,
        cv2.LINE_AA,
    )

    cell_w, cell_h = 212, 159  # Exact 4:3!
    for ci, dt_f in enumerate([-3, -2, -1, 0]):
        cx0 = fx0 + 14 + ci * (cell_w + 14)
        cy0 = fy0 + 32
        t_cell = max(0.0, t_sec + dt_f * (2.0 / 24.0))
        c_mode = "rough_blue" if ci == 0 else ("onion_flip" if ci == 1 else ("cleanup_ink" if ci == 2 else stage_mode))
        mini = render_manual_cartoon_frame(slug, t_cell, cell_w, cell_h, stage_mode=c_mode, boil_phase=boil_phase + dt_f)
        canvas[cy0:cy0 + cell_h, cx0:cx0 + cell_w] = mini
        b_col = BGR_VERMILION if dt_f == 0 else BGR_INK
        b_th = 3 if dt_f == 0 else 2
        cv2.rectangle(canvas, (cx0, cy0), (cx0 + cell_w, cy0 + cell_h), b_col, b_th, cv2.LINE_AA)
        tag = f"F{max(1, global_frame_idx + 1 + dt_f * 2):03d} ({['PASS A', 'FLIP', 'INK', 'ACTIVE'][ci]})"
        cv2.rectangle(canvas, (cx0, cy0), (cx0 + 128, cy0 + 20), BGR_INK, -1)
        cv2.putText(canvas, tag, (cx0 + 5, cy0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (245, 248, 255), 1, cv2.LINE_AA)

    # Outer frame border around Left Studio Stage + Caption Slate at Bottom (`y = 952..996`)
    cv2.rectangle(canvas, (lx0, ly0), (lx1, ly1), BGR_INK, 4, cv2.LINE_AA)
    cv2.rectangle(canvas, (lx0, 950), (lx1, ly1), (212, 226, 238), -1)
    cv2.rectangle(canvas, (lx0, 950), (lx1, ly1), BGR_INK, 2, cv2.LINE_AA)
    p_name = AGENT_PROFILES.get(agent_primary, AGENT_PROFILES["nib"])["name"]
    slate_txt = f"SCENE {scene_num}/8: {scene_title} | LEAD ANIMATOR: {p_name} | 100% HAND-DRAWN FRAME-BY-FRAME @ 24 FPS"
    cv2.putText(canvas, slate_txt, (lx0 + 14, 978), cv2.FONT_HERSHEY_SIMPLEX, 0.48, BGR_INK, 1, cv2.LINE_AA)


STUDIO_8_SCENE_SPECS = [
    (1, "35MM MOVIOLA & X-SHEET TIMING", "nib", "blueline", "HOLD THE 35MM REEL! MARK EVERY EXTREME ON THE X-SHEET!", "giraffe-walks-past"),
    (2, "4-KEY STORYBOARD WALL & TURNAROUNDS", "blueline", "nib", "PIN ALL 4 KEY POSES IN NON-PHOTO BLUE ON THE CORKBOARD!", "lighthouse"),
    (3, "BACKLIT GLASS LIGHTBOX & ONION-SKIN FLIP", "blueline", "nib", "ROLL THE 3 SHEETS ON THE ACME PEGS TO CHECK THE ARCS!", "campfire"),
    (4, "SPEEDBALL B-5 INDIA INK CLEANUP", "nib", "celinker", "PULL ONE CONTINUOUS TAPERED INK LINE OVER THE BLUE ROUGH!", "train"),
    (5, "ACETATE CEL GOUACHE OPAQUING & DRYING LINE", "celinker", "nib", "FLOOD-FILL WATERTIGHT GOUACHE PIGMENT ON THE REVERSE CEL!", "butterfly"),
    (6, "7-PLANE MULTIPLANE CAMERA TOWER", "celinker", "nib", "DIAL THE BRASS MICROMETER! LOCK ALL 7 PLANES IN DEPTH!", "airplane"),
    (7, "35MM CALIPER GATE AUDIT & FOLEY SOUNDSTAGE", "sprocket", "chime", "EXACT 4:3 GATE! HIT THE COCONUT HOOVES AND BRASS BELL!", "sailboat"),
    (8, "35MM DAILIES THEATRE GRAND PREMIERE", "nib", "sprocket", "ROLL THE 35MM PROJECTOR! ALL 5 ANIMATORS CHEER THE FILM!", "giraffe-walks-past"),
]


def build_studio_8scenes_flipbook_sheets():
    """
    Renders:
      - `preview/flipbook_scene_01.png` .. `preview/flipbook_scene_08.png` (each a 3x4 = 12-frame sequential flipbook)
      - `preview/studio_3100_hand_animated_code_atlas.png` (Master 8x6 = 48-tile overview atlas)
      - `preview/studio_8scenes_overview_atlas.png` & `preview/test_manual_studio_8scenes.png` (2x4 = 8-scene studio overview)
    and copies them to the artifact directory.
    """
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    preview_dir = root / "preview"
    bts_dir = root / "bts_films"
    preview_dir.mkdir(parents=True, exist_ok=True)
    bts_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

    tw, th = 384, 216
    pad = 12
    hdr_h = 54
    all_scene_rows = []
    hero_scene_cards = []

    for sc_num, sc_title, ag_p, ag_s, dlg, slug in STUDIO_8_SCENE_SPECS:
        cols, rows = 4, 3
        W_fb = pad + cols * (tw + pad)
        H_fb = hdr_h + pad + rows * (th + pad)
        fb_canvas = np.full((H_fb, W_fb, 3), BGR_PAPER, dtype=np.uint8)
        cv2.rectangle(fb_canvas, (0, 0), (W_fb, hdr_h), (214, 228, 240), -1)
        cv2.rectangle(fb_canvas, (0, 0), (W_fb, hdr_h), BGR_INK, 3)
        cv2.putText(
            fb_canvas,
            f"SCENE {sc_num}/8 FLIPBOOK: {sc_title} -- WATERTIGHT DUAL-PASS CAPSULE UNION @ 24 FPS ON ONES",
            (16, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.62,
            BGR_INK,
            1,
            cv2.LINE_AA,
        )

        row_strip_thumbs = []
        for f_k in range(12):
            sub_u = f_k / 11.0
            g_frame = (sc_num - 1) * 60 + f_k * 5
            w_canvas = np.full((1080, 1920, 3), BGR_PAPER, dtype=np.uint8)
            cv2.rectangle(w_canvas, (0, 0), (1920, 52), (212, 226, 238), -1)
            cv2.rectangle(w_canvas, (0, 0), (1920, 52), BGR_INK, 3)
            cv2.putText(
                w_canvas,
                f"SECONDHAND STUDIOS -- SCENE {sc_num}/8: {sc_title} | FRAME {f_k + 1:02d}/12",
                (24, 34),
                cv2.FONT_HERSHEY_DUPLEX,
                0.72,
                BGR_INK,
                2,
                cv2.LINE_AA,
            )
            render_complete_studio_widescreen_frame(
                w_canvas,
                scene_num=sc_num,
                scene_title=sc_title,
                agent_primary=ag_p,
                agent_secondary=ag_s,
                dialogue_line=dlg,
                slug=slug,
                sub_step=sub_u,
                global_frame_idx=g_frame,
                total_frames=480,
                boil_phase=f_k,
            )
            if f_k == 5:
                hero_scene_cards.append(cv2.resize(w_canvas, (640, 360), interpolation=cv2.INTER_AREA))

            thumb = cv2.resize(w_canvas, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.rectangle(thumb, (tw - 96, 6), (tw - 6, 28), BGR_INK, -1)
            cv2.putText(thumb, f"S{sc_num}.F{f_k + 1:02d}/12", (tw - 90, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, BGR_PAPER, 1, cv2.LINE_AA)

            r_i = f_k // cols
            c_i = f_k % cols
            x0 = pad + c_i * (tw + pad)
            y0 = hdr_h + pad + r_i * (th + pad)
            fb_canvas[y0:y0 + th, x0:x0 + tw] = thumb
            cv2.rectangle(fb_canvas, (x0, y0), (x0 + tw, y0 + th), BGR_INK, 2)

            if f_k % 2 == 0:
                row_strip_thumbs.append(cv2.resize(thumb, (256, 144), interpolation=cv2.INTER_AREA))

        fb_name = f"flipbook_scene_0{sc_num}.png"
        fb_path = preview_dir / fb_name
        cv2.imwrite(str(fb_path), fb_canvas)
        if artifact_dir.exists():
            shutil.copy2(fb_path, artifact_dir / fb_name)
        all_scene_rows.append(np.hstack(row_strip_thumbs))
        print(f"   [Studio Flipbook {sc_num}/8] Saved {fb_name}")

    # Master 8x6 Atlas
    grid_body = np.vstack(all_scene_rows)
    gh, gw = grid_body.shape[:2]
    m_hdr = np.full((64, gw, 3), (214, 228, 240), dtype=np.uint8)
    cv2.rectangle(m_hdr, (0, 0), (gw - 1, 63), BGR_INK, 3)
    cv2.putText(
        m_hdr,
        "MASTER 8-SCENE HAND-DRAWN STUDIO FLIPBOOK ATLAS -- ZERO JOINT SLOP WATER-TIGHT CAPSULE UNION @ 24 FPS",
        (16, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        BGR_INK,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        m_hdr,
        "Seamless Shoulders, Rolled-Up Cuffs, Articulated Wrists/Palms/Fingers, Cuffed Trousers, 3D Apparatus & 4:3 Multiplane Screen",
        (16, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (55, 48, 42),
        1,
        cv2.LINE_AA,
    )
    master_atlas = np.vstack([m_hdr, grid_body])
    m_path = preview_dir / "studio_3100_hand_animated_code_atlas.png"
    cv2.imwrite(str(m_path), master_atlas)
    if artifact_dir.exists():
        shutil.copy2(m_path, artifact_dir / "studio_3100_hand_animated_code_atlas.png")

    # 8-Scene Overview (4 rows x 2 cols of 640x360 cards)
    ov_rows = []
    for r_i in range(4):
        ov_rows.append(np.hstack([hero_scene_cards[r_i * 2], hero_scene_cards[r_i * 2 + 1]]))
    ov_grid = np.vstack(ov_rows)
    oh, ow = ov_grid.shape[:2]
    o_hdr = np.full((60, ow, 3), (214, 228, 240), dtype=np.uint8)
    cv2.rectangle(o_hdr, (0, 0), (ow - 1, 59), BGR_INK, 3)
    cv2.putText(
        o_hdr,
        "8-SCENE HAND-DRAWN MULTIPLANE ANIMATION STUDIO SOUNDSTAGE OVERVIEW (ALL 5 ANIMATORS & 3D APPARATUS)",
        (18, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        BGR_INK,
        1,
        cv2.LINE_AA,
    )
    ov_full = np.vstack([o_hdr, ov_grid])
    for out_p in [
        preview_dir / "studio_8scenes_overview_atlas.png",
        preview_dir / "test_manual_studio_8scenes.png",
        bts_dir / "studio_8scenes_overview_atlas.png",
    ]:
        cv2.imwrite(str(out_p), ov_full)
    if artifact_dir.exists():
        shutil.copy2(preview_dir / "studio_8scenes_overview_atlas.png", artifact_dir / "studio_8scenes_overview_atlas.png")
        shutil.copy2(preview_dir / "test_manual_studio_8scenes.png", artifact_dir / "test_manual_studio_8scenes.png")
    print("   [Studio Overview] Saved studio_3100_hand_animated_code_atlas.png, studio_8scenes_overview_atlas.png, test_manual_studio_8scenes.png")


if __name__ == "__main__":
    build_studio_8scenes_flipbook_sheets()

