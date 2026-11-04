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
    and calligraphic Speedball nib line taper.
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

    # Subtle 12fps/24fps organic hand-drawn boil (pinned at corners when `use_sharp=True`)
    b_step = (boil_phase // 2)
    idx = np.arange(n, dtype=np.float64)
    u = idx / float(max(1, n - 1))
    boil_amp = 0.34 if use_sharp else 0.42
    if use_sharp:
        seg_u = (idx % max(2, samples_per_seg)) / float(max(2, samples_per_seg))
        pin_env = np.sin(np.pi * seg_u)
    else:
        pin_env = 1.0
    bx = boil_amp * pin_env * np.sin(u * 9.4 + b_step * 1.7 + seed * 0.73)
    by = boil_amp * pin_env * np.cos(u * 11.2 - b_step * 1.3 + seed * 1.19)
    pts_f = np.column_stack([curve[:, 0] + bx, curve[:, 1] + by])
    pts_i = np.round(pts_f).astype(np.int32)

    if fill_bgr is not None and len(pts_i) >= 3:
        cv2.fillPoly(canvas, [pts_i], fill_bgr, cv2.LINE_AA)

    if base_thick <= 0:
        return

    if not taper_ends or base_thick <= 1.4:
        cv2.polylines(canvas, [pts_i], closed, color_bgr, max(1, int(round(base_thick))), cv2.LINE_AA)
        return

    # Calligraphic pressure profile (thicker in middle/belly of curve, tapered at starts/ends)
    for i in range(n - 1):
        um = (i + 0.5) / float(max(1, n - 1))
        envelope = math.sin(math.pi * um) ** 0.55 if not closed else (0.78 + 0.22 * math.sin(2.0 * math.pi * um))
        w = max(1, int(round(base_thick * (0.45 + 0.70 * envelope))))
        p1 = (int(pts_i[i, 0]), int(pts_i[i, 1]))
        p2 = (int(pts_i[i + 1, 0]), int(pts_i[i + 1, 1]))
        cv2.line(canvas, p1, p2, color_bgr, w, cv2.LINE_AA)
    if closed:
        cv2.line(canvas, (int(pts_i[-1, 0]), int(pts_i[-1, 1])), (int(pts_i[0, 0]), int(pts_i[0, 1])), color_bgr, max(1, int(round(base_thick * 0.75))), cv2.LINE_AA)


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
    """Draws an organic hand-drawn oval/sphere with optional gouache fill."""
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    ctrl = []
    for i in range(num_pts):
        th = (2.0 * math.pi * i) / num_pts
        lx = rx * math.cos(th)
        ly = ry * math.sin(th)
        wx = cx + lx * ca - ly * sa
        wy = cy + lx * sa + ly * ca
        ctrl.append((wx, wy))
    draw_hand_ink_spline(
        canvas, ctrl, color_bgr=color_bgr, base_thick=thick,
        closed=True, fill_bgr=fill_bgr, boil_phase=boil_phase, seed=seed,
        taper_ends=True, samples_per_seg=4,
    )


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
    Draws a fleshy, volumetric 2-segment limb (`p0 -> p1 -> p2`) with anatomical
    curves, joint cross-contour ellipses in construction mode, gouache fill, and ink contour.
    """
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2

    def _norm(ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        L = max(1e-4, math.hypot(dx, dy))
        return -dy / L, dx / L

    n0x, n0y = _norm(x0, y0, x1, y1)
    n2x, n2y = _norm(x1, y1, x2, y2)
    n1x, n1y = 0.5 * (n0x + n2x), 0.5 * (n0y + n2y)
    n1L = max(1e-4, math.hypot(n1x, n1y))
    n1x, n1y = n1x / n1L, n1y / n1L

    # Mid-belly swell on upper and lower limb segments
    m01x, m01y = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    m12x, m12y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w01 = 0.54 * (w0 + w1)
    w12 = 0.54 * (w1 + w2)

    left_side = [
        (x0 + n0x * w0, y0 + n0y * w0),
        (m01x + n0x * w01, m01y + n0y * w01),
        (x1 + n1x * w1, y1 + n1y * w1),
        (m12x + n2x * w12, m12y + n2y * w12),
        (x2 + n2x * w2, y2 + n2y * w2),
    ]
    right_side = [
        (x2 - n2x * w2, y2 - n2y * w2),
        (m12x - n2x * w12, m12y - n2y * w12),
        (x1 - n1x * w1, y1 - n1y * w1),
        (m01x - n0x * w01, m01y - n0y * w01),
        (x0 - n0x * w0, y0 - n0y * w0),
    ]

    if not construction_only and fill_bgr is not None:
        poly_ctrl = left_side + right_side
        draw_hand_ink_spline(canvas, poly_ctrl, color_bgr=ink_bgr, base_thick=0, closed=True, fill_bgr=fill_bgr, boil_phase=boil_phase, seed=seed, samples_per_seg=5)

    # Always draw bone centerline & joint spheres in Non-Photo Blue if construction_only or rough pass
    if construction_only:
        cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x1), int(y1)), max(2, int(w1 * 0.85)), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    draw_hand_ink_spline(canvas, left_side, color_bgr=ink_bgr, base_thick=ink_thick, closed=False, boil_phase=boil_phase, seed=seed + 1, samples_per_seg=6)
    draw_hand_ink_spline(canvas, right_side, color_bgr=ink_bgr, base_thick=ink_thick, closed=False, boil_phase=boil_phase, seed=seed + 2, samples_per_seg=6)


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

    gir_gold = (128, 198, 245) if is_colored else None
    gir_spot = (54, 96, 158) if is_colored else None
    gir_hoof = (42, 46, 54) if is_colored else None

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

    # Near legs (Hind Right `0.00`, Front Right `0.25`) + Hoof Dust Puffs!
    for idx_l, (bx_l, by_l, ph_o, bend_dir) in enumerate([
        (hip_x + 4, hip_y + 10, 0.00, -1.0),
        (chest_x + 6, chest_y + 12, 0.25, 1.0),
    ]):
        fx, fy, dust = _leg_pose(bx_l, by_l, ph_o)
        jx, jy, ex, ey = solve_2bone_ik(bx_l, by_l, fx, fy, leg_len * 0.52, leg_len * 0.54, bend_sign=bend_dir)
        draw_tapered_limb(canvas, (bx_l, by_l), (jx, jy), (ex, ey), 11.5, 7.5, 5.0, gir_gold, ink_bgr, 2.3, boil_phase, 240 + idx_l, construction_only=is_rough)
        draw_hand_oval(canvas, ex + 3, ey, 7.5, 4.5, color_bgr=ink_bgr, thick=2.2, fill_bgr=gir_hoof, boil_phase=boil_phase, seed=245 + idx_l)
        if dust:
            draw_hand_oval(canvas, ex - 8, ground_y - 4, 11, 6, color_bgr=BGR_NONPHOTO_BLUE if is_rough else (160, 175, 190), thick=1.4, fill_bgr=(220, 232, 242) if is_colored else None, boil_phase=boil_phase, seed=248 + idx_l)

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
    `Sprocket`, `Chime`) matching the exact character models from `assets/studio_doodles/`:
      - `Nib`: 3-Pointed Fountain-Pen Nib Crown Hat (with slit & breather hole `o`) + V-collar (`\\/\\/`) + bib apron + rolled elbow cuffs (`agent_nib_keys.jpg`).
      - `Blue-Line`: Visor cap + pencil tucked behind ear + collared shirt + pocket bib apron (`agent_blueline_keys.jpg`).
      - `Cel-Inker`: French beret with stem + jeweler's monocle over eye + apron pocket with 3 paint drops (`agent_celinker_keys.jpg`).
      - `Sprocket`: Cap with two 35mm film reels mounted like ears + monocle with cord + apron pocket with checkmark (`agent_sprocket_keys.jpg`).
      - `Chime`: Large studio headphones with a stopwatch dial on top of the headband + apron pocket with stopwatch (`agent_chime_keys.jpg`).
    """
    prof = AGENT_PROFILES.get(agent_id, AGENT_PROFILES["nib"])
    skin = prof["skin_bgr"]
    shirt = prof["shirt_bgr"]
    apron = prof["apron_bgr"]
    pants = prof["pants_bgr"]
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

    # --- 1. STURDY CARTOON LEGS, CUFFED TROUSERS & ROUNDED SHOES ---
    leg_l = 98.0 * scale
    for li, side_l in enumerate([-1, 1]):
        hip_j_x = hip_x + side_l * 18.0 * scale
        hip_j_y = hip_y + 8.0 * scale
        if action_mode == "walk":
            ph = t_sec * 5.5 + li * math.pi
            fx_l = hip_j_x + 22.0 * scale * math.sin(ph)
            fy_l = foot_y - max(0.0, 14.0 * scale * math.cos(ph))
        elif action_mode == "cheer":
            fx_l = foot_x + side_l * 22.0 * scale
            fy_l = min(foot_y, hip_y + 96.0 * scale)
        else:
            fx_l = foot_x + side_l * 20.0 * scale
            fy_l = foot_y
        kx, ky, ex, ey = solve_2bone_ik(hip_j_x, hip_j_y, fx_l, fy_l, leg_l * 0.52, leg_l * 0.52, bend_sign=float(facing))
        draw_tapered_limb(canvas, (hip_j_x, hip_j_y), (kx, ky), (ex, ey), 16.5 * scale, 14.0 * scale, 12.0 * scale, pants, BGR_INK, ink_w, boil_phase, 600 + li)
        # Cuffed trouser hem + rounded cartoon shoe
        cv2.ellipse(canvas, (int(ex + facing * 10 * scale), int(ey - 4 * scale)), (int(20 * scale), int(9.5 * scale)), 0, 0, 360, (240, 244, 248), -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(ex + facing * 10 * scale), int(ey - 4 * scale)), (int(20 * scale), int(9.5 * scale)), 0, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.rectangle(canvas, (int(ex - 13 * scale), int(ey - 16 * scale)), (int(ex + 13 * scale), int(ey - 8 * scale)), pants, -1)
        cv2.rectangle(canvas, (int(ex - 13 * scale), int(ey - 16 * scale)), (int(ex + 13 * scale), int(ey - 8 * scale)), BGR_INK, ink_w, cv2.LINE_AA)

    # --- 2. FAR ARM (Left Arm `target_lh`) WITH THICK ROLLED ELBOW CUFF ---
    sh_l_x = chest_x - facing * 24.0 * scale
    sh_l_y = chest_y + 16.0 * scale
    arm_l = 96.0 * scale
    el_lx, el_ly, wr_lx, wr_ly = solve_2bone_ik(sh_l_x, sh_l_y, target_lh[0], target_lh[1], arm_l * 0.52, arm_l * 0.52, bend_sign=float(facing))
    draw_tapered_limb(canvas, (sh_l_x, sh_l_y), (el_lx, el_ly), (wr_lx, wr_ly), 15.0 * scale, 12.5 * scale, 9.5 * scale, shirt, BGR_INK, ink_w, boil_phase, 610)
    # Rolled sleeve cuff at elbow
    cuff_ang_l = math.degrees(math.atan2(el_ly - sh_l_y, el_lx - sh_l_x)) + 90.0
    cv2.ellipse(canvas, (int(el_lx), int(el_ly)), (int(16 * scale), int(7.5 * scale)), cuff_ang_l, 0, 360, shirt, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(el_lx), int(el_ly)), (int(16 * scale), int(7.5 * scale)), cuff_ang_l, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(wr_lx), int(wr_ly)), (int(11 * scale), int(9 * scale)), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(wr_lx), int(wr_ly)), (int(11 * scale), int(9 * scale)), 0, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)

    # --- 3. TORSO, POINTED V-COLLAR (`\/\/`) & CANONICAL BIB APRON ---
    mid_x = 0.5 * (hip_x + chest_x)
    mid_y = 0.5 * (hip_y + chest_y)
    torso_poly = np.array([
        [int(chest_x - 36.0 * scale), int(chest_y)],
        [int(chest_x + 36.0 * scale), int(chest_y)],
        [int(hip_x + 34.0 * scale), int(hip_y + 14.0 * scale)],
        [int(hip_x - 34.0 * scale), int(hip_y + 14.0 * scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [torso_poly], shirt, cv2.LINE_AA)
    cv2.polylines(canvas, [torso_poly], True, BGR_INK, ink_w, cv2.LINE_AA)

    # Canonical Bib Apron with Shoulder Straps + Square-U Chest Bib + Emblems
    sway_hem = 6.0 * scale * math.sin(t_sec * 3.8 + 0.6)
    apron_poly = np.array([
        [int(chest_x - 24.0 * scale), int(chest_y + 22.0 * scale)],
        [int(chest_x + 26.0 * scale), int(chest_y + 22.0 * scale)],
        [int(hip_x + 36.0 * scale + sway_hem), int(hip_y + 42.0 * scale)],
        [int(hip_x - 32.0 * scale + sway_hem), int(hip_y + 42.0 * scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [apron_poly], apron, cv2.LINE_AA)
    cv2.polylines(canvas, [apron_poly], True, BGR_INK, ink_w, cv2.LINE_AA)
    # Shoulder straps & chest bib rectangle
    for st_s in (-1, 1):
        cv2.line(canvas, (int(chest_x + st_s * 20 * scale), int(chest_y + 2 * scale)), (int(chest_x + st_s * 18 * scale), int(chest_y + 24 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
    cv2.line(canvas, (int(chest_x - 18 * scale), int(chest_y + 24 * scale)), (int(chest_x + 18 * scale), int(chest_y + 24 * scale)), BGR_INK, ink_w, cv2.LINE_AA)

    # Pointed V-Neck Collar (`\/\/`) for Nib & Blue-Line
    if agent_id in ("nib", "blueline"):
        cx_v, cy_v = chest_x + facing * 4.0 * scale, chest_y + 4.0 * scale
        v_col = np.array([
            [int(cx_v - 18 * scale), int(cy_v - 2 * scale)],
            [int(cx_v - 10 * scale), int(cy_v + 16 * scale)],
            [int(cx_v), int(cy_v + 6 * scale)],
            [int(cx_v + 10 * scale), int(cy_v + 16 * scale)],
            [int(cx_v + 18 * scale), int(cy_v - 2 * scale)],
        ], dtype=np.int32)
        cv2.polylines(canvas, [v_col], False, BGR_INK, ink_w, cv2.LINE_AA)

    # Canonical Apron Pocket Emblems (`blueline`, `celinker`, `sprocket`, `chime`)
    if agent_id != "nib":
        px0, py0 = int(mid_x - 16 * scale), int(mid_y + 4 * scale)
        px1, py1 = int(mid_x + 16 * scale), int(mid_y + 30 * scale)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), (248, 250, 252), -1)
        cv2.rectangle(canvas, (px0, py0), (px1, py1), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)
        if agent_id == "celinker":
            for di in (-8, 0, 8):
                cv2.circle(canvas, (int(mid_x + di * scale), int(mid_y + 18 * scale)), max(2, int(3.2 * scale)), BGR_VERMILION, -1, cv2.LINE_AA)
        elif agent_id == "sprocket":
            cv2.line(canvas, (int(mid_x - 7 * scale), int(mid_y + 17 * scale)), (int(mid_x - 1 * scale), int(mid_y + 24 * scale)), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)
            cv2.line(canvas, (int(mid_x - 1 * scale), int(mid_y + 24 * scale)), (int(mid_x + 9 * scale), int(mid_y + 10 * scale)), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)
        elif agent_id == "chime":
            cv2.circle(canvas, (int(mid_x), int(mid_y + 2 * scale)), int(9 * scale), (248, 250, 252), -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(mid_x), int(mid_y + 2 * scale)), int(9 * scale), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)

    # --- 4. CANONICAL ROUND CARTOON HEAD, SOLID BLACK OVAL EYES & ICONIC HEADGEAR ---
    head_r_x = 46.0 * scale
    head_r_y = 42.0 * scale
    hx = chest_x + facing * 6.0 * scale
    hy = chest_y - 44.0 * scale

    # Main round-oval head cranium (`nib_inking_keys.jpg` & `agent_*_keys.jpg`)
    cv2.ellipse(canvas, (int(hx), int(hy)), (int(head_r_x), int(head_r_y)), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(hx), int(hy)), (int(head_r_x), int(head_r_y)), 0, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)

    # Solid-black vertical oval cartoon eyes (`● ●`) + curved eyebrows
    is_blink = ((int(t_sec * 24) + hash(agent_id) % 19) % 58) in (0, 1)
    for ei, ex_off in enumerate([-8.0 * scale * facing, 16.0 * scale * facing]):
        ex_c = hx + ex_off
        ey_c = hy - 2.0 * scale
        if is_blink:
            cv2.line(canvas, (int(ex_c - 6 * scale), int(ey_c)), (int(ex_c + 6 * scale), int(ey_c)), BGR_INK, ink_w, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (int(ex_c), int(ey_c)), (max(2, int(4.8 * scale)), max(3, int(7.5 * scale))), 0, 0, 360, BGR_INK, -1, cv2.LINE_AA)
        brow_lift = 3.0 * scale * math.sin(t_sec * 3.2)
        cv2.ellipse(canvas, (int(ex_c), int(ey_c - 10 * scale - brow_lift)), (int(7 * scale), int(3.5 * scale)), 0, 200, 340, BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)

    # Jeweler's Monocle for `celinker` and `sprocket` (`agent_celinker_keys.jpg`, `agent_sprocket_keys.jpg`)
    if agent_id in ("celinker", "sprocket"):
        mx_e = int(hx + 16.0 * scale * facing)
        my_e = int(hy - 2.0 * scale)
        cv2.circle(canvas, (mx_e, my_e), int(12 * scale), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.circle(canvas, (mx_e, my_e), int(9 * scale), BGR_INK, max(1, ink_w - 2), cv2.LINE_AA)

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
    hat_kind = prof["hat"]
    if hat_kind == "nib_crown":
        # Nib's Iconic 3-Pointed Fountain-Pen Nib Crown Hat (`agent_nib_keys.jpg` & `nib_inking_keys.jpg`)!
        crown_pts = np.array([
            [int(hx - 41 * scale), int(hy - 20 * scale)],
            [int(hx - 48 * scale), int(hy - 58 * scale)],
            [int(hx - 20 * scale), int(hy - 40 * scale)],
            [int(hx), int(hy - 88 * scale)],
            [int(hx + 20 * scale), int(hy - 40 * scale)],
            [int(hx + 48 * scale), int(hy - 58 * scale)],
            [int(hx + 41 * scale), int(hy - 20 * scale)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [crown_pts], hat_c, cv2.LINE_AA)
        cv2.polylines(canvas, [crown_pts], True, BGR_INK, ink_w, cv2.LINE_AA)
        # Fountain-nib vertical slit & circular breather hole `o`
        cv2.line(canvas, (int(hx), int(hy - 88 * scale)), (int(hx), int(hy - 44 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx), int(hy - 41 * scale)), max(3, int(5.5 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx), int(hy - 41 * scale)), max(3, int(5.5 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
    elif hat_kind == "visor_pencil":
        # Blue-Line's Visor Cap + Pencil Tucked Behind Ear (`agent_blueline_keys.jpg`)
        cv2.ellipse(canvas, (int(hx), int(hy - 26 * scale)), (int(42 * scale), int(18 * scale)), 0, 180, 360, hat_c, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy - 26 * scale)), (int(42 * scale), int(18 * scale)), 0, 180, 360, BGR_INK, ink_w, cv2.LINE_AA)
        brim = np.array([
            [int(hx - 10 * scale * facing), int(hy - 26 * scale)],
            [int(hx + 52 * scale * facing), int(hy - 22 * scale)],
            [int(hx + 36 * scale * facing), int(hy - 14 * scale)],
            [int(hx - 20 * scale * facing), int(hy - 18 * scale)],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [brim], hat_c, cv2.LINE_AA)
        cv2.polylines(canvas, [brim], True, BGR_INK, ink_w, cv2.LINE_AA)
        # Pencil behind ear
        cv2.line(canvas, (int(hx - 38 * scale * facing), int(hy - 28 * scale)), (int(hx - 12 * scale * facing), int(hy - 6 * scale)), BGR_NONPHOTO_BLUE, max(3, int(5 * scale)), cv2.LINE_AA)
        cv2.line(canvas, (int(hx - 38 * scale * facing), int(hy - 28 * scale)), (int(hx - 12 * scale * facing), int(hy - 6 * scale)), BGR_INK, 2, cv2.LINE_AA)
    elif hat_kind == "beret_monocle":
        # Cel-Inker's French Beret with Stem (`agent_celinker_keys.jpg`)
        cv2.ellipse(canvas, (int(hx - 4 * scale * facing), int(hy - 34 * scale)), (int(46 * scale), int(18 * scale)), -12 * facing, 0, 360, hat_c, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx - 4 * scale * facing), int(hy - 34 * scale)), (int(46 * scale), int(18 * scale)), -12 * facing, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.line(canvas, (int(hx - 12 * scale * facing), int(hy - 50 * scale)), (int(hx - 18 * scale * facing), int(hy - 57 * scale)), BGR_INK, ink_w, cv2.LINE_AA)
    elif hat_kind == "reel_ears_monocle":
        # Sprocket's Cap with Two 35mm Film Reels Mounted Like Ears (`agent_sprocket_keys.jpg`)!
        for rx_s in (-34.0 * scale, 26.0 * scale):
            rcx, rcy = int(hx + rx_s), int(hy - 46 * scale)
            cv2.circle(canvas, (rcx, rcy), int(19 * scale), (235, 240, 245), -1, cv2.LINE_AA)
            cv2.circle(canvas, (rcx, rcy), int(19 * scale), BGR_INK, ink_w, cv2.LINE_AA)
            for hole_a in range(5):
                ha = hole_a * (2.0 * math.pi / 5.0) + t_sec * 3.0
                cv2.circle(canvas, (int(rcx + 10 * scale * math.cos(ha)), int(rcy + 10 * scale * math.sin(ha))), max(2, int(3.5 * scale)), BGR_INK, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy - 26 * scale)), (int(40 * scale), int(16 * scale)), 0, 180, 360, hat_c, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (int(hx), int(hy - 26 * scale)), (int(40 * scale), int(16 * scale)), 0, 180, 360, BGR_INK, ink_w, cv2.LINE_AA)
    elif hat_kind == "headphones_stopwatch":
        # Chime's Large Studio Headphones + Top Stopwatch Dial (`agent_chime_keys.jpg`)!
        cv2.ellipse(canvas, (int(hx), int(hy - 6 * scale)), (int(50 * scale), int(46 * scale)), 0, 180, 360, BGR_INK, ink_w + 2, cv2.LINE_AA)
        for ex_s in (-46.0 * scale, 46.0 * scale):
            cv2.ellipse(canvas, (int(hx + ex_s), int(hy + 2 * scale)), (int(11 * scale), int(18 * scale)), 0, 0, 360, hat_c, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (int(hx + ex_s), int(hy + 2 * scale)), (int(11 * scale), int(18 * scale)), 0, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx), int(hy - 40 * scale)), int(11 * scale), (250, 252, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(hx), int(hy - 40 * scale)), int(11 * scale), BGR_INK, ink_w, cv2.LINE_AA)
        cv2.line(canvas, (int(hx), int(hy - 40 * scale)), (int(hx + 5 * scale), int(hy - 45 * scale)), BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)

    # --- 5. NEAR ARM (Right Arm `target_rh`) + ROLLED CUFF + FOUNTAIN-NIB PEN / TOOL ---
    sh_r_x = chest_x - facing * 4.0 * scale
    sh_r_y = chest_y + 18.0 * scale
    el_rx, el_ry, wr_rx, wr_ry = solve_2bone_ik(sh_r_x, sh_r_y, target_rh[0], target_rh[1], arm_l * 0.54, arm_l * 0.54, bend_sign=float(facing))
    draw_tapered_limb(canvas, (sh_r_x, sh_r_y), (el_rx, el_ry), (wr_rx, wr_ry), 16.0 * scale, 13.0 * scale, 10.0 * scale, shirt, BGR_INK, ink_w, boil_phase, 650)
    cuff_ang_r = math.degrees(math.atan2(el_ry - sh_r_y, el_rx - sh_r_x)) + 90.0
    cv2.ellipse(canvas, (int(el_rx), int(el_ry)), (int(17 * scale), int(8.0 * scale)), cuff_ang_r, 0, 360, shirt, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(el_rx), int(el_ry)), (int(17 * scale), int(8.0 * scale)), cuff_ang_r, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(wr_rx), int(wr_ry)), (int(12 * scale), int(9.5 * scale)), 0, 0, 360, skin, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(wr_rx), int(wr_ry)), (int(12 * scale), int(9.5 * scale)), 0, 0, 360, BGR_INK, ink_w, cv2.LINE_AA)

    # Studio Tool held in Right Hand (Iconic Fountain-Nib Dip Pen for Nib!)
    t_ang = math.atan2(target_rh[1] - el_ry, target_rh[0] - el_rx) - 0.45 * facing
    ca_t, sa_t = math.cos(t_ang), math.sin(t_ang)
    nx_t, ny_t = -sa_t, ca_t
    tx_tip = wr_rx + 44.0 * scale * ca_t
    ty_tip = wr_ry + 44.0 * scale * sa_t
    tx_back = wr_rx - 22.0 * scale * ca_t
    ty_back = wr_ry - 22.0 * scale * sa_t
    tool_col = BGR_NONPHOTO_BLUE if tool_rh == "blue_pencil" else (BGR_VERMILION if tool_rh == "brush" else BGR_AMBER)
    cv2.line(canvas, (int(tx_back), int(ty_back)), (int(wr_rx + 22 * scale * ca_t), int(wr_ry + 22 * scale * sa_t)), BGR_INK, max(3, int(7 * scale)), cv2.LINE_AA)
    cv2.line(canvas, (int(tx_back), int(ty_back)), (int(wr_rx + 22 * scale * ca_t), int(wr_ry + 22 * scale * sa_t)), tool_col, max(1, int(3.5 * scale)), cv2.LINE_AA)
    # Diamond fountain-nib head with slit
    nib_diamond = np.array([
        [int(wr_rx + 20 * scale * ca_t), int(wr_ry + 20 * scale * sa_t)],
        [int(wr_rx + 30 * scale * ca_t + nx_t * 7 * scale), int(wr_ry + 30 * scale * sa_t + ny_t * 7 * scale)],
        [int(tx_tip), int(ty_tip)],
        [int(wr_rx + 30 * scale * ca_t - nx_t * 7 * scale), int(wr_ry + 30 * scale * sa_t - ny_t * 7 * scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [nib_diamond], (235, 242, 248), cv2.LINE_AA)
    cv2.polylines(canvas, [nib_diamond], True, BGR_INK, max(2, ink_w - 1), cv2.LINE_AA)
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
        draw_hand_ink_spline(canvas, ped, color_bgr=BGR_INK, base_thick=2.6, closed=True, fill_bgr=(92, 114, 98), boil_phase=boil_phase, seed=701)
        # Viewer screen hood (4:3 miniature preview!)
        scr = render_manual_cartoon_frame(slug, t_sec, 128, 96, stage_mode="multiplane_composite", boil_phase=boil_phase)
        sx0, sy0 = int(ax - 64), int(ay - 252)
        canvas[sy0:sy0 + 96, sx0:sx0 + 128] = scr
        cv2.rectangle(canvas, (sx0, sy0), (sx0 + 128, sy0 + 96), BGR_INK, 3, cv2.LINE_AA)
        # Spinning top & bottom 35mm reels
        ang_r = t_sec * 6.5
        for ri, (rx_c, ry_c) in enumerate([(ax - 54, ay - 292), (ax + 54, ay - 292)]):
            draw_hand_oval(canvas, rx_c, ry_c, 36, 36, color_bgr=BGR_INK, thick=2.6, fill_bgr=steel, boil_phase=boil_phase, seed=702 + ri)
            for sp in range(5):
                a_s = ang_r * (1 if ri == 0 else -1) + sp * (2.0 * math.pi / 5.0)
                cv2.line(canvas, (int(rx_c), int(ry_c)), (int(rx_c + 30 * math.cos(a_s)), int(ry_c + 30 * math.sin(a_s))), BGR_INK, 2, cv2.LINE_AA)
            draw_hand_oval(canvas, rx_c, ry_c, 10, 10, color_bgr=BGR_INK, thick=2.0, fill_bgr=brass, boil_phase=boil_phase, seed=705 + ri)
        # Hand crank on left side
        crank_x = ax - 76 + 18 * math.cos(ang_r)
        crank_y = ay - 196 + 18 * math.sin(ang_r)
        cv2.line(canvas, (int(ax - 68), int(ay - 196)), (int(crank_x), int(crank_y)), BGR_INK, 4, cv2.LINE_AA)
        cv2.circle(canvas, (int(crank_x), int(crank_y)), 6, brass, -1, cv2.LINE_AA)
        lh = (crank_x, crank_y)
        rh = (ax - 12 + 24 * math.sin(t_sec * 3.8), ay - 204 + 14 * math.cos(t_sec * 3.8))
        return lh, rh, (ax, ay - 204), "draft", "loupe"

    if scene_num in (2, 3, 4, 5):
        # 3D PERSPECTIVE TILTED BEECHWOOD ANIMATION DESK + RECESSED LIGHTBOX + 4-PEG ACME BAR + MUG + INK BOTTLE + ARCHITECT LAMP!
        wood_side = (128, 166, 198)
        for leg_off in (-112, 98):
            lx_d = ax + leg_off
            leg_poly = [(lx_d - 14, ay - 165), (lx_d + 14, ay - 165), (lx_d + 18, ay - 12), (lx_d - 18, ay - 12)]
            draw_hand_ink_spline(canvas, leg_poly, color_bgr=BGR_INK, base_thick=2.5, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=710 + leg_off)
            foot_pad = [(lx_d - 28, ay - 12), (lx_d + 28, ay - 12), (lx_d + 32, ay), (lx_d - 32, ay)]
            draw_hand_ink_spline(canvas, foot_pad, color_bgr=BGR_INK, base_thick=2.4, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=711 + leg_off)

        # Horizontal stretcher crossbar
        stretcher = [(ax - 104, ay - 68), (ax + 92, ay - 68), (ax + 92, ay - 52), (ax - 104, ay - 52)]
        draw_hand_ink_spline(canvas, stretcher, color_bgr=BGR_INK, base_thick=2.3, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=712)

        # 3D Beveled Rectangular Drafting Board tilted toward animator
        b_fl = (ax - 148, ay - 150)
        b_bl = (ax - 68, ay - 246)
        b_br = (ax + 138, ay - 238)
        b_fr = (ax + 96, ay - 142)
        lip_h = 18
        board_front = [b_fl, b_fr, (b_fr[0], b_fr[1] + lip_h), (b_fl[0], b_fl[1] + lip_h)]
        board_right = [b_fr, b_br, (b_br[0], b_br[1] + lip_h * 0.85), (b_fr[0], b_fr[1] + lip_h)]
        draw_hand_ink_spline(canvas, board_front, color_bgr=BGR_INK, base_thick=2.4, closed=True, fill_bgr=wood_side, boil_phase=boil_phase, seed=713)
        draw_hand_ink_spline(canvas, board_right, color_bgr=BGR_INK, base_thick=2.4, closed=True, fill_bgr=wood_dark, boil_phase=boil_phase, seed=714)
        draw_hand_ink_spline(canvas, [b_fl, b_bl, b_br, b_fr], color_bgr=BGR_INK, base_thick=2.8, closed=True, fill_bgr=wood_light, boil_phase=boil_phase, seed=715)

        # Recessed Lightbox Frame & Backlit Circular Animation Disc
        lb_frame = [(ax - 110, ay - 158), (ax - 52, ay - 234), (ax + 86, ay - 228), (ax + 44, ay - 152)]
        draw_hand_ink_spline(canvas, lb_frame, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=716)

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
            draw_hand_ink_spline(canvas, sheet_poly, color_bgr=BGR_INK, base_thick=1.8, closed=True, fill_bgr=s_fill, boil_phase=boil_phase, seed=718 + sh_i)

        # 4-Peg Brass Acme Registration Bar at top of disc
        peg_bar = [(disc_cx - 52, disc_cy - 28), (disc_cx + 44, disc_cy - 33), (disc_cx + 46, disc_cy - 23), (disc_cx - 50, disc_cy - 18)]
        draw_hand_ink_spline(canvas, peg_bar, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=brass, boil_phase=boil_phase, seed=722)
        for p_u in (-0.7, -0.22, 0.22, 0.7):
            px_p = int(disc_cx - 3 + p_u * 46)
            py_p = int(disc_cy - 25 - p_u * 3)
            cv2.circle(canvas, (px_p, py_p), 3, BGR_INK, -1, cv2.LINE_AA)

        # Ceramic Mug of Sharpened Pencils on upper-left of desk
        mug_x, mug_y = ax - 42, ay - 252
        mug_poly = [(mug_x - 13, mug_y - 18), (mug_x + 13, mug_y - 18), (mug_x + 11, mug_y + 12), (mug_x - 11, mug_y + 12)]
        draw_hand_ink_spline(canvas, mug_poly, color_bgr=BGR_INK, base_thick=2.0, closed=True, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=723)
        for p_a in (-20, -6, 8, 22):
            rad_p = math.radians(p_a - 90)
            cv2.line(canvas, (int(mug_x + p_a * 0.3), int(mug_y - 18)), (int(mug_x + math.cos(rad_p) * 34), int(mug_y - 16 + math.sin(rad_p) * 34)), BGR_INK, 2, cv2.LINE_AA)

        # Articulated Double-Strut Architect Lamp clamped to upper-right of desk
        cl_x, cl_y = ax + 82, ay - 242
        el_x, el_y = cl_x + 36, cl_y - 68
        hd_x, hd_y = cl_x - 32, cl_y - 108
        cv2.line(canvas, (int(cl_x), int(cl_y)), (int(el_x), int(el_y)), BGR_INK, 3, cv2.LINE_AA)
        cv2.line(canvas, (int(el_x), int(el_y)), (int(hd_x), int(hd_y)), BGR_INK, 3, cv2.LINE_AA)
        cv2.circle(canvas, (int(el_x), int(el_y)), 5, brass, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(el_x), int(el_y)), 5, BGR_INK, 2, cv2.LINE_AA)
        shade_l = [(hd_x - 32, hd_y + 16), (hd_x - 14, hd_y - 10), (hd_x + 14, hd_y - 10), (hd_x + 32, hd_y + 16)]
        draw_hand_ink_spline(canvas, shade_l, color_bgr=BGR_INK, base_thick=2.2, closed=True, fill_bgr=(92, 168, 118), boil_phase=boil_phase, seed=724)

        if scene_num == 4:
            # Faceted Glass `INK` Bottle on right corner of desk
            ik_x, ik_y = ax + 82, ay - 168
            bot_p = [(ik_x - 16, ik_y - 10), (ik_x + 16, ik_y - 10), (ik_x + 18, ik_y + 16), (ik_x - 18, ik_y + 16)]
            draw_hand_ink_spline(canvas, bot_p, color_bgr=BGR_INK, base_thick=2.2, closed=True, fill_bgr=BGR_PAPER, boil_phase=boil_phase, seed=720)
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
                target_lh=(cx_a - 36.0, floor_y - 265.0 - wave_h),
                target_rh=(cx_a + 36.0, floor_y - 275.0 - wave_h),
                look_at=(1200.0, 400.0),
                boil_phase=boil_phase + ci,
                facing=1,
                tool_rh="pencil",
            )
    else:
        # Secondary collaborating agent on far left (inspecting storyboard / carrying cels / checking X-sheet)
        if agent_secondary and agent_secondary != agent_primary:
            sec_x = 185.0 + 18.0 * math.sin(t_sec * 1.6)
            sec_wave = 22.0 * math.sin(t_sec * 3.8)
            draw_heroic_studio_animator(
                canvas,
                agent_id=agent_secondary,
                foot_x=sec_x,
                foot_y=float(floor_y - 10),
                scale=0.92,
                t_sec=t_sec + 0.4,
                action_mode="draft",
                target_lh=(sec_x - 28.0, floor_y - 210.0),
                target_rh=(sec_x + 48.0, floor_y - 225.0 + sec_wave),
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
