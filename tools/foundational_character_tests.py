#!/usr/bin/env python3
"""
foundational_character_tests.py
================================================================================
STAGE 01 -> STAGE 06: ZERO-DRIFT FOUNDATIONAL CHARACTER SHEETS & PROGRESSIVE
ANIMATION PENCIL-TEST SUITE (`Nib`, `Blue-Line`, `Cel-Inker`, `Sprocket`, `Chime`)

Solves character drift, proportion shifts, and joint slop by establishing a
single, 100% deterministic, locked-volume 3-head-tall cartoon character rig
matching the canonical studio mascot sheets (`agent_nib_keys.jpg`,
`studio_crew_celebration_keys.jpg`, `agent_blueline_keys.jpg`,
`agent_celinker_keys.jpg`, `agent_sprocket_keys.jpg`, `agent_chime_keys.jpg`)
and building up slowly through a 6-Tier Classical Animation Curriculum:

  - Sheet 1 (`preview/basic_sheet_01_nib_model_and_turnaround.png`):
      * Row 1: 5-View Full-Body Orthographic Turnaround (`Front 0°`, `3/4 Left -35°`,
        `Profile -90°`, `3/4 Right +35°`, `Back 180°`) locked to horizontal
        animator ruler lines (`0.0H Crown Tip` down to `3.0H Ground Plane`).
      * Row 2: 4-Pass Construction Progression (`Pass 1 Blue Sphere`, `Pass 2
        Underdrawing`, `Pass 3 Speedball Ink`, `Pass 4 Gouache Cel`) + 4 Canonical
        Facial Expressions (`Neutral Focus`, `Idea Brow Lift`, `Tongue-Out
        Concentration`, `Joyful Open Laugh`).
      * Row 3: Zero-Slop Joint & Prop Anatomy Callouts (`Rolled Elbow Cuff`,
        `Collar & Apron Strap Junction`, `Fountain-Nib Pen Pinch`, `Two-Finger
        Pegbar Flip`, `Apron Hem -> Cuffed Trousers -> Animator Shoe`).

  - Sheet 2 (`preview/basic_sheet_02_5agents_lineup_proportions.png`):
      * Full-Body Orthographic Lineup of all 5 Studio Agents (`Nib`, `Blue-Line`,
        `Cel-Inker`, `Sprocket`, `Chime`) across all 3 passes (`Pass 1 Rough Blue`,
        `Pass 2 Clean Speedball Ink`, `Pass 3 Watertight Gouache Cel`).

  - Sheet 3 (`preview/basic_tests_tier1_to_tier6_flipbooks.png` + MP4/GIF):
      * Tier 1: Locked-Body Head Turn, Eye Blink & Expression Shift (`0% body drift`)
      * Tier 2: Constant-Volume Squash & Stretch Hop (`W * H = 1.00`)
      * Tier 3: Stationary-Body Single-Arm Pendulum & Wrist Whip Arc
      * Tier 4: Bilateral Coordination (Left Paper-Pinch + Right Pen-Stroke)
      * Tier 5: Grounded Weight Shift & 4-Beat Walk Cycle
      * Tier 6: Compound Acting Performance Built from Tiers 1–5

  - Sheet 4 (`preview/omni_basic_tests_start_end_pairs.png` + `omni_keyframe_packs/basic_tests/`):
      * Clean `1024x1024` Start & End Keyframe Pairs + Prompts for Google Omni / Veo.
================================================================================
"""

import json
import math
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
PREVIEW_DIR = ROOT / "preview"
FILMS_DIR = ROOT / "bts_films"
OMNI_BASIC_DIR = ROOT / "omni_keyframe_packs" / "basic_tests"
ARTIFACT_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

# Canonical Studio Palette (BGR)
BGR_PAPER = (234, 244, 250)         # Warm archival bond paper
BGR_PAPER_CARD = (244, 250, 254)    # Clean cel card background
BGR_WHITE = (248, 252, 255)         # Crisp shirt / crown / paper white
BGR_INK = (16, 18, 22)              # Speedball B-5 India ink black
BGR_NONPHOTO_BLUE = (224, 148, 56)  # Animator's non-photo blue pencil
BGR_BLUE_FAINT = (234, 202, 162)    # Faint construction ruler line
BGR_RULER_RED = (142, 148, 228)     # Horizontal proportion ruler red
BGR_ARC_RED = (48, 72, 212)         # Trajectory arc crimson
BGR_ONION_PREV = (132, 148, 218)    # Previous frame onion skin
BGR_ONION_NEXT = (218, 178, 118)    # Next frame onion skin

BGR_SKIN = (212, 232, 248)          # Warm cartoon skin gouache
BGR_SKIN_SHADE = (188, 210, 232)    # Shaded neck / back-arm skin
BGR_TONGUE = (98, 112, 224)         # Warm coral-pink tongue
BGR_SHOE = (58, 68, 92)             # Dark sepia-charcoal animator shoes
BGR_PANTS = (118, 94, 72)           # Warm slate-denim studio trousers
BGR_PANTS_DARK = (96, 76, 58)       # Shaded back trouser leg
BGR_BRASS = (62, 174, 236)          # Polished brass ferrule / dial / loupe

AGENT_STYLES = {
    "nib": {
        "name": "NIB (Lead Animator)",
        "apron_bgr": (54, 88, 204),      # Terracotta red apron
        "hat_bgr": (246, 251, 255),      # Crisp white fountain-nib crown
        "accent_bgr": (62, 174, 236),    # Brass breather hole & pen holder
        "tool": "fountain_pen",
        "emblem": "none",
    },
    "blueline": {
        "name": "BLUE-LINE (Rough Layout)",
        "apron_bgr": (156, 102, 48),     # Prussian blue apron
        "hat_bgr": (218, 172, 112),      # Sky-blue visor cap
        "accent_bgr": (224, 148, 56),    # Non-photo blue pencil
        "tool": "blue_pencil",
        "emblem": "pocket",
    },
    "celinker": {
        "name": "CEL-INKER (Ink & Paint)",
        "apron_bgr": (64, 164, 234),     # Golden ochre apron
        "hat_bgr": (58, 92, 208),        # Vermilion artist beret
        "accent_bgr": (196, 142, 62),    # Teal paint drops
        "tool": "brush",
        "emblem": "drops",
    },
    "sprocket": {
        "name": "SPROCKET (Multiplane Camera)",
        "apron_bgr": (84, 128, 58),      # Forest green apron
        "hat_bgr": (172, 184, 194),      # Silver-steel 35mm reel cap
        "accent_bgr": (62, 174, 236),    # Brass monocle
        "tool": "loupe",
        "emblem": "check",
    },
    "chime": {
        "name": "CHIME (Foley & Timing)",
        "apron_bgr": (72, 118, 224),     # Coral-amber apron
        "hat_bgr": (78, 88, 104),        # Studio headphones
        "accent_bgr": (62, 174, 236),    # Brass stopwatch dial
        "tool": "bell",
        "emblem": "watch",
    },
}


# ==============================================================================
# I. GEOMETRIC & WATER-TIGHT VECTOR SPLINE PRIMITIVES (ZERO SLOP)
# ==============================================================================

def _capsule_poly(p0: tuple, p1: tuple, r0: float, r1: float, steps: int = 18) -> np.ndarray:
    """Constructs a smooth closed stadium/capsule polygon from `(p0, r0)` to `(p1, r1)`."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-3:
        r = max(1.0, 0.5 * (r0 + r1))
        angs = np.linspace(0.0, 2.0 * math.pi, steps * 2, endpoint=False)
        pts = np.column_stack([x0 + r * np.cos(angs), y0 + r * np.sin(angs)])
        return np.round(pts).astype(np.int32)
    ang = math.atan2(dy, dx)
    pts = []
    for i in range(steps + 1):
        a = ang + math.pi * 0.5 + (math.pi * i) / steps
        pts.append((x0 + r0 * math.cos(a), y0 + r0 * math.sin(a)))
    for i in range(steps + 1):
        a = ang - math.pi * 0.5 + (math.pi * i) / steps
        pts.append((x1 + r1 * math.cos(a), y1 + r1 * math.sin(a)))
    return np.round(np.array(pts, dtype=np.float64)).astype(np.int32)


def _solve_2bone_ik(sx: float, sy: float, tx: float, ty: float, l1: float, l2: float, bend_sign: float = 1.0) -> tuple:
    """Analytical 2-bone inverse kinematics solver with invariant bone lengths `l1, l2`."""
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    max_reach = max(1e-3, (l1 + l2) * 0.992)
    min_reach = abs(l1 - l2) + 1e-3
    if d > max_reach:
        dx *= max_reach / d
        dy *= max_reach / d
        d = max_reach
    elif d < min_reach:
        dx *= min_reach / max(d, 1e-3)
        dy *= min_reach / max(d, 1e-3)
        d = min_reach
    cos_a = max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)))
    ang_base = math.atan2(dy, dx)
    ang_el = ang_base + bend_sign * math.acos(cos_a)
    ex = sx + l1 * math.cos(ang_el)
    ey = sy + l1 * math.sin(ang_el)
    wx = sx + dx
    wy = sy + dy
    return ex, ey, wx, wy


def _draw_outlined_poly(canvas: np.ndarray, pts: np.ndarray, fill_bgr: tuple, ink_bgr: tuple, ink_w: int, pass_mode: str):
    """Draws a closed polygon according to `pass_mode` (`rough_blue`, `cleanup_ink`, `gouache_cel`)."""
    if pass_mode == "rough_blue":
        cv2.polylines(canvas, [pts], True, BGR_NONPHOTO_BLUE, max(1, ink_w - 1), cv2.LINE_AA)
    elif pass_mode == "cleanup_ink":
        cv2.fillPoly(canvas, [pts], BGR_WHITE, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, ink_bgr, ink_w, cv2.LINE_AA)
    else:
        if fill_bgr is not None:
            cv2.fillPoly(canvas, [pts], fill_bgr, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], True, ink_bgr, ink_w, cv2.LINE_AA)


def _draw_proportion_ruler_grid(
    canvas: np.ndarray,
    x0: int,
    x1: int,
    ground_y: int,
    unit_h: float,
    show_labels: bool = True,
):
    """
    Draws horizontal classical animator model-sheet ruler lines (`0.0H` Crown Tip
    down to `3.0H` Ground Plane) across `[x0, x1]` so any vertical drift is
    immediately visible to the eye.
    """
    levels = [
        (3.00, "0.0H  CROWN TIP", BGR_RULER_RED, 1),
        (2.45, "0.55H HEAD / EYES", BGR_BLUE_FAINT, 1),
        (2.00, "1.0H  CHIN / COLLAR", BGR_RULER_RED, 1),
        (1.45, "1.55H ELBOW CUFF", BGR_BLUE_FAINT, 1),
        (0.85, "2.15H APRON HEM", BGR_RULER_RED, 1),
        (0.22, "2.78H ANKLE CUFF", BGR_BLUE_FAINT, 1),
        (0.00, "3.0H  GROUND PLANE", BGR_INK, 2),
    ]
    for h_mult, label, col, thick in levels:
        y = int(round(ground_y - h_mult * unit_h))
        if h_mult == 0.0:
            cv2.line(canvas, (x0, y), (x1, y), col, thick, cv2.LINE_AA)
        else:
            # Dashed animator construction line
            step = 14
            for sx in range(x0, x1, step):
                ex = min(x1, sx + 8)
                cv2.line(canvas, (sx, y), (ex, y), col, thick, cv2.LINE_AA)
        if show_labels:
            cv2.putText(
                canvas, label, (x0 + 6, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                BGR_INK if h_mult in (0.0, 2.0, 3.0) else BGR_NONPHOTO_BLUE,
                1, cv2.LINE_AA,
            )


# ==============================================================================
# II. UNIFIED ZERO-DRIFT CANONICAL CHARACTER RIG (`draw_canonical_character`)
# ==============================================================================

def _draw_arm_and_tool(
    canvas: np.ndarray,
    sh_pt: tuple,
    hand_pt: tuple,
    scale: float,
    ink_w: int,
    pass_mode: str,
    ink_col: tuple,
    shirt_fill: tuple,
    skin_fill: tuple,
    bend_sign: float = 1.0,
    tool: str = "none",
    hand_pose: str = "pinch",
    accent_bgr: tuple = BGR_BRASS,
) -> tuple:
    """
    Draws a complete, 100% watertight cartoon arm (`Upper Sleeve -> Puffy Rolled
    Elbow Cuff -> Bare Forearm -> Sculpted Hand & Fingers -> Studio Prop`) using
    strict painter's order so the rolled cuff cleanly wraps the elbow joint with
    ZERO joint slop.
    """
    l_up = 34.0 * scale
    l_low = 32.0 * scale
    ex, ey, wx, wy = _solve_2bone_ik(sh_pt[0], sh_pt[1], hand_pt[0], hand_pt[1], l_up, l_low, bend_sign=bend_sign)

    u_ang = math.atan2(ey - sh_pt[1], ex - sh_pt[0])
    f_ang = math.atan2(wy - ey, wx - ex)
    uca, usa = math.cos(u_ang), math.sin(u_ang)
    unx, uny = -usa, uca
    fca, fsa = math.cos(f_ang), math.sin(f_ang)
    fnx, fny = -fsa, fca

    if pass_mode == "rough_blue":
        cv2.line(canvas, (int(sh_pt[0]), int(sh_pt[1])), (int(ex), int(ey)), BGR_NONPHOTO_BLUE, 2, cv2.LINE_AA)
        cv2.line(canvas, (int(ex), int(ey)), (int(wx), int(wy)), BGR_NONPHOTO_BLUE, 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(ex), int(ey)), max(3, int(9 * scale)), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(wx), int(wy)), max(3, int(9 * scale)), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

    # 1. Upper Arm Shirt Sleeve (Shoulder to Elbow)
    r_sh = 12.5 * scale
    r_el = 11.0 * scale
    up_poly = _capsule_poly(sh_pt, (ex, ey), r_sh, r_el)
    _draw_outlined_poly(canvas, up_poly, shirt_fill, ink_col, ink_w, pass_mode)

    # 2. Bare Forearm + Wrist + Palm Base (Drawn BEFORE the rolled cuff so cuff overlaps the elbow seam!)
    r_fore = 8.8 * scale
    r_wrist = 7.2 * scale
    fore_poly = _capsule_poly((ex + fca * 2.0 * scale, ey + fsa * 2.0 * scale), (wx, wy), r_fore, r_wrist)

    palm_cx = wx + fca * 6.5 * scale
    palm_cy = wy + fsa * 6.5 * scale
    r_palm = 9.2 * scale

    # Fingers
    finger_polys = []
    if hand_pose == "open":
        spreads = [-0.45, -0.15, 0.15, 0.45]
        lens = [11.0, 13.0, 12.5, 10.0]
        for sp, fl in zip(spreads, lens):
            fa = f_ang + sp
            p0 = (palm_cx + math.cos(fa) * 4.0 * scale, palm_cy + math.sin(fa) * 4.0 * scale)
            p1 = (p0[0] + math.cos(fa) * fl * scale, p0[1] + math.sin(fa) * fl * scale)
            finger_polys.append(_capsule_poly(p0, p1, 3.3 * scale, 2.8 * scale))
    else:
        for sp, fl, noff in [(-0.18, 10.5, -3.2), (0.08, 11.0, 0.0), (0.32, 9.5, 3.2)]:
            fa = f_ang + sp
            p0 = (palm_cx + fca * 3.5 * scale + fnx * noff * scale, palm_cy + fsa * 3.5 * scale + fny * noff * scale)
            p1 = (p0[0] + math.cos(fa) * fl * scale, p0[1] + math.sin(fa) * fl * scale)
            finger_polys.append(_capsule_poly(p0, p1, 3.5 * scale, 3.0 * scale))

    # Thumb
    th_ang = f_ang - 0.65 * (1.0 if bend_sign >= 0 else -1.0)
    th_p0 = (palm_cx - fnx * 4.5 * scale, palm_cy - fny * 4.5 * scale)
    th_p1 = (th_p0[0] + math.cos(th_ang) * 9.5 * scale, th_p0[1] + math.sin(th_ang) * 9.5 * scale)
    thumb_poly = _capsule_poly(th_p0, th_p1, 3.6 * scale, 3.1 * scale)

    # Dual-Pass Watertight Forearm + Hand Union (Pass A: Outer Ink Casing -> Pass B: Seamless Skin Fill)
    if pass_mode == "rough_blue":
        cv2.polylines(canvas, [fore_poly], True, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        for fp in finger_polys:
            cv2.polylines(canvas, [fp], True, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
    else:
        fill_c = BGR_WHITE if pass_mode == "cleanup_ink" else skin_fill
        # Outer ink casing
        cv2.fillPoly(canvas, [_capsule_poly((ex + fca * 2.0 * scale, ey + fsa * 2.0 * scale), (wx, wy), r_fore + ink_w, r_wrist + ink_w)], ink_col, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(palm_cx)), int(round(palm_cy))), int(round(r_palm + ink_w)), ink_col, -1, cv2.LINE_AA)
        for fp in finger_polys:
            cv2.polylines(canvas, [fp], True, ink_col, ink_w * 2, cv2.LINE_AA)
            cv2.fillPoly(canvas, [fp], ink_col, cv2.LINE_AA)
        cv2.polylines(canvas, [thumb_poly], True, ink_col, ink_w * 2, cv2.LINE_AA)
        cv2.fillPoly(canvas, [thumb_poly], ink_col, cv2.LINE_AA)

        # Inner seamless skin fill
        cv2.fillPoly(canvas, [fore_poly], fill_c, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(palm_cx)), int(round(palm_cy))), int(round(r_palm)), fill_c, -1, cv2.LINE_AA)
        for fp in finger_polys:
            cv2.fillPoly(canvas, [fp], fill_c, cv2.LINE_AA)
        cv2.fillPoly(canvas, [thumb_poly], fill_c, cv2.LINE_AA)

    # 3. Puffy Rolled Shirt Cuff (Sits cleanly OVER the elbow junction, 100% sealing the seam!)
    mid_ang = math.atan2(usa + fsa, uca + fca)
    mnx, mny = -math.sin(mid_ang), math.cos(mid_ang)
    cuff_w = 8.2 * scale
    cuff_r = 4.8 * scale
    c_p0 = (ex - mnx * cuff_w, ey - mny * cuff_w)
    c_p1 = (ex + mnx * cuff_w, ey + mny * cuff_w)
    cuff_poly = _capsule_poly(c_p0, c_p1, cuff_r, cuff_r)
    _draw_outlined_poly(canvas, cuff_poly, shirt_fill, ink_col, ink_w, pass_mode)
    # Subtle fold crease inside rolled cuff
    if pass_mode != "rough_blue":
        cf0 = (int(round(ex - mnx * cuff_w * 0.48)), int(round(ey - mny * cuff_w * 0.48)))
        cf1 = (int(round(ex + mnx * cuff_w * 0.48)), int(round(ey + mny * cuff_w * 0.48)))
        cv2.line(canvas, cf0, cf1, ink_col, max(1, ink_w - 1), cv2.LINE_AA)

    # 4. Studio Tool / Prop held in hand (`fountain_pen`, `blue_pencil`, `brush`, `loupe`, `bell`, `paper_sheet`)
    tip_pt = (palm_cx + fca * 12.0 * scale, palm_cy + fsa * 12.0 * scale)
    if tool in ("fountain_pen", "blue_pencil", "brush"):
        t_ang = f_ang - 0.52
        tca, tsa = math.cos(t_ang), math.sin(t_ang)
        tnx, tny = -tsa, tca
        gx, gy = palm_cx + fca * 3.5 * scale, palm_cy + fsa * 3.5 * scale
        bx, by = gx - tca * 28.0 * scale, gy - tsa * 28.0 * scale
        fx, fy = gx + tca * 16.0 * scale, gy + tsa * 16.0 * scale
        tx, ty = gx + tca * 42.0 * scale, gy + tsa * 42.0 * scale
        tip_pt = (tx, ty)

        shaft_col = BGR_NONPHOTO_BLUE if tool == "blue_pencil" else (BGR_ARC_RED if tool == "brush" else accent_bgr)
        shaft_poly = _capsule_poly((bx, by), (fx, fy), 3.2 * scale, 3.8 * scale)
        _draw_outlined_poly(canvas, shaft_poly, shaft_col, ink_col, ink_w, pass_mode)

        if tool == "fountain_pen":
            # Iconic diamond fountain-pen nib with slit & breather hole
            mid_nx, mid_ny = gx + tca * 27.0 * scale, gy + tsa * 27.0 * scale
            nw = 6.2 * scale
            nib_pts = np.array([
                [fx + tnx * 3.5 * scale, fy + tny * 3.5 * scale],
                [mid_nx + tnx * nw, mid_ny + tny * nw],
                [tx, ty],
                [mid_nx - tnx * nw, mid_ny - tny * nw],
                [fx - tnx * 3.5 * scale, fy - tny * 3.5 * scale],
            ], dtype=np.int32)
            _draw_outlined_poly(canvas, nib_pts, BGR_WHITE, ink_col, ink_w, pass_mode)
            if pass_mode != "rough_blue":
                cv2.line(canvas, (int(tx), int(ty)), (int(mid_nx), int(mid_ny)), ink_col, max(1, ink_w - 1), cv2.LINE_AA)
                cv2.circle(canvas, (int(mid_nx), int(mid_ny)), max(2, int(2.2 * scale)), ink_col, -1, cv2.LINE_AA)
        elif tool == "blue_pencil":
            cone_pts = np.array([
                [fx + tnx * 3.6 * scale, fy + tny * 3.6 * scale],
                [tx, ty],
                [fx - tnx * 3.6 * scale, fy - tny * 3.6 * scale],
            ], dtype=np.int32)
            _draw_outlined_poly(canvas, cone_pts, BGR_NONPHOTO_BLUE, ink_col, ink_w, pass_mode)
        elif tool == "brush":
            mid_bx, mid_by = gx + tca * 28.0 * scale, gy + tsa * 28.0 * scale
            br_pts = np.array([
                [fx + tnx * 3.5 * scale, fy + tny * 3.5 * scale],
                [mid_bx + tnx * 6.5 * scale, mid_by + tny * 6.5 * scale],
                [tx, ty],
                [mid_bx - tnx * 5.0 * scale, mid_by - tny * 5.0 * scale],
                [fx - tnx * 3.5 * scale, fy - tny * 3.5 * scale],
            ], dtype=np.int32)
            _draw_outlined_poly(canvas, br_pts, accent_bgr, ink_col, ink_w, pass_mode)

        # Re-draw thumb overlapping the tool shaft for a true anatomical pinch
        if pass_mode != "rough_blue":
            fill_c = BGR_WHITE if pass_mode == "cleanup_ink" else skin_fill
            cv2.polylines(canvas, [thumb_poly], True, ink_col, ink_w * 2, cv2.LINE_AA)
            cv2.fillPoly(canvas, [thumb_poly], fill_c, cv2.LINE_AA)

    return (ex, ey), (wx, wy), tip_pt


def draw_canonical_character(
    canvas: np.ndarray,
    agent_id: str = "nib",
    foot_x: float = 256.0,
    foot_y: float = 450.0,
    unit_h: float = 120.0,
    yaw_deg: float = 0.0,
    head_yaw_deg: float = None,
    head_pitch_deg: float = 0.0,
    head_roll_deg: float = 0.0,
    squash_stretch: float = 0.0,
    hip_offset_x: float = 0.0,
    hip_offset_y: float = 0.0,
    torso_lean_deg: float = 0.0,
    lh_target: tuple = None,
    rh_target: tuple = None,
    lh_pose: str = "open",
    rh_pose: str = "pinch",
    rh_tool: str = None,
    l_foot_offset: tuple = (0.0, 0.0),
    r_foot_offset: tuple = (0.0, 0.0),
    blink: float = 0.0,
    expression: str = "smile",
    pass_mode: str = "gouache_cel",
    ink_override: tuple = None,
) -> dict:
    """
    Draws a 100% deterministic, zero-drift, full-body Studio Character (`Nib`,
    `Blue-Line`, `Cel-Inker`, `Sprocket`, `Chime`) locked to the `3.0 * unit_h`
    proportion system:
      - Total height from Ground (`foot_y`) to Crown Tip = `3.0 * unit_h` (when `squash_stretch == 0`).
      - Strict constant-volume conservation: `scale_y = 1.0 + squash_stretch`,
        `scale_x = 1.0 / scale_y` so `scale_x * scale_y == 1.000`.
      - Supports continuous 360-degree orthographic rotation (`yaw_deg in [-180, 180]`)
        plus independent head rotation (`head_yaw_deg`, `head_pitch_deg`, `head_roll_deg`).
    """
    style = AGENT_STYLES.get(agent_id, AGENT_STYLES["nib"])
    scale = unit_h / 120.0
    sy = max(0.72, min(1.32, 1.0 + squash_stretch))
    sx = 1.0 / sy  # Exact constant-volume conservation (sx * sy == 1.0)

    ink_col = ink_override if ink_override is not None else (BGR_NONPHOTO_BLUE if pass_mode == "rough_blue" else BGR_INK)
    ink_w = max(2, int(round(3.8 * scale)))

    if head_yaw_deg is None:
        head_yaw_deg = yaw_deg
    if rh_tool is None:
        rh_tool = style["tool"]

    rad_yaw = math.radians(yaw_deg)
    sin_y = math.sin(rad_yaw)
    cos_y = math.cos(rad_yaw)
    is_back = abs(yaw_deg) > 105.0
    facing_sign = 1.0 if sin_y >= 0.0 else -1.0

    # --- Key Vertical Landmarks Locked to `unit_h` (`0H..3H`) ---
    # Soles at `foot_y`, Ankles at `0.22H`, Hips/Apron Hem at `0.85H`, Shoulders/Collar at `1.92H`,
    # Head Center at `2.30H`, Crown Tip at `3.00H`.
    hip_cx = foot_x + hip_offset_x * scale
    hip_cy = foot_y - (0.88 * unit_h * sy) + hip_offset_y * scale

    lean_rad = math.radians(torso_lean_deg)
    torso_len = 1.04 * unit_h * sy
    neck_cx = hip_cx + torso_len * math.sin(lean_rad)
    neck_cy = hip_cy - torso_len * math.cos(lean_rad)

    head_cx = neck_cx + (6.0 * scale * math.sin(lean_rad))
    head_cy = neck_cy - (0.35 * unit_h * sy)

    # Shoulder anchors (foreshortened by `cos(yaw)`)
    sh_half_w = 26.5 * scale * sx * max(0.32, abs(cos_y))
    # Screen-left shoulder vs screen-right shoulder
    sh_l_pt = (neck_cx - sh_half_w + sin_y * 5.0 * scale, neck_cy + 14.0 * scale * sy)
    sh_r_pt = (neck_cx + sh_half_w + sin_y * 5.0 * scale, neck_cy + 14.0 * scale * sy)

    if lh_target is None:
        lh_target = (sh_l_pt[0] - 24.0 * scale * sx, hip_cy + 4.0 * scale * sy)
    if rh_target is None:
        rh_target = (sh_r_pt[0] + 28.0 * scale * sx, hip_cy - 10.0 * scale * sy)

    # Determine arm draw order based on `yaw_deg` (when turned, one arm is behind the torso, one is in front)
    draw_left_behind = (sin_y > 0.25)
    draw_right_behind = (sin_y < -0.25)

    # ==========================================================================
    # LAYER 1: BACK ARM (if character is in 3/4 or profile turn)
    # ==========================================================================
    if draw_left_behind:
        _draw_arm_and_tool(
            canvas, sh_l_pt, lh_target, scale, ink_w, pass_mode, ink_col,
            BGR_WHITE, BGR_SKIN_SHADE, bend_sign=1.0, tool="none", hand_pose=lh_pose,
            accent_bgr=style["accent_bgr"],
        )
    elif draw_right_behind:
        _draw_arm_and_tool(
            canvas, sh_r_pt, rh_target, scale, ink_w, pass_mode, ink_col,
            BGR_WHITE, BGR_SKIN_SHADE, bend_sign=-1.0, tool=rh_tool, hand_pose=rh_pose,
            accent_bgr=style["accent_bgr"],
        )

    # ==========================================================================
    # LAYER 2: STURDY TROUSER LEGS, ROLLED ANKLE CUFFS & FLAT-SOLED SHOES
    # ==========================================================================
    leg_sep = 17.0 * scale * sx * max(0.25, abs(cos_y))
    leg_defs = [
        (-1, hip_cx - leg_sep, foot_x - leg_sep + l_foot_offset[0] * scale, foot_y + l_foot_offset[1] * scale, BGR_PANTS_DARK if sin_y > 0.2 else BGR_PANTS),
        (1,  hip_cx + leg_sep, foot_x + leg_sep + r_foot_offset[0] * scale, foot_y + r_foot_offset[1] * scale, BGR_PANTS_DARK if sin_y < -0.2 else BGR_PANTS),
    ]
    if sin_y < -0.2:
        leg_defs = leg_defs[::-1]  # Draw far leg first

    for side_s, hx_l, fx_l, fy_l, p_col in leg_defs:
        ank_x = fx_l
        ank_y = fy_l - 14.0 * scale
        thigh_l = 44.0 * scale
        calf_l = 44.0 * scale
        bend_dir = -facing_sign if abs(sin_y) > 0.2 else float(side_s) * 0.35
        kx, ky, ax, ay = _solve_2bone_ik(hx_l, hip_cy, ank_x, ank_y, thigh_l, calf_l, bend_sign=bend_dir)

        # 2A. Rounded Flat-Soled Animator Shoe (drawn first, under the rolled ankle cuff)
        toe_dir = math.copysign(1.0, sin_y) if abs(sin_y) > 0.22 else float(side_s)
        heel_x = int(round(ax - toe_dir * 13.0 * scale * sx))
        toe_x = int(round(ax + toe_dir * 25.0 * scale * sx))
        sole_y = int(round(fy_l))
        top_y = int(round(fy_l - 15.0 * scale))
        shoe_pts = np.array([
            [heel_x, sole_y - int(3 * scale)],
            [heel_x + int(toe_dir * 2 * scale), top_y + int(3 * scale)],
            [int(ax + toe_dir * 6 * scale), top_y],
            [toe_x - int(toe_dir * 4 * scale), top_y + int(3 * scale)],
            [toe_x, sole_y - int(4 * scale)],
            [toe_x - int(toe_dir * 2 * scale), sole_y],
            [heel_x + int(toe_dir * 2 * scale), sole_y],
        ], dtype=np.int32)
        _draw_outlined_poly(canvas, shoe_pts, BGR_SHOE, ink_col, ink_w, pass_mode)

        # 2B. Trouser Leg (Thigh + Calf Dual-Pass Union)
        r_th = 13.5 * scale * sx
        r_kn = 12.2 * scale * sx
        r_an = 11.2 * scale * sx
        if pass_mode == "rough_blue":
            cv2.line(canvas, (int(hx_l), int(hip_cy)), (int(kx), int(ky)), BGR_NONPHOTO_BLUE, 2, cv2.LINE_AA)
            cv2.line(canvas, (int(kx), int(ky)), (int(ax), int(ay)), BGR_NONPHOTO_BLUE, 2, cv2.LINE_AA)
        else:
            fill_p = BGR_WHITE if pass_mode == "cleanup_ink" else p_col
            cv2.fillPoly(canvas, [_capsule_poly((hx_l, hip_cy), (kx, ky), r_th + ink_w, r_kn + ink_w)], ink_col, cv2.LINE_AA)
            cv2.fillPoly(canvas, [_capsule_poly((kx, ky), (ax, ay), r_kn + ink_w, r_an + ink_w)], ink_col, cv2.LINE_AA)
            cv2.fillPoly(canvas, [_capsule_poly((hx_l, hip_cy), (kx, ky), r_th, r_kn)], fill_p, cv2.LINE_AA)
            cv2.fillPoly(canvas, [_capsule_poly((kx, ky), (ax, ay), r_kn, r_an)], fill_p, cv2.LINE_AA)

        # 2C. Rolled Trouser Cuff at Ankle (matches `studio_crew_celebration_keys.jpg` bottom row!)
        cuff_p0 = (ax - 12.5 * scale * sx, ay)
        cuff_p1 = (ax + 12.5 * scale * sx, ay)
        cuff_leg_poly = _capsule_poly(cuff_p0, cuff_p1, 4.2 * scale, 4.2 * scale)
        _draw_outlined_poly(canvas, cuff_leg_poly, p_col, ink_col, ink_w, pass_mode)

    # ==========================================================================
    # LAYER 3: TORSO SHIRT, A-LINE STUDIO APRON, BIB, STRAPS & V-NECK COLLAR
    # ==========================================================================
    top_w = 34.0 * scale * sx * max(0.58, abs(cos_y) * 0.85 + 0.15)
    hem_w = 45.0 * scale * sx * max(0.62, abs(cos_y) * 0.85 + 0.15)
    hem_y = hip_cy + 14.0 * scale * sy
    sh_y = neck_cy + 8.0 * scale * sy

    # 3A. Base White Shirt Torso
    torso_pts = np.array([
        [neck_cx - top_w, sh_y],
        [neck_cx + top_w, sh_y],
        [hip_cx + hem_w * 0.92, hem_y - 16.0 * scale * sy],
        [hip_cx - hem_w * 0.92, hem_y - 16.0 * scale * sy],
    ], dtype=np.int32)
    _draw_outlined_poly(canvas, torso_pts, BGR_WHITE, ink_col, ink_w, pass_mode)

    # 3B. Iconic Studio Overall Apron (Bib + Shoulder Straps + Flared Skirt Hem)
    bib_w = top_w * 0.54
    bib_top_y = sh_y + 20.0 * scale * sy
    apron_pts = np.array([
        # Left shoulder strap top
        [neck_cx - top_w * 0.78, sh_y + 2.0 * scale],
        [neck_cx - bib_w, sh_y + 2.0 * scale],
        # Inner bib top (unless back view)
        [neck_cx - bib_w, bib_top_y],
        [neck_cx + bib_w, bib_top_y],
        # Right shoulder strap top
        [neck_cx + bib_w, sh_y + 2.0 * scale],
        [neck_cx + top_w * 0.78, sh_y + 2.0 * scale],
        # Right apron side & flared hem
        [hip_cx + hem_w, hem_y],
        [hip_cx - hem_w, hem_y],
    ], dtype=np.int32)
    _draw_outlined_poly(canvas, apron_pts, style["apron_bgr"], ink_col, ink_w, pass_mode)

    # Apron side seam / pocket details (matching `agent_*_keys.jpg` and `studio_crew_celebration_keys.jpg`)
    if pass_mode != "rough_blue" and not is_back:
        # Side hip cutaway lines on apron
        for s_dir in (-1, 1):
            sx_line = int(round(hip_cx + s_dir * hem_w * 0.74))
            cv2.line(
                canvas,
                (sx_line, int(round(bib_top_y + 18.0 * scale * sy))),
                (int(round(hip_cx + s_dir * hem_w * 0.80)), int(round(hem_y))),
                ink_col, max(1, ink_w - 1), cv2.LINE_AA,
            )

        # Character-specific front pocket emblem
        emblem = style["emblem"]
        px_c = int(round(0.5 * (neck_cx + hip_cx) + sin_y * 10.0 * scale * sx))
        py_c = int(round(bib_top_y + 34.0 * scale * sy))
        pw_h = int(round(16.0 * scale * sx))
        ph_h = int(round(15.0 * scale * sy))
        if emblem in ("pocket", "drops", "check", "watch"):
            p_rect = np.array([
                [px_c - pw_h, py_c - ph_h],
                [px_c + pw_h, py_c - ph_h],
                [px_c + pw_h, py_c + int(ph_h * 0.7)],
                [px_c, py_c + ph_h + int(3 * scale)],
                [px_c - pw_h, py_c + int(ph_h * 0.7)],
            ], dtype=np.int32)
            _draw_outlined_poly(canvas, p_rect, BGR_WHITE, ink_col, max(2, ink_w - 1), pass_mode)
            if emblem == "pocket":
                cv2.line(canvas, (px_c - pw_h + 2, py_c - ph_h + 5), (px_c + pw_h - 2, py_c - ph_h + 5), ink_col, max(1, ink_w - 1), cv2.LINE_AA)
            elif emblem == "check":
                cv2.line(canvas, (px_c - 6, py_c), (px_c - 1, py_c + 6), ink_col, max(2, ink_w - 1), cv2.LINE_AA)
                cv2.line(canvas, (px_c - 1, py_c + 6), (px_c + 8, py_c - 6), ink_col, max(2, ink_w - 1), cv2.LINE_AA)
            elif emblem == "drops":
                for dx_o, dy_o in [(-7, 2), (0, 5), (7, 2)]:
                    cv2.circle(canvas, (px_c + int(dx_o * scale), py_c + int(dy_o * scale)), max(2, int(3.0 * scale)), style["accent_bgr"], -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px_c + int(dx_o * scale), py_c + int(dy_o * scale)), max(2, int(3.0 * scale)), ink_col, 1, cv2.LINE_AA)
            elif emblem == "watch":
                cv2.circle(canvas, (px_c, py_c - ph_h - int(2 * scale)), max(3, int(6.5 * scale)), style["accent_bgr"], -1, cv2.LINE_AA)
                cv2.circle(canvas, (px_c, py_c - ph_h - int(2 * scale)), max(3, int(6.5 * scale)), ink_col, max(1, ink_w - 1), cv2.LINE_AA)

    # 3C. Sturdy Cartoon Neck Cylinder + Crisp Pointed V-Neck Shirt Collar
    neck_poly = _capsule_poly(
        (neck_cx, sh_y + 4.0 * scale * sy),
        (head_cx, head_cy + 22.0 * scale * sy),
        10.5 * scale * sx, 10.5 * scale * sx,
    )
    _draw_outlined_poly(canvas, neck_poly, BGR_SKIN_SHADE, ink_col, ink_w, pass_mode)

    if not is_back:
        col_cx = neck_cx + sin_y * 6.0 * scale * sx
        col_top_y = sh_y - 3.5 * scale * sy
        col_bot_y = bib_top_y - 2.0 * scale * sy
        col_w = 18.0 * scale * sx
        l_col = np.array([
            [col_cx - col_w, col_top_y],
            [col_cx, col_top_y + 8.0 * scale * sy],
            [col_cx - col_w * 0.48, col_bot_y],
        ], dtype=np.int32)
        r_col = np.array([
            [col_cx + col_w, col_top_y],
            [col_cx + col_w * 0.48, col_bot_y],
            [col_cx, col_top_y + 8.0 * scale * sy],
        ], dtype=np.int32)
        _draw_outlined_poly(canvas, l_col, BGR_WHITE, ink_col, ink_w, pass_mode)
        _draw_outlined_poly(canvas, r_col, BGR_WHITE, ink_col, ink_w, pass_mode)

    # ==========================================================================
    # LAYER 4: HEAD CRANIUM SPHERE, EARS, FACIAL EXPRESSIONS & SIGNATURE HAT
    # ==========================================================================
    h_rad_yaw = math.radians(head_yaw_deg)
    h_sin = math.sin(h_rad_yaw)
    h_cos = math.cos(h_rad_yaw)
    h_back = abs(head_yaw_deg) > 105.0

    rx_head = int(round(46.0 * scale * sx))
    ry_head = int(round(42.0 * scale * sy))
    h_cx = int(round(head_cx))
    h_cy = int(round(head_cy + head_pitch_deg * 0.45 * scale))

    # 4A. Ears (drawn behind cranium silhouette when visible)
    for ear_s in (-1, 1):
        # Hide the front-occluded ear when turned beyond 45 degrees
        if (ear_s == 1 and h_sin > 0.65) or (ear_s == -1 and h_sin < -0.65):
            continue
        ear_x = int(round(h_cx + ear_s * (rx_head - 2.0 * scale) * max(0.35, abs(h_cos)) - h_sin * 6.0 * scale))
        ear_y = int(round(h_cy + 4.0 * scale))
        ear_r = int(round(10.5 * scale))
        if pass_mode == "rough_blue":
            cv2.circle(canvas, (ear_x, ear_y), ear_r, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        else:
            fill_e = BGR_WHITE if pass_mode == "cleanup_ink" else BGR_SKIN
            cv2.circle(canvas, (ear_x, ear_y), ear_r + ink_w, ink_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (ear_x, ear_y), ear_r, fill_e, -1, cv2.LINE_AA)
            cv2.circle(canvas, (ear_x, ear_y), max(2, ear_r // 2), ink_col, max(1, ink_w - 1), cv2.LINE_AA)

    # 4B. Rounded Cartoon Cranium Sphere
    if pass_mode == "rough_blue":
        cv2.ellipse(canvas, (h_cx, h_cy), (rx_head, ry_head), head_roll_deg, 0, 360, BGR_NONPHOTO_BLUE, 2, cv2.LINE_AA)
        # Construction crosshairs (horizontal eye line + vertical centerline wrap)
        eye_off_x = int(round(h_sin * rx_head * 0.55))
        eye_off_y = int(round(head_pitch_deg * 0.6 * scale))
        cv2.line(canvas, (h_cx - rx_head, h_cy + eye_off_y), (h_cx + rx_head, h_cy + eye_off_y), BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.ellipse(canvas, (h_cx, h_cy), (abs(eye_off_x) + 2, ry_head), head_roll_deg, -90, 90, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
    else:
        fill_h = BGR_WHITE if pass_mode == "cleanup_ink" else BGR_SKIN
        cv2.ellipse(canvas, (h_cx, h_cy), (rx_head + ink_w, ry_head + ink_w), head_roll_deg, 0, 360, ink_col, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (h_cx, h_cy), (rx_head, ry_head), head_roll_deg, 0, 360, fill_h, -1, cv2.LINE_AA)

    # 4C. Facial Features (Eyes, Brows, C-Curve Nose, Expressive Mouth)
    if not h_back:
        face_cx = h_cx + h_sin * (rx_head * 0.38)
        face_cy = h_cy + 2.0 * scale + head_pitch_deg * 0.45 * scale
        eye_sep = 15.0 * scale * sx * max(0.30, abs(h_cos))

        for ei, es in enumerate((-1, 1)):
            # In strict 90-deg profile, only draw the near eye
            if abs(h_sin) > 0.88 and ((h_sin > 0 and es < 0) or (h_sin < 0 and es > 0)):
                continue
            ex_i = int(round(face_cx + es * eye_sep))
            ey_i = int(round(face_cy - 5.0 * scale * sy))
            erx = max(2, int(round(4.6 * scale * sx)))
            ery = max(1, int(round(6.8 * scale * sy * (1.0 - 0.88 * blink))))
            if blink > 0.75:
                cv2.line(canvas, (ex_i - erx - 2, ey_i), (ex_i + erx + 2, ey_i), ink_col, ink_w, cv2.LINE_AA)
            else:
                cv2.ellipse(canvas, (ex_i, ey_i), (erx, ery), 0, 0, 360, ink_col, -1, cv2.LINE_AA)

            # Expressive Eyebrow
            brow_lift = 4.5 * scale if expression in ("idea", "laugh") else (-2.5 * scale if expression == "focus" else 0.0)
            brow_tilt = -es * 12.0 if expression == "focus" else es * 8.0
            cv2.ellipse(
                canvas,
                (ex_i, int(round(ey_i - 10.5 * scale - brow_lift))),
                (int(round(7.5 * scale)), int(round(3.5 * scale))),
                brow_tilt, 200, 340, ink_col, ink_w, cv2.LINE_AA,
            )

        # Character-specific eyewear (`celinker` monocle loupe, `sprocket` monocle + chain)
        if agent_id in ("celinker", "sprocket") and pass_mode != "rough_blue":
            lx_m = int(round(face_cx + eye_sep))
            ly_m = int(round(face_cy - 5.0 * scale * sy))
            r_loupe = int(round(12.5 * scale))
            cv2.circle(canvas, (lx_m, ly_m), r_loupe, style["accent_bgr"], max(2, ink_w), cv2.LINE_AA)
            cv2.circle(canvas, (lx_m, ly_m), r_loupe, ink_col, max(2, ink_w - 1), cv2.LINE_AA)
            if agent_id == "celinker":
                cv2.circle(canvas, (lx_m, ly_m), int(round(r_loupe * 0.72)), ink_col, max(1, ink_w - 1), cv2.LINE_AA)
            else:
                cv2.line(canvas, (lx_m - r_loupe, ly_m + 3), (lx_m - r_loupe - int(4 * scale), ly_m + int(24 * scale)), ink_col, max(1, ink_w - 1), cv2.LINE_AA)

        # Rounded C-Curve Cartoon Nose (`agent_nib_keys.jpg`)
        nx_c = int(round(face_cx + (4.0 if h_sin >= 0 else -4.0) * scale))
        ny_c = int(round(face_cy + 3.5 * scale))
        n_start, n_end = (-70, 110) if h_sin >= 0 else (70, 250)
        cv2.ellipse(canvas, (nx_c, ny_c), (int(round(6.2 * scale)), int(round(5.0 * scale))), 0, n_start, n_end, ink_col, ink_w, cv2.LINE_AA)

        # Expressive Mouth (`smile`, `laugh`, `focus`, `o_mouth`)
        mx_c = int(round(face_cx))
        my_c = int(round(face_cy + 14.5 * scale * sy))
        if expression == "laugh":
            mw = int(round(14.5 * scale * sx))
            mh = int(round(10.5 * scale * sy))
            cv2.ellipse(canvas, (mx_c, my_c), (mw, mh), 0, 0, 180, ink_col, -1, cv2.LINE_AA)
            if pass_mode == "gouache_cel":
                cv2.ellipse(canvas, (mx_c, my_c + int(3 * scale)), (int(mw * 0.65), int(mh * 0.55)), 0, 0, 180, BGR_TONGUE, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (mx_c, my_c), (mw, mh), 0, 0, 180, ink_col, ink_w, cv2.LINE_AA)
            cv2.line(canvas, (mx_c - mw, my_c), (mx_c + mw, my_c), ink_col, ink_w, cv2.LINE_AA)
        elif expression == "focus":
            # Concentration smile with cute tongue tip sticking out to the side (`nib_inking_keys.jpg` Q1!)
            cv2.ellipse(canvas, (mx_c, my_c), (int(round(12.0 * scale)), int(round(5.5 * scale))), 0, 15, 165, ink_col, ink_w, cv2.LINE_AA)
            tx_c = mx_c + int(round(11.0 * scale))
            ty_c = my_c + int(round(1.5 * scale))
            t_r = int(round(5.2 * scale))
            if pass_mode == "gouache_cel":
                cv2.circle(canvas, (tx_c, ty_c), t_r, BGR_TONGUE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (tx_c, ty_c), t_r, ink_col, ink_w, cv2.LINE_AA)
        elif expression == "idea":
            cv2.ellipse(canvas, (mx_c, my_c - int(2 * scale)), (int(round(14.0 * scale)), int(round(7.5 * scale))), 0, 10, 170, ink_col, ink_w, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (mx_c, my_c - int(2 * scale)), (int(round(12.5 * scale)), int(round(6.0 * scale))), 0, 15, 165, ink_col, ink_w, cv2.LINE_AA)

    # 4D. Signature Studio Headgear (Rigidly Attached to Cranium Top at `3.0H`)
    crown_base_y = h_cy - int(round(24.0 * scale * sy))
    crown_tip_y = crown_base_y - int(round(62.0 * scale * sy))

    if agent_id == "nib":
        # Iconic 3-Pointed Fountain-Pen Nib Crown Hat (`agent_nib_keys.jpg`!)
        w_base = 41.0 * scale * sx
        w_flare = 49.0 * scale * sx
        mid_shift = h_sin * 10.0 * scale * sx
        side_y = crown_tip_y + 24.0 * scale * sy
        valley_y = crown_tip_y + 38.0 * scale * sy

        crown_pts = np.array([
            [h_cx - w_base, crown_base_y + 4.0 * scale],
            [h_cx - w_flare, side_y],
            [h_cx - w_base * 0.42 + mid_shift * 0.5, valley_y],
            [h_cx + mid_shift, crown_tip_y],
            [h_cx + w_base * 0.42 + mid_shift * 0.5, valley_y],
            [h_cx + w_flare, side_y],
            [h_cx + w_base, crown_base_y + 4.0 * scale],
            [h_cx, crown_base_y - 3.0 * scale],
        ], dtype=np.int32)
        _draw_outlined_poly(canvas, crown_pts, style["hat_bgr"], ink_col, ink_w, pass_mode)

        # Central fountain-pen nib slit + circular breather hole (`O`)
        if not h_back:
            hole_x = int(round(h_cx + mid_shift * 0.72))
            hole_y = int(round(crown_tip_y + 39.0 * scale * sy))
            hole_r = max(3, int(round(5.2 * scale)))
            cv2.line(canvas, (int(round(h_cx + mid_shift)), crown_tip_y + 2), (hole_x, hole_y - hole_r), ink_col, ink_w, cv2.LINE_AA)
            if pass_mode == "gouache_cel":
                cv2.circle(canvas, (hole_x, hole_y), hole_r, style["accent_bgr"], -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (hole_x, hole_y), hole_r, BGR_WHITE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (hole_x, hole_y), hole_r, ink_col, ink_w, cv2.LINE_AA)

    elif agent_id == "blueline":
        # Animator Visor Cap + Tucked Non-Photo Blue Pencil over ear (`agent_blueline_keys.jpg`)
        dome_pts = np.array([
            [h_cx - rx_head * 0.92, crown_base_y + 4 * scale],
            [h_cx - rx_head * 0.75, crown_tip_y + 18 * scale],
            [h_cx + rx_head * 0.75, crown_tip_y + 18 * scale],
            [h_cx + rx_head * 0.92, crown_base_y + 4 * scale],
        ], dtype=np.int32)
        _draw_outlined_poly(canvas, dome_pts, style["hat_bgr"], ink_col, ink_w, pass_mode)
        brim_dir = 1.0 if h_sin >= 0 else -1.0
        brim_pts = np.array([
            [h_cx - rx_head * 0.85, crown_base_y + 4 * scale],
            [h_cx + brim_dir * rx_head * 1.25, crown_base_y - 4 * scale],
            [h_cx + rx_head * 0.85, crown_base_y + 8 * scale],
        ], dtype=np.int32)
        _draw_outlined_poly(canvas, brim_pts, style["hat_bgr"], ink_col, ink_w, pass_mode)
        # Pencil tucked behind ear
        p_c = _capsule_poly((h_cx + rx_head * 0.35, crown_base_y - 4 * scale), (h_cx + rx_head * 1.08, crown_base_y + 14 * scale), 3.2 * scale, 3.2 * scale)
        _draw_outlined_poly(canvas, p_c, BGR_NONPHOTO_BLUE, ink_col, max(2, ink_w - 1), pass_mode)

    elif agent_id == "celinker":
        # Artist Beret with stem (`agent_celinker_keys.jpg`)
        beret_pts = _capsule_poly(
            (h_cx - rx_head * 0.85, crown_base_y - 2 * scale),
            (h_cx + rx_head * 0.95, crown_tip_y + 22 * scale),
            16.0 * scale, 22.0 * scale,
        )
        _draw_outlined_poly(canvas, beret_pts, style["hat_bgr"], ink_col, ink_w, pass_mode)
        cv2.line(canvas, (h_cx - int(8 * scale), int(crown_tip_y + 10 * scale)), (h_cx - int(16 * scale), int(crown_tip_y + 2 * scale)), ink_col, ink_w, cv2.LINE_AA)

    elif agent_id == "sprocket":
        # Visor Cap topped with Twin 35mm Film-Reel Ears (`agent_sprocket_keys.jpg`)
        for rs in (-1, 1):
            rx_r = int(round(h_cx + rs * 32.0 * scale * sx))
            ry_r = int(round(crown_tip_y + 22.0 * scale * sy))
            rr = int(round(20.0 * scale))
            cv2.circle(canvas, (rx_r, ry_r), rr + ink_w, ink_col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (rx_r, ry_r), rr, BGR_WHITE if pass_mode == "cleanup_ink" else style["hat_bgr"], -1, cv2.LINE_AA)
            for hi in range(5):
                ha = hi * (2.0 * math.pi / 5.0)
                hx_h = int(round(rx_r + math.cos(ha) * rr * 0.55))
                hy_h = int(round(ry_r + math.sin(ha) * rr * 0.55))
                cv2.circle(canvas, (hx_h, hy_h), max(2, int(round(3.8 * scale))), BGR_WHITE, -1, cv2.LINE_AA)
                cv2.circle(canvas, (hx_h, hy_h), max(2, int(round(3.8 * scale))), ink_col, max(1, ink_w - 1), cv2.LINE_AA)
        dome_pts = _capsule_poly((h_cx - rx_head * 0.78, crown_base_y + 2 * scale), (h_cx + rx_head * 0.78, crown_base_y + 2 * scale), 12.0 * scale, 12.0 * scale)
        _draw_outlined_poly(canvas, dome_pts, style["hat_bgr"], ink_col, ink_w, pass_mode)

    elif agent_id == "chime":
        # Studio Headphones + Forehead Stopwatch Dial (`agent_chime_keys.jpg`)
        cv2.ellipse(canvas, (h_cx, h_cy - int(4 * scale)), (rx_head + int(9 * scale), ry_head + int(12 * scale)), 0, 185, 355, ink_col, ink_w + 3, cv2.LINE_AA)
        for cs in (-1, 1):
            cx_cup = h_cx + int(round(cs * (rx_head + 6 * scale)))
            cy_cup = h_cy + int(round(4 * scale))
            cup_poly = _capsule_poly((cx_cup, cy_cup - 10 * scale), (cx_cup, cy_cup + 10 * scale), 9.5 * scale, 9.5 * scale)
            _draw_outlined_poly(canvas, cup_poly, style["hat_bgr"], ink_col, ink_w, pass_mode)
        # Forehead stopwatch dial
        dial_y = crown_base_y - int(round(4 * scale))
        dial_r = int(round(11.5 * scale))
        cv2.circle(canvas, (h_cx, dial_y), dial_r + ink_w, ink_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (h_cx, dial_y), dial_r, BGR_WHITE if pass_mode != "gouache_cel" else style["accent_bgr"], -1, cv2.LINE_AA)
        cv2.line(canvas, (h_cx, dial_y), (h_cx, dial_y - int(dial_r * 0.65)), ink_col, max(1, ink_w - 1), cv2.LINE_AA)
        cv2.line(canvas, (h_cx, dial_y), (h_cx + int(dial_r * 0.55), dial_y + int(dial_r * 0.35)), ink_col, max(1, ink_w - 1), cv2.LINE_AA)

    # ==========================================================================
    # LAYER 5: FRONT ARMS, ROLLED ELBOW CUFFS, HANDS & STUDIO TOOLS
    # ==========================================================================
    l_el, l_wr, l_tip = (0, 0), (0, 0), (0, 0)
    r_el, r_wr, r_tip = (0, 0), (0, 0), (0, 0)
    if not draw_left_behind:
        l_el, l_wr, l_tip = _draw_arm_and_tool(
            canvas, sh_l_pt, lh_target, scale, ink_w, pass_mode, ink_col,
            BGR_WHITE, BGR_SKIN, bend_sign=1.0, tool="none", hand_pose=lh_pose,
            accent_bgr=style["accent_bgr"],
        )
    if not draw_right_behind:
        r_el, r_wr, r_tip = _draw_arm_and_tool(
            canvas, sh_r_pt, rh_target, scale, ink_w, pass_mode, ink_col,
            BGR_WHITE, BGR_SKIN, bend_sign=-1.0, tool=rh_tool, hand_pose=rh_pose,
            accent_bgr=style["accent_bgr"],
        )

    return {
        "head_center": (h_cx, h_cy),
        "hip_center": (hip_cx, hip_cy),
        "lh_wrist": l_wr,
        "rh_wrist": r_wr,
        "rh_tip": r_tip,
    }


# ==============================================================================
# III. SHEET 1: BASIC CHARACTER MODEL, TURNAROUND & ANATOMY SHEET (`Nib`)
# ==============================================================================

def build_sheet_01_nib_model_and_turnaround() -> Path:
    """
    Renders `preview/basic_sheet_01_nib_model_and_turnaround.png` (`2200x1680`):
      - Row 1: 5-View Full-Body Orthographic Turnaround (`Front 0°`, `3/4 Left -35°`,
        `Side Profile -85°`, `3/4 Right +35°`, `Back 180°`) on a continuous
        horizontal animator proportion grid (`0.0H` to `3.0H`).
      - Row 2: 4-Pass Classical Construction Progression + 4 Canonical Expressions.
      - Row 3: Zero-Slop Joint & Prop Anatomy Callouts (closeup studies proving
        watertight cuffs, collar, hand pinch, and shoe cuffs).
    """
    W, H = 2200, 1680
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)

    # Header Banner
    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "STAGE 01 -- FOUNDATIONAL ZERO-DRIFT CHARACTER MODEL SHEET: 'NIB' (LOCKED 3.0-HEAD PROPORTION SYSTEM)",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.88, (240, 248, 255), 2, cv2.LINE_AA,
    )

    # --------------------------------------------------------------------------
    # ROW 1: 5-VIEW ORTHOGRAPHIC FULL-BODY TURNAROUND (`y = 86..650`)
    # --------------------------------------------------------------------------
    r1_x0, r1_y0, r1_x1, r1_y1 = 24, 84, W - 24, 650
    cv2.rectangle(sheet, (r1_x0, r1_y0), (r1_x1, r1_y1), BGR_PAPER_CARD, -1)
    cv2.rectangle(sheet, (r1_x0, r1_y0), (r1_x1, r1_y1), BGR_INK, 2, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "ROW 1: 5-VIEW FULL-BODY ORTHOGRAPHIC TURNAROUND -- CONTINUOUS HORIZONTAL RULER LINES (0.0H CROWN TIP TO 3.0H GROUND PLANE = ZERO DRIFT)",
        (r1_x0 + 18, r1_y0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, BGR_INK, 1, cv2.LINE_AA,
    )

    ground_y = r1_y1 - 48
    unit_h = 152.0
    _draw_proportion_ruler_grid(sheet, r1_x0 + 16, r1_x1 - 16, ground_y, unit_h, show_labels=True)

    turn_views = [
        ("1. FRONT (0 deg)", 0.0, "smile", "fountain_pen"),
        ("2. 3/4 LEFT (-35 deg)", -35.0, "idea", "fountain_pen"),
        ("3. PROFILE (-82 deg)", -82.0, "focus", "fountain_pen"),
        ("4. 3/4 RIGHT (+35 deg)", 35.0, "laugh", "fountain_pen"),
        ("5. BACK (180 deg)", 180.0, "smile", "none"),
    ]
    for idx, (v_label, yaw, expr, tool_v) in enumerate(turn_views):
        cx = int(round(360 + idx * 410))
        # Subtle vertical plumb line (`center of gravity`)
        cv2.line(sheet, (cx, r1_y0 + 44), (cx, ground_y + 12), BGR_BLUE_FAINT, 1, cv2.LINE_AA)
        draw_canonical_character(
            sheet,
            agent_id="nib",
            foot_x=float(cx),
            foot_y=float(ground_y),
            unit_h=unit_h,
            yaw_deg=yaw,
            expression=expr,
            rh_tool=tool_v,
            pass_mode="gouache_cel",
        )
        cv2.putText(sheet, v_label, (cx - 88, ground_y + 32), cv2.FONT_HERSHEY_DUPLEX, 0.50, BGR_INK, 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # ROW 2: 4-PASS CLASSICAL CONSTRUCTION + 4 CANONICAL EXPRESSIONS (`y = 670..1165`)
    # --------------------------------------------------------------------------
    r2_x0, r2_y0, r2_x1, r2_y1 = 24, 670, W - 24, 1165
    cv2.rectangle(sheet, (r2_x0, r2_y0), (r2_x1, r2_y1), BGR_PAPER_CARD, -1)
    cv2.rectangle(sheet, (r2_x0, r2_y0), (r2_x1, r2_y1), BGR_INK, 2, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "ROW 2: 4-PASS CLASSICAL HAND-DRAWN CONSTRUCTION BREAKDOWN (LEFT)  &  4 CANONICAL ACTING EXPRESSIONS (RIGHT)",
        (r2_x0 + 18, r2_y0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, BGR_INK, 1, cv2.LINE_AA,
    )

    g2_y = r2_y1 - 44
    u2_h = 128.0
    _draw_proportion_ruler_grid(sheet, r2_x0 + 16, r2_x1 - 16, g2_y, u2_h, show_labels=False)

    row2_items = [
        ("PASS 1: ROUGH BLUE", "rough_blue", "smile", 0.0, 0.0),
        ("PASS 2: BLUE + INK", "overlay_both", "smile", 12.0, 0.0),
        ("PASS 3: CLEANUP INK", "cleanup_ink", "smile", 12.0, 0.0),
        ("PASS 4: GOUACHE CEL", "gouache_cel", "smile", 12.0, 0.0),
        ("EXPR A: BLINK / CALM", "gouache_cel", "smile", -15.0, 0.92),
        ("EXPR B: IDEA LIFT", "gouache_cel", "idea", 18.0, 0.0),
        ("EXPR C: TONGUE FOCUS", "gouache_cel", "focus", -22.0, 0.0),
        ("EXPR D: OPEN LAUGH", "gouache_cel", "laugh", 20.0, 0.0),
    ]
    for idx, (lbl, pmode, expr, h_yaw, blk) in enumerate(row2_items):
        cx = int(round(155 + idx * 268))
        if pmode == "overlay_both":
            draw_canonical_character(
                sheet, agent_id="nib", foot_x=float(cx), foot_y=float(g2_y),
                unit_h=u2_h, yaw_deg=0.0, head_yaw_deg=h_yaw, expression=expr,
                blink=blk, pass_mode="cleanup_ink",
            )
            draw_canonical_character(
                sheet, agent_id="nib", foot_x=float(cx), foot_y=float(g2_y),
                unit_h=u2_h, yaw_deg=0.0, head_yaw_deg=h_yaw, expression=expr,
                blink=blk, pass_mode="rough_blue",
            )
        else:
            draw_canonical_character(
                sheet, agent_id="nib", foot_x=float(cx), foot_y=float(g2_y),
                unit_h=u2_h, yaw_deg=0.0, head_yaw_deg=h_yaw, expression=expr,
                blink=blk, pass_mode=pmode,
            )
        cv2.putText(sheet, lbl, (cx - 88, g2_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.44, BGR_INK, 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # ROW 3: ZERO-SLOP ANATOMICAL JOINT & PROP CALLOUTS (`y = 1185..1656`)
    # --------------------------------------------------------------------------
    r3_x0, r3_y0, r3_x1, r3_y1 = 24, 1185, W - 24, 1656
    cv2.rectangle(sheet, (r3_x0, r3_y0), (r3_x1, r3_y1), BGR_PAPER_CARD, -1)
    cv2.rectangle(sheet, (r3_x0, r3_y0), (r3_x1, r3_y1), BGR_INK, 2, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "ROW 3: ZERO-SLOP ANATOMICAL JOINT & PROP CONSTRUCTION CALLOUTS (WATERTIGHT CUFFS, COLLAR, HAND PINCH & ANKLE CUFFS)",
        (r3_x0 + 18, r3_y0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, BGR_INK, 1, cv2.LINE_AA,
    )

    callout_titles = [
        "1. CROWN HAT & BREATHER HOLE",
        "2. V-COLLAR & APRON STRAPS",
        "3. ROLLED CUFF & PEN PINCH",
        "4. OPEN WAVE HAND STUDY",
        "5. CUFFED TROUSER & SHOE",
    ]
    cw_box = 412
    ch_box = 395
    for ci, ctitle in enumerate(callout_titles):
        bx0 = r3_x0 + 18 + ci * (cw_box + 14)
        by0 = r3_y0 + 44
        bx1 = bx0 + cw_box
        by1 = by0 + ch_box
        cv2.rectangle(sheet, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        cv2.rectangle(sheet, (bx0, by0), (bx1, by1), BGR_INK, 2, cv2.LINE_AA)

        sub = sheet[by0 + 4:by1 - 36, bx0 + 4:bx1 - 4]
        sh, sw = sub.shape[:2]
        if ci == 0:
            # Closeup of Head & Crown Hat
            draw_canonical_character(
                sub, agent_id="nib", foot_x=sw * 0.5, foot_y=sh + 340.0,
                unit_h=210.0, yaw_deg=15.0, expression="smile", pass_mode="gouache_cel",
            )
        elif ci == 1:
            # Closeup of V-Collar, Apron Bib & Shoulders
            draw_canonical_character(
                sub, agent_id="nib", foot_x=sw * 0.5, foot_y=sh + 190.0,
                unit_h=210.0, yaw_deg=0.0, expression="smile", pass_mode="gouache_cel",
            )
        elif ci == 2:
            # Closeup of Rolled Elbow Cuff & Fountain-Nib Pen Pinch
            _draw_arm_and_tool(
                sub, (75.0, 95.0), (245.0, 205.0), scale=2.15, ink_w=6,
                pass_mode="gouache_cel", ink_col=BGR_INK, shirt_fill=BGR_WHITE,
                skin_fill=BGR_SKIN, bend_sign=-1.0, tool="fountain_pen", hand_pose="pinch",
            )
        elif ci == 3:
            # Closeup of Open Expressive 4-Finger + Thumb Hand
            _draw_arm_and_tool(
                sub, (85.0, 245.0), (255.0, 120.0), scale=2.15, ink_w=6,
                pass_mode="gouache_cel", ink_col=BGR_INK, shirt_fill=BGR_WHITE,
                skin_fill=BGR_SKIN, bend_sign=1.0, tool="none", hand_pose="open",
            )
        elif ci == 4:
            # Closeup of Apron Hem -> Trouser Legs -> Rolled Ankle Cuffs -> Animator Shoes
            draw_canonical_character(
                sub, agent_id="nib", foot_x=sw * 0.5, foot_y=sh - 24.0,
                unit_h=220.0, yaw_deg=18.0, expression="smile", pass_mode="gouache_cel",
            )

        cv2.putText(sheet, ctitle, (bx0 + 14, by1 - 12), cv2.FONT_HERSHEY_DUPLEX, 0.48, BGR_INK, 1, cv2.LINE_AA)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEW_DIR / "basic_sheet_01_nib_model_and_turnaround.png"
    cv2.imwrite(str(out_path), sheet)
    shutil.copy2(out_path, ARTIFACT_DIR / out_path.name)
    return out_path


# ==============================================================================
# IV. SHEET 2: 5-CHARACTER ORTHOGRAPHIC LINEUP ACROSS ALL 3 PASSES
# ==============================================================================

def build_sheet_02_5agents_lineup_proportions() -> Path:
    """
    Renders `preview/basic_sheet_02_5agents_lineup_proportions.png` (`2200x1620`):
      - All 5 Studio Characters (`Nib`, `Blue-Line`, `Cel-Inker`, `Sprocket`, `Chime`)
        standing on the exact same continuous `0.0H..3.0H` proportion ruler grid
        across 3 rows (`Pass 1 Rough Blue`, `Pass 2 Clean Speedball Ink`, `Pass 3 Gouache Cel`).
    """
    W, H = 2200, 1620
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "STAGE 01B -- 5-CHARACTER ORTHOGRAPHIC STUDIO LINEUP (100% LOCKED 3-HEAD PROPORTION GRID ACROSS ALL 3 PASSES)",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.86, (240, 248, 255), 2, cv2.LINE_AA,
    )

    passes = [
        ("ROW 1 -- PASS 1: NON-PHOTO BLUE VOLUMETRIC CONSTRUCTION & RULER GRID", "rough_blue"),
        ("ROW 2 -- PASS 2: SPEEDBALL B-5 INDIA INK CLEANUP CONTOURS", "cleanup_ink"),
        ("ROW 3 -- PASS 3: WATERTIGHT OPAQUE GOUACHE CEL & SIGNATURE STUDIO EMBLEMS", "gouache_cel"),
    ]
    agents = [
        ("nib", 10.0, "smile"),
        ("blueline", -12.0, "idea"),
        ("celinker", 12.0, "smile"),
        ("sprocket", -10.0, "focus"),
        ("chime", 8.0, "laugh"),
    ]

    for r_idx, (r_title, p_mode) in enumerate(passes):
        ry0 = 84 + r_idx * 504
        ry1 = ry0 + 488
        cv2.rectangle(sheet, (24, ry0), (W - 24, ry1), BGR_PAPER_CARD, -1)
        cv2.rectangle(sheet, (24, ry0), (W - 24, ry1), BGR_INK, 2, cv2.LINE_AA)
        cv2.putText(sheet, r_title, (42, ry0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, BGR_INK, 1, cv2.LINE_AA)

        g_y = ry1 - 42
        u_h = 132.0
        _draw_proportion_ruler_grid(sheet, 40, W - 40, g_y, u_h, show_labels=True)

        for a_idx, (aid, yaw_a, expr_a) in enumerate(agents):
            cx = int(round(360 + a_idx * 410))
            sh_x = cx + 52.0
            sh_y = g_y - 1.95 * u_h
            draw_canonical_character(
                sheet,
                agent_id=aid,
                foot_x=float(cx),
                foot_y=float(g_y),
                unit_h=u_h,
                yaw_deg=yaw_a,
                expression=expr_a,
                rh_target=(sh_x + 36.0, sh_y - 18.0),
                lh_target=(cx - 64.0, g_y - 1.15 * u_h),
                lh_pose="open",
                pass_mode=p_mode,
            )
            lbl = AGENT_STYLES[aid]["name"]
            cv2.putText(sheet, lbl, (cx - 105, g_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.48, BGR_INK, 1, cv2.LINE_AA)

    out_path = PREVIEW_DIR / "basic_sheet_02_5agents_lineup_proportions.png"
    cv2.imwrite(str(out_path), sheet)
    shutil.copy2(out_path, ARTIFACT_DIR / out_path.name)
    return out_path


# ==============================================================================
# V. STAGE 02 -> STAGE 06: 6-TIER PROGRESSIVE ANIMATION TEST CURRICULUM
# ==============================================================================

TIER_SPECS = [
    {
        "tier": 1,
        "slug": "tier_01_head_turn_and_blink",
        "title": "TIER 1 (BASIC TEST 01): LOCKED-BODY HEAD TURN, EYE BLINK & EXPRESSION SHIFT",
        "subtitle": "Feet, legs, apron, torso & arms 100% stationary (0.0px drift) while head rotates -30 deg -> 0 deg (Blink) -> +30 deg (Smile)",
    },
    {
        "tier": 2,
        "slug": "tier_02_squash_and_stretch_hop",
        "title": "TIER 2 (BASIC TEST 02): CONSTANT-VOLUME SQUASH & STRETCH VERTICAL HOP (W * H = 1.00)",
        "subtitle": "Strict volume conservation: Rest (1.00x) -> Anticipation Squat (0.82x H, 1.22x W) -> Launch Stretch (1.16x H, 0.86x W) -> Settle",
    },
    {
        "tier": 3,
        "slug": "tier_03_single_arm_pendulum_arc",
        "title": "TIER 3 (BASIC TEST 03): STATIONARY-BODY SINGLE-ARM PENDULUM SWEEP & WRIST WHIP ARC",
        "subtitle": "Body & head 100% locked while right arm + fountain-nib pen sweeps along a circular trajectory arc from hip rest to high flourish",
    },
    {
        "tier": 4,
        "slug": "tier_04_bilateral_paper_and_pen",
        "title": "TIER 4 (INTERMEDIATE TEST 04): BILATERAL COORDINATION -- LEFT PAPER PINCH & RIGHT INKING STROKE",
        "subtitle": "Grounded torso with slight contrapposto lean as left hand lifts a sheet corner and right hand pulls a smooth S-curve ink stroke",
    },
    {
        "tier": 5,
        "slug": "tier_05_weight_shift_walk_cycle",
        "title": "TIER 5 (ADVANCED TEST 05): GROUNDED WEIGHT SHIFT & 4-BEAT WALK CYCLE ON FIXED GROUND LINE",
        "subtitle": "Classic Contact -> Down -> Passing -> Up -> Contact walk cycle with clear hip/head wave and opposite arm-leg swing",
    },
    {
        "tier": 6,
        "slug": "tier_06_compound_acting_shot",
        "title": "TIER 6 (MASTER TEST 06): COMPOUND ACTING PERFORMANCE COMBINING TIERS 1 THROUGH 5",
        "subtitle": "1. Spots Idea (Tier 1) -> 2. Anticipation Crouch (Tier 2) -> 3. Sweeps Pen Arc (Tier 3/4) -> 4. Joyful Step & Open Laugh (Tier 5)",
    },
]


def evaluate_tier_pose(tier_num: int, u: float, cx: float, ground_y: float, unit_h: float) -> dict:
    """
    Evaluates the exact deterministic pose dictionary for `tier_num in 1..6` at
    normalized phase `u in [0.0, 1.0]`.
    """
    scale = unit_h / 120.0
    # Smooth ease-in-out S-curve `s`
    s = u * u * (3.0 - 2.0 * u)
    ang = u * 2.0 * math.pi

    pose = {
        "agent_id": "nib",
        "foot_x": cx,
        "foot_y": ground_y,
        "unit_h": unit_h,
        "yaw_deg": 0.0,
        "head_yaw_deg": 0.0,
        "head_pitch_deg": 0.0,
        "head_roll_deg": 0.0,
        "squash_stretch": 0.0,
        "hip_offset_x": 0.0,
        "hip_offset_y": 0.0,
        "torso_lean_deg": 0.0,
        "lh_target": (cx - 58.0 * scale, ground_y - 102.0 * scale),
        "rh_target": (cx + 58.0 * scale, ground_y - 112.0 * scale),
        "lh_pose": "open",
        "rh_pose": "pinch",
        "rh_tool": "fountain_pen",
        "l_foot_offset": (0.0, 0.0),
        "r_foot_offset": (0.0, 0.0),
        "blink": 0.0,
        "expression": "smile",
    }

    if tier_num == 1:
        # TIER 1: 100% Locked Body — ONLY Head Yaw, Pitch, Blink & Expression change!
        pose["head_yaw_deg"] = -32.0 + 64.0 * s
        pose["head_pitch_deg"] = 10.0 * math.sin(u * math.pi)  # Slight dip at midpoint (`Breakdown`)
        pose["head_roll_deg"] = -5.0 * math.sin(u * math.pi)
        # Eye blink centered right on the breakdown (`u in [0.35, 0.60]`)
        pose["blink"] = max(0.0, 1.0 - abs(u - 0.46) / 0.14)
        pose["expression"] = "smile" if u < 0.52 else ("idea" if u < 0.78 else "laugh")

    elif tier_num == 2:
        # TIER 2: Constant-Volume Squash & Stretch Vertical Hop
        # u=0.0: Rest -> u=0.25: Deep Squash Crouch -> u=0.55: Airborne Stretch Apex -> u=0.82: Landing Cushion -> u=1.0: Rest
        if u < 0.28:
            k = math.sin((u / 0.28) * math.pi * 0.5)
            sq = -0.18 * k
            jump_y = 0.0
        elif u < 0.72:
            hu = (u - 0.28) / 0.44
            jump_y = -28.0 * math.sin(hu * math.pi)
            sq = 0.14 * math.sin(hu * math.pi)
        else:
            lu = (u - 0.72) / 0.28
            jump_y = 0.0
            sq = -0.12 * math.sin(lu * math.pi)

        pose["squash_stretch"] = sq
        pose["hip_offset_y"] = jump_y + (18.0 * (-sq) if sq < 0 else 0.0)
        pose["l_foot_offset"] = (0.0, jump_y * scale)
        pose["r_foot_offset"] = (0.0, jump_y * scale)
        arm_lift = -jump_y * 1.3 + sq * 85.0
        pose["lh_target"] = (cx - (62.0 - sq * 30.0) * scale, ground_y - (105.0 + arm_lift) * scale)
        pose["rh_target"] = (cx + (62.0 - sq * 30.0) * scale, ground_y - (112.0 + arm_lift) * scale)
        pose["expression"] = "focus" if sq < -0.05 else ("laugh" if jump_y < -15.0 else "smile")

    elif tier_num == 3:
        # TIER 3: Stationary Body — Single-Arm Circular Pendulum Sweep & Wrist Whip Arc
        # Right hand traces a clean circular arc of radius `68 * scale` around right shoulder `(cx + 34*scale, ground_y - 216*scale)`
        sh_rx = cx + 34.0 * scale
        sh_ry = ground_y - 216.0 * scale
        arc_ang = math.radians(78.0 - 142.0 * s)  # Sweeps from +78 deg (down at side) to -64 deg (high overhead flourish)
        r_arc = 58.0 * scale
        pose["rh_target"] = (sh_rx + r_arc * math.cos(arc_ang), sh_ry + r_arc * math.sin(arc_ang))
        pose["head_yaw_deg"] = 14.0 * s
        pose["expression"] = "smile" if u < 0.6 else "idea"

    elif tier_num == 4:
        # TIER 4: Bilateral Coordination — Left-Hand Paper Pinch & Right-Hand Inking Stroke
        pose["yaw_deg"] = 12.0
        pose["torso_lean_deg"] = 4.0 + 5.0 * math.sin(u * math.pi)
        pose["head_yaw_deg"] = 15.0 - 8.0 * math.cos(u * math.pi)
        pose["head_pitch_deg"] = 8.0
        # Left hand lifts paper sheet corner upward
        pose["lh_target"] = (cx - 48.0 * scale, ground_y - (128.0 + 36.0 * math.sin(u * math.pi)) * scale)
        pose["lh_pose"] = "pinch"
        # Right hand pulls a figure-8 calligraphic stroke across the desk plane
        pose["rh_target"] = (
            cx + (28.0 + 34.0 * s) * scale,
            ground_y - (142.0 - 16.0 * math.sin(u * 2.0 * math.pi)) * scale,
        )
        pose["expression"] = "focus" if 0.2 < u < 0.8 else "smile"

    elif tier_num == 5:
        # TIER 5: Grounded Weight Shift & 4-Beat Walk Cycle
        pose["yaw_deg"] = 28.0
        pose["head_yaw_deg"] = 24.0
        pose["torso_lean_deg"] = 6.0
        stride = 26.0
        lift = 16.0
        lx_off = stride * math.sin(ang)
        ly_off = -lift * max(0.0, math.cos(ang))
        rx_off = stride * math.sin(ang + math.pi)
        ry_off = -lift * max(0.0, math.cos(ang + math.pi))
        pose["l_foot_offset"] = (lx_off, ly_off)
        pose["r_foot_offset"] = (rx_off, ry_off)
        # Double-bounce hip bob (`2 * ang`)
        pose["hip_offset_y"] = 6.5 * abs(math.sin(ang)) - 3.0
        # Counter-swinging arms
        pose["lh_target"] = (cx - (32.0 - 28.0 * math.sin(ang)) * scale, ground_y - (112.0 + 14.0 * math.cos(ang)) * scale)
        pose["rh_target"] = (cx + (38.0 + 28.0 * math.sin(ang)) * scale, ground_y - (118.0 - 14.0 * math.cos(ang)) * scale)
        pose["expression"] = "smile"

    else:
        # TIER 6: Compound Acting Performance Combining Tiers 1–5
        # Phase 1 (0..0.28): Head turns & blinks (Tier 1)
        # Phase 2 (0.28..0.52): Anticipation crouch squash (Tier 2)
        # Phase 3 (0.52..0.80): Upward stretch + sweeping right-arm pen flourish (Tier 3/4)
        # Phase 4 (0.80..1.00): Joyful step settle & open laugh (Tier 5)
        if u < 0.28:
            p = u / 0.28
            pose["head_yaw_deg"] = -25.0 + 45.0 * p
            pose["blink"] = max(0.0, 1.0 - abs(p - 0.5) / 0.35)
            pose["expression"] = "smile"
        elif u < 0.52:
            p = (u - 0.28) / 0.24
            sq_k = math.sin(p * math.pi)
            pose["squash_stretch"] = -0.15 * sq_k
            pose["hip_offset_y"] = 16.0 * sq_k
            pose["torso_lean_deg"] = -8.0 * sq_k
            pose["head_yaw_deg"] = 20.0
            pose["lh_target"] = (cx - 64.0 * scale, ground_y - (95.0 - 18.0 * sq_k) * scale)
            pose["rh_target"] = (cx + 52.0 * scale, ground_y - (95.0 - 18.0 * sq_k) * scale)
            pose["expression"] = "focus"
        elif u < 0.82:
            p = (u - 0.52) / 0.30
            sp = p * p * (3.0 - 2.0 * p)
            pose["squash_stretch"] = 0.12 * math.sin(p * math.pi)
            pose["yaw_deg"] = 18.0 * sp
            pose["head_yaw_deg"] = 22.0
            pose["torso_lean_deg"] = 8.0 * sp
            pose["r_foot_offset"] = (18.0 * sp, -12.0 * math.sin(p * math.pi))
            pose["lh_target"] = (cx - 62.0 * scale, ground_y - (135.0 + 35.0 * sp) * scale)
            pose["rh_target"] = (cx + (64.0 + 18.0 * sp) * scale, ground_y - (145.0 + 58.0 * sp) * scale)
            pose["expression"] = "idea"
        else:
            p = (u - 0.82) / 0.18
            pose["yaw_deg"] = 18.0
            pose["head_yaw_deg"] = 18.0
            pose["torso_lean_deg"] = 5.0
            pose["r_foot_offset"] = (18.0, 0.0)
            pose["lh_target"] = (cx - 58.0 * scale, ground_y - 162.0 * scale)
            pose["rh_target"] = (cx + 76.0 * scale, ground_y - 196.0 * scale)
            pose["expression"] = "laugh"

    return pose


def build_sheet_03_progressive_tests_and_video() -> tuple:
    """
    Renders:
      1. `preview/basic_tests_tier1_to_tier6_flipbooks.png` (`2240x2160`):
         All 6 Progressive Animation Tiers (`8 frames` per row on continuous
         horizontal proportion ruler grids + crimson motion-arc overlays).
      2. `bts_films/basic_tests_tier1_to_tier6_showcase.mp4` & `.gif` (`24 fps`
         animated pencil test loop showing all 6 progressive tiers simultaneously!).
    """
    W, H = 2240, 2160
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "STAGE 02 -> STAGE 06: 6-TIER PROGRESSIVE ANIMATION PENCIL-TEST SUITE (BUILDING UP SLOWLY WITH ZERO DRIFT)",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.86, (240, 248, 255), 2, cv2.LINE_AA,
    )

    num_cols = 8
    row_h = 338
    unit_h = 84.0

    for t_idx, spec in enumerate(TIER_SPECS):
        tier_num = spec["tier"]
        ry0 = 80 + t_idx * (row_h + 6)
        ry1 = ry0 + row_h
        cv2.rectangle(sheet, (20, ry0), (W - 20, ry1), BGR_PAPER_CARD, -1)
        cv2.rectangle(sheet, (20, ry0), (W - 20, ry1), BGR_INK, 2, cv2.LINE_AA)

        cv2.putText(sheet, spec["title"], (36, ry0 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.54, BGR_INK, 1, cv2.LINE_AA)
        cv2.putText(sheet, spec["subtitle"], (36, ry0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

        g_y = ry1 - 26
        _draw_proportion_ruler_grid(sheet, 32, W - 32, g_y, unit_h, show_labels=(t_idx == 0))

        # Precompute trajectory points for crimson motion arc overlay
        for f_idx in range(num_cols):
            u = f_idx / float(num_cols - 1)
            cx = 175.0 + f_idx * 268.0

            # Draw faint onion-skin of previous frame (`f_idx > 0`) to prove smooth spacing
            if f_idx > 0:
                u_prev = (f_idx - 1) / float(num_cols - 1)
                prev_pose = evaluate_tier_pose(tier_num, u_prev, cx, float(g_y), unit_h)
                prev_pose["pass_mode"] = "rough_blue"
                prev_pose["ink_override"] = BGR_ONION_PREV
                draw_canonical_character(sheet, **prev_pose)

            # Draw current frame
            cur_pose = evaluate_tier_pose(tier_num, u, cx, float(g_y), unit_h)
            # Show Pass 1/2/3 progression across the flipbook or full Gouache Cel with construction arcs
            info = draw_canonical_character(sheet, **cur_pose)

            # For Tier 3 & Tier 6, plot the circular right-hand pen trajectory arc right on the cell!
            if tier_num in (3, 4, 6):
                arc_pts = []
                for k_s in range(13):
                    uk = k_s / 12.0
                    pk = evaluate_tier_pose(tier_num, uk, cx, float(g_y), unit_h)
                    arc_pts.append((int(round(pk["rh_target"][0])), int(round(pk["rh_target"][1]))))
                for ai in range(len(arc_pts) - 1):
                    cv2.line(sheet, arc_pts[ai], arc_pts[ai + 1], BGR_ARC_RED, 1, cv2.LINE_AA)
                cv2.circle(sheet, (int(round(cur_pose["rh_target"][0])), int(round(cur_pose["rh_target"][1]))), 4, BGR_ARC_RED, -1, cv2.LINE_AA)

            role_tag = "KEYSTART" if f_idx == 0 else ("BREAKDOWN" if f_idx == 3 else ("KEYEND" if f_idx == num_cols - 1 else f"INB F{f_idx + 1:02d}"))
            cv2.putText(
                sheet,
                f"F{f_idx + 1:02d} [{role_tag}]",
                (int(cx - 58), g_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                BGR_ARC_RED if "KEY" in role_tag else BGR_INK,
                1, cv2.LINE_AA,
            )

    sheet_path = PREVIEW_DIR / "basic_tests_tier1_to_tier6_flipbooks.png"
    cv2.imwrite(str(sheet_path), sheet)
    shutil.copy2(sheet_path, ARTIFACT_DIR / sheet_path.name)

    # --------------------------------------------------------------------------
    # Render 24fps Animated Showcase Video (3x2 Grid of All 6 Progressive Tests)
    # --------------------------------------------------------------------------
    FILMS_DIR.mkdir(parents=True, exist_ok=True)
    mp4_path = FILMS_DIR / "basic_tests_tier1_to_tier6_showcase.mp4"
    gif_path = FILMS_DIR / "basic_tests_tier1_to_tier6_showcase.gif"

    vw, vh = 1440, 960
    total_frames = 72  # 3.0 seconds at 24fps (smooth back-and-forth ping-pong or cycle)
    tmp_dir = FILMS_DIR / "_tmp_basic_tests"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for fi in range(total_frames):
        # Smooth ping-pong phase `u in [0, 1]` for Tiers 1-4 & 6, continuous cycle for Tier 5 walk
        phase = fi / float(total_frames)
        u_ping = 0.5 - 0.5 * math.cos(phase * 2.0 * math.pi)
        u_walk = (phase * 2.0) % 1.0

        frame = np.full((vh, vw, 3), BGR_PAPER, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (vw, 48), (34, 42, 54), -1)
        cv2.putText(
            frame,
            f"FOUNDATIONAL ZERO-DRIFT PENCIL TESTS (TIERS 1-6) | 24 FPS ON ONES | FRAME {fi + 1:02d}/{total_frames:02d}",
            (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.66, (240, 248, 255), 1, cv2.LINE_AA,
        )

        cw, ch = 460, 430
        for idx, spec in enumerate(TIER_SPECS):
            r = idx // 3
            c = idx % 3
            x0 = 16 + c * (cw + 14)
            y0 = 60 + r * (ch + 14)
            x1, y1 = x0 + cw, y0 + ch
            cv2.rectangle(frame, (x0, y0), (x1, y1), BGR_PAPER_CARD, -1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), BGR_INK, 2, cv2.LINE_AA)

            t_num = spec["tier"]
            u_use = u_walk if t_num == 5 else u_ping
            g_cell_y = y1 - 38
            u_cell_h = 112.0
            _draw_proportion_ruler_grid(frame, x0 + 10, x1 - 10, g_cell_y, u_cell_h, show_labels=False)

            cx_cell = x0 + cw * 0.5
            # Onion skin of slightly earlier phase
            u_prev = max(0.0, u_use - 0.08)
            p_prev = evaluate_tier_pose(t_num, u_prev, cx_cell, float(g_cell_y), u_cell_h)
            p_prev["pass_mode"] = "rough_blue"
            p_prev["ink_override"] = BGR_ONION_PREV
            draw_canonical_character(frame, **p_prev)

            # Current pose
            p_cur = evaluate_tier_pose(t_num, u_use, cx_cell, float(g_cell_y), u_cell_h)
            draw_canonical_character(frame, **p_cur)

            short_t = spec["title"].split(":")[0] + ":" + spec["title"].split(":")[1][:26]
            cv2.putText(frame, short_t, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.44, BGR_INK, 1, cv2.LINE_AA)

        cv2.imwrite(str(tmp_dir / f"f_{fi:04d}.png"), frame)

    subprocess.run([
        "ffmpeg", "-y", "-framerate", "24", "-i", str(tmp_dir / "f_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4_path),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(mp4_path),
        "-vf", "fps=16,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        str(gif_path),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.copy2(gif_path, ARTIFACT_DIR / gif_path.name)

    return sheet_path, mp4_path, gif_path


# ==============================================================================
# VI. SHEET 4: GOOGLE OMNI / VEO START & END KEYFRAME PAIRS FOR BASIC TESTS
# ==============================================================================

OMNI_BASIC_PROMPTS = {
    1: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "The cartoon animator character Nib (wearing his white 3-pointed fountain-pen crown hat "
        "with center slit and brass breather hole, white V-neck collar, terracotta-red overall "
        "apron, rolled elbow cuffs, denim trousers with rolled ankle cuffs, and dark shoes) "
        "keeps his feet, legs, torso, and arms 100% stationary and locked on the horizontal "
        "proportion lines while smoothly rotating ONLY his head from 3/4-left view in the Start "
        "Frame through a mid-turn eye blink to a cheerful 3/4-right smile in the End Frame. "
        "Zero body drift, constant cranium sphere volume, crisp unchanging ink line weight."
    ),
    2: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "The cartoon character Nib performs a classic constant-volume squash-and-stretch hop "
        "in place: starting from a compressed anticipation crouch in the Start Frame (knees bent, "
        "torso squashed wider, arms cocked back) and springing smoothly upward into an airborne "
        "vertical stretch at the End Frame with arms raised cheerfully. Strict volume conservation, "
        "zero character off-model drift."
    ),
    3: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "The cartoon character Nib stands completely grounded with his feet, legs, apron, torso, "
        "and left hand 100% locked in place while sweeping ONLY his right arm and fountain-nib dip "
        "pen along a smooth circular pendulum arc from a low hip rest in the Start Frame up to a "
        "high raised idea flourish in the End Frame, with natural wrist drag and overlapping follow-through."
    ),
    4: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "The cartoon character Nib stands grounded in a 3/4 view while coordinating both hands: "
        "his left hand pinches and lifts an animation sheet upward while his right hand pulls "
        "his fountain-nib dip pen smoothly across the drawing plane from the Start Frame to the "
        "End Frame, with focused concentration and locked body proportions."
    ),
    5: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "The cartoon character Nib performs a clean, grounded weight-shift step forward on the "
        "fixed ground plane from the back-foot contact pose in the Start Frame through a dipped "
        "passing breakdown into the front-foot stride contact pose in the End Frame, with natural "
        "opposite arm-and-leg swing and zero foot sliding."
    ),
    6: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "The cartoon character Nib transitions smoothly from a crouched anticipation pose looking "
        "left in the Start Frame into a triumphant upward-stretched celebration flourish in the "
        "End Frame, sweeping his fountain-nib pen high overhead with a joyful open laugh while "
        "maintaining 100% locked character model proportions."
    ),
}


def build_sheet_04_omni_basic_keyframe_packs() -> Path:
    """
    Exports clean `1024x1024` Start & End Keyframe PNG pairs + `prompt.txt` for each
    of the 6 Progressive Tests (`omni_keyframe_packs/basic_tests/tier_01..tier_06`)
    and renders `preview/omni_basic_tests_start_end_pairs.png`.
    """
    OMNI_BASIC_DIR.mkdir(parents=True, exist_ok=True)
    pairs_for_sheet = []

    for spec in TIER_SPECS:
        t_num = spec["tier"]
        slug = spec["slug"]
        pack_dir = OMNI_BASIC_DIR / slug
        pack_dir.mkdir(parents=True, exist_ok=True)

        # Choose the two most descriptive Start/End extremes (`u_start, u_end`) for each test
        u_start, u_end = (0.18, 0.56) if t_num == 2 else ((0.32, 0.88) if t_num == 6 else (0.0, 1.0))

        for tag, u_val in [("start_frame", u_start), ("end_frame", u_end)]:
            for pass_suffix, p_mode in [("gouache", "gouache_cel"), ("ink", "cleanup_ink")]:
                img = np.full((1024, 1024, 3), BGR_PAPER_CARD, dtype=np.uint8)
                g_y = 910.0
                u_h = 228.0
                pose = evaluate_tier_pose(t_num, u_val, 512.0, g_y, u_h)
                pose["pass_mode"] = p_mode
                draw_canonical_character(img, **pose)
                cv2.imwrite(str(pack_dir / f"{tag}_{pass_suffix}.png"), img)

        prompt_str = OMNI_BASIC_PROMPTS[t_num]
        (pack_dir / "prompt.txt").write_text(prompt_str + "\n", encoding="utf-8")
        meta = {
            "tier": t_num,
            "slug": slug,
            "title": spec["title"],
            "u_start": u_start,
            "u_end": u_end,
            "start_frame_gouache": f"{slug}/start_frame_gouache.png",
            "end_frame_gouache": f"{slug}/end_frame_gouache.png",
            "start_frame_ink": f"{slug}/start_frame_ink.png",
            "end_frame_ink": f"{slug}/end_frame_ink.png",
            "omni_prompt": prompt_str,
        }
        (pack_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        pairs_for_sheet.append((spec, pack_dir / "start_frame_gouache.png", pack_dir / "end_frame_gouache.png", prompt_str))

    # Render Contact Sheet `preview/omni_basic_tests_start_end_pairs.png` (`2100x1680`)
    W, H = 2100, 1680
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)
    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "GOOGLE OMNI / VEO -- ZERO-DRIFT BASIC PENCIL-TEST START & END KEYFRAME PACKS (TIERS 1 TO 6)",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.86, (240, 248, 255), 2, cv2.LINE_AA,
    )

    cw, ch = 1010, 510
    for idx, (spec, p_start, p_end, prompt_txt) in enumerate(pairs_for_sheet):
        r = idx // 2
        c = idx % 2
        x0 = 24 + c * (cw + 28)
        y0 = 84 + r * (ch + 18)
        x1, y1 = x0 + cw, y0 + ch
        cv2.rectangle(sheet, (x0, y0), (x1, y1), BGR_PAPER_CARD, -1)
        cv2.rectangle(sheet, (x0, y0), (x1, y1), BGR_INK, 2, cv2.LINE_AA)

        cv2.putText(sheet, spec["title"][:68], (x0 + 16, y0 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.50, BGR_INK, 1, cv2.LINE_AA)

        img_s = cv2.resize(cv2.imread(str(p_start)), (340, 340), interpolation=cv2.INTER_AREA)
        img_e = cv2.resize(cv2.imread(str(p_end)), (340, 340), interpolation=cv2.INTER_AREA)

        sy_box = y0 + 42
        sx_s = x0 + 36
        sx_e = x0 + 634
        sheet[sy_box:sy_box + 340, sx_s:sx_s + 340] = img_s
        sheet[sy_box:sy_box + 340, sx_e:sx_e + 340] = img_e
        cv2.rectangle(sheet, (sx_s, sy_box), (sx_s + 340, sy_box + 340), BGR_NONPHOTO_BLUE, 2)
        cv2.rectangle(sheet, (sx_e, sy_box), (sx_e + 340, sy_box + 340), BGR_ARC_RED, 2)

        cv2.putText(sheet, "START KEYFRAME (1024x1024)", (sx_s + 32, sy_box + 360), cv2.FONT_HERSHEY_DUPLEX, 0.46, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)
        cv2.putText(sheet, "END KEYFRAME (1024x1024)", (sx_e + 42, sy_box + 360), cv2.FONT_HERSHEY_DUPLEX, 0.46, BGR_ARC_RED, 1, cv2.LINE_AA)

        # Arrow in middle between Start and End
        mid_x = (sx_s + 340 + sx_e) // 2
        mid_y = sy_box + 170
        cv2.arrowedLine(sheet, (mid_x - 85, mid_y), (mid_x + 85, mid_y), BGR_ARC_RED, 4, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(sheet, "GOOGLE OMNI", (mid_x - 58, mid_y - 14), cv2.FONT_HERSHEY_DUPLEX, 0.46, BGR_ARC_RED, 1, cv2.LINE_AA)
        cv2.putText(sheet, "ZERO-DRIFT", (mid_x - 52, mid_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.44, BGR_INK, 1, cv2.LINE_AA)

        # Prompt snippet at bottom of card
        words = prompt_txt.split()
        lines = []
        cur = "PROMPT: "
        for w in words:
            if len(cur) + len(w) + 1 > 118:
                lines.append(cur)
                cur = "  " + w
            else:
                cur += (" " if cur != "PROMPT: " else "") + w
        if cur:
            lines.append(cur)
        for li, ln in enumerate(lines[:3]):
            cv2.putText(sheet, ln, (x0 + 16, sy_box + 388 + li * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.39, BGR_INK, 1, cv2.LINE_AA)

    out_path = PREVIEW_DIR / "omni_basic_tests_start_end_pairs.png"
    cv2.imwrite(str(out_path), sheet)
    shutil.copy2(out_path, ARTIFACT_DIR / out_path.name)
    return out_path


def main():
    print("[1/4] Rendering Sheet 01: Nib Foundational Model, 5-View Turnaround & Zero-Slop Joint Callouts...")
    s1 = build_sheet_01_nib_model_and_turnaround()
    print(f"      -> {s1}")

    print("[2/4] Rendering Sheet 02: 5-Agent Orthographic Studio Lineup Across All 3 Passes...")
    s2 = build_sheet_02_5agents_lineup_proportions()
    print(f"      -> {s2}")

    print("[3/4] Rendering Sheet 03 & 24fps Showcase Video: 6-Tier Progressive Animation Curriculum...")
    s3, mp4, gif = build_sheet_03_progressive_tests_and_video()
    print(f"      -> {s3}")
    print(f"      -> {mp4}")
    print(f"      -> {gif}")

    print("[4/4] Exporting Sheet 04 & Google Omni Start/End Keyframe Packs for Basic Tests...")
    s4 = build_sheet_04_omni_basic_keyframe_packs()
    print(f"      -> {s4}")


if __name__ == "__main__":
    main()
