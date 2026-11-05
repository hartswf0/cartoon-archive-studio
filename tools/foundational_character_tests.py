#!/usr/bin/env python3
"""
foundational_character_tests.py
================================================================================
STAGE 01 -> STAGE 06: 100% REFERENCE-TRACED FOUNDATIONAL CHARACTER MODEL SHEETS
& 6-TIER PROGRESSIVE ANIMATION PENCIL-TEST SUITE

Directly traces, segments, gouache-colors, and animates our 7 canonical full-body
hand-drawn reference sheets in `assets/studio_doodles/`:
  1. `basic_nib_turnaround_keys.jpg`      — Full-Body 4-Angle Turnaround (`Front`, `3/4 Left`, `Profile`, `Back`)
  2. `basic_crew_lineup_keys.jpg`         — Full-Body Standing Crew (`Blue-Line`, `Cel-Inker`, `Sprocket`, `Chime`)
  3. `basic_nib_test01_head_expressions.jpg` — Tier 1: Locked-Body Head Turn, Eye Blink & Laugh
  4. `basic_nib_test02_squash_stretch_hop.jpg` — Tier 2: Constant-Volume Squash Crouch & Airborne Stretch Hop
  5. `basic_nib_test03_arm_arc_and_walk.jpg`   — Tier 3: Single-Arm Fountain-Pen Sweep Arc & Stride
  6. `basic_nib_test04_walk_cycle.jpg`         — Tier 4: Classical 4-Beat Walk Cycle (`Contact -> Down -> Passing -> Up`)
  7. `basic_nib_test05_compound_acting.jpg`    — Tier 5/6: Compound Acting (`Chin Tap -> Idea Crouch -> Tongue-Out Sketch -> Proud Laugh`)

Every single character pose on every sheet is 100% traced from the canonical
hand-drawn reference artwork via `reference_traced_cel_animator.py` across all
3 classical passes (`Pass 1: Rough Non-Photo Blue + Onion Skin`, `Pass 2:
Speedball B-5 India Ink Cleanup`, `Pass 3: Watertight Gouache Cel`).
================================================================================
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
sys.path.insert(0, str(ROOT / "tools"))

from reference_traced_cel_animator import (
    BGR_PAPER,
    BGR_GUTTER,
    BGR_INK,
    BGR_NONPHOTO_BLUE,
    BGR_BLUE_FAINT,
    BGR_ARC_RED,
    render_traced_reference_cel_onto_canvas,
)

DOODLES_DIR = ROOT / "assets" / "studio_doodles"
PREVIEW_DIR = ROOT / "preview"
FILMS_DIR = ROOT / "bts_films"
OMNI_BASIC_DIR = ROOT / "omni_keyframe_packs" / "basic_tests"
ARTIFACT_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

BGR_PAPER_CARD = (244, 250, 254)
BGR_RULER_RED = (142, 148, 228)


def _draw_proportion_ruler_grid(
    canvas: np.ndarray,
    x0: int,
    x1: int,
    top_y: int,
    ground_y: int,
    show_labels: bool = True,
):
    """
    Draws horizontal classical animator model-sheet ruler lines (`0.0H` Crown Tip
    down to `3.0H` Ground Plane) across `[x0, x1]`.
    """
    span_h = float(max(40, ground_y - top_y))
    levels = [
        (0.00, "0.0H  CROWN TIP", BGR_RULER_RED, 1),
        (0.24, "0.72H HEAD / EYES", BGR_BLUE_FAINT, 1),
        (0.45, "1.35H CHIN / COLLAR", BGR_RULER_RED, 1),
        (0.60, "1.80H ELBOW CUFF", BGR_BLUE_FAINT, 1),
        (0.75, "2.25H APRON HEM", BGR_RULER_RED, 1),
        (0.89, "2.67H ANKLE CUFF", BGR_BLUE_FAINT, 1),
        (1.00, "3.0H  GROUND PLANE", BGR_INK, 2),
    ]
    for rel_u, label, col, thick in levels:
        y = int(round(top_y + rel_u * span_h))
        if rel_u == 1.00:
            cv2.line(canvas, (x0, y), (x1, y), col, thick, cv2.LINE_AA)
        else:
            step = 14
            for sx in range(x0, x1, step):
                ex = min(x1, sx + 8)
                cv2.line(canvas, (sx, y), (ex, y), col, thick, cv2.LINE_AA)
        if show_labels:
            cv2.putText(
                canvas, label, (x0 + 6, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                BGR_INK if rel_u in (0.0, 0.45, 1.0) else BGR_NONPHOTO_BLUE,
                1, cv2.LINE_AA,
            )


def _place_traced_fullbody_cel(
    canvas: np.ndarray,
    sheet_name: str,
    q_idx: int,
    cx: float,
    top_y: float,
    ground_y: float,
    pass_mode: str = "gouache",
    flip_x: bool = False,
    morph_to_sheet: str = None,
    morph_to_q: int = None,
    morph_t: float = 0.0,
    onion_prev_pose: tuple = None,
    onion_next_pose: tuple = None,
    head_nod_deg: float = 0.0,
    arm_swing_deg: float = 0.0,
    squash_stretch: float = 0.0,
    boil_phase: int = 0,
):
    """
    Renders a full-body traced reference cel anchored cleanly between `top_y` (`0.0H`)
    and `ground_y` (`3.0H`) centered at `cx`.
    """
    cel_h = float(max(60.0, ground_y - top_y))
    cel_w = cel_h * 0.96
    cy = 0.5 * (top_y + ground_y)
    render_traced_reference_cel_onto_canvas(
        canvas,
        sheet_name=sheet_name,
        q_idx=q_idx,
        cx=cx,
        cy=cy,
        target_w=cel_w,
        target_h=cel_h,
        pass_mode=pass_mode,
        is_agent=True,
        flip_x=flip_x,
        draw_prog=1.0,
        bob_phase=0.0,
        boil_phase=boil_phase,
        head_nod_deg=head_nod_deg,
        arm_swing_deg=arm_swing_deg,
        squash_stretch=squash_stretch,
        morph_to_sheet=morph_to_sheet,
        morph_to_q=morph_to_q,
        morph_t=morph_t,
        onion_prev_pose=onion_prev_pose,
        onion_next_pose=onion_next_pose,
    )


# ==============================================================================
# I. SHEET 1: CANONICAL REFERENCE-TRACED TURNAROUND & EXPRESSION SHEET (`Nib`)
# ==============================================================================

def build_sheet_01_nib_model_and_turnaround() -> Path:
    """
    Renders `preview/basic_sheet_01_nib_model_and_turnaround.png` (`2200x1680`)
    100% traced from `basic_nib_turnaround_keys.jpg` and `basic_nib_test01_head_expressions.jpg`.
    """
    W, H = 2200, 1680
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "STAGE 01 -- 100% REFERENCE-TRACED CHARACTER MODEL & TURNAROUND SHEET: 'NIB' (ZERO-DRIFT 3.0-HEAD SYSTEM)",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.84, (240, 248, 255), 2, cv2.LINE_AA,
    )

    # --------------------------------------------------------------------------
    # ROW 1: 5-VIEW FULL-BODY ORTHOGRAPHIC TURNAROUND (`basic_nib_turnaround_keys.jpg`)
    # --------------------------------------------------------------------------
    r1_x0, r1_y0, r1_x1, r1_y1 = 24, 84, W - 24, 650
    cv2.rectangle(sheet, (r1_x0, r1_y0), (r1_x1, r1_y1), BGR_PAPER_CARD, -1)
    cv2.rectangle(sheet, (r1_x0, r1_y0), (r1_x1, r1_y1), BGR_INK, 2, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "ROW 1: 5-VIEW FULL-BODY ORTHOGRAPHIC TURNAROUND (TRACED FROM basic_nib_turnaround_keys.jpg ON CONTINUOUS 0.0H..3.0H RULERS)",
        (r1_x0 + 18, r1_y0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.56, BGR_INK, 1, cv2.LINE_AA,
    )

    top_y1 = r1_y0 + 54
    ground_y1 = r1_y1 - 46
    _draw_proportion_ruler_grid(sheet, r1_x0 + 16, r1_x1 - 16, top_y1, ground_y1, show_labels=True)

    turn_views = [
        ("1. FRONT VIEW (0 deg)", "basic_nib_turnaround_keys", 0, False),
        ("2. 3/4 LEFT VIEW (-35 deg)", "basic_nib_turnaround_keys", 1, False),
        ("3. SIDE PROFILE (90 deg)", "basic_nib_turnaround_keys", 2, False),
        ("4. 3/4 RIGHT VIEW (+35 deg)", "basic_nib_turnaround_keys", 1, True),
        ("5. BACK VIEW (180 deg)", "basic_nib_turnaround_keys", 3, False),
    ]
    for idx, (v_label, s_name, q_i, fl_x) in enumerate(turn_views):
        cx = 365.0 + idx * 410.0
        cv2.line(sheet, (int(cx), top_y1 - 8), (int(cx), ground_y1 + 10), BGR_BLUE_FAINT, 1, cv2.LINE_AA)
        _place_traced_fullbody_cel(
            sheet, s_name, q_i, cx, float(top_y1), float(ground_y1),
            pass_mode="gouache", flip_x=fl_x,
        )
        cv2.putText(sheet, v_label, (int(cx - 102), ground_y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.48, BGR_INK, 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # ROW 2: 3-PASS CLASSICAL PROGRESSION + 4 CANONICAL EXPRESSIONS (`basic_nib_test01_head_expressions.jpg`)
    # --------------------------------------------------------------------------
    r2_x0, r2_y0, r2_x1, r2_y1 = 24, 670, W - 24, 1165
    cv2.rectangle(sheet, (r2_x0, r2_y0), (r2_x1, r2_y1), BGR_PAPER_CARD, -1)
    cv2.rectangle(sheet, (r2_x0, r2_y0), (r2_x1, r2_y1), BGR_INK, 2, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "ROW 2: 4-STAGE CLASSICAL PASS PROGRESSION (LEFT)  &  4 LOCKED-BODY HEAD EXPRESSIONS (TRACED FROM basic_nib_test01_head_expressions.jpg)",
        (r2_x0 + 18, r2_y0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.54, BGR_INK, 1, cv2.LINE_AA,
    )

    top_y2 = r2_y0 + 50
    ground_y2 = r2_y1 - 42
    _draw_proportion_ruler_grid(sheet, r2_x0 + 16, r2_x1 - 16, top_y2, ground_y2, show_labels=False)

    row2_items = [
        ("PASS 1: ROUGH BLUE", "basic_nib_turnaround_keys", 0, "rough_blue", None),
        ("PASS 1B: ONION BREAKDOWN", "basic_nib_test01_head_expressions", 1, "rough_blue", ("basic_nib_test01_head_expressions", 0)),
        ("PASS 2: SPEEDBALL INK", "basic_nib_turnaround_keys", 0, "clean_ink", None),
        ("PASS 3: GOUACHE CEL", "basic_nib_turnaround_keys", 0, "gouache", None),
        ("EXPR 1: 3/4 LEFT CALM", "basic_nib_test01_head_expressions", 0, "gouache", None),
        ("EXPR 2: CENTER EYE BLINK", "basic_nib_test01_head_expressions", 1, "gouache", None),
        ("EXPR 3: 3/4 RIGHT 'AHA!'", "basic_nib_test01_head_expressions", 2, "gouache", None),
        ("EXPR 4: JOYFUL LAUGH", "basic_nib_test01_head_expressions", 3, "gouache", None),
    ]
    for idx, (lbl, s_name, q_i, pmode, onion_p) in enumerate(row2_items):
        cx = 160.0 + idx * 268.0
        _place_traced_fullbody_cel(
            sheet, s_name, q_i, cx, float(top_y2), float(ground_y2),
            pass_mode=pmode, onion_prev_pose=onion_p,
        )
        cv2.putText(sheet, lbl, (int(cx - 95), ground_y2 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.42, BGR_INK, 1, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # ROW 3: CANONICAL 2x2 REFERENCE DOODLE SHEETS & CLOSEUP STUDIES (`y = 1185..1656`)
    # --------------------------------------------------------------------------
    r3_x0, r3_y0, r3_x1, r3_y1 = 24, 1185, W - 24, 1656
    cv2.rectangle(sheet, (r3_x0, r3_y0), (r3_x1, r3_y1), BGR_PAPER_CARD, -1)
    cv2.rectangle(sheet, (r3_x0, r3_y0), (r3_x1, r3_y1), BGR_INK, 2, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "ROW 3: SOURCE 1024x1024 HAND-DRAWN REFERENCE SHEETS IN assets/studio_doodles/ (100% MATCHED TO TRACED CELS ABOVE)",
        (r3_x0 + 18, r3_y0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.56, BGR_INK, 1, cv2.LINE_AA,
    )

    ref_cards = [
        ("1. basic_nib_turnaround_keys.jpg", "basic_nib_turnaround_keys.jpg"),
        ("2. basic_nib_test01_head_expressions.jpg", "basic_nib_test01_head_expressions.jpg"),
        ("3. basic_nib_test02_squash_stretch_hop.jpg", "basic_nib_test02_squash_stretch_hop.jpg"),
        ("4. basic_nib_test03_arm_arc_and_walk.jpg", "basic_nib_test03_arm_arc_and_walk.jpg"),
        ("5. basic_nib_test05_compound_acting.jpg", "basic_nib_test05_compound_acting.jpg"),
    ]
    cw_box = 412
    ch_box = 395
    for ci, (ctitle, fname) in enumerate(ref_cards):
        bx0 = r3_x0 + 18 + ci * (cw_box + 14)
        by0 = r3_y0 + 44
        bx1 = bx0 + cw_box
        by1 = by0 + ch_box
        cv2.rectangle(sheet, (bx0, by0), (bx1, by1), (255, 255, 255), -1)
        cv2.rectangle(sheet, (bx0, by0), (bx1, by1), BGR_INK, 2, cv2.LINE_AA)

        sub = sheet[by0 + 4:by1 - 34, bx0 + 4:bx1 - 4]
        sh, sw = sub.shape[:2]
        img_p = DOODLES_DIR / fname
        if img_p.exists():
            raw = cv2.imread(str(img_p))
            side = min(sh, sw)
            resized = cv2.resize(raw, (side, side), interpolation=cv2.INTER_AREA)
            ox = (sw - side) // 2
            oy = (sh - side) // 2
            sub[oy:oy + side, ox:ox + side] = resized
        cv2.putText(sheet, ctitle, (bx0 + 10, by1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.42, BGR_INK, 1, cv2.LINE_AA)

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREVIEW_DIR / "basic_sheet_01_nib_model_and_turnaround.png"
    cv2.imwrite(str(out_path), sheet)
    shutil.copy2(out_path, ARTIFACT_DIR / out_path.name)
    return out_path


# ==============================================================================
# II. SHEET 2: 5-CHARACTER FULL-BODY STUDIO LINEUP (`basic_crew_lineup_keys.jpg`)
# ==============================================================================

def build_sheet_02_5agents_lineup_proportions() -> Path:
    """
    Renders `preview/basic_sheet_02_5agents_lineup_proportions.png` (`2200x1620`):
      - All 5 Studio Characters (`Nib`, `Blue-Line`, `Cel-Inker`, `Sprocket`, `Chime`)
        traced directly from `basic_nib_turnaround_keys.jpg` and `basic_crew_lineup_keys.jpg`
        across all 3 classical passes (`Pass 1 Rough Blue`, `Pass 2 Speedball Ink`,
        `Pass 3 Watertight Gouache Cel`).
    """
    W, H = 2200, 1620
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "STAGE 01B -- 100% REFERENCE-TRACED 5-CHARACTER STUDIO LINEUP (LOCKED 3.0-HEAD PROPORTION GRID ACROSS ALL 3 PASSES)",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.80, (240, 248, 255), 2, cv2.LINE_AA,
    )

    passes = [
        ("ROW 1 -- PASS 1: NON-PHOTO BLUE VOLUMETRIC CONSTRUCTION & RULER GRID", "rough_blue"),
        ("ROW 2 -- PASS 2: SPEEDBALL B-5 INDIA INK CLEANUP CONTOURS", "clean_ink"),
        ("ROW 3 -- PASS 3: WATERTIGHT OPAQUE GOUACHE CEL & SIGNATURE STUDIO EMBLEMS", "gouache"),
    ]
    agents = [
        ("NIB (Lead Animator)", "basic_nib_turnaround_keys", 0),
        ("BLUE-LINE (Rough Layout)", "basic_crew_lineup_keys", 0),
        ("CEL-INKER (Ink & Paint)", "basic_crew_lineup_keys", 1),
        ("SPROCKET (Multiplane Camera)", "basic_crew_lineup_keys", 2),
        ("CHIME (Foley & Timing)", "basic_crew_lineup_keys", 3),
    ]

    for r_idx, (r_title, p_mode) in enumerate(passes):
        ry0 = 84 + r_idx * 504
        ry1 = ry0 + 488
        cv2.rectangle(sheet, (24, ry0), (W - 24, ry1), BGR_PAPER_CARD, -1)
        cv2.rectangle(sheet, (24, ry0), (W - 24, ry1), BGR_INK, 2, cv2.LINE_AA)
        cv2.putText(sheet, r_title, (42, ry0 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, BGR_INK, 1, cv2.LINE_AA)

        top_y = ry0 + 50
        ground_y = ry1 - 42
        _draw_proportion_ruler_grid(sheet, 40, W - 40, top_y, ground_y, show_labels=True)

        for a_idx, (lbl, s_name, q_i) in enumerate(agents):
            cx = 365.0 + a_idx * 410.0
            _place_traced_fullbody_cel(
                sheet, s_name, q_i, cx, float(top_y), float(ground_y),
                pass_mode=p_mode,
            )
            cv2.putText(sheet, lbl, (int(cx - 108), ground_y + 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, BGR_INK, 1, cv2.LINE_AA)

    out_path = PREVIEW_DIR / "basic_sheet_02_5agents_lineup_proportions.png"
    cv2.imwrite(str(out_path), sheet)
    shutil.copy2(out_path, ARTIFACT_DIR / out_path.name)
    return out_path


# ==============================================================================
# III. SHEET 3 & VIDEO: 6-TIER REFERENCE-TRACED PROGRESSIVE PENCIL-TEST SUITE
# ==============================================================================

TIER_SPECS = [
    {
        "tier": 1,
        "slug": "tier_01_head_turn_and_blink",
        "title": "TIER 1 (BASIC TEST 01): LOCKED-BODY HEAD TURN, EYE BLINK & EXPRESSION SHIFT (basic_nib_test01_head_expressions.jpg)",
        "subtitle": "Full-body Nib stays grounded while rotating head from 3/4 Left (Q0) -> Center Blink (Q1) -> 3/4 Right 'Aha!' (Q2) -> Open Laugh (Q3)",
        "sheet": "basic_nib_test01_head_expressions",
        "keyframes": [0, 1, 2, 3],
        "start_q": 0,
        "end_q": 2,
    },
    {
        "tier": 2,
        "slug": "tier_02_squash_and_stretch_hop",
        "title": "TIER 2 (BASIC TEST 02): CONSTANT-VOLUME SQUASH & STRETCH VERTICAL HOP (basic_nib_test02_squash_stretch_hop.jpg)",
        "subtitle": "Neutral Standing (Q0) -> Deep Anticipation Squat Squash (Q1) -> Airborne Stretch Hop Apex (Q2) -> Cushioned Landing Settle (Q3)",
        "sheet": "basic_nib_test02_squash_stretch_hop",
        "keyframes": [0, 1, 2, 3],
        "start_q": 1,
        "end_q": 2,
    },
    {
        "tier": 3,
        "slug": "tier_03_single_arm_pendulum_arc",
        "title": "TIER 3 (BASIC TEST 03): SINGLE-ARM FOUNTAIN-PEN PENDULUM SWEEP & STRIDE (basic_nib_test03_arm_arc_and_walk.jpg)",
        "subtitle": "Pen Lowered at Side (Q0) -> Horizontal Mid-Arc Point Breakdown (Q1) -> High Overhead 'Eureka!' Flourish (Q2) -> Lively Stride (Q3)",
        "sheet": "basic_nib_test03_arm_arc_and_walk",
        "keyframes": [0, 1, 2, 3],
        "start_q": 0,
        "end_q": 2,
    },
    {
        "tier": 4,
        "slug": "tier_04_bilateral_paper_and_pen",
        "title": "TIER 4 (INTERMEDIATE TEST 04): TURNAROUND ROTATION & CONTRAPPOSTO STANCE (basic_nib_turnaround_keys.jpg)",
        "subtitle": "Smooth full-body orthographic rotation through Front (Q0) -> 3/4 Left (Q1) -> Side Profile (Q2) -> Back (Q3) on locked height rulers",
        "sheet": "basic_nib_turnaround_keys",
        "keyframes": [0, 1, 2, 3],
        "start_q": 0,
        "end_q": 1,
    },
    {
        "tier": 5,
        "slug": "tier_05_weight_shift_walk_cycle",
        "title": "TIER 5 (ADVANCED TEST 05): CLASSICAL 4-BEAT FULL-BODY WALK CYCLE (basic_nib_test04_walk_cycle.jpg)",
        "subtitle": "Contact Stride (Q0) -> Down / Recoil Weight Absorption (Q1) -> Passing Pose (Q2) -> Up / High-Point Whistling Push-Off (Q3)",
        "sheet": "basic_nib_test04_walk_cycle",
        "keyframes": [0, 1, 2, 3],
        "start_q": 0,
        "end_q": 2,
    },
    {
        "tier": 6,
        "slug": "tier_06_compound_acting_shot",
        "title": "TIER 6 (MASTER TEST 06): COMPOUND ACTING PERFORMANCE -- IDEA TO SKETCH FLOURISH (basic_nib_test05_compound_acting.jpg)",
        "subtitle": "Chin-Tap Thinking (Q0) -> Excited Anticipation Crouch (Q1) -> Tongue-Out Sketching Curve (Q2) -> Proud Sketch Display & Laugh (Q3)",
        "sheet": "basic_nib_test05_compound_acting",
        "keyframes": [0, 1, 2, 3],
        "start_q": 0,
        "end_q": 3,
    },
]


def _render_tier_frame_on_box(
    canvas: np.ndarray,
    spec: dict,
    u: float,
    cx: float,
    top_y: float,
    ground_y: float,
    pass_mode: str = "gouache",
    boil_phase: int = 0,
    show_onion: bool = True,
):
    """
    Renders a single frame of `spec` at normalized progress `u in [0.0, 1.0]` by
    stepping through all 4 traced reference quadrants `Q0 -> Q1 -> Q2 -> Q3 -> Q0`
    with smooth diffeomorphic breakdown morphing (`morph_t`) and subtle secondary motion.
    """
    sname = spec["sheet"]
    kfs = list(spec["keyframes"]) + [spec["keyframes"][0]]
    n_segs = len(kfs) - 1  # 4 segments: Q0->Q1, Q1->Q2, Q2->Q3, Q3->Q0
    pos = max(0.0, min(float(n_segs) - 1e-6, u * n_segs))
    idx_a = int(math.floor(pos))
    idx_b = idx_a + 1
    local_t = pos - idx_a

    q_a = kfs[idx_a]
    q_b = kfs[idx_b]

    # Choose the nearest keyframe as primary and morph toward the partner keyframe
    if local_t <= 0.55:
        q_primary = q_a
        q_target = q_b
        morph_t = local_t * 1.4
    else:
        q_primary = q_b
        q_target = q_a
        morph_t = (1.0 - local_t) * 1.4

    nod = 2.4 * math.sin(local_t * math.pi)
    swing = 3.8 * math.sin(local_t * math.pi)
    sq = 0.035 * math.sin(local_t * math.pi) if spec["tier"] in (2, 5, 6) else 0.0

    # Vertical hop offset for Tier 2 airborne apex (`idx_a == 2`, i.e. `Q2`)
    y_shift = 0.0
    if spec["tier"] == 2:
        hop_weight = math.exp(-((pos - 2.0) ** 2) / 0.35)
        y_shift = -18.0 * hop_weight

    if show_onion and 0.12 < local_t < 0.88:
        _place_traced_fullbody_cel(
            canvas,
            sheet_name=sname,
            q_idx=q_a,
            cx=cx - 7.0,
            top_y=top_y + y_shift,
            ground_y=ground_y + y_shift,
            pass_mode="rough_blue",
            boil_phase=boil_phase,
        )

    _place_traced_fullbody_cel(
        canvas,
        sheet_name=sname,
        q_idx=q_primary,
        cx=cx,
        top_y=top_y + y_shift,
        ground_y=ground_y + y_shift,
        pass_mode=pass_mode,
        morph_to_sheet=sname if morph_t > 0.05 else None,
        morph_to_q=q_target if morph_t > 0.05 else None,
        morph_t=morph_t,
        head_nod_deg=nod,
        arm_swing_deg=swing,
        squash_stretch=sq,
        boil_phase=boil_phase,
    )


def build_sheet_03_progressive_tests_and_video() -> tuple:
    """
    Renders:
      1. `preview/basic_tests_tier1_to_tier6_flipbooks.png` (`2240x2160`)
      2. `bts_films/basic_tests_tier1_to_tier6_showcase.mp4` & `.gif` (`24 fps`)
    """
    W, H = 2240, 2160
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "STAGE 02 -> STAGE 06: 100% REFERENCE-TRACED 6-TIER PROGRESSIVE ANIMATION PENCIL-TEST FLIPBOOKS",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.84, (240, 248, 255), 2, cv2.LINE_AA,
    )

    num_cols = 8
    row_h = 338

    for t_idx, spec in enumerate(TIER_SPECS):
        ry0 = 80 + t_idx * (row_h + 6)
        ry1 = ry0 + row_h
        cv2.rectangle(sheet, (20, ry0), (W - 20, ry1), BGR_PAPER_CARD, -1)
        cv2.rectangle(sheet, (20, ry0), (W - 20, ry1), BGR_INK, 2, cv2.LINE_AA)

        cv2.putText(sheet, spec["title"], (36, ry0 + 23), cv2.FONT_HERSHEY_DUPLEX, 0.52, BGR_INK, 1, cv2.LINE_AA)
        cv2.putText(sheet, spec["subtitle"], (36, ry0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.41, BGR_NONPHOTO_BLUE, 1, cv2.LINE_AA)

        top_y = ry0 + 62
        g_y = ry1 - 26
        _draw_proportion_ruler_grid(sheet, 32, W - 32, top_y, g_y, show_labels=(t_idx == 0))

        # 8 columns across:
        #   Odd columns (`F01, F03, F05, F07`) = the 4 canonical keyframes (`Q0, Q1, Q2, Q3`)
        #   Even columns (`F02, F04, F06, F08`) = the 4 breakdown/inbetween frames (`INB 1-2`, `INB 2-3`, `INB 3-4`, `INB 4-1`)
        for f_idx in range(num_cols):
            u = f_idx / float(num_cols)  # 0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875
            cx = 175.0 + f_idx * 268.0
            is_key = (f_idx % 2 == 0)
            _render_tier_frame_on_box(
                sheet, spec, u, cx, float(top_y), float(g_y),
                pass_mode="gouache", boil_phase=f_idx, show_onion=(not is_key),
            )
            q_num = (f_idx // 2) + 1
            next_q = (q_num % 4) + 1
            role_tag = f"KEY Q{q_num}" if is_key else f"INB Q{q_num}->Q{next_q}"
            cv2.putText(
                sheet,
                f"F{f_idx + 1:02d} [{role_tag}]",
                (int(cx - 58), g_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                BGR_ARC_RED if is_key else BGR_INK,
                1, cv2.LINE_AA,
            )

    sheet_path = PREVIEW_DIR / "basic_tests_tier1_to_tier6_flipbooks.png"
    cv2.imwrite(str(sheet_path), sheet)
    shutil.copy2(sheet_path, ARTIFACT_DIR / sheet_path.name)

    # Render 24fps Animated Showcase Video (3x2 Grid of All 6 Reference-Traced Progressive Tests)
    FILMS_DIR.mkdir(parents=True, exist_ok=True)
    mp4_path = FILMS_DIR / "basic_tests_tier1_to_tier6_showcase.mp4"
    gif_path = FILMS_DIR / "basic_tests_tier1_to_tier6_showcase.gif"

    vw, vh = 1440, 960
    total_frames = 72
    tmp_dir = FILMS_DIR / "_tmp_basic_tests"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for fi in range(total_frames):
        phase = fi / float(total_frames)
        u_ping = 0.5 - 0.5 * math.cos(phase * 2.0 * math.pi)
        u_cycle = (phase * 2.0) % 1.0

        frame = np.full((vh, vw, 3), BGR_PAPER, dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (vw, 48), (34, 42, 54), -1)
        cv2.putText(
            frame,
            f"100% REFERENCE-TRACED PENCIL TESTS (TIERS 1-6) | 24 FPS ON ONES | FRAME {fi + 1:02d}/{total_frames:02d}",
            (20, 32), cv2.FONT_HERSHEY_DUPLEX, 0.64, (240, 248, 255), 1, cv2.LINE_AA,
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
            u_use = u_cycle if t_num == 5 else u_ping
            top_cell_y = y0 + 36
            g_cell_y = y1 - 30
            _draw_proportion_ruler_grid(frame, x0 + 10, x1 - 10, top_cell_y, g_cell_y, show_labels=False)

            cx_cell = x0 + cw * 0.5
            _render_tier_frame_on_box(
                frame, spec, u_use, cx_cell, float(top_cell_y), float(g_cell_y),
                pass_mode="gouache", boil_phase=fi, show_onion=True,
            )

            short_t = spec["title"].split("(")[0].strip()
            cv2.putText(frame, short_t[:44], (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_DUPLEX, 0.42, BGR_INK, 1, cv2.LINE_AA)

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
# IV. SHEET 4: GOOGLE OMNI / VEO START & END KEYFRAME PACKS (100% REF-TRACED)
# ==============================================================================

OMNI_BASIC_PROMPTS = {
    1: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "Full-body standing cartoon animator character Nib (wearing his white 3-pointed fountain-pen "
        "crown hat with center slit and brass breather hole, white V-neck collar, terracotta-red "
        "overall apron, rolled elbow cuffs, denim trousers with rolled ankle cuffs, and dark shoes) "
        "keeps his feet, legs, apron, torso, and arms 100% stationary while smoothly rotating ONLY "
        "his head from looking 3/4-left in the Start Frame through a center eye-blink breakdown to "
        "a cheerful 3/4-right smile in the End Frame. Zero body drift, constant line weight."
    ),
    2: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "Full-body cartoon character Nib performs a classic constant-volume squash-and-stretch hop "
        "in place: starting from the deep anticipation crouch squash in the Start Frame (knees bent "
        "wide, arms swung out, focused expression) and springing smoothly upward into the cheerful "
        "airborne stretch jump in the End Frame with both arms raised high. Zero off-model drift."
    ),
    3: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "Full-body cartoon character Nib stands grounded in place and sweeps his drawing arm and "
        "fountain-nib dip pen along a smooth circular arc from lowered at his side in the Start "
        "Frame through a horizontal forward point to raised high overhead in a joyful 'Eureka!' "
        "flourish in the End Frame. Clean wrist overlap and locked character proportions."
    ),
    4: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "Full-body cartoon character Nib performs a smooth orthographic character turnaround rotation "
        "on a fixed ground plane from the Front standing view in the Start Frame to the 3/4-Left "
        "standing view holding his fountain-nib pen in the End Frame, maintaining 100% locked "
        "3-head-tall proportions across the turn."
    ),
    5: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "Full-body cartoon character Nib performs a lively 3/4-right cartoon walk stride on a "
        "fixed ground plane from the wide heel-strike Contact Pose in the Start Frame through a "
        "weight-absorbing Down recoil pose into the upright Passing Pose in the End Frame, with "
        "natural opposite arm-and-leg swing."
    ),
    6: (
        "2D classical hand-drawn cel animation on warm cream archival paper, 24fps on ones, "
        "bold black Speedball B-5 India ink outlines and flat opaque gouache fills. "
        "Full-body cartoon character Nib transitions from thoughtfully tapping his chin with his "
        "fountain-nib pen while holding a blank animation sheet in the Start Frame, through an "
        "excited anticipation crouch and focused tongue-out sketching beat, into proudly holding up "
        "the finished sketch sheet and raising his pen with a joyful open laugh in the End Frame."
    ),
}


def build_sheet_04_omni_basic_keyframe_packs() -> Path:
    """
    Exports clean `1024x1024` Start & End Keyframe PNG pairs + `prompt.txt` for each
    of the 6 Reference-Traced Progressive Tests (`omni_keyframe_packs/basic_tests/tier_01..tier_06`)
    and renders `preview/omni_basic_tests_start_end_pairs.png`.
    """
    OMNI_BASIC_DIR.mkdir(parents=True, exist_ok=True)
    pairs_for_sheet = []

    for spec in TIER_SPECS:
        t_num = spec["tier"]
        slug = spec["slug"]
        sname = spec["sheet"]
        q_start = spec["start_q"]
        q_end = spec["end_q"]

        pack_dir = OMNI_BASIC_DIR / slug
        pack_dir.mkdir(parents=True, exist_ok=True)

        for tag, q_val in [("start_frame", q_start), ("end_frame", q_end)]:
            for pass_suffix, p_mode in [("gouache", "gouache"), ("ink", "clean_ink")]:
                img = np.full((1024, 1024, 3), BGR_PAPER_CARD, dtype=np.uint8)
                _place_traced_fullbody_cel(
                    img, sname, q_val, cx=512.0, top_y=96.0, ground_y=928.0,
                    pass_mode=p_mode,
                )
                cv2.imwrite(str(pack_dir / f"{tag}_{pass_suffix}.png"), img)

        # Also export all 4 quadrants Q0..Q3 as `key_q0..key_q3` so the user can feed any pair or 4-beat sequence!
        for qi in range(4):
            img_q = np.full((1024, 1024, 3), BGR_PAPER_CARD, dtype=np.uint8)
            _place_traced_fullbody_cel(
                img_q, sname, qi, cx=512.0, top_y=96.0, ground_y=928.0,
                pass_mode="gouache",
            )
            cv2.imwrite(str(pack_dir / f"key_q{qi}_gouache.png"), img_q)

        prompt_str = OMNI_BASIC_PROMPTS[t_num]
        (pack_dir / "prompt.txt").write_text(prompt_str + "\n", encoding="utf-8")
        meta = {
            "tier": t_num,
            "slug": slug,
            "title": spec["title"],
            "source_sheet": f"assets/studio_doodles/{sname}.jpg",
            "start_quadrant": q_start,
            "end_quadrant": q_end,
            "start_frame_gouache": f"{slug}/start_frame_gouache.png",
            "end_frame_gouache": f"{slug}/end_frame_gouache.png",
            "start_frame_ink": f"{slug}/start_frame_ink.png",
            "end_frame_ink": f"{slug}/end_frame_ink.png",
            "omni_prompt": prompt_str,
        }
        (pack_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        pairs_for_sheet.append((spec, pack_dir / "start_frame_gouache.png", pack_dir / "end_frame_gouache.png", prompt_str))

    W, H = 2100, 1680
    sheet = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)
    cv2.rectangle(sheet, (0, 0), (W, 68), (34, 42, 54), -1)
    cv2.putText(
        sheet,
        "GOOGLE OMNI / VEO -- 100% REFERENCE-TRACED START & END KEYFRAME PACKS (TIERS 1 TO 6)",
        (24, 44), cv2.FONT_HERSHEY_DUPLEX, 0.84, (240, 248, 255), 2, cv2.LINE_AA,
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

        clean_title = spec["title"].split(" (basic_")[0]
        cv2.putText(sheet, clean_title, (x0 + 16, y0 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.46, BGR_INK, 1, cv2.LINE_AA)

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

        mid_x = (sx_s + 340 + sx_e) // 2
        mid_y = sy_box + 170
        cv2.arrowedLine(sheet, (mid_x - 85, mid_y), (mid_x + 85, mid_y), BGR_ARC_RED, 4, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(sheet, "GOOGLE OMNI", (mid_x - 58, mid_y - 14), cv2.FONT_HERSHEY_DUPLEX, 0.46, BGR_ARC_RED, 1, cv2.LINE_AA)
        cv2.putText(sheet, "REF-TRACED", (mid_x - 52, mid_y + 26), cv2.FONT_HERSHEY_DUPLEX, 0.44, BGR_INK, 1, cv2.LINE_AA)

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
    print("[1/4] Rendering Sheet 01: 100% Reference-Traced Nib Model, 5-View Turnaround & Expressions...")
    s1 = build_sheet_01_nib_model_and_turnaround()
    print(f"      -> {s1}")

    print("[2/4] Rendering Sheet 02: 100% Reference-Traced 5-Agent Studio Lineup Across All 3 Passes...")
    s2 = build_sheet_02_5agents_lineup_proportions()
    print(f"      -> {s2}")

    print("[3/4] Rendering Sheet 03 & 24fps Showcase Video: 6-Tier Reference-Traced Progressive Curriculum...")
    s3, mp4, gif = build_sheet_03_progressive_tests_and_video()
    print(f"      -> {s3}")
    print(f"      -> {mp4}")
    print(f"      -> {gif}")

    print("[4/4] Exporting Sheet 04 & Google Omni Reference-Traced Start/End Keyframe Packs...")
    s4 = build_sheet_04_omni_basic_keyframe_packs()
    print(f"      -> {s4}")


if __name__ == "__main__":
    main()
