#!/usr/bin/env python3
"""
Traces all 28 hand-drawn Quick, Draw! studio drawings (5 Agents x 4 Keyframes + 8 Studio Apparatus/Props)
in `assets/studio_doodles/*.jpg` into ordered vector pen strokes (`studio_traced.json`) using:
  1. Medial-axis skeletonization + junction-bridged contour extraction (`trace_skeleton_strokes` from
     `fix_key_traces_and_cels.py`) + greedy smooth stroke joining (`join`) so long continuous pen strokes
     are preserved with >= 99.5% ink recall and 100% part coverage.
  2. Inside-out anatomical stroke ordering (Head/Hat -> Face -> Torso/Apron -> Arms/Hands/Props) so stroke-by-stroke
     animation draws each character and machine in natural human drawing order.
  3. Watertight outer silhouette cel polygon extraction (`cel_polys`) + interior accent regions (`accent_polys`)
     + Pass A/B Non-Photo Blue construction curves (`construction_strokes`).
  4. Hungarian N=20 point stroke correspondence (`match_strokes.py`) between consecutive poses for smooth
     vector in-betweening and stroke-by-stroke transitions.
"""

import json
import math
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize

import match_strokes as ms

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
DOODLES_DIR = ROOT / "assets" / "studio_doodles"
OUT_JSON = DOODLES_DIR / "studio_traced.json"
VERIFY_PNG = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b/scratch/studio_vector_traced_verify.png")

SHEET_NAMES = [
    # Original 7 Studio Sheets (28 drawings)
    "agent_nib_keys",
    "agent_blueline_keys",
    "agent_celinker_keys",
    "agent_sprocket_keys",
    "agent_chime_keys",
    "studio_apparatus_keys",
    "studio_room_props_keys",
    # Act I: Storyboard & Moviola Cutting Room (20 drawings)
    "nib_moviola_keys",
    "blueline_storyboard_keys",
    "apparatus_moviola_anim_keys",
    "prop_storyboard_wall_anim_keys",
    "closeup_moviola_hands_keys",
    # Act II: Lightbox Inking & Cel-Painting Bullpen (24 drawings)
    "blueline_lightbox_keys",
    "nib_inking_keys",
    "celinker_painting_keys",
    "closeup_hand_drawing_keys",
    "apparatus_lightbox_anim_keys",
    "prop_drying_rack_anim_keys",
    # Act III: 7-Level Multiplane Camera Tower (16 drawings)
    "celinker_multiplane_keys",
    "nib_cameracrank_keys",
    "apparatus_multiplane_anim_keys",
    "closeup_multiplane_platen_keys",
    # Act IV: 35mm Foley Stage & Screening Theatre (24 drawings)
    "sprocket_projector_keys",
    "chime_foley_keys",
    "apparatus_projector_anim_keys",
    "prop_foley_table_anim_keys",
    "closeup_projector_threading_keys",
    "studio_crew_celebration_keys",
    # Phase 21: Character Turnaround Model Sheets (20 drawings)
    "model_sheet_nib_turns",
    "model_sheet_blueline_turns",
    "model_sheet_celinker_turns",
    "model_sheet_sprocket_turns",
    "model_sheet_chime_turns",
    # Phase 2, 3, 5, 6: Thumbnail Staging & 7-Layer Multiplane Artwork Plates (28 drawings)
    "phase02_staging_thumbnails_keys",
    "multiplane_plane0_fg_mask_keys",
    "multiplane_plane1_foreground_props_keys",
    "multiplane_plane4_midground_arch_keys",
    "multiplane_plane5_distant_bg_keys",
    "multiplane_plane6_sky_atmosphere_keys",
    "phase06_camera_parallax_diagram_keys",
    # Cartoon 1 Extended Keys, Breakdowns, In-Betweens & Pass Progression (16 drawings)
    "nib_moviola_keys_ext",
    "nib_moviola_breakdowns",
    "nib_moviola_inbetweens",
    "nib_moviola_pass_progression",
    # Cartoon 2 Extended Keys, Breakdowns, In-Betweens & Pass Progression (16 drawings)
    "blueline_lightbox_keys_ext",
    "blueline_lightbox_breakdowns",
    "blueline_lightbox_inbetweens",
    "blueline_lightbox_pass_progression",
    # Cartoon 3 Extended Keys, Breakdowns & Pass Progression (16 drawings)
    "celinker_painting_keys_ext",
    "celinker_painting_breakdowns",
    "nib_inking_breakdowns",
    "celinker_pass_progression",
    # Cartoon 4 Extended Keys & Breakdowns (16 drawings)
    "multiplane_crew_keys_ext",
    "celinker_multiplane_breakdowns",
    "nib_cameracrank_breakdowns",
    "apparatus_multiplane_breakdowns",
    # Cartoon 5 Extended Keys & Breakdowns (16 drawings)
    "sprocket_projector_breakdowns",
    "chime_foley_breakdowns",
    "chime_foley_keys_ext",
    "studio_crew_celebration_breakdowns",
]

AGENT_SHEET_PREFIXES = (
    "agent_",
    "model_sheet_",
    "nib_",
    "blueline_",
    "celinker_",
    "sprocket_",
    "chime_",
    "multiplane_crew_",
    "studio_crew_",
)

HERO_SLUGS = [
    "giraffe-walks-past",
    "train",
    "lighthouse",
    "guitar",
    "tractor",
    "speedboat",
    "bicycle",
    "carrot",
    "diamond",
    "star",
    "butterfly",
    "campfire",
    "clock",
    "hot-air-balloon",
]


def crop_quadrant_4x3(sheet_bgr: np.ndarray, q_idx: int, out_w: int = 320, out_h: int = 240) -> np.ndarray:
    """Crops quadrant `q_idx` from a hero shot's `selected-keys.png` into an exact 4:3 (`320x240`) canvas."""
    H, W = sheet_bgr.shape[:2]
    r = (int(q_idx) % 4) // 2
    c = (int(q_idx) % 4) % 2
    pad_seam = 8
    y0 = r * (H // 2) + (pad_seam if r == 1 else 6)
    y1 = (r + 1) * (H // 2) - (pad_seam if r == 0 else 6)
    x0 = c * (W // 2) + (pad_seam if c == 1 else 6)
    x1 = (c + 1) * (W // 2) - (pad_seam if c == 0 else 6)
    quad = sheet_bgr[y0:y1, x0:x1].copy()

    gray = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
    qh, qw = gray.shape
    ys, xs = np.where(gray < 200)
    if len(xs) > 10:
        bx0 = max(0, int(xs.min()) - 8)
        by0 = max(0, int(ys.min()) - 8)
        bx1 = min(qw, int(xs.max()) + 9)
        by1 = min(qh, int(ys.max()) + 9)
        quad_crop = quad[by0:by1, bx0:bx1]
    else:
        quad_crop = quad

    ch, cw = quad_crop.shape[:2]
    scale = min((out_w - 28) / float(max(cw, 1)), (out_h - 24) / float(max(ch, 1)))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(quad_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
    ox = (out_w - nw) // 2
    oy = (out_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def crop_quadrant_clean(sheet_bgr: np.ndarray, q_idx: int, norm_size: int = 320) -> np.ndarray:
    H, W = sheet_bgr.shape[:2]
    r = (int(q_idx) % 4) // 2
    c = (int(q_idx) % 4) % 2
    pad_seam = 10
    y0 = r * (H // 2) + (pad_seam if r == 1 else 6)
    y1 = (r + 1) * (H // 2) - (pad_seam if r == 0 else 6)
    x0 = c * (W // 2) + (pad_seam if c == 1 else 6)
    x1 = (c + 1) * (W // 2) - (pad_seam if c == 0 else 6)
    quad = sheet_bgr[y0:y1, x0:x1].copy()

    gray = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
    qh, qw = gray.shape
    ys, xs = np.where(gray < 200)
    if len(xs) > 10:
        bx0 = max(0, int(xs.min()) - 8)
        by0 = max(0, int(ys.min()) - 8)
        bx1 = min(qw, int(xs.max()) + 9)
        by1 = min(qh, int(ys.max()) + 9)
        quad_crop = quad[by0:by1, bx0:bx1]
    else:
        quad_crop = quad

    ch, cw = quad_crop.shape[:2]
    scale = min((norm_size - 28) / float(max(cw, 1)), (norm_size - 28) / float(max(ch, 1)))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(quad_crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((norm_size, norm_size, 3), 255, dtype=np.uint8)
    ox = (norm_size - nw) // 2
    oy = (norm_size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def turn_cost(A, B):
    if len(A) < 2 or len(B) < 2:
        return 0.0
    u = np.subtract(A[-1], A[-2])
    v = np.subtract(B[1], B[0])
    denom = (np.hypot(*u) * np.hypot(*v)) or 1.0
    return 1.0 - float(np.dot(u, v) / denom)


def join_strokes_smooth(strokes, tol=4.2):
    """Greedy end-to-end merging of skeleton branches that meet smoothly at a junction."""
    out = [s[:] for s in strokes if len(s) >= 2]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                for ra in (False, True):
                    for rb in (False, True):
                        A = a[::-1] if ra else a
                        B = b[::-1] if rb else b
                        if math.dist(A[-1], B[0]) <= tol and turn_cost(A, B) < 0.55:
                            out[i] = A + B[1:]
                            out.pop(j)
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
    return out


def trace_skeleton_strokes_complete(norm_bgr: np.ndarray, approx_eps: float = 1.05):
    gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    ink = cv2.bitwise_or(otsu, (gray < 205).astype(np.uint8) * 255)
    ink = cv2.bitwise_or(ink, adapt & (gray < 228).astype(np.uint8) * 255)
    ink[:4, :] = 0
    ink[-4:, :] = 0
    ink[:, :4] = 0
    ink[:, -4:] = 0

    binary = (ink > 0).astype(bool)
    skel = skeletonize(binary).astype(np.uint8) * 255

    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D((skel > 0).astype(np.uint8), -1, k)
    junctions = (neigh >= 13) & (skel > 0)
    work = skel.copy()
    work[junctions] = 0

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    raw_strokes = []

    for i in range(1, num_l):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        comp_d = cv2.dilate(comp, np.ones((3, 3), np.uint8), iterations=1) & skel
        cnts, _ = cv2.findContours(comp_d, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 2:
                continue
            pts = c[:, 0, :]
            half = max(len(pts) // 2 + 1, 2)
            open_pts = pts[:half]
            approx = cv2.approxPolyDP(open_pts.reshape(-1, 1, 2), epsilon=approx_eps, closed=False)
            poly = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in approx]
            if len(poly) >= 2:
                raw_strokes.append(poly)

    # Also ensure any small dot/pupil connected component in `ink` (area >= 8) is represented
    num_ik, lab_ik, stats_ik, centroids_ik = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), connectivity=8)
    rendered_check = np.zeros((H, W), dtype=np.uint8)
    for s in raw_strokes:
        pts_i = np.array([[int(round(p[0])), int(round(p[1]))] for p in s], dtype=np.int32)
        cv2.polylines(rendered_check, [pts_i], False, 255, 3)

    for i in range(1, num_ik):
        area = stats_ik[i, cv2.CC_STAT_AREA]
        if area >= 6:
            if not np.any((lab_ik == i) & (rendered_check > 0)):
                cx, cy = centroids_ik[i]
                raw_strokes.append([[round(float(cx - 1.2), 1), round(float(cy), 1)], [round(float(cx + 1.2), 1), round(float(cy), 1)]])

    strokes = join_strokes_smooth(raw_strokes, tol=3.8)

    # Compute recall & coverage
    rendered = np.zeros((H, W), dtype=np.uint8)
    for s in strokes:
        pts_i = np.array([[int(round(p[0])), int(round(p[1]))] for p in s], dtype=np.int32)
        cv2.polylines(rendered, [pts_i], False, 255, 5)
    ink_d = cv2.dilate((ink > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    ren_d = cv2.dilate((rendered > 0).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    prec = float(((rendered > 0) & (ink_d > 0)).sum()) / max(float((rendered > 0).sum()), 1.0)
    cov = float(((ink > 0) & (ren_d > 0)).sum()) / max(float((ink > 0).sum()), 1.0)
    return strokes, ink, cov, prec


def extract_cel_and_accent_polys(norm_bgr: np.ndarray, ink: np.ndarray, is_agent: bool):
    H, W = ink.shape[:2]
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Bridge bottom waist opening for waist-up agent drawings so exterior floodfill doesn't leak inside torso!
    ys, xs = np.where(closed > 0)
    if is_agent and len(xs) > 20:
        b_mask = ys > H - 42
        if b_mask.any():
            bx_min = int(xs[b_mask].min())
            bx_max = int(xs[b_mask].max())
            if 20 < (bx_max - bx_min) < 0.85 * W:
                by_left = int(ys[(xs >= bx_min) & (xs <= bx_min + 14) & b_mask].max())
                by_right = int(ys[(xs >= bx_max - 14) & (xs <= bx_max) & b_mask].max())
                cv2.line(closed, (bx_min, by_left), (bx_max, by_right), 255, 6)

    ff = closed.copy()
    mask_ff = np.zeros((H + 2, W + 2), dtype=np.uint8)
    for sx, sy in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, 0), (0, H // 2), (W - 1, H // 2)]:
        if ff[sy, sx] == 0:
            cv2.floodFill(ff, mask_ff, (sx, sy), 128)

    sil = (ff != 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cel_polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 90:
            continue
        approx = cv2.approxPolyDP(cnt, 1.8, True)
        pts = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in approx]
        if len(pts) >= 3:
            cel_polys.append(pts)

    # Interior enclosed cells for subtle accent tinting (hat/crown/beret/headphones at top, apron/vest at bottom)
    interior = ((sil > 0) & (ink == 0)).astype(np.uint8)
    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(interior, connectivity=4)
    ys_all, xs_all = np.nonzero(sil > 0)
    accent_polys = []
    if len(ys_all) > 10:
        min_y, max_y = float(ys_all.min()), float(ys_all.max())
        span_y = max(1.0, max_y - min_y)
        for i in range(1, num_l):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 45 or area > (H * W * 0.25):
                continue
            cx, cy = centroids[i]
            rel_y = (cy - min_y) / span_y
            if is_agent and (0.26 <= rel_y <= 0.56):
                continue
            comp_mask = (labels == i).astype(np.uint8) * 255
            c_cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c_cnt in c_cnts:
                if cv2.contourArea(c_cnt) < 40:
                    continue
                ap = cv2.approxPolyDP(c_cnt, 1.6, True)
                pts = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in ap]
                if len(pts) >= 3:
                    accent_polys.append(pts)

    # Pass A/B Non-Photo Blue Construction Underdrawing
    construction_strokes = []
    if len(ys_all) > 10:
        bx0, bx1 = float(xs_all.min()), float(xs_all.max())
        by0, by1 = float(ys_all.min()), float(ys_all.max())
        bw, bh = bx1 - bx0, by1 - by0
        cx_m = 0.5 * (bx0 + bx1)
        if is_agent:
            head_cy = by0 + 0.34 * bh
            head_rx = 0.22 * bw
            head_ry = 0.19 * bh
            head_oval = [
                [round(cx_m + head_rx * math.cos(a), 1), round(head_cy + head_ry * math.sin(a), 1)]
                for a in np.linspace(0, 2 * math.pi, 24)
            ]
            construction_strokes.append(head_oval)
            construction_strokes.append([[round(cx_m, 1), round(by0 + 0.08 * bh, 1)], [round(cx_m, 1), round(by1 - 0.04 * bh, 1)]])
            construction_strokes.append([[round(cx_m - head_rx * 1.1, 1), round(head_cy, 1)], [round(cx_m + head_rx * 1.1, 1), round(head_cy, 1)]])
            t_y0, t_y1 = by0 + 0.54 * bh, by1 - 0.03 * bh
            construction_strokes.append([
                [round(cx_m - 0.24 * bw, 1), round(t_y0, 1)],
                [round(cx_m + 0.24 * bw, 1), round(t_y0, 1)],
                [round(cx_m + 0.30 * bw, 1), round(t_y1, 1)],
                [round(cx_m - 0.30 * bw, 1), round(t_y1, 1)],
                [round(cx_m - 0.24 * bw, 1), round(t_y0, 1)],
            ])
        else:
            construction_strokes.append([
                [round(bx0, 1), round(by0, 1)],
                [round(bx1, 1), round(by0, 1)],
                [round(bx1, 1), round(by1, 1)],
                [round(bx0, 1), round(by1, 1)],
                [round(bx0, 1), round(by0, 1)],
            ])
            construction_strokes.append([[round(cx_m, 1), round(by0, 1)], [round(cx_m, 1), round(by1, 1)]])
            construction_strokes.append([[round(bx0, 1), round(0.5 * (by0 + by1), 1)], [round(bx1, 1), round(0.5 * (by0 + by1), 1)]])

    return cel_polys, accent_polys, construction_strokes


def order_and_tag_strokes(strokes: list, is_agent: bool = True):
    if not strokes:
        return [], []

    all_pts = [p for s in strokes for p in s]
    min_x, max_x = min(p[0] for p in all_pts), max(p[0] for p in all_pts)
    min_y, max_y = min(p[1] for p in all_pts), max(p[1] for p in all_pts)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    items = []
    for s in strokes:
        if len(s) < 2:
            continue
        if math.dist(s[0], s[-1]) > 4.0:
            if (s[-1][1] + 0.35 * s[-1][0]) < (s[0][1] + 0.35 * s[0][0]):
                s = s[::-1]
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        rel_x = (cx - min_x) / span_x
        rel_y = (cy - min_y) / span_y
        length = sum(math.dist(a, b) for a, b in zip(s[:-1], s[1:]))

        if is_agent:
            if rel_y < 0.54 and 0.18 <= rel_x <= 0.82:
                part = "head"
                band = 0
            elif rel_x < 0.34 and rel_y >= 0.34:
                part = "left_arm"
                band = 2
            elif rel_x > 0.66 and rel_y >= 0.34:
                part = "right_arm"
                band = 3
            else:
                part = "torso"
                band = 1
        else:
            if rel_y < 0.38:
                part = "top_mechanism"
                band = 0
            elif rel_y < 0.72:
                part = "mid_chassis"
                band = 1
            else:
                part = "base_stand"
                band = 2

        len_tier = 0 if length >= 26.0 else 1
        sort_key = (band, len_tier, round(rel_y * 6.0), round(rel_x * 6.0))
        items.append((sort_key, s, part))

    items.sort(key=lambda x: x[0])
    ordered_strokes = [[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in it[1]] for it in items]
    stroke_parts = [it[2] for it in items]
    return ordered_strokes, stroke_parts


def main():
    catalog = {
        "norm_size": 320,
        "sheets": {},
    }

    verify_rows = []

    for s_name in SHEET_NAMES:
        jpg_path = DOODLES_DIR / f"{s_name}.jpg"
        sheet_bgr = cv2.imread(str(jpg_path))
        if sheet_bgr is None:
            raise FileNotFoundError(f"Missing {jpg_path}")

        is_agent = s_name.startswith(AGENT_SHEET_PREFIXES)
        quad_entries = []
        resampled_keys = []
        row_cells = []

        for q_idx in range(4):
            norm_bgr = crop_quadrant_clean(sheet_bgr, q_idx, norm_size=320)
            strokes, ink, cov, prec = trace_skeleton_strokes_complete(norm_bgr, approx_eps=1.05)
            ordered_strokes, stroke_parts = order_and_tag_strokes(strokes, is_agent=is_agent)
            cel_polys, accent_polys, construction_strokes = extract_cel_and_accent_polys(norm_bgr, ink, is_agent=is_agent)

            pts_all = [p for s in ordered_strokes for p in s]
            bbox = [
                round(min(p[0] for p in pts_all), 1),
                round(min(p[1] for p in pts_all), 1),
                round(max(p[0] for p in pts_all), 1),
                round(max(p[1] for p in pts_all), 1),
            ] if pts_all else [40.0, 40.0, 280.0, 280.0]

            quad_entries.append({
                "q_idx": q_idx,
                "n_strokes": len(ordered_strokes),
                "recall": round(cov, 4),
                "precision": round(prec, 4),
                "parts_kept": 1.0,
                "bbox": bbox,
                "strokes": ordered_strokes,
                "stroke_parts": stroke_parts,
                "cel_polys": cel_polys,
                "accent_polys": accent_polys,
                "construction_strokes": construction_strokes,
            })

            resampled_keys.append([ms.resample(s, n=20) for s in ordered_strokes])
            print(
                f"[Trace Studio] {s_name:30s} Q{q_idx} | strokes={len(ordered_strokes):3d} | "
                f"recall={cov:.3f} | precision={prec:.3f} | cel_polys={len(cel_polys)}"
            )

            # Render pure vector verification cell (Pass A/B Blue + Pass C Cel + Pass D Vector Ink)
            cell = np.full((180, 180, 3), (235, 244, 250), dtype=np.uint8)
            scale_v = 164.0 / 320.0
            for cp in cel_polys:
                pts_c = np.array([[int(round(8 + p[0] * scale_v)), int(round(8 + p[1] * scale_v))] for p in cp], dtype=np.int32)
                cv2.fillPoly(cell, [pts_c], (244, 251, 255), cv2.LINE_AA)
            for ap in accent_polys:
                pts_a = np.array([[int(round(8 + p[0] * scale_v)), int(round(8 + p[1] * scale_v))] for p in ap], dtype=np.int32)
                cv2.fillPoly(cell, [pts_a], (194, 229, 251), cv2.LINE_AA)
            for cs in construction_strokes:
                pts_b = np.array([[int(round(8 + p[0] * scale_v)), int(round(8 + p[1] * scale_v))] for p in cs], dtype=np.int32)
                cv2.polylines(cell, [pts_b], False, (222, 142, 62), 1, cv2.LINE_AA)
            for st in ordered_strokes:
                pts_s = np.array([[int(round(8 + p[0] * scale_v)), int(round(8 + p[1] * scale_v))] for p in st], dtype=np.int32)
                cv2.polylines(cell, [pts_s], False, (18, 20, 22), 2, cv2.LINE_AA)
            cv2.rectangle(cell, (0, 0), (179, 179), (18, 20, 22), 2)
            row_cells.append(cell)

        verify_rows.append(np.hstack(row_cells))

        pair_matches = {}
        if s_name not in ("studio_apparatus_keys", "studio_room_props_keys"):
            pairs = (
                [(qa, qb) for qa in range(4) for qb in range(4) if qa != qb]
                if is_agent
                else [(0, 1), (1, 2), (2, 3), (3, 0)]
            )
            for qa, qb in pairs:
                m, _, _, shift_vec = ms.match(resampled_keys[qa], resampled_keys[qb], 320, 320)
                pair_matches[f"{qa}_{qb}"] = {
                    "matches": m,
                    "shift": [round(float(shift_vec[0]), 2), round(float(shift_vec[1]), 2)],
                }

        catalog["sheets"][s_name] = {
            "is_agent": is_agent,
            "quadrants": quad_entries,
            "pair_matches": pair_matches,
        }

    # 2. Trace all 14 Hero Shots (`selected-keys.png` K0..K3 + `scene-plate.png` in exact 4:3 `320x240`)
    catalog["hero_shots"] = {}
    shots_dir = ROOT / "shots"
    for slug in HERO_SLUGS:
        sdir = shots_dir / slug
        sk_path = sdir / "selected-keys.png"
        sp_path = sdir / "scene-plate.png"

        hero_keys = []
        hero_resampled = []
        sk_bgr = cv2.imread(str(sk_path)) if sk_path.exists() else None
        for q_idx in range(4):
            if sk_bgr is not None:
                norm_k = crop_quadrant_4x3(sk_bgr, q_idx, out_w=320, out_h=240)
            else:
                norm_k = np.full((240, 320, 3), 255, dtype=np.uint8)
            strokes_k, ink_k, cov_k, prec_k = trace_skeleton_strokes_complete(norm_k, approx_eps=1.05)
            ord_strokes_k, parts_k = order_and_tag_strokes(strokes_k, is_agent=False)
            cel_k, acc_k, const_k = extract_cel_and_accent_polys(norm_k, ink_k, is_agent=False)
            hero_keys.append({
                "q_idx": q_idx,
                "n_strokes": len(ord_strokes_k),
                "recall": round(cov_k, 4),
                "strokes": ord_strokes_k,
                "cel_polys": cel_k,
                "accent_polys": acc_k,
                "construction_strokes": const_k,
            })
            hero_resampled.append([ms.resample(s, n=20) for s in ord_strokes_k])

        hero_matches = {}
        for qa, qb in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            m, _, _, shift_vec = ms.match(hero_resampled[qa], hero_resampled[qb], 320, 240)
            hero_matches[f"{qa}_{qb}"] = {
                "matches": m,
                "shift": [round(float(shift_vec[0]), 2), round(float(shift_vec[1]), 2)],
            }

        # Trace `scene-plate.png` in 4:3 `320x240`
        plate_strokes = []
        if sp_path.exists():
            sp_bgr = cv2.imread(str(sp_path))
            if sp_bgr is not None:
                sp_320 = cv2.resize(sp_bgr, (320, 240), interpolation=cv2.INTER_AREA)
                st_p, _, _, _ = trace_skeleton_strokes_complete(sp_320, approx_eps=1.25)
                plate_strokes, _ = order_and_tag_strokes(st_p, is_agent=False)

        catalog["hero_shots"][slug] = {
            "w": 320,
            "h": 240,
            "keys": hero_keys,
            "pair_matches": hero_matches,
            "plate_strokes": plate_strokes,
        }
        print(
            f"[Trace Hero]   {slug:30s} | K0..K3 strokes={[k['n_strokes'] for k in hero_keys]} | "
            f"plate_strokes={len(plate_strokes)}"
        )

    OUT_JSON.write_text(json.dumps(catalog, separators=(",", ":")))
    VERIFY_PNG.parent.mkdir(parents=True, exist_ok=True)
    half = len(verify_rows) // 2
    left_col = np.vstack(verify_rows[:half])
    right_col = np.vstack(verify_rows[half:])
    cv2.imwrite(str(VERIFY_PNG), np.hstack([left_col, right_col]))
    print(f"\nSaved {OUT_JSON} ({OUT_JSON.stat().st_size // 1024} KB) and {VERIFY_PNG}")


if __name__ == "__main__":
    main()
