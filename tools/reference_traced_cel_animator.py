#!/usr/bin/env python3
"""
High-Resolution Reference-Traced Cel & Variable-Width Stroke Animator (`reference_traced_cel_animator.py`)
===========================================================================================================
Directly traces over the canonical 1024x1024 (`512x512` per quadrant) hand-drawn studio reference sheets
in `assets/studio_doodles/*.jpg` and animates directly from the traced drawings:

  1. FULL-RESOLUTION (`512x512`) VARIABLE-WIDTH STROKE & CONTOUR TRACING:
     - Extracts anti-aliased ink coverage (`ink_alpha`), binary ink mask (`ink_bin`), and exact Euclidean
       distance transform (`dist_map = cv2.distanceTransform`) so every medial-axis stroke point `(x, y, r)`
       carries its exact calligraphic stroke radius `r` from the reference drawing.
     - Orders strokes Inside-Out (`Head/Crown -> Face -> Torso/Collar/Apron -> Left Arm -> Right Arm/Prop`)
       so progressive stroke-by-stroke animation (`draw_prog in [0.0, 1.0]`) physically draws the reference
       pose with a live moving Fountain-Nib Dip Pen or Non-Photo Blue Pencil.

  2. TOPOLOGICAL REGION SEGMENTATION & CANONICAL GOUACHE CEL COLORING:
     - Extracts the watertight outer silhouette (`sil_mask`) of the traced character/apparatus so the
       exterior is 100% transparent multiplane glass.
     - Segments every enclosed interior region via 4-connected component topology and classifies each
       region (`hat_crown`, `skin_face_hands`, `mouth_tongue`, `apron_vest`, `shirt_paper`, `wood_desk`,
       `lightbox_glow`, `brass_metal`) into canonical studio gouache washes.

  3. ARTICULATED ANATOMICAL MESH DEFORMATION, BREAKDOWNS & 24FPS IN-BETWEENING:
     - Animates each traced reference drawing continuously at 24fps On Ones using a smooth diffeomorphic
       2D anatomical deformation grid (`head` nod/tilt, `torso` squash-and-stretch breathing, `left_arm`
       and `right_arm` curved-arc articulation, and `6 Hz` organic paper-tooth line boil).
     - Generates true curved-arc Breakdowns (`B01..B16`) and In-Betweens (`I01..I32`) between consecutive
       traced reference poses using part-aligned displacement fields + Pass 1 Non-Photo Blue / Sepia
       Onion-Skin ghosting and dotted red trajectory arcs.
"""

import math
from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
DOODLES_DIR = ROOT / "assets" / "studio_doodles"

# Canonical Studio Palette (BGR)
BGR_PAPER = (227, 239, 246)         # #F6EFE3 Warm archival animation bond paper
BGR_GUTTER = (244, 251, 255)        # #FFFBF4 Crisp cel white-cream (shirt, crown hat, paper sheets)
BGR_PAPER_SHADOW = (206, 224, 235)  # #EBE0CE Warm paper cel drop-shadow
BGR_INK = (18, 20, 22)              # #161412 Speedball B-5 India ink
BGR_NONPHOTO_BLUE = (212, 145, 58)  # #3A91D4 Animator's non-photo blue pencil
BGR_BLUE_FAINT = (232, 198, 152)    # Faint non-photo blue construction underdrawing
BGR_ONION_PREV = (128, 148, 212)    # Warm terracotta-sepia onion-skin (previous drawing)
BGR_ONION_NEXT = (218, 175, 118)    # Cool blue onion-skin (next drawing)
BGR_ARC_RED = (48, 72, 208)         # Breakdown motion-arc crimson pencil

BGR_SKIN = (212, 232, 248)          # #F8E8D4 Warm cartoon skin gouache
BGR_APRON_RED = (48, 82, 196)       # #C45230 Nib's terracotta red apron
BGR_BLUELINE_BLUE = (142, 96, 46)   # #2E608E Blue-Line's Prussian blue apron/visor
BGR_CELINKER_GOLD = (62, 162, 236)  # #ECA23E Cel-Inker's ochre beret/apron
BGR_SPROCKET_GREEN = (79, 118, 52)  # #34764F Sprocket's forest green vest/cap
BGR_CHIME_CORAL = (68, 116, 222)    # #DE7444 Chime's coral-amber apron/headphones
BGR_WOOD_LIGHT = (142, 184, 218)    # #DAB88E Beechwood drafting table & apparatus
BGR_WOOD_DARK = (92, 128, 162)      # #A2805C Shaded beechwood frame
BGR_LIGHTBOX_GLOW = (212, 246, 255) # #FFF6D4 Backlit milk-glass lightbox disc
BGR_BRASS = (62, 172, 234)          # #EAAC3E Polished brass Acme pegbar, wheel & breather hole

_TRACED_CEL_CACHE = {}
_SHEET_BGR_CACHE = {}


def _load_sheet_bgr(sheet_name: str) -> np.ndarray:
    if sheet_name not in _SHEET_BGR_CACHE:
        p = DOODLES_DIR / f"{sheet_name}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Missing canonical reference sheet: {p}")
        _SHEET_BGR_CACHE[sheet_name] = img
    return _SHEET_BGR_CACHE[sheet_name]


def crop_reference_quadrant_highres(sheet_name: str, q_idx: int, norm_size: int = 512) -> np.ndarray:
    """
    Crops quadrant `q_idx` (`0..3`) from `assets/studio_doodles/<sheet_name>.jpg` at full `512x512`
    resolution with clean white margin normalization.
    """
    sheet_bgr = _load_sheet_bgr(sheet_name)
    H, W = sheet_bgr.shape[:2]
    r = (int(q_idx) % 4) // 2
    c = (int(q_idx) % 4) % 2
    pad_seam = 12
    y0 = r * (H // 2) + (pad_seam if r == 1 else 8)
    y1 = (r + 1) * (H // 2) - (pad_seam if r == 0 else 8)
    x0 = c * (W // 2) + (pad_seam if c == 1 else 8)
    x1 = (c + 1) * (W // 2) - (pad_seam if c == 0 else 8)
    quad = sheet_bgr[y0:y1, x0:x1].copy()

    gray = cv2.cvtColor(quad, cv2.COLOR_BGR2GRAY)
    qh, qw = gray.shape
    ys, xs = np.where(gray < 205)
    if len(xs) > 20:
        bx0 = max(0, int(xs.min()) - 12)
        by0 = max(0, int(ys.min()) - 12)
        bx1 = min(qw, int(xs.max()) + 13)
        by1 = min(qh, int(ys.max()) + 13)
        quad_crop = quad[by0:by1, bx0:bx1]
    else:
        quad_crop = quad

    ch, cw = quad_crop.shape[:2]
    margin = 36
    scale = min((norm_size - margin) / float(max(cw, 1)), (norm_size - margin) / float(max(ch, 1)))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(quad_crop, (nw, nh), interpolation=interp)

    canvas = np.full((norm_size, norm_size, 3), 255, dtype=np.uint8)
    ox = (norm_size - nw) // 2
    oy = (norm_size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def _join_variable_width_strokes(strokes: list, tol: float = 5.5) -> list:
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
                        d = math.hypot(A[-1][0] - B[0][0], A[-1][1] - B[0][1])
                        if d <= tol:
                            u = (A[-1][0] - A[-2][0], A[-1][1] - A[-2][1])
                            v = (B[1][0] - B[0][0], B[1][1] - B[0][1])
                            denom = (math.hypot(*u) * math.hypot(*v)) or 1.0
                            cos_ang = (u[0] * v[0] + u[1] * v[1]) / denom
                            if cos_ang > 0.40:
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


def _extract_variable_width_strokes_and_contours(gray: np.ndarray, ink_bin: np.ndarray, dist_map: np.ndarray, is_agent: bool):
    """
    Extracts ordered variable-width medial-axis strokes `[[x, y, r], ...]` from `ink_bin` + `dist_map`,
    plus closed contour polygons for thick solid ink features (eyes, inkwell reservoir, etc.).
    """
    H, W = gray.shape
    binary = (ink_bin > 0).astype(bool)
    skel = skeletonize(binary).astype(np.uint8) * 255

    k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neigh = cv2.filter2D((skel > 0).astype(np.uint8), -1, k)
    junctions = (neigh >= 13) & (skel > 0)
    work = skel.copy()
    work[junctions] = 0

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    raw_strokes = []

    for i in range(1, num_l):
        if stats[i, cv2.CC_STAT_AREA] < 2:
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
            approx = cv2.approxPolyDP(open_pts.reshape(-1, 1, 2), epsilon=1.1, closed=False)
            poly = []
            for p in approx:
                px = int(np.clip(round(float(p[0][0])), 0, W - 1))
                py = int(np.clip(round(float(p[0][1])), 0, H - 1))
                rad = max(1.15, float(dist_map[py, px]) * 1.08)
                poly.append([float(px), float(py), round(rad, 2)])
            if len(poly) >= 2:
                raw_strokes.append(poly)

    joined = _join_variable_width_strokes(raw_strokes, tol=5.2)

    # Order strokes Inside-Out (Head/Crown -> Face -> Torso/Apron -> Left Arm -> Right Arm/Prop)
    all_pts = [p for s in joined for p in s]
    if not all_pts:
        return [], []
    min_x, max_x = min(p[0] for p in all_pts), max(p[0] for p in all_pts)
    min_y, max_y = min(p[1] for p in all_pts), max(p[1] for p in all_pts)
    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)

    items = []
    for s in joined:
        if math.hypot(s[0][0] - s[-1][0], s[0][1] - s[-1][1]) > 6.0:
            if (s[-1][1] + 0.35 * s[-1][0]) < (s[0][1] + 0.35 * s[0][0]):
                s = s[::-1]
        cx = float(np.mean([p[0] for p in s]))
        cy = float(np.mean([p[1] for p in s]))
        rel_x = (cx - min_x) / span_x
        rel_y = (cy - min_y) / span_y
        length = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(s[:-1], s[1:]))

        if is_agent:
            if rel_y < 0.27 and 0.16 <= rel_x <= 0.84:
                part = "hat"
                band = 0
            elif rel_y < 0.54 and 0.18 <= rel_x <= 0.82:
                part = "head"
                band = 1
            elif rel_x < 0.32 and rel_y >= 0.32:
                part = "left_arm"
                band = 3
            elif rel_x > 0.68 and rel_y >= 0.32:
                part = "right_arm"
                band = 4
            else:
                part = "torso"
                band = 2
        else:
            if rel_y < 0.36:
                part = "top_mechanism"
                band = 0
            elif rel_y < 0.72:
                part = "mid_chassis"
                band = 1
            else:
                part = "base_stand"
                band = 2

        len_tier = 0 if length >= 36.0 else 1
        sort_key = (band, len_tier, round(rel_y * 8.0), round(rel_x * 8.0))
        items.append((sort_key, s, part))

    items.sort(key=lambda x: x[0])
    ordered_strokes = [it[1] for it in items]
    stroke_parts = [it[2] for it in items]
    return ordered_strokes, stroke_parts


def _segment_cel_gouache_regions(sheet_name: str, gray: np.ndarray, ink_bin: np.ndarray, ink_alpha: np.ndarray, is_agent: bool):
    """
    Builds a full-resolution (`512x512`) multi-color Gouache Cel Fill layer (`gouache_bgr`),
    a Cream Paper Cel Fill layer (`paper_cel_bgr`), a Watertight Outer Silhouette mask (`sil_mask`),
    Pass A/B Non-Photo Blue Construction lines (`construction_strokes`), and the character's
    horizontal spine center `spine_x` directly from the topological enclosed regions of the
    traced reference drawing.
    """
    H, W = ink_bin.shape[:2]
    sname = sheet_name.lower()
    if is_agent:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed = cv2.morphologyEx(ink_bin, cv2.MORPH_CLOSE, k_close, iterations=1)
    else:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(ink_bin, cv2.MORPH_CLOSE, k_close, iterations=1)

    ys, xs = np.where(closed > 0)
    if len(xs) > 25:
        min_x0, max_x0 = float(xs.min()), float(xs.max())
        min_y0, max_y0 = float(ys.min()), float(ys.max())
        span_x0 = max(1.0, max_x0 - min_x0)
        span_y0 = max(1.0, max_y0 - min_y0)
        head_band = ys <= (min_y0 + 0.36 * span_y0)
        spine_x = float(np.median(xs[head_band])) if head_band.any() else 0.5 * (min_x0 + max_x0)
    else:
        min_x0, max_x0, min_y0, max_y0 = 32.0, float(W - 32), 32.0, float(H - 32)
        span_x0, span_y0 = max(1.0, max_x0 - min_x0), max(1.0, max_y0 - min_y0)
        spine_x = 0.5 * W

    max_y_ink = int(max_y0) if len(ys) > 25 else (H - 16)
    y_hem = max(16, min(H - 4, max_y_ink - 1))
    y_bridge = min(H - 3, max_y_ink + 2)

    partition_ink = cv2.dilate(ink_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    ap_x0, ap_x1 = int(spine_x - 0.18 * span_x0), int(spine_x + 0.18 * span_x0)

    if len(xs) > 25 and is_agent and ("celebration" not in sname):
        # Find bottom waist/apron ink columns aligned with the character's torso (`spine_x ± 0.33 * span_x0`)
        torso_b_band = (ys >= (max_y_ink - 36)) & (xs >= (spine_x - 0.33 * span_x0)) & (xs <= (spine_x + 0.33 * span_x0))
        b_band = torso_b_band if torso_b_band.any() else (ys >= (max_y_ink - 34))
        if b_band.any():
            bx_min = int(xs[b_band].min())
            bx_max = int(xs[b_band].max())
            ap_x0, ap_x1 = bx_min, bx_max
            if 24 < (bx_max - bx_min) < int(0.92 * W):
                y_win0 = max(0, max_y_ink - 24)
                bottom_strip = closed[y_win0:max_y_ink + 1, bx_min:bx_max + 1]
                col_has_ink = np.where(bottom_strip.any(axis=0))[0]
                # Group contiguous columns into distinct stroke tips so slanted strokes never become thick blocks
                clusters = []
                for c_off in col_has_ink:
                    if not clusters or (int(c_off) - clusters[-1][-1]) > 8:
                        clusters.append([int(c_off)])
                    else:
                        clusters[-1].append(int(c_off))
                for cl in clusters:
                    best_x = bx_min + cl[len(cl) // 2]
                    best_y = y_win0
                    for c_off in cl:
                        col_x = bx_min + c_off
                        col_ys = np.where(closed[y_win0:max_y_ink + 1, col_x] > 0)[0]
                        if len(col_ys) > 0:
                            y_cand = y_win0 + int(col_ys.max())
                            if y_cand >= best_y:
                                best_y = y_cand
                                best_x = col_x
                    if (max_y_ink - best_y) <= 22:
                        cv2.line(closed, (best_x, best_y), (best_x, y_bridge), 255, 3)
                        cv2.line(partition_ink, (best_x, best_y), (best_x, y_bridge), 255, 3)
                        cv2.line(ink_bin, (best_x, best_y), (best_x, y_hem), 255, 3)
                        ink_alpha[min(best_y, y_hem):max(best_y, y_hem) + 1, max(0, best_x - 1):min(W, best_x + 2)] = 1.0
                # Seal the bottom of the waist/apron with a crisp 4px black ink hem line!
                cv2.line(closed, (bx_min, y_bridge), (bx_max, y_bridge), 255, 5)
                cv2.line(partition_ink, (bx_min, y_bridge), (bx_max, y_bridge), 255, 5)
                cv2.line(ink_bin, (bx_min, y_hem), (bx_max, y_hem), 255, 4)
                ink_alpha[max(0, y_hem - 2):min(H, y_hem + 2), bx_min:bx_max + 1] = 1.0

    ff = closed.copy()
    mask_ff = np.zeros((H + 2, W + 2), dtype=np.uint8)
    seeds = [
        (0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1),
        (W // 2, 0), (0, H // 2), (W - 1, H // 2), (W // 2, H - 1),
    ]
    for sx, sy in seeds:
        if ff[sy, sx] == 0:
            cv2.floodFill(ff, mask_ff, (sx, sy), 128)

    sil_mask = (ff != 128).astype(np.uint8) * 255
    if is_agent and y_hem < H - 1 and ("celebration" not in sname):
        sil_mask[y_hem + 1:, :] = 0
    sil_mask = cv2.morphologyEx(sil_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Determine primary character costume color from sheet_name
    if "blueline" in sname:
        costume_bgr = BGR_BLUELINE_BLUE
    elif "celinker" in sname:
        costume_bgr = BGR_CELINKER_GOLD
    elif "sprocket" in sname:
        costume_bgr = BGR_SPROCKET_GREEN
    elif "chime" in sname:
        costume_bgr = BGR_CHIME_CORAL
    else:
        costume_bgr = BGR_APRON_RED

    gouache_bgr = np.full((H, W, 3), BGR_GUTTER, dtype=np.uint8)

    interior = ((sil_mask > 0) & (partition_ink == 0)).astype(np.uint8)
    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(interior, connectivity=4)
    ys_all, xs_all = np.nonzero(sil_mask > 0)

    if len(ys_all) > 20:
        min_x, max_x = float(xs_all.min()), float(xs_all.max())
        min_y, max_y = float(ys_all.min()), float(ys_all.max())
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        spine_rel_x = (spine_x - min_x) / span_x

        for i in range(1, num_l):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < 10:
                continue
            cx, cy = centroids[i]
            comp_top = float(stats[i, cv2.CC_STAT_TOP])
            comp_h = float(stats[i, cv2.CC_STAT_HEIGHT])
            comp_max_y = comp_top + comp_h
            rel_x = (cx - min_x) / span_x
            rel_y = (cy - min_y) / span_y
            rw = stats[i, cv2.CC_STAT_WIDTH] / span_x
            rh = comp_h / span_y
            dist_spine = abs(rel_x - spine_rel_x)
            touches_hem = (comp_max_y >= (max_y - 18.0)) or (rel_y >= 0.79)
            comp_mask = (labels == i)

            if is_agent:
                # 0. Cut out trapped exterior air pockets (e.g. gap between raised arm and crown hat)
                if (rel_y < 0.42 and (rel_x < 0.24 or rel_x > 0.76) and area > 2800 and rh > 0.28):
                    sil_mask[comp_mask] = 0
                    continue
                # 1. Crown breather hole / monocle ring / stopwatch dial / brass gears on right apparatus
                if (rel_y < 0.34 and 20 <= area <= 750 and dist_spine <= 0.24) or (
                    rel_x > spine_rel_x + 0.14 and 0.52 <= rel_y <= 0.86 and 80 <= area <= 1400 and not touches_hem
                ):
                    gouache_bgr[comp_mask] = BGR_BRASS
                # 2. Crown Hat / Visor / Beret / Reel Ears (`rel_y < 0.27`)
                elif rel_y < 0.27 and area > 550:
                    if "celinker" in sname:
                        gouache_bgr[comp_mask] = BGR_CELINKER_GOLD
                    elif "sprocket" in sname:
                        gouache_bgr[comp_mask] = BGR_WOOD_LIGHT if rel_y < 0.17 else BGR_SPROCKET_GREEN
                    else:
                        gouache_bgr[comp_mask] = BGR_GUTTER
                # 3. Face / Cranium / Ears (`0.24 <= rel_y <= 0.56` near `spine_rel_x`)
                elif 0.24 <= rel_y <= 0.56 and dist_spine <= 0.32 and area > 800 and rw < 0.62:
                    gouache_bgr[comp_mask] = BGR_SKIN
                # 4. Tongue / Open Laugh Mouth (`0.39 <= rel_y <= 0.55`, small enclosed region inside face)
                elif 0.39 <= rel_y <= 0.55 and dist_spine <= 0.22 and 35 <= area <= 900:
                    gouache_bgr[comp_mask] = BGR_APRON_RED
                # 5. Apron Skirt (touches bottom hem within `dist_spine <= 0.28`, including all side fold panels)
                #    & Central Chest Apron Bib (`0.55 <= rel_y < 0.69` and `dist_spine <= 0.13`)
                elif (
                    (touches_hem and rel_y >= 0.66 and dist_spine <= 0.28 and area >= 40 and rw < 0.52)
                    or (0.55 <= rel_y < 0.69 and dist_spine <= 0.13 and 220 <= area <= 3600 and rw < 0.28)
                ):
                    if "inking" in sname and rel_x < 0.52 and 0.46 <= rel_y <= 0.82 and area > 1800:
                        gouache_bgr[comp_mask] = BGR_LIGHTBOX_GLOW
                    elif "celebration" in sname and rel_x > 0.52:
                        gouache_bgr[comp_mask] = BGR_BLUELINE_BLUE
                    else:
                        gouache_bgr[comp_mask] = costume_bgr
                # 6. Forearms, Hands & Fingers (above the bottom hem, or lateral hands)
                elif (
                    (not touches_hem and 0.56 <= rel_y <= 0.78 and dist_spine <= 0.24 and 80 <= area <= 3800 and rw > 0.95 * rh)
                    or ((rel_x < 0.30 or rel_x > 0.70 or (0.42 <= rel_y <= 0.84)) and (35 <= area <= 2600) and rw < 0.32 and rh < 0.28 and not touches_hem)
                ):
                    if 0.48 <= rel_y <= 0.66 and dist_spine <= 0.14 and rw < 0.18:
                        gouache_bgr[comp_mask] = BGR_GUTTER
                    elif "inking" in sname and rel_x < 0.56 and rel_y > 0.48 and area > 1800:
                        gouache_bgr[comp_mask] = BGR_LIGHTBOX_GLOW
                    else:
                        gouache_bgr[comp_mask] = BGR_SKIN
                # 7. Right-Side Apparatus (Moviola / Camera Box / Tilted Drafting Board: `rel_x > spine_rel_x + 0.15`)
                elif area > 2200 and (rel_x > spine_rel_x + 0.15) and rel_y > 0.36:
                    if ("inking" in sname or "lightbox" in sname or "painting" in sname) and area > 9500:
                        gouache_bgr[comp_mask] = BGR_GUTTER
                    else:
                        gouache_bgr[comp_mask] = BGR_WOOD_LIGHT
            else:
                if "lightbox" in sname:
                    if (rel_x < 0.35 and 0.15 < rel_y < 0.46 and area > 1200) or (rel_y > 0.74 and 0.30 < rel_x < 0.70 and area > 1500):
                        sil_mask[comp_mask] = 0
                        continue
                    if 0.25 <= rel_x <= 0.78 and 0.20 <= rel_y <= 0.70 and area > 1800:
                        gouache_bgr[comp_mask] = BGR_LIGHTBOX_GLOW if area < 12000 else BGR_GUTTER
                    elif rel_y < 0.22 and rel_x < 0.48:
                        gouache_bgr[comp_mask] = (92, 168, 118)
                    else:
                        gouache_bgr[comp_mask] = BGR_WOOD_LIGHT
                elif "plane1_foreground" in sname:
                    if area > 14000 and (rel_x < 0.15 or rel_x > 0.85 or rel_y < 0.15):
                        sil_mask[comp_mask] = 0
                        continue
                    if 25 <= area <= 650:
                        palette_cycle = [BGR_APRON_RED, BGR_CELINKER_GOLD, BGR_BLUELINE_BLUE, BGR_SPROCKET_GREEN, BGR_CHIME_CORAL, BGR_BRASS]
                        gouache_bgr[comp_mask] = palette_cycle[i % len(palette_cycle)]
                    elif rel_y < 0.48 and rel_x < 0.55:
                        gouache_bgr[comp_mask] = (186, 198, 208)
                    elif rel_y > 0.48 and rel_x > 0.55 and area > 1400:
                        gouache_bgr[comp_mask] = BGR_GUTTER
                    elif rel_x > 0.62 and 0.25 <= rel_y <= 0.68:
                        gouache_bgr[comp_mask] = (224, 244, 252)
                    else:
                        gouache_bgr[comp_mask] = BGR_WOOD_LIGHT
                elif "plane4_midground" in sname or "plane5_distant_bg" in sname:
                    if area > 9500 and (rel_y > 0.52 or rel_x < 0.28 or rel_x > 0.72):
                        sil_mask[comp_mask] = 0
                        continue
                    if "plane5" in sname and 120 <= area <= 3800 and 0.18 <= rel_y <= 0.72:
                        gouache_bgr[comp_mask] = (226, 246, 255)
                    elif rel_y < 0.26:
                        gouache_bgr[comp_mask] = (92, 168, 118) if area < 1600 else BGR_WOOD_LIGHT
                    elif 0.26 <= rel_y <= 0.68 and area > 1600:
                        gouache_bgr[comp_mask] = BGR_GUTTER if area < 6500 else BGR_WOOD_LIGHT
                    else:
                        gouache_bgr[comp_mask] = BGR_WOOD_LIGHT
                elif "closeup" in sname:
                    if area > 14000:
                        gouache_bgr[comp_mask] = BGR_GUTTER
                    elif 1200 <= area <= 14000 and (rel_x > 0.45 or rel_y > 0.45):
                        gouache_bgr[comp_mask] = BGR_SKIN
                    else:
                        gouache_bgr[comp_mask] = BGR_WOOD_LIGHT
                else:
                    if area > 4200 and (rel_y > 0.68 or (rel_x < 0.22 or rel_x > 0.78)):
                        sil_mask[comp_mask] = 0
                        continue
                    if rel_y < 0.35:
                        gouache_bgr[comp_mask] = BGR_BRASS if area < 2500 else BGR_WOOD_LIGHT
                    elif rel_y < 0.72:
                        gouache_bgr[comp_mask] = BGR_LIGHTBOX_GLOW if (0.28 <= rel_x <= 0.72 and 800 <= area <= 8500) else BGR_WOOD_LIGHT
                    else:
                        gouache_bgr[comp_mask] = BGR_WOOD_DARK

        # Extend interior gouache colors down through the bottom 5 rows of `sil_mask` so there is ZERO white fringe
        if is_agent and ("celebration" not in sname):
            src_row = max(0, y_hem - 6)
            for yr in range(max(0, y_hem - 5), min(H, y_hem + 1)):
                gouache_bgr[yr, ap_x0:ap_x1 + 1] = gouache_bgr[src_row, ap_x0:ap_x1 + 1]

    # Build Pass A/B Non-Photo Blue Construction Underdrawing curves (`construction_strokes`)
    construction_strokes = []
    if len(ys_all) > 20 and is_agent:
        bx0, bx1 = float(xs_all.min()), float(xs_all.max())
        by0, by1 = float(ys_all.min()), float(ys_all.max())
        bw, bh = bx1 - bx0, by1 - by0
        cx_m = spine_x
        # 1. Line of Action spine curve
        spine = [
            [cx_m, by0 + 0.10 * bh],
            [cx_m - 0.03 * bw, by0 + 0.45 * bh],
            [cx_m, by1 - 0.05 * bh],
        ]
        construction_strokes.append(spine)
        # 2. Cranium sphere & cross-hairs
        h_cy = by0 + 0.36 * bh
        h_rx, h_ry = 0.19 * bw, 0.17 * bh
        head_oval = [
            [cx_m + h_rx * math.cos(a), h_cy + h_ry * math.sin(a)]
            for a in np.linspace(0, 2.0 * math.pi, 28)
        ]
        construction_strokes.append(head_oval)
        construction_strokes.append([[cx_m - h_rx * 1.12, h_cy], [cx_m + h_rx * 1.12, h_cy]])
        construction_strokes.append([[cx_m, h_cy - h_ry * 1.15], [cx_m, h_cy + h_ry * 1.15]])
        # 3. Ribcage / Torso volumetric mass
        t_cy = by0 + 0.72 * bh
        t_rx, t_ry = 0.20 * bw, 0.20 * bh
        torso_oval = [
            [cx_m + t_rx * math.cos(a), t_cy + t_ry * math.sin(a)]
            for a in np.linspace(0, 2.0 * math.pi, 28)
        ]
        construction_strokes.append(torso_oval)

    return sil_mask, gouache_bgr, construction_strokes, spine_x


def get_traced_reference_cel(sheet_name: str, q_idx: int, is_agent: bool = True) -> dict:
    """
    Loads, traces, and caches the full `512x512` reference drawing `(sheet_name, q_idx)`.
    """
    key = (sheet_name, int(q_idx) % 4, bool(is_agent))
    if key in _TRACED_CEL_CACHE:
        return _TRACED_CEL_CACHE[key]

    norm_bgr = crop_reference_quadrant_highres(sheet_name, q_idx, norm_size=512)
    gray = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY)

    # Anti-aliased ink coverage float [0.0 .. 1.0]
    ink_alpha = np.clip((235.0 - gray.astype(np.float32)) / 145.0, 0.0, 1.0)
    ink_bin = (gray < 200).astype(np.uint8) * 255
    ink_bin[:4, :] = 0
    ink_bin[-4:, :] = 0
    ink_bin[:, :4] = 0
    ink_bin[:, -4:] = 0

    dist_map = cv2.distanceTransform(ink_bin, cv2.DIST_L2, 5)
    strokes, stroke_parts = _extract_variable_width_strokes_and_contours(gray, ink_bin, dist_map, is_agent=is_agent)
    sil_mask, gouache_bgr, construction_strokes, spine_x = _segment_cel_gouache_regions(
        sheet_name, gray, ink_bin, ink_alpha, is_agent=is_agent
    )

    ys, xs = np.nonzero(sil_mask > 0)
    if len(xs) > 10:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    else:
        bbox = (32, 32, 480, 480)

    # Compute part centroids for smooth anatomical deformation & curved-arc breakdowns
    part_pts = {"hat": [], "head": [], "torso": [], "left_arm": [], "right_arm": []}
    for st, pt_name in zip(strokes, stroke_parts):
        bucket = pt_name if pt_name in part_pts else "torso"
        for p in st:
            part_pts[bucket].append((p[0], p[1]))

    bx0, by0, bx1, by1 = bbox
    cx_b, cy_b = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
    part_centroids = {}
    for k_p, pts_l in part_pts.items():
        if pts_l:
            part_centroids[k_p] = (float(np.mean([p[0] for p in pts_l])), float(np.mean([p[1] for p in pts_l])))
        else:
            part_centroids[k_p] = (cx_b, cy_b)

    entry = {
        "sheet_name": sheet_name,
        "q_idx": int(q_idx) % 4,
        "is_agent": is_agent,
        "norm_size": 512,
        "gray": gray,
        "ink_alpha": ink_alpha,
        "ink_bin": ink_bin,
        "dist_map": dist_map,
        "sil_mask": sil_mask,
        "gouache_bgr": gouache_bgr,
        "strokes": strokes,
        "stroke_parts": stroke_parts,
        "construction_strokes": construction_strokes,
        "bbox": bbox,
        "spine_x": spine_x,
        "part_centroids": part_centroids,
    }
    _TRACED_CEL_CACHE[key] = entry
    return entry


def _build_anatomical_warp_maps(
    cel_a: dict,
    cel_b: dict = None,
    morph_t: float = 0.0,
    bob_phase: float = 0.0,
    boil_phase: int = 0,
    head_nod_deg: float = 0.0,
    arm_swing_deg: float = 0.0,
    squash_stretch: float = 0.0,
) -> tuple:
    """
    Constructs smooth `512x512` `cv2.remap` coordinate grids `(map_x, map_y)` plus a forward
    point transformer `transform_pt(x, y)` that applies:
      1. Volume-conserving squash-and-stretch breathing (`squash_stretch`),
      2. Smooth head nod/tilt around the neck pivot (`head_nod_deg`),
      3. Left & right arm curved-arc articulation (`arm_swing_deg`),
      4. Coherent part-aligned displacement toward `cel_b` (`morph_t in [0, 1]`) with a vertical
         breakdown arc lift (`-18 * sin(pi * morph_t)`),
      5. Authentic 6 Hz hand-drawn paper-tooth line boil (`boil_phase`).
    """
    N = 512
    bx0, by0, bx1, by1 = cel_a["bbox"]
    bw = max(32.0, float(bx1 - bx0))
    bh = max(32.0, float(by1 - by0))
    cx = 0.5 * (bx0 + bx1)
    cy = 0.5 * (by0 + by1)

    # Low-res 32x32 control grid upsampled via cubic interpolation for fast, C2-smooth diffeomorphic warp
    G = 32
    gy, gx = np.mgrid[0:G, 0:G].astype(np.float32) * (float(N - 1) / float(G - 1))
    u = (gx - cx) / bw
    v = (gy - cy) / bh

    # 1. Volume-conserving squash & stretch
    breath = 1.0 + squash_stretch + 0.015 * math.sin(bob_phase)
    sx = breath
    sy = 1.0 / breath
    dx_total = gx - (cx + u * bw * sx)
    dy_total = gy - (cy + v * bh * sy)

    if cel_a["is_agent"]:
        # 2. Smooth head nod & tilt around neck pivot `(cx, by0 + 0.50 * bh)`
        neck_x = cx
        neck_y = by0 + 0.50 * bh
        w_head = np.clip((neck_y - gy) / (0.35 * bh), 0.0, 1.0) ** 1.5
        ang_h = math.radians(head_nod_deg + 2.8 * math.sin(bob_phase * 1.2 + 0.4))
        ca_h, sa_h = math.cos(ang_h), math.sin(ang_h)
        rx_h = gx - neck_x
        ry_h = gy - neck_y
        rot_hx = neck_x + rx_h * ca_h - ry_h * sa_h
        rot_hy = neck_y + rx_h * sa_h + ry_h * ca_h
        dx_total += w_head * (gx - rot_hx)
        dy_total += w_head * (gy - rot_hy)

        # 3. Left & Right Arm articulation around bilateral shoulder pivots
        sh_ly = by0 + 0.54 * bh
        sh_lx = cx - 0.18 * bw
        sh_rx = cx + 0.18 * bw
        w_larm = np.clip((sh_lx - gx) / (0.32 * bw), 0.0, 1.0) * np.clip((gy - (by0 + 0.28 * bh)) / (0.25 * bh), 0.0, 1.0)
        w_rarm = np.clip((gx - sh_rx) / (0.32 * bw), 0.0, 1.0) * np.clip((gy - (by0 + 0.28 * bh)) / (0.25 * bh), 0.0, 1.0)

        ang_la = math.radians(arm_swing_deg + 4.5 * math.sin(bob_phase * 2.0))
        ca_l, sa_l = math.cos(ang_la), math.sin(ang_la)
        rlx, rly = gx - sh_lx, gy - sh_ly
        dx_total += w_larm * (rlx - (rlx * ca_l - rly * sa_l))
        dy_total += w_larm * (rly - (rlx * sa_l + rly * ca_l))

        ang_ra = math.radians(-arm_swing_deg - 5.2 * math.sin(bob_phase * 2.0 + 0.6))
        ca_r, sa_r = math.cos(ang_ra), math.sin(ang_ra)
        rrx, rry = gx - sh_rx, gy - sh_ly
        dx_total += w_rarm * (rrx - (rrx * ca_r - rry * sa_r))
        dy_total += w_rarm * (rry - (rrx * sa_r + rry * ca_r))

    # 4. Part-aligned curved-arc breakdown displacement toward `cel_b` when `morph_t > 0`
    if cel_b is not None and abs(morph_t) > 1e-3:
        arc_lift = -16.0 * math.sin(math.pi * abs(morph_t))
        pa = cel_a["part_centroids"]
        pb = cel_b["part_centroids"]
        # Soft radial basis blend of part displacements (head, torso, left_arm, right_arm)
        num_dx = np.zeros_like(gx)
        num_dy = np.zeros_like(gy)
        den_w = np.full_like(gx, 1e-4)
        for p_name, sigma in [("head", 110.0), ("torso", 140.0), ("left_arm", 105.0), ("right_arm", 105.0)]:
            ax, ay = pa.get(p_name, (cx, cy))
            bx, by = pb.get(p_name, (cx, cy))
            # Clamp max part shift so distinct poses warp cohesively without tearing
            ddx = float(np.clip((bx - ax) * morph_t, -52.0, 52.0))
            ddy = float(np.clip((by - ay) * morph_t + arc_lift * (1.2 if "arm" in p_name else 0.6), -44.0, 44.0))
            d2 = (gx - ax) ** 2 + (gy - ay) ** 2
            w_p = np.exp(-d2 / (2.0 * (sigma ** 2)))
            num_dx += w_p * ddx
            num_dy += w_p * ddy
            den_w += w_p
        dx_total -= (num_dx / den_w)
        dy_total -= (num_dy / den_w)

    # 5. Authentic 6 Hz hand-drawn paper-tooth line boil
    boil_amp = 0.85
    ang_b = gx * 0.075 + gy * 0.068 + boil_phase * 1.73
    dx_total += boil_amp * np.sin(ang_b)
    dy_total += boil_amp * np.cos(ang_b * 1.31)

    map_x_low = (gx + dx_total).astype(np.float32)
    map_y_low = (gy + dy_total).astype(np.float32)
    map_x = cv2.resize(map_x_low, (N, N), interpolation=cv2.INTER_CUBIC)
    map_y = cv2.resize(map_y_low, (N, N), interpolation=cv2.INTER_CUBIC)

    # Forward point transformer (inverse of backward remap displacement)
    dx_full = cv2.resize(dx_total, (N, N), interpolation=cv2.INTER_CUBIC)
    dy_full = cv2.resize(dy_total, (N, N), interpolation=cv2.INTER_CUBIC)

    def transform_pt(px: float, py: float) -> tuple:
        ix = int(np.clip(round(px), 0, N - 1))
        iy = int(np.clip(round(py), 0, N - 1))
        return (float(px - dx_full[iy, ix]), float(py - dy_full[iy, ix]))

    return map_x, map_y, transform_pt


def draw_active_fountain_pen_or_pencil(
    canvas: np.ndarray,
    tip_x: float,
    tip_y: float,
    is_blue_pencil: bool = False,
    scale: float = 1.0,
):
    """
    Draws the live moving animator's tool (Non-Photo Blue Col-Erase Pencil in Pass 1 or
    Speedball B-5 Fountain-Nib Dip Pen in Pass 2/3) actively tracing the reference strokes!
    """
    tx, ty = int(round(tip_x)), int(round(tip_y))
    ang = math.radians(-42.0)
    ux, uy = math.cos(ang), -math.sin(ang)
    nx, ny = -uy, ux

    L_nib = 22.0 * scale
    L_shaft = 64.0 * scale
    w_nib = 6.5 * scale

    if is_blue_pencil:
        # Sharpened Non-Photo Blue Col-Erase Pencil tip
        cone = np.array([
            [tx, ty],
            [int(round(tx + ux * L_nib + nx * w_nib * 0.7)), int(round(ty + uy * L_nib + ny * w_nib * 0.7))],
            [int(round(tx + ux * L_nib - nx * w_nib * 0.7)), int(round(ty + uy * L_nib - ny * w_nib * 0.7))],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [cone], BGR_SKIN, cv2.LINE_AA)
        cv2.polylines(canvas, [cone], True, BGR_INK, 2, cv2.LINE_AA)
        lead = np.array([
            [tx, ty],
            [int(round(tx + ux * 8 * scale + nx * 2.5 * scale)), int(round(ty + uy * 8 * scale + ny * 2.5 * scale))],
            [int(round(tx + ux * 8 * scale - nx * 2.5 * scale)), int(round(ty + uy * 8 * scale - ny * 2.5 * scale))],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [lead], BGR_NONPHOTO_BLUE, cv2.LINE_AA)
        bx1 = tx + ux * (L_nib + L_shaft)
        by1 = ty + uy * (L_nib + L_shaft)
        shaft = np.array([
            [int(round(tx + ux * L_nib + nx * w_nib * 0.7)), int(round(ty + uy * L_nib + ny * w_nib * 0.7))],
            [int(round(bx1 + nx * w_nib * 0.7)), int(round(by1 + ny * w_nib * 0.7))],
            [int(round(bx1 - nx * w_nib * 0.7)), int(round(by1 - ny * w_nib * 0.7))],
            [int(round(tx + ux * L_nib - nx * w_nib * 0.7)), int(round(ty + uy * L_nib - ny * w_nib * 0.7))],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [shaft], BGR_NONPHOTO_BLUE, cv2.LINE_AA)
        cv2.polylines(canvas, [shaft], True, BGR_INK, 2, cv2.LINE_AA)
    else:
        # Diamond-shaped brass Speedball B-5 Fountain-Nib Dip Pen with slit & breather hole
        nib_mid_x = tx + ux * (L_nib * 0.55)
        nib_mid_y = ty + uy * (L_nib * 0.55)
        nib_base_x = tx + ux * L_nib
        nib_base_y = ty + uy * L_nib
        nib_poly = np.array([
            [tx, ty],
            [int(round(nib_mid_x + nx * w_nib)), int(round(nib_mid_y + ny * w_nib))],
            [int(round(nib_base_x + nx * w_nib * 0.55)), int(round(nib_base_y + ny * w_nib * 0.55))],
            [int(round(nib_base_x - nx * w_nib * 0.55)), int(round(nib_base_y - ny * w_nib * 0.55))],
            [int(round(nib_mid_x - nx * w_nib)), int(round(nib_mid_y - ny * w_nib))],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [nib_poly], BGR_BRASS, cv2.LINE_AA)
        cv2.polylines(canvas, [nib_poly], True, BGR_INK, 2, cv2.LINE_AA)
        cv2.line(canvas, (tx, ty), (int(round(nib_mid_x)), int(round(nib_mid_y))), BGR_INK, 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(round(nib_mid_x)), int(round(nib_mid_y))), max(2, int(round(2.2 * scale))), BGR_INK, -1, cv2.LINE_AA)
        bx1 = tx + ux * (L_nib + L_shaft)
        by1 = ty + uy * (L_nib + L_shaft)
        shaft = np.array([
            [int(round(nib_base_x + nx * w_nib * 0.58)), int(round(nib_base_y + ny * w_nib * 0.58))],
            [int(round(bx1 + nx * w_nib * 0.25)), int(round(by1 + ny * w_nib * 0.25))],
            [int(round(bx1 - nx * w_nib * 0.25)), int(round(by1 - ny * w_nib * 0.25))],
            [int(round(nib_base_x - nx * w_nib * 0.58)), int(round(nib_base_y - ny * w_nib * 0.58))],
        ], dtype=np.int32)
        cv2.fillPoly(canvas, [shaft], BGR_APRON_RED, cv2.LINE_AA)
        cv2.polylines(canvas, [shaft], True, BGR_INK, 2, cv2.LINE_AA)


def render_traced_reference_cel_onto_canvas(
    canvas: np.ndarray,
    sheet_name: str,
    q_idx: int,
    cx: float,
    cy: float,
    target_w: float,
    target_h: float,
    pass_mode: str = "gouache",
    is_agent: bool = True,
    flip_x: bool = False,
    draw_prog: float = 1.0,
    bob_phase: float = 0.0,
    boil_phase: int = 0,
    head_nod_deg: float = 0.0,
    arm_swing_deg: float = 0.0,
    squash_stretch: float = 0.0,
    morph_to_sheet: str = None,
    morph_to_q: int = None,
    morph_t: float = 0.0,
    onion_prev_pose: tuple = None,
    onion_next_pose: tuple = None,
    blink_amount: float = 0.0,
    opacity: float = 1.0,
):
    """
    Renders a canonical hand-drawn reference pose `(sheet_name, q_idx)` directly traced from
    `assets/studio_doodles/<sheet_name>.jpg` onto `canvas` at `(cx, cy)` within `(target_w, target_h)`:

      - `pass_mode == "rough_blue"`:
          Pass 1 Animator's Non-Photo Blue construction lines + traced variable-width blue pencil
          strokes + Sepia/Blue Onion-Skin ghosting of `onion_prev_pose` / `onion_next_pose` +
          dotted crimson breakdown trajectory arcs!
      - `pass_mode == "clean_ink"`:
          Pass 2 Opaque cream bond paper cel silhouette + faint non-photo blue underdrawing +
          full-resolution variable-width Speedball B-5 India Ink traced contours!
      - `pass_mode == "gouache"`:
          Pass 3 Watertight topological gouache cel fills (warm skin, terracotta/blue/gold/green
          costume, beechwood desk, glowing lightbox disc, brass pegs) + full-resolution traced
          Speedball B-5 India Ink contours + subtle cel drop-shadow!
      - Supports progressive stroke-by-stroke animation (`draw_prog in [0.0, 1.0]`) with a live
        moving Dip-Pen Nib or Non-Photo Blue Pencil, plus smooth articulated breakdown/in-between
        deformation toward `(morph_to_sheet, morph_to_q)` at `morph_t in [0.0, 1.0]`.
    """
    cel_a = get_traced_reference_cel(sheet_name, q_idx, is_agent=is_agent)
    cel_b = None
    if morph_to_sheet is not None and morph_to_q is not None and 0.01 < morph_t < 0.99:
        cel_b = get_traced_reference_cel(morph_to_sheet, morph_to_q, is_agent=is_agent)

    # 1. Render Onion-Skin Ghosts first if in `rough_blue` (or when onion poses are supplied)
    if pass_mode == "rough_blue":
        for ghost_pose, ghost_col, dx_g in [
            (onion_prev_pose, BGR_ONION_PREV, -10.0),
            (onion_next_pose, BGR_ONION_NEXT, +10.0),
        ]:
            if ghost_pose is not None:
                g_sheet, g_q = ghost_pose
                g_cel = get_traced_reference_cel(g_sheet, g_q, is_agent=is_agent)
                _composite_single_traced_layer(
                    canvas,
                    g_cel,
                    cx=cx + dx_g,
                    cy=cy,
                    target_w=target_w,
                    target_h=target_h,
                    mode="onion_ghost",
                    ghost_color=ghost_col,
                    flip_x=flip_x,
                    map_x=None,
                    map_y=None,
                    transform_pt=None,
                    draw_prog=1.0,
                    opacity=0.38 * opacity,
                )

    # 2. Build smooth diffeomorphic anatomical deformation maps for `cel_a`
    map_x, map_y, tf_pt = _build_anatomical_warp_maps(
        cel_a,
        cel_b=cel_b,
        morph_t=morph_t if cel_b is not None else 0.0,
        bob_phase=bob_phase,
        boil_phase=boil_phase,
        head_nod_deg=head_nod_deg,
        arm_swing_deg=arm_swing_deg,
        squash_stretch=squash_stretch,
    )

    # If we are in an active breakdown/in-between (`0.15 < morph_t < 0.85`) and `cel_b` is present:
    # blend the forward-warped `cel_a` and backward-warped `cel_b` smoothly during the midpoint breakdown,
    # or show `cel_a` warping into `cel_b` with breakdown motion arcs!
    if cel_b is not None and 0.22 < morph_t < 0.78:
        map_xb, map_yb, tf_pt_b = _build_anatomical_warp_maps(
            cel_b,
            cel_b=cel_a,
            morph_t=(1.0 - morph_t),
            bob_phase=bob_phase,
            boil_phase=boil_phase,
            head_nod_deg=head_nod_deg,
            arm_swing_deg=arm_swing_deg,
            squash_stretch=squash_stretch,
        )
        if pass_mode == "rough_blue":
            # In Rough Blue pass, show both the departing and arriving warped breakdowns + trajectory arc!
            _composite_single_traced_layer(
                canvas, cel_a, cx, cy, target_w, target_h,
                mode="onion_ghost", ghost_color=BGR_ONION_PREV,
                flip_x=flip_x, map_x=map_x, map_y=map_y, transform_pt=tf_pt,
                draw_prog=1.0, opacity=0.48 * opacity,
            )
            _composite_single_traced_layer(
                canvas, cel_b, cx, cy, target_w, target_h,
                mode="rough_blue", ghost_color=BGR_NONPHOTO_BLUE,
                flip_x=flip_x, map_x=map_xb, map_y=map_yb, transform_pt=tf_pt_b,
                draw_prog=draw_prog, opacity=opacity,
            )
            return
        elif morph_t >= 0.50:
            cel_a, map_x, map_y, tf_pt = cel_b, map_xb, map_yb, tf_pt_b

    _composite_single_traced_layer(
        canvas,
        cel_a,
        cx=cx,
        cy=cy,
        target_w=target_w,
        target_h=target_h,
        mode=pass_mode,
        ghost_color=BGR_NONPHOTO_BLUE,
        flip_x=flip_x,
        map_x=map_x,
        map_y=map_y,
        transform_pt=tf_pt,
        draw_prog=draw_prog,
        opacity=opacity,
    )


def _composite_single_traced_layer(
    canvas: np.ndarray,
    cel: dict,
    cx: float,
    cy: float,
    target_w: float,
    target_h: float,
    mode: str,
    ghost_color: tuple,
    flip_x: bool,
    map_x: np.ndarray,
    map_y: np.ndarray,
    transform_pt,
    draw_prog: float,
    opacity: float = 1.0,
):
    H_c, W_c = canvas.shape[:2]
    N = cel["norm_size"]
    bx0, by0, bx1, by1 = cel["bbox"]
    bw = max(24.0, float(bx1 - bx0))
    bh = max(24.0, float(by1 - by0))
    scx = 0.5 * (bx0 + bx1)
    scy = 0.5 * (by0 + by1)

    scale = min(float(target_w) / bw, float(target_h) / bh)

    # Warp high-res `512x512` cel buffers through `(map_x, map_y)`
    if map_x is not None and map_y is not None:
        ink_alpha_w = cv2.remap(cel["ink_alpha"], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        sil_w = cv2.remap(cel["sil_mask"], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        gouache_w = cv2.remap(cel["gouache_bgr"], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=BGR_GUTTER)
    else:
        ink_alpha_w = cel["ink_alpha"].copy()
        sil_w = cel["sil_mask"].copy()
        gouache_w = cel["gouache_bgr"].copy()

    # If progressive stroke-by-stroke animation is active (`draw_prog < 0.995`), build a progressive
    # stroke reveal mask along the Inside-Out ordered variable-width strokes!
    active_tip_512 = None
    if draw_prog < 0.995:
        reveal_mask = np.zeros((N, N), dtype=np.float32)
        strokes = cel["strokes"]
        total_pts = sum(len(s) for s in strokes)
        max_pts = max(2, int(round(float(draw_prog) * total_pts)))
        drawn = 0
        for s in strokes:
            if drawn >= max_pts:
                break
            take = min(len(s), max_pts - drawn)
            pts_w = []
            for p in s[:take]:
                px, py = transform_pt(p[0], p[1]) if transform_pt is not None else (p[0], p[1])
                pts_w.append((px, py, p[2]))
            for a, b in zip(pts_w[:-1], pts_w[1:]):
                rad_px = max(4, int(round(max(a[2], b[2]) * 2.6 + 5)))
                cv2.line(
                    reveal_mask,
                    (int(round(a[0])), int(round(a[1]))),
                    (int(round(b[0])), int(round(b[1]))),
                    1.0,
                    rad_px * 2,
                    cv2.LINE_AA,
                )
            if pts_w:
                active_tip_512 = (pts_w[-1][0], pts_w[-1][1])
            drawn += take

        reveal_mask = cv2.GaussianBlur(reveal_mask, (9, 9), 2.2)
        reveal_mask = np.clip(reveal_mask, 0.0, 1.0)
    else:
        reveal_mask = None

    # Compute crop window in `512x512` space and destination ROI on `canvas`
    pad = 28
    x0_s = max(0, bx0 - pad)
    y0_s = max(0, by0 - pad)
    x1_s = min(N, bx1 + pad)
    y1_s = min(N, by1 + pad)
    sw = max(1, x1_s - x0_s)
    sh = max(1, y1_s - y0_s)

    dw = max(1, int(round(sw * scale)))
    dh = max(1, int(round(sh * scale)))

    # Anchor `spine_x` (for agents) or `scx` (for props) in `512x512` to `(cx, cy)` on `canvas`
    anchor_x = float(cel.get("spine_x", scx)) if cel.get("is_agent", False) else scx
    rel_cx = (anchor_x - x0_s) * scale
    rel_cy = (scy - y0_s) * scale
    if flip_x:
        rel_cx = dw - rel_cx

    dst_x0 = int(round(cx - rel_cx))
    dst_y0 = int(round(cy - rel_cy))
    dst_x1 = dst_x0 + dw
    dst_y1 = dst_y0 + dh

    # Clip against canvas bounds
    cx0 = max(0, dst_x0)
    cy0 = max(0, dst_y0)
    cx1 = min(W_c, dst_x1)
    cy1 = min(H_c, dst_y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return

    # Resize cropped `512x512` patches to `(dw, dh)`
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    ink_patch = cv2.resize(ink_alpha_w[y0_s:y1_s, x0_s:x1_s], (dw, dh), interpolation=interp)
    sil_patch = cv2.resize(sil_w[y0_s:y1_s, x0_s:x1_s].astype(np.float32) / 255.0, (dw, dh), interpolation=interp)
    gou_patch = cv2.resize(gouache_w[y0_s:y1_s, x0_s:x1_s], (dw, dh), interpolation=interp)
    rev_patch = (
        cv2.resize(reveal_mask[y0_s:y1_s, x0_s:x1_s], (dw, dh), interpolation=interp)
        if reveal_mask is not None
        else np.ones((dh, dw), dtype=np.float32)
    )

    if flip_x:
        ink_patch = cv2.flip(ink_patch, 1)
        sil_patch = cv2.flip(sil_patch, 1)
        gou_patch = cv2.flip(gou_patch, 1)
        rev_patch = cv2.flip(rev_patch, 1)

    sx0 = cx0 - dst_x0
    sy0 = cy0 - dst_y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    ink_r = np.clip(ink_patch[sy0:sy1, sx0:sx1] * 1.18, 0.0, 1.0)[..., None]
    sil_r = np.clip(sil_patch[sy0:sy1, sx0:sx1], 0.0, 1.0)[..., None]
    gou_r = gou_patch[sy0:sy1, sx0:sx1].astype(np.float32)
    rev_r = np.clip(rev_patch[sy0:sy1, sx0:sx1], 0.0, 1.0)[..., None]

    if mode == "onion_ghost":
        g_col = np.array(ghost_color, dtype=np.float32).reshape(1, 1, 3)
        alpha = ink_r * float(opacity)
        roi = roi * (1.0 - alpha) + g_col * alpha
        canvas[cy0:cy1, cx0:cx1] = np.clip(roi, 0, 255).astype(np.uint8)
        return

    if mode == "rough_blue":
        # 1. Draw Pass A/B Non-Photo Blue construction ovals/spine first
        _draw_transformed_construction_strokes(
            canvas, cel, x0_s, y0_s, dst_x0, dst_y0, dw, scale, flip_x, transform_pt,
            color_bgr=BGR_BLUE_FAINT, thickness=1,
        )
        # 2. Composite Non-Photo Blue traced strokes (revealed progressively by `rev_r`)
        blue_col = np.array(BGR_NONPHOTO_BLUE, dtype=np.float32).reshape(1, 1, 3)
        faint_col = np.array(BGR_BLUE_FAINT, dtype=np.float32).reshape(1, 1, 3)
        roi_after = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
        # Unrevealed strokes show as faint construction blue; revealed strokes show as bold Non-Photo Blue
        alpha_faint = ink_r * (1.0 - rev_r) * 0.35 * float(opacity)
        alpha_bold = ink_r * rev_r * 0.95 * float(opacity)
        roi_after = roi_after * (1.0 - alpha_faint) + faint_col * alpha_faint
        roi_after = roi_after * (1.0 - alpha_bold) + blue_col * alpha_bold
        canvas[cy0:cy1, cx0:cx1] = np.clip(roi_after, 0, 255).astype(np.uint8)

    elif mode == "clean_ink":
        # Pass 2: Opaque cream paper cel fill + crisp Speedball B-5 India Ink contours
        paper_col = np.array(BGR_GUTTER, dtype=np.float32).reshape(1, 1, 3)
        cel_fill_alpha = sil_r * (0.35 + 0.65 * float(draw_prog)) * float(opacity)
        roi = roi * (1.0 - cel_fill_alpha) + paper_col * cel_fill_alpha
        canvas[cy0:cy1, cx0:cx1] = np.clip(roi, 0, 255).astype(np.uint8)

        roi_after = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
        blue_under = np.array(BGR_NONPHOTO_BLUE, dtype=np.float32).reshape(1, 1, 3)
        ink_col = np.array(BGR_INK, dtype=np.float32).reshape(1, 1, 3)
        # Un-inked strokes show in non-photo blue underdrawing; inked strokes show in crisp Speedball India ink
        alpha_under = ink_r * (1.0 - rev_r) * 0.55 * float(opacity)
        alpha_ink = ink_r * rev_r * float(opacity)
        roi_after = roi_after * (1.0 - alpha_under) + blue_under * alpha_under
        roi_after = roi_after * (1.0 - alpha_ink) + ink_col * alpha_ink
        canvas[cy0:cy1, cx0:cx1] = np.clip(roi_after, 0, 255).astype(np.uint8)

    else:
        # Pass 3 (`gouache`): Subtle multiplane cel drop-shadow + watertight gouache cel fills + Speedball India Ink
        sh_dx = max(2, int(round(4 * scale)))
        sh_dy = max(2, int(round(5 * scale)))
        if (sy1 - sy0) > sh_dy and (sx1 - sx0) > sh_dx:
            sh_mask = np.zeros_like(sil_r)
            sh_mask[sh_dy:, sh_dx:] = sil_r[:-sh_dy, :-sh_dx]
            sh_alpha = sh_mask * (1.0 - sil_r) * 0.30 * float(opacity)
            sh_col = np.array((172, 188, 202), dtype=np.float32).reshape(1, 1, 3)
            roi = roi * (1.0 - sh_alpha) + sh_col * sh_alpha

        cel_fill_alpha = sil_r * (0.40 + 0.60 * float(draw_prog)) * float(opacity)
        roi = roi * (1.0 - cel_fill_alpha) + gou_r * cel_fill_alpha

        ink_col = np.array(BGR_INK, dtype=np.float32).reshape(1, 1, 3)
        blue_under = np.array(BGR_NONPHOTO_BLUE, dtype=np.float32).reshape(1, 1, 3)
        alpha_under = ink_r * (1.0 - rev_r) * 0.50 * float(opacity)
        alpha_ink = ink_r * rev_r * float(opacity)
        roi = roi * (1.0 - alpha_under) + blue_under * alpha_under
        roi = roi * (1.0 - alpha_ink) + ink_col * alpha_ink
        canvas[cy0:cy1, cx0:cx1] = np.clip(roi, 0, 255).astype(np.uint8)

    # Draw live moving Fountain-Nib Dip Pen or Non-Photo Blue Pencil when `draw_prog < 0.985`
    if active_tip_512 is not None and draw_prog < 0.985:
        tx_loc = (active_tip_512[0] - x0_s) * scale
        ty_loc = (active_tip_512[1] - y0_s) * scale
        if flip_x:
            tx_loc = dw - tx_loc
        draw_active_fountain_pen_or_pencil(
            canvas,
            dst_x0 + tx_loc,
            dst_y0 + ty_loc,
            is_blue_pencil=(mode == "rough_blue"),
            scale=max(0.65, min(1.2, scale)),
        )


def _draw_transformed_construction_strokes(
    canvas: np.ndarray,
    cel: dict,
    x0_s: int,
    y0_s: int,
    dst_x0: int,
    dst_y0: int,
    dw: int,
    scale: float,
    flip_x: bool,
    transform_pt,
    color_bgr: tuple,
    thickness: int = 1,
):
    for cs in cel.get("construction_strokes", []):
        pts = []
        for p in cs:
            px, py = transform_pt(p[0], p[1]) if transform_pt is not None else (p[0], p[1])
            lx = (px - x0_s) * scale
            ly = (py - y0_s) * scale
            if flip_x:
                lx = dw - lx
            pts.append([int(round(dst_x0 + lx)), int(round(dst_y0 + ly))])
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color_bgr, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    test_items = [
        ("nib_inking_keys", 0, True),
        ("nib_inking_keys", 1, True),
        ("nib_inking_keys", 2, True),
        ("nib_inking_keys", 3, True),
        ("agent_nib_keys", 0, True),
        ("agent_nib_keys", 3, True),
        ("apparatus_lightbox_anim_keys", 0, False),
        ("apparatus_lightbox_anim_keys", 2, False),
        ("blueline_lightbox_keys", 1, True),
        ("celinker_painting_keys", 3, True),
        ("sprocket_projector_keys", 0, True),
        ("studio_crew_celebration_keys", 1, True),
    ]
    rows = []
    for s_name, q_i, is_ag in test_items:
        row_cells = []
        # Col 0: Raw reference crop
        raw_crop = crop_reference_quadrant_highres(s_name, q_i, norm_size=260)
        row_cells.append(raw_crop)
        for p_mode, dp in [("rough_blue", 0.72), ("clean_ink", 1.0), ("gouache", 1.0)]:
            c = np.full((260, 260, 3), BGR_PAPER, dtype=np.uint8)
            prev_p = (s_name, (q_i - 1) % 4) if p_mode == "rough_blue" else None
            render_traced_reference_cel_onto_canvas(
                c, s_name, q_i, cx=130, cy=130, target_w=230, target_h=230,
                pass_mode=p_mode, is_agent=is_ag, draw_prog=dp,
                onion_prev_pose=prev_p,
            )
            row_cells.append(c)
        rows.append(np.hstack(row_cells))
    grid = np.vstack([
        np.hstack([rows[0], rows[1]]),
        np.hstack([rows[2], rows[3]]),
        np.hstack([rows[4], rows[5]]),
        np.hstack([rows[6], rows[7]]),
        np.hstack([rows[8], rows[9]]),
        np.hstack([rows[10], rows[11]]),
    ])
    out_p = ROOT / "preview" / "test_reference_traced_cels.png"
    cv2.imwrite(str(out_p), grid)
    print(f"Saved {out_p} ({grid.shape[1]}x{grid.shape[0]})")

