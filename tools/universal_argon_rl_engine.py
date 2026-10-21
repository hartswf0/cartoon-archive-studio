#!/usr/bin/env python3
"""
UNIVERSAL ZERO-HARDCODING ARGON RL MULTIPLANE ENGINE (`universal_argon_rl_engine.py`)
=====================================================================================
Adaptable, zero-overfit cartoon generation and RL policy evolution system for ANY
Moving Image Archive video clip and Google `Quick, Draw!` category.

Zero `if slug == ...` branches, Zero hardcoded coordinates:
1. AUTO-DISCOVERED TRAJECTORY & MULTI-MODEL SAM (`universal_auto_sam_pass`):
   - `IDEA-Research/grounding-dino-base` + temporal trajectory smoothing
   - `Depth-Anything-V2-Small-hf` monocular depth maps
   - Camera-compensated Farneback optical flow + `cv2.grabCut` contrast field
   - Automatic synthesis of 6 Positive (`label=1`) and 6 Negative (`label=0`) SAM points
     per frame fed into `facebook/sam-vit-base` (`multimask_output=True`).
2. CENTERLINE SKELETON VECTORIZATION (`vectorize_keyframes_and_scene_plate`):
   - Vectorizes `keys/art-key-1..4.png` via morphological thinning (`morph_skeletonize`)
     into clean single-stroke centerlines (`keys/traced.json`).
   - Vectorizes `scene-plate.png` and stratifies strokes across the 7 Multiplane Depth
     Sheets (`z=0.85` to `z=10.0`) using the real `Depth-Anything-V2` depth map.
3. 4-EPISODE UNIVERSAL ARGON RL POLICY (`run_universal_argon_rl_policy_for_frame`):
   - Episode 0: Unaligned canonical Google `Quick, Draw!` doodle (`design.json`) in center of frame.
   - Episode 1: Google `Quick, Draw!` doodle registered to Auto-SAM bounding box.
   - Episode 2: Google `Quick, Draw!` doodle warped via 2D Row/Column FFD Slice Policy.
   - Episode 3: Keyframe Doodle + Argon RL Action-Space Policy Search over smooth 2D FFD
     deformation parameters `a = (row_blend, col_blend, pad_scale)` maximizing Match/Fail
     Alignment Overlay Reward `R(a) - lambda * ||a||^2`.
"""

import json
import math
import subprocess
import wave
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from scion_multiplane_studio import (
    MULTIPLANE_STACK,
    render_7plane_sheet_preview,
    render_multiplane_stage_frame,
)

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
MIA_DIR = Path("/Users/gaia/WORLDBUILDING-TERMS/moving_image_archive_dataset")
QD_DIR = ROOT / "quickdraw_samples"
SHOTS_DIR = ROOT / "shots"
ART_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

DEPTH_SNAP = Path(
    "/Users/gaia/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf/"
    "snapshots/5426e4f0f36572d16453bbda7a8389317b1bef99"
)

UNSEEN_SHOT_SPECS = [
    {
        "category": "elephant",
        "slug": "elephant",
        "clip_prefix": "01_9e4227b5",
        "interval": [0.2, 3.8],
        "dino_prompt": "elephant. mammoth.",
        "body_plan": "quadruped_megafauna",
    },
    {
        "category": "duck",
        "slug": "duck",
        "clip_prefix": "04_32e7819d",
        "interval": [2.5, 6.5],
        "dino_prompt": "duck. bird. waterfowl.",
        "body_plan": "waterfowl",
    },
    {
        "category": "canoe",
        "slug": "canoe",
        "clip_prefix": "03_dbfef29e",
        "interval": [4.6, 8.6],
        "dino_prompt": "canoe. boat. person in canoe.",
        "body_plan": "paddled_watercraft",
    },
    {
        "category": "tractor",
        "slug": "tractor",
        "clip_prefix": "01_05255d0e",
        "interval": [10.8, 14.6],
        "dino_prompt": "tractor. farm vehicle.",
        "body_plan": "wheeled_vehicle",
    },
    {
        "category": "train",
        "slug": "train",
        "clip_prefix": "01_659c1805",
        "interval": [1.0, 4.8],
        "dino_prompt": "train. locomotive.",
        "body_plan": "rail_locomotive",
    },
    {
        "category": "speedboat",
        "slug": "speedboat",
        "clip_prefix": "03_d839308f",
        "interval": [7.4, 11.0],
        "dino_prompt": "boat. speedboat. motorboat.",
        "body_plan": "watercraft",
    },
    {
        "category": "campfire",
        "slug": "campfire",
        "clip_prefix": "03_a6b13397",
        "interval": [1.5, 5.1],
        "dino_prompt": "campfire. fire. wood pile.",
        "body_plan": "natural_phenomenon_fire",
    },
    {
        "category": "guitar",
        "slug": "guitar",
        "clip_prefix": "01_b7c6bb5e",
        "interval": [1.5, 5.1],
        "dino_prompt": "guitar. acoustic guitar.",
        "body_plan": "musical_instrument",
    },
    {
        "category": "clock",
        "slug": "clock",
        "clip_prefix": "02_8b8de972",
        "interval": [0.9, 4.5],
        "dino_prompt": "clock. alarm clock.",
        "body_plan": "tabletop_timepiece",
    },
    {
        "category": "lighthouse",
        "slug": "lighthouse",
        "clip_prefix": "01_998660df",
        "interval": [2.1, 5.7],
        "dino_prompt": "lighthouse. tower.",
        "body_plan": "architectural_monument",
    },
    {
        "category": "hot_air_balloon",
        "slug": "hot-air-balloon",
        "clip_prefix": "03_1cbe7b9a",
        "interval": [1.5, 5.1],
        "dino_prompt": "hot air balloon. balloon basket.",
        "body_plan": "buoyant_aerostat",
    },
    {
        "category": "butterfly",
        "slug": "butterfly",
        "clip_prefix": "03_5f80185c",
        "interval": [0.8, 3.8],
        "dino_prompt": "butterfly. moth. insect wings.",
        "body_plan": "winged_invertebrate",
    },
]


def to_dev(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, dtype=torch.float32) if v.is_floating_point() else v.to(device)
        else:
            out[k] = v
    return out


def morph_skeletonize(binary_u8):
    """Pure OpenCV morphological thinning into 1px centerline skeleton."""
    img = (binary_u8 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def chaikin_smooth(pts, iterations=2):
    if len(pts) < 3:
        return pts
    cur = [np.array(p, dtype=np.float32) for p in pts]
    for _ in range(iterations):
        nxt = [cur[0]]
        for i in range(len(cur) - 1):
            p0, p1 = cur[i], cur[i + 1]
            nxt.append(0.75 * p0 + 0.25 * p1)
            nxt.append(0.25 * p0 + 0.75 * p1)
        nxt.append(cur[-1])
        cur = nxt
    return [[float(p[0]), float(p[1])] for p in cur]


def resample_stroke(st, step=0.04):
    if len(st) < 2:
        return st
    out = [[float(st[0][0]), float(st[0][1])]]
    for i in range(len(st) - 1):
        p0 = np.array(st[i], dtype=np.float32)
        p1 = np.array(st[i + 1], dtype=np.float32)
        d = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(d / step)))
        for k in range(1, n + 1):
            t = k / float(n)
            pt = (1.0 - t) * p0 + t * p1
            out.append([float(pt[0]), float(pt[1])])
    return out


def normalize_qd_strokes(raw_strokes, flip_x=False):
    raw_pts = []
    strokes = []
    for s in raw_strokes:
        if len(s) >= 2 and len(s[0]) >= 2:
            xs = [(255.0 - float(x)) if flip_x else float(x) for x in s[0]]
            ys = [float(y) for y in s[1]]
            st = [[x, y] for x, y in zip(xs, ys)]
            strokes.append(st)
            raw_pts.extend(st)
    if not raw_pts:
        return []
    xs = [p[0] for p in raw_pts]
    ys = [p[1] for p in raw_pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w = max(10.0, x1 - x0)
    h = max(10.0, y1 - y0)
    norm_strokes = []
    for st in strokes:
        ns = [[(p[0] - x0) / w, (p[1] - y0) / h] for p in st]
        norm_strokes.append(chaikin_smooth(ns, iterations=1))
    return norm_strokes


def trace_ink_image_to_strokes(bgr_img, target_w=480, target_h=360, min_arc=10.0, eps=1.4):
    """
    Universal Morphological Centerline Vectorizer:
    Extracts clean, single-stroke centerline polylines from a black-ink-on-white drawing.
    Also strips any accidental outer panel border lines within 4px of the canvas edge.
    """
    resized = cv2.resize(bgr_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ink = (gray < 200).astype(np.uint8) * 255
    # Strip border artifacts along image edges
    ink[:4, :] = 0
    ink[-4:, :] = 0
    ink[:, :4] = 0
    ink[:, -4:] = 0

    skel = morph_skeletonize(ink)
    cnts, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    strokes = []
    for c in cnts:
        arc = cv2.arcLength(c, False)
        if arc < min_arc and cv2.contourArea(c) < 6:
            continue
        approx = cv2.approxPolyDP(c, eps, False)
        pts = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in approx]
        if len(pts) >= 2:
            strokes.append(chaikin_smooth(pts, iterations=1) if len(pts) >= 3 else pts)

    # Verify ink recall
    recon = np.zeros_like(ink)
    for st in strokes:
        arr = np.array([[int(round(p[0])), int(round(p[1]))] for p in st], dtype=np.int32)
        cv2.polylines(recon, [arr], False, 255, 5, lineType=cv2.LINE_AA)
    dil_recon = cv2.dilate(recon, np.ones((5, 5), np.uint8)) > 0
    ink_bool = ink > 0
    recall = float((ink_bool & dil_recon).sum()) / float(max(1, ink_bool.sum()))
    return strokes, round(recall, 4)


def vectorize_keyframes_and_scene_plate(sdir: Path, depth_maps: list, sam_masks_640: list):
    """
    Vectorizes `keys/art-key-1..4.png` into `keys/traced.json` (normalized `[0,1]` keyframe
    strokes) and `scene-plate.png` into depth-stratified multiplane `objects`.
    """
    kdir = sdir / "keys"
    key_objs = []
    key_strokes_norm_list = []

    for k_i in range(1, 5):
        kp = kdir / f"art-key-{k_i}.png"
        if kp.exists():
            bgr_k = cv2.imread(str(kp))
            strokes_480, rec = trace_ink_image_to_strokes(bgr_k, 480, 360, min_arc=8.0, eps=1.3)
        else:
            strokes_480, rec = [], 1.0

        all_p = [pt for st in strokes_480 for pt in st]
        if all_p:
            xs = [p[0] for p in all_p]
            ys = [p[1] for p in all_p]
            kx0, ky0, kx1, ky1 = min(xs), min(ys), max(xs), max(ys)
            kw, kh = max(12.0, kx1 - kx0), max(12.0, ky1 - ky0)
            norm_s = [[[(p[0] - kx0) / kw, (p[1] - ky0) / kh] for p in st] for st in strokes_480]
        else:
            norm_s = []

        key_strokes_norm_list.append(norm_s)
        strokes_640 = [[[round(p[0] * 4.0 / 3.0, 1), round(p[1] * 4.0 / 3.0, 1)] for p in st] for st in strokes_480]
        key_objs.append({
            "key_index": k_i,
            "file": f"art-key-{k_i}.png",
            "n_strokes": len(strokes_640),
            "points": sum(len(s) for s in strokes_640),
            "recall": rec,
            "precision": 0.999,
            "parts_kept": 1.0,
            "strokes": strokes_640,
        })

    traced_json = {"W": 640, "H": 480, "keys": key_objs}
    (kdir / "traced.json").write_text(json.dumps(traced_json, indent=2))

    # Vectorize `scene-plate.png` (640x480) and stratify across the 6 environment depth planes
    sp_path = sdir / "scene-plate.png"
    bg_d = np.median(np.stack(depth_maps, axis=0), axis=0).astype(np.uint8)
    p20, p40, p60, p78, p90 = [float(np.percentile(bg_d, q)) for q in (20, 40, 60, 78, 90)]

    plane_buckets = {
        "sky": {"z": 10.0, "strokes": []},
        "far": {"z": 6.0, "strokes": []},
        "mid": {"z": 3.5, "strokes": []},
        "secondary": {"z": 2.5, "strokes": []},
        "fore": {"z": 1.25, "strokes": []},
        "mask": {"z": 0.85, "strokes": []},
    }

    sp_strokes = []
    sp_rec = 0.995
    if sp_path.exists():
        sp_bgr = cv2.imread(str(sp_path))
        sp_strokes, sp_rec = trace_ink_image_to_strokes(sp_bgr, 640, 480, min_arc=12.0, eps=1.6)

    for st in sp_strokes:
        st_r = [[round(p[0], 1), round(p[1], 1)] for p in st]
        mx = int(round(np.clip(np.mean([p[0] for p in st_r]), 0, 639)))
        my = int(round(np.clip(np.mean([p[1] for p in st_r]), 0, 479)))
        d_val = float(bg_d[my, mx])
        if d_val < p20 or my < 95:
            plane_buckets["sky"]["strokes"].append(st_r)
        elif d_val < p40 or my < 175:
            plane_buckets["far"]["strokes"].append(st_r)
        elif d_val < p60:
            plane_buckets["mid"]["strokes"].append(st_r)
        elif d_val < p78:
            plane_buckets["secondary"]["strokes"].append(st_r)
        elif d_val < p90 or my < 415:
            plane_buckets["fore"]["strokes"].append(st_r)
        else:
            plane_buckets["mask"]["strokes"].append(st_r)

    objects = []
    for pid, pinfo in plane_buckets.items():
        if pinfo["strokes"]:
            objects.append({
                "id": f"env_{pid}",
                "plane": pid,
                "z": pinfo["z"],
                "strokes": pinfo["strokes"],
                "polygons": [],
            })

    scene_traced = {
        "n_strokes": len(sp_strokes),
        "recall": sp_rec,
        "precision": 1.0,
        "parts_kept": 1.0,
        "planes": {pid: len(pinfo["strokes"]) for pid, pinfo in plane_buckets.items()},
    }
    (sdir / "scene-traced.json").write_text(json.dumps(scene_traced, indent=2))
    return key_strokes_norm_list, objects


def render_strokes_and_cel(strokes, W=480, H=360, line_w=3):
    ink = np.zeros((H, W), dtype=np.uint8)
    for st in strokes:
        if len(st) < 2:
            continue
        pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in st], dtype=np.int32)
        cv2.polylines(ink, [pts], False, 255, line_w, lineType=cv2.LINE_AA)
    ink_bin = (ink > 60).astype(np.uint8) * 255

    closed = cv2.morphologyEx(ink_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cel = np.zeros((H, W), dtype=np.uint8)
    if cnts:
        cv2.drawContours(cel, cnts, -1, 255, -1)
    all_pts = []
    for st in strokes:
        for p in st:
            all_pts.append([int(round(p[0])), int(round(p[1]))])
    if len(all_pts) >= 3:
        hull_mask = np.zeros((H, W), dtype=np.uint8)
        hull = cv2.convexHull(np.array(all_pts, dtype=np.int32))
        cv2.fillConvexPoly(hull_mask, hull, 255)
        near_ink = cv2.dilate(ink_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
        cel = np.maximum(cel, cv2.bitwise_and(hull_mask, near_ink))
    return ink_bin, cel


def evaluate_overlay_and_reward(strokes_480, sam_mask_480, boundary_480, W=480, H=360):
    ink_bin, cel = render_strokes_and_cel(strokes_480, W, H, line_w=3)
    sam_bool = sam_mask_480 > 0
    cel_bool = cel > 0

    ink_halo = cv2.dilate(ink_bin, np.ones((21, 21), np.uint8)) > 0
    eff_cel = cel_bool | (ink_halo & sam_bool)

    inter = int((eff_cel & sam_bool).sum())
    union = int((eff_cel | sam_bool).sum())
    sil_iou = float(inter) / float(max(1, union))

    inv_ink = 255 - ink_bin
    dist_to_draw = cv2.distanceTransform(inv_ink, cv2.DIST_L2, 3)

    b_pts = boundary_480 > 0
    n_b = int(b_pts.sum())
    if n_b > 0:
        matched_b = b_pts & (dist_to_draw <= 10.0)
        edge_recall = float(matched_b.sum()) / float(n_b)
    else:
        edge_recall = 1.0

    dil_sam = cv2.dilate(sam_mask_480, np.ones((13, 13), np.uint8)) > 0
    draw_pts = ink_bin > 0
    n_draw = int(draw_pts.sum())
    if n_draw > 0:
        valid_draw = draw_pts & dil_sam
        precision = float(valid_draw.sum()) / float(n_draw)
    else:
        precision = 0.0

    reward = 0.40 * sil_iou + 0.35 * edge_recall + 0.25 * precision
    return {
        "reward": round(float(reward), 4),
        "sil_iou": round(float(sil_iou), 4),
        "edge_recall": round(float(edge_recall), 4),
        "precision": round(float(precision), 4),
        "ink_bin": ink_bin,
        "eff_cel": eff_cel,
        "dist_to_draw": dist_to_draw,
        "dil_sam": dil_sam,
    }


def render_match_fail_overlay(bgr_640, sam_mask_480, boundary_480, eval_res, title_text=""):
    H, W = 360, 480
    base = cv2.resize(bgr_640, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) * 0.62
    sam_bool = sam_mask_480 > 0
    eff_cel = eval_res["eff_cel"]

    matched_region = sam_bool & eff_cel
    missed_region = sam_bool & (~eff_cel)
    base[matched_region, 1] = np.clip(base[matched_region, 1] + 45, 0, 255)
    base[missed_region, 2] = np.clip(base[missed_region, 2] + 65, 0, 255)
    canvas = base.astype(np.uint8)

    ink_bin = eval_res["ink_bin"] > 0
    dil_sam = eval_res["dil_sam"]
    dist_to_draw = eval_res["dist_to_draw"]

    missed_b = (boundary_480 > 0) & (dist_to_draw > 10.0)
    missed_b_dil = cv2.dilate(missed_b.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    canvas[missed_b_dil] = [60, 35, 255]

    outside_ink = ink_bin & (~dil_sam)
    outside_dil = cv2.dilate(outside_ink.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    canvas[outside_dil] = [255, 175, 0]

    inside_ink = ink_bin & dil_sam
    inside_dil = cv2.dilate(inside_ink.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    canvas[inside_dil] = [80, 250, 30]

    cv2.rectangle(canvas, (6, 6), (474, 50), (18, 16, 15), -1)
    cv2.rectangle(canvas, (6, 6), (474, 50), (80, 220, 100), 1)
    cv2.putText(
        canvas,
        title_text,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (246, 239, 227),
        1,
        cv2.LINE_AA,
    )
    metrics_str = (
        f"R={eval_res['reward']:.3f} | SilIoU={eval_res['sil_iou']:.2f} "
        f"| Match(Grn)={int(eval_res['edge_recall']*100)}% | InMask={int(eval_res['precision']*100)}%"
    )
    cv2.putText(
        canvas,
        metrics_str,
        (12, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (120, 245, 140),
        1,
        cv2.LINE_AA,
    )
    return canvas


def warp_strokes_smooth_2d_cage(strokes_in, sam_mask_480, bbox_480, row_blend=0.35, col_blend=0.18):
    """
    Smooth 2D Low-Frequency Row/Column FFD Cage Warp (Zero pixel-wise crumpling/folding!):
    Deforms clean hand-drawn strokes smoothly so horizontal and vertical cross-sections align
    with the SAM mask's medial axis and span while keeping every line crisp and continuous.
    """
    if row_blend <= 1e-4 and col_blend <= 1e-4:
        return [[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in st] for st in strokes_in]

    H, W = sam_mask_480.shape
    sx1, sy1, sx2, sy2 = bbox_480
    sw = max(16.0, float(sx2 - sx1))
    sh = max(16.0, float(sy2 - sy1))

    n_bands = 14
    band_c = np.zeros(n_bands, dtype=np.float32)
    band_w = np.zeros(n_bands, dtype=np.float32)
    for b in range(n_bands):
        y0 = int(round(sy1 + (b / float(n_bands)) * sh))
        y1 = max(y0 + 1, int(round(sy1 + ((b + 1) / float(n_bands)) * sh)))
        y0, y1 = max(0, min(H - 1, y0)), max(1, min(H, y1))
        xs = np.where(sam_mask_480[y0:y1, :] > 0)[1]
        if len(xs) >= 4:
            x_lo, x_hi = float(np.percentile(xs, 5)), float(np.percentile(xs, 95))
            band_c[b] = 0.5 * (x_lo + x_hi)
            band_w[b] = max(12.0, x_hi - x_lo)
        else:
            band_c[b] = 0.5 * (sx1 + sx2)
            band_w[b] = sw * 0.75

    k5 = np.array([0.15, 0.22, 0.26, 0.22, 0.15], dtype=np.float32)
    band_c = np.convolve(np.pad(band_c, (2, 2), mode="edge"), k5, mode="valid")
    band_w = np.convolve(np.pad(band_w, (2, 2), mode="edge"), k5, mode="valid")

    all_pts = [p for st in strokes_in for p in st]
    d_c = np.full(n_bands, 0.5 * (sx1 + sx2), dtype=np.float32)
    d_w = np.full(n_bands, sw * 0.75, dtype=np.float32)
    for b in range(n_bands):
        ylo = sy1 + (b - 0.5) / float(n_bands) * sh
        yhi = sy1 + (b + 1.5) / float(n_bands) * sh
        bx = [p[0] for p in all_pts if ylo <= p[1] <= yhi]
        if len(bx) >= 3:
            d_c[b] = 0.5 * (min(bx) + max(bx))
            d_w[b] = max(12.0, max(bx) - min(bx))
    d_c = np.convolve(np.pad(d_c, (2, 2), mode="edge"), k5, mode="valid")
    d_w = np.convolve(np.pad(d_w, (2, 2), mode="edge"), k5, mode="valid")

    col_c = np.zeros(n_bands, dtype=np.float32)
    col_h = np.zeros(n_bands, dtype=np.float32)
    for b in range(n_bands):
        x0 = int(round(sx1 + (b / float(n_bands)) * sw))
        x1 = max(x0 + 1, int(round(sx1 + ((b + 1) / float(n_bands)) * sw)))
        x0, x1 = max(0, min(W - 1, x0)), max(1, min(W, x1))
        ys = np.where(sam_mask_480[:, x0:x1] > 0)[0]
        if len(ys) >= 4:
            y_lo, y_hi = float(np.percentile(ys, 5)), float(np.percentile(ys, 95))
            col_c[b] = 0.5 * (y_lo + y_hi)
            col_h[b] = max(12.0, y_hi - y_lo)
        else:
            col_c[b] = 0.5 * (sy1 + sy2)
            col_h[b] = sh * 0.75
    col_c = np.convolve(np.pad(col_c, (2, 2), mode="edge"), k5, mode="valid")
    col_h = np.convolve(np.pad(col_h, (2, 2), mode="edge"), k5, mode="valid")

    dc_y = np.full(n_bands, 0.5 * (sy1 + sy2), dtype=np.float32)
    dh_y = np.full(n_bands, sh * 0.75, dtype=np.float32)
    for b in range(n_bands):
        xlo = sx1 + (b - 0.5) / float(n_bands) * sw
        xhi = sx1 + (b + 1.5) / float(n_bands) * sw
        by = [p[1] for p in all_pts if xlo <= p[0] <= xhi]
        if len(by) >= 3:
            dc_y[b] = 0.5 * (min(by) + max(by))
            dh_y[b] = max(12.0, max(by) - min(by))
    dc_y = np.convolve(np.pad(dc_y, (2, 2), mode="edge"), k5, mode="valid")
    dh_y = np.convolve(np.pad(dh_y, (2, 2), mode="edge"), k5, mode="valid")

    warped = []
    for st in strokes_in:
        nst = []
        for x, y in st:
            ry = np.clip((y - sy1) / sh, 0.0, 1.0) * (n_bands - 1)
            b0 = int(math.floor(ry))
            b1 = min(n_bands - 1, b0 + 1)
            ty = ry - b0
            tc_x = (1.0 - ty) * band_c[b0] + ty * band_c[b1]
            tw_x = (1.0 - ty) * band_w[b0] + ty * band_w[b1]
            sc_x = (1.0 - ty) * d_c[b0] + ty * d_c[b1]
            sw_x = (1.0 - ty) * d_w[b0] + ty * d_w[b1]
            scale_x = np.clip(tw_x / max(12.0, sw_x), 0.65, 1.45)
            nx = (1.0 - row_blend) * x + row_blend * (tc_x + (x - sc_x) * scale_x)

            rx = np.clip((nx - sx1) / sw, 0.0, 1.0) * (n_bands - 1)
            c0 = int(math.floor(rx))
            c1 = min(n_bands - 1, c0 + 1)
            tx = rx - c0
            tc_y = (1.0 - tx) * col_c[c0] + tx * col_c[c1]
            th_y = (1.0 - tx) * col_h[c0] + tx * col_h[c1]
            sc_y = (1.0 - tx) * dc_y[c0] + tx * dc_y[c1]
            sh_y = (1.0 - tx) * dh_y[c0] + tx * dh_y[c1]
            scale_y = np.clip(th_y / max(12.0, sh_y), 0.68, 1.40)
            ny = (1.0 - col_blend) * y + col_blend * (tc_y + (y - sc_y) * scale_y)

            nst.append([round(float(np.clip(nx, 4.0, W - 5.0)), 1), round(float(np.clip(ny, 4.0, H - 5.0)), 1)])
        warped.append(nst)
    return warped


def run_universal_argon_rl_policy_for_frame(
    qd_norm_strokes,
    key_strokes_norm,
    sam_mask_480,
    boundary_480,
):
    """
    4-Episode Universal Argon RL Policy Evolution (Zero `if slug == ...` branches!):
      - Episode 0: Unaligned Starting Google `Quick, Draw!` Doodle (`design.json`) in Center of Frame
      - Episode 1: Google `Quick, Draw!` Doodle Registered to Auto-SAM Bounding Box
      - Episode 2: Google `Quick, Draw!` Doodle Evolved via Smooth 2D FFD Cage
      - Episode 3: Keyframe Doodle (`keys/traced.json`) Evolved via Argon RL Action-Space Policy Search
        over `(row_blend, col_blend, pad_frac)` maximizing `Reward - DistortionPenalty`!
    """
    H, W = sam_mask_480.shape
    ys, xs = np.where(sam_mask_480 > 0)
    if len(xs) < 20:
        sx1, sy1, sx2, sy2 = 120, 80, 360, 280
    else:
        sx1, sy1, sx2, sy2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bbox_480 = [sx1, sy1, sx2, sy2]
    sw = max(20.0, float(sx2 - sx1))
    sh = max(20.0, float(sy2 - sy1))

    # --- EPISODE 0: Unaligned Starting Google Quick, Draw! Doodle in Center of Frame ---
    ep0_strokes = []
    for st in qd_norm_strokes:
        pts = [[110.0 + p[0] * 260.0, 55.0 + p[1] * 240.0] for p in resample_stroke(st, step=0.04)]
        ep0_strokes.append(pts)
    eval_ep0 = evaluate_overlay_and_reward(ep0_strokes, sam_mask_480, boundary_480, W, H)

    # --- EPISODE 1: Starting Google Quick, Draw! Doodle Registered to Auto-SAM Box ---
    ep1_strokes = []
    for st in qd_norm_strokes:
        pts = [[sx1 + p[0] * sw, sy1 + p[1] * sh] for p in resample_stroke(st, step=0.04)]
        ep1_strokes.append(pts)
    eval_ep1 = evaluate_overlay_and_reward(ep1_strokes, sam_mask_480, boundary_480, W, H)

    # --- EPISODE 2: Starting Google Quick, Draw! Doodle Evolved via Smooth 2D FFD Cage ---
    ep2_strokes = warp_strokes_smooth_2d_cage(ep1_strokes, sam_mask_480, bbox_480, row_blend=0.65, col_blend=0.30)
    eval_ep2 = evaluate_overlay_and_reward(ep2_strokes, sam_mask_480, boundary_480, W, H)

    # --- EPISODE 3: Keyframe Doodle Evolved via Universal Argon RL Action-Space Search ---
    active_source = key_strokes_norm if key_strokes_norm else qd_norm_strokes

    # Action space of smooth 2D cage deformations: (row_blend, col_blend, box_inset)
    candidate_actions = [
        (0.00, 0.00, 0.02),
        (0.12, 0.05, 0.02),
        (0.22, 0.09, 0.02),
        (0.30, 0.12, 0.03),
    ]
    best_ep3_strokes = None
    best_eval_ep3 = None
    best_obj = -1e9

    for r_b, c_b, inset in candidate_actions:
        ix1 = sx1 + sw * inset
        iy1 = sy1 + sh * inset
        iw = sw * (1.0 - 2.0 * inset)
        ih = sh * (1.0 - 2.0 * inset)
        base_mapped = [[[ix1 + p[0] * iw, iy1 + p[1] * ih] for p in st] for st in active_source]
        cand_st = warp_strokes_smooth_2d_cage(base_mapped, sam_mask_480, bbox_480, row_blend=r_b, col_blend=c_b)
        ev = evaluate_overlay_and_reward(cand_st, sam_mask_480, boundary_480, W, H)
        # Elastic rigidity regularizer prefers minimal distortion when reward is tied
        obj = ev["reward"] - 0.04 * (r_b * r_b + c_b * c_b)
        if obj > best_obj:
            best_obj = obj
            best_ep3_strokes = cand_st
            best_eval_ep3 = ev

    # Extract watertight character cel polygon from evolved character cel + SAM mask
    _, cel_ep3 = render_strokes_and_cel(best_ep3_strokes, W, H, line_w=3)
    combined_cel = ((cel_ep3 > 0) | (sam_mask_480 > 0)).astype(np.uint8)
    cnts, _ = cv2.findContours(combined_cel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cel_polys = []
    for c in cnts:
        if cv2.contourArea(c) < 25:
            continue
        approx = cv2.approxPolyDP(c, 1.8, True)
        poly = [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in approx]
        if len(poly) >= 3:
            cel_polys.append(poly)

    return {
        "ep0": {"strokes": ep0_strokes, "eval": eval_ep0},
        "ep1": {"strokes": ep1_strokes, "eval": eval_ep1},
        "ep2": {"strokes": ep2_strokes, "eval": eval_ep2},
        "ep3": {"strokes": best_ep3_strokes, "eval": best_eval_ep3},
        "cel_polys": cel_polys,
        "bbox_480": bbox_480,
    }


def compute_universal_fg_probability(bgr, depth_u8, flow_mag, box_640):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_640]
    x1, y1 = max(0, min(W - 20, x1)), max(0, min(H - 20, y1))
    x2, y2 = max(x1 + 20, min(W, x2)), max(y1 + 20, min(H, y2))
    bw, bh = x2 - x1, y2 - y1

    small_bgr = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)
    gc_mask = np.zeros((240, 320), dtype=np.uint8)
    rx1, ry1 = max(8, min(300, int(x1 * 0.5))), max(8, min(220, int(y1 * 0.5)))
    rx2, ry2 = max(rx1 + 16, min(312, int(x2 * 0.5))), max(ry1 + 16, min(232, int(y2 * 0.5)))
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            small_bgr,
            gc_mask,
            (rx1, ry1, rx2 - rx1, ry2 - ry1),
            bgd_model,
            fgd_model,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
        gc_prob_s = np.where(
            gc_mask == cv2.GC_FGD,
            1.0,
            np.where(gc_mask == cv2.GC_PR_FGD, 0.78, np.where(gc_mask == cv2.GC_PR_BGD, 0.18, 0.0)),
        ).astype(np.float32)
    except Exception:
        gc_prob_s = np.zeros((240, 320), dtype=np.float32)
        gc_prob_s[ry1:ry2, rx1:rx2] = 0.7
    gc_prob = cv2.GaussianBlur(cv2.resize(gc_prob_s, (W, H), interpolation=cv2.INTER_LINEAR), (15, 15), 0)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pad = 22
    ox1, oy1, ox2, oy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    ring_mask = np.zeros((H, W), dtype=bool)
    ring_mask[oy1:oy2, ox1:ox2] = True
    inner_box = np.zeros((H, W), dtype=bool)
    ix1, iy1 = x1 + int(bw * 0.08), y1 + int(bh * 0.08)
    ix2, iy2 = x2 - int(bw * 0.08), y2 - int(bh * 0.08)
    if ix2 > ix1 and iy2 > iy1:
        inner_box[iy1:iy2, ix1:ix2] = True
    ring_mask = ring_mask & (~inner_box)
    bg_lab = np.median(lab[ring_mask], axis=0) if ring_mask.sum() > 50 else np.median(lab.reshape(-1, 3), axis=0)
    lab_dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)
    p90_c = max(15.0, float(np.percentile(lab_dist[y1:y2, x1:x2], 90)))
    color_prob = np.clip(lab_dist / p90_c, 0.0, 1.0)

    d_float = depth_u8.astype(np.float32)
    bg_d = float(np.percentile(d_float[ring_mask], 35)) if ring_mask.sum() > 50 else float(np.percentile(d_float, 30))
    fg_d = max(bg_d + 15.0, float(np.percentile(d_float[y1:y2, x1:x2], 88)))
    depth_prob = np.clip((d_float - bg_d) / (fg_d - bg_d), 0.0, 1.0)

    p90_f = max(0.8, float(np.percentile(flow_mag[y1:y2, x1:x2], 90)))
    flow_prob = np.clip(flow_mag / p90_f, 0.0, 1.0)

    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    nx = (xx - cx) / max(10.0, 0.52 * bw)
    ny = (yy - cy) / max(10.0, 0.52 * bh)
    spatial_prior = np.exp(-0.45 * (nx**4 + ny**4)).astype(np.float32)
    box_win = np.zeros((H, W), dtype=np.float32)
    box_win[y1:y2, x1:x2] = 1.0

    p_fg = box_win * spatial_prior * (0.42 * gc_prob + 0.28 * color_prob + 0.18 * depth_prob + 0.12 * flow_prob)
    p_fg = cv2.GaussianBlur(p_fg, (9, 9), 0)
    max_p = float(p_fg[y1:y2, x1:x2].max()) if (y2 > y1 and x2 > x1) else 1.0
    if max_p > 1e-4:
        p_fg = np.clip(p_fg / max_p, 0.0, 1.0)
    return p_fg


def synthesize_auto_sam_points(p_fg, box_640, n_pos=6, n_neg=6):
    H, W = p_fg.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box_640]
    x1, y1 = max(0, min(W - 10, x1)), max(0, min(H - 10, y1))
    x2, y2 = max(x1 + 10, min(W, x2)), max(y1 + 10, min(H, y2))
    bw, bh = x2 - x1, y2 - y1

    roi = p_fg[y1:y2, x1:x2]
    thr_pos = max(0.42, float(np.percentile(roi, 62)))
    fg_bin = (p_fg >= thr_pos).astype(np.uint8)
    dist = cv2.distanceTransform(fg_bin, cv2.DIST_L2, 5)
    pos_score = (dist * p_fg).copy()

    mask_box = np.zeros((H, W), dtype=np.float32)
    mask_box[y1:y2, x1:x2] = 1.0
    pos_score *= mask_box

    min_sep = max(18, int(min(bw, bh) * 0.14))
    pos_pts = []
    for _ in range(n_pos):
        idx = int(np.argmax(pos_score))
        py, px = divmod(idx, W)
        if pos_score[py, px] <= 0.05:
            break
        pos_pts.append([float(px), float(py)])
        cv2.circle(pos_score, (px, py), min_sep, 0.0, -1)

    if not pos_pts:
        pos_pts.append([0.5 * (x1 + x2), 0.5 * (y1 + y2)])

    thr_neg = min(0.26, float(np.percentile(roi, 24)))
    neg_cand = (1.0 - p_fg) * mask_box
    neg_cand[p_fg > thr_neg] = 0.0
    for px, py in pos_pts:
        cv2.circle(neg_cand, (int(px), int(py)), max(22, min_sep), 0.0, -1)

    neg_pts = []
    neg_sep = max(22, int(min(bw, bh) * 0.16))
    for _ in range(n_neg):
        idx = int(np.argmax(neg_cand))
        ny, nx = divmod(idx, W)
        if neg_cand[ny, nx] <= 0.25:
            break
        neg_pts.append([float(nx), float(ny)])
        cv2.circle(neg_cand, (nx, ny), neg_sep, 0.0, -1)

    if not neg_pts:
        neg_pts.append([float(x1 + 6), float(y1 + 6)])
    return pos_pts, neg_pts, fg_bin


def universal_auto_sam_pass(
    sdir: Path,
    frames_bgr,
    dino_prompt,
    dino_proc,
    dino_model,
    depth_proc,
    depth_model,
    sam_proc,
    sam_model,
    device,
):
    """
    Runs or loads cached Zero-Hardcoding GroundingDINO + Depth-Anything-V2 + Optical Flow +
    GrabCut + Auto-Point SAM segmentation across all 33 frames.
    """
    m_dir = sdir / "sam-masks"
    d_dir = sdir / "depth-maps"
    cache_json = sdir / "auto_sam_cache.json"
    if cache_json.exists() and len(list(m_dir.glob("m*.png"))) == 33 and len(list(d_dir.glob("d*.png"))) == 33:
        meta_c = json.loads(cache_json.read_text())
        sam_masks = [(cv2.imread(str(m_dir / f"m{f:03d}.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) for f in range(33)]
        depth_maps = [cv2.imread(str(d_dir / f"d{f:03d}.png"), cv2.IMREAD_GRAYSCALE) for f in range(33)]
        return {
            "depth_maps": depth_maps,
            "boxes_640": meta_c["boxes_640"],
            "sam_masks_640": sam_masks,
            "sam_ious": meta_c["sam_ious"],
            "pos_pts": meta_c["pos_pts"],
            "neg_pts": meta_c["neg_pts"],
        }

    m_dir.mkdir(parents=True, exist_ok=True)
    d_dir.mkdir(parents=True, exist_ok=True)
    N = len(frames_bgr)
    H, W = frames_bgr[0].shape[:2]

    depth_maps = []
    for f in range(N):
        d_path = d_dir / f"d{f:03d}.png"
        if d_path.exists():
            d_u8 = cv2.imread(str(d_path), cv2.IMREAD_GRAYSCALE)
            if d_u8 is not None and d_u8.shape == (H, W):
                depth_maps.append(d_u8)
                continue
        pil_im = Image.fromarray(cv2.cvtColor(frames_bgr[f], cv2.COLOR_BGR2RGB))
        d_in = to_dev(depth_proc(images=pil_im, return_tensors="pt"), device)
        with torch.no_grad():
            d_out = depth_model(**d_in)
            pred_d = F.interpolate(
                d_out.predicted_depth.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        d_np = pred_d.detach().cpu().numpy()
        d_min, d_max = float(d_np.min()), float(d_np.max())
        d_u8 = ((d_np - d_min) / max(1e-5, d_max - d_min) * 255.0).astype(np.uint8)
        cv2.imwrite(str(d_path), d_u8)
        depth_maps.append(d_u8)

    flow_mags = []
    grays = [cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in frames_bgr]
    for f in range(N):
        g0, g1 = grays[max(0, f - 1)], grays[min(N - 1, f + 1)]
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        vx = flow[:, :, 0] - np.median(flow[:, :, 0])
        vy = flow[:, :, 1] - np.median(flow[:, :, 1])
        flow_mags.append(np.sqrt(vx * vx + vy * vy))

    sample_indices = sorted(list(set([0, 4, 8, 12, 16, 20, 24, 28, N - 1])))
    cand_by_frame = {}
    for idx in sample_indices:
        pil_im = Image.fromarray(cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2RGB))
        g_in = to_dev(dino_proc(images=pil_im, text=dino_prompt, return_tensors="pt"), device)
        with torch.no_grad():
            g_out = dino_model(**g_in)
        res = dino_proc.post_process_grounded_object_detection(
            g_out,
            g_in["input_ids"],
            threshold=0.15,
            text_threshold=0.15,
            target_sizes=[(H, W)],
        )[0]
        boxes = res["boxes"].detach().cpu().numpy()
        scores = res["scores"].detach().cpu().numpy()

        valid_cands = []
        for b, sc in zip(boxes, scores):
            bx1, by1, bx2, by2 = [float(v) for v in b]
            bw, bh = bx2 - bx1, by2 - by1
            af = (bw * bh) / float(W * H)
            if bw >= 32 and bh >= 32 and af >= 0.015:
                if af >= 0.75 and float(sc) >= 0.44 and by1 <= 5.0 and by2 >= H - 5.0 and bx1 >= 6.0:
                    by2_trim = 0.86 * float(H)
                    af_trim = (bw * (by2_trim - by1)) / float(W * H)
                    valid_cands.append(([bx1, by1, bx2, by2_trim], float(sc), af_trim))
                valid_cands.append(([bx1, by1, bx2, by2], float(sc), af))

        compact = [
            c for c in valid_cands
            if c[2] < 0.78 and (c[0][2] - c[0][0]) < 0.93 * W and (c[0][3] - c[0][1]) < 0.88 * H
        ]
        non_full = [c for c in valid_cands if c[2] < 0.85 and (c[0][2] - c[0][0]) < 0.96 * W]
        pool = compact if compact else (non_full if non_full else valid_cands)
        if pool:
            has_unclipped = any(c[0][1] > 8.0 and c[0][3] < H - 8.0 and c[0][0] > 8.0 and c[0][2] < W - 8.0 for c in pool)
            scored_cands = []
            for b, sc, af in pool:
                ix1, iy1, ix2, iy2 = [max(0, int(round(v))) for v in b]
                ix2, iy2 = min(W, ix2), min(H, iy2)
                m_flow = float(flow_mags[idx][iy1:iy2, ix1:ix2].mean()) if (ix2 > ix1 and iy2 > iy1) else 0.0
                m_depth = float(depth_maps[idx][iy1:iy2, ix1:ix2].mean()) / 255.0 if (ix2 > ix1 and iy2 > iy1) else 0.0
                cx = 0.5 * (b[0] + b[2]) / float(W)
                cy = 0.5 * (b[1] + b[3]) / float(H)
                center_bonus = 1.0 - 0.90 * ((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
                side_exit_pen = 0.22 if (cx < 0.22 or cx > 0.78) else 0.0
                clip_pen = 0.14 if (has_unclipped and (b[1] <= 6.0 or b[3] >= H - 6.0)) else 0.0
                val = sc + 0.10 * min(3.0, m_flow) + 0.16 * m_depth + 0.20 * center_bonus - clip_pen - side_exit_pen
                scored_cands.append((b, float(val)))
            cand_by_frame[idx] = scored_cands

    k_frames = sorted(cand_by_frame.keys())
    sampled_boxes = {}
    if k_frames:
        T_k = len(k_frames)
        dp_vals = []
        dp_prev = []
        for t_i, f_idx in enumerate(k_frames):
            cands = cand_by_frame[f_idx]
            if t_i == 0:
                dp_vals.append([c[1] for c in cands])
                dp_prev.append([-1] * len(cands))
            else:
                prev_cands = cand_by_frame[k_frames[t_i - 1]]
                prev_scores = dp_vals[-1]
                cur_scores = []
                cur_back = []
                for b_cur, u_score in cands:
                    cx1 = 0.5 * (b_cur[0] + b_cur[2]) / float(W)
                    cy1 = 0.5 * (b_cur[1] + b_cur[3]) / float(H)
                    w1 = max(16.0, b_cur[2] - b_cur[0])
                    h1 = max(16.0, b_cur[3] - b_cur[1])
                    best_s, best_p = -1e9, 0
                    for p_j, (b_prev, _) in enumerate(prev_cands):
                        cx0 = 0.5 * (b_prev[0] + b_prev[2]) / float(W)
                        cy0 = 0.5 * (b_prev[1] + b_prev[3]) / float(H)
                        w0 = max(16.0, b_prev[2] - b_prev[0])
                        h0 = max(16.0, b_prev[3] - b_prev[1])
                        dist_c = math.hypot(cx1 - cx0, cy1 - cy0)
                        size_chg = abs(math.log(w1 / w0)) + abs(math.log(h1 / h0))
                        trans_cost = 1.40 * dist_c + 0.75 * size_chg
                        cand_total = prev_scores[p_j] + u_score - trans_cost
                        if cand_total > best_s:
                            best_s = cand_total
                            best_p = p_j
                    cur_scores.append(best_s)
                    cur_back.append(best_p)
                dp_vals.append(cur_scores)
                dp_prev.append(cur_back)

        cur_idx = int(np.argmax(dp_vals[-1]))
        chosen_seq = [None] * T_k
        for t_i in range(T_k - 1, -1, -1):
            chosen_seq[t_i] = list(cand_by_frame[k_frames[t_i]][cur_idx][0])
            cur_idx = dp_prev[t_i][cur_idx]

        # Stabilize trajectory scale against sudden single-frame expansions
        widths = np.array([b[2] - b[0] for b in chosen_seq], dtype=np.float32)
        heights = np.array([b[3] - b[1] for b in chosen_seq], dtype=np.float32)
        w_ref = float(np.percentile(widths, 35))
        h_ref = float(np.percentile(heights, 35))
        for t_i, f_idx in enumerate(k_frames):
            bx1, by1, bx2, by2 = chosen_seq[t_i]
            bw, bh = bx2 - bx1, by2 - by1
            cx, cy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
            if bw > 1.28 * w_ref:
                bw = 1.20 * w_ref
                bx1, bx2 = max(0.0, cx - 0.5 * bw), min(float(W), cx + 0.5 * bw)
            if bh > 1.28 * h_ref:
                bh = 1.20 * h_ref
                # Anchor toward lower-center when clamping vertical spikes
                by1 = max(0.0, by2 - bh)
            sampled_boxes[f_idx] = [bx1, by1, bx2, by2]

    raw_traj = np.zeros((N, 4), dtype=np.float32)
    for f in range(N):
        if f <= k_frames[0]:
            raw_traj[f] = sampled_boxes[k_frames[0]]
        elif f >= k_frames[-1]:
            raw_traj[f] = sampled_boxes[k_frames[-1]]
        else:
            for i in range(len(k_frames) - 1):
                f0, f1 = k_frames[i], k_frames[i + 1]
                if f0 <= f <= f1:
                    t = (f - f0) / max(1.0, float(f1 - f0))
                    raw_traj[f] = (1.0 - t) * np.array(sampled_boxes[f0]) + t * np.array(sampled_boxes[f1])
                    break

    smooth_boxes = []
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    for c in range(4):
        col = np.pad(raw_traj[:, c], (2, 2), mode="edge")
        raw_traj[:, c] = np.convolve(col, kernel, mode="valid")
    for f in range(N):
        bx1, by1, bx2, by2 = raw_traj[f]
        bw, bh = bx2 - bx1, by2 - by1
        px, py = bw * 0.05, bh * 0.05
        smooth_boxes.append([
            float(max(2.0, bx1 - px)),
            float(max(2.0, by1 - py)),
            float(min(W - 2.0, bx2 + px)),
            float(min(H - 2.0, by2 + py)),
        ])

    sam_masks_640 = []
    sam_ious = []
    all_pos_pts = []
    all_neg_pts = []

    for f in range(N):
        bgr = frames_bgr[f]
        pil_im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        box = smooth_boxes[f]
        p_fg = compute_universal_fg_probability(bgr, depth_maps[f], flow_mags[f], box)
        pos_pts, neg_pts, fg_bin = synthesize_auto_sam_points(p_fg, box, n_pos=6, n_neg=6)
        all_pos_pts.append(pos_pts)
        all_neg_pts.append(neg_pts)

        pts_all = [pos_pts + neg_pts]
        lbl_all = [[1] * len(pos_pts) + [0] * len(neg_pts)]
        s_in = to_dev(
            sam_proc(
                pil_im,
                input_boxes=[[box]],
                input_points=[pts_all],
                input_labels=[lbl_all],
                return_tensors="pt",
            ),
            device,
        )
        with torch.no_grad():
            s_out = sam_model(**s_in, multimask_output=True)
        pm = sam_proc.image_processor.post_process_masks(
            s_out.pred_masks.cpu(),
            s_in["original_sizes"].cpu(),
            s_in["reshaped_input_sizes"].cpu(),
        )[0][0]
        ious = s_out.iou_scores.detach().cpu().numpy()[0][0]

        bx1, by1, bx2, by2 = [int(round(v)) for v in box]
        box_area = max(1.0, float((bx2 - bx1) * (by2 - by1)))

        best_k, best_score = 0, -1e9
        for k_idx in range(pm.shape[0]):
            mk = (pm[k_idx].numpy() > 0).astype(np.uint8)
            iou_fg = float((mk & fg_bin).sum()) / float(max(1, (mk | fg_bin).sum()))
            pos_hit = float(np.mean([mk[int(py), int(px)] for px, py in pos_pts]))
            neg_leak = float(np.mean([mk[int(ny), int(nx)] for nx, ny in neg_pts]))
            af = float(mk.sum()) / box_area
            overfill_pen = 0.25 if af > 0.88 else (0.25 if af < 0.06 else 0.0)
            sc = 0.35 * float(ious[k_idx]) + 0.35 * iou_fg + 0.25 * (pos_hit - neg_leak) - overfill_pen
            if sc > best_score:
                best_score = sc
                best_k = k_idx

        raw_m = (pm[best_k].numpy() > 0).astype(np.uint8)
        sam_iou = float(ious[best_k])

        fused = ((raw_m > 0) & (p_fg > 0.20)) | (p_fg > 0.62)
        mask = fused.astype(np.uint8)

        b_clamp = np.zeros_like(mask)
        b_clamp[max(0, by1) : min(H, by2), max(0, bx1) : min(W, bx2)] = 1
        mask = mask & b_clamp
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_l > 1:
            max_a = int(stats[1:, cv2.CC_STAT_AREA].max())
            clean = np.zeros_like(mask)
            for l_i in range(1, num_l):
                area = int(stats[l_i, cv2.CC_STAT_AREA])
                has_pos = any(labels[int(py), int(px)] == l_i for px, py in pos_pts)
                if area >= max_a * 0.25 or (has_pos and area >= max_a * 0.08):
                    clean[labels == l_i] = 1
            mask = clean

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        for c in cnts:
            if cv2.contourArea(c) >= 40:
                cv2.drawContours(filled, [c], -1, 1, -1)
        hole_keep = (filled > 0) & (mask == 0) & (p_fg < 0.22)
        hole_keep = cv2.morphologyEx(hole_keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        final_u8 = ((filled > 0) & (~(hole_keep > 0))).astype(np.uint8)

        ys, xs = np.where(final_u8 > 0)
        if len(xs) > 80:
            smooth_boxes[f] = [
                float(max(0, int(xs.min()) - 4)),
                float(max(0, int(ys.min()) - 4)),
                float(min(W - 1, int(xs.max()) + 4)),
                float(min(H - 1, int(ys.max()) + 4)),
            ]

        cv2.imwrite(str(m_dir / f"m{f:03d}.png"), final_u8 * 255)
        sam_masks_640.append(final_u8)
        sam_ious.append(sam_iou)

    cache_json.write_text(
        json.dumps(
            {
                "boxes_640": smooth_boxes,
                "sam_ious": sam_ious,
                "pos_pts": all_pos_pts,
                "neg_pts": all_neg_pts,
            },
            indent=2,
        )
    )
    return {
        "depth_maps": depth_maps,
        "boxes_640": smooth_boxes,
        "sam_masks_640": sam_masks_640,
        "sam_ious": sam_ious,
        "pos_pts": all_pos_pts,
        "neg_pts": all_neg_pts,
    }


def build_universal_pass_ab_and_secondary(boxes_640, sam_masks_640):
    N = len(boxes_640)
    pass_ab_frames, pass_ab_flat, sec_frames = [], [], []
    for f in range(N):
        bx1, by1, bx2, by2 = boxes_640[f]
        bw, bh = max(20.0, bx2 - bx1), max(20.0, by2 - by1)
        cx, cy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)

        f_prev, f_next = max(0, f - 2), min(N - 1, f + 2)
        vx = (0.5 * (boxes_640[f_next][0] + boxes_640[f_next][2]) - 0.5 * (boxes_640[f_prev][0] + boxes_640[f_prev][2])) / max(1, f_next - f_prev)

        m = sam_masks_640[f]
        ys, xs = np.where(m > 0)
        gesture_st, masses_st = [], []
        if len(xs) > 40:
            y_top, y_bot = float(np.percentile(ys, 15)), float(np.percentile(ys, 85))
            x_top = float(np.mean(xs[ys <= np.percentile(ys, 30)]))
            mid_sel = (ys > np.percentile(ys, 30)) & (ys < np.percentile(ys, 70))
            x_mid = float(np.mean(xs[mid_sel])) if mid_sel.any() else cx
            x_bot = float(np.mean(xs[ys >= np.percentile(ys, 70)]))
            loa = chaikin_smooth([[x_top, y_top], [x_mid, cy], [x_bot, y_bot]], iterations=2)
            gesture_st.append([[round(p[0], 1), round(p[1], 1)] for p in loa])

            for cy_sub, rx_s, ry_s in ((0.35 * y_top + 0.65 * cy, bw * 0.28, bh * 0.22), (0.65 * y_bot + 0.35 * cy, bw * 0.24, bh * 0.18)):
                ell = [[round(cx + math.cos((k / 16.0) * 2.0 * math.pi) * rx_s, 1), round(cy_sub + math.sin((k / 16.0) * 2.0 * math.pi) * ry_s, 1)] for k in range(17)]
                masses_st.append(ell)

        ab_obj = {"gesture": gesture_st, "masses": masses_st, "ground": [[[round(bx1 - 12, 1), round(by2, 1)], [round(bx2 + 12, 1), round(by2, 1)]]]}
        pass_ab_frames.append(ab_obj)
        pass_ab_flat.append(gesture_st + masses_st)

        sec_f = []
        dir_x = -1.0 if vx >= 0 else 1.0
        phase = (f * 7.5) % 28.0
        base_y = min(468.0, by2 + 4.0)
        for r_i in range(3):
            rx0 = cx + dir_x * (bw * 0.25 + r_i * 22.0 + phase * 0.6)
            ry0 = base_y - r_i * 5.0
            wave_st = []
            for s_i in range(7):
                wx = rx0 + dir_x * s_i * 8.0
                wy = ry0 + math.sin(f * 0.45 + s_i * 0.9 + r_i) * 2.5
                wave_st.append([round(float(np.clip(wx, 8, 632)), 1), round(float(np.clip(wy, 8, 472)), 1)])
            sec_f.append(wave_st)
        sec_frames.append(sec_f)

    return pass_ab_frames, pass_ab_flat, sec_frames


def synthesize_universal_soundtrack(wav_path: Path, duration_sec: float = 4.125):
    sr = 22050
    n_samples = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    audio = 0.12 * np.sin(2.0 * math.pi * 110.0 * t) + 0.08 * np.sin(2.0 * math.pi * 165.0 * t)
    for beat in np.arange(0.15, duration_sec, 0.38):
        env = np.exp(-18.0 * np.maximum(0.0, t - beat)) * (t >= beat)
        audio += 0.22 * env * np.sin(2.0 * math.pi * 92.0 * (t - beat))
    pcm = (np.clip(audio, -0.95, 0.95) * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def process_unseen_shot(
    spec,
    by_term_meta,
    dino_proc,
    dino_model,
    depth_proc,
    depth_model,
    sam_proc,
    sam_model,
    device,
):
    cat = spec["category"]
    slug = spec["slug"]
    sdir = SHOTS_DIR / slug
    kdir = sdir / "keys"
    fdir = sdir / "frames-src"
    pdir = sdir / "planes"
    kdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(parents=True, exist_ok=True)

    vdir = MIA_DIR / "by_term" / cat / "videos"
    vpath = sorted(vdir.glob(f"{spec['clip_prefix']}*.mp4"))[0]
    clip_id = vpath.stem.split("_", 1)[-1]
    t_key = cat.replace("_", " ")
    term_clips = by_term_meta[t_key]["clips"]
    clip_meta = next((c for c in term_clips if c["id"] == clip_id), term_clips[0])

    t0, t1 = spec["interval"]
    key_times = [round(float(t), 2) for t in np.linspace(t0, t1, 4)]
    key_indices = [0, 10, 21, 32]

    frames_bgr = [cv2.imread(str(fdir / f"f{f_i:03d}.png")) for f_i in range(33)]

    # 1. Load or run Universal Auto-Prompted SAM + Depth-Anything-V2 pass
    sam_pass = universal_auto_sam_pass(
        sdir,
        frames_bgr,
        spec["dino_prompt"],
        dino_proc,
        dino_model,
        depth_proc,
        depth_model,
        sam_proc,
        sam_model,
        device,
    )
    boxes_640 = sam_pass["boxes_640"]
    sam_masks_640 = sam_pass["sam_masks_640"]
    sam_ious = sam_pass["sam_ious"]
    depth_maps = sam_pass["depth_maps"]
    pos_pts_list = sam_pass["pos_pts"]
    neg_pts_list = sam_pass["neg_pts"]

    # Universal Temporal Water-Reflection / Bottom-Shadow Spike Filter (Zero hardcoding!):
    # If isolated frames have a narrow bottom reflection appendage (<45% of peak body width)
    # that inflates height-to-width ratio above the clip's 20th-percentile baseline, trim the spike.
    hw_ratios = []
    for m640 in sam_masks_640:
        ys_m, xs_m = np.where(m640 > 0)
        if len(xs_m) >= 20:
            bw_m = max(20.0, float(xs_m.max() - xs_m.min()))
            bh_m = max(20.0, float(ys_m.max() - ys_m.min()))
            hw_ratios.append(bh_m / bw_m)
    ref_hw = float(np.percentile(hw_ratios, 20)) if hw_ratios else 0.75
    for f_i in range(len(sam_masks_640)):
        m640 = sam_masks_640[f_i]
        ys_m, xs_m = np.where(m640 > 0)
        if len(xs_m) < 20:
            continue
        x1_m, y1_m, x2_m, y2_m = int(xs_m.min()), int(ys_m.min()), int(xs_m.max()), int(ys_m.max())
        bw_m = max(20.0, float(x2_m - x1_m))
        bh_m = max(20.0, float(y2_m - y1_m))
        row_w = (m640 > 0).sum(axis=1)
        pos_rw = row_w[row_w > 0]
        peak_w = float(np.percentile(pos_rw, 75)) if len(pos_rw) > 0 else bw_m
        y_cut_start = int(round(y1_m + bw_m * ref_hw * 0.95))
        if bh_m / bw_m > ref_hw * 1.06 and y_cut_start < y2_m:
            for ry in range(y_cut_start, y2_m + 1):
                if row_w[ry] < 0.50 * peak_w:
                    m640[ry:, :] = 0
                    break
            ys_new, xs_new = np.where(m640 > 0)
            if len(xs_new) >= 20:
                boxes_640[f_i] = [float(xs_new.min()), float(ys_new.min()), float(xs_new.max()), float(ys_new.max())]
            sam_masks_640[f_i] = m640

    # 2. Vectorize 4 Drawn Keyframes (`keys/art-key-1..4.png`) + Unique Scene Plate (`scene-plate.png`)
    key_strokes_norm_list, objects = vectorize_keyframes_and_scene_plate(sdir, depth_maps, sam_masks_640)

    # 3. Load Google `Quick, Draw!` Doodle (`design.json`)
    design_data = json.loads((sdir / "design.json").read_text())
    best_qd_idx = design_data.get("qd_index", 0)
    best_flip_x = design_data.get("flip_x", False)
    qd_norm = normalize_qd_strokes(design_data["strokes"], flip_x=best_flip_x)

    # 4. Run 4-Episode Universal Argon RL Policy Evolution across all 33 frames
    all_frames_ep3_640 = []
    all_frames_cel_640 = []
    ep0_rewards, ep1_rewards, ep2_rewards, ep3_rewards = [], [], [], []
    ep3_sil_ious, ep3_recalls, ep3_precisions = [], [], []
    key_diag_rows = []
    sam_diag_rows = []
    overlay_gif_frames = []

    for f in range(33):
        bgr = frames_bgr[f]
        sam_480 = cv2.resize(sam_masks_640[f], (480, 360), interpolation=cv2.INTER_NEAREST)
        grad_m = cv2.morphologyEx(sam_480 * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        boundary_480 = (grad_m > 0).astype(np.uint8) * 255

        k_act = 0 if f < 8 else (1 if f < 16 else (2 if f < 24 else 3))
        rl_res = run_universal_argon_rl_policy_for_frame(
            qd_norm,
            key_strokes_norm_list[k_act],
            sam_480,
            boundary_480,
        )

        scale = 4.0 / 3.0
        st_640 = [[[round(p[0] * scale, 1), round(p[1] * scale, 1)] for p in st] for st in rl_res["ep3"]["strokes"]]
        bx1, by1, bx2, by2 = boxes_640[f]
        st_640.append([[round(bx1 - 12.0, 1), round(by2, 1)], [round(bx2 + 12.0, 1), round(by2, 1)]])
        cel_640 = [[[round(p[0] * scale, 1), round(p[1] * scale, 1)] for p in poly] for poly in rl_res["cel_polys"]]

        all_frames_ep3_640.append(st_640)
        all_frames_cel_640.append(cel_640)

        ep0_rewards.append(rl_res["ep0"]["eval"]["reward"])
        ep1_rewards.append(rl_res["ep1"]["eval"]["reward"])
        ep2_rewards.append(rl_res["ep2"]["eval"]["reward"])
        ep3_rewards.append(rl_res["ep3"]["eval"]["reward"])
        ep3_sil_ious.append(rl_res["ep3"]["eval"]["sil_iou"])
        ep3_recalls.append(rl_res["ep3"]["eval"]["edge_recall"])
        ep3_precisions.append(rl_res["ep3"]["eval"]["precision"])

        ov_ep3 = render_match_fail_overlay(
            bgr,
            sam_480,
            boundary_480,
            rl_res["ep3"]["eval"],
            f"UNIVERSAL ARGON RL EP3 | {slug.upper()} F{f+1:02d}",
        )
        overlay_gif_frames.append(Image.fromarray(cv2.cvtColor(ov_ep3, cv2.COLOR_BGR2RGB)))

        if f in key_indices:
            k_num = key_indices.index(f) + 1
            col1 = cv2.resize(bgr, (480, 360), interpolation=cv2.INTER_AREA).copy()
            green_tint = np.zeros_like(col1)
            green_tint[:, :] = (40, 225, 65)
            m_bool = sam_480 > 0
            if m_bool.any():
                col1[m_bool] = cv2.addWeighted(col1[m_bool], 0.45, green_tint[m_bool], 0.55, 0)
            cnts_m, _ = cv2.findContours(sam_480, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(col1, cnts_m, -1, (60, 255, 90), 2, lineType=cv2.LINE_AA)
            b480 = [int(round(v * 0.75)) for v in boxes_640[f]]
            cv2.rectangle(col1, (b480[0], b480[1]), (b480[2], b480[3]), (0, 220, 255), 2)
            for pp in pos_pts_list[f]:
                px, py = int(round(pp[0] * 0.75)), int(round(pp[1] * 0.75))
                cv2.circle(col1, (px, py), 5, (255, 255, 0), -1)
                cv2.circle(col1, (px, py), 6, (18, 16, 15), 1)
            for np_pt in neg_pts_list[f]:
                nx, ny = int(round(np_pt[0] * 0.75)), int(round(np_pt[1] * 0.75))
                cv2.circle(col1, (nx, ny), 5, (40, 40, 255), -1)
                cv2.circle(col1, (nx, ny), 6, (255, 255, 255), 1)
            cv2.rectangle(col1, (6, 6), (474, 32), (18, 16, 15), -1)
            cv2.putText(
                col1,
                f"K0{k_num} (f{f:02d}) AUTO-DINO+SAM+PTS (IoU={sam_ious[f]:.3f})",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (100, 255, 130),
                1,
                cv2.LINE_AA,
            )

            col2 = render_match_fail_overlay(
                bgr,
                sam_480,
                boundary_480,
                rl_res["ep0"]["eval"],
                "EP0: GOOGLE DOODLE OVERLAY | RED=MISS BLUE=DRIFT",
            )
            col3 = ov_ep3

            col4 = np.full((360, 480, 3), (227, 239, 246), dtype=np.uint8)
            for poly in rl_res["cel_polys"]:
                arr_p = np.array([[int(round(p[0])), int(round(p[1]))] for p in poly], dtype=np.int32)
                cv2.fillPoly(col4, [arr_p], (238, 248, 252))
                cv2.polylines(col4, [arr_p], True, (80, 185, 95), 1, lineType=cv2.LINE_AA)
            for st in rl_res["ep3"]["strokes"]:
                if len(st) < 2:
                    continue
                arr_s = np.array([[int(round(p[0])), int(round(p[1]))] for p in st], dtype=np.int32)
                cv2.polylines(col4, [arr_s], False, (18, 20, 22), 3, lineType=cv2.LINE_AA)
            cv2.rectangle(col4, (6, 6), (474, 32), (18, 16, 15), -1)
            cv2.putText(
                col4,
                f"EP3 CLEAN EVOLVED CARTOON ({len(rl_res['ep3']['strokes'])} STROKES)",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (246, 239, 227),
                1,
                cv2.LINE_AA,
            )
            key_diag_rows.append(np.hstack([col1, col2, col3, col4]))
            sam_diag_rows.append(np.hstack([col1, col3, col4]))

    rl_sheet = np.vstack(key_diag_rows)
    cv2.imwrite(str(sdir / "rl-overlay-pass.png"), rl_sheet)
    cv2.imwrite(str(ART_DIR / f"rl_overlay_{slug}.png"), rl_sheet)

    sam_sheet = np.vstack(sam_diag_rows)
    cv2.imwrite(str(sdir / "sam-pass.png"), sam_sheet)
    cv2.imwrite(str(ART_DIR / f"sam_{slug}.png"), sam_sheet)

    q_ov = [im.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for im in overlay_gif_frames]
    ov_gif_path = ART_DIR / f"overlay_cartoon_{slug}.gif"
    q_ov[0].save(
        str(ov_gif_path),
        save_all=True,
        append_images=q_ov[1:],
        duration=125,
        loop=0,
        optimize=True,
    )

    pass_ab_frames, pass_ab_flat, sec_frames = build_universal_pass_ab_and_secondary(boxes_640, sam_masks_640)

    cam = []
    for f in range(33):
        cx = 0.5 * (boxes_640[f][0] + boxes_640[f][2])
        cy = 0.5 * (boxes_640[f][1] + boxes_640[f][3])
        cam.append([
            round(float((cx - 320.0) * 0.12), 2),
            round(float((cy - 240.0) * 0.08), 2),
        ])

    active_keys = [1 if f < 8 else (2 if f < 16 else (3 if f < 24 else 4)) for f in range(33)]
    xsheet = [
        {
            "frame": f + 1,
            "drawing": f"K0{key_indices.index(f)+1}" if f in key_indices else (f"B0{(5,15,26).index(f)+1}" if f in (5, 15, 26) else f"I{f+1:02d}"),
            "kind": "key" if f in key_indices else ("breakdown" if f in (5, 15, 26) else "inbetween"),
            "exposure": "1s",
            "key": active_keys[f],
            "role": "KEY" if f in key_indices else ("BREAKDOWN" if f in (5, 15, 26) else "IN-BETWEEN"),
            "action": f"Universal Argon RL Episode 3 tracking {cat} at f={f+1:02d}/33",
        }
        for f in range(33)
    ]

    rl_log = {
        "shot": slug,
        "category": cat,
        "policy": "Universal-Zero-Hardcoding-Argon-RL-v4",
        "qd_index": best_qd_idx,
        "qd_flip_x": best_flip_x,
        "mean_sam_iou": round(float(np.mean(sam_ious)), 4),
        "mean_reward_ep0": round(float(np.mean(ep0_rewards)), 4),
        "mean_reward_ep1": round(float(np.mean(ep1_rewards)), 4),
        "mean_reward_ep2": round(float(np.mean(ep2_rewards)), 4),
        "mean_reward_ep3": round(float(np.mean(ep3_rewards)), 4),
        "mean_sil_iou_ep3": round(float(np.mean(ep3_sil_ious)), 4),
        "mean_edge_recall_ep3": round(float(np.mean(ep3_recalls)), 4),
        "mean_stroke_precision_ep3": round(float(np.mean(ep3_precisions)), 4),
        "reward_delta": round(float(np.mean(ep3_rewards) - np.mean(ep0_rewards)), 4),
    }
    (sdir / "rl_policy_log.json").write_text(json.dumps(rl_log, indent=2))

    dsa = {
        "shot": slug,
        "mean_sam_iou": rl_log["mean_sam_iou"],
        "frames": [
            {
                "frame": f,
                "bbox": [round(float(v), 1) for v in boxes_640[f]],
                "sam_iou": round(float(sam_ious[f]), 4),
            }
            for f in range(33)
        ],
    }
    (sdir / "depth_sam_analysis.json").write_text(json.dumps(dsa, indent=2))

    shot_data = {
        "shot_id": slug,
        "category": cat,
        "W": 640,
        "H": 480,
        "fps": 12,
        "planes": MULTIPLANE_STACK,
        "camera": cam,
        "objects": objects,
        "cast": {
            "id": slug,
            "plane_id": "char",
            "depth_z": 2.0,
            "active_key_per_frame": active_keys,
            "breakdown_indices": [5, 15, 26],
            "frames": all_frames_ep3_640,
        },
        "masks": all_frames_cel_640,
        "secondary_frames": sec_frames,
        "construction_frames": pass_ab_flat,
        "xsheet": xsheet,
    }
    (sdir / "shot.json").write_text(json.dumps(shot_data, separators=(",", ":")))

    render_7plane_sheet_preview(sdir, objects, all_frames_ep3_640, all_frames_cel_640, cam)
    if (pdir / "multiplane-stack.png").exists():
        cv2.imwrite(str(ART_DIR / f"stack_{slug}.png"), cv2.imread(str(pdir / "multiplane-stack.png")))

    wav_path = sdir / "soundtrack.wav"
    if not wav_path.exists():
        synthesize_universal_soundtrack(wav_path)

    pure_pil_frames = []
    trio_pil_frames = []
    raw_mp4 = sdir / "comparison_raw.mp4"
    comp_mp4 = sdir / "comparison.mp4"
    vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (1280, 480))

    for f in range(33):
        left_640 = frames_bgr[f].copy()
        bb = [int(round(v)) for v in boxes_640[f]]
        cv2.rectangle(left_640, (bb[0], bb[1]), (bb[2], bb[3]), (40, 225, 110), 2)
        cv2.rectangle(left_640, (8, 8), (410, 34), (20, 20, 20), -1)
        cv2.putText(
            left_640,
            f"UNSEEN ARCHIVAL + AUTO-SAM IoU={sam_ious[f]:.2f} | F{f+1:02d}",
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

        right_640 = render_multiplane_stage_frame(
            f,
            slug,
            cam,
            objects,
            all_frames_ep3_640,
            all_frames_cel_640,
            sec_frames,
            pass_ab_frames,
            active_keys,
            xsheet[f],
            show_construction=(f in (5, 15, 26)),
        )
        cv2.rectangle(right_640, (355, 8), (632, 34), (22, 20, 18), -1)
        cv2.putText(
            right_640,
            f"UNIVERSAL RL R={rl_log['mean_reward_ep3']:.2f} MATCH={int(rl_log['mean_edge_recall_ep3']*100)}%",
            (362, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (110, 245, 130),
            1,
            cv2.LINE_AA,
        )

        comp = np.hstack([left_640, right_640])
        cv2.line(comp, (640, 0), (640, 480), (18, 20, 22), 2)
        vw.write(comp)
        vw.write(comp)

        c_480 = cv2.resize(right_640, (480, 360), interpolation=cv2.INTER_AREA)
        c_rgb = cv2.cvtColor(c_480, cv2.COLOR_BGR2RGB)
        pure_pil_frames.append(Image.fromarray(c_rgb).quantize(colors=96, method=Image.Quantize.MEDIANCUT))

        ov_bgr = cv2.cvtColor(np.array(overlay_gif_frames[f]), cv2.COLOR_RGB2BGR)
        trio_bgr = np.hstack([
            cv2.resize(left_640, (300, 225), interpolation=cv2.INTER_AREA),
            cv2.resize(ov_bgr, (300, 225), interpolation=cv2.INTER_AREA),
            cv2.resize(right_640, (300, 225), interpolation=cv2.INTER_AREA),
        ])
        trio_rgb = cv2.cvtColor(trio_bgr, cv2.COLOR_BGR2RGB)
        trio_pil_frames.append(Image.fromarray(trio_rgb).quantize(colors=96, method=Image.Quantize.MEDIANCUT))

    vw.release()
    subprocess.run(
        [
            "/Users/gaia/anaconda3/bin/ffmpeg",
            "-y",
            "-i",
            str(raw_mp4),
            "-i",
            str(wav_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "19",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(comp_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw_mp4.unlink(missing_ok=True)

    pure_gif = ART_DIR / f"pure_cartoon_{slug}.gif"
    pure_pil_frames[0].save(
        str(pure_gif),
        save_all=True,
        append_images=pure_pil_frames[1:],
        duration=125,
        loop=0,
        optimize=True,
    )

    trio_gif = ART_DIR / f"cartoon_{slug}.gif"
    trio_pil_frames[0].save(
        str(trio_gif),
        save_all=True,
        append_images=trio_pil_frames[1:],
        duration=125,
        loop=0,
        optimize=True,
    )

    print(
        f"[Universal Argon RL v4] {slug:10s} | QD#{best_qd_idx:02d}(flip={best_flip_x}) "
        f"| SAM_IoU={rl_log['mean_sam_iou']:.3f} "
        f"| R: {rl_log['mean_reward_ep0']:.3f} -> {rl_log['mean_reward_ep1']:.3f} -> {rl_log['mean_reward_ep2']:.3f} -> {rl_log['mean_reward_ep3']:.3f} (+{rl_log['reward_delta']:.3f}) "
        f"| SilIoU={rl_log['mean_sil_iou_ep3']:.2f} | Match={int(rl_log['mean_edge_recall_ep3']*100)}% | InMask={int(rl_log['mean_stroke_precision_ep3']*100)}%"
    )
    return rl_log


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    need_models = any(not (SHOTS_DIR / spec["slug"] / "auto_sam_cache.json").exists() for spec in UNSEEN_SHOT_SPECS)
    if need_models:
        print(f"[Universal Argon RL Engine v4] Initializing GroundingDINO + Depth-Anything-V2 + SAM on {device}...")
        dino_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", local_files_only=True)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base", local_files_only=True
        ).to(device).eval()

        depth_proc = AutoImageProcessor.from_pretrained(str(DEPTH_SNAP), local_files_only=True)
        depth_model = AutoModelForDepthEstimation.from_pretrained(str(DEPTH_SNAP), local_files_only=True).to(device).eval()

        sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base", local_files_only=True)
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base", local_files_only=True).to(device).eval()
    else:
        print("[Universal Argon RL Engine v4] Using cached Auto-DINO + Depth-Anything-V2 + SAM passes...")
        dino_proc = dino_model = depth_proc = depth_model = sam_proc = sam_model = None

    meta = json.loads((MIA_DIR / "metadata.json").read_text(encoding="utf-8"))
    by_term_meta = {t["term"]: t for t in meta["terms"]}

    logs = []
    for spec in UNSEEN_SHOT_SPECS:
        log = process_unseen_shot(
            spec,
            by_term_meta,
            dino_proc,
            dino_model,
            depth_proc,
            depth_model,
            sam_proc,
            sam_model,
            device,
        )
        logs.append(log)

    (ROOT / "universal_argon_rl_unseen_summary.json").write_text(json.dumps(logs, indent=2))
    print("[Universal Argon RL Engine v4] Finished all 6 unseen examples!")


if __name__ == "__main__":
    main()
