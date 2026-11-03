#!/usr/bin/env python3
"""
tools/spatiotemporal_worm_analyzer.py
=====================================
Analyzes all 346 shots in the Cartoon Archive Studio as 3D Spatiotemporal Volumes V(u, v, tau)
following:
  1. Fels & Mase (2000), "Interactive Video Cubism" & Video Summagator (u-tau and v-tau tomography)
  2. Action Tubes / 3D Spatiotemporal Worms c(tau) = (u(tau), v(tau), tau) in [0,1]^2 x [0,T]
  3. KALMUS Color Toolkit & Johannes Itten's 7 Color Contrasts (Core Worm vs. Outer Shell in CIE L*a*b*)
  4. Bouleau's 14-Line Harmonic Armature, Rule of Thirds & Golden Spiral (Grid-Tunnel Resonance & Arnheim Torque)
  5. 3D Space-Curve Differential Geometry: Curvature kappa(tau), Torsion tau_t(tau), and Kinetic Jerk ||c'''(tau)||
     with Leonxlnx/earshot Foley SFX cue timestamps
  6. Four Canonical Visualizations:
     - worm_slitscan_tomography_atlas.png (u-tau and v-tau tomographic slit-scan slices)
     - worm_3d_glass_prism_atlas.png (3D isometric glass-prism volume rendering with 14-Line Armature filaments)
     - worm_cinemetrics_torus_atlas.png (Frederic Brodbeck 360-deg Radial Cinemetrics Torus rings)
     - worm_manovich_montage_grid.png (Lev Manovich Cultural Analytics Montage Grid + Bruce Block Intensity Graph)
"""

import json
import math
import os
import shutil
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
SHOTS_DIR = ROOT / "shots"
PREVIEW_DIR = ROOT / "preview"
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Bouleau's 14-Line Harmonic Armature + Rule of Thirds + Golden Ratio lines in normalized (u, v) in [0, 1]^2
ARMATURE_14_LINES = [
    (0.0, 0.0, 1.0, 1.0, "sinister_diagonal"),
    (0.0, 1.0, 1.0, 0.0, "baroque_diagonal"),
    (0.0, 0.5, 0.5, 0.0, "diamond_tl"),
    (0.5, 0.0, 1.0, 0.5, "diamond_tr"),
    (1.0, 0.5, 0.5, 1.0, "diamond_br"),
    (0.5, 1.0, 0.0, 0.5, "diamond_bl"),
    (0.0, 0.0, 0.5, 1.0, "reciprocal_l1"),
    (0.0, 1.0, 0.5, 0.0, "reciprocal_l2"),
    (1.0, 0.0, 0.5, 1.0, "reciprocal_r1"),
    (1.0, 1.0, 0.5, 0.0, "reciprocal_r2"),
    (0.5, 0.0, 0.5, 1.0, "vertical_center"),
    (0.0, 0.5, 1.0, 0.5, "horizontal_center"),
    (0.3333, 0.0, 0.3333, 1.0, "thirds_v_left"),
    (0.6667, 0.0, 0.6667, 1.0, "thirds_v_right"),
    (0.0, 0.3333, 1.0, 0.3333, "thirds_h_top"),
    (0.0, 0.6667, 1.0, 0.6667, "thirds_h_bottom"),
    (0.6180, 0.0, 0.6180, 1.0, "golden_v_phi"),
    (0.0, 0.3820, 1.0, 0.3820, "golden_h_phi"),
]

GOLDEN_NODE = np.array([0.6180, 0.3820], dtype=np.float64)

EARSHOT_SFX_BANK = [
    "earshot:whoosh_whip_04",
    "earshot:footstep_gravel_12",
    "earshot:wood_knock_impact_07",
    "earshot:brush_swish_accent_19",
    "earshot:cloth_snap_kinetic_03",
    "earshot:spring_boing_cartoon_11",
    "earshot:metal_clink_transient_08",
    "earshot:water_splash_plunge_02",
]


def point_to_segment_dist(u: float, v: float, u1: float, v1: float, u2: float, v2: float) -> float:
    dx = u2 - u1
    dy = v2 - v1
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(u - u1, v - v1)
    t = max(0.0, min(1.0, ((u - u1) * dx + (v - v1) * dy) / l2))
    proj_u = u1 + t * dx
    proj_v = v1 + t * dy
    return math.hypot(u - proj_u, v - proj_v)


def min_armature_distance(u: float, v: float):
    best_d = 999.0
    best_name = "thirds_v_left"
    for u1, v1, u2, v2, name in ARMATURE_14_LINES:
        d = point_to_segment_dist(u, v, u1, v1, u2, v2)
        if d < best_d:
            best_d = d
            best_name = name
    return best_d, best_name


def bgr_to_hex(bgr):
    b, g, r = [int(np.clip(x, 0, 255)) for x in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_worm_from_shot_json(shot_dir: Path):
    """Extracts 33-frame (u(tau), v(tau), r(tau)) worm trajectory and camera pan from shot.json."""
    shot_json_path = shot_dir / "shot.json"
    W, H = 480.0, 360.0
    u_arr = np.full(33, 0.5, dtype=np.float64)
    v_arr = np.full(33, 0.55, dtype=np.float64)
    r_arr = np.full(33, 0.16, dtype=np.float64)
    sec_u_arr = np.full(33, 0.28, dtype=np.float64)
    sec_v_arr = np.full(33, 0.52, dtype=np.float64)
    has_secondary = False
    cam_dx = np.zeros(33, dtype=np.float64)
    cam_dy = np.zeros(33, dtype=np.float64)

    if not shot_json_path.exists():
        return u_arr, v_arr, r_arr, sec_u_arr, sec_v_arr, has_secondary, cam_dx, cam_dy

    try:
        data = json.loads(shot_json_path.read_text())
        W = float(data.get("W", 480))
        H = float(data.get("H", 360))
        cam = data.get("camera", [])
        for i in range(min(33, len(cam))):
            if isinstance(cam[i], (list, tuple)) and len(cam[i]) >= 2:
                cam_dx[i] = float(cam[i][0]) / max(W, 1.0)
                cam_dy[i] = float(cam[i][1]) / max(H, 1.0)

        objs = data.get("objects", [])
        char_obj = None
        sec_objs = []
        for o in objs:
            if o.get("plane") == "char" or o.get("id") in ("char", "hero", "giraffe"):
                if char_obj is None:
                    char_obj = o
            elif o.get("plane") in ("secondary", "mid", "fore"):
                sec_objs.append(o)

        char_frames = None
        if isinstance(data.get("cast"), dict) and data["cast"].get("frames"):
            char_frames = data["cast"]["frames"]
        elif isinstance(data.get("cast"), list) and data["cast"]:
            if isinstance(data["cast"][0], dict) and data["cast"][0].get("frames"):
                char_frames = data["cast"][0]["frames"]
            else:
                char_frames = data["cast"]
        elif char_obj and char_obj.get("frames"):
            char_frames = char_obj["frames"]
        elif data.get("frames"):
            char_frames = data["frames"]

        if char_frames:
            n_f = len(char_frames)
            for t in range(33):
                fr = char_frames[min(t, n_f - 1)]
                strokes = fr.get("strokes", fr) if isinstance(fr, dict) else fr
                pts = []
                if isinstance(strokes, list):
                    for st in strokes:
                        if isinstance(st, dict):
                            st = st.get("points", st.get("pts", []))
                        if isinstance(st, list):
                            for p in st:
                                if isinstance(p, (list, tuple)) and len(p) >= 2:
                                    pts.append((float(p[0]), float(p[1])))
                if pts:
                    pts_np = np.array(pts, dtype=np.float64)
                    umin, vmin = pts_np.min(axis=0)
                    umax, vmax = pts_np.max(axis=0)
                    uc = 0.5 * (umin + umax) / max(W, 1.0)
                    vc = 0.5 * (vmin + vmax) / max(H, 1.0)
                    bw = (umax - umin) / max(W, 1.0)
                    bh = (vmax - vmin) / max(H, 1.0)
                    u_arr[t] = float(np.clip(uc, 0.05, 0.95))
                    v_arr[t] = float(np.clip(vc, 0.05, 0.95))
                    r_arr[t] = float(np.clip(0.5 * math.hypot(bw, bh), 0.05, 0.48))

        if sec_objs:
            has_secondary = True
            so = sec_objs[0]
            s_strokes = so.get("strokes", [])
            if not s_strokes and so.get("frames"):
                s_strokes = so["frames"][0]
            pts = []
            if isinstance(s_strokes, list):
                for st in s_strokes:
                    if isinstance(st, list):
                        for p in st:
                            if isinstance(p, (list, tuple)) and len(p) >= 2:
                                pts.append((float(p[0]), float(p[1])))
            if pts:
                pts_np = np.array(pts, dtype=np.float64)
                uc0 = float(pts_np[:, 0].mean() / max(W, 1.0))
                vc0 = float(pts_np[:, 1].mean() / max(H, 1.0))
                z_sec = float(so.get("z", 2.5))
                z_ref = float(data.get("z_ref", 3.0))
                for t in range(33):
                    sec_u_arr[t] = np.clip(uc0 + cam_dx[t] * (z_ref / max(z_sec, 0.5)), 0.04, 0.96)
                    sec_v_arr[t] = np.clip(vc0 + cam_dy[t] * (z_ref / max(z_sec, 0.5)), 0.04, 0.96)
    except Exception:
        pass

    return u_arr, v_arr, r_arr, sec_u_arr, sec_v_arr, has_secondary, cam_dx, cam_dy


def analyze_single_shot(shot_dir: Path, rank_idx: int):
    slug = shot_dir.name
    u_arr, v_arr, r_arr, sec_u, sec_v, has_sec, cam_dx, cam_dy = extract_worm_from_shot_json(shot_dir)

    comp_path = shot_dir / "comparison.mp4"
    arch_frames = []
    cart_frames = []
    if comp_path.exists():
        cap = cv2.VideoCapture(str(comp_path))
        while len(arch_frames) < 33:
            ok, fr = cap.read()
            if not ok:
                break
            fh, fw = fr.shape[:2]
            half_w = fw // 2
            y_top = 30
            y_bot = fh - 26
            crop_h = y_bot - y_top
            crop_w = int(round(crop_h * 4.0 / 3.0))
            x_pad = max(0, (half_w - crop_w) // 2)
            arch_frames.append(cv2.resize(fr[y_top:y_bot, x_pad:x_pad + crop_w], (160, 120)))
            cart_frames.append(cv2.resize(fr[y_top:y_bot, half_w + x_pad:half_w + x_pad + crop_w], (160, 120)))
        cap.release()

    if not arch_frames:
        sk = cv2.imread(str(shot_dir / "source-keys.png"))
        if sk is not None:
            f0 = cv2.resize(sk[:sk.shape[0] // 2, :sk.shape[1] // 2], (160, 120))
        else:
            f0 = np.full((120, 160, 3), 180, dtype=np.uint8)
        arch_frames = [f0] * 33
        cart_frames = [f0] * 33
    while len(arch_frames) < 33:
        arch_frames.append(arch_frames[-1])
        cart_frames.append(cart_frames[-1])

    T = 33
    tau = np.linspace(0.0, 1.0, T)
    dt = 1.0 / (T - 1)

    if float(np.std(u_arr)) < 0.008 and float(np.std(v_arr)) < 0.008:
        for t in range(T):
            gray = cv2.cvtColor(arch_frames[t], cv2.COLOR_BGR2GRAY).astype(np.float64)
            diff = np.abs(gray - np.mean(gray))
            ys, xs = np.mgrid[0:120, 0:160]
            w_sum = float(np.sum(diff)) + 1e-6
            u_opt = float(np.sum(xs * diff) / w_sum) / 160.0
            v_opt = float(np.sum(ys * diff) / w_sum) / 120.0
            u_arr[t] = 0.65 * u_arr[t] + 0.35 * u_opt
            v_arr[t] = 0.65 * v_arr[t] + 0.35 * v_opt

    # 1. Differential Geometry of the 3D Space Curve c(tau) = (u(tau), v(tau), tau)
    du = np.gradient(u_arr, dt)
    dv = np.gradient(v_arr, dt)
    dtau = np.ones_like(du)

    d2u = np.gradient(du, dt)
    d2v = np.gradient(dv, dt)
    d2tau = np.zeros_like(du)

    d3u = np.gradient(d2u, dt)
    d3v = np.gradient(d2v, dt)
    d3tau = np.zeros_like(du)

    c1 = np.stack([du, dv, dtau], axis=-1)
    c2 = np.stack([d2u, d2v, d2tau], axis=-1)
    c3 = np.stack([d3u, d3v, d3tau], axis=-1)

    cross_12 = np.cross(c1, c2)
    norm_cross = np.linalg.norm(cross_12, axis=-1)
    norm_c1 = np.linalg.norm(c1, axis=-1)

    curvature = norm_cross / np.maximum(norm_c1 ** 3, 1e-6)
    triple_prod = np.sum(cross_12 * c3, axis=-1)
    torsion = triple_prod / np.maximum(norm_cross ** 2, 1e-4)
    jerk_mag = np.hypot(d3u, d3v) * (dt ** 2)

    jerk_order = np.argsort(jerk_mag)[::-1]
    sfx_cues = []
    for idx in jerk_order:
        f_i = int(idx)
        if all(abs(f_i - c["frame"]) >= 5 for c in sfx_cues):
            sfx_name = EARSHOT_SFX_BANK[(rank_idx * 3 + len(sfx_cues) * 2) % len(EARSHOT_SFX_BANK)]
            sfx_cues.append({
                "frame": f_i + 1,
                "tau": round(float(tau[f_i]), 3),
                "time_s": round(float(f_i / 8.0), 2),
                "jerk": round(float(jerk_mag[f_i]), 4),
                "sfx": sfx_name,
            })
        if len(sfx_cues) >= 3:
            break
    sfx_cues.sort(key=lambda x: x["frame"])

    # 2. KALMUS Core-Worm vs. Outer-Shell Chromatic Stratigraphy & Johannes Itten's 7 Color Contrasts
    delta_L_arr = np.zeros(T, dtype=np.float64)
    delta_H_arr = np.zeros(T, dtype=np.float64)
    area_ratio_arr = np.zeros(T, dtype=np.float64)
    core_bgr_samples = []
    shell_bgr_samples = []

    ys_grid, xs_grid = np.mgrid[0:120, 0:160]
    u_grid = xs_grid / 160.0
    v_grid = ys_grid / 120.0

    for t in range(T):
        fr_bgr = arch_frames[t]
        fr_lab = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        fr_hsv = cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)

        dist_sq = (u_grid - u_arr[t]) ** 2 + (v_grid - v_arr[t]) ** 2
        rad = max(float(r_arr[t]), 0.08)
        core_mask = dist_sq <= (rad ** 2)
        shell_mask = ~core_mask
        if not np.any(core_mask):
            core_mask[50:70, 70:90] = True
            shell_mask = ~core_mask

        L_core = float(np.mean(fr_lab[:, :, 0][core_mask])) * (100.0 / 255.0)
        L_shell = float(np.mean(fr_lab[:, :, 0][shell_mask])) * (100.0 / 255.0)
        delta_L_arr[t] = abs(L_core - L_shell)

        H_core = float(np.mean(fr_hsv[:, :, 0][core_mask])) * 2.0
        H_shell = float(np.mean(fr_hsv[:, :, 0][shell_mask])) * 2.0
        dh = abs(H_core - H_shell) % 360.0
        delta_H_arr[t] = min(dh, 360.0 - dh)

        area_ratio_arr[t] = float(np.mean(core_mask))

        core_bgr_samples.append(np.mean(fr_bgr[core_mask], axis=0))
        shell_bgr_samples.append(np.mean(fr_bgr[shell_mask], axis=0))

    core_swatches = [
        bgr_to_hex(core_bgr_samples[0]),
        bgr_to_hex(core_bgr_samples[T // 2]),
        bgr_to_hex(core_bgr_samples[-1]),
    ]
    shell_swatches = [
        bgr_to_hex(shell_bgr_samples[0]),
        bgr_to_hex(shell_bgr_samples[-1]),
    ]

    # 3. Grid-Tunnel Resonance & Arnheim Torque (Bouleau 14-Line Armature)
    d_grid_arr = np.zeros(T, dtype=np.float64)
    nearest_lines = []
    for t in range(T):
        dg, lname = min_armature_distance(float(u_arr[t]), float(v_arr[t]))
        d_grid_arr[t] = dg
        nearest_lines.append(lname)

    mean_d_grid = float(np.mean(d_grid_arr))
    lock_share = float(np.mean(d_grid_arr < 0.045))
    arnheim_tension_share = float(np.mean((d_grid_arr >= 0.06) & (d_grid_arr <= 0.14)))
    dominant_filament = max(set(nearest_lines), key=nearest_lines.count)

    # 4. Classify the 3D Spatiotemporal Worm Topology in (u, v, tau)
    net_du = float(u_arr[-1] - u_arr[0])
    net_dv = float(v_arr[-1] - v_arr[0])
    shear_deg = float(math.degrees(math.atan2(net_du, 1.0)))
    radius_ratio = float((r_arr[-1] + 1e-4) / (r_arr[0] + 1e-4))
    golden_dist_end = float(np.linalg.norm(np.array([u_arr[-1], v_arr[-1]]) - GOLDEN_NODE))
    golden_dist_start = float(np.linalg.norm(np.array([u_arr[0], v_arr[0]]) - GOLDEN_NODE))
    sign_changes_u = int(np.sum(np.diff(np.sign(du)) != 0))
    sign_changes_v = int(np.sum(np.diff(np.sign(dv)) != 0))

    if abs(net_du) < 0.045 and abs(net_dv) < 0.045 and np.std(u_arr) < 0.03:
        topology = "Vertical Pillar (Locked Tripod + Static Blocking)"
        topology_code = "vertical_pillar"
        grammar_note = f"High stability/affinity (Ozu/Anderson channel lock on {dominant_filament}); near-zero saccade load."
    elif radius_ratio > 1.22 or radius_ratio < 0.80:
        topology = "Expanding Cone / Funnel (Dolly / Scale Compression)"
        topology_code = "expanding_cone"
        grammar_note = f"Cross-sectional radius r(tau) scales by {radius_ratio:.2f}x as negative-space shell volume compresses."
    elif sign_changes_u >= 3 and sign_changes_v >= 3 or (golden_dist_end < golden_dist_start - 0.04):
        topology = "Helical Corkscrew (S-Curve / Golden Spiral Staging)"
        topology_code = "helical_corkscrew"
        grammar_note = f"Kinetic grace distributed across both u and v, spiraling toward Golden Ratio node (d_phi={golden_dist_end:.3f})."
    elif abs(shear_deg) >= 6.5:
        topology = "Sheared Diagonal Ribbon (Camera Pan / Lateral Track)"
        topology_code = "sheared_ribbon"
        grammar_note = f"Worm shears at theta = {shear_deg:+.1f} deg across the u-tau plane, separating 7-plane parallax velocities."
    else:
        topology = "Double Helix / Braided Counterpoint (Hero + Secondary Tube)"
        topology_code = "double_helix"
        gap_delta = float(np.mean(np.abs(u_arr - sec_u)))
        grammar_note = f"Hero worm and secondary counterpoint tube braid across Rabatment channels (mean Delta-u = {gap_delta:.3f})."

    speed = np.hypot(du, dv)
    intensity = np.clip(
        0.40 * (speed / (np.max(speed) + 1e-4))
        + 0.35 * (delta_L_arr / 60.0)
        + 0.25 * (curvature / (np.max(curvature) + 1e-4)),
        0.05,
        1.0,
    )

    return {
        "slug": slug,
        "rank": rank_idx + 1,
        "topology": topology,
        "topology_code": topology_code,
        "grammar_note": grammar_note,
        "shear_deg": round(shear_deg, 2),
        "radius_ratio": round(radius_ratio, 3),
        "dominant_filament": dominant_filament,
        "mean_d_grid": round(mean_d_grid, 4),
        "compositional_lock_pct": round(lock_share * 100.0, 1),
        "arnheim_tension_pct": round(arnheim_tension_share * 100.0, 1),
        "mean_chiaroscuro_dL": round(float(np.mean(delta_L_arr)), 2),
        "mean_hue_polarity_dH": round(float(np.mean(delta_H_arr)), 1),
        "mean_extension_ratio": round(float(np.mean(area_ratio_arr)), 3),
        "mean_curvature": round(float(np.mean(curvature)), 3),
        "mean_torsion": round(float(np.mean(np.abs(torsion))), 3),
        "peak_jerk": round(float(np.max(jerk_mag)), 4),
        "core_palette": core_swatches,
        "shell_palette": shell_swatches,
        "earshot_cues": sfx_cues,
        "worm_u": [round(float(x), 4) for x in u_arr],
        "worm_v": [round(float(x), 4) for x in v_arr],
        "worm_r": [round(float(x), 4) for x in r_arr],
        "sec_u": [round(float(x), 4) for x in sec_u],
        "sec_v": [round(float(x), 4) for x in sec_v],
        "d_grid_curve": [round(float(x), 4) for x in d_grid_arr],
        "dL_curve": [round(float(x), 2) for x in delta_L_arr],
        "dH_curve": [round(float(x), 1) for x in delta_H_arr],
        "jerk_curve": [round(float(x), 4) for x in jerk_mag],
        "intensity_curve": [round(float(x), 3) for x in intensity],
        "_arch_frames": arch_frames,
        "_cart_frames": cart_frames,
    }


from PIL import Image, ImageDraw, ImageFont

QD_DIR = ROOT / "quickdraw_samples"
_QD_CACHE = {}

# Unified Quick, Draw! Paper & Gouache Palette (BGR)
PAPER_CREAM = (227, 239, 246)     # #F6EFE3
PAPER_LIGHT = (244, 251, 255)     # #FFFBF4
PAPER_DARK = (206, 226, 236)      # #ECE2CE
INK_BLACK = (18, 20, 22)          # #161412
INK_GRAPHITE = (102, 112, 122)    # #7A7066
GOUACHE_AMBER = (194, 229, 251)   # #FBE5C2
GOUACHE_MINT = (220, 232, 212)    # #D4E8DC
GOUACHE_BLUE = (240, 228, 214)    # #D6E4F0
GOUACHE_CORAL = (206, 216, 247)   # #F7D8CE
ACCENT_RED = (41, 74, 201)        # #C94A29
ACCENT_GREEN = (79, 107, 47)      # #2F6B4F
ACCENT_BLUE = (132, 91, 43)       # #2B5B84
ACCENT_GOLD = (58, 159, 239)      # #EF9F3A

_FONT_CACHE = {}


def get_comic_font(size: int, style: str = "bold"):
    key = (size, style)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    paths = [
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]
    if style == "comic":
        paths = [
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/System/Library/Fonts/Supplemental/Futura.ttc",
        ]
    for p in paths:
        if Path(p).exists():
            try:
                f = ImageFont.truetype(p, size)
                _FONT_CACHE[key] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def fit_pil_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, start_size: int, min_size: int = 10, style: str = "bold"):
    for sz in range(start_size, min_size - 1, -1):
        f = get_comic_font(sz, style)
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_w:
            return f
    return get_comic_font(min_size, style)


def human_motion_story(topo_code: str) -> str:
    stories = {
        "vertical_pillar": "STEADY CENTER PILLAR (Holds center while 7 planes parallax)",
        "sheared_ribbon": "GLIDES ACROSS FRAME (Smooth horizontal camera pan)",
        "expanding_cone": "GROWS TOWARD CAMERA (Expands from background to foreground)",
        "helical_corkscrew": "SPIRALS THROUGH TIME (Curved arc across 4 keyframes)",
        "double_helix": "TWIN ACTION PATHS (Subject & background move together)",
    }
    return stories.get(topo_code, "SEQUENTIAL TIME-FRAMES (4 keyframes stacked in space)")


def load_qd_doodle(slug: str):
    cat = "giraffe" if slug == "giraffe-walks-past" else slug.replace("-", "_")
    if cat in _QD_CACHE:
        return _QD_CACHE[cat]
    p = QD_DIR / f"{cat}.json"
    if not p.exists():
        p = QD_DIR / "star.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        recs = data.get("drawings", data.get("records", [])) if isinstance(data, dict) else data
        raw_strokes = None
        for r in recs:
            st = r.get("strokes", r.get("drawing", []))
            if 2 <= len(st) <= 16:
                raw_strokes = st
                break
        if not raw_strokes and recs:
            raw_strokes = recs[0].get("strokes", recs[0].get("drawing", []))
        if not raw_strokes:
            return []
        pts = []
        norm = []
        for s in raw_strokes:
            if len(s) == 2 and isinstance(s[0], list) and isinstance(s[0][0], (int, float)):
                st = [(float(x), float(y)) for x, y in zip(s[0], s[1])]
            else:
                st = [(float(pt[0]), float(pt[1])) for pt in s if len(pt) >= 2]
            if len(st) >= 2:
                norm.append(st)
                pts.extend(st)
        if not pts:
            return []
        arr = np.array(pts, dtype=np.float32)
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)
        span = float(max(mx[0] - mn[0], mx[1] - mn[1], 1.0))
        cx = 0.5 * (mn[0] + mx[0])
        cy = 0.5 * (mn[1] + mx[1])
        out = [[((x - cx) / span, (y - cy) / span) for (x, y) in st] for st in norm]
        _QD_CACHE[cat] = out
        return out
    except Exception:
        return []


def draw_qd_doodle_stamp(canvas: np.ndarray, slug: str, cx: int, cy: int, size: float, fill_bgr=PAPER_LIGHT, thickness: int = 2):
    strokes = load_qd_doodle(slug)
    if not strokes:
        return
    all_pts = []
    scaled = []
    for st in strokes:
        pts = []
        for nx, ny in st:
            px = int(round(cx + nx * size))
            py = int(round(cy + ny * size))
            pts.append([px, py])
            all_pts.append([px, py])
        if len(pts) >= 2:
            scaled.append(np.array(pts, dtype=np.int32))
    if len(all_pts) >= 3 and fill_bgr is not None:
        hull = cv2.convexHull(np.array(all_pts, dtype=np.int32))
        cv2.fillConvexPoly(canvas, hull + np.array([2, 2], dtype=np.int32), INK_BLACK, cv2.LINE_AA)
        cv2.fillConvexPoly(canvas, hull, fill_bgr, cv2.LINE_AA)
    for arr in scaled:
        cv2.polylines(canvas, [arr], False, INK_BLACK, thickness, cv2.LINE_AA)


def draw_qd_card(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill_bgr=PAPER_LIGHT, shadow: int = 3):
    if shadow > 0:
        cv2.rectangle(canvas, (x0 + shadow, y0 + shadow), (x1 + shadow, y1 + shadow), INK_BLACK, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill_bgr, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), INK_BLACK, 2)


DOODLE_ASSETS_DIR = ROOT / "assets" / "studio_doodles"
_DOODLE_QUAD_CACHE = {}


def load_doodle_quadrant(sheet_name: str, quad_idx: int) -> tuple:
    key = (sheet_name, int(quad_idx) % 4)
    if key in _DOODLE_QUAD_CACHE:
        return _DOODLE_QUAD_CACHE[key]
    p = DOODLE_ASSETS_DIR / f"{sheet_name}.jpg"
    if not p.exists():
        blank_g = np.full((120, 120), 255, dtype=np.uint8)
        blank_m = np.zeros((120, 120), dtype=np.uint8)
        _DOODLE_QUAD_CACHE[key] = (blank_g, blank_m)
        return blank_g, blank_m
    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape[:2]
    r = (int(quad_idx) % 4) // 2
    c = (int(quad_idx) % 4) % 2
    pad_seam = 10
    y0 = r * (H // 2) + (pad_seam if r == 1 else 6)
    y1 = (r + 1) * (H // 2) - (pad_seam if r == 0 else 6)
    x0 = c * (W // 2) + (pad_seam if c == 1 else 6)
    x1 = (c + 1) * (W // 2) - (pad_seam if c == 0 else 6)
    sub = gray[y0:y1, x0:x1]
    ys, xs = np.where(sub < 200)
    if len(xs) > 10:
        bx0 = max(0, int(xs.min()) - 10)
        by0 = max(0, int(ys.min()) - 10)
        bx1 = min(sub.shape[1], int(xs.max()) + 11)
        by1 = min(sub.shape[0], int(ys.max()) + 11)
        q_gray = sub[by0:by1, bx0:bx1].copy()
    else:
        q_gray = sub.copy()
    qh, qw = q_gray.shape[:2]
    ink_bin = (q_gray < 210).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_closed = cv2.morphologyEx(ink_bin, cv2.MORPH_CLOSE, kernel)
    ff_img = (ink_closed == 0).astype(np.uint8) * 255
    mask = np.zeros((qh + 2, qw + 2), dtype=np.uint8)
    for sx in range(0, qw, 4):
        if ff_img[0, sx] == 255:
            cv2.floodFill(ff_img, mask, (sx, 0), 128)
        if ff_img[qh - 1, sx] == 255:
            cv2.floodFill(ff_img, mask, (sx, qh - 1), 128)
    for sy in range(0, qh, 4):
        if ff_img[sy, 0] == 255:
            cv2.floodFill(ff_img, mask, (0, sy), 128)
        if ff_img[sy, qw - 1] == 255:
            cv2.floodFill(ff_img, mask, (qw - 1, sy), 128)
    cel_mask = (ff_img != 128).astype(np.uint8) * 255
    _DOODLE_QUAD_CACHE[key] = (q_gray, cel_mask)
    return q_gray, cel_mask


def draw_studio_doodle_sprite(
    canvas: np.ndarray,
    sheet_name: str,
    quad_idx: int,
    cx: int,
    cy: int,
    max_w: int,
    max_h: int,
    tint_bgr: tuple = None,
    tint_strength: float = 0.28,
):
    q_gray, cel_mask = load_doodle_quadrant(sheet_name, quad_idx)
    qh, qw = q_gray.shape[:2]
    if qw < 4 or qh < 4:
        return
    scale = min(float(max_w) / float(qw), float(max_h) / float(qh))
    tw = max(8, int(round(qw * scale)))
    th = max(8, int(round(qh * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    g_res = cv2.resize(q_gray, (tw, th), interpolation=interp)
    m_res = cv2.resize(cel_mask, (tw, th), interpolation=interp)
    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - th / 2.0))
    x1, y1 = x0 + tw, y0 + th
    H, W = canvas.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)
    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    g_sub = g_res[sy0:sy1, sx0:sx1].astype(np.float32)
    m_sub = (m_res[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]
    base_cel = np.array(PAPER_LIGHT, dtype=np.float32)
    fill_col = base_cel * (1.0 - tint_strength) + np.array(tint_bgr, dtype=np.float32) * tint_strength if tint_bgr is not None else base_cel
    roi = roi * (1.0 - m_sub) + fill_col * m_sub
    ink_alpha = (np.clip((235.0 - g_sub) / 185.0, 0.0, 1.0) ** 0.85)[..., None]
    roi = roi * (1.0 - ink_alpha) + np.array(INK_BLACK, dtype=np.float32) * ink_alpha
    canvas[cy0:cy1, cx0:cx1] = np.clip(roi, 0, 255).astype(np.uint8)



def archival_slice_to_qd_sketch(bgr_img: np.ndarray) -> np.ndarray:
    """Converts a raw archival photographic slit-scan slice into a Quick, Draw! graphite sketch on cream paper."""
    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    ink_mask = cv2.bitwise_or(edges, adapt)
    out = np.full((h, w, 3), PAPER_CREAM, dtype=np.uint8)
    out[blur < 115] = PAPER_DARK
    out[ink_mask > 0] = (52, 58, 66)
    return out


def render_slitscan_tomography_atlas(records):
    """Visualization 1: Scott McCloud 'Time Frames' Slit-Scan Atlas (How Space Shows Time Across 33 Frames)."""
    featured_slugs = [
        "giraffe-walks-past", "train", "lighthouse", "guitar",
        "tractor", "speedboat", "bicycle", "carrot",
        "sandwich", "diamond", "star", "zigzag",
    ]
    rec_map = {r["slug"]: r for r in records}
    chosen = [rec_map[s] for s in featured_slugs if s in rec_map]

    cols, rows = 4, 3
    cell_w, cell_h = 420, 290
    header_h = 74
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (cols * cell_w, header_h - 2), INK_BLACK, 3)
    draw_studio_doodle_sprite(canvas, "agent_nib_keys", 0, cols * cell_w - 135, header_h // 2, 64, 66, tint_bgr=GOUACHE_CORAL, tint_strength=0.35)
    draw_studio_doodle_sprite(canvas, "agent_chime_keys", 0, cols * cell_w - 55, header_h // 2, 64, 66, tint_bgr=GOUACHE_MINT, tint_strength=0.35)

    text_overlays = []

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        draw_qd_card(canvas, x0 + 6, y0 + 6, x0 + cell_w - 8, y0 + cell_h - 8, PAPER_LIGHT, shadow=3)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 34), PAPER_DARK, -1)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 34), INK_BLACK, 2)
        draw_qd_doodle_stamp(canvas, rec["slug"], x0 + cell_w - 26, y0 + 20, 18.0, fill_bgr=GOUACHE_AMBER, thickness=2)

        arch_frs = rec["_arch_frames"]
        cart_frs = rec["_cart_frames"]
        v_slice_idx = int(np.clip(np.mean(rec["worm_v"]) * 119, 12, 108))
        u_slice_idx = int(np.clip(np.mean(rec["worm_u"]) * 159, 16, 144))

        arch_utau = np.stack([fr[v_slice_idx, :, :] for fr in arch_frs], axis=0)
        cart_utau = np.stack([fr[v_slice_idx, :, :] for fr in cart_frs], axis=0)
        cart_vtau = np.stack([fr[:, u_slice_idx, :] for fr in cart_frs], axis=0)

        arch_utau_img = archival_slice_to_qd_sketch(cv2.resize(arch_utau, (124, 188), interpolation=cv2.INTER_LINEAR))
        cart_utau_img = cv2.resize(cart_utau, (124, 188), interpolation=cv2.INTER_LINEAR)
        cart_vtau_img = cv2.resize(cart_vtau, (124, 188), interpolation=cv2.INTER_LINEAR)

        for panel_img, coords in [(arch_utau_img, rec["worm_u"]), (cart_utau_img, rec["worm_u"]), (cart_vtau_img, rec["worm_v"])]:
            ph, pw = panel_img.shape[:2]
            pts = []
            for t_i, val in enumerate(coords):
                px = int(np.clip(val * (pw - 1), 3, pw - 4))
                py = int(round((t_i / 32.0) * (ph - 1)))
                pts.append([px, py])
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(panel_img, [pts_np], False, INK_BLACK, 5, cv2.LINE_AA)
            cv2.polylines(panel_img, [pts_np], False, ACCENT_GOLD, 2, cv2.LINE_AA)
            for cue in rec["earshot_cues"]:
                py = int(round(((cue["frame"] - 1) / 32.0) * (ph - 1)))
                px = pts[cue["frame"] - 1][0]
                cv2.circle(panel_img, (px, py), 5, ACCENT_RED, -1, cv2.LINE_AA)
                cv2.circle(panel_img, (px, py), 5, INK_BLACK, 1, cv2.LINE_AA)

        py0 = y0 + 40
        for p_idx, (p_im, p_title) in enumerate([
            (arch_utau_img, "1. Pencil Trace"),
            (cart_utau_img, "2. Left-Right Motion"),
            (cart_vtau_img, "3. Up-Down Motion"),
        ]):
            px0 = x0 + 13 + p_idx * 132
            canvas[py0 + 18:py0 + 206, px0:px0 + 124] = p_im
            cv2.rectangle(canvas, (px0, py0 + 18), (px0 + 124, py0 + 206), INK_BLACK, 2)
            text_overlays.append((px0 + 4, py0 + 1, p_title, 12, "bold", (22, 20, 18), 120))

        lbl = f"#{rec['rank']:03d} {rec['slug'].upper()}"
        text_overlays.append((x0 + 14, y0 + 11, lbl, 15, "bold", (22, 20, 18), cell_w - 60))
        story = human_motion_story(rec["topology_code"])
        text_overlays.append((x0 + 14, y0 + cell_h - 30, story, 12, "comic", (47, 107, 79), cell_w - 28))

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 12), "1. AGENT ANIMATOR MOTION SLICES: SLICING 33 FRAMES ACROSS SPACE (NIB & CHIME)", font=get_comic_font(24, "bold"), fill=(22, 20, 18))
    draw.text((20, 44), "Top to Bottom = Frame 1 to Frame 33  |  Golden Ribbon = Subject's Path Through Time  |  Red Dots = Chime's Foley Cues", font=get_comic_font(15, "comic"), fill=(90, 82, 74))
    for tx, ty, txt, sz, st, col, mw in text_overlays:
        f = fit_pil_font(draw, txt, mw, sz, min_size=10, style=st)
        draw.text((tx, ty), txt, font=f, fill=col)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out_path = PREVIEW_DIR / "worm_slitscan_tomography_atlas.png"
    cv2.imwrite(str(out_path), out_bgr)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_slitscan_tomography_atlas.png")
    return out_path


def project_3d_iso(u, v, tau, x_center, y_center, scale_w=150.0, scale_h=110.0, scale_t=140.0):
    du = u - 0.5
    dv = v - 0.5
    dt = tau - 0.5
    sx = x_center + scale_w * du * 0.88 + scale_t * dt * 0.62
    sy = y_center + scale_h * dv * 0.88 - scale_w * du * 0.22 + scale_t * dt * 0.34
    return int(round(sx)), int(round(sy))


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def render_3d_glass_prism_atlas(records):
    """Visualization 2: Agent Animator 3D Time-Cubes (4 Keyframes Stacked in Space + Golden Motion Ribbon)."""
    featured_slugs = [
        "giraffe-walks-past", "train", "lighthouse", "guitar",
        "tractor", "speedboat", "bicycle", "carrot",
        "sandwich", "diamond", "star", "zigzag",
    ]
    rec_map = {r["slug"]: r for r in records}
    chosen = [rec_map[s] for s in featured_slugs if s in rec_map]

    cols, rows = 4, 3
    cell_w, cell_h = 420, 300
    header_h = 74
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (cols * cell_w, header_h - 2), INK_BLACK, 3)
    draw_studio_doodle_sprite(canvas, "agent_blueline_keys", 1, cols * cell_w - 135, header_h // 2, 64, 66, tint_bgr=GOUACHE_BLUE, tint_strength=0.35)
    draw_studio_doodle_sprite(canvas, "studio_apparatus_keys", 1, cols * cell_w - 55, header_h // 2, 64, 66, tint_bgr=GOUACHE_AMBER, tint_strength=0.35)

    text_overlays = []

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        draw_qd_card(canvas, x0 + 6, y0 + 6, x0 + cell_w - 8, y0 + cell_h - 8, PAPER_LIGHT, shadow=3)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 36), PAPER_DARK, -1)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 36), INK_BLACK, 2)

        cx = x0 + cell_w // 2 - 4
        cy = y0 + cell_h // 2 + 6

        # Draw 3 Stacked Keyframe Cards (K1 at tau=0.0, K2 at tau=0.5, K4 at tau=1.0) inside the 3D Time-Cube
        for tau_plane, plane_fill in [(0.0, GOUACHE_BLUE), (0.5, GOUACHE_AMBER), (1.0, GOUACHE_MINT)]:
            p00 = project_3d_iso(0.0, 0.0, tau_plane, cx, cy)
            p10 = project_3d_iso(1.0, 0.0, tau_plane, cx, cy)
            p11 = project_3d_iso(1.0, 1.0, tau_plane, cx, cy)
            p01 = project_3d_iso(0.0, 1.0, tau_plane, cx, cy)
            poly = np.array([p00, p10, p11, p01], dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, poly, plane_fill, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0, canvas)
            cv2.polylines(canvas, [poly], True, INK_BLACK, 2, cv2.LINE_AA)

        # Connect the 4 outer corners of the Time-Cube
        for u_c, v_c in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
            pa = project_3d_iso(u_c, v_c, 0.0, cx, cy)
            pb = project_3d_iso(u_c, v_c, 1.0, cx, cy)
            cv2.line(canvas, pa, pb, INK_GRAPHITE, 1, cv2.LINE_AA)

        # Clean Golden Motion Ribbon connecting Frame 1 to Frame 33
        worm_pts = []
        for t_i in range(33):
            tau_v = t_i / 32.0
            px, py = project_3d_iso(rec["worm_u"][t_i], rec["worm_v"][t_i], tau_v, cx, cy)
            worm_pts.append((px, py))

        pts_np = np.array(worm_pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_np], False, INK_BLACK, 9, cv2.LINE_AA)
        cv2.polylines(canvas, [pts_np], False, ACCENT_GOLD, 5, cv2.LINE_AA)

        # Stamp the genuine Quick, Draw! doodle at Frame 1 (K1) and Frame 33 (K4)
        draw_qd_doodle_stamp(canvas, rec["slug"], worm_pts[0][0], worm_pts[0][1], 28.0, fill_bgr=GOUACHE_AMBER, thickness=2)
        draw_qd_doodle_stamp(canvas, rec["slug"], worm_pts[16][0], worm_pts[16][1], 24.0, fill_bgr=PAPER_LIGHT, thickness=2)
        draw_qd_doodle_stamp(canvas, rec["slug"], worm_pts[-1][0], worm_pts[-1][1], 32.0, fill_bgr=GOUACHE_MINT, thickness=2)

        lbl = f"#{rec['rank']:03d} {rec['slug'].upper()} // 3D TIME-CUBE"
        text_overlays.append((x0 + 14, y0 + 11, lbl, 15, "bold", (22, 20, 18), cell_w - 28))
        story = human_motion_story(rec["topology_code"])
        text_overlays.append((x0 + 14, y0 + cell_h - 30, story, 12, "comic", (43, 91, 132), cell_w - 28))

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 12), "2. BLUE-LINE'S 3D TIME-CUBES: 4 KEYFRAMES STACKED IN SPACE (K1..K4)", font=get_comic_font(24, "bold"), fill=(22, 20, 18))
    draw.text((20, 44), "3 Translucent Cards = Keyframes K1, K2, K4 Stacked in Space  |  Golden Ribbon = Motion Path Connecting Them", font=get_comic_font(15, "comic"), fill=(90, 82, 74))
    for tx, ty, txt, sz, st, col, mw in text_overlays:
        f = fit_pil_font(draw, txt, mw, sz, min_size=10, style=st)
        draw.text((tx, ty), txt, font=f, fill=col)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out_path = PREVIEW_DIR / "worm_3d_glass_prism_atlas.png"
    cv2.imwrite(str(out_path), out_bgr)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_3d_glass_prism_atlas.png")
    return out_path


def render_cinemetrics_torus_atlas(records):
    """Visualization 3: Cel-Inker & Chime's Color & Rhythm Wheel Atlas (5 Gouache Palette Rings + Action Spikes)."""
    featured_slugs = [
        "giraffe-walks-past", "train", "lighthouse", "guitar",
        "tractor", "speedboat", "bicycle", "carrot",
        "sandwich", "diamond", "star", "zigzag",
    ]
    rec_map = {r["slug"]: r for r in records}
    chosen = [rec_map[s] for s in featured_slugs if s in rec_map]

    cols, rows = 4, 3
    cell_w, cell_h = 400, 300
    header_h = 74
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (cols * cell_w, header_h - 2), INK_BLACK, 3)
    draw_studio_doodle_sprite(canvas, "agent_celinker_keys", 0, cols * cell_w - 135, header_h // 2, 64, 66, tint_bgr=GOUACHE_AMBER, tint_strength=0.35)
    draw_studio_doodle_sprite(canvas, "agent_chime_keys", 3, cols * cell_w - 55, header_h // 2, 64, 66, tint_bgr=GOUACHE_MINT, tint_strength=0.35)

    text_overlays = []

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        draw_qd_card(canvas, x0 + 6, y0 + 6, x0 + cell_w - 8, y0 + cell_h - 8, PAPER_LIGHT, shadow=3)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 36), PAPER_DARK, -1)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 36), INK_BLACK, 2)

        cx = x0 + cell_w // 2
        cy = y0 + cell_h // 2 + 8

        swatches = rec["core_palette"] + rec["shell_palette"]
        gouache_bands = [GOUACHE_AMBER, GOUACHE_CORAL, GOUACHE_MINT, GOUACHE_BLUE, PAPER_DARK]
        radii = [34, 44, 54, 64, 74]
        for b_i, r_px in enumerate(reversed(radii)):
            raw_col = hex_to_bgr(swatches[b_i % len(swatches)])
            gt = gouache_bands[b_i % len(gouache_bands)]
            col = tuple(int(0.42 * raw_col[k] + 0.58 * gt[k]) for k in range(3))
            cv2.circle(canvas, (cx, cy), r_px, col, 9, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), r_px + 4, INK_BLACK, 1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 28, PAPER_LIGHT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 28, INK_BLACK, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 79, INK_BLACK, 2, cv2.LINE_AA)

        draw_qd_doodle_stamp(canvas, rec["slug"], cx, cy, 36.0, fill_bgr=GOUACHE_AMBER, thickness=2)

        max_j = max(rec["jerk_curve"]) + 1e-4
        for t_i in range(33):
            ang = -math.pi / 2.0 + (2.0 * math.pi * t_i / 33.0)
            inten = rec["intensity_curve"][t_i]
            j_norm = rec["jerk_curve"][t_i] / max_j
            spike_len = int(12 + 30 * (0.55 * inten + 0.45 * j_norm))
            x1 = int(round(cx + 80 * math.cos(ang)))
            y1 = int(round(cy + 80 * math.sin(ang)))
            x2 = int(round(cx + (80 + spike_len) * math.cos(ang)))
            y2 = int(round(cy + (80 + spike_len) * math.sin(ang)))
            is_key = t_i in (0, 10, 21, 32)
            col = ACCENT_RED if is_key else INK_BLACK
            thick = 3 if is_key else 2
            cv2.line(canvas, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)

        lbl = f"#{rec['rank']:03d} {rec['slug'].upper()} // COLOR & RHYTHM WHEEL"
        text_overlays.append((x0 + 14, y0 + 11, lbl, 15, "bold", (22, 20, 18), cell_w - 28))
        story = f"5 Gouache Plane Colors  |  4 Red Keyframe Beats (K1, K2, K3, K4)"
        text_overlays.append((x0 + 14, y0 + cell_h - 30, story, 12, "comic", (90, 82, 74), cell_w - 28))

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 12), "3. CEL-INKER & CHIME'S COLOR & RHYTHM WHEELS: 7-PLANE PALETTES & BEATS", font=get_comic_font(24, "bold"), fill=(22, 20, 18))
    draw.text((20, 44), "Center = Quick, Draw! Icon  |  5 Colored Rings = 7-Plane Gouache Palette  |  Red Spikes = 4 Keyframe Accents", font=get_comic_font(15, "comic"), fill=(90, 82, 74))
    for tx, ty, txt, sz, st, col, mw in text_overlays:
        f = fit_pil_font(draw, txt, mw, sz, min_size=10, style=st)
        draw.text((tx, ty), txt, font=f, fill=col)

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out_path = PREVIEW_DIR / "worm_cinemetrics_torus_atlas.png"
    cv2.imwrite(str(out_path), out_bgr)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_cinemetrics_torus_atlas.png")
    return out_path


def render_manovich_montage_grid(records):
    """Visualization 4: All 346 Cartoons Side-by-Side at Exact 4:3 Cell Aspect Ratio (`76x57` inner thumbnail = 4:3!)."""
    cols, rows = 20, 18
    cw, ch = 78, 59  # Inner cell is 76x57 -> 76 / 57 = 4 / 3 (EXACT 4:3 aspect ratio!)
    header_h = 74
    footer_h = 128
    W = cols * cw
    H = header_h + rows * ch + footer_h
    canvas = np.full((H, W, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (W, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (W, header_h - 2), INK_BLACK, 3)
    draw_studio_doodle_sprite(canvas, "agent_sprocket_keys", 0, W - 135, header_h // 2, 64, 66, tint_bgr=GOUACHE_BLUE, tint_strength=0.35)
    draw_studio_doodle_sprite(canvas, "studio_apparatus_keys", 0, W - 55, header_h // 2, 64, 66, tint_bgr=GOUACHE_AMBER, tint_strength=0.35)

    macro_intensities = []
    macro_dL = []

    topo_colors = {
        "vertical_pillar": GOUACHE_AMBER,
        "sheared_ribbon": GOUACHE_MINT,
        "expanding_cone": GOUACHE_CORAL,
        "helical_corkscrew": GOUACHE_BLUE,
        "double_helix": PAPER_DARK,
    }

    for idx, rec in enumerate(records[:346]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cw
        y0 = header_h + r_i * ch

        mid_fr = cv2.resize(rec["_cart_frames"][16], (cw - 2, ch - 2))
        pts = []
        for t_i in range(33):
            px = int(np.clip(rec["worm_u"][t_i] * (cw - 4), 1, cw - 4))
            py = int(np.clip(rec["worm_v"][t_i] * (ch - 4), 1, ch - 4))
            pts.append([px, py])
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(mid_fr, [pts_np], False, INK_BLACK, 3, cv2.LINE_AA)
        cv2.polylines(mid_fr, [pts_np], False, ACCENT_GOLD, 1, cv2.LINE_AA)

        tcol = topo_colors.get(rec["topology_code"], PAPER_DARK)
        cv2.rectangle(mid_fr, (0, 0), (cw - 2, 7), tcol, -1)
        cv2.line(mid_fr, (0, 7), (cw - 2, 7), INK_BLACK, 1)

        canvas[y0 + 1:y0 + ch - 1, x0 + 1:x0 + cw - 1] = mid_fr
        cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + cw - 1, y0 + ch - 1), INK_BLACK, 1)
        macro_intensities.append(float(np.mean(rec["intensity_curve"])))
        macro_dL.append(float(rec["mean_chiaroscuro_dL"]))

    fy0 = header_h + rows * ch + 10
    fh = footer_h - 22
    draw_qd_card(canvas, 14, fy0, W - 16, fy0 + fh, PAPER_LIGHT, shadow=4)

    for gy in range(fy0 + 38, fy0 + fh - 10, 22):
        cv2.line(canvas, (24, gy), (W - 26, gy), PAPER_DARK, 1)

    pts_int = []
    pts_dl = []
    for i, (val_i, val_l) in enumerate(zip(macro_intensities, macro_dL)):
        px = int(round(24 + (i / max(len(macro_intensities) - 1, 1)) * (W - 50)))
        py_i = int(round(fy0 + fh - 12 - np.clip(val_i, 0.0, 1.0) * (fh - 42)))
        py_l = int(round(fy0 + fh - 12 - np.clip(val_l / 55.0, 0.0, 1.0) * (fh - 42)))
        pts_int.append([px, py_i])
        pts_dl.append([px, py_l])

    cv2.polylines(canvas, [np.array(pts_dl, dtype=np.int32)], False, ACCENT_RED, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(pts_int, dtype=np.int32)], False, ACCENT_GREEN, 2, cv2.LINE_AA)

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 12), "4. SPROCKET'S 346-SHOT ARCHIVE WALL (EVERY 4:3 CARTOON IN ONE STUDIO GRID)", font=get_comic_font(24, "bold"), fill=(22, 20, 18))
    draw.text((20, 44), "Every 4:3 Panel = 1 Multiplane Cartoon (#001 to #346) with Golden Motion Ribbon  |  Bottom Wave = Studio Rhythm Across All 346 Films", font=get_comic_font(15, "comic"), fill=(90, 82, 74))
    draw.text((26, fy0 + 8), "STUDIO RHYTHM WAVE ACROSS ALL 346 CARTOONS (GREEN = MOTION ENERGY  |  RED = LIGHT/DARK CONTRAST)", font=get_comic_font(15, "bold"), fill=(22, 20, 18))

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out_path = PREVIEW_DIR / "worm_manovich_montage_grid.png"
    cv2.imwrite(str(out_path), out_bgr)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_manovich_montage_grid.png")
    return out_path


def main():
    all_dirs = sorted([d for d in SHOTS_DIR.iterdir() if d.is_dir() and (d / "shot.json").exists()])
    print(f"Analyzing {len(all_dirs)} shots as 3D Spatiotemporal Volumes V(u,v,tau)...")

    records = []
    for idx, d in enumerate(all_dirs):
        rec = analyze_single_shot(d, idx)
        records.append(rec)
        if (idx + 1) % 50 == 0 or idx + 1 == len(all_dirs):
            print(f"  [{idx + 1:03d}/{len(all_dirs):03d}] {d.name:20s} | {rec['topology_code']:18s} | d_grid={rec['mean_d_grid']:.3f} | dL*={rec['mean_chiaroscuro_dL']:.1f}")

    p1 = render_slitscan_tomography_atlas(records)
    p2 = render_3d_glass_prism_atlas(records)
    p3 = render_cinemetrics_torus_atlas(records)
    p4 = render_manovich_montage_grid(records)
    print("Rendered 4 Spatiotemporal Worm Atlases:", p1.name, p2.name, p3.name, p4.name)

    clean_records = []
    from collections import Counter
    topo_counts = Counter()
    filament_counts = Counter()
    for r in records:
        rc = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_records.append(rc)
        topo_counts[rc["topology_code"]] += 1
        filament_counts[rc["dominant_filament"]] += 1

    summary = {
        "total_shots": len(clean_records),
        "topology_counts": dict(topo_counts),
        "dominant_filament_counts": dict(filament_counts),
        "mean_d_grid": round(float(np.mean([r["mean_d_grid"] for r in clean_records])), 4),
        "mean_compositional_lock_pct": round(float(np.mean([r["compositional_lock_pct"] for r in clean_records])), 1),
        "mean_arnheim_tension_pct": round(float(np.mean([r["arnheim_tension_pct"] for r in clean_records])), 1),
        "mean_chiaroscuro_dL": round(float(np.mean([r["mean_chiaroscuro_dL"] for r in clean_records])), 2),
        "mean_hue_polarity_dH": round(float(np.mean([r["mean_hue_polarity_dH"] for r in clean_records])), 1),
        "mean_curvature": round(float(np.mean([r["mean_curvature"] for r in clean_records])), 3),
        "mean_peak_jerk": round(float(np.mean([r["peak_jerk"] for r in clean_records])), 4),
        "shots": clean_records,
    }

    json_path = DATA_DIR / "spatiotemporal_worms_346.json"
    json_path.write_text(json.dumps(summary, indent=2))

    js_path = ROOT / "spatiotemporal_worms_data.js"
    js_path.write_text("window.SPATIOTEMPORAL_WORMS_DATA = " + json.dumps(summary, separators=(",", ":")) + ";\n")
    print(f"Wrote {json_path} and {js_path} ({js_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
