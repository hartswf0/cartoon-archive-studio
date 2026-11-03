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
            arch_frames.append(cv2.resize(fr[:, :fw // 2], (160, 120)))
            cart_frames.append(cv2.resize(fr[:, fw // 2:], (160, 120)))
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
    """Visualization 1: Fels & Mase Interactive Video Cubism / Video Summagator u-tau & v-tau Slit-Scan Atlas (Unified Quick, Draw! Aesthetic)."""
    featured_slugs = [
        "giraffe-walks-past", "train", "lighthouse", "guitar",
        "tractor", "speedboat", "bicycle", "carrot",
        "sandwich", "diamond", "star", "zigzag",
    ]
    rec_map = {r["slug"]: r for r in records}
    chosen = [rec_map[s] for s in featured_slugs if s in rec_map]

    cols, rows = 4, 3
    cell_w, cell_h = 420, 290
    header_h = 68
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (cols * cell_w, header_h - 2), INK_BLACK, 2)
    cv2.putText(
        canvas,
        "1. QUICK, DRAW! TOMOGRAPHIC SLIT-SCAN SLICES (u-tau & v-tau CROSS-SECTIONS // INTERACTIVE VIDEO CUBISM)",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, INK_BLACK, 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Left: Archival Graphite u-tau Horizon Slice + Worm c(tau) | Mid: Quick, Draw! Cartoon u-tau Slice | Right: Vertical v-tau Slice + Jerk Cues",
        (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK_GRAPHITE, 1, cv2.LINE_AA
    )

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        draw_qd_card(canvas, x0 + 6, y0 + 6, x0 + cell_w - 8, y0 + cell_h - 8, PAPER_LIGHT, shadow=3)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 32), PAPER_DARK, -1)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 32), INK_BLACK, 2)
        lbl = f"#{rec['rank']:03d} {rec['slug'].upper()} // {rec['topology_code'].upper()} (shear={rec['shear_deg']:+.1f}deg)"
        cv2.putText(canvas, lbl, (x0 + 12, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.39, INK_BLACK, 1, cv2.LINE_AA)
        draw_qd_doodle_stamp(canvas, rec["slug"], x0 + cell_w - 24, y0 + 19, 18.0, fill_bgr=GOUACHE_AMBER, thickness=1)

        arch_frs = rec["_arch_frames"]
        cart_frs = rec["_cart_frames"]
        v_slice_idx = int(np.clip(np.mean(rec["worm_v"]) * 119, 12, 108))
        u_slice_idx = int(np.clip(np.mean(rec["worm_u"]) * 159, 16, 144))

        arch_utau = np.stack([fr[v_slice_idx, :, :] for fr in arch_frs], axis=0)
        cart_utau = np.stack([fr[v_slice_idx, :, :] for fr in cart_frs], axis=0)
        cart_vtau = np.stack([fr[:, u_slice_idx, :] for fr in cart_frs], axis=0)

        arch_utau_img = archival_slice_to_qd_sketch(cv2.resize(arch_utau, (124, 196), interpolation=cv2.INTER_LINEAR))
        cart_utau_img = cv2.resize(cart_utau, (124, 196), interpolation=cv2.INTER_LINEAR)
        cart_vtau_img = cv2.resize(cart_vtau, (124, 196), interpolation=cv2.INTER_LINEAR)

        for panel_img, coords in [(arch_utau_img, rec["worm_u"]), (cart_utau_img, rec["worm_u"]), (cart_vtau_img, rec["worm_v"])]:
            ph, pw = panel_img.shape[:2]
            for gx in [int(0.333 * pw), int(0.667 * pw), int(0.618 * pw)]:
                cv2.line(panel_img, (gx, 0), (gx, ph - 1), ACCENT_BLUE, 1, cv2.LINE_AA)
            pts = []
            for t_i, val in enumerate(coords):
                px = int(np.clip(val * (pw - 1), 2, pw - 3))
                py = int(round((t_i / 32.0) * (ph - 1)))
                pts.append([px, py])
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(panel_img, [pts_np], False, INK_BLACK, 4, cv2.LINE_AA)
            cv2.polylines(panel_img, [pts_np], False, ACCENT_GOLD, 2, cv2.LINE_AA)
            for cue in rec["earshot_cues"]:
                py = int(round(((cue["frame"] - 1) / 32.0) * (ph - 1)))
                px = pts[cue["frame"] - 1][0]
                cv2.circle(panel_img, (px, py), 5, ACCENT_RED, -1, cv2.LINE_AA)
                cv2.circle(panel_img, (px, py), 5, INK_BLACK, 1, cv2.LINE_AA)

        py0 = y0 + 38
        for p_idx, (p_im, p_title) in enumerate([
            (arch_utau_img, "Pencil u-tau"),
            (cart_utau_img, "Cartoon u-tau"),
            (cart_vtau_img, "Vertical v-tau"),
        ]):
            px0 = x0 + 13 + p_idx * 132
            canvas[py0 + 16:py0 + 212, px0:px0 + 124] = p_im
            cv2.rectangle(canvas, (px0, py0 + 16), (px0 + 124, py0 + 212), INK_BLACK, 2)
            cv2.putText(canvas, p_title, (px0 + 4, py0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, INK_BLACK, 1, cv2.LINE_AA)

        cap_txt = f"dL*={rec['mean_chiaroscuro_dL']:.1f} | dH={rec['mean_hue_polarity_dH']:.0f}deg | d_grid={rec['mean_d_grid']:.3f} | Jerk={rec['peak_jerk']:.2f}"
        cv2.putText(canvas, cap_txt, (x0 + 12, y0 + cell_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, INK_GRAPHITE, 1, cv2.LINE_AA)

    out_path = PREVIEW_DIR / "worm_slitscan_tomography_atlas.png"
    cv2.imwrite(str(out_path), canvas)
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
    """Visualization 2: 3D Quick, Draw! Paper-Prism Volume Rendering with Extruded Action Tubes & 14-Line Armature Filaments."""
    featured_slugs = [
        "giraffe-walks-past", "train", "lighthouse", "guitar",
        "tractor", "speedboat", "bicycle", "carrot",
        "sandwich", "diamond", "star", "zigzag",
    ]
    rec_map = {r["slug"]: r for r in records}
    chosen = [rec_map[s] for s in featured_slugs if s in rec_map]

    cols, rows = 4, 3
    cell_w, cell_h = 420, 300
    header_h = 68
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (cols * cell_w, header_h - 2), INK_BLACK, 2)
    cv2.putText(
        canvas,
        "2. 3D QUICK, DRAW! PAPER-PRISM VOLUME V(u,v,tau) // EXTRUDED DOODLE WORMS & 14-LINE ARMATURE",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, INK_BLACK, 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Red/Blue Filaments: Bouleau 14-Line Armature + Thirds | Gouache Ink Tube: Core Worm + Stamped Quick, Draw! Doodles at tau=0 & tau=1",
        (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, INK_GRAPHITE, 1, cv2.LINE_AA
    )

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        draw_qd_card(canvas, x0 + 6, y0 + 6, x0 + cell_w - 8, y0 + cell_h - 8, PAPER_LIGHT, shadow=3)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 34), PAPER_DARK, -1)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 34), INK_BLACK, 2)

        cx = x0 + cell_w // 2 - 8
        cy = y0 + cell_h // 2 + 10

        # Subtle drafting-paper grid inside cell
        for gx in range(x0 + 24, x0 + cell_w - 16, 28):
            cv2.line(canvas, (gx, y0 + 36), (gx, y0 + cell_h - 38), (232, 240, 246), 1)
        for gy in range(y0 + 52, y0 + cell_h - 38, 28):
            cv2.line(canvas, (x0 + 12, gy), (x0 + cell_w - 14, gy), (232, 240, 246), 1)

        corners = {}
        for u_c in (0.0, 1.0):
            for v_c in (0.0, 1.0):
                for t_c in (0.0, 1.0):
                    corners[(u_c, v_c, t_c)] = project_3d_iso(u_c, v_c, t_c, cx, cy)

        prism_edges = [
            ((0,0,0), (1,0,0)), ((1,0,0), (1,1,0)), ((1,1,0), (0,1,0)), ((0,1,0), (0,0,0)),
            ((0,0,1), (1,0,1)), ((1,0,1), (1,1,1)), ((1,1,1), (0,1,1)), ((0,1,1), (0,0,1)),
            ((0,0,0), (0,0,1)), ((1,0,0), (1,0,1)), ((1,1,0), (1,1,1)), ((0,1,0), (0,1,1)),
        ]
        for p1, p2 in prism_edges:
            cv2.line(canvas, corners[p1], corners[p2], INK_BLACK, 2, cv2.LINE_AA)

        for g_u in (0.3333, 0.6180, 0.6667):
            p_start = project_3d_iso(g_u, 0.3820, 0.0, cx, cy)
            p_end = project_3d_iso(g_u, 0.3820, 1.0, cx, cy)
            cv2.line(canvas, p_start, p_end, ACCENT_RED, 1, cv2.LINE_AA)
        for u1, v1, u2, v2, _ in ARMATURE_14_LINES[:6]:
            pa = project_3d_iso(u1, v1, 0.5, cx, cy)
            pb = project_3d_iso(u2, v2, 0.5, cx, cy)
            cv2.line(canvas, pa, pb, ACCENT_BLUE, 1, cv2.LINE_AA)

        sec_pts = []
        for t_i in range(33):
            tau_v = t_i / 32.0
            sec_pts.append(project_3d_iso(rec["sec_u"][t_i], rec["sec_v"][t_i], tau_v, cx, cy))
        for t_i in range(len(sec_pts) - 1):
            cv2.line(canvas, sec_pts[t_i], sec_pts[t_i + 1], ACCENT_BLUE, 3, cv2.LINE_AA)

        gouache_cycle = [GOUACHE_AMBER, GOUACHE_MINT, GOUACHE_BLUE, GOUACHE_CORAL]
        g_tint0 = gouache_cycle[idx % len(gouache_cycle)]
        g_tint1 = gouache_cycle[(idx + 1) % len(gouache_cycle)]
        c_bgr0 = tuple(int(0.45 * hex_to_bgr(rec["core_palette"][0])[k] + 0.55 * g_tint0[k]) for k in range(3))
        c_bgr1 = tuple(int(0.45 * hex_to_bgr(rec["core_palette"][-1])[k] + 0.55 * g_tint1[k]) for k in range(3))
        worm_pts = []
        for t_i in range(33):
            tau_v = t_i / 32.0
            px, py = project_3d_iso(rec["worm_u"][t_i], rec["worm_v"][t_i], tau_v, cx, cy)
            rad_px = int(np.clip(rec["worm_r"][t_i] * 54.0, 6, 22))
            alpha = t_i / 32.0
            col = tuple(int(np.clip((1 - alpha) * c_bgr0[k] + alpha * c_bgr1[k], 70, 250)) for k in range(3))
            worm_pts.append((px, py))
            if t_i % 2 == 0 or t_i == 32:
                cv2.circle(canvas, (px + 2, py + 2), rad_px, INK_BLACK, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), rad_px, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), rad_px, INK_BLACK, 2, cv2.LINE_AA)

        pts_np = np.array(worm_pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_np], False, INK_BLACK, 4, cv2.LINE_AA)
        cv2.polylines(canvas, [pts_np], False, ACCENT_GOLD, 2, cv2.LINE_AA)

        # Stamp the genuine Quick, Draw! doodle at tau=0 (entry) and tau=1 (exit) of the 3D worm!
        draw_qd_doodle_stamp(canvas, rec["slug"], worm_pts[0][0], worm_pts[0][1], 28.0, fill_bgr=GOUACHE_AMBER, thickness=2)
        draw_qd_doodle_stamp(canvas, rec["slug"], worm_pts[-1][0], worm_pts[-1][1], 32.0, fill_bgr=PAPER_LIGHT, thickness=2)

        for cue in rec["earshot_cues"]:
            qx, qy = worm_pts[cue["frame"] - 1]
            cv2.circle(canvas, (qx, qy), 6, ACCENT_RED, -1, cv2.LINE_AA)
            cv2.circle(canvas, (qx, qy), 7, INK_BLACK, 2, cv2.LINE_AA)

        cv2.putText(
            canvas, f"#{rec['rank']:03d} {rec['slug'].upper()} // {rec['topology'].split('(')[0].strip()}",
            (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, INK_BLACK, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Filament: {rec['dominant_filament']} (Lock {rec['compositional_lock_pct']:.0f}% | Arnheim {rec['arnheim_tension_pct']:.0f}%)",
            (x0 + 12, y0 + cell_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, INK_BLACK, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"kappa={rec['mean_curvature']:.2f} | tau={rec['mean_torsion']:.2f} | SFX@{[c['frame'] for c in rec['earshot_cues']]}",
            (x0 + 12, y0 + cell_h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, ACCENT_GREEN, 1, cv2.LINE_AA
        )

    out_path = PREVIEW_DIR / "worm_3d_glass_prism_atlas.png"
    cv2.imwrite(str(out_path), canvas)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_3d_glass_prism_atlas.png")
    return out_path


def render_cinemetrics_torus_atlas(records):
    """Visualization 3: Frederic Brodbeck 360-deg Radial Cinemetrics Torus Atlas (Unified Quick, Draw! Aesthetic)."""
    featured_slugs = [
        "giraffe-walks-past", "train", "lighthouse", "guitar",
        "tractor", "speedboat", "bicycle", "carrot",
        "sandwich", "diamond", "star", "zigzag",
    ]
    rec_map = {r["slug"]: r for r in records}
    chosen = [rec_map[s] for s in featured_slugs if s in rec_map]

    cols, rows = 4, 3
    cell_w, cell_h = 400, 300
    header_h = 68
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (cols * cell_w, header_h - 2), INK_BLACK, 2)
    cv2.putText(
        canvas,
        "3. QUICK, DRAW! RADIAL CINEMETRICS TORUS (BRODBECK 360-DEG RING // KALMUS BANDS + JERK SPIKES)",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, INK_BLACK, 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Inner 5 Gouache Bands: 3 Core-Worm + 2 Outer-Shell KALMUS Palette | Center: Quick, Draw! Doodle | Outer Spikes: Kinetic Jerk ||c'''(tau)||",
        (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, INK_GRAPHITE, 1, cv2.LINE_AA
    )

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        draw_qd_card(canvas, x0 + 6, y0 + 6, x0 + cell_w - 8, y0 + cell_h - 8, PAPER_LIGHT, shadow=3)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 32), PAPER_DARK, -1)
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x0 + cell_w - 8, y0 + 32), INK_BLACK, 2)

        cx = x0 + cell_w // 2
        cy = y0 + cell_h // 2 + 12

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

        # Stamp the genuine Quick, Draw! category doodle inside the center of the torus!
        draw_qd_doodle_stamp(canvas, rec["slug"], cx, cy, 36.0, fill_bgr=GOUACHE_AMBER, thickness=2)

        max_j = max(rec["jerk_curve"]) + 1e-4
        for t_i in range(33):
            ang = -math.pi / 2.0 + (2.0 * math.pi * t_i / 33.0)
            inten = rec["intensity_curve"][t_i]
            j_norm = rec["jerk_curve"][t_i] / max_j
            spike_len = int(12 + 32 * (0.55 * inten + 0.45 * j_norm))
            x1 = int(round(cx + 80 * math.cos(ang)))
            y1 = int(round(cy + 80 * math.sin(ang)))
            x2 = int(round(cx + (80 + spike_len) * math.cos(ang)))
            y2 = int(round(cy + (80 + spike_len) * math.sin(ang)))
            is_key = t_i in (0, 10, 21, 32)
            col = ACCENT_RED if is_key else INK_BLACK
            thick = 3 if is_key else 2
            cv2.line(canvas, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)

        cv2.putText(
            canvas, f"#{rec['rank']:03d} {rec['slug'].upper()} // {rec['topology_code'].upper()}",
            (x0 + 12, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.41, INK_BLACK, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Itten dL*={rec['mean_chiaroscuro_dL']:.1f} | dH={rec['mean_hue_polarity_dH']:.0f}deg | Area={rec['mean_extension_ratio']*100:.1f}%",
            (x0 + 12, y0 + cell_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, INK_GRAPHITE, 1, cv2.LINE_AA
        )

    out_path = PREVIEW_DIR / "worm_cinemetrics_torus_atlas.png"
    cv2.imwrite(str(out_path), canvas)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_cinemetrics_torus_atlas.png")
    return out_path


def render_manovich_montage_grid(records):
    """Visualization 4: Lev Manovich Cultural Analytics Montage Grid + Bruce Block Visual Intensity Graph across all 346 shots (Unified Quick, Draw! Aesthetic)."""
    cols, rows = 20, 18
    cw, ch = 84, 54
    header_h = 68
    footer_h = 128
    W = cols * cw
    H = header_h + rows * ch + footer_h
    canvas = np.full((H, W, 3), PAPER_CREAM, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (W, header_h), GOUACHE_AMBER, -1)
    cv2.line(canvas, (0, header_h - 2), (W, header_h - 2), INK_BLACK, 2)
    cv2.putText(
        canvas,
        "4. QUICK, DRAW! CULTURAL ANALYTICS MONTAGE GRID (LEV MANOVICH + BRUCE BLOCK INTENSITY ACROSS ALL 346 SHOTS)",
        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, INK_BLACK, 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Every Cell = 1 Shot (#001..#346): Midpoint Quick, Draw! Multiplane Frame + Extruded Worm Ribbon + Gouache Topology Header + Bruce Block Wave Below",
        (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, INK_GRAPHITE, 1, cv2.LINE_AA
    )

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
        cv2.rectangle(mid_fr, (0, 0), (cw - 2, 8), tcol, -1)
        cv2.line(mid_fr, (0, 8), (cw - 2, 8), INK_BLACK, 1)
        cv2.putText(mid_fr, f"{rec['rank']:03d}", (2, ch - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, INK_BLACK, 1, cv2.LINE_AA)

        canvas[y0 + 1:y0 + ch - 1, x0 + 1:x0 + cw - 1] = mid_fr
        cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + cw - 1, y0 + ch - 1), INK_BLACK, 1)
        macro_intensities.append(float(np.mean(rec["intensity_curve"])))
        macro_dL.append(float(rec["mean_chiaroscuro_dL"]))

    fy0 = header_h + rows * ch + 10
    fh = footer_h - 22
    draw_qd_card(canvas, 14, fy0, W - 16, fy0 + fh, PAPER_LIGHT, shadow=4)
    cv2.putText(
        canvas,
        "BRUCE BLOCK MACRO VISUAL INTENSITY GRAPH (FOREST GREEN) & KALMUS CHIAROSCURO CONTRAST dL* (WAX RED) ACROSS ALL 346 SHOTS (#001 -> #346)",
        (26, fy0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK_BLACK, 1, cv2.LINE_AA
    )

    for gy in range(fy0 + 34, fy0 + fh - 10, 22):
        cv2.line(canvas, (24, gy), (W - 26, gy), PAPER_DARK, 1)

    pts_int = []
    pts_dl = []
    for i, (val_i, val_l) in enumerate(zip(macro_intensities, macro_dL)):
        px = int(round(24 + (i / max(len(macro_intensities) - 1, 1)) * (W - 50)))
        py_i = int(round(fy0 + fh - 12 - np.clip(val_i, 0.0, 1.0) * (fh - 38)))
        py_l = int(round(fy0 + fh - 12 - np.clip(val_l / 55.0, 0.0, 1.0) * (fh - 38)))
        pts_int.append([px, py_i])
        pts_dl.append([px, py_l])

    cv2.polylines(canvas, [np.array(pts_dl, dtype=np.int32)], False, ACCENT_RED, 2, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(pts_int, dtype=np.int32)], False, ACCENT_GREEN, 2, cv2.LINE_AA)

    out_path = PREVIEW_DIR / "worm_manovich_montage_grid.png"
    cv2.imwrite(str(out_path), canvas)
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
