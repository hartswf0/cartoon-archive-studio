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


def render_slitscan_tomography_atlas(records):
    """Visualization 1: Fels & Mase Interactive Video Cubism / Video Summagator u-tau & v-tau Slit-Scan Atlas."""
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
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), (235, 242, 247), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), (20, 22, 26), -1)
    cv2.putText(
        canvas,
        "1. TOMOGRAPHIC SLIT-SCAN SLICES (u-tau & v-tau CROSS-SECTIONS // INTERACTIVE VIDEO CUBISM & SUMMAGATOR)",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 230, 210), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Left panel: Archival u-tau Horizon Slice + Extruded Worm Centerline c(tau) | Mid: Cartoon u-tau Multiplane Slice | Right: v-tau Vertical Slice + Jerk Spikes",
        (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 205, 195), 1, cv2.LINE_AA
    )

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + cell_w - 4, y0 + cell_h - 4), (248, 251, 254), -1)
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + cell_w - 4, y0 + cell_h - 4), (30, 30, 30), 2)

        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + cell_w - 4, y0 + 30), (28, 32, 36), -1)
        lbl = f"#{rec['rank']:03d} {rec['slug'].upper()} // {rec['topology_code'].upper()} (shear={rec['shear_deg']:+.1f}deg)"
        cv2.putText(canvas, lbl, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (245, 238, 220), 1, cv2.LINE_AA)

        arch_frs = rec["_arch_frames"]
        cart_frs = rec["_cart_frames"]
        v_slice_idx = int(np.clip(np.mean(rec["worm_v"]) * 119, 12, 108))
        u_slice_idx = int(np.clip(np.mean(rec["worm_u"]) * 159, 16, 144))

        arch_utau = np.stack([fr[v_slice_idx, :, :] for fr in arch_frs], axis=0)
        cart_utau = np.stack([fr[v_slice_idx, :, :] for fr in cart_frs], axis=0)
        cart_vtau = np.stack([fr[:, u_slice_idx, :] for fr in cart_frs], axis=0)

        arch_utau_img = cv2.resize(arch_utau, (126, 200), interpolation=cv2.INTER_LINEAR)
        cart_utau_img = cv2.resize(cart_utau, (126, 200), interpolation=cv2.INTER_LINEAR)
        cart_vtau_img = cv2.resize(cart_vtau, (126, 200), interpolation=cv2.INTER_LINEAR)

        for panel_img, coords in [(arch_utau_img, rec["worm_u"]), (cart_utau_img, rec["worm_u"]), (cart_vtau_img, rec["worm_v"])]:
            ph, pw = panel_img.shape[:2]
            for gx in [int(0.333 * pw), int(0.667 * pw), int(0.618 * pw)]:
                cv2.line(panel_img, (gx, 0), (gx, ph - 1), (90, 190, 245), 1, cv2.LINE_AA)
            pts = []
            for t_i, val in enumerate(coords):
                px = int(np.clip(val * (pw - 1), 2, pw - 3))
                py = int(round((t_i / 32.0) * (ph - 1)))
                pts.append([px, py])
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(panel_img, [pts_np], False, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.polylines(panel_img, [pts_np], False, (50, 235, 120), 2, cv2.LINE_AA)
            for cue in rec["earshot_cues"]:
                py = int(round(((cue["frame"] - 1) / 32.0) * (ph - 1)))
                px = pts[cue["frame"] - 1][0]
                cv2.circle(panel_img, (px, py), 5, (30, 70, 235), -1, cv2.LINE_AA)
                cv2.circle(panel_img, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

        py0 = y0 + 38
        for p_idx, (p_im, p_title) in enumerate([
            (arch_utau_img, "Archival u-tau"),
            (cart_utau_img, "Cartoon u-tau"),
            (cart_vtau_img, "Vertical v-tau"),
        ]):
            px0 = x0 + 12 + p_idx * 134
            canvas[py0 + 16:py0 + 216, px0:px0 + 126] = p_im
            cv2.rectangle(canvas, (px0, py0 + 16), (px0 + 126, py0 + 216), (30, 30, 30), 1)
            cv2.putText(canvas, p_title, (px0 + 4, py0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1, cv2.LINE_AA)

        cap_txt = f"dL*={rec['mean_chiaroscuro_dL']:.1f} | dH={rec['mean_hue_polarity_dH']:.0f}deg | d_grid={rec['mean_d_grid']:.3f} | Jerk={rec['peak_jerk']:.2f}"
        cv2.putText(canvas, cap_txt, (x0 + 12, y0 + cell_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (45, 45, 45), 1, cv2.LINE_AA)

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
    """Visualization 2: 3D Glass-Prism Volume Rendering with Extruded Action Tubes & 14-Line Armature Filaments."""
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
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), (18, 20, 26), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), (12, 14, 18), -1)
    cv2.putText(
        canvas,
        "2. 3D GLASS-PRISM VOLUME V(u,v,tau) // EXTRUDED ACTION WORMS & BOULEAU 14-LINE ARMATURE FILAMENTS",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (110, 215, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Gold Filaments: Bouleau 14-Line Armature + Golden Spiral | Solid Tube: Core Worm (KALMUS Palette) | Cyan Tube: Secondary Counterpoint Worm",
        (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (195, 205, 215), 1, cv2.LINE_AA
    )

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        cv2.rectangle(canvas, (x0 + 5, y0 + 5), (x0 + cell_w - 5, y0 + cell_h - 5), (26, 29, 37), -1)
        cv2.rectangle(canvas, (x0 + 5, y0 + 5), (x0 + cell_w - 5, y0 + cell_h - 5), (75, 85, 105), 1)

        cx = x0 + cell_w // 2 - 10
        cy = y0 + cell_h // 2 + 12

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
            cv2.line(canvas, corners[p1], corners[p2], (70, 82, 102), 1, cv2.LINE_AA)

        for g_u in (0.3333, 0.6180, 0.6667):
            p_start = project_3d_iso(g_u, 0.3820, 0.0, cx, cy)
            p_end = project_3d_iso(g_u, 0.3820, 1.0, cx, cy)
            cv2.line(canvas, p_start, p_end, (55, 165, 225), 1, cv2.LINE_AA)
        for u1, v1, u2, v2, _ in ARMATURE_14_LINES[:6]:
            pa = project_3d_iso(u1, v1, 0.5, cx, cy)
            pb = project_3d_iso(u2, v2, 0.5, cx, cy)
            cv2.line(canvas, pa, pb, (45, 130, 175), 1, cv2.LINE_AA)

        sec_pts = []
        for t_i in range(33):
            tau_v = t_i / 32.0
            sec_pts.append(project_3d_iso(rec["sec_u"][t_i], rec["sec_v"][t_i], tau_v, cx, cy))
        for t_i in range(len(sec_pts) - 1):
            cv2.line(canvas, sec_pts[t_i], sec_pts[t_i + 1], (210, 195, 75), 3, cv2.LINE_AA)

        c_bgr0 = hex_to_bgr(rec["core_palette"][0])
        c_bgr1 = hex_to_bgr(rec["core_palette"][-1])
        worm_pts = []
        for t_i in range(33):
            tau_v = t_i / 32.0
            px, py = project_3d_iso(rec["worm_u"][t_i], rec["worm_v"][t_i], tau_v, cx, cy)
            rad_px = int(np.clip(rec["worm_r"][t_i] * 58.0, 6, 24))
            alpha = t_i / 32.0
            col = tuple(int(np.clip((1 - alpha) * c_bgr0[k] + alpha * c_bgr1[k] + 45, 50, 250)) for k in range(3))
            worm_pts.append((px, py))
            cv2.circle(canvas, (px, py), rad_px + 2, (20, 22, 28), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), rad_px, col, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), rad_px, (235, 240, 245), 1, cv2.LINE_AA)

        pts_np = np.array(worm_pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_np], False, (255, 255, 255), 2, cv2.LINE_AA)
        for cue in rec["earshot_cues"]:
            qx, qy = worm_pts[cue["frame"] - 1]
            cv2.circle(canvas, (qx, qy), 6, (40, 65, 250), -1, cv2.LINE_AA)
            cv2.circle(canvas, (qx, qy), 8, (110, 230, 255), 1, cv2.LINE_AA)

        cv2.putText(
            canvas, f"#{rec['rank']:03d} {rec['slug'].upper()}",
            (x0 + 14, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 245, 220), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, rec["topology"].split("(")[0].strip(),
            (x0 + 14, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (115, 220, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Filament: {rec['dominant_filament']} (Lock {rec['compositional_lock_pct']:.0f}% | Arnheim {rec['arnheim_tension_pct']:.0f}%)",
            (x0 + 14, y0 + cell_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (195, 205, 215), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"kappa={rec['mean_curvature']:.2f} | tau={rec['mean_torsion']:.2f} | SFX@{[c['frame'] for c in rec['earshot_cues']]}",
            (x0 + 14, y0 + cell_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 235, 175), 1, cv2.LINE_AA
        )

    out_path = PREVIEW_DIR / "worm_3d_glass_prism_atlas.png"
    cv2.imwrite(str(out_path), canvas)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_3d_glass_prism_atlas.png")
    return out_path


def render_cinemetrics_torus_atlas(records):
    """Visualization 3: Frederic Brodbeck 360-deg Radial Cinemetrics Torus Atlas."""
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
    canvas = np.full((header_h + rows * cell_h, cols * cell_w, 3), (242, 238, 230), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (cols * cell_w, header_h), (22, 20, 18), -1)
    cv2.putText(
        canvas,
        "3. RADIAL CINEMETRICS TORUS (FREDERIC BRODBECK 360-DEG RING // KALMUS BANDS + MOTION JERK SPIKES)",
        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 235, 215), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Inner 5 Concentric Bands: 3 Core-Worm + 2 Outer-Shell KALMUS Palette | Outer Radial Spikes: Kinetic Jerk ||c'''(tau)|| & Visual Intensity",
        (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (195, 210, 200), 1, cv2.LINE_AA
    )

    for idx, rec in enumerate(chosen[:12]):
        r_i = idx // cols
        c_i = idx % cols
        x0 = c_i * cell_w
        y0 = header_h + r_i * cell_h

        cv2.rectangle(canvas, (x0 + 5, y0 + 5), (x0 + cell_w - 5, y0 + cell_h - 5), (252, 250, 245), -1)
        cv2.rectangle(canvas, (x0 + 5, y0 + 5), (x0 + cell_w - 5, y0 + cell_h - 5), (30, 28, 25), 2)

        cx = x0 + cell_w // 2
        cy = y0 + cell_h // 2 + 8

        swatches = rec["core_palette"] + rec["shell_palette"]
        radii = [34, 44, 54, 64, 74]
        for b_i, r_px in enumerate(reversed(radii)):
            col = hex_to_bgr(swatches[b_i % len(swatches)])
            cv2.circle(canvas, (cx, cy), r_px, col, 9, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 26, (252, 250, 245), -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 79, (25, 25, 25), 2, cv2.LINE_AA)

        max_j = max(rec["jerk_curve"]) + 1e-4
        for t_i in range(33):
            ang = -math.pi / 2.0 + (2.0 * math.pi * t_i / 33.0)
            inten = rec["intensity_curve"][t_i]
            j_norm = rec["jerk_curve"][t_i] / max_j
            spike_len = int(12 + 34 * (0.55 * inten + 0.45 * j_norm))
            x1 = int(round(cx + 80 * math.cos(ang)))
            y1 = int(round(cy + 80 * math.sin(ang)))
            x2 = int(round(cx + (80 + spike_len) * math.cos(ang)))
            y2 = int(round(cy + (80 + spike_len) * math.sin(ang)))
            is_key = t_i in (0, 10, 21, 32)
            col = (35, 65, 215) if is_key else (35, 110, 75)
            thick = 3 if is_key else 2
            cv2.line(canvas, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)

        cv2.putText(canvas, f"#{rec['rank']:03d}", (cx - 19, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

        cv2.putText(
            canvas, f"{rec['slug'].upper()} // {rec['topology_code'].upper()}",
            (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"Itten dL*={rec['mean_chiaroscuro_dL']:.1f} | dH={rec['mean_hue_polarity_dH']:.0f}deg | Area={rec['mean_extension_ratio']*100:.1f}%",
            (x0 + 12, y0 + cell_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 55, 50), 1, cv2.LINE_AA
        )

    out_path = PREVIEW_DIR / "worm_cinemetrics_torus_atlas.png"
    cv2.imwrite(str(out_path), canvas)
    shutil.copy2(out_path, ARTIFACT_DIR / "worm_cinemetrics_torus_atlas.png")
    return out_path


def render_manovich_montage_grid(records):
    """Visualization 4: Lev Manovich Cultural Analytics Montage Grid + Bruce Block Visual Intensity Graph across all 346 shots."""
    cols, rows = 20, 18
    cw, ch = 84, 54
    header_h = 68
    footer_h = 120
    W = cols * cw
    H = header_h + rows * ch + footer_h
    canvas = np.full((H, W, 3), (20, 22, 26), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (W, header_h), (12, 14, 18), -1)
    cv2.putText(
        canvas,
        "4. CULTURAL ANALYTICS MONTAGE GRID (LEV MANOVICH VISUALIZING VERTOV + BRUCE BLOCK INTENSITY ACROSS ALL 346 SHOTS)",
        (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 232, 205), 2, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "Every cell = 1 Shot (#001..#346): Midpoint Cartoon Frame + Extruded Worm Trajectory Ribbon (Green) + KALMUS Core/Shell Swatches + Macro Bruce Block Wave Below",
        (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (185, 205, 195), 1, cv2.LINE_AA
    )

    macro_intensities = []
    macro_dL = []

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
        cv2.polylines(mid_fr, [pts_np], False, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.polylines(mid_fr, [pts_np], False, (55, 240, 125), 1, cv2.LINE_AA)

        topo_colors = {
            "vertical_pillar": (210, 175, 65),
            "sheared_ribbon": (65, 205, 115),
            "expanding_cone": (55, 115, 235),
            "helical_corkscrew": (215, 95, 215),
            "double_helix": (70, 210, 235),
        }
        tcol = topo_colors.get(rec["topology_code"], (180, 180, 180))
        cv2.rectangle(mid_fr, (0, 0), (cw - 2, 7), tcol, -1)
        cv2.putText(mid_fr, f"{rec['rank']:03d}", (2, ch - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (15, 15, 15), 1, cv2.LINE_AA)

        canvas[y0 + 1:y0 + ch - 1, x0 + 1:x0 + cw - 1] = mid_fr
        macro_intensities.append(float(np.mean(rec["intensity_curve"])))
        macro_dL.append(float(rec["mean_chiaroscuro_dL"]))

    fy0 = header_h + rows * ch + 10
    fh = footer_h - 20
    cv2.rectangle(canvas, (14, fy0), (W - 14, fy0 + fh), (28, 31, 38), -1)
    cv2.rectangle(canvas, (14, fy0), (W - 14, fy0 + fh), (80, 90, 110), 1)
    cv2.putText(
        canvas,
        "BRUCE BLOCK MACRO VISUAL INTENSITY GRAPH (GREEN) & KALMUS CHIAROSCURO CONTRAST dL* (ORANGE) ACROSS ALL 346 SHOTS (#001 -> #346)",
        (24, fy0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 235, 240), 1, cv2.LINE_AA
    )

    pts_int = []
    pts_dl = []
    for i, (val_i, val_l) in enumerate(zip(macro_intensities, macro_dL)):
        px = int(round(24 + (i / max(len(macro_intensities) - 1, 1)) * (W - 48)))
        py_i = int(round(fy0 + fh - 10 - np.clip(val_i, 0.0, 1.0) * (fh - 34)))
        py_l = int(round(fy0 + fh - 10 - np.clip(val_l / 55.0, 0.0, 1.0) * (fh - 34)))
        pts_int.append([px, py_i])
        pts_dl.append([px, py_l])

    cv2.polylines(canvas, [np.array(pts_dl, dtype=np.int32)], False, (55, 155, 245), 1, cv2.LINE_AA)
    cv2.polylines(canvas, [np.array(pts_int, dtype=np.int32)], False, (75, 235, 130), 2, cv2.LINE_AA)

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
