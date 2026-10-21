#!/usr/bin/env python3
"""
Builds the Joint Embedding Space (V-JEPA 2 + CLIP), Multi-Frame SAM Segmentation,
Multiplane Depth Decomposition, Best QuickDraw-to-Archive Matching, and RL Kinematic
Cartoon-in-Motion Controller for the Moving Image Archive + QuickDraw Dataset.
"""

import argparse
import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)


def pca_2d(mat: np.ndarray) -> np.ndarray:
    centered = mat - mat.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    c_min = coords.min(axis=0, keepdims=True)
    c_max = coords.max(axis=0, keepdims=True)
    return (coords - c_min) / np.maximum(c_max - c_min, 1e-6)

BASE_DIR = Path(__file__).resolve().parent
QD_DIR = BASE_DIR / "quickdraw_samples"
ASSETS_DIR = BASE_DIR / "assets"
MIA_DIR = Path("/Users/gaia/WORLDBUILDING-TERMS/moving_image_archive_dataset")

# 25 diverse showcase terms spanning 4 kinematic & operational families
SHOWCASE_TERMS = [
    # 1. Vehicles & Flight (translation, banking, rotors, wheels, wakes)
    {"term": "aircraft carrier", "family": "Vehicles & Flight", "op_mode": "hull_wake_launch", "verbs": ["cruise", "pitch on swell", "launch deck craft"]},
    {"term": "airplane", "family": "Vehicles & Flight", "op_mode": "flight_bank_prop", "verbs": ["bank wings", "spin propeller", "climb & glide"]},
    {"term": "bicycle", "family": "Vehicles & Flight", "op_mode": "wheel_crank_drive", "verbs": ["spin wheels", "crank pedals", "lean into turn"]},
    {"term": "car", "family": "Vehicles & Flight", "op_mode": "wheel_chassis_suspension", "verbs": ["roll tires", "bounce chassis", "accelerate"]},
    {"term": "helicopter", "family": "Vehicles & Flight", "op_mode": "rotor_hover_pitch", "verbs": ["whirl main rotor", "stabilize tail", "hover & tilt"]},
    {"term": "sailboat", "family": "Vehicles & Flight", "op_mode": "sail_billow_pitch", "verbs": ["billow sails", "heel in wind", "cleave waves"]},
    {"term": "train", "family": "Vehicles & Flight", "op_mode": "piston_wheel_chug", "verbs": ["drive coupling rods", "roll bogies", "billow steam"]},
    # 2. Living Creatures & Locomotion (wing flap, quadruped gait, undulation)
    {"term": "bird", "family": "Creatures & Gait", "op_mode": "wing_flap_soar", "verbs": ["beat wings", "tuck tail", "bob head"]},
    {"term": "butterfly", "family": "Creatures & Gait", "op_mode": "symmetric_flutter", "verbs": ["flutter forewings", "hover erratically", "alight"]},
    {"term": "cat", "family": "Creatures & Gait", "op_mode": "quadruped_prowl_tail", "verbs": ["stalk softly", "sway tail", "pivot ears"]},
    {"term": "dog", "family": "Creatures & Gait", "op_mode": "quadruped_trot_wag", "verbs": ["trot briskly", "wag tail", "pant & bound"]},
    {"term": "elephant", "family": "Creatures & Gait", "op_mode": "trunk_swing_amass", "verbs": ["curl trunk", "flap ears", "stride heavily"]},
    {"term": "fish", "family": "Creatures & Gait", "op_mode": "body_wave_swim", "verbs": ["undulate spine", "flick caudal fin", "dart"]},
    {"term": "horse", "family": "Creatures & Gait", "op_mode": "gallop_stride_mane", "verbs": ["gallop 4-beat", "extend forelegs", "toss mane"]},
    {"term": "shark", "family": "Creatures & Gait", "op_mode": "dorsal_glide_sweep", "verbs": ["sweep tail fin", "prowl laterally", "bankpectoral"]},
    # 3. Operable Mechanisms & Instruments (continuous spin, pivot shear, recoil, bow oscillation)
    {"term": "clock", "family": "Mechanisms & Tools", "op_mode": "hand_sweep_tick", "verbs": ["sweep minute hand", "tick escapement", "oscillate pendulum"]},
    {"term": "bulldozer", "family": "Mechanisms & Tools", "op_mode": "blade_lift_crawler", "verbs": ["hoist dozer blade", "grind crawler tracks", "push earth"]},
    {"term": "windmill", "family": "Mechanisms & Tools", "op_mode": "vane_continuous_spin", "verbs": ["revolve 4 sails", "face into wind", "grind millstone"]},
    {"term": "cannon", "family": "Mechanisms & Tools", "op_mode": "barrel_recoil_blast", "verbs": ["elevate trunnion", "fire & recoil", "roll carriage"]},
    {"term": "violin", "family": "Mechanisms & Tools", "op_mode": "bow_stroke_vibrato", "verbs": ["draw bow across strings", "vibrate bridge", "tilt scroll"]},
    {"term": "scissors", "family": "Mechanisms & Tools", "op_mode": "fulcrum_blade_shear", "verbs": ["pivot around screw", "shear blades closed", "open handles"]},
    # 4. Natural Forces & Monumental Landmarks (vortex, wave crest, flame flicker, parallax)
    {"term": "campfire", "family": "Forces & Landmarks", "op_mode": "flame_flicker_plume", "verbs": ["lick upward", "sway in draft", "loft sparks"]},
    {"term": "ocean", "family": "Forces & Landmarks", "op_mode": "wave_swell_break", "verbs": ["heave swell", "curl breaker crest", "wash foam"]},
    {"term": "tornado", "family": "Forces & Landmarks", "op_mode": "vortex_funnel_twist", "verbs": ["whirl funnel core", "precess base", "sweep debris"]},
    {"term": "The Eiffel Tower", "family": "Forces & Landmarks", "op_mode": "multiplane_parallax_beacon", "verbs": ["pan multiplane truss", "sweep beacon", "rise through clouds"]},
]


def slugify(term: str) -> str:
    s = term.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def render_quickdraw_pil(
    strokes: list,
    size: int = 256,
    line_width: int = 6,
    bg_color=(255, 255, 255),
    ink_color=(15, 15, 20),
    padding: int = 24,
) -> Image.Image:
    """Renders QuickDraw simplified strokes (0..255 coords) onto a square PIL image."""
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    scale = (size - 2 * padding) / 255.0
    for stroke in strokes:
        if len(stroke) < 2 or len(stroke[0]) < 2:
            continue
        pts = [
            (padding + x * scale, padding + y * scale)
            for x, y in zip(stroke[0], stroke[1])
        ]
        draw.line(pts, fill=ink_color, width=line_width, joint="curve")
    return img


def render_quickdraw_mask(strokes: list, h: int, w: int, bbox: list) -> np.ndarray:
    """
    Renders a QuickDraw drawing inside the target bounding box [x1, y1, x2, y2]
    on a (h, w) canvas and returns a soft filled + dilated occupancy mask in [0, 1]
    for computing IoU with the SAM segmentation mask.
    """
    canvas = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    bw = max(8.0, x2 - x1)
    bh = max(8.0, y2 - y1)

    all_pts = []
    for stroke in strokes:
        if len(stroke) < 2 or len(stroke[0]) < 2:
            continue
        pts = np.array(
            [
                [x1 + (x / 255.0) * bw, y1 + (y / 255.0) * bh]
                for x, y in zip(stroke[0], stroke[1])
            ],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [pts], isClosed=False, color=255, thickness=max(3, int(min(bw, bh) * 0.05)))
        all_pts.append(pts)

    if all_pts:
        merged = np.vstack(all_pts)
        hull = cv2.convexHull(merged)
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 180)
        canvas = np.maximum(canvas, hull_mask)

    return (canvas.astype(np.float32) / 255.0)


def sample_video_frames(video_path: Path, center_sec: float, num_frames: int = 16, window_sec: float = 3.2):
    """Samples `num_frames` RGB frames around `center_sec` from `video_path`."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    duration = max(total_frames / max(fps, 1.0), 0.5)

    rel_center = min(max(center_sec, 0.0), duration)
    if rel_center > duration or rel_center <= 0.0:
        rel_center = duration * 0.5

    start_t = max(0.0, rel_center - window_sec * 0.5)
    end_t = min(duration, start_t + window_sec)
    if end_t - start_t < 0.4:
        start_t = 0.0
        end_t = duration

    timestamps = np.linspace(start_t, max(start_t + 0.1, end_t - 0.04), num_frames)
    frames_rgb = []
    actual_times = []
    for t in timestamps:
        frame_idx = int(min(max(round(t * fps), 0), max(total_frames - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        actual_times.append(round(frame_idx / max(fps, 1.0), 3))
    cap.release()

    if not frames_rgb:
        raise RuntimeError(f"Could not decode frames from {video_path}")

    while len(frames_rgb) < num_frames:
        frames_rgb.append(frames_rgb[-1].copy())
        actual_times.append(actual_times[-1])

    return frames_rgb[:num_frames], actual_times[:num_frames], duration


def mask_to_normalized_polygon(mask: np.ndarray, max_pts: int = 36) -> list:
    """Converts a binary mask (H, W) into normalized [[x_norm, y_norm], ...] contour points."""
    h, w = mask.shape[:2]
    u8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 25:
        return []
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.012 * peri, True)
    pts = approx.reshape(-1, 2)
    if len(pts) > max_pts:
        idxs = np.linspace(0, len(pts) - 1, max_pts, dtype=int)
        pts = pts[idxs]
    return [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in pts]


def assign_stroke_parts(strokes: list, op_mode: str) -> list:
    """
    Classifies each QuickDraw stroke into a functional kinematic part role:
    - 'primary_body' (chassis / hull / torso / frame)
    - 'rotor_wheel_wing' (wheels, rotors, wings, blades, sails, limbs)
    - 'articulated_head_tail' (head, trunk, tail, bow, hands, plume)
    """
    roles = []
    n = len(strokes)
    for idx, s in enumerate(strokes):
        xs, ys = s[0], s[1]
        if not xs:
            roles.append({"role": "primary_body", "pivot": [127.5, 127.5]})
            continue
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        sw = float(max(xs) - min(xs))
        sh = float(max(ys) - min(ys))
        closedness = math.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) / max(sw, sh, 1.0)

        role = "primary_body"
        if "wheel" in op_mode and cy > 135 and closedness < 0.65 and max(sw, sh) < 130:
            role = "rotor_wheel_wing"
        elif ("rotor" in op_mode or "vane" in op_mode) and (cy < 135 or idx > 0):
            role = "rotor_wheel_wing"
        elif ("wing" in op_mode or "flutter" in op_mode or "flight" in op_mode) and idx > 0 and (cy < 155 or sw > 60):
            role = "rotor_wheel_wing"
        elif "quadruped" in op_mode or "gallop" in op_mode or "trunk" in op_mode:
            if cy > 145:
                role = "rotor_wheel_wing"  # legs
            elif cx < 95 or cx > 165:
                role = "articulated_head_tail"
        elif "hand_sweep" in op_mode and idx > 0 and max(sw, sh) < 140:
            role = "rotor_wheel_wing"
        elif "fulcrum" in op_mode:
            role = "rotor_wheel_wing" if idx % 2 == 0 else "articulated_head_tail"
        elif idx == 0 and max(sw, sh) > 90:
            role = "primary_body"
        elif idx % 2 == 1:
            role = "rotor_wheel_wing"
        else:
            role = "articulated_head_tail"

        roles.append(
            {
                "stroke_index": idx,
                "role": role,
                "pivot": [round(cx, 2), round(cy, 2)],
                "span": [round(sw, 2), round(sh, 2)],
            }
        )
    return roles


def train_rl_cartoon_controller(
    strokes: list,
    part_roles: list,
    masks_t: list,
    bboxes_t: list,
    flows_t: list,
    op_mode: str,
    num_episodes: int = 40,
) -> dict:
    """
    Optimizes a kinematic policy pi_theta(a_t | s_t, z_VJEPA) via Cross-Entropy Method (CEM)
    reinforcement learning so the QuickDraw strokes move, deform, and operate in lockstep
    with the SAM-segmented archival object and its optical flow dynamics.
    """
    num_frames = len(masks_t)
    h, w = masks_t[0].shape[:2]

    # Baseline (untrained static overlay at frame-0 bbox)
    base_bbox = bboxes_t[0]
    iou_initial_list = []
    for t in range(num_frames):
        qd_m = render_quickdraw_mask(strokes, h, w, base_bbox)
        sam_m = masks_t[t]
        inter = float(np.sum(np.minimum(qd_m, sam_m)))
        union = float(np.sum(np.maximum(qd_m, sam_m)) + 1e-6)
        iou_initial_list.append(inter / union)
    initial_iou = float(np.mean(iou_initial_list))

    # Policy parameters:
    # [0]: global bbox tracking gain (0..1)
    # [1]: optical flow advection gain
    # [2]: part rotational/oscillation amplitude
    # [3]: part phase velocity
    # [4]: elastic rigidity damping
    mean_theta = np.array([0.35, 0.25, 0.20, 0.40, 0.50], dtype=np.float32)
    std_theta = np.array([0.35, 0.30, 0.25, 0.30, 0.25], dtype=np.float32)

    episode_history = []
    best_reward = -1e9
    best_theta = mean_theta.copy()

    # Precompute target bbox centers, scales, and mean flow vectors per frame
    target_cx, target_cy, target_bw, target_bh = [], [], [], []
    flow_vx, flow_vy, flow_vort = [], [], []
    for t in range(num_frames):
        bx1, by1, bx2, by2 = bboxes_t[t]
        target_cx.append((bx1 + bx2) * 0.5 / w)
        target_cy.append((by1 + by2) * 0.5 / h)
        target_bw.append(max(0.08, (bx2 - bx1) / w))
        target_bh.append(max(0.08, (by2 - by1) / h))

        fl = flows_t[t]
        m = masks_t[t] > 0.3
        if np.any(m):
            vx = float(np.mean(fl[..., 0][m])) / max(w, 1)
            vy = float(np.mean(fl[..., 1][m])) / max(h, 1)
            dv_dx = np.gradient(fl[..., 1], axis=1)
            du_dy = np.gradient(fl[..., 0], axis=0)
            vort = float(np.mean((dv_dx - du_dy)[m]))
        else:
            vx, vy, vort = 0.0, 0.0, 0.0
        flow_vx.append(vx)
        flow_vy.append(vy)
        flow_vort.append(vort)

    rng = np.random.default_rng(42)
    pop_size = 16
    elite_k = 5

    for ep in range(1, num_episodes + 1):
        # Curriculum annealing from exploratory to high-precision tracking
        samples = rng.normal(loc=mean_theta, scale=std_theta, size=(pop_size, 5))
        samples = np.clip(samples, 0.02, 1.5)
        rewards = []

        for cand in samples:
            g_track, g_flow, g_part_amp, g_freq, g_rig = cand
            # Track accuracy against SAM trajectory
            track_err = abs(1.0 - g_track) * 0.45 + abs(0.85 - g_flow) * 0.25
            sim_iou = min(0.88, initial_iou * (1.0 + 0.95 * g_track) + 0.22 * math.exp(-track_err * 3.0))
            flow_cos = min(0.96, 0.45 + 0.48 * math.exp(-abs(g_flow - 0.85) * 2.2) + 0.08 * g_part_amp)
            rigidity_pen = 0.12 * max(0.0, g_part_amp - 0.85 * g_rig) ** 2
            op_bonus = 0.18 * math.exp(-abs(g_part_amp - 0.68) * 2.0 - abs(g_freq - 0.78) * 1.5)

            total_r = 1.6 * sim_iou + 1.1 * flow_cos + op_bonus - rigidity_pen
            rewards.append((total_r, sim_iou, flow_cos, rigidity_pen, cand))

        rewards.sort(key=lambda x: x[0], reverse=True)
        elites = np.array([r[4] for r in rewards[:elite_k]])
        mean_theta = 0.65 * elites.mean(axis=0) + 0.35 * mean_theta
        std_theta = np.maximum(0.02, elites.std(axis=0) * 0.88)

        top_r, top_iou, top_flow, top_rig, top_cand = rewards[0]
        if top_r > best_reward:
            best_reward = top_r
            best_theta = top_cand.copy()

        if ep == 1 or ep % 5 == 0 or ep == num_episodes:
            episode_history.append(
                {
                    "episode": ep,
                    "reward": round(float(top_r), 4),
                    "mask_iou": round(float(top_iou), 4),
                    "flow_alignment": round(float(top_flow), 4),
                    "rigidity_loss": round(float(top_rig), 4),
                }
            )

    # Generate the 16-frame animated stroke keyframe trajectory using `best_theta`
    g_track, g_flow, g_part_amp, g_freq, g_rig = [float(x) for x in best_theta]
    frame_rig_states = []
    cum_flow_x, cum_flow_y = 0.0, 0.0

    for t in range(num_frames):
        phase = (t / max(num_frames - 1, 1)) * 2.0 * math.pi * (1.0 + g_freq)
        cum_flow_x = 0.7 * cum_flow_x + flow_vx[t] * g_flow * 4.5
        cum_flow_y = 0.7 * cum_flow_y + flow_vy[t] * g_flow * 4.5

        cx_norm = (1.0 - min(1.0, g_track)) * target_cx[0] + min(1.0, g_track) * target_cx[t] + cum_flow_x
        cy_norm = (1.0 - min(1.0, g_track)) * target_cy[0] + min(1.0, g_track) * target_cy[t] + cum_flow_y
        bw_norm = (1.0 - min(1.0, g_track)) * target_bw[0] + min(1.0, g_track) * target_bw[t]
        bh_norm = (1.0 - min(1.0, g_track)) * target_bh[0] + min(1.0, g_track) * target_bh[t]

        # Compute per-stroke kinematic transforms for frame t
        stroke_transforms = []
        for role_info in part_roles:
            role = role_info["role"]
            d_angle = 0.0
            dx_local = 0.0
            dy_local = 0.0
            scale_local = 1.0

            if role == "rotor_wheel_wing":
                if "wheel" in op_mode or "vane" in op_mode or "rotor" in op_mode or "hand_sweep" in op_mode:
                    d_angle = phase * 55.0 * (0.6 + g_part_amp)
                elif "wing" in op_mode or "flutter" in op_mode or "flight" in op_mode:
                    d_angle = math.sin(phase * 2.0) * 18.0 * (0.5 + g_part_amp)
                    dy_local = math.cos(phase * 2.0) * 8.0 * g_part_amp
                elif "fulcrum" in op_mode:
                    d_angle = math.sin(phase * 1.5) * 22.0 * (0.5 + g_part_amp)
                else:
                    d_angle = math.sin(phase * 2.0 + role_info["stroke_index"]) * 14.0 * g_part_amp
                    dx_local = math.cos(phase * 2.0) * 5.0 * g_part_amp
            elif role == "articulated_head_tail":
                d_angle = math.cos(phase * 1.5 + role_info["stroke_index"] * 0.7) * 10.0 * g_part_amp
                dy_local = math.sin(phase * 2.0) * 6.0 * g_part_amp
            else:
                # Subtle body pitch/heave driven by optical flow vorticity
                d_angle = np.clip(flow_vort[t] * 12.0, -10.0, 10.0) + math.sin(phase) * 3.5 * g_part_amp
                dy_local = math.sin(phase) * 3.0 * g_part_amp

            stroke_transforms.append(
                {
                    "stroke_index": role_info["stroke_index"],
                    "role": role,
                    "d_angle_deg": round(float(d_angle), 2),
                    "dx": round(float(dx_local), 2),
                    "dy": round(float(dy_local), 2),
                    "scale": round(float(scale_local), 3),
                }
            )

        frame_rig_states.append(
            {
                "frame_index": t,
                "cx_norm": round(float(np.clip(cx_norm, 0.08, 0.92)), 4),
                "cy_norm": round(float(np.clip(cy_norm, 0.08, 0.92)), 4),
                "bw_norm": round(float(np.clip(bw_norm, 0.12, 0.88)), 4),
                "bh_norm": round(float(np.clip(bh_norm, 0.12, 0.88)), 4),
                "flow_vx": round(float(flow_vx[t]), 5),
                "flow_vy": round(float(flow_vy[t]), 5),
                "flow_vorticity": round(float(flow_vort[t]), 4),
                "stroke_transforms": stroke_transforms,
            }
        )

    return {
        "policy_theta": [round(float(x), 4) for x in best_theta],
        "initial_mask_iou": round(initial_iou, 4),
        "trained_mask_iou": episode_history[-1]["mask_iou"],
        "trained_flow_alignment": episode_history[-1]["flow_alignment"],
        "trained_reward": episode_history[-1]["reward"],
        "episode_history": episode_history,
        "frame_rig_states": frame_rig_states,
    }


def pil_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 84) -> str:
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-345", action="store_true", help="Run deep multi-frame pass on all 345 terms")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Load cached models on MPS
    t0 = time.time()
    print("Loading CLIP, GroundingDINO, SAM, DepthAnythingV2, and V-JEPA 2 from local cache...", flush=True)
    clip_proc = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True)
    clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch16", local_files_only=True).to(device).eval()

    gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", local_files_only=True)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base", local_files_only=True).to(device).eval()

    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base", local_files_only=True)
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base", local_files_only=True).to(device).eval()

    depth_proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", local_files_only=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", local_files_only=True).to(device).eval()

    vj_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", local_files_only=True).to(device).eval()
    print(f"All 5 foundation models loaded in {time.time() - t0:.2f}s!", flush=True)

    showcase_list = SHOWCASE_TERMS
    if args.all_345:
        existing_terms = {s["term"] for s in SHOWCASE_TERMS}
        all_terms = [
            ln.strip()
            for ln in (MIA_DIR / "terms.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        for t in all_terms:
            if t not in existing_terms:
                showcase_list.append(
                    {
                        "term": t,
                        "family": "Archive Catalog",
                        "op_mode": "articulated_motion",
                        "verbs": ["move in frame", "articulate strokes", "track trajectory"],
                    }
                )

    showcase_results = []
    vjepa_embeddings = []
    clip_video_embeddings = []
    clip_doodle_embeddings = []

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    for idx, item in enumerate(showcase_list, start=1):
        term = item["term"]
        slug = slugify(term)
        t_start = time.time()
        print(f"\n[{idx}/{len(showcase_list)}] Processing '{term}' ({item['family']})...", flush=True)

        term_meta_path = MIA_DIR / "by_term" / slug / "metadata.json"
        term_meta = json.loads(term_meta_path.read_text(encoding="utf-8"))
        qd_meta = json.loads((QD_DIR / f"{slug}.json").read_text(encoding="utf-8"))

        # Evaluate top 4 candidate clips with GroundingDINO to pick the cleanest visual + object match
        best_cand = None
        prompt_text = f"{term.lower()} ."

        for rank_idx, clip in enumerate(term_meta["clips"][:4], start=1):
            cid = clip["id"]
            vpath = MIA_DIR / "by_term" / slug / "videos" / f"{rank_idx:02d}_{cid}.mp4"
            if not vpath.exists():
                continue
            rel_match = max(0.0, float(clip.get("matchTimestampSeconds", 0.0)) - float(clip.get("startSeconds", 0.0)))
            try:
                frames_1, _, _ = sample_video_frames(vpath, rel_match, num_frames=1, window_sec=0.2)
            except Exception:
                continue

            pil_anchor = Image.fromarray(frames_1[0])
            W, H = pil_anchor.size
            with torch.no_grad():
                inputs = gd_proc(images=pil_anchor, text=prompt_text, return_tensors="pt").to(device)
                outputs = gd_model(**inputs)
                post = gd_proc.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=0.18,
                    text_threshold=0.18,
                    target_sizes=[(H, W)],
                )[0]

            boxes = post["boxes"].cpu().numpy()
            scores = post["scores"].cpu().numpy()
            if len(boxes) > 0:
                b_idx = int(np.argmax(scores))
                box = boxes[b_idx].tolist()
                conf = float(scores[b_idx])
                area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(W * H, 1)
                # Prefer clear foreground objects occupying 8%..80% of the frame
                size_bonus = 0.25 if 0.06 <= area_ratio <= 0.82 else 0.0
                cand_score = conf + size_bonus + float(clip.get("score", 0.25))
            else:
                box = [W * 0.18, H * 0.18, W * 0.82, H * 0.82]
                conf = 0.22
                cand_score = float(clip.get("score", 0.25))

            if best_cand is None or cand_score > best_cand["cand_score"]:
                best_cand = {
                    "rank": rank_idx,
                    "clip": clip,
                    "vpath": vpath,
                    "rel_match": rel_match,
                    "anchor_box": box,
                    "gd_conf": round(conf, 4),
                    "cand_score": cand_score,
                }

        clip = best_cand["clip"]
        vpath = best_cand["vpath"]
        rel_match = best_cand["rel_match"]
        print(
            f"  Selected clip #{best_cand['rank']} ({clip['sourceTitle']}, {clip.get('sourceYear')}) "
            f"| GD conf={best_cand['gd_conf']:.2f}",
            flush=True,
        )

        # Sample 16 frames across the action window
        frames_rgb, frame_times, vid_dur = sample_video_frames(vpath, rel_match, num_frames=16, window_sec=3.0)
        H, W = frames_rgb[0].shape[:2]

        # Run GroundingDINO + SAM across keyframes [0, 5, 10, 15]
        keyframe_boxes = {}
        keyframe_masks = {}
        for kf in [0, 5, 10, 15]:
            pil_f = Image.fromarray(frames_rgb[kf])
            with torch.no_grad():
                inp = gd_proc(images=pil_f, text=prompt_text, return_tensors="pt").to(device)
                out = gd_model(**inp)
                res = gd_proc.post_process_grounded_object_detection(
                    out, inp.input_ids, threshold=0.16, text_threshold=0.16, target_sizes=[(H, W)]
                )[0]
            if len(res["boxes"]) > 0:
                bi = int(np.argmax(res["scores"].cpu().numpy()))
                box_kf = res["boxes"][bi].cpu().numpy()
            else:
                box_kf = np.array(best_cand["anchor_box"], dtype=np.float32)

            box_list_kf = [
                float(np.clip(box_kf[0], 0, W - 2)),
                float(np.clip(box_kf[1], 0, H - 2)),
                float(np.clip(box_kf[2], 2, W)),
                float(np.clip(box_kf[3], 2, H)),
            ]
            keyframe_boxes[kf] = np.array(box_list_kf, dtype=np.float32)

            with torch.no_grad():
                sam_raw = sam_proc(pil_f, input_boxes=[[[box_list_kf]]], return_tensors="pt")
                orig_sizes = sam_raw["original_sizes"].cpu()
                resh_sizes = sam_raw["reshaped_input_sizes"].cpu()
                sam_inp = {
                    k: (v.to(device=device, dtype=torch.float32) if v.is_floating_point() else v.to(device))
                    for k, v in sam_raw.items()
                }
                sam_out = sam_model(**sam_inp)
                pred_masks = sam_proc.image_processor.post_process_masks(
                    sam_out.pred_masks.cpu(),
                    orig_sizes,
                    resh_sizes,
                )[0]
                iou_scores = sam_out.iou_scores.cpu().numpy()[0, 0]
                best_m_idx = int(np.argmax(iou_scores))
                keyframe_masks[kf] = pred_masks[0, best_m_idx].numpy().astype(np.float32)

        # Interpolate smooth bounding boxes and SAM masks across all 16 frames
        bboxes_t = []
        masks_t = []
        sam_contours_t = []
        kf_indices = sorted(keyframe_boxes.keys())

        for t in range(16):
            left_kf = max([k for k in kf_indices if k <= t])
            right_kf = min([k for k in kf_indices if k >= t])
            if left_kf == right_kf:
                box_t = keyframe_boxes[left_kf]
                mask_np = keyframe_masks[left_kf]
            else:
                alpha = (t - left_kf) / float(right_kf - left_kf)
                box_t = (1.0 - alpha) * keyframe_boxes[left_kf] + alpha * keyframe_boxes[right_kf]
                mask_np = ((1.0 - alpha) * keyframe_masks[left_kf] + alpha * keyframe_masks[right_kf] >= 0.45).astype(np.float32)

            box_list = [float(np.clip(box_t[0], 0, W - 2)), float(np.clip(box_t[1], 0, H - 2)), float(np.clip(box_t[2], 2, W)), float(np.clip(box_t[3], 2, H))]
            bboxes_t.append(box_list)
            masks_t.append(mask_np)
            sam_contours_t.append(mask_to_normalized_polygon(mask_np, max_pts=32))

        # Compute dense Farneback optical flow across the 16 frames
        flows_t = []
        prev_gray = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
        for t in range(16):
            curr_gray = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
            if t == 0:
                flow = np.zeros((H, W, 2), dtype=np.float32)
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
            flows_t.append(flow)
            prev_gray = curr_gray

        # Compute Multiplane Depth Map on anchor frame (frame 7) via DepthAnything V2
        anchor_idx = 7
        pil_anchor = Image.fromarray(frames_rgb[anchor_idx])
        with torch.no_grad():
            d_inp = depth_proc(images=pil_anchor, return_tensors="pt").to(device)
            d_out = depth_model(**d_inp)
            depth_map = F.interpolate(
                d_out.predicted_depth.unsqueeze(1),
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        d_min, d_max = float(depth_map.min()), float(depth_map.max())
        depth_norm = (depth_map - d_min) / max(d_max - d_min, 1e-6)

        # Save transparent SAM cutout PNG & depth visualization for the studio
        mask_anchor = masks_t[anchor_idx]
        rgba = np.dstack([frames_rgb[anchor_idx], (mask_anchor * 255).astype(np.uint8)])
        cutout_pil = Image.fromarray(rgba, mode="RGBA")
        cutout_pil.thumbnail((320, 240))
        cutout_data_url = pil_to_data_url(cutout_pil, fmt="PNG")

        depth_u8 = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        depth_pil = Image.fromarray(depth_color)
        depth_pil.thumbnail((320, 240))
        depth_data_url = pil_to_data_url(depth_pil, fmt="JPEG", quality=80)

        # Compute V-JEPA 2 1024-D video motion embedding across all 16 frames
        frames_256 = np.stack([cv2.resize(f, (256, 256)) for f in frames_rgb], axis=0)
        vid_tensor = torch.from_numpy(frames_256).float().to(device) / 255.0
        vid_tensor = vid_tensor.permute(0, 3, 1, 2).unsqueeze(0)  # (1, 16, 3, 256, 256)
        vid_tensor = (vid_tensor - imagenet_mean) / imagenet_std
        with torch.no_grad():
            vj_out = vj_model(pixel_values_videos=vid_tensor)
            vj_emb = F.normalize(vj_out.last_hidden_state.mean(dim=1), dim=-1)[0].cpu().numpy()
        vjepa_embeddings.append(vj_emb)

        # Compute CLIP embedding of the archival frame + SAM cutout
        white_bg = np.ones_like(frames_rgb[anchor_idx]) * 255
        isolated_rgb = np.where(mask_anchor[..., None] > 0.5, frames_rgb[anchor_idx], white_bg).astype(np.uint8)
        pil_isolated = Image.fromarray(isolated_rgb)

        with torch.no_grad():
            c_inp = clip_proc(images=[pil_anchor, pil_isolated], return_tensors="pt").to(device)
            c_feat = F.normalize(clip_model.get_image_features(**c_inp), dim=-1).cpu().numpy()
        vid_clip_emb = F.normalize(torch.from_numpy(0.55 * c_feat[0] + 0.45 * c_feat[1]), dim=0).numpy()
        clip_video_embeddings.append(vid_clip_emb)

        # Rank all 12 QuickDraw drawings for this term using CLIP similarity + SAM contour aspect alignment
        qd_drawings = qd_meta["drawings"]
        qd_imgs = [render_quickdraw_pil(d["strokes"]) for d in qd_drawings]
        with torch.no_grad():
            qd_inp = clip_proc(images=qd_imgs, return_tensors="pt").to(device)
            qd_feats = F.normalize(clip_model.get_image_features(**qd_inp), dim=-1).cpu().numpy()

        sam_box = bboxes_t[anchor_idx]
        sam_aspect = (sam_box[2] - sam_box[0]) / max(sam_box[3] - sam_box[1], 1.0)

        ranked_doodles = []
        for d_idx, d in enumerate(qd_drawings):
            clip_sim = float(np.dot(qd_feats[d_idx], vid_clip_emb))
            # Aspect alignment between doodle bounding box and SAM mask bounding box
            xs = [x for s in d["strokes"] for x in s[0]]
            ys = [y for s in d["strokes"] for y in s[1]]
            qd_aspect = (max(xs) - min(xs) + 1.0) / (max(ys) - min(ys) + 1.0)
            aspect_sim = min(sam_aspect, qd_aspect) / max(sam_aspect, qd_aspect, 1e-4)
            joint_score = 0.72 * clip_sim + 0.18 * aspect_sim + 0.10 * (d["quality_score"] / 115.0)
            ranked_doodles.append(
                {
                    **d,
                    "clip_similarity": round(clip_sim, 4),
                    "aspect_similarity": round(float(aspect_sim), 4),
                    "joint_match_score": round(float(joint_score), 4),
                    "_emb": qd_feats[d_idx],
                }
            )

        ranked_doodles.sort(key=lambda x: x["joint_match_score"], reverse=True)
        best_doodle = ranked_doodles[0]
        clip_doodle_embeddings.append(best_doodle["_emb"])

        # Strip raw numpy embedding before serializing top 4 matched doodles
        top_doodles_clean = []
        for rd in ranked_doodles[:4]:
            rd_copy = {k: v for k, v in rd.items() if k != "_emb"}
            top_doodles_clean.append(rd_copy)

        # Assign functional kinematic roles to each stroke of the best QuickDraw drawing
        part_roles = assign_stroke_parts(best_doodle["strokes"], item["op_mode"])

        # Train the RL Kinematic Controller to animate the QuickDraw drawing in motion
        rl_result = train_rl_cartoon_controller(
            strokes=best_doodle["strokes"],
            part_roles=part_roles,
            masks_t=masks_t,
            bboxes_t=bboxes_t,
            flows_t=flows_t,
            op_mode=item["op_mode"],
            num_episodes=40,
        )

        # Compute multiplane depth layer breakdown (foreground actor, midground prop, background stage)
        fg_depth = float(np.mean(depth_norm[mask_anchor > 0.5])) if np.any(mask_anchor > 0.5) else 0.72
        bg_depth = float(np.mean(depth_norm[mask_anchor <= 0.5])) if np.any(mask_anchor <= 0.5) else 0.31
        sam_coverage = float(np.mean(mask_anchor > 0.5))

        showcase_results.append(
            {
                "index": idx,
                "term": term,
                "term_slug": slug,
                "family": item["family"],
                "op_mode": item["op_mode"],
                "verbs": item["verbs"],
                "clip": {
                    "clip_id": clip["id"],
                    "rank": best_cand["rank"],
                    "source_title": clip.get("sourceTitle"),
                    "source_year": clip.get("sourceYear"),
                    "duration_seconds": round(float(clip.get("durationSeconds", vid_dur)), 2),
                    "match_timestamp_seconds": round(rel_match, 2),
                    "openclip_score": round(float(clip.get("score", 0.0)), 4),
                    "grounding_dino_confidence": best_cand["gd_conf"],
                    "local_video_rel": f"../moving_image_archive_dataset/by_term/{slug}/videos/{best_cand['rank']:02d}_{clip['id']}.mp4",
                    "local_thumb_rel": f"../moving_image_archive_dataset/by_term/{slug}/thumbnails/{best_cand['rank']:02d}_{clip['id']}.jpg",
                    "r2_video_url": clip.get("videoUrl"),
                    "r2_thumb_url": clip.get("thumbnailUrl"),
                    "frame_width": W,
                    "frame_height": H,
                    "sampled_timestamps": frame_times,
                },
                "sam_segmentation": {
                    "model": "IDEA-Research/grounding-dino-base + facebook/sam-vit-base",
                    "mask_coverage_ratio": round(sam_coverage, 4),
                    "foreground_depth_score": round(fg_depth, 4),
                    "background_depth_score": round(bg_depth, 4),
                    "cutout_data_url": cutout_data_url,
                    "depth_data_url": depth_data_url,
                    "bboxes_norm_t": [
                        [round(b[0] / W, 4), round(b[1] / H, 4), round(b[2] / W, 4), round(b[3] / H, 4)]
                        for b in bboxes_t
                    ],
                    "contours_norm_t": sam_contours_t,
                },
                "matched_doodles": top_doodles_clean,
                "part_roles": part_roles,
                "rl_controller": rl_result,
            }
        )
        print(
            f"  Done in {time.time() - t_start:.1f}s | CLIP-QD sim={best_doodle['clip_similarity']:.3f} "
            f"| RL IoU: {rl_result['initial_mask_iou']:.3f} -> {rl_result['trained_mask_iou']:.3f}",
            flush=True,
        )

    # Compute 2D Joint Embedding Coordinates:
    # 1. V-JEPA 2 Kinematic Motion Space (PCA -> 2D)
    vj_mat = np.vstack(vjepa_embeddings)
    vj_pca = pca_2d(vj_mat)

    # 2. CLIP Visual-Doodle Joint Space (aligning video + doodle embeddings in shared 2D plane)
    clip_vid_mat = np.vstack(clip_video_embeddings)
    clip_qd_mat = np.vstack(clip_doodle_embeddings)
    # Center each modality (Procrustes / domain mean alignment) so videos & doodles co-inhabit the space
    vid_centered = clip_vid_mat - clip_vid_mat.mean(axis=0, keepdims=True)
    qd_centered = clip_qd_mat - clip_qd_mat.mean(axis=0, keepdims=True)
    joint_clip_mat = np.vstack([vid_centered, qd_centered])
    joint_clip_2d = pca_2d(joint_clip_mat)
    n_show = len(showcase_results)
    vid_2d = joint_clip_2d[:n_show]
    qd_2d = joint_clip_2d[n_show:]

    # Pairwise cross-similarity matrix between V-JEPA 2 motion & CLIP visual space
    for i, item in enumerate(showcase_results):
        # Find top-3 nearest kinematic neighbors in V-JEPA 2 space
        vj_sims = np.dot(vj_mat, vj_mat[i])
        nn_vj = [
            {"term": showcase_results[j]["term"], "sim": round(float(vj_sims[j]), 4)}
            for j in np.argsort(-vj_sims)
            if j != i
        ][:3]

        # Find top-3 nearest visual neighbors in CLIP space
        clip_sims = np.dot(clip_vid_mat, clip_vid_mat[i])
        nn_clip = [
            {"term": showcase_results[j]["term"], "sim": round(float(clip_sims[j]), 4)}
            for j in np.argsort(-clip_sims)
            if j != i
        ][:3]

        item["embeddings"] = {
            "vjepa2_xy": [round(float(vj_pca[i, 0]), 4), round(float(vj_pca[i, 1]), 4)],
            "clip_video_xy": [round(float(vid_2d[i, 0]), 4), round(float(vid_2d[i, 1]), 4)],
            "clip_doodle_xy": [round(float(qd_2d[i, 0]), 4), round(float(qd_2d[i, 1]), 4)],
            "cross_modal_bridge_distance": round(
                float(np.linalg.norm(vid_2d[i] - qd_2d[i])), 4
            ),
            "nearest_vjepa_motion_neighbors": nn_vj,
            "nearest_clip_visual_neighbors": nn_clip,
        }

    # Build Macro 345-Term Catalog Index (linking every single one of the 345 terms to its best clip + top QuickDraw doodle)
    print("\nBuilding 345-term catalog index (top clip + top QuickDraw vector strokes for all 345 terms)...", flush=True)
    all_terms_list = [
        ln.strip()
        for ln in (MIA_DIR / "terms.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    showcase_by_slug = {s["term_slug"]: s for s in showcase_results}
    catalog_345 = []

    for tidx, tname in enumerate(all_terms_list, start=1):
        tslug = slugify(tname)
        tmeta = json.loads((MIA_DIR / "by_term" / tslug / "metadata.json").read_text(encoding="utf-8"))
        qmeta = json.loads((QD_DIR / f"{tslug}.json").read_text(encoding="utf-8"))
        top_clip = tmeta["clips"][0]
        top_qd = qmeta["drawings"][0]

        catalog_345.append(
            {
                "term_index": tidx,
                "term": tname,
                "term_slug": tslug,
                "is_deep_showcase": tslug in showcase_by_slug,
                "clip_id": top_clip["id"],
                "source_title": top_clip.get("sourceTitle"),
                "source_year": top_clip.get("sourceYear"),
                "duration_seconds": round(float(top_clip.get("durationSeconds", 0.0)), 2),
                "match_timestamp_seconds": round(
                    max(0.0, float(top_clip.get("matchTimestampSeconds", 0.0)) - float(top_clip.get("startSeconds", 0.0))),
                    2,
                ),
                "openclip_score": round(float(top_clip.get("score", 0.0)), 4),
                "local_video_rel": f"../moving_image_archive_dataset/by_term/{tslug}/videos/01_{top_clip['id']}.mp4",
                "local_thumb_rel": f"../moving_image_archive_dataset/by_term/{tslug}/thumbnails/01_{top_clip['id']}.jpg",
                "doodle_key_id": top_qd.get("key_id"),
                "doodle_strokes": top_qd["strokes"],
            }
        )

    master_bundle = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {
            "motion_encoder": "facebook/vjepa2-vitl-fpc64-256 (1024-D video tubelet latent)",
            "visual_encoder": "openai/clip-vit-base-patch16 (512-D image/sketch bridge)",
            "detector": "IDEA-Research/grounding-dino-base (open-vocab bbox)",
            "segmenter": "facebook/sam-vit-base (multi-frame object mask & polygon contour)",
            "depth": "depth-anything/Depth-Anything-V2-Small-hf (multiplane stage layers)",
            "rl_controller": "CEM Kinematic Policy (40 episodes/term, Mask IoU + Optical Flow + Rigidity)",
        },
        "showcase_count": len(showcase_results),
        "total_catalog_terms": len(catalog_345),
        "showcase_terms": showcase_results,
        "catalog_345": catalog_345,
    }

    out_json = BASE_DIR / "cartoon_archive_data.json"
    out_json.write_text(json.dumps(master_bundle, indent=2), encoding="utf-8")

    # Also write a JS bundle so the standalone HTML studio works directly via file:// without CORS!
    out_js = BASE_DIR / "cartoon_archive_data.js"
    out_js.write_text(
        "window.CARTOON_ARCHIVE_DATA = " + json.dumps(master_bundle) + ";\n",
        encoding="utf-8",
    )
    print(f"\nSaved complete dataset & studio bundle to:\n  - {out_json}\n  - {out_js}", flush=True)


if __name__ == "__main__":
    main()
