#!/usr/bin/env python3
"""
Renders a side-by-side showcase video (`Cartoon_Archive_Living_Doodles_Showcase.mp4`)
combining:
  - Left Pane: Moving Image Archive clip + SAM segmentation mask + Jordan Mechner 12-frame
    rotoscoped ridges & contour-warped QuickDraw strokes.
  - Right Pane: HALFWORLD × Scribbling Speech Multiplane Paper-and-Ink Stage (#f2f1ec paper,
    #141414 ink, #e01b1b signal red) with SkyObject / AirObject / FloorObject depth layers,
    halftone dot matrix, Mechner rotoscope ridges, and 12-frame exposure sheet indicator.
"""

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

PAPER_BGR = (236, 241, 242)   # #f2f1ec in BGR
PAPER2_BGR = (220, 228, 230)  # #e6e4dc in BGR
INK_BGR = (20, 20, 20)        # #141414 in BGR
RED_BGR = (27, 27, 224)       # #e01b1b in BGR
MUTED_BGR = (120, 124, 126)


def draw_strokes_norm_cv2(
    canvas: np.ndarray,
    strokes: list,
    box_px: tuple,
    color_bgr: tuple,
    thickness: int = 2,
    boil_amp: float = 0.8,
    time_sec: float = 0.0,
    paper_halo: bool = False,
):
    x1, y1, bw, bh = box_px
    for s_idx, stroke in enumerate(strokes):
        if not stroke or len(stroke) < 2:
            continue
        xs, ys = stroke[0], stroke[1]
        if len(xs) < 2:
            continue
        pts = []
        for i in range(len(xs)):
            bx = math.sin(time_sec * 13.0 + s_idx * 5.1 + i * 1.9) * boil_amp
            by = math.cos(time_sec * 12.0 + s_idx * 4.3 + i * 2.1) * boil_amp
            px = x1 + (xs[i] / 255.0) * bw + bx
            py = y1 + (ys[i] / 255.0) * bh + by
            pts.append([int(round(px)), int(round(py))])
        pts_np = np.array(pts, dtype=np.int32)
        if paper_halo:
            cv2.polylines(canvas, [pts_np], False, PAPER_BGR, thickness=thickness + 3, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts_np], False, color_bgr, thickness=thickness, lineType=cv2.LINE_AA)


def main():
    data = json.loads((BASE_DIR / "cartoon_archive_data.json").read_text(encoding="utf-8"))
    showcase = data["showcase_terms"][:12]  # Top 12 showcase clips for a crisp 24s 12fps reel

    W, H = 1280, 560
    half_W = W // 2
    fps = 12
    out_raw = BASE_DIR / "showcase_raw.mp4"
    out_final = BASE_DIR / "Cartoon_Archive_Living_Doodles_Showcase.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_raw), fourcc, fps, (W, H))

    for item in showcase:
        clip_info = item["clip"]
        vpath = (BASE_DIR / clip_info["local_video_rel"]).resolve()
        cap = cv2.VideoCapture(str(vpath))
        v_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

        doodle = item["matched_doodles"][0]
        roto_frames = item.get("mechner_rotoscope", {}).get("frames", [])
        speech_layers = item.get("scribbling_speech_layers", [])
        num_roto = len(roto_frames) if roto_frames else 12

        # Play 2 full 12-frame Jordan Mechner rotoscope cycles (24 frames = 2.0s per scene)
        for loop_idx in range(24):
            r_idx = loop_idx % num_roto
            rf = roto_frames[r_idx] if r_idx < len(roto_frames) else None
            t_sec = rf["timestamp"] if rf else 1.0
            f_num = int(min(max(round(t_sec * v_fps), 0), max(total_f - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                bgr = np.zeros((H, half_W, 3), dtype=np.uint8)
            else:
                bgr = cv2.resize(bgr, (half_W, H))

            frame = np.zeros((H, W, 3), dtype=np.uint8)
            left = bgr.copy()

            # 1. Left Pane: Archival Reference + Mechner Roto Ridges + Contour-Warped QuickDraw
            if rf:
                bw_px = rf["bw_norm"] * half_W
                bh_px = rf["bh_norm"] * H
                x1 = rf["cx_norm"] * half_W - bw_px * 0.5
                y1 = rf["cy_norm"] * H - bh_px * 0.5

                draw_strokes_norm_cv2(
                    left,
                    rf.get("roto_strokes", []),
                    (x1, y1, bw_px, bh_px),
                    RED_BGR,
                    thickness=2,
                    boil_amp=0.35,
                    time_sec=loop_idx / fps,
                )
                draw_strokes_norm_cv2(
                    left,
                    rf.get("warped_qd_strokes", doodle["strokes"]),
                    (x1, y1, bw_px, bh_px),
                    PAPER_BGR,
                    thickness=2,
                    boil_amp=0.6,
                    time_sec=loop_idx / fps,
                )

            # 2. Right Pane: HALFWORLD Paper-and-Ink Multiplane World
            right = np.full((H, half_W, 3), PAPER_BGR, dtype=np.uint8)
            horizon_y = int(H * 0.67)
            cv2.rectangle(right, (0, horizon_y), (half_W, H), PAPER2_BGR, -1)
            cv2.line(right, (0, horizon_y), (half_W, horizon_y), INK_BGR, 2)

            # Perspective ground grid lines
            for gx in range(-6, 7):
                x_top = int(half_W * 0.5 + gx * 26)
                x_bot = int(half_W * 0.5 + gx * 95)
                cv2.line(right, (x_top, horizon_y), (x_bot, H - 58), (195, 202, 204), 1, lineType=cv2.LINE_AA)

            # Draw Scribbling Speech SkyObject / AirObject / FloorObject layers
            for l_idx, layer in enumerate(speech_layers):
                stratum = layer.get("stratum", "FloorObject")
                lx = ((l_idx * 173 + 60) % (half_W - 140)) + 30
                if stratum == "SkyObject":
                    ly = 56 + (l_idx * 29) % 68
                    l_col = (115, 115, 115)
                    l_scale = 76
                elif stratum == "AirObject":
                    ly = 135 + int(math.sin(loop_idx * 0.35 + l_idx) * 14)
                    l_col = RED_BGR
                    l_scale = 82
                else:
                    ly = horizon_y - 25 + (l_idx * 23) % 55
                    l_col = INK_BGR
                    l_scale = 92

                draw_strokes_norm_cv2(
                    right,
                    layer.get("strokes", []),
                    (lx, ly, l_scale, l_scale),
                    l_col,
                    thickness=2,
                    boil_amp=0.7,
                    time_sec=loop_idx / fps,
                )
                cv2.putText(
                    right,
                    f"{layer['term'].upper()} [{stratum}]",
                    (lx, ly - 4),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.82,
                    MUTED_BGR,
                    1,
                    cv2.LINE_AA,
                )

            # Draw Primary Actor with Halftone Dot Grid + Mechner Rotoscope + Warped QuickDraw
            if rf:
                bw_px = max(rf["bw_norm"] * half_W * 1.08, 130)
                bh_px = max(rf["bh_norm"] * H * 1.08, 130)
                x1 = rf["cx_norm"] * half_W - bw_px * 0.5
                y1 = rf["cy_norm"] * H - bh_px * 0.5

                # Halftone 24x24 dot grid inside silhouette
                dot_grid = rf.get("halftone_24", [])
                for pt in dot_grid:
                    gx, gy, darkness = pt[0], pt[1], pt[2]
                    px = int(x1 + ((gx + 0.5) / 24.0) * bw_px)
                    py = int(y1 + ((gy + 0.5) / 24.0) * bh_px)
                    rad = max(1, int(darkness * (bw_px / 24.0) * 0.44))
                    cv2.circle(right, (px, py), rad, (165, 170, 172), -1, lineType=cv2.LINE_AA)

                # Mechner high-contrast rotoscope ridges in signal red
                draw_strokes_norm_cv2(
                    right,
                    rf.get("roto_strokes", []),
                    (x1, y1, bw_px, bh_px),
                    RED_BGR,
                    thickness=2,
                    boil_amp=0.35,
                    time_sec=loop_idx / fps,
                )

                # Contour-warped QuickDraw strokes in deep ink with paper halo
                draw_strokes_norm_cv2(
                    right,
                    rf.get("warped_qd_strokes", doodle["strokes"]),
                    (x1, y1, bw_px, bh_px),
                    INK_BGR,
                    thickness=3,
                    boil_amp=0.9,
                    time_sec=loop_idx / fps,
                    paper_halo=True,
                )

            # Bottom 12-frame Mechner Rotoscope Exposure Sheet strip on Right Pane
            strip_y1 = H - 56
            cv2.rectangle(right, (0, strip_y1), (half_W, H - 32), PAPER2_BGR, -1)
            cv2.line(right, (0, strip_y1), (half_W, strip_y1), INK_BGR, 1)
            cell_w = half_W // 12
            for c_i in range(12):
                cx1 = c_i * cell_w
                cx2 = cx1 + cell_w - 2
                is_cur = c_i == r_idx
                cv2.rectangle(
                    right,
                    (cx1 + 2, strip_y1 + 3),
                    (cx2, H - 35),
                    RED_BGR if is_cur else INK_BGR,
                    2 if is_cur else 1,
                )
                cv2.putText(
                    right,
                    f"F{c_i+1:02d}",
                    (cx1 + 8, strip_y1 + 17),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.85,
                    RED_BGR if is_cur else INK_BGR,
                    1,
                    cv2.LINE_AA,
                )

            frame[:, :half_W] = left
            frame[:, half_W:] = right
            cv2.line(frame, (half_W, 0), (half_W, H), INK_BGR, 3)

            # Top architectural header bar
            cv2.rectangle(frame, (0, 0), (W, 38), INK_BGR, -1)
            stratum_lbl = item.get("stratum", "FloorObject")
            title_txt = (
                f"{item['term'].upper()} [{stratum_lbl} / {item['family']}]   |   "
                f"LEFT: ARCHIVE + SAM + MECHNER ROTO   |   RIGHT: HALFWORLD x SCRIBBLING SPEECH STAGE"
            )
            cv2.putText(frame, title_txt, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.54, PAPER_BGR, 1, cv2.LINE_AA)

            # Bottom metadata bar
            cv2.rectangle(frame, (0, H - 32), (W, H), INK_BGR, -1)
            meta_txt = (
                f"Archive: {clip_info['source_title']} ({clip_info['source_year']})  |  "
                f"Mechner Roto Cycle: Frame {r_idx+1:02d}/12  |  "
                f"RL IoU: {item['rl_controller']['initial_mask_iou']:.2f} -> {item['rl_controller']['trained_mask_iou']:.2f}"
            )
            cv2.putText(frame, meta_txt, (14, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, PAPER_BGR, 1, cv2.LINE_AA)

            writer.write(frame)
        cap.release()

    writer.release()
    subprocess.run(
        [
            "/Users/gaia/anaconda3/bin/ffmpeg",
            "-y",
            "-i",
            str(out_raw),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(out_final),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    out_raw.unlink(missing_ok=True)
    print(f"Rendered upgraded showcase video: {out_final} ({out_final.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
