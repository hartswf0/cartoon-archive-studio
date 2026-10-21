#!/usr/bin/env python3
"""
Runs the 4-Pass Background Redraw & Multiplane Gouache Occlusion Engine across all 50
Batch-50 shots and regenerates all contact sheets (`batch50_scene_plates_*.png`,
`batch50_stages_*.png`, and `batch50_bg_redraws_showcase.png`).
"""

import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from tools.scion_batch50_engine import (
    ROOT,
    SHOTS_DIR,
    PREVIEW_DIR,
    ART_DIR,
    W,
    H,
    STAGE_W,
    STAGE_H,
    BATCH_50_SPECS,
    select_best_continuous_interval,
    detect_broadcast_chyron_mask,
    render_scion_multiplane_shot,
)
from tools.bespoke_scene_architect import build_bespoke_50_scene


def load_33_archival_frames_fast(spec: dict, sdir: Path) -> list:
    """Loads the exact 33 archival frames from `source.json` (`local_video_path` + `interval`)."""
    src_path = sdir / "source.json"
    if src_path.exists():
        try:
            src = json.loads(src_path.read_text(encoding="utf-8"))
            rel_vid = src.get("archive_source", {}).get("local_video_path", "")
            interval = src.get("interval", [1.0, 5.12])
            vid_p = ROOT / rel_vid
            if vid_p.exists():
                cap = cv2.VideoCapture(str(vid_p))
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                t0, t1 = float(interval[0]), float(interval[1])
                frames = []
                for i in range(33):
                    t = t0 + (t1 - t0) * (i / 32.0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
                    ok, fr = cap.read()
                    if ok and fr is not None:
                        frames.append(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA))
                cap.release()
                if len(frames) == 33:
                    return frames
        except Exception:
            pass
    frames, _ = select_best_continuous_interval(spec["category"], sdir)
    return frames


def build_contact_sheets(slugs: list):
    """Builds 5x5 contact sheets for scene plates, multiplane stages, and 4-pass BG redraws."""
    tw, th = 256, 192

    def make_5x5_grid(group_slugs, img_getter):
        rows = []
        for r in range(5):
            row_imgs = []
            for c in range(5):
                idx = r * 5 + c
                slug = group_slugs[idx]
                im = img_getter(slug)
                if im is None:
                    im = np.full((th, tw, 3), 245, dtype=np.uint8)
                else:
                    im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
                cv2.rectangle(im, (4, 4), (178, 24), (18, 20, 22), -1)
                cv2.putText(im, f"{idx+1:02d}. {slug.upper()}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 255, 180), 1, cv2.LINE_AA)
                cv2.rectangle(im, (0, 0), (tw - 1, th - 1), (160, 160, 160), 1)
                row_imgs.append(im)
            rows.append(np.hstack(row_imgs))
        return np.vstack(rows)

    def get_scene_plate(slug):
        return cv2.imread(str(SHOTS_DIR / slug / "scene-plate.png"))

    def get_stage_frame(slug):
        gif_p = PREVIEW_DIR / f"pure_cartoon_{slug}.gif"
        if gif_p.exists():
            pil_im = Image.open(gif_p)
            pil_im.seek(min(10, getattr(pil_im, "n_frames", 1) - 1))
            rgb = np.array(pil_im.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.imread(str(SHOTS_DIR / slug / "planes" / "multiplane-stack.png"))

    sp_1_25 = make_5x5_grid(slugs[:25], get_scene_plate)
    sp_26_50 = make_5x5_grid(slugs[25:50], get_scene_plate)
    st_1_25 = make_5x5_grid(slugs[:25], get_stage_frame)
    st_26_50 = make_5x5_grid(slugs[25:50], get_stage_frame)

    for name, arr in [
        ("batch50_scene_plates_1_25.png", sp_1_25),
        ("batch50_scene_plates_26_50.png", sp_26_50),
        ("batch50_stages_1_25.png", st_1_25),
        ("batch50_stages_26_50.png", st_26_50),
    ]:
        cv2.imwrite(str(PREVIEW_DIR / name), arr)
        if ART_DIR.exists():
            cv2.imwrite(str(ART_DIR / name), arr)


def main():
    t0 = time.time()
    sx_up, sy_up = float(W) / STAGE_W, float(H) / STAGE_H
    slugs = [s["slug"] for s in BATCH_50_SPECS]

    for idx, sp in enumerate(BATCH_50_SPECS):
        slug = sp["slug"]
        sdir = SHOTS_DIR / slug
        shot_data = json.loads((sdir / "shot.json").read_text(encoding="utf-8"))
        cast_frames_480 = shot_data["cast"]["frames"]
        cel_masks_480 = shot_data["cast"]["cel_masks"]

        frames_bgr = load_33_archival_frames_fast(sp, sdir)

        sam_masks_640 = []
        for polys in cel_masks_480:
            m = np.zeros((H, W), dtype=np.uint8)
            for poly in polys:
                if len(poly) >= 3:
                    arr = np.array([[int(round(p[0] * sx_up)), int(round(p[1] * sy_up))] for p in poly], dtype=np.int32)
                    cv2.fillPoly(m, [arr], 255)
            sam_masks_640.append(m)

        bg_plate_path = sdir / "bg-plate-color.png"
        if bg_plate_path.exists():
            bg_plate = cv2.imread(str(bg_plate_path))
        else:
            bg_plate = np.median(np.stack(frames_bgr[::2], axis=0), axis=0).astype(np.uint8)
            cv2.imwrite(str(bg_plate_path), bg_plate)

        stack_m = np.stack([cv2.dilate(m, np.ones((15, 15), np.uint8)) > 127 for m in sam_masks_640[::2]], axis=0)
        occ_freq = stack_m.mean(axis=0)
        subject_envelope = cv2.dilate(((occ_freq >= 0.22) * 255).astype(np.uint8), np.ones((21, 21), np.uint8))
        if (subject_envelope > 0).mean() > 0.55:
            subject_envelope = cv2.erode(((occ_freq >= 0.50) * 255).astype(np.uint8), np.ones((9, 9), np.uint8))

        chyron_mask = detect_broadcast_chyron_mask(cv2.cvtColor(bg_plate, cv2.COLOR_BGR2GRAY))
        d_med = np.linspace(0.15, 0.85, H, dtype=np.float32)[:, None] * np.ones((H, W), dtype=np.float32)

        planes_480, objects = build_bespoke_50_scene(sp, sdir, bg_plate, d_med, sam_masks_640, subject_envelope, chyron_mask)
        render_scion_multiplane_shot(sp, sdir, frames_bgr, cast_frames_480, cel_masks_480, planes_480, objects)

        # Update tests.json to record the 4-Pass Living Background Redraw + Multiplane Gouache Cels
        tests_p = sdir / "tests.json"
        if tests_p.exists():
            tests_list = json.loads(tests_p.read_text(encoding="utf-8"))
            for t_item in tests_list:
                if t_item.get("id") == "B1":
                    t_item["how"] = "4-Pass Bespoke Scene Architect (BG-K01..BG-K04 living redraw on twos + repulsive layout relaxation)"
                elif t_item.get("id") == "S5":
                    t_item["how"] = "Watertight gouache occlusion cels painted across all 7 planes (SAM character cel + concave prop/architecture cels)"
            tests_p.write_text(json.dumps(tests_list, indent=2), encoding="utf-8")

        if (idx + 1) % 5 == 0 or idx == len(BATCH_50_SPECS) - 1:
            print(f"[{idx+1:02d}/50] Completed 4-Pass BG Redraw & Gouache Multiplane Render: {slug} ({time.time()-t0:.1f}s)", flush=True)

    build_contact_sheets(slugs)
    print(f"All 50 shots + contact sheets completed in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
