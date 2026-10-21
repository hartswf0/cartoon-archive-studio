#!/usr/bin/env python3
"""
SCION + UNIVERSAL ARGON RL + 4-PASS BG REDRAW — BATCH-2 OF 50 ENGINE (SHOTS #71–#120)
======================================================================================
Processes the next 50 diverse categories from the Moving Image Archive + Google Quick, Draw!
into complete 33-frame Full SCION 7-Plane Multiplane Cartoons (expanding the studio
from 70 to 120 complete shots across 8 kinematic & ontological regimes).

Every shot executes:
  1. Automatic Continuous Archival Interval Selection (33 frames @ 640x480)
  2. Batched Depth-Anything-V2 + GroundingDINO + Foreground Depth Guard + SAM (`facebook/sam-vit-base`)
  3. Zero-Shot Quick, Draw! Prototype Selection (Top-25 recognized human drawings x flip)
  4. 4-Keyframe SCION Rotoscope-Doodle Synthesis + 4-Episode Argon RL Smooth Cage Evolution (`rl-overlay-pass.png`)
  5. 4-Pass Living Background Redraw & Multiplane Gouache Occlusion Engine (`scene-plate.png`, `bg-redraw-pass.png`)
  6. Full SCION 7-Plane Multiplane Rendering + Frame-Synced Foley Audio (`soundtrack.wav`, `comparison.mp4`)
  7. 16/16 Verification Gates (`tests.json`) + 5x5 Contact Sheets (`batch2_stages_*.png`, `batch2_scene_plates_*.png`).
"""

import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from tools.scion_batch50_engine import (
    ROOT,
    SHOTS_DIR,
    PREVIEW_DIR,
    ART_DIR,
    DEPTH_SNAP,
    select_best_continuous_interval,
    run_fast_auto_sam_and_depth,
    select_best_quickdraw_prototype,
    synthesize_4_scion_doodle_keyframes_and_rl,
    synthesize_unique_7plane_scene,
    render_scion_multiplane_shot,
    write_complete_metadata_and_16_gates,
)

BATCH_NEXT_50_SPECS = [
    # Regime I: Terrestrial Mammals, Reptiles & Arthropods (10)
    {"category": "cat", "slug": "cat", "prompt": "cat. kitten. feline.", "regime": "Regime I: Terrestrial Locomotion", "water": False, "grounded": True},
    {"category": "rhinoceros", "slug": "rhinoceros", "prompt": "rhinoceros. rhino. animal.", "regime": "Regime I: Terrestrial Locomotion", "water": False, "grounded": True},
    {"category": "crocodile", "slug": "crocodile", "prompt": "crocodile. alligator. reptile.", "regime": "Regime I: Terrestrial & Amphibious", "water": True, "grounded": True},
    {"category": "snake", "slug": "snake", "prompt": "snake. serpent. reptile.", "regime": "Regime VIII: Biological Articulation", "water": False, "grounded": True},
    {"category": "hedgehog", "slug": "hedgehog", "prompt": "hedgehog. porcupine. mammal.", "regime": "Regime I: Terrestrial Locomotion", "water": False, "grounded": True},
    {"category": "mouse", "slug": "mouse", "prompt": "mouse. rat. rodent.", "regime": "Regime I: Terrestrial Locomotion", "water": False, "grounded": True},
    {"category": "scorpion", "slug": "scorpion", "prompt": "scorpion. arachnid. insect.", "regime": "Regime VIII: Biological Articulation", "water": False, "grounded": True},
    {"category": "ant", "slug": "ant", "prompt": "ant. insect. bug.", "regime": "Regime VIII: Biological Articulation", "water": False, "grounded": True},
    {"category": "bee", "slug": "bee", "prompt": "bee. bumblebee. insect.", "regime": "Regime II: Aerial & Wingbeat", "water": False, "grounded": False},
    {"category": "bat", "slug": "bat", "prompt": "bat. flying bat. mammal.", "regime": "Regime II: Aerial & Wingbeat", "water": False, "grounded": False},

    # Regime II: Avian, Marine & Mythic Creatures (6)
    {"category": "bird", "slug": "bird", "prompt": "bird. songbird. sparrow.", "regime": "Regime II: Aerial & Wingbeat", "water": False, "grounded": True},
    {"category": "fish", "slug": "fish", "prompt": "fish. swimming fish.", "regime": "Regime III: Aquatic & Marine", "water": True, "grounded": False},
    {"category": "octopus", "slug": "octopus", "prompt": "octopus. squid. marine animal.", "regime": "Regime III: Aquatic & Marine", "water": True, "grounded": False},
    {"category": "lobster", "slug": "lobster", "prompt": "lobster. crab. crustacean.", "regime": "Regime III: Aquatic & Marine", "water": True, "grounded": True},
    {"category": "mermaid", "slug": "mermaid", "prompt": "mermaid. swimmer. person.", "regime": "Regime III: Aquatic & Marine", "water": True, "grounded": False},
    {"category": "dragon", "slug": "dragon", "prompt": "dragon. monster. dinosaur.", "regime": "Regime II: Aerial & Mythic", "water": False, "grounded": True},

    # Regime III: Road, Emergency & Recreational Vehicles + Submarine (8)
    {"category": "pickup_truck", "slug": "pickup-truck", "prompt": "pickup truck. truck. vehicle.", "regime": "Regime IV: Wheeled & Heavy Vehicles", "water": False, "grounded": True},
    {"category": "police_car", "slug": "police-car", "prompt": "police car. patrol car. car.", "regime": "Regime IV: Wheeled & Heavy Vehicles", "water": False, "grounded": True},
    {"category": "school_bus", "slug": "school-bus", "prompt": "school bus. bus. vehicle.", "regime": "Regime IV: Wheeled & Heavy Vehicles", "water": False, "grounded": True},
    {"category": "truck", "slug": "truck", "prompt": "truck. semi truck. lorry.", "regime": "Regime IV: Wheeled & Heavy Vehicles", "water": False, "grounded": True},
    {"category": "van", "slug": "van", "prompt": "van. minivan. vehicle.", "regime": "Regime IV: Wheeled & Heavy Vehicles", "water": False, "grounded": True},
    {"category": "skateboard", "slug": "skateboard", "prompt": "skateboard. skater. board.", "regime": "Regime IV: Wheeled & Kinetic", "water": False, "grounded": True},
    {"category": "rollerskates", "slug": "rollerskates", "prompt": "roller skates. skate. shoe.", "regime": "Regime IV: Wheeled & Kinetic", "water": False, "grounded": True},
    {"category": "submarine", "slug": "submarine", "prompt": "submarine. boat. vessel.", "regime": "Regime V: Naval & Maritime", "water": True, "grounded": False},

    # Regime IV: Musical Instruments & Audio Performance (6)
    {"category": "clarinet", "slug": "clarinet", "prompt": "clarinet. flute. instrument.", "regime": "Regime VI: Musical Performance", "water": False, "grounded": False},
    {"category": "drums", "slug": "drums", "prompt": "drums. drum kit. drummer.", "regime": "Regime VI: Musical Performance", "water": False, "grounded": True},
    {"category": "harp", "slug": "harp", "prompt": "harp. musical instrument.", "regime": "Regime VI: Musical Performance", "water": False, "grounded": True},
    {"category": "saxophone", "slug": "saxophone", "prompt": "saxophone. horn. instrument.", "regime": "Regime VI: Musical Performance", "water": False, "grounded": False},
    {"category": "trombone", "slug": "trombone", "prompt": "trombone. brass horn. instrument.", "regime": "Regime VI: Musical Performance", "water": False, "grounded": False},
    {"category": "microphone", "slug": "microphone", "prompt": "microphone. mic. singer.", "regime": "Regime VI: Musical Performance", "water": False, "grounded": True},

    # Regime V: World Monuments, Architecture & Outdoor Structures (7)
    {"category": "the_eiffel_tower", "slug": "the-eiffel-tower", "prompt": "Eiffel Tower. tower. monument.", "regime": "Regime VII: Monumental Architecture", "water": False, "grounded": True},
    {"category": "the_great_wall_of_china", "slug": "the-great-wall-of-china", "prompt": "Great Wall. stone wall. fortress.", "regime": "Regime VII: Monumental Architecture", "water": False, "grounded": True},
    {"category": "skyscraper", "slug": "skyscraper", "prompt": "skyscraper. tall building. tower.", "regime": "Regime VII: Monumental Architecture", "water": False, "grounded": True},
    {"category": "house", "slug": "house", "prompt": "house. home. building.", "regime": "Regime VII: Monumental Architecture", "water": False, "grounded": True},
    {"category": "hospital", "slug": "hospital", "prompt": "hospital. building. facade.", "regime": "Regime VII: Monumental Architecture", "water": False, "grounded": True},
    {"category": "tent", "slug": "tent", "prompt": "tent. camping tent. shelter.", "regime": "Regime VII: Monumental Architecture", "water": False, "grounded": True},
    {"category": "cannon", "slug": "cannon", "prompt": "cannon. artillery. wheel.", "regime": "Regime VII: Monumental & Mechanical", "water": False, "grounded": True},

    # Regime VI: Natural Forces, Landscapes & Botanicals (6)
    {"category": "hurricane", "slug": "hurricane", "prompt": "hurricane. storm. cyclone.", "regime": "Regime VIII: Elemental & Botanical", "water": True, "grounded": False},
    {"category": "lightning", "slug": "lightning", "prompt": "lightning. storm cloud. sky.", "regime": "Regime VIII: Elemental & Botanical", "water": False, "grounded": False},
    {"category": "rainbow", "slug": "rainbow", "prompt": "rainbow. sky. landscape.", "regime": "Regime VIII: Elemental & Botanical", "water": False, "grounded": False},
    {"category": "mountain", "slug": "mountain", "prompt": "mountain. peak. ridge.", "regime": "Regime VII: Monumental & Landscape", "water": False, "grounded": True},
    {"category": "palm_tree", "slug": "palm-tree", "prompt": "palm tree. tree. foliage.", "regime": "Regime VIII: Elemental & Botanical", "water": False, "grounded": True},
    {"category": "cactus", "slug": "cactus", "prompt": "cactus. desert plant.", "regime": "Regime VIII: Elemental & Botanical", "water": False, "grounded": True},

    # Regime VII: Kinetic Sports, Light & Time Instruments (7)
    {"category": "basketball", "slug": "basketball", "prompt": "basketball. ball. player.", "regime": "Regime IV: Kinetic & Sports", "water": False, "grounded": False},
    {"category": "soccer_ball", "slug": "soccer-ball", "prompt": "soccer ball. football. ball.", "regime": "Regime IV: Kinetic & Sports", "water": False, "grounded": True},
    {"category": "baseball_bat", "slug": "baseball-bat", "prompt": "baseball bat. batter. player.", "regime": "Regime IV: Kinetic & Sports", "water": False, "grounded": False},
    {"category": "ceiling_fan", "slug": "ceiling-fan", "prompt": "ceiling fan. fan blades. lamp.", "regime": "Regime VII: Mechanical & Interior", "water": False, "grounded": False},
    {"category": "lantern", "slug": "lantern", "prompt": "lantern. lamp. light.", "regime": "Regime VII: Mechanical & Interior", "water": False, "grounded": True},
    {"category": "hourglass", "slug": "hourglass", "prompt": "hourglass. glass timer. object.", "regime": "Regime VII: Mechanical & Interior", "water": False, "grounded": True},
    {"category": "umbrella", "slug": "umbrella", "prompt": "umbrella. parasol. person.", "regime": "Regime VII: Mechanical & Exterior", "water": False, "grounded": False},
]


def build_batch2_contact_sheets(slugs: list):
    """Builds 5x5 contact sheets for Batch-2 of 50 (`batch2_scene_plates_*.png` and `batch2_stages_*.png`)."""
    tw, th = 256, 192

    def make_5x5_grid(group_slugs, img_getter, start_num=1):
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
                cv2.rectangle(im, (4, 4), (195, 24), (18, 20, 22), -1)
                cv2.putText(
                    im,
                    f"{start_num+idx:02d}. {slug.upper()[:16]}",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (240, 255, 180),
                    1,
                    cv2.LINE_AA,
                )
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

    sp_1_25 = make_5x5_grid(slugs[:25], get_scene_plate, start_num=51)
    sp_26_50 = make_5x5_grid(slugs[25:50], get_scene_plate, start_num=76)
    st_1_25 = make_5x5_grid(slugs[:25], get_stage_frame, start_num=51)
    st_26_50 = make_5x5_grid(slugs[25:50], get_stage_frame, start_num=76)

    for name, arr in [
        ("batch2_scene_plates_1_25.png", sp_1_25),
        ("batch2_scene_plates_26_50.png", sp_26_50),
        ("batch2_stages_1_25.png", st_1_25),
        ("batch2_stages_26_50.png", st_26_50),
    ]:
        cv2.imwrite(str(PREVIEW_DIR / name), arr)
        if ART_DIR.exists():
            cv2.imwrite(str(ART_DIR / name), arr)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[SCION Batch-2 of 50 Engine] Loading GroundingDINO + Depth-Anything-V2 + SAM on {device}...", flush=True)

    dino_id = "IDEA-Research/grounding-dino-tiny"
    dino_proc = AutoProcessor.from_pretrained(dino_id, local_files_only=True)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id, local_files_only=True).to(device).eval()

    depth_proc = AutoImageProcessor.from_pretrained(str(DEPTH_SNAP), local_files_only=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained(str(DEPTH_SNAP), local_files_only=True).to(device).eval()

    sam_id = "facebook/sam-vit-base"
    sam_proc = SamProcessor.from_pretrained(sam_id, local_files_only=True)
    sam_model = SamModel.from_pretrained(sam_id, local_files_only=True).to(device).eval()

    summary_rows = []
    t_start = time.time()

    for idx, spec in enumerate(BATCH_NEXT_50_SPECS):
        slug = spec["slug"]
        sdir = SHOTS_DIR / slug
        sdir.mkdir(parents=True, exist_ok=True)
        t_s = time.time()

        frames_bgr, interval_meta = select_best_continuous_interval(spec["category"], sdir)
        sam_masks_640, depths, smooth_boxes, sam_ious, pos_pts, neg_pts = run_fast_auto_sam_and_depth(
            spec, sdir, frames_bgr, dino_proc, dino_model, depth_proc, depth_model, sam_proc, sam_model, device
        )
        qd_polylines, design_meta = select_best_quickdraw_prototype(spec["category"], sdir, sam_masks_640, frames_bgr)
        cast_frames_480, cel_masks_480, rl_log = synthesize_4_scion_doodle_keyframes_and_rl(
            spec, sdir, frames_bgr, sam_masks_640, smooth_boxes, sam_ious, pos_pts, neg_pts, qd_polylines
        )
        planes_480, objects = synthesize_unique_7plane_scene(spec, sdir, frames_bgr, sam_masks_640, depths)
        render_scion_multiplane_shot(spec, sdir, frames_bgr, cast_frames_480, cel_masks_480, planes_480, objects)
        write_complete_metadata_and_16_gates(spec, sdir, interval_meta, design_meta, rl_log)

        dt = time.time() - t_s
        row = {
            "idx": idx + 51,
            "slug": slug,
            "category": spec["category"],
            "clip": interval_meta["clip_prefix"],
            "interval": interval_meta["interval"],
            "sam_iou": rl_log["mean_sam_iou"],
            "ep0": rl_log["mean_reward_ep0"],
            "ep3": rl_log["mean_reward_ep3"],
            "sil_iou": rl_log["mean_sil_iou_ep3"],
            "in_mask": rl_log["mean_stroke_precision_ep3"],
            "qd_key_id": design_meta["key_id"],
            "sec": round(dt, 1),
        }
        summary_rows.append(row)
        print(
            f"[{idx+1:02d}/50 | #{idx+51:03d}] {slug:24s} | Clip {row['clip']} ({row['interval'][0]:.1f}..{row['interval'][1]:.1f}s) "
            f"| SAM IoU: {row['sam_iou']:.3f} | RL Ep0->Ep3: {row['ep0']:.3f}->{row['ep3']:.3f} "
            f"| InMask: {row['in_mask']*100:.1f}% | {dt:.1f}s",
            flush=True,
        )

    (ROOT / "data" / "batch_next50_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    slugs = [s["slug"] for s in BATCH_NEXT_50_SPECS]
    build_batch2_contact_sheets(slugs)
    print(f"\n[SCION Batch-2 of 50 Engine] Completed all 50 new shots (#51..#100 -> 120 total) in {time.time() - t_start:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
