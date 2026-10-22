#!/usr/bin/env python3
"""
SCION-UNIQUEWORLD-v9 + UNIVERSAL ARGON RL v9 — BATCH-6 (FINAL 76 CATEGORIES, SHOTS #271–#346)
+ REFRESH BATCH-4 & BATCH-5 (#171–#270) WITH ONE-OF-A-KIND PER-SHOT BACKGROUNDS
==============================================================================================
Completes 100% of the 345-category Moving Image Archive x Google Quick, Draw! corpus
(expanding the studio from 270 to 346 complete Full-Frame SCION 7-Plane Multiplane Cartoons)
and upgrades every background across Shots #171–#346 to `SCION-UniqueWorld-v9` so no two
shots ever share a repeated room template, window layout, radial floorboard grid, or prop topology:
  - 36 distinct Architectural & Landscape Blueprints (18 Indoor + 18 Outdoor)
  - 16 distinct Foreground Surface / Floor / Ground Patterns (Herringbone Parquet, Diamond Tiles,
    Braided Fringe Rug, Running-Bond Flagstones, S-Curve Stepping Stones, Bolted Pier Planks,
    Drawer-Front Counter Apron, Arched Cobblestone Courses, Ascending Stair Treads, Brook Boulders,
    Converging Rail Sleepers, Stone Balustrade, Scalloped Surf, Diagonal Furrows, Court Chalk Arc,
    Woodland Roots & Tufts)
  - 12 distinct Sky / Ceiling Accents & 12 Asymmetric Multiplane Prop Topologies + Exemplar Rotation
"""

import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
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
from tools.scion_batch4_5_100_engine import (
    BATCH_4_50_SPECS,
    BATCH_5_50_SPECS,
)

BATCH_6_76_SPECS = [
    # Regime I: Expressive Human Anatomy, Gesture & Portraiture (12)
    {"category": "arm", "slug": "arm", "prompt": "human arm. flexed arm. hand and arm.", "regime": "Regime VIII: Expressive Anatomy & Gesture", "water": False, "grounded": False},
    {"category": "elbow", "slug": "elbow", "prompt": "elbow. bent arm. human arm.", "regime": "Regime VIII: Expressive Anatomy & Gesture", "water": False, "grounded": False},
    {"category": "finger", "slug": "finger", "prompt": "pointing finger. hand. fingers.", "regime": "Regime VIII: Expressive Anatomy & Gesture", "water": False, "grounded": False},
    {"category": "foot", "slug": "foot", "prompt": "human foot. bare foot. stepping foot.", "regime": "Regime VIII: Expressive Anatomy & Gesture", "water": False, "grounded": True},
    {"category": "toe", "slug": "toe", "prompt": "toes. bare foot. foot.", "regime": "Regime VIII: Expressive Anatomy & Gesture", "water": False, "grounded": True},
    {"category": "knee", "slug": "knee", "prompt": "knee. human leg. bent knee.", "regime": "Regime VIII: Expressive Anatomy & Gesture", "water": False, "grounded": True},
    {"category": "leg", "slug": "leg", "prompt": "human leg. walking legs. stride.", "regime": "Regime VIII: Expressive Anatomy & Gesture", "water": False, "grounded": True},
    {"category": "brain", "slug": "brain", "prompt": "brain. head. science diagram.", "regime": "Regime V: Medical & Anatomical", "water": False, "grounded": False},
    {"category": "tooth", "slug": "tooth", "prompt": "tooth. smile. dentist.", "regime": "Regime V: Medical & Anatomical", "water": False, "grounded": False},
    {"category": "goatee", "slug": "goatee", "prompt": "bearded man. goatee. chin beard.", "regime": "Regime VIII: Expressive Portrait", "water": False, "grounded": False},
    {"category": "smiley_face", "slug": "smiley-face", "prompt": "smiling face. happy person. portrait.", "regime": "Regime VIII: Expressive Portrait", "water": False, "grounded": False},
    {"category": "stitches", "slug": "stitches", "prompt": "sewing stitches. medical stitches. thread.", "regime": "Regime V: Medical & Craft", "water": False, "grounded": False},

    # Regime II: Botanical Harvest, Vegetables & Pantry Produce (10)
    {"category": "asparagus", "slug": "asparagus", "prompt": "asparagus. green vegetable. stalks.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "blackberry", "slug": "blackberry", "prompt": "blackberry. dark berry. fruit.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "broccoli", "slug": "broccoli", "prompt": "broccoli. green vegetable. floret.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "carrot", "slug": "carrot", "prompt": "carrot. root vegetable. orange carrot.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "onion", "slug": "onion", "prompt": "onion. bulb vegetable. kitchen onion.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "peanut", "slug": "peanut", "prompt": "peanut. nut. snack.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "peas", "slug": "peas", "prompt": "peas. pea pod. green peas.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "potato", "slug": "potato", "prompt": "potato. root vegetable. harvest.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "string_bean", "slug": "string-bean", "prompt": "green beans. string bean. pod.", "regime": "Regime VIII: Botanical Harvest", "water": False, "grounded": True},
    {"category": "basket", "slug": "basket", "prompt": "wicker basket. picnic basket. woven basket.", "regime": "Regime VI: Woven Vessel & Harvest", "water": False, "grounded": True},

    # Regime III: Culinary Tableware, Cutlery & Confectionery (12)
    {"category": "bottlecap", "slug": "bottlecap", "prompt": "bottle cap. soda bottle. cap.", "regime": "Regime VI: Culinary & Tableware", "water": False, "grounded": True},
    {"category": "cup", "slug": "cup", "prompt": "drinking cup. teacup. vessel.", "regime": "Regime VI: Culinary & Tableware", "water": False, "grounded": True},
    {"category": "mug", "slug": "mug", "prompt": "coffee mug. ceramic mug. handle cup.", "regime": "Regime VI: Culinary & Tableware", "water": False, "grounded": True},
    {"category": "fork", "slug": "fork", "prompt": "dining fork. silverware. utensil.", "regime": "Regime IV: Culinary Cutlery", "water": False, "grounded": True},
    {"category": "knife", "slug": "knife", "prompt": "kitchen knife. blade. cutlery.", "regime": "Regime IV: Culinary Cutlery", "water": False, "grounded": True},
    {"category": "spoon", "slug": "spoon", "prompt": "spoon. soup spoon. silverware.", "regime": "Regime IV: Culinary Cutlery", "water": False, "grounded": True},
    {"category": "lollipop", "slug": "lollipop", "prompt": "lollipop. candy on stick. sweet.", "regime": "Regime VI: Confectionery & Bakery", "water": False, "grounded": False},
    {"category": "popsicle", "slug": "popsicle", "prompt": "popsicle. ice pop. frozen treat.", "regime": "Regime VI: Confectionery & Summer", "water": False, "grounded": False},
    {"category": "sandwich", "slug": "sandwich", "prompt": "sandwich. bread slices. lunch.", "regime": "Regime VI: Culinary & Hearth", "water": False, "grounded": True},
    {"category": "steak", "slug": "steak", "prompt": "steak. meat on plate. grill.", "regime": "Regime VI: Culinary & Hearth", "water": False, "grounded": True},
    {"category": "cooler", "slug": "cooler", "prompt": "ice cooler. picnic chest. box.", "regime": "Regime VII: Outdoor & Picnic", "water": False, "grounded": True},
    {"category": "dishwasher", "slug": "dishwasher", "prompt": "dishwasher. kitchen appliance. rack.", "regime": "Regime V: Domestic Mechanism", "water": True, "grounded": True},

    # Regime IV: Wardrobe, Apparel, Jewelry & Personal Accessories (12)
    {"category": "belt", "slug": "belt", "prompt": "leather belt. waist belt. buckle.", "regime": "Regime VII: Apparel & Haberdashery", "water": False, "grounded": False},
    {"category": "bracelet", "slug": "bracelet", "prompt": "bracelet. wrist bangle. jewelry.", "regime": "Regime VII: Apparel & Regalia", "water": False, "grounded": False},
    {"category": "diamond", "slug": "diamond", "prompt": "diamond gem. jewel. crystal.", "regime": "Regime VII: Regalia & Optics", "water": False, "grounded": True},
    {"category": "helmet", "slug": "helmet", "prompt": "helmet. protective helmet. headgear.", "regime": "Regime VII: Protective Gear", "water": False, "grounded": True},
    {"category": "pants", "slug": "pants", "prompt": "trousers. pants. slacks.", "regime": "Regime VII: Apparel & Haberdashery", "water": False, "grounded": True},
    {"category": "purse", "slug": "purse", "prompt": "handbag. purse. shoulder bag.", "regime": "Regime VII: Apparel & Haberdashery", "water": False, "grounded": True},
    {"category": "shorts", "slug": "shorts", "prompt": "shorts. summer shorts. clothing.", "regime": "Regime VII: Apparel & Recreation", "water": False, "grounded": True},
    {"category": "sock", "slug": "sock", "prompt": "sock. footwear. knitted sock.", "regime": "Regime VII: Apparel & Haberdashery", "water": False, "grounded": True},
    {"category": "sweater", "slug": "sweater", "prompt": "sweater. pullover. knitwear.", "regime": "Regime VII: Apparel & Haberdashery", "water": False, "grounded": False},
    {"category": "t_shirt", "slug": "t-shirt", "prompt": "t-shirt. shirt. garment.", "regime": "Regime VII: Apparel & Haberdashery", "water": False, "grounded": False},
    {"category": "underwear", "slug": "underwear", "prompt": "briefs. garment. laundry.", "regime": "Regime VII: Apparel & Domestic", "water": False, "grounded": False},
    {"category": "camouflage", "slug": "camouflage", "prompt": "camouflage uniform. foliage pattern. woods.", "regime": "Regime VII: Field & Woodland", "water": False, "grounded": True},

    # Regime V: Domestic Interior, Hearth, Electronics & Correspondence (16)
    {"category": "bandage", "slug": "bandage", "prompt": "bandage. first aid. medical wrap.", "regime": "Regime V: Medical & Care", "water": False, "grounded": False},
    {"category": "toothpaste", "slug": "toothpaste", "prompt": "toothpaste tube. dental care. bathroom.", "regime": "Regime VI: Domestic Sanctuary", "water": False, "grounded": True},
    {"category": "toilet", "slug": "toilet", "prompt": "porcelain fixture. bathroom. plumbing.", "regime": "Regime VI: Domestic Sanctuary", "water": True, "grounded": True},
    {"category": "pillow", "slug": "pillow", "prompt": "pillow. cushion. bed pillow.", "regime": "Regime VI: Interior Sanctuary", "water": False, "grounded": True},
    {"category": "candle", "slug": "candle", "prompt": "lit candle. candle flame. candlestick.", "regime": "Regime V: Luminous Hearth", "water": False, "grounded": True},
    {"category": "lighter", "slug": "lighter", "prompt": "pocket lighter. flame. metal lighter.", "regime": "Regime IV: Ignition Mechanism", "water": False, "grounded": False},
    {"category": "matches", "slug": "matches", "prompt": "matchstick. lit match. matchbox.", "regime": "Regime IV: Ignition Mechanism", "water": False, "grounded": False},
    {"category": "nail", "slug": "nail", "prompt": "metal nail. carpentry nail. hardware.", "regime": "Regime IV: Carpentry & Hardware", "water": False, "grounded": True},
    {"category": "power_outlet", "slug": "power-outlet", "prompt": "wall socket. electrical outlet. plug.", "regime": "Regime V: Electrical & Utility", "water": False, "grounded": False},
    {"category": "headphones", "slug": "headphones", "prompt": "headphones. audio headset. earphones.", "regime": "Regime V: Acoustic & Broadcast", "water": False, "grounded": False},
    {"category": "remote_control", "slug": "remote-control", "prompt": "remote control. handheld keypad. television remote.", "regime": "Regime V: Broadcast & Electronics", "water": False, "grounded": False},
    {"category": "stereo", "slug": "stereo", "prompt": "stereo boombox. speakers. radio receiver.", "regime": "Regime V: Acoustic & Broadcast", "water": False, "grounded": True},
    {"category": "eraser", "slug": "eraser", "prompt": "rubber eraser. desk supply. chalkboard eraser.", "regime": "Regime IV: Atelier & Stationer", "water": False, "grounded": True},
    {"category": "paper_clip", "slug": "paper-clip", "prompt": "wire paper clip. office stationery. desk.", "regime": "Regime IV: Atelier & Stationer", "water": False, "grounded": True},
    {"category": "spreadsheet", "slug": "spreadsheet", "prompt": "ledger grid. chart table. office document.", "regime": "Regime V: Literary & Computing", "water": False, "grounded": True},
    {"category": "passport", "slug": "passport", "prompt": "passport booklet. travel document. stamp.", "regime": "Regime VII: Travel & Correspondence", "water": False, "grounded": True},

    # Regime VI: Civic Fixtures, Parklands & Geometric / Celestial Icons (14)
    {"category": "postcard", "slug": "postcard", "prompt": "postcard. vintage card. mail.", "regime": "Regime VII: Travel & Correspondence", "water": False, "grounded": True},
    {"category": "mailbox", "slug": "mailbox", "prompt": "mailbox. post box. letterbox.", "regime": "Regime VI: Civic & Roadside", "water": False, "grounded": True},
    {"category": "bench", "slug": "bench", "prompt": "park bench. wooden bench. seat.", "regime": "Regime VI: Civic & Parkland", "water": False, "grounded": True},
    {"category": "jail", "slug": "jail", "prompt": "iron bars. cell window. stone fortress.", "regime": "Regime VI: Civic Architecture", "water": False, "grounded": True},
    {"category": "rifle", "slug": "rifle", "prompt": "vintage rifle. long barrel. target.", "regime": "Regime IV: Ballistic & Historical", "water": False, "grounded": False},
    {"category": "star", "slug": "star", "prompt": "night star. glowing star. astronomy.", "regime": "Regime VIII: Celestial & Geometric", "water": False, "grounded": False},
    {"category": "circle", "slug": "circle", "prompt": "circle ring. round wheel. sphere.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": False},
    {"category": "square", "slug": "square", "prompt": "square frame. box. geometric square.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": True},
    {"category": "triangle", "slug": "triangle", "prompt": "triangle. pyramid peak. geometric.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": True},
    {"category": "hexagon", "slug": "hexagon", "prompt": "hexagon. honeycomb cell. polygon.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": False},
    {"category": "octagon", "slug": "octagon", "prompt": "octagon sign. eight-sided shape. stop sign.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": True},
    {"category": "line", "slug": "line", "prompt": "horizon line. path line. tightrope.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": True},
    {"category": "squiggle", "slug": "squiggle", "prompt": "wavy line. ribbon curve. squiggle.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": False},
    {"category": "zigzag", "slug": "zigzag", "prompt": "zigzag bolt. sawtooth wave. lightning.", "regime": "Regime VIII: Geometric & Graphic", "water": False, "grounded": False},
]


def refresh_cached_shot_background_v9(spec: dict, idx: int, total: int):
    """
    Fast scene-only refresh for already-computed shots (`Batch-4` & `Batch-5`):
    loads cached `frames-src/`, `sam-masks/`, `depth-maps/`, and `shot.json` (`cast_frames_480` + `cel_masks_480`),
    runs `synthesize_unique_7plane_scene` with `SCION-UniqueWorld-v9`, and re-renders `comparison.mp4`,
    `stack.png`, `bg-redraw-pass.png`, `scene-plate.png`, and `shot.json`.
    """
    t0 = time.time()
    slug = spec["slug"]
    sdir = SHOTS_DIR / slug
    f_paths = sorted((sdir / "frames-src").glob("f*.png"))
    m_paths = sorted((sdir / "sam-masks").glob("m*.png"))
    d_paths = sorted((sdir / "depth-maps").glob("d*.png"))
    shot_json_path = sdir / "shot.json"

    if len(f_paths) < 33 or len(m_paths) < 33 or len(d_paths) < 33 or not shot_json_path.exists():
        return False

    frames_bgr = [cv2.imread(str(p)) for p in f_paths[:33]]
    sam_masks_640 = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in m_paths[:33]]
    depths = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in d_paths[:33]]

    old_shot = json.loads(shot_json_path.read_text())
    cast_frames_480 = old_shot["cast"]["frames"]
    cel_masks_480 = old_shot["cast"].get("cel_masks", old_shot.get("masks", []))

    planes_480, objects = synthesize_unique_7plane_scene(spec, sdir, frames_bgr, sam_masks_640, depths)
    render_scion_multiplane_shot(spec, sdir, frames_bgr, cast_frames_480, cel_masks_480, planes_480, objects)

    dt = time.time() - t0
    print(f"[REFRESH-v9 {idx+1:03d}/{total:03d}] {slug:18s} | UniqueWorld-v9 Stage & Render Updated | {dt:.2f}s", flush=True)
    return True


def build_contact_sheets_generic(specs: list, prefix: str, chunks_tags: list):
    """Builds 5x5 contact sheets of final multiplane cartoon frames (`f=10`) and full-frame scene plates."""
    for chunk, tag in chunks_tags:
        stage_rows = []
        plate_rows = []
        for r in range(5):
            s_row = []
            p_row = []
            for c in range(5):
                idx_c = r * 5 + c
                spec = chunk[idx_c] if idx_c < len(chunk) else chunk[-1]
                slug = spec["slug"]
                sdir = SHOTS_DIR / slug

                cap = cv2.VideoCapture(str(sdir / "comparison.mp4"))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
                ok, comp = cap.read()
                cap.release()
                if ok and comp is not None and comp.shape[1] >= 1280:
                    st = comp[:, 640:1280]
                else:
                    st = np.full((480, 640, 3), 242, dtype=np.uint8)
                st_sm = cv2.resize(st, (320, 240), interpolation=cv2.INTER_AREA)
                cv2.rectangle(st_sm, (0, 0), (320, 24), (18, 20, 22), -1)
                cv2.putText(st_sm, f"{slug.upper()} (UNIQUE-WORLD v9)", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (130, 255, 160), 1, cv2.LINE_AA)
                cv2.rectangle(st_sm, (0, 0), (319, 239), (180, 180, 180), 1)
                s_row.append(st_sm)

                sp = cv2.imread(str(sdir / "scene-plate.png"))
                if sp is None:
                    sp = np.full((480, 640, 3), 255, dtype=np.uint8)
                sp_sm = cv2.resize(sp, (320, 240), interpolation=cv2.INTER_AREA)
                cv2.rectangle(sp_sm, (0, 0), (320, 24), (18, 20, 22), -1)
                cv2.putText(sp_sm, f"{slug.upper()} (UNIQUE PLATE v9)", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (140, 225, 255), 1, cv2.LINE_AA)
                cv2.rectangle(sp_sm, (0, 0), (319, 239), (180, 180, 180), 1)
                p_row.append(sp_sm)

            stage_rows.append(np.hstack(s_row))
            plate_rows.append(np.hstack(p_row))

        st_out = PREVIEW_DIR / f"{prefix}_stages_{tag}.png"
        pl_out = PREVIEW_DIR / f"{prefix}_scene_plates_{tag}.png"
        cv2.imwrite(str(st_out), np.vstack(stage_rows))
        cv2.imwrite(str(pl_out), np.vstack(plate_rows))
        if ART_DIR.exists():
            cv2.imwrite(str(ART_DIR / f"{prefix}_stages_{tag}.png"), np.vstack(stage_rows))
            cv2.imwrite(str(ART_DIR / f"{prefix}_scene_plates_{tag}.png"), np.vstack(plate_rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-refresh", action="store_true", help="Skip refreshing Batch-4 and Batch-5 backgrounds")
    parser.add_argument("--limit", type=int, default=76, help="Number of Batch-6 shots to process (default: all 76)")
    args = parser.parse_args()

    t_start = time.time()

    # 1. Refresh all 100 shots of Batch-4 & Batch-5 (#171-#270) with SCION-UniqueWorld-v9
    if not args.skip_refresh:
        print("[SCION-UniqueWorld-v9] Refreshing Batch-4 & Batch-5 (100 shots #171-#270) with unique per-shot worlds...", flush=True)
        b45_specs = BATCH_4_50_SPECS + BATCH_5_50_SPECS
        for idx, spec in enumerate(b45_specs):
            refresh_cached_shot_background_v9(spec, idx, len(b45_specs))
        build_contact_sheets_generic(BATCH_4_50_SPECS, "batch4", [(BATCH_4_50_SPECS[:25], "1_25"), (BATCH_4_50_SPECS[25:50], "26_50")])
        build_contact_sheets_generic(BATCH_5_50_SPECS, "batch5", [(BATCH_5_50_SPECS[:25], "1_25"), (BATCH_5_50_SPECS[25:50], "26_50")])
        print("[SCION-UniqueWorld-v9] Refreshed Batch-4 & Batch-5 contact sheets!", flush=True)

    # 2. Process all 76 remaining categories (Batch-6: Shots #271-#346)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[SCION-UniqueWorld-v9] Loading GroundingDINO + Depth-Anything-V2 + SAM on {device} for Batch-6 (76 shots)...", flush=True)

    dino_id = "IDEA-Research/grounding-dino-tiny"
    dino_proc = AutoProcessor.from_pretrained(dino_id, local_files_only=True)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id, local_files_only=True).to(device).eval()

    depth_proc = AutoImageProcessor.from_pretrained(str(DEPTH_SNAP), local_files_only=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained(str(DEPTH_SNAP), local_files_only=True).to(device).eval()

    sam_id = "facebook/sam-vit-base"
    sam_proc = SamProcessor.from_pretrained(sam_id, local_files_only=True)
    sam_model = SamModel.from_pretrained(sam_id, local_files_only=True).to(device).eval()

    target_specs = BATCH_6_76_SPECS[: args.limit]
    summary_rows = []

    for idx, spec in enumerate(target_specs):
        slug = spec["slug"]
        sdir = SHOTS_DIR / slug
        sdir.mkdir(parents=True, exist_ok=True)
        if (sdir / "shot.json").exists() and (sdir / "rl_policy_log.json").exists() and (sdir / "tests.json").exists():
            rl_log = json.loads((sdir / "rl_policy_log.json").read_text())
            summary_rows.append(rl_log)
            print(f"[{idx+1:02d}/{len(target_specs):02d}] {slug:18s} | Already completed (SAM_IoU={rl_log['mean_sam_iou']:.3f}, SilIoU={rl_log['mean_sil_iou_ep3']:.2f})", flush=True)
            continue
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
        print(
            f"[{idx+1:02d}/{len(target_specs):02d}] {slug:18s} | QD#{design_meta['qd_index']:02d}({design_meta['country']}) "
            f"| SAM_IoU={rl_log['mean_sam_iou']:.3f} "
            f"| R: {rl_log['mean_reward_ep0']:.3f} -> {rl_log['mean_reward_ep3']:.3f} (+{rl_log['reward_delta']:.3f}) "
            f"| SilIoU={rl_log['mean_sil_iou_ep3']:.2f} | InMask={int(rl_log['mean_stroke_precision_ep3']*100)}% | {dt:.1f}s",
            flush=True,
        )
        summary_rows.append(rl_log)

    if len(target_specs) >= 76:
        build_contact_sheets_generic(
            BATCH_6_76_SPECS,
            "batch6",
            [
                (BATCH_6_76_SPECS[:25], "1_25"),
                (BATCH_6_76_SPECS[25:50], "26_50"),
                (BATCH_6_76_SPECS[51:76], "51_76"),
            ],
        )

    (ROOT / "scion_batch6_76_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"[SCION-UniqueWorld-v9] Completed Refresh + Batch-6 (76 new shots) in {(time.time() - t_start)/60.0:.1f} min!", flush=True)


if __name__ == "__main__":
    main()
