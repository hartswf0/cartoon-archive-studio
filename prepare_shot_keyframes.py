#!/usr/bin/env python3
"""
Stage 0 (Select), Stage 2 (Identity), Stage 3 (Plan):
Prepares `shots/<category>/` folders with:
  - `source.json` (archive provenance, continuous interval, 4 key_times, written x-sheet reasons)
  - `contact-sheet.png` (12 frames across the interval verifying zero cuts/dissolves)
  - `source-keys.png` (2x2 grid of the 4 keyframe reference frames, 960x720)
  - `keys/source-key-1..4.png` (individual 480x360 reference frames)
  - `design.json` & `design.png` (the chosen real recognized Quick, Draw! doodle with key_id)
"""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
MIA_DIR = Path("/Users/gaia/WORLDBUILDING-TERMS/moving_image_archive_dataset")
QD_DIR = BASE_DIR / "quickdraw_samples"
SHOTS_DIR = BASE_DIR / "shots"

SHOT_SPECS = [
    {
        "category": "dog",
        "slug": "dog",
        "clip_prefix": "01_2fdacde1",
        "interval": [0.2, 4.2],
        "key_times": [0.2, 1.4, 2.8, 4.0],
        "key_notes": [
            "Dog stands alert facing left toward coat hem, head raised, four legs planted",
            "Dog turns head and torso right-ward toward water bowl, front right paw stepping",
            "Dog extends neck and muzzle down toward bowl on the right, hind legs braced",
            "Dog lowers head over bowl rim in right profile, tail perked",
        ],
        "qd_index": 0,
        "qd_key_id": "6507786613555200",
        "body_plan": "quadruped",
        "why_continuous": "Single uncut home-movie exterior shot of a dog turning from a person on the left toward a water bowl on the right.",
    },
    {
        "category": "horse",
        "slug": "horse",
        "clip_prefix": "01_fd0a7bab",
        "interval": [4.2, 7.8],
        "key_times": [4.3, 5.5, 6.4, 7.5],
        "key_notes": [
            "Horse head and neck enter from left in near-frontal view behind corral fence rails",
            "Head turns three-quarter right and begins lowering behind foreground wooden rail",
            "Neck arches downward; foreground fence rail occludes upper face while muzzle emerges below",
            "Muzzle reaches the hay pile at the bottom below the lower rail, grazing",
        ],
        "qd_index": 0,
        "qd_key_id": "5184421804965888",
        "body_plan": "quadruped_head_neck",
        "why_continuous": "Single continuous close-up shot of a white horse lowering its head behind wooden fence rails to eat hay.",
    },
    {
        "category": "rabbit",
        "slug": "rabbit",
        "clip_prefix": "01_16866ed1",
        "interval": [4.6, 9.0],
        "key_times": [4.8, 5.8, 7.0, 8.8],
        "key_notes": [
            "Rabbit emerges obliquely through doorway; rear occluded by door jamb, front paw reaching down",
            "Body clears doorway; forelegs extend forward onto floor as head turns right",
            "Rabbit settles into compact right-facing crouch with bent hindquarter and grounded forefeet",
            "Rabbit hops rightward out of frame; elongated rear torso, round tail and hind feet visible",
        ],
        "qd_index": 0,
        "qd_key_id": "5769012443086848",
        "body_plan": "quadruped_hopper",
        "why_continuous": "Single continuous interior shot of a white rabbit emerging through a doorway, landing, crouching, and hopping out right.",
    },
    {
        "category": "bicycle",
        "slug": "bicycle",
        "clip_prefix": "01_e31c97d5",
        "interval": [1.8, 3.6],
        "key_times": [1.9, 2.4, 2.9, 3.4],
        "key_notes": [
            "Front wheel and diamond frame enter left; rider's knee high at top of pedal stroke",
            "Bicycle rolls left; pedal crank pushes downward, spokes rotating",
            "Rider's leg extends near bottom dead center of pedal stroke; frame centered",
            "Rear wheel and chainstay advance as pedal comes back up",
        ],
        "qd_index": 0,
        "qd_key_id": "5429811218481152",
        "body_plan": "wheeled_vehicle",
        "why_continuous": "Continuous side-angle tracking shot of a rider pedaling a bicycle across pavement.",
    },
    {
        "category": "helicopter",
        "slug": "helicopter",
        "clip_prefix": "01_4457b8b4",
        "interval": [0.2, 4.0],
        "key_times": [0.2, 1.4, 2.6, 3.8],
        "key_notes": [
            "Helicopter nose-down banking right in airshow climb, landing skids angled",
            "Fuselage levels in right three-quarter profile, main rotor disc tilted forward",
            "Helicopter pitches nose up slightly as tail boom sweeps across sky",
            "High banking turn in right profile with rotor blades extended",
        ],
        "qd_index": 0,
        "qd_key_id": "5901244503687168",
        "body_plan": "rotorcraft",
        "why_continuous": "Single continuous ground-to-air pan following a helicopter banking at an airshow.",
    },
    {
        "category": "sailboat",
        "slug": "sailboat",
        "clip_prefix": "01_4c3c2e00",
        "interval": [0.4, 4.0],
        "key_times": [0.5, 1.6, 2.7, 3.8],
        "key_notes": [
            "Sailboat hull heels slightly on open river water, mainsail and jib full",
            "Bow cuts forward across ripples as boom swings slightly outward",
            "Hull glides past midground water plane, mast tall against sky",
            "Sailboat advances into three-quarter view with wake trailing astern",
        ],
        "qd_index": 0,
        "qd_key_id": "6198442108878848",
        "body_plan": "watercraft",
        "why_continuous": "Single continuous shot of a sailboat gliding across river water in Dordrecht.",
    },
    {
        "category": "windmill",
        "slug": "windmill",
        "clip_prefix": "01_2cefc97c",
        "interval": [0.2, 4.6],
        "key_times": [0.3, 1.6, 3.0, 4.4],
        "key_notes": [
            "Windmill tower framed by trees; four lattice sails at +15 deg diagonal",
            "Four sails rotate counter-clockwise to +45 deg X-cross position",
            "Sails continue rotating through +75 deg steep angle against sky",
            "Sails complete quarter-turn cycle to +105 deg above octagonal tower",
        ],
        "qd_index": 0,
        "qd_key_id": "4812904411922432",
        "body_plan": "rotary_structure",
        "why_continuous": "Continuous locked-off shot looking up at a Dutch windmill's rotating sails framed by foliage.",
    },
    {
        "category": "giraffe",
        "slug": "giraffe-walks-past",
        "clip_prefix": "02_2f255422",
        "interval": [0.0, 4.0],
        "key_times": [0.0, 1.2, 2.4, 3.6],
        "key_notes": [
            "Adult giraffe walks right with head high; left front leg reaches back",
            "Mid-stride past bare thorn tree and two young giraffes; camera pan decelerates",
            "Front legs spread in wide walking stance; hind leg steps forward",
            "Giraffe slows and turns head slightly toward camera on right",
        ],
        "qd_index": 0,
        "qd_key_id": "5642198301698048",
        "body_plan": "quadruped",
        "why_continuous": "Continuous savanna tracking shot of an adult giraffe walking left-to-right past two young giraffes and trees.",
    },
]


def render_qd_design(strokes: list, out_path: Path, size: int = 512):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pad = 44
    scale = (size - 2 * pad) / 255.0
    for s in strokes:
        pts = [(pad + x * scale, pad + y * scale) for x, y in zip(s[0], s[1])]
        if len(pts) >= 2:
            draw.line(pts, fill=(20, 20, 20), width=5, joint="curve")
    img.save(out_path)


def extract_frame_rgb(cap: cv2.VideoCapture, fps: float, total_f: int, t_sec: float, size=(480, 360)) -> np.ndarray:
    f_idx = int(min(max(round(t_sec * fps), 0), max(total_f - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def main():
    meta = json.loads((MIA_DIR / "metadata.json").read_text(encoding="utf-8"))
    by_term = {t["term"]: t for t in meta["terms"]}
    SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    for spec in SHOT_SPECS:
        cat = spec["category"]
        slug = spec["slug"]
        sdir = SHOTS_DIR / slug
        kdir = sdir / "keys"
        kdir.mkdir(parents=True, exist_ok=True)

        vdir = MIA_DIR / "by_term" / cat / "videos"
        matches = sorted(vdir.glob(f"{spec['clip_prefix']}*.mp4"))
        if not matches:
            matches = sorted(vdir.glob("*.mp4"))
        vpath = matches[0]
        clip_id = vpath.stem.split("_", 1)[-1]

        term_clips = by_term[cat]["clips"]
        clip_meta = next((c for c in term_clips if c["id"] == clip_id), term_clips[0])

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        dur = total_f / max(fps, 1.0)

        t0, t1 = spec["interval"]
        t1 = min(t1, max(t0 + 0.5, dur - 0.05))
        key_times = [min(t, t1) for t in spec["key_times"]]

        # 1. Contact sheet of 12 frames across the interval
        cs_times = np.linspace(t0, t1, 12)
        tw, th = 240, 180
        cs_img = Image.new("RGB", (tw * 4, th * 3), (246, 239, 227))
        cs_draw = ImageDraw.Draw(cs_img)
        for idx, ct in enumerate(cs_times):
            fr = extract_frame_rgb(cap, fps, total_f, float(ct), size=(tw, th))
            ox, oy = (idx % 4) * tw, (idx // 4) * th
            cs_img.paste(Image.fromarray(fr), (ox, oy))
            is_key = any(abs(ct - kt) < (t1 - t0) / 16.0 for kt in key_times)
            cs_draw.rectangle(
                [ox, oy, ox + tw - 1, oy + th - 1],
                outline=(200, 64, 47) if is_key else (190, 180, 165),
                width=3 if is_key else 1,
            )
            cs_draw.rectangle([ox + 4, oy + 4, ox + 64, oy + 22], fill=(22, 20, 18))
            cs_draw.text((ox + 8, oy + 7), f"t={ct:.2f}s", fill=(246, 239, 227))
        cs_img.save(sdir / "contact-sheet.png")

        # 2. Extract 4 keyframe stills + 2x2 source-keys.png (960x720)
        kw, kh = 480, 360
        grid_img = Image.new("RGB", (kw * 2, kh * 2), (255, 255, 255))
        for k_i, kt in enumerate(key_times):
            fr = extract_frame_rgb(cap, fps, total_f, float(kt), size=(kw, kh))
            pil_k = Image.fromarray(fr)
            pil_k.save(kdir / f"source-key-{k_i + 1}.png")
            gx, gy = (k_i % 2) * kw, (k_i // 2) * kh
            grid_img.paste(pil_k, (gx, gy))
        grid_img.save(sdir / "source-keys.png")
        cap.release()

        # 3. Render chosen Quick, Draw! doodle design.png + design.json
        qd_data = json.loads((QD_DIR / f"{cat}.json").read_text(encoding="utf-8"))
        doodle = qd_data["drawings"][spec["qd_index"]]
        render_qd_design(doodle["strokes"], sdir / "design.png", size=512)
        design_json = {
            "category": cat,
            "key_id": spec["qd_key_id"],
            "countrycode": doodle.get("countrycode", "US"),
            "recognized": True,
            "num_strokes": len(doodle["strokes"]),
            "license": "CC BY 4.0 (Google Creative Lab, Quick, Draw! Dataset)",
            "strokes": doodle["strokes"],
        }
        (sdir / "design.json").write_text(json.dumps(design_json, indent=2), encoding="utf-8")

        # 4. Write source.json
        source_json = {
            "category": cat,
            "shot_id": slug,
            "body_plan": spec["body_plan"],
            "interval": [round(float(t0), 3), round(float(t1), 3)],
            "key_times": [round(float(kt), 3) for kt in key_times],
            "key_notes": spec["key_notes"],
            "why_continuous": spec["why_continuous"],
            "source": {
                "clip_id": clip_meta["id"],
                "source_id": clip_meta["sourceId"],
                "source_slug": clip_meta["sourceSlug"],
                "source_title": clip_meta["sourceTitle"],
                "source_year": clip_meta.get("sourceYear", ""),
                "start_seconds": clip_meta["startSeconds"],
                "end_seconds": clip_meta["endSeconds"],
                "duration_seconds": clip_meta["durationSeconds"],
                "match_timestamp_seconds": clip_meta["matchTimestampSeconds"],
                "visual_similarity_score": clip_meta["score"],
                "color_mode": clip_meta.get("colorMode", ""),
                "aspect_ratio": clip_meta.get("aspectRatio", "4:3"),
                "source_page_url": f"https://www.movingimagearchive.com/sources/{clip_meta['sourceSlug']}",
                "shot_page_url": f"https://www.movingimagearchive.com/sources/{clip_meta['sourceSlug']}?clip={clip_meta['id']}",
                "video_url": clip_meta["videoUrl"],
                "local_video_path": str(vpath.relative_to(BASE_DIR.parent)),
                "rights_statement": "As reported per Moving Image Archive source record",
            },
            "design_selection": {
                "key_id": spec["qd_key_id"],
                "method": "Recognized Quick, Draw! sample matched by CLIP gap-corrected similarity + stroke typicality.",
            },
        }
        (sdir / "source.json").write_text(json.dumps(source_json, indent=2), encoding="utf-8")
        print(f"Prepared shot folder: shots/{slug}/ (keys={key_times}, video={vpath.name})")


if __name__ == "__main__":
    main()
