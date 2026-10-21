#!/usr/bin/env python3
"""
Stage 4b & 5b: Scene Environment Plate Installer & Skeleton Vectorizer (`trace_scene_plates.py`).
Takes each shot's generated `scene-plate.png`, cleans & resizes it to 480x360, runs the
Secondhand Motion skeleton-graph tracer (`trace_keys.trace`) to produce `scene-traced.json`,
and decomposes the traced environment strokes into multiplane depth layers (`sky`, `far`, `mid`,
`secondary`, `fore`) tailored to that specific archival film clip.
"""
import json
import math
from pathlib import Path
from PIL import Image
import numpy as np

from trace_keys import trace

W_STAGE, H_STAGE = 480, 360

STUDIO_DIR = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
SHOTS_DIR = STUDIO_DIR / "shots"


def stroke_bbox_and_center(s: list) -> tuple:
    pts = np.array(s, dtype=float)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = pts.mean(0)
    return float(x0), float(y0), float(x1), float(y1), float(cx), float(cy)


def assign_stroke_plane(slug: str, s: list) -> str:
    """
    Assigns each traced vector stroke from `scene-plate.png` to its true multiplane depth layer
    (`sky`, `far`, `mid`, `secondary`, `fore`) based on the specific shot's archival layout.
    """
    x0, y0, x1, y1, cx, cy = stroke_bbox_and_center(s)

    if slug == "giraffe-walks-past":
        if cy < 85:
            return "sky"
        if cy > 295:
            return "fore"
        if 210 <= cx <= 355 and 180 <= cy <= 295:
            return "secondary"  # juvenile giraffes & central brush
        if cy < 175 and (cx < 95 or (265 < cx < 335)):
            return "far"
        return "mid"  # bare thorn tree & right bush tree

    if slug == "dog":
        if cy < 95:
            return "far"  # winter shrub & distant path ball
        if cx > 280 and cy > 255:
            return "secondary"  # metal washbasin water bowl at bottom-right
        if cx < 130 and cy < 275:
            return "mid"  # person's coat & legs on left
        if cx > 355 and cy < 290:
            return "mid"  # cobblestone wall on right
        if cy > 290:
            return "fore"  # foreground gravel/pebbles
        return "mid"

    if slug == "horse":
        if cy > 275 and cx > 180:
            return "fore"  # foreground hay mound & bottom fence rail in front of horse muzzle
        if cy > 285:
            return "fore"
        if cx > 390:
            return "mid"  # right vertical timber post
        if cy < 70:
            return "far"  # barn roofline & top rail
        return "mid"  # corral fence rails

    if slug == "rabbit":
        # Indoor room doorway: NO sky!
        if cy > 295:
            return "fore"  # foreground floorboard threshold
        if cy > 245:
            return "secondary"  # interior wooden floorboards
        if cx < 185:
            return "mid"  # left white painted clapboard doorway wall & jamb
        return "far"  # recessed open doorway frame & interior hall on right

    if slug == "bicycle":
        if cx < 105 and cy < 295:
            return "secondary"  # companion lead bicycle rear wheel on left edge
        if cy < 155:
            return "far"  # street-side bush & upper wall
        if cy < 225:
            return "mid"  # sidewalk curb line
        return "fore"  # foreground road surface texture

    if slug == "helicopter":
        if cy > 295:
            return "far"  # distant low airfield tree line & hangar at bottom of frame
        if cy < 135:
            return "sky"  # high cumulus clouds
        if cy < 235:
            return "mid"  # mid-altitude cloud wisps
        return "fore"  # rotor slipstream streaks

    if slug == "sailboat":
        if cy < 155:
            return "sky"  # upper harbor sky
        if cy < 228:
            return "far"  # steam tugboat & smoke on left horizon + distant ships
        if cy > 298:
            return "fore"  # foreground harbor wave ripples
        if cx > 190 and cy > 235:
            return "secondary"  # shimmering hull reflection waves
        return "mid"  # mid-harbor swells

    if slug == "windmill":
        if cx < 135 or cx > 315 or cy > 285:
            return "fore"  # framing pine/cypress tree branches on left, right, and bottom
        if cy < 140:
            return "sky"  # high windswept cloud streaks
        if cy > 235:
            return "far"  # distant polder horizon
        return "mid"

    return "mid"


def trace_all_scene_plates():
    print("Tracing generated scene-plate.png environment drawings across all 8 shots...")
    for sdir in sorted(SHOTS_DIR.iterdir()):
        if not sdir.is_dir():
            continue
        slug = sdir.name
        plate_path = sdir / "scene-plate.png"
        if not plate_path.exists():
            print(f"  [WARN] {slug}: scene-plate.png not found yet")
            continue

        # Ensure clean 480x360 RGB PNG on white ground
        im = Image.open(plate_path).convert("RGB").resize((W_STAGE, H_STAGE), Image.Resampling.LANCZOS)
        im.save(plate_path)

        from trace_keys import score
        strokes, ink, hw = trace(im, eps=1.5)
        recall, precision, parts = score(strokes, ink, W_STAGE, H_STAGE, hw)
        strokes = [[[round(float(x), 1), round(float(y), 1)] for x, y in s] for s in strokes if len(s) >= 2]
        plane_buckets = {"sky": [], "far": [], "mid": [], "secondary": [], "fore": []}
        for s in strokes:
            pl = assign_stroke_plane(slug, s)
            plane_buckets[pl].append(s)

        out = {
            "shot_id": slug,
            "W": W_STAGE,
            "H": H_STAGE,
            "file": "scene-plate.png",
            "n_strokes": len(strokes),
            "points": int(sum(len(s) for s in strokes)),
            "recall": round(float(recall), 4),
            "precision": round(float(precision), 4),
            "parts_kept": round(float(parts), 4),
            "strokes": strokes,
            "planes": plane_buckets,
        }
        (sdir / "scene-traced.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
        counts_str = ", ".join(f"{k}:{len(v)}" for k, v in plane_buckets.items())
        print(
            f"  [SCENE TRACE] {slug:18s} | {len(strokes):3d} strokes | recall={recall:.3f} parts={parts:.3f} | planes=({counts_str})"
        )


if __name__ == "__main__":
    trace_all_scene_plates()
