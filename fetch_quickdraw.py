#!/usr/bin/env python3
"""
Fetches real Quick, Draw! simplified vector drawings (.ndjson) from Google Cloud Storage
for all 345 terms using HTTP Range requests, selects the cleanest recognized vector drawings
per category, and saves them to `quickdraw_samples/` for joint CLIP + V-JEPA 2 + SAM alignment.
"""

import json
import math
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
QD_DIR = BASE_DIR / "quickdraw_samples"
MIA_DIR = Path("/Users/gaia/WORLDBUILDING-TERMS/moving_image_archive_dataset")

TERMS = [
    line.strip()
    for line in (MIA_DIR / "terms.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def slugify(term: str) -> str:
    s = term.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def score_drawing_quality(d: dict) -> float:
    """
    Scores a QuickDraw vector drawing by recognition status, stroke richness,
    aspect balance, and point density so we pick iconic, well-drawn examples
    rather than 1-stroke scribbles.
    """
    strokes = d.get("drawing", [])
    if not strokes:
        return -1e9
    n_strokes = len(strokes)
    n_points = sum(len(s[0]) for s in strokes if len(s) >= 2)
    if n_points < 8:
        return -1e6

    xs = [x for s in strokes for x in s[0]]
    ys = [y for s in strokes for y in s[1]]
    if not xs or not ys:
        return -1e6
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w < 40 and h < 40:
        return -1e5

    rec_bonus = 100.0 if d.get("recognized", False) else 0.0
    # Ideal stroke count is ~3 to 14 strokes, ideal point count ~25 to 120 points
    stroke_score = -abs(n_strokes - 6.5) * 2.0
    point_score = -abs(n_points - 55.0) * 0.25
    aspect_balance = min(w, h) / max(w, h, 1.0) * 15.0
    return rec_bonus + stroke_score + point_score + aspect_balance


def fetch_term_quickdraw(idx: int, term: str, top_k: int = 12) -> dict:
    slug = slugify(term)
    out_file = QD_DIR / f"{slug}.json"
    if out_file.exists():
        try:
            cached = json.loads(out_file.read_text(encoding="utf-8"))
            if cached.get("drawings"):
                return cached
        except Exception:
            pass

    url = (
        "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
        + urllib.parse.quote(term)
        + ".ndjson"
    )
    last_err = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Range": "bytes=0-98303",  # First 96 KB (~200 drawings)
                    "User-Agent": USER_AGENT,
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                chunk = resp.read().decode("utf-8", errors="ignore")

            lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
            # Drop the last line as it may be truncated by the byte range
            if len(lines) > 1:
                lines = lines[:-1]

            parsed = []
            for ln in lines:
                try:
                    obj = json.loads(ln)
                    if "drawing" in obj:
                        obj["_quality"] = round(score_drawing_quality(obj), 3)
                        parsed.append(obj)
                except Exception:
                    continue

            if not parsed:
                raise ValueError(f"No valid drawings parsed for {term}")

            parsed.sort(key=lambda x: x["_quality"], reverse=True)
            selected = []
            for rank, item in enumerate(parsed[:top_k], start=1):
                strokes = item["drawing"]
                selected.append(
                    {
                        "rank": rank,
                        "key_id": item.get("key_id"),
                        "word": item.get("word", term),
                        "countrycode": item.get("countrycode"),
                        "recognized": item.get("recognized", True),
                        "quality_score": item["_quality"],
                        "num_strokes": len(strokes),
                        "num_points": sum(len(s[0]) for s in strokes),
                        "strokes": strokes,
                    }
                )

            payload = {
                "term_index": idx,
                "term": term,
                "term_slug": slug,
                "source_url": url,
                "total_sampled": len(parsed),
                "recognized_rate": round(
                    sum(1 for p in parsed if p.get("recognized")) / len(parsed), 4
                ),
                "drawings": selected,
            }
            out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"Failed QuickDraw fetch for {term!r}: {last_err}")


def main():
    QD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching QuickDraw vector drawings for {len(TERMS)} terms...", flush=True)
    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {
            pool.submit(fetch_term_quickdraw, idx + 1, t): t
            for idx, t in enumerate(TERMS)
        }
        done = 0
        for fut in as_completed(futs):
            t = futs[fut]
            done += 1
            res = fut.result()
            results[t] = res
            if done % 50 == 0 or done == len(TERMS):
                print(
                    f"  [QuickDraw {done}/{len(TERMS)}] Last: {t!r} "
                    f"(rec_rate={res['recognized_rate']:.2f}) - {time.time()-t0:.1f}s",
                    flush=True,
                )

    summary = {
        "total_terms": len(TERMS),
        "drawings_per_term": 12,
        "total_drawings": sum(len(r["drawings"]) for r in results.values()),
        "avg_recognized_rate": round(
            sum(r["recognized_rate"] for r in results.values()) / len(results), 4
        ),
    }
    (BASE_DIR / "quickdraw_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("QuickDraw fetch complete:", summary, flush=True)


if __name__ == "__main__":
    main()
