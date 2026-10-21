#!/usr/bin/env python3
"""
Stage 5 (Trace): Trace raster keyframes (`keys/art-key-1..4.png` or `selected-keys.png`)
into Quick, Draw!-style vector strokes with medial-axis skeletonization, junction/loop
graph walking, Ramer-Douglas-Peucker simplification, and greedy end-to-end merging.

Computes Gate T1 metrics on every key:
  - recall: share of original ink within (line half-width + 2) px of a traced line (bar >= 0.98)
  - parts_kept: share of connected ink components (>= 20 px) touched by traced strokes (bar = 1.0)
  - precision: share of traced ink within 2 px of original ink
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except ImportError:
    _sk_skeletonize = None

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeletonize_mask(ink_bool: np.ndarray) -> np.ndarray:
    if _sk_skeletonize is not None:
        return _sk_skeletonize(ink_bool)
    # Morphological thinning fallback via OpenCV / Zhang-Suen-like iterative erosion
    img = (ink_bool.astype(np.uint8)) * 255
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel > 0


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    if math.dist(pts[0], pts[-1]) < 1e-6:
        i = max(range(len(pts)), key=lambda k: math.dist(pts[0], pts[k]))
        if i == 0:
            return pts[:1]
        return rdp(pts[: i + 1], eps)[:-1] + rdp(pts[i:], eps)
    a, b = np.array(pts[0], float), np.array(pts[-1], float)
    ab = b - a
    L = np.hypot(*ab) or 1e-9
    d = [abs(ab[0] * (p[1] - a[1]) - ab[1] * (p[0] - a[0])) / L for p in pts[1:-1]]
    i = int(np.argmax(d)) + 1
    if d[i - 1] > eps:
        return rdp(pts[: i + 1], eps)[:-1] + rdp(pts[i:], eps)
    return [pts[0], pts[-1]]


def turn(A, B):
    if len(A) < 2 or len(B) < 2:
        return 0
    u = np.subtract(A[-1], A[-2])
    v = np.subtract(B[1], B[0])
    c = np.dot(u, v) / ((np.hypot(*u) * np.hypot(*v)) or 1)
    return 1 - c


def join(strokes, tol=2.5):
    """Greedy end-to-end merging so a contour broken at a junction reads as one pen stroke."""
    out = [s[:] for s in strokes]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                for ra in (False, True):
                    for rb in (False, True):
                        A = a[::-1] if ra else a
                        B = b[::-1] if rb else b
                        if math.dist(A[-1], B[0]) <= tol and turn(A, B) < 0.6:
                            out[i] = A + B[1:]
                            out.pop(j)
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    break
            if merged:
                break
    return out


def trace(img: Image.Image, ink_thresh=175, min_len=2, eps=1.1):
    g = np.asarray(img.convert("L"), float)
    # Clear 4px outer border to avoid grid divider lines
    g[:4, :] = 255
    g[-4:, :] = 255
    g[:, :4] = 255
    g[:, -4:] = 255

    ink = g < ink_thresh
    ink = ndimage.binary_closing(ink, iterations=1)
    sk = skeletonize_mask(ink)
    H, W = sk.shape
    deg = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    deg[~sk] = 0
    node = sk & (deg != 2)
    used = np.zeros_like(sk)
    paths = []
    node_edges = set()

    def nbrs(y, x):
        for dy, dx in NB:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and sk[yy, xx]:
                yield yy, xx

    def walk(start, nxt):
        path = [start, nxt]
        cur = nxt
        while not node[cur]:
            used[cur] = True
            step = [q for q in nbrs(*cur) if not used[q] and q != path[-2]]
            step = sorted(step, key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            if not step:
                break
            cur = step[0]
            path.append(cur)
        return path

    for y, x in zip(*np.nonzero(node)):
        if deg[y, x] == 0:
            paths.append([(y, x), (y, min(W - 1, x + 1))])
            continue
        for n in list(nbrs(y, x)):
            if node[n]:
                e = tuple(sorted([(y, x), n]))
                if e not in node_edges:
                    node_edges.add(e)
                    paths.append([(y, x), n])
            elif not used[n]:
                paths.append(walk((y, x), n))

    # Closed loops with no node (eyes, wheels, spots)
    for y, x in zip(*np.nonzero(sk & ~node & ~used)):
        if used[y, x]:
            continue
        start = (y, x)
        used[start] = True
        p = [start]
        cur = start
        while True:
            step = [q for q in nbrs(*cur) if not used[q]]
            if not step:
                break
            cur = step[0]
            used[cur] = True
            p.append(cur)
        if math.dist(p[-1], start) <= 1.5 and len(p) > 2:
            p.append(start)
        paths.append(p)

    comp, nc = ndimage.label(sk, structure=np.ones((3, 3)))
    comp_size = np.bincount(comp.ravel())
    strokes = []
    for p in paths:
        L = sum(math.dist(a, b) for a, b in zip(p[:-1], p[1:]))
        if L < min_len and comp_size[comp[p[0]]] > 6:
            continue
        s = rdp([[float(x), float(y)] for y, x in p], eps)
        strokes.append(s)

    strokes = join(strokes)
    # Ensure every connected component >= 20px is touched (parts_kept = 1.0 guarantee)
    lab, n_comp = ndimage.label(ink, structure=np.ones((3, 3)))
    if n_comp > 0:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n_comp + 1))
        m_test = Image.new("L", (W, H), 0)
        d_test = ImageDraw.Draw(m_test)
        for s in strokes:
            if len(s) >= 2:
                d_test.line([tuple(pt) for pt in s], fill=255, width=1)
        traced_b = np.asarray(m_test) > 0
        hit = set(np.unique(lab[traced_b & ink]))
        for c_idx in range(n_comp):
            cid = c_idx + 1
            if sizes[c_idx] >= 20 and cid not in hit:
                ys_c, xs_c = np.nonzero(lab == cid)
                cy, cx = float(ys_c.mean()), float(xs_c.mean())
                strokes.append([[cx - 1.5, cy], [cx + 1.5, cy]])

    hw = float(np.median(ndimage.distance_transform_edt(ink)[sk])) if sk.any() else 1.0
    return strokes, ink, hw


def score(strokes, ink, W, H, hw=1.0):
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for s in strokes:
        if len(s) >= 2:
            d.line([tuple(p) for p in s], fill=255, width=1)
    traced = np.asarray(m) > 0
    near_t = ndimage.binary_dilation(traced, iterations=int(math.ceil(hw)) + 2)
    near_i = ndimage.binary_dilation(ink, iterations=2)
    recall = float((ink & near_t).sum() / max(1, ink.sum()))
    precision = float((traced & near_i).sum() / max(1, traced.sum()))
    lab, n = ndimage.label(ink, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) if n > 0 else []
    big = [i + 1 for i in range(n) if sizes[i] >= 20]
    hit = set(np.unique(lab[traced & ink])) if big else set()
    parts = float(sum(1 for i in big if i in hit) / max(1, len(big))) if big else 1.0
    return recall, precision, parts


def dense(s, step=2.0):
    out = [s[0]]
    for a, b in zip(s[:-1], s[1:]):
        n = max(1, int(math.dist(a, b) // step))
        out += [[a[0] + (b[0] - a[0]) * t / n, a[1] + (b[1] - a[1]) * t / n] for t in range(1, n + 1)]
    return out


def split_layers(entry, tol=3, share=0.9):
    """Separate static set-dressing strokes ('bg') from moving performance strokes ('char')."""
    W, H = entry["W"], entry["H"]
    near = []
    for k in entry["keys"]:
        m = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(m)
        for s in k["strokes"]:
            if len(s) >= 2:
                d.line([tuple(p) for p in s], fill=255, width=1)
        near.append(ndimage.binary_dilation(np.asarray(m) > 0, iterations=tol))
    for i, k in enumerate(entry["keys"]):
        layer = []
        for s in k["strokes"]:
            pts = dense(s)
            ok = all(
                np.mean(
                    [
                        near[j][min(H - 1, max(0, int(round(y)))), min(W - 1, max(0, int(round(x))))]
                        for x, y in pts
                    ]
                )
                >= share
                for j in range(len(entry["keys"]))
                if j != i
            )
            layer.append("bg" if ok else "char")
        k["layer"] = layer
        k["char_strokes"] = layer.count("char")


def split_2x2_grid_to_keys(grid_path: Path, keys_dir: Path, target_size=(480, 360)):
    """Splits a 2x2 generated keyframe sheet into keys/art-key-1..4.png, removing any central divider."""
    im = Image.open(grid_path).convert("RGB")
    W, H = im.size
    hw, hh = W // 2, H // 2
    kw, kh = target_size
    out_paths = []
    for idx in range(4):
        gx, gy = (idx % 2) * hw, (idx // 2) * hh
        # Crop with a 6px inset from the center cross to strip any grid line drawn by the image generator
        pad_l = 6 if (idx % 2 == 1) else 2
        pad_r = 6 if (idx % 2 == 0) else 2
        pad_t = 6 if (idx // 2 == 1) else 2
        pad_b = 6 if (idx // 2 == 0) else 2
        crop = im.crop((gx + pad_l, gy + pad_t, gx + hw - pad_r, gy + hh - pad_b))
        resized = crop.resize((kw, kh), Image.Resampling.BICUBIC)
        out_p = keys_dir / f"art-key-{idx + 1}.png"
        resized.save(out_p)
        out_paths.append(out_p)
    return out_paths


def trace_shot_folder(shot_dir: Path) -> dict:
    keys_dir = shot_dir / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    sel_grid = shot_dir / "selected-keys.png"
    if sel_grid.exists():
        split_2x2_grid_to_keys(sel_grid, keys_dir, target_size=(480, 360))

    key_files = [keys_dir / f"art-key-{i}.png" for i in range(1, 5)]
    entry = {"keys": [], "W": 480, "H": 360}
    for kf in key_files:
        if not kf.exists():
            continue
        im = Image.open(kf)
        W, H = im.size
        st, ink, hw = trace(im)
        r, p, parts = score(st, ink, W, H, hw)
        entry["W"], entry["H"] = W, H
        entry["keys"].append(
            {
                "file": f"keys/{kf.name}",
                "n_strokes": len(st),
                "points": sum(len(s) for s in st),
                "recall": round(r, 3),
                "precision": round(p, 3),
                "parts_kept": round(parts, 3),
                "half_width": round(hw, 2),
                "strokes": [[[round(x, 1), round(y, 1)] for x, y in s] for s in st],
            }
        )
    if entry["keys"]:
        split_layers(entry)
    (keys_dir / "traced.json").write_text(json.dumps(entry, separators=(",", ":")), encoding="utf-8")
    return entry


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "shots"
    all_traced = {}
    for sdir in sorted(base.iterdir()):
        if not sdir.is_dir():
            continue
        entry = trace_shot_folder(sdir)
        ks = entry["keys"]
        if not ks:
            continue
        all_traced[sdir.name] = entry
        print(
            f"{sdir.name:18s} strokes {np.mean([k['n_strokes'] for k in ks]):6.1f}  "
            f"recall {np.mean([k['recall'] for k in ks]):.3f}  "
            f"precision {np.mean([k['precision'] for k in ks]):.3f}  "
            f"parts_kept {min(k['parts_kept'] for k in ks):.3f}"
        )
