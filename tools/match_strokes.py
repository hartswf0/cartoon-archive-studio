#!/usr/bin/env python3
"""
Stage 5 -> 7 (Match Strokes): In-between traced keys by stroke correspondence.

For each pair of consecutive keys, every stroke is resampled to N=16 points along its length
and matched to a stroke in the next key via Hungarian assignment (`linear_sum_assignment`)
on mean point distance (forward or reversed). Keys are aligned by Chamfer translation (`cKDTree`).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

N = 16


def resample(s, n=N):
    a = np.array(s, float)
    if len(a) == 1:
        a = np.vstack([a, a])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(a, axis=0).T))]
    if d[-1] == 0:
        return np.repeat(a[:1], n, 0)
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, a[:, 0]), np.interp(t, d, a[:, 1])]


def length(a):
    return float(np.hypot(*np.diff(a, axis=0).T).sum())


def shift(A, B, W, H):
    """Translation that best lays A's ink onto B's (chamfer distance, coarse grid then fine)."""
    if not A or not B:
        return np.zeros(2)
    pa = np.concatenate(A)
    tb = cKDTree(np.concatenate(B))
    cost = lambda t: float(np.minimum(tb.query(pa + t)[0], 30).mean())
    best = min(
        (
            (cost(np.array([dx, dy])), dx, dy)
            for dx in np.arange(-W / 2, W / 2 + 1, W / 24)
            for dy in np.arange(-H / 3, H / 3 + 1, H / 18)
        )
    )
    _, bx, by = best
    for step in (W / 48, W / 96, 1):
        best = min(
            (
                (cost(np.array([bx + dx, by + dy])), bx + dx, by + dy)
                for dx in (-step, 0, step)
                for dy in (-step, 0, step)
            )
        )
        _, bx, by = best
    return np.array([bx, by]) if cost(np.array([bx, by])) < cost(np.zeros(2)) - 0.5 else np.zeros(2)


def match(A, B, W, H=None):
    """A, B: lists of resampled strokes. Returns [(i, j, reversed)], lengths, and the shift applied to A."""
    if not A or not B:
        return [], np.array([1.0]), np.array([1.0]), np.zeros(2)
    t = shift(A, B, W, H or W * 0.75)
    A_shifted = [a + t for a in A]
    na, nb = len(A_shifted), len(B)
    C = np.full((na, nb), 1e6)
    R = np.zeros((na, nb), bool)
    LA = np.array([length(a) for a in A_shifted])
    LB = np.array([length(b) for b in B])
    for i, a in enumerate(A_shifted):
        for j, b in enumerate(B):
            f = np.hypot(*(a - b).T).mean()
            r = np.hypot(*(a - b[::-1]).T).mean()
            ratio = max(LA[i], LB[j], 1) / max(min(LA[i], LB[j]), 1)
            C[i, j] = min(f, r) * (1 + 0.5 * (ratio - 1))
            R[i, j] = r < f
    rows, cols = linear_sum_assignment(C)
    limit = 0.14 * W
    return [(int(i), int(j), bool(R[i, j])) for i, j in zip(rows, cols) if C[i, j] < limit], LA, LB, t


def match_shot_folder(shot_dir: Path) -> dict:
    traced_p = shot_dir / "keys" / "traced.json"
    src_p = shot_dir / "source.json"
    if not traced_p.exists() or not src_p.exists():
        return {}
    e = json.loads(traced_p.read_text(encoding="utf-8"))
    src = json.loads(src_p.read_text(encoding="utf-8"))
    keys = [[resample(s) for s in k["strokes"]] for k in e["keys"]]
    pairs, share, shifts = [], [], []
    for a in range(len(keys) - 1):
        m, LA, LB, t = match(keys[a], keys[a + 1], e["W"], e["H"])
        denom = float(LA.sum() + LB.sum()) or 1.0
        ink = (sum(LA[i] for i, _, _ in m) + sum(LB[j] for _, j, _ in m)) / denom
        pairs.append(m)
        share.append(round(float(ink), 3))
        shifts.append([round(float(t[0]), 2), round(float(t[1]), 2)])
    res = {
        "W": e["W"],
        "H": e["H"],
        "fps": 12,
        "interval": src["interval"],
        "key_times": src["key_times"],
        "keys": [[[[round(x, 1), round(y, 1)] for x, y in s] for s in k] for k in keys],
        "matches": pairs,
        "matched_ink": share,
        "shifts": shifts,
    }
    (shot_dir / "keys" / "matches.json").write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8")
    return res


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "shots"
    for sdir in sorted(base.iterdir()):
        if sdir.is_dir():
            r = match_shot_folder(sdir)
            if r:
                print(f"{sdir.name:18s} matched ink per key pair: {r['matched_ink']}")
