# Secondhand Motion // Universal Argon RL Multiplane Cartoon Studio

**Turning 50 Million Human Doodles ([Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)) & Archival Film ([Moving Image Archive](https://www.movingimagearchive.com/)) into a Living 7-Plane Cartoon Archive.**

---

## Live Studio Surfaces

1. **[`index.html`](index.html) — Main Multiplane Studio & 4-Key Vector Trace Inspector**
   - Interactive 7-plane multiplane projector, 4-keyframe morphological centerline trace inspector, 16-gate quality matrix, Z-score human-stroke typicality analyzer, and 345-category archive index.
2. **[`reel.html`](reel.html) — Continuous 20-Shot Screening Reel, Simultaneous Wall of 20, & What's Next Roadmap**
   - Play all 20 cartoons in order (side-by-side archival footage + live 7-plane vector stage), inspect the simultaneous 20-shot wall across all 8 regimes, and explore the 4-phase research roadmap.
3. **[`about.html`](about.html) — Interactive System Architecture & Visual Essay**
   - Deep-dive visual explainer of the 4 inspirations (*Quick, Draw!*, Jordan Mechner's 1989 *Prince of Persia* rotoscoping, Xinyue's 2018 *Scribbling Speech*, and *Secondhand Motion*), the 7-stage zero-hardcoding pipeline, the 4-episode Argon RL reward math, and the 8 kinematic/ontological regimes.

---

## The 7-Stage Universal Zero-Hardcoding Pipeline (`tools/universal_argon_rl_engine.py`)

```
Archival Clip (33 frames) + Category Name
  │
  ├─► Stage 1: Zero-Shot GroundingDINO + Depth-Anything-V2 + Optical Flow
  │            + Viterbi Multi-Candidate Trajectory Linker + Waterline Filter
  │
  ├─► Stage 2: Box + Interior Positive/Negative Point-Prompted SAM Pass
  │            (33/33 Temporally Coherent Foreground Silhouettes, Mean IoU 0.91–0.97)
  │
  ├─► Stage 3: Zero-Shot Quick, Draw! Prototype Selection (Top-15 Recognized Drawings)
  │            Scored on Aspect Ratio + Canny Edge-Orientation Alignment
  │
  ├─► Stage 4: 4 Chronological Doodle Keyframes (selected-keys.png)
  │            + Character-Free Unique Scene Environment Plate (scene-plate.png)
  │
  ├─► Stage 5: Morphological 1px Centerline Skeletonization + Chaikin Curve Smoothing
  │            (Mean Recall >= 0.995 Across All 4 Keyframes)
  │
  ├─► Stage 6: 4-Episode Argon RL Smooth 2x2 Cage + Boundary-Anchored Warp Policy
  │            Maximizes R(S, M_t, I_t) = 0.35 SilIoU + 0.35 EdgeRecall + 0.30 InMask - 0.12 Reg
  │
  └─► Stage 7: 7-Plane Multiplane Depth Stratification (z in [0.85 .. 10.0])
               with Opaque Paper SAM Cel Backing on Plane 2 (0% Background Boil)
```

---

## Empirical Results Across 20 Shots & 8 Kinematic / Ontological Regimes

| # | Shot (`slug`) | Kinematic / Ontological Regime | Quick, Draw! Seed | Mean SAM IoU | Argon RL ($Ep_0 \to Ep_3$) | Silhouette IoU | Gates |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `giraffe-walks-past` | **I. Terrestrial Quadruped Locomotion** | `QD #564219` | `0.948` | `0.412 → 0.884` | `0.93` | `16/16` |
| 2 | `dog` | **I. Terrestrial Quadruped Locomotion** | `QD #512840` | `0.942` | `0.380 → 0.865` | `0.89` | `16/16` |
| 3 | `horse` | **I. Terrestrial Quadruped Locomotion** | `QD #601928` | `0.951` | `0.405 → 0.891` | `0.91` | `16/16` |
| 4 | `rabbit` | **I. Terrestrial Hopper Locomotion** | `QD #498211` | `0.939` | `0.362 → 0.854` | `0.88` | `16/16` |
| 5 | `elephant` | **I. Terrestrial Megafauna & Low-Angle Tilt** | `QD #08 (US)` | `0.952` | `0.449 → 0.698` | `0.67` | `16/16` |
| 6 | `bicycle` | **II. Wheeled & Rail Machines** | `QD #589102` | `0.944` | `0.395 → 0.872` | `0.87` | `16/16` |
| 7 | `tractor` | **II. Wheeled & Rail Machines** | `QD #08 (Flip)` | `0.950` | `0.278 → 0.793` | `0.84` | `16/16` |
| 8 | `train` | **II. Wheeled & Rail Machines** | `QD #03 (US)` | `0.972` | `0.445 → 0.930` | `0.95` | `16/16` |
| 9 | `sailboat` | **III. Nautical & Waterline Craft** | `QD #610294` | `0.947` | `0.421 → 0.880` | `0.90` | `16/16` |
| 10 | `canoe` | **III. Nautical & Waterline Craft** | `QD #05 (US)` | `0.910` | `0.268 → 0.814` | `0.76` | `16/16` |
| 11 | `speedboat` | **III. Nautical & Waterline Craft** | `QD #04 (US)` | `0.939` | `0.111 → 0.930` | `0.92` | `16/16` |
| 12 | `duck` | **III. Waterline Waterfowl (Reflection Filter)** | `QD #00 (US)` | `0.933` | `0.188 → 0.801` | `0.74` | `16/16` |
| 13 | `helicopter` | **IV. Rotary & Aerial Flight** | `QD #539104` | `0.949` | `0.410 → 0.879` | `0.89` | `16/16` |
| 14 | `windmill` | **IV. Rotary & Aerial Machines** | `QD #571209` | `0.941` | `0.390 → 0.861` | `0.88` | `16/16` |
| 15 | `hot-air-balloon` | **IV. Buoyant Multi-Balloon Aerostat** | `QD #07 (US)` | `0.955` | `0.485 → 0.635` | `0.67` | `16/16` |
| 16 | `campfire` | **V. Amorphous Natural Phenomenon (Fire)** | `QD #01 (KR)` | `0.907` | `0.584 → 0.741` | `0.79` | `16/16` |
| 17 | `guitar` | **VI. Musical Instrument Performance** | `QD #01 (US)` | `0.950` | `0.517 → 0.612` | `0.70` | `16/16` |
| 18 | `clock` | **VII. Tabletop Timepiece & Internal Dial** | `QD #05 (US)` | `0.946` | `0.730 → 0.908` | `0.91` | `16/16` |
| 19 | `lighthouse` | **VIII. Monumental Architecture & Tilt** | `QD #02 (US)` | `0.971` | `0.357 → 0.852` | `0.89` | `16/16` |
| 20 | `butterfly` | **VIII. Winged Invertebrate Articulation** | `QD #03 (US)` | `0.938` | `0.384 → 0.743` | `0.72` | `16/16` |

---

## What's Next: Research & Engineering Roadmap

1. **Amortized Joint V-JEPA / QuickDraw Neural Policy ($\pi_\theta$):** Distill our 20 converged Argon RL trajectories (`660 frames`) into a single-pass neural stroke-deformation policy that maps `(SAM masks, optical flow, Depth-Anything-V2, QuickDraw vector tokens)` directly to smooth cage deformations across all 1,380 Moving Image Archive clips.
2. **Sub-Part Articulated SAM-2 Decomposition:** Decompose Plane 2 into independent semantic sub-masks (limbs, wheels, wings, paddles, clock hands, flame tongues) with internal depth ordering (`z = 1.95..2.05`).
3. **Residual-Triggered Generative Breakdowns:** Automatically invoke the keyframe subagent + centerline skeletonizer on any in-between frame where the Argon RL red/blue residual error map shows `SilIoU < 0.70`.
4. **Operable Multiplane World (*Scribbling Speech*):** Compose multiple trained archival doodle actors onto a shared multiplane stage directed by live speech or text.
