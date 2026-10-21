# Secondhand Motion // Full SCION & Universal Argon RL Multiplane Cartoon Studio (70 Shots)

**Turning 50 Million Human Doodles ([Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)) & Archival Film ([Moving Image Archive](https://www.movingimagearchive.com/)) into a Living 7-Plane Cartoon Archive (`70` Complete Multiplane Cartoons · `2,310` Frames · `8` Kinematic & Ontological Regimes).**

---

## Live Studio Surfaces

1. **[`index.html`](index.html) — Main Multiplane Studio & 4-Key Vector Trace Inspector (`70 Shots`)**
   - Interactive 7-plane multiplane projector, 4-keyframe morphological centerline trace inspector, 16-gate quality matrix, Z-score human-stroke typicality analyzer, and 345-category archive index.
2. **[`reel.html`](reel.html) — Continuous 70-Shot Screening Reel, Simultaneous Wall of 70, & What's Next Roadmap**
   - Play all 70 cartoons in order (side-by-side archival footage + live 7-plane vector stage), inspect the simultaneous 70-shot wall across all 8 regimes (`2,310` live vector frames), and explore the 4-phase research roadmap.
3. **[`about.html`](about.html) — Interactive System Architecture & Visual Essay (`70-Shot Walkthrough`)**
   - Deep-dive visual explainer of the 4 inspirations (*Quick, Draw!*, Jordan Mechner's 1989 *Prince of Persia* rotoscoping, Xinyue's 2018 *Scribbling Speech*, and *Secondhand Motion*), the 7-stage zero-hardcoding pipeline, the 4-episode Argon RL reward math, and the 8 kinematic/ontological regimes.

---

## The 7-Stage Universal Zero-Hardcoding Pipeline (`tools/universal_argon_rl_engine.py` & `tools/scion_batch50_engine.py`)

```
Archival Clip (33 frames) + Category Name
  │
  ├─► Stage 1: Zero-Shot GroundingDINO + Depth-Anything-V2 + Optical Flow
  │            + Viterbi Multi-Candidate Trajectory Linker + Waterline Filter
  │
  ├─► Stage 2: Box + Interior Positive/Negative Point-Prompted SAM Pass
  │            (33/33 Temporally Coherent Foreground Silhouettes, Mean IoU 0.90–0.98)
  │
  ├─► Stage 3: Zero-Shot Quick, Draw! Prototype Selection (Top-25 Recognized Drawings)
  │            Scored on Aspect Ratio + Structural Contour Alignment
  │
  ├─► Stage 4: 4 Chronological SCION Rotoscope-Doodle Keyframes (selected-keys.png)
  │            + Character-Free Unique Scene Environment Plate (scene-plate.png)
  │
  ├─► Stage 5: Morphological 1px Centerline Skeletonization + Chaikin Curve Smoothing
  │            (Mean Recall >= 0.995 Across All 280 Keyframes)
  │
  ├─► Stage 6: 4-Episode Argon RL Smooth 2x2 Cage + 16-Ray Boundary-Anchored Warp Policy
  │            Maximizes R(S, M_t, I_t) = 0.35 SilIoU + 0.35 EdgeRecall + 0.30 InMask - 0.12 Reg
  │
  └─► Stage 7: 7-Plane Multiplane Depth Stratification (z in [0.85 .. 10.0])
               with Watertight Gouache SAM Cel Backing on Plane 2 (0% Background Boil) + Foley Audio
```

---

## Coverage Across All 70 Full SCION Multiplane Cartoons & 8 Regimes

- **Regime I — Terrestrial Predators, Ungulates, Hoppers & Primates (18 Shots):** `giraffe-walks-past`, `dog`, `horse`, `rabbit`, `elephant`, `lion`, `tiger`, `bear`, `zebra`, `camel`, `kangaroo`, `cow`, `sheep`, `pig`, `squirrel`, `monkey`, `panda`, `raccoon`
- **Regime II — Wheeled, Tracked & Rail Machines (10 Shots):** `bicycle`, `tractor`, `train`, `car`, `bus`, `firetruck`, `ambulance`, `motorbike`, `bulldozer`, `roller-coaster`
- **Regime III — Nautical Craft, Waterfowl & Marine Life (14 Shots):** `sailboat`, `canoe`, `speedboat`, `aircraft-carrier`, `cruise-ship`, `duck`, `swan`, `penguin`, `flamingo`, `shark`, `whale`, `dolphin`, `sea-turtle`, `crab`
- **Regime IV — Rotary, Avian & Aerial Flight (8 Shots):** `helicopter`, `windmill`, `hot-air-balloon`, `airplane`, `parachute`, `flying-saucer`, `owl`, `parrot`
- **Regime V — Amorphous Natural Phenomena (3 Shots):** `campfire`, `tornado`, `ocean`
- **Regime VI — Musical Performance & Instruments (5 Shots):** `guitar`, `piano`, `violin`, `cello`, `trumpet`
- **Regime VII — Everyday Tabletop & Optical Mechanisms (2 Shots):** `clock`, `camera`
- **Regime VIII — Monumental Architecture, Invertebrates & Flora (10 Shots):** `lighthouse`, `bridge`, `castle`, `church`, `barn`, `butterfly`, `frog`, `snail`, `spider`, `flower`

---

## What's Next: Research & Engineering Roadmap

1. **Amortized Joint V-JEPA / QuickDraw Neural Policy ($\pi_\theta$):** Distill our 70 converged Argon RL trajectories (`2,310 frames`) into a single-pass neural stroke-deformation policy that maps `(SAM masks, optical flow, Depth-Anything-V2, QuickDraw vector tokens)` directly to smooth cage deformations across all 1,380 Moving Image Archive clips.
2. **Sub-Part Articulated SAM-2 Decomposition:** Decompose Plane 2 into independent semantic sub-masks (limbs, wheels, wings, paddles, clock hands, flame tongues) with internal depth ordering (`z = 1.95..2.05`).
3. **Residual-Triggered Generative Breakdowns:** Automatically invoke the keyframe subagent + centerline skeletonizer on any in-between frame where the Argon RL red/blue residual error map shows `SilIoU < 0.70`.
4. **Operable Multiplane World (*Scribbling Speech*):** Compose multiple trained archival doodle actors onto a shared multiplane stage directed by live speech or text.
