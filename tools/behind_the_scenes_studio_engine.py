#!/usr/bin/env python3
"""
tools/behind_the_scenes_studio_engine.py
========================================
Generates the Behind-the-Scenes Multiplane Animation Studio Documentary Film +
18 Bespoke Studio Process Films ("make for each"):
  - Part I (10 Virtual Agent Office & Kanban Worktree Architectures):
      01. AgentSystemLabs/agent-office (3D Multiplayer Office + Multi-Floor Repo Worktrees + MCP Hire/Dismiss)
      02. bill10/agent-007 (Pixel Office + Billion Manager Agent + To-Do -> In-Progress -> Review Kanban Worktrees)
      03. W17ant/Claude-Office (Isometric Pixel Office + Specialist Debugger/Reviewer/Frontend Routing + Token/Payroll Meters)
      04. KalebKE/PixelOffice (Event-Driven 2D Office: Walk to Whiteboards in Plan Mode & Celebrate Passing 16/16 Gates)
      05. pixel-agents-hq/pixel-agents (Canvas 2D Pathfinding & State-Machine Transcript Watcher for Claude/Codex/Gemini)
      06. khanh020905/my-office (3D Office: Walk to Archive Library When Reading Footage & Type at Multiplane Desks)
      07. proactive-agent/workbench (Kanban + Worktrees + AI Semantic Merge & Multi-Account Rate-Limit Swapping)
      08. langwatch/kanban-code (Native Kanban + Worktree + tmux Terminal + Pushover Mobile Review Alerts)
      09. BloopAI/vibe-kanban (10-Agent Multi-Harness Kanban with Inline Diff Comments & Live Multiplane Previews)
      10. NikiforovAll/claude-code-kanban (Localhost:3541 Hook-Driven Kanban + 55-Min Prompt-Cache Auto-Compaction)
  - Part II (8 AI Video-Editing & Motion-Director Skills):
      11. alchaincyf/huashu-motion (Beat-Synced Launch Film, Animated Data Charts & Timeline Review Studio)
      12. rossark0/rossark-motion-skill (60 FPS Single-Scene-Clock SVG Mascot & Character Arcs via MOTION.md)
      13. talsiach/take-one (Strategist -> Creative Director -> Storyboard -> Independent Critic Multi-Pass Film)
      14. NiazMorshed2007/reanimate (Frame-by-Frame Reverse-Engineering & Iterative Pixel-Diff Matching Loop)
      15. shreyaskarnik/argo (Playwright Browser Recording with Cursor Highlights, Callout Overlays & Voiceover Sync)
      16. vincentsch/explainroo (Narrated Explainer Walkthrough with Kokoro TTS, Whisper Word-Level Subtitles & Thumbnails)
      17. win4r/video-director (Pre-Build check_timeline.py Validation + Post-Render qa_video.py Text/Flash/Loudness QA)
      18. Leonxlnx/earshot (CPU Audio Perception + 252-Sound SFX Bank Synced to Spatiotemporal Worm Jerk Spikes)
"""

import json
import math
import os
import shutil
import subprocess
import wave
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
BTS_DIR = ROOT / "bts_films"
PREVIEW_DIR = ROOT / "preview"
DATA_DIR = ROOT / "data"
SHOTS_DIR = ROOT / "shots"
ARTIFACT_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

BTS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Named Studio Agents (with persistent souls, colors, and roles)
STUDIO_AGENTS = [
    {"name": "Maestro", "role": "Orchestrator & Kanban Lead", "color": (45, 95, 225), "hat": "crown"},
    {"name": "Auditor (Codex)", "role": "Cross-Model 16-Gate & CLIP Critic", "color": (55, 165, 95), "hat": "visor"},
    {"name": "3D Office Eng", "role": "Multiplane & Spatiotemporal Worm Architect", "color": (205, 115, 45), "hat": "hardhat"},
    {"name": "Look-Lane Artist", "role": "4-Key Quick, Draw! Rotoscope Animator", "color": (165, 75, 195), "hat": "beret"},
    {"name": "Earshot Foley", "role": "252-SFX Jerk-Spike Audio Syncer", "color": (40, 175, 215), "hat": "headphones"},
    {"name": "Merge Guard", "role": "FIFO Worktree Queue & 55m Cache Compactor", "color": (95, 95, 105), "hat": "cap"},
]

AGENT_OFFICE_SPECS = [
    {
        "id": "office-01-agent-office",
        "num": 1,
        "group": "office",
        "repo": "AgentSystemLabs/agent-office",
        "url": "https://github.com/AgentSystemLabs/agent-office",
        "category": "3D Office + Worktrees (Closest Kohnny Match)",
        "install": "curl -fsSL https://raw.githubusercontent.com/AgentSystemLabs/agent-office/main/install.sh | bash",
        "title": "3D Multi-Floor Studio & Worktree Desks",
        "featured_shot": "giraffe-walks-past",
        "blurb": "Full 3D multiplayer studio where each repo batch is a building floor; Maestro hires Claude Code & Codex animators at isolated Git worktree desks via the agent-office MCP server while corkboard PR queues track all 346 shots.",
        "stages": ["Floor 1: Archive Survey", "Desk Worktree: 4-Key Draw", "Corkboard PR: 16/16 Audit", "Coffee Lounge & MCP Hire"],
    },
    {
        "id": "office-02-agent-007",
        "num": 2,
        "group": "office",
        "repo": "bill10/agent-007",
        "url": "https://github.com/bill10/agent-007",
        "category": "Kanban Worktrees + Pixel Office + Manager Agent",
        "install": "git clone https://github.com/bill10/agent-007",
        "title": "Billion Manager Agent & Kanban Worktrees",
        "featured_shot": "train",
        "blurb": "Combines a pixel animation studio, web terminals, and a To-Do -> In-Progress -> Review Kanban board where Billion (the always-on Manager Agent) spawns a Git worktree per shot, audits diffs, and merges PRs.",
        "stages": ["Billion Plans Lot Card", "Spawn git worktree shots/train", "Review 7-Plane Diff", "Auto-Merge to main"],
    },
    {
        "id": "office-03-claude-office",
        "num": 3,
        "group": "office",
        "repo": "W17ant/Claude-Office",
        "url": "https://github.com/W17ant/Claude-Office",
        "category": "Isometric Pixel Office + Task Board",
        "install": "git clone https://github.com/W17ant/Claude-Office.git && cd Claude-Office && npm i && npm run install-hooks",
        "title": "Isometric Studio, Specialist Routing & Token Meters",
        "featured_shot": "lighthouse",
        "blurb": "Real-time isometric studio driven by hooks + WebSocket events with a live task whiteboard, specialist routing (SAM Debugger, CLIP Reviewer, Multiplane Frontend), permission-approval cards, and token/payroll meters.",
        "stages": ["Hook Event: PreToolUse", "Route to SAM/CLIP Specialist", "Approve Gouache Cel Card", "Payroll & Token Meter Sync"],
    },
    {
        "id": "office-04-pixel-office",
        "num": 4,
        "group": "office",
        "repo": "KalebKE/PixelOffice",
        "url": "https://github.com/KalebKE/PixelOffice",
        "category": "2D Pixel Office (Claude Code + Codex)",
        "install": "git clone https://github.com/KalebKE/PixelOffice && cd PixelOffice && python3 hooks/manage_hooks.py install && ./start.sh",
        "title": "Whiteboard Plan Mode & 16-Gate Celebration",
        "featured_shot": "guitar",
        "blurb": "Event-driven desktop pixel studio supporting both Claude Code and Codex hooks: animators walk to the whiteboard during Stage 2–4 planning, code at multiplane desks, and throw confetti when all 16/16 gates pass.",
        "stages": ["Walk to Whiteboard (Plan Mode)", "Code 4-Cycle BG Redraw at Desk", "Run 16-Gate Test Battery", "Studio Celebration & Lounge"],
    },
    {
        "id": "office-05-pixel-agents",
        "num": 5,
        "group": "office",
        "repo": "pixel-agents-hq/pixel-agents",
        "url": "https://github.com/pixel-agents-hq/pixel-agents",
        "category": "Canvas 2D Pixel Office",
        "install": "git clone https://github.com/pixel-agents-hq/pixel-agents",
        "title": "A* Pathfinding & Transcript State Machine",
        "featured_shot": "tractor",
        "blurb": "Lightweight Canvas 2D virtual studio with A* grid pathfinding and character state machines that watches Claude Code, Codex, and Gemini JSONL transcripts live without modifying the underlying CLI.",
        "stages": ["Tail transcript.jsonl", "FSM State: READING_ARCHIVE", "A* Path to Multiplane Camera", "FSM State: RENDERING_MP4"],
    },
    {
        "id": "office-06-my-office",
        "num": 6,
        "group": "office",
        "repo": "khanh020905/my-office",
        "url": "https://github.com/khanh020905/my-office",
        "category": "3D Claude Code Office",
        "install": "git clone https://github.com/khanh020905/my-office",
        "title": "3D Archive Library Walk & Multiplane Desk Typing",
        "featured_shot": "speedboat",
        "blurb": "3D studio visualization where each active session spawns a 3D animator character that walks to the Moving Image Archive bookshelves when reading source footage and types at its multiplane desk when editing shot.json.",
        "stages": ["Spawn 3D Animator Avatar", "Walk to Archive Library Shelf", "Read Quick, Draw! NDJSON", "Type shot.json at Desk"],
    },
    {
        "id": "office-07-workbench",
        "num": 7,
        "group": "office",
        "repo": "proactive-agent/workbench",
        "url": "https://github.com/proactive-agent/workbench",
        "category": "Kanban + Worktrees + Semantic Merge",
        "install": "git clone https://github.com/proactive-agent/workbench",
        "title": "Per-Task Worktree Isolation & AI Semantic Merge",
        "featured_shot": "bicycle",
        "blurb": "Desktop workbench featuring a Kanban board, per-shot Git worktree isolation, AI-powered semantic merge across concurrent batch branches, conflict detection on cartoon_archive_data.js, and automatic rate-limit swapping.",
        "stages": ["Isolate Batch Worktrees", "Triage Model & Account Pool", "Detect AST/JSON Overlap", "AI Semantic Merge to Main"],
    },
    {
        "id": "office-08-kanban-code",
        "num": 8,
        "group": "office",
        "repo": "langwatch/kanban-code",
        "url": "https://github.com/langwatch/kanban-code",
        "category": "Kanban + Worktrees + Mobile Alerts",
        "install": "git clone https://github.com/langwatch/kanban-code",
        "title": "Native Worktree + tmux + Pushover Mobile Review",
        "featured_shot": "carrot",
        "blurb": "Native macOS studio linking each Kanban shot card to a dedicated Git worktree, live tmux terminal, and GitHub PR, firing a Pushover notification to your phone whenever a batch contact sheet is ready for visual review.",
        "stages": ["Card Linked to tmux + Worktree", "Render 4-Pass BG Redraw Sheet", "Fire Pushover Phone Alert", "1-Tap Mobile PR Approval"],
    },
    {
        "id": "office-09-vibe-kanban",
        "num": 9,
        "group": "office",
        "repo": "BloopAI/vibe-kanban",
        "url": "https://github.com/BloopAI/vibe-kanban",
        "category": "Multi-Agent Kanban + Worktrees",
        "install": "npx vibe-kanban",
        "title": "10-Harness Kanban + Inline Diff & Live Stage Preview",
        "featured_shot": "sandwich",
        "blurb": "Web-based Kanban orchestrating 10+ coding harnesses (Claude Code, Codex, Gemini CLI, OpenCode) with per-card Git worktrees, inline diff comments on bespoke_scene_architect.py, and built-in live multiplane previews.",
        "stages": ["Dispatch 10 Parallel Agents", "Preview Live HTML5 Multiplane", "Inline Diff Comment on Cels", "FIFO Merge Queue"],
    },
    {
        "id": "office-10-claude-code-kanban",
        "num": 10,
        "group": "office",
        "repo": "NikiforovAll/claude-code-kanban",
        "url": "https://github.com/NikiforovAll/claude-code-kanban",
        "category": "Local Kanban + Cost/Context & 55m Cache Compactor",
        "install": "npm install -g claude-code-kanban && claude-code-kanban --install && claude-code-kanban --open",
        "title": "Localhost:3541 Hook Board & 55-Min Cache Compaction",
        "featured_shot": "star",
        "blurb": "Standalone browser Kanban board (localhost:3541) that installs hooks to track live agent activity, preview HTML before/after mockups in Planning, and auto-compact with handoff notes at 55 minutes before the 1-hour prompt cache expires.",
        "stages": ["Planning: HTML Before/After", "Doing: 7-Plane Multiplane Code", "55-Min Prompt Cache Compactor", "Handoff Notes -> FIFO Merge"],
    },
]

VIDEO_SKILL_SPECS = [
    {
        "id": "skill-01-huashu-motion",
        "num": 11,
        "group": "skill",
        "timestamp": "00:46",
        "repo": "alchaincyf/huashu-motion",
        "url": "https://github.com/alchaincyf/huashu-motion",
        "category": "Beat-Synced Launch Films & Timeline Review Studio",
        "install": "npx skills add alchaincyf/huashu-motion",
        "title": "Beat-Synced Studio Launch Film & Animated Data Charts",
        "featured_shot": "giraffe-walks-past",
        "blurb": "Open-source alternative to Claude Motion that builds beat-synced launch films, animated CLIP & 16-gate data charts, and captioned showcase reels inside a browser-based timeline review studio.",
        "stages": ["Extract Audio Beat Grid", "Animate 346-Shot Data Chart", "Sync Captions to Beat Cuts", "Timeline Review Studio Export"],
    },
    {
        "id": "skill-02-rossark-motion-skill",
        "num": 12,
        "group": "skill",
        "timestamp": "03:51",
        "repo": "rossark0/rossark-motion-skill",
        "url": "https://github.com/rossark0/rossark-motion-skill",
        "category": "60 FPS Single-Scene-Clock SVG Character Animation",
        "install": "npx skills@latest add rossark0/rossark-motion-skill",
        "title": "60 FPS Single-Scene-Clock SVG Arcs via MOTION.md",
        "featured_shot": "bicycle",
        "blurb": "Animates traced Quick, Draw! SVG strokes, mascots, and multiplane props at 60 FPS using a single deterministic scene clock tau and a declarative MOTION.md choreography ruleset with natural curvature kappa(tau).",
        "stages": ["Parse MOTION.md Ruleset", "Bind Single Scene Clock tau", "Evaluate Cubic Bezier Arcs", "60 FPS Deterministic SVG Render"],
    },
    {
        "id": "skill-03-take-one",
        "num": 13,
        "group": "skill",
        "timestamp": "05:03",
        "repo": "talsiach/take-one",
        "url": "https://github.com/talsiach/take-one",
        "category": "Strategist -> Creative Director -> Storyboard -> Critic",
        "install": "git clone https://github.com/talsiach/take-one ~/.claude/skills/take-one",
        "title": "4-Role Creative Director & Independent Critic Pipeline",
        "featured_shot": "lighthouse",
        "blurb": "Turns a single-line archive prompt into a story-driven multiplane film using four sequential passes: Strategist (thesis), Creative Director (Bouleau 14-line staging), Storyboarder (4 keys), and Independent Reviewer.",
        "stages": ["Pass 1: Strategist Thesis", "Pass 2: Creative Director", "Pass 3: 4-Key Storyboard", "Pass 4: Blind Critic Audit"],
    },
    {
        "id": "skill-04-reanimate",
        "num": 14,
        "group": "skill",
        "timestamp": "06:55",
        "repo": "NiazMorshed2007/reanimate",
        "url": "https://github.com/NiazMorshed2007/reanimate",
        "category": "Frame-by-Frame Reverse Engineering & Iterative Diff Match",
        "install": "npx skills add NiazMorshed2007/reanimate -g",
        "title": "Frame-by-Frame Archive Reverse-Engineering Loop",
        "featured_shot": "train",
        "blurb": "Extracts every frame of an archival MP4 or reference animation, measures pixel/silhouette IoU and optical flow frame by frame, and iteratively refines the multiplane code until the recreated animation matches the original.",
        "stages": ["Decompose 33 Archival Frames", "Measure Silhouette & Flow Diff", "Iterative Pose Refinement", "Converge to IoU >= 0.89"],
    },
    {
        "id": "skill-05-argo",
        "num": 15,
        "group": "skill",
        "timestamp": "09:02",
        "repo": "shreyaskarnik/argo",
        "url": "https://github.com/shreyaskarnik/argo",
        "category": "Playwright Web Studio Recording + Callout Overlays",
        "install": "npm i -D @argo-video/cli",
        "title": "Playwright Multiplane Projector Walkthrough & Callouts",
        "featured_shot": "guitar",
        "blurb": "Drives our interactive Multiplane Projector and 3D Spatiotemporal Worm Studio in Playwright, recording smooth cursor highlights, zoom callout overlays on the 7 depth planes, and scene-synced voiceover.",
        "stages": ["Launch Playwright Studio", "Highlight 7-Plane Toggles", "Render Zoom Callout Overlay", "Sync Scene to Voiceover Track"],
    },
    {
        "id": "skill-06-explainroo",
        "num": 16,
        "group": "skill",
        "timestamp": "09:50",
        "repo": "vincentsch/explainroo",
        "url": "https://github.com/vincentsch/explainroo",
        "category": "Kokoro TTS Narrated Explainer + Whisper Word Timing",
        "install": "git clone https://github.com/vincentsch/explainroo.git",
        "title": "Narrated Behind-the-Scenes Explainer & Word Subtitles",
        "featured_shot": "diamond",
        "blurb": "Generates narrated behind-the-scenes explainer videos of our 13-stage Look + Motion Join pipeline with local TTS voiceover, Whisper word-level subtitle karaoke timing, and automatic chapter thumbnails.",
        "stages": ["Script 13-Stage Narration", "Synthesize Local Voiceover", "Whisper Word-Level Alignment", "Burn Karaoke Subtitles & Thumb"],
    },
    {
        "id": "skill-07-video-director",
        "num": 17,
        "group": "skill",
        "timestamp": "11:20",
        "repo": "win4r/video-director",
        "url": "https://github.com/win4r/video-director",
        "category": "Pre-Build check_timeline.py + Post-Render qa_video.py",
        "install": "git clone https://github.com/win4r/video-director.git && cp -R video-director/video-director ~/.claude/skills/",
        "title": "Pre-Build Timeline Validation & Post-Render Video QA",
        "featured_shot": "carrot",
        "blurb": "Directs code-rendered multiplane videos with pre-build timeline validation (check_timeline.py verifying X-sheet frame continuity) and post-render QA (qa_video.py checking text legibility, zero flash frames, and LUFS loudness).",
        "stages": ["Run check_timeline.py", "Render 7-Plane Canvas to MP4", "Run qa_video.py (Text/Flash)", "Verify Audio Loudness & Export"],
    },
    {
        "id": "skill-08-earshot",
        "num": 18,
        "group": "skill",
        "timestamp": "12:57",
        "repo": "Leonxlnx/earshot",
        "url": "https://github.com/Leonxlnx/earshot",
        "category": "CPU Audio Perception + 252-SFX Bank Synced to Jerk Spikes",
        "install": "git clone https://github.com/Leonxlnx/earshot.git",
        "title": "CPU Audio Perception & 252-SFX Synced to Worm Jerk ||c'''(tau)||",
        "featured_shot": "zigzag",
        "blurb": "Gives our studio agents local CPU audio perception and a 252-sound SFX bank to place footsteps, brush swishes, and impacts at the exact timestamps where the 3D Spatiotemporal Worm's Kinetic Jerk ||c'''(tau)|| spikes.",
        "stages": ["Compute 3D Worm Jerk ||c'''||", "Detect Top-3 Jerk Spike Frames", "Retrieve SFX from 252-Bank", "CPU Audio Critique & Mix"],
    },
]


def draw_agent_avatar(canvas, x, y, agent, frame_idx=0, state="walk", scale=1.0):
    """Draws an expressive hand-drawn + isometric studio agent character with hat, speech/thought indicator, and legs."""
    col = agent["color"]
    bob = int(round(math.sin(frame_idx * 0.55 + x * 0.03) * 3.0 * scale)) if state == "walk" else 0
    cy = y + bob
    r_head = int(round(11 * scale))
    # Shadow on floor
    cv2.ellipse(canvas, (x, y + int(24 * scale)), (int(12 * scale), int(5 * scale)), 0, 0, 360, (190, 182, 168), -1, cv2.LINE_AA)
    # Legs (scissoring when walking)
    stride = int(round(math.sin(frame_idx * 0.65 + y * 0.05) * 7.0 * scale)) if state == "walk" else 0
    cv2.line(canvas, (x - int(4 * scale), cy + int(10 * scale)), (x - int(4 * scale) - stride, y + int(23 * scale)), (25, 25, 25), 2, cv2.LINE_AA)
    cv2.line(canvas, (x + int(4 * scale), cy + int(10 * scale)), (x + int(4 * scale) + stride, y + int(23 * scale)), (25, 25, 25), 2, cv2.LINE_AA)
    # Torso coat
    pts_body = np.array([
        [x - int(9 * scale), cy - int(2 * scale)],
        [x + int(9 * scale), cy - int(2 * scale)],
        [x + int(11 * scale), cy + int(12 * scale)],
        [x - int(11 * scale), cy + int(12 * scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts_body], col, cv2.LINE_AA)
    cv2.polylines(canvas, [pts_body], True, (20, 20, 20), 2, cv2.LINE_AA)
    # Head
    cv2.circle(canvas, (x, cy - int(12 * scale)), r_head, (238, 246, 252), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, cy - int(12 * scale)), r_head, (20, 20, 20), 2, cv2.LINE_AA)
    # Eyes
    cv2.circle(canvas, (x - int(3 * scale), cy - int(13 * scale)), max(1, int(2 * scale)), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + int(4 * scale), cy - int(13 * scale)), max(1, int(2 * scale)), (20, 20, 20), -1, cv2.LINE_AA)
    # Hat badge
    cv2.rectangle(canvas, (x - int(10 * scale), cy - int(24 * scale)), (x + int(10 * scale), cy - int(19 * scale)), col, -1)
    cv2.rectangle(canvas, (x - int(10 * scale), cy - int(24 * scale)), (x + int(10 * scale), cy - int(19 * scale)), (20, 20, 20), 1)
    # Name tag
    cv2.putText(canvas, agent["name"], (x - int(22 * scale), cy - int(28 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, (20, 20, 20), 1, cv2.LINE_AA)


def load_shot_pair(slug: str, frame_idx: int):
    """Loads (archival_frame, cartoon_frame) at frame_idx from shots/<slug>/comparison.mp4."""
    comp_path = SHOTS_DIR / slug / "comparison.mp4"
    if comp_path.exists():
        cap = cv2.VideoCapture(str(comp_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % 33)
        ok, fr = cap.read()
        cap.release()
        if ok and fr is not None:
            h, w = fr.shape[:2]
            return fr[:, :w // 2], fr[:, w // 2:]
    f = np.full((240, 320, 3), 240, dtype=np.uint8)
    return f, f


def generate_foley_wav(wav_path: Path, n_frames: int, fps: int = 8, seed: int = 1):
    """Generates a beat-synced + Earshot Foley WAV soundtrack for a studio film."""
    sr = 22050
    dur = n_frames / float(fps)
    n_samples = int(sr * dur)
    t = np.linspace(0.0, dur, n_samples, endpoint=False)
    rng = np.random.default_rng(seed)

    # Warm studio projector hum + melodic pentatonic marimba/synth arpeggio on every stage beat
    notes = [220.0, 277.18, 329.63, 440.0, 554.37, 659.25]
    sig = 0.03 * np.sin(2.0 * math.pi * 110.0 * t)
    for beat_i in range(int(dur * 2)):
        t0 = beat_i * 0.5
        mask = (t >= t0) & (t < t0 + 0.42)
        freq = notes[(beat_i + seed) % len(notes)]
        env = np.exp(-6.5 * (t[mask] - t0))
        sig[mask] += 0.18 * env * np.sin(2.0 * math.pi * freq * (t[mask] - t0))
        # Subtle footstep/typewriter click transient
        click_mask = (t >= t0) & (t < t0 + 0.035)
        sig[click_mask] += 0.12 * rng.standard_normal(np.sum(click_mask)) * np.exp(-45.0 * (t[click_mask] - t0))

    sig = np.clip(sig, -0.95, 0.95)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def render_studio_film_frame(spec: dict, f_idx: int, total_frames: int = 36):
    """Renders one 640x360 frame of a Behind-the-Scenes Studio Film for `spec`."""
    W, H = 640, 360
    canvas = np.full((H, W, 3), (227, 239, 246), dtype=np.uint8)  # warm paper BGR (#F6EFE3)

    # Subtle isometric floor grid
    for gx in range(0, W, 40):
        cv2.line(canvas, (gx, 52), (gx, H - 36), (212, 224, 232), 1)
    for gy in range(52, H - 36, 32):
        cv2.line(canvas, (0, gy), (W, gy), (212, 224, 232), 1)

    stage_idx = min(3, int((f_idx / float(total_frames)) * 4))
    stage_progress = (f_idx % (total_frames // 4)) / float(max(1, total_frames // 4 - 1))

    # Top Pegbar Header
    cv2.rectangle(canvas, (0, 0), (W, 48), (20, 22, 26), -1)
    tag_col = (55, 155, 240) if spec["group"] == "office" else (95, 195, 115)
    cv2.rectangle(canvas, (10, 8), (165, 26), tag_col, -1)
    group_lbl = f"OFFICE #{spec['num']:02d}" if spec["group"] == "office" else f"VIDEO SKILL #{spec['num']-10:02d} ({spec.get('timestamp','')})"
    cv2.putText(canvas, group_lbl, (16, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (15, 15, 15), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{spec['repo']} // {spec['title']}", (174, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 240, 228), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"CMD: {spec['install'][:78]}", (12, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (175, 210, 235), 1, cv2.LINE_AA)

    # Left Zone (x=12..345): Isometric Virtual Office / Kanban Worktree / Skill Pipeline Floor
    cv2.rectangle(canvas, (12, 56), (348, 314), (244, 250, 254), -1)
    cv2.rectangle(canvas, (12, 56), (348, 314), (25, 25, 25), 2)

    # Draw 4 Kanban / Pipeline Stage Columns at top of Left Zone
    col_w = 80
    for s_i, st_name in enumerate(spec["stages"]):
        sx0 = 18 + s_i * 81
        is_active = (s_i == stage_idx)
        is_done = (s_i < stage_idx)
        bg_c = (195, 235, 205) if is_done else ((190, 225, 252) if is_active else (232, 236, 240))
        cv2.rectangle(canvas, (sx0, 62), (sx0 + col_w - 3, 108), bg_c, -1)
        cv2.rectangle(canvas, (sx0, 62), (sx0 + col_w - 3, 108), (25, 25, 25), 2 if is_active else 1)
        cv2.putText(canvas, f"STAGE {s_i+1}", (sx0 + 5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (30, 30, 30), 1, cv2.LINE_AA)
        words = st_name.split()
        line1 = " ".join(words[:2])[:13]
        line2 = " ".join(words[2:])[:13]
        cv2.putText(canvas, line1, (sx0 + 4, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (15, 15, 15), 1, cv2.LINE_AA)
        if line2:
            cv2.putText(canvas, line2, (sx0 + 4, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (55, 55, 55), 1, cv2.LINE_AA)

    # Draw Studio Desks, Archive Bookshelf, Whiteboard & Coffee Machine on the Isometric Floor
    # Bookshelf (Archive Library)
    cv2.rectangle(canvas, (22, 125), (82, 185), (155, 185, 215), -1)
    cv2.rectangle(canvas, (22, 125), (82, 185), (25, 25, 25), 2)
    cv2.putText(canvas, "ARCHIVE", (27, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "345 CATS", (27, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (20, 20, 20), 1, cv2.LINE_AA)

    # Whiteboard (Plan & 14-Line Armature)
    cv2.rectangle(canvas, (102, 122), (222, 175), (255, 255, 255), -1)
    cv2.rectangle(canvas, (102, 122), (222, 175), (25, 25, 25), 2)
    cv2.line(canvas, (102, 122), (222, 175), (180, 215, 240), 1, cv2.LINE_AA)
    cv2.line(canvas, (102, 175), (222, 122), (180, 215, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "WHITEBOARD // PLAN", (108, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"worktree: shots/{spec['featured_shot']}", (108, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (40, 105, 65), 1, cv2.LINE_AA)
    cv2.putText(canvas, "55m Cache Timer: OK", (108, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (45, 75, 185), 1, cv2.LINE_AA)

    # FIFO Merge & QA Gate Station
    cv2.rectangle(canvas, (242, 122), (338, 175), (225, 242, 228), -1)
    cv2.rectangle(canvas, (242, 122), (338, 175), (25, 25, 25), 2)
    cv2.putText(canvas, "FIFO MERGE", (252, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 75, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "16/16 GATES", (252, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (20, 75, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "CLIP + WORM", (252, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (50, 50, 50), 1, cv2.LINE_AA)

    # Two Worktree Desks in foreground of Left Zone
    for d_i, (dx, dy, d_lbl) in enumerate([(45, 245, "DESK A: CLAUDE WORKTREE"), (195, 245, "DESK B: CODEX AUDITOR")]):
        cv2.rectangle(canvas, (dx, dy), (dx + 125, dy + 42), (185, 208, 228), -1)
        cv2.rectangle(canvas, (dx, dy), (dx + 125, dy + 42), (25, 25, 25), 2)
        cv2.putText(canvas, d_lbl, (dx + 6, dy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (20, 20, 20), 1, cv2.LINE_AA)

    # Animate 4 Studio Agents walking between stations according to f_idx
    waypoints = [
        (55, 205),   # Archive Library
        (160, 205),  # Whiteboard
        (105, 240),  # Desk A
        (260, 240),  # Desk B / FIFO Merge
    ]
    for a_i in range(4):
        ag = STUDIO_AGENTS[(spec["num"] + a_i) % len(STUDIO_AGENTS)]
        wp_start = waypoints[(a_i + stage_idx) % 4]
        wp_end = waypoints[(a_i + stage_idx + 1) % 4]
        ax = int(round((1.0 - stage_progress) * wp_start[0] + stage_progress * wp_end[0]))
        ay = int(round((1.0 - stage_progress) * wp_start[1] + stage_progress * wp_end[1]))
        draw_agent_avatar(canvas, ax, ay, ag, frame_idx=f_idx + a_i * 5, state="walk", scale=0.82)

    # Right Zone (x=358..628): Live Multiplane Stage + Spatiotemporal Worm Prism + Telemetry
    arch_fr, cart_fr = load_shot_pair(spec["featured_shot"], f_idx)
    arch_thumb = cv2.resize(arch_fr, (128, 96))
    cart_thumb = cv2.resize(cart_fr, (128, 96))

    cv2.rectangle(canvas, (358, 56), (628, 314), (248, 251, 254), -1)
    cv2.rectangle(canvas, (358, 56), (628, 314), (25, 25, 25), 2)
    cv2.putText(
        canvas, f"LIVE WORKTREE OUTPUT // {spec['featured_shot'].upper()}",
        (368, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA
    )

    canvas[82:178, 366:494] = arch_thumb
    cv2.rectangle(canvas, (366, 82), (494, 178), (25, 25, 25), 1)
    cv2.putText(canvas, "Archival Source", (370, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (255, 255, 255), 1, cv2.LINE_AA)

    canvas[82:178, 496:624] = cart_thumb
    cv2.rectangle(canvas, (496, 82), (624, 178), (25, 25, 25), 1)
    cv2.putText(canvas, "7-Plane Cartoon", (500, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (20, 80, 40), 1, cv2.LINE_AA)

    # Mini Spatiotemporal Worm & Earshot Jerk / Timeline Meter in bottom-right (y=186..306)
    cv2.rectangle(canvas, (366, 186), (620, 306), (22, 25, 32), -1)
    cv2.rectangle(canvas, (366, 186), (620, 306), (70, 80, 100), 1)
    cv2.putText(
        canvas, "3D SPATIOTEMPORAL WORM V(u,v,tau) & EARSHOT JERK",
        (374, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (115, 220, 255), 1, cv2.LINE_AA
    )

    # Draw animated worm wave & playhead
    wave_pts = []
    for t_i in range(33):
        wx = 376 + int(round((t_i / 32.0) * 234))
        wy = 248 + int(round(math.sin(t_i * 0.32 + spec["num"]) * 18.0))
        wave_pts.append([wx, wy])
    cv2.polylines(canvas, [np.array(wave_pts, dtype=np.int32)], False, (75, 235, 130), 2, cv2.LINE_AA)
    cur_t = f_idx % 33
    px_cur, py_cur = wave_pts[cur_t]
    cv2.line(canvas, (px_cur, 210), (px_cur, 282), (80, 180, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, (px_cur, py_cur), 6, (45, 85, 245), -1, cv2.LINE_AA)
    cv2.circle(canvas, (px_cur, py_cur), 6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"Active Stage [{stage_idx+1}/4]: {spec['stages'][stage_idx]}",
        (374, 297), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (235, 240, 220), 1, cv2.LINE_AA
    )

    # Bottom Subtitle / Explainer Bar
    cv2.rectangle(canvas, (0, H - 36), (W, H), (20, 22, 26), -1)
    sub_text = spec["blurb"]
    # Scroll or slice subtitle cleanly across stages
    chunk_len = 92
    start_c = min(max(0, len(sub_text) - chunk_len), (stage_idx * 32))
    sub_slice = sub_text[start_c:start_c + chunk_len]
    cv2.putText(
        canvas, f"[{spec['repo']}] {sub_slice}",
        (14, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (245, 238, 215), 1, cv2.LINE_AA
    )

    return canvas


def build_single_bts_film(spec: dict):
    """Builds MP4 (with Foley audio), animated GIF, and poster PNG for one Behind-the-Scenes Studio Film."""
    slug = spec["id"]
    total_frames = 36
    fps = 8
    frames_bgr = [render_studio_film_frame(spec, f_i, total_frames) for f_i in range(total_frames)]

    poster_path = BTS_DIR / f"{slug}_poster.png"
    cv2.imwrite(str(poster_path), frames_bgr[10])

    gif_path = BTS_DIR / f"{slug}.gif"
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr[::2]]
    pil_frames[0].save(
        str(gif_path), save_all=True, append_images=pil_frames[1:], duration=240, loop=0, optimize=True
    )

    raw_mp4 = BTS_DIR / f"_{slug}_raw.mp4"
    wav_path = BTS_DIR / f"_{slug}.wav"
    final_mp4 = BTS_DIR / f"{slug}.mp4"

    H, W = frames_bgr[0].shape[:2]
    vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames_bgr:
        vw.write(f)
    vw.release()

    generate_foley_wav(wav_path, total_frames, fps=fps, seed=spec["num"])

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(raw_mp4), "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-c:a", "aac", "-b:a", "96k", "-shortest",
        str(final_mp4)
    ], check=False)

    if raw_mp4.exists():
        raw_mp4.unlink()
    if wav_path.exists():
        wav_path.unlink()

    return {
        **spec,
        "mp4": f"bts_films/{slug}.mp4",
        "gif": f"bts_films/{slug}.gif",
        "poster": f"bts_films/{slug}_poster.png",
        "poster_bgr": frames_bgr[10],
    }


def render_contact_sheet(results, title, subtitle, out_name, cols=2):
    rows = math.ceil(len(results) / cols)
    cw, ch = 640, 360
    header_h = 68
    canvas = np.full((header_h + rows * ch, cols * cw, 3), (20, 22, 26), dtype=np.uint8)
    cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 235, 210), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 210, 195), 1, cv2.LINE_AA)

    for idx, r in enumerate(results):
        ri = idx // cols
        ci = idx % cols
        x0 = ci * cw
        y0 = header_h + ri * ch
        canvas[y0:y0 + ch, x0:x0 + cw] = r["poster_bgr"]
        cv2.rectangle(canvas, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (80, 90, 105), 2)

    out_path = PREVIEW_DIR / out_name
    cv2.imwrite(str(out_path), canvas)
    shutil.copy2(out_path, ARTIFACT_DIR / out_name)
    return out_path


def main():
    all_specs = AGENT_OFFICE_SPECS + VIDEO_SKILL_SPECS
    results = []
    for sp in all_specs:
        res = build_single_bts_film(sp)
        results.append(res)
        print(f"  Built BTS Studio Film [{sp['num']:02d}/18]: {sp['id']} ({sp['repo']})")

    office_res = [r for r in results if r["group"] == "office"]
    skill_res = [r for r in results if r["group"] == "skill"]

    p_off = render_contact_sheet(
        office_res,
        "BEHIND THE SCENES PART I // 10 VIRTUAL AGENT OFFICE & KANBAN WORKTREE STUDIO FILMS",
        "Kohnny Architecture + 10 Open-Source Multi-Agent Worktree Studios Building the 346 Quick, Draw! Multiplane Cartoons",
        "bts_10_agent_offices_atlas.png",
        cols=2,
    )
    p_sk = render_contact_sheet(
        skill_res,
        "BEHIND THE SCENES PART II // 8 AI VIDEO DIRECTOR & MOTION SKILL STUDIO FILMS",
        "All 8 Video Editing & Motion Skills (huashu-motion, rossark, take-one, reanimate, argo, explainroo, video-director, earshot)",
        "bts_8_video_skills_atlas.png",
        cols=2,
    )
    print("Rendered BTS contact sheets:", p_off.name, p_sk.name)

    # Concatenate all 18 films into the Master Behind-the-Scenes Documentary Film
    concat_txt = BTS_DIR / "_concat_list.txt"
    concat_txt.write_text("\n".join(f"file '{ROOT / r['mp4']}'" for r in results) + "\n")
    master_mp4 = ROOT / "Behind_The_Scenes_Multiplane_Studio_Film.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-c", "copy", str(master_mp4)
    ], check=False)
    if concat_txt.exists():
        concat_txt.unlink()
    print(f"Built Master Behind-the-Scenes Film: {master_mp4.name} ({master_mp4.stat().st_size / 1e6:.2f} MB)")

    clean_results = [{k: v for k, v in r.items() if k != "poster_bgr"} for r in results]
    payload = {
        "total_films": len(clean_results),
        "master_film": "Behind_The_Scenes_Multiplane_Studio_Film.mp4",
        "agents": STUDIO_AGENTS,
        "office_films": [r for r in clean_results if r["group"] == "office"],
        "skill_films": [r for r in clean_results if r["group"] == "skill"],
    }
    (DATA_DIR / "bts_studio_films.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "bts_studio_data.js").write_text("window.BTS_STUDIO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
