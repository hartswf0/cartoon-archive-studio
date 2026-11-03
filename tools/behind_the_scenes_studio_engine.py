#!/usr/bin/env python3
"""
tools/behind_the_scenes_studio_engine.py
========================================
Unified Quick, Draw! Multiplane Animation Studio Soundstage Engine.
Renders:
  - 4 Master Architectural Studio Soundstage Plates (`studio_bg_1..4.jpg` + `studio_bg_5_bullpen_unoccupied.jpg`)
  - 18 Behind-the-Scenes Studio Soundstage Films (10 Virtual Agent Office & Worktree Films + 8 AI Video Director Skill Films)
  - 2 Master Contact Sheets (`preview/bts_10_agent_offices_atlas.png` & `preview/bts_8_video_skills_atlas.png`)
  - Master Documentary Film (`Behind_The_Scenes_Multiplane_Studio_Film.mp4`)

100% UNIFIED IN THE QUICK, DRAW! AESTHETIC:
  - Warm archival cream paper (`#F6EFE3` / `#FFFBF4`), soft pastel gouache washes (`#FBE5C2`, `#D4E8DC`, `#D6E4F0`, `#F7D8CE`),
    and crisp `#161412` hand-drawn Quick, Draw! ink strokes with 4-keyframe living line boil (`K01..K04`) and `3px` hard ink drop-shadows.
  - Every studio room, camera rig, drafting table, lamp, microphone, clock, prop, and animator character is constructed from
    REAL Quick, Draw! vector doodles (`quickdraw_samples/<category>.json`) with watertight gouache occlusion cels (0 X-ray bleeds).
  - Archival source frames are rendered as unified graphite/blueprint pencil traces on `#F6EFE3` paper alongside the 7-plane
    Quick, Draw! multiplane cartoons.
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
QD_DIR = ROOT / "quickdraw_samples"
BTS_DIR = ROOT / "bts_films"
PREVIEW_DIR = ROOT / "preview"
DATA_DIR = ROOT / "data"
SHOTS_DIR = ROOT / "shots"
ARTIFACT_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

BTS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Unified Quick, Draw! Studio Palette (in OpenCV BGR matching CSS variables in index.html)
COL_PAPER       = (227, 239, 246)  # #F6EFE3 warm archival paper
COL_PAPER_DEEP  = (206, 226, 236)  # #ECE2CE deeper paper wash
COL_SHEET       = (244, 251, 255)  # #FFFBF4 bright cream cel sheet
COL_INK         = (18, 20, 22)     # #161412 Quick, Draw! fountain-pen ink
COL_INK_SOFT    = (50, 56, 61)     # #3D3832 soft graphite ink
COL_RULE        = (168, 191, 203)  # #CBBFA8 drafting rule line
COL_TAG         = (58, 159, 239)   # #EF9F3A warm amber tag
COL_TAG_PALE    = (194, 229, 251)  # #FBE5C2 pale amber gouache wash
COL_WAX         = (41, 74, 201)    # #C94A29 red-orange wax seal
COL_WAX_PALE    = (206, 216, 247)  # #F7D8CE pale coral gouache wash
COL_FOREST      = (79, 107, 47)    # #2F6B4F studio forest green
COL_FOREST_PALE = (220, 232, 212)  # #D4E8DC pale sage gouache wash
COL_BLUE        = (132, 91, 43)    # #2B5B84 blueprint slate blue
COL_BLUE_PALE   = (240, 228, 214)  # #D6E4F0 pale sky gouache wash

# Named Quick, Draw! Studio Crew (constructed from real Quick, Draw! doodles!)
STUDIO_AGENTS = [
    {"name": "Maestro", "role": "Studio Director & Kanban Lead", "color": COL_TAG_PALE, "accent": COL_TAG, "hat_qd": "crown", "prop_qd": "megaphone"},
    {"name": "Auditor (Codex)", "role": "16-Gate & CLIP Dailies Critic", "color": COL_FOREST_PALE, "accent": COL_FOREST, "hat_qd": "eyeglasses", "prop_qd": "flashlight"},
    {"name": "Camera Op", "role": "7-Plane Multiplane Rig Operator", "color": COL_BLUE_PALE, "accent": COL_BLUE, "hat_qd": "hat", "prop_qd": "camera"},
    {"name": "Lead Animator", "role": "4-Key Quick, Draw! Cel Inker", "color": COL_WAX_PALE, "accent": COL_WAX, "hat_qd": "bowtie", "prop_qd": "paintbrush"},
    {"name": "Earshot Foley", "role": "252-SFX Jerk-Spike Sound Artist", "color": COL_TAG_PALE, "accent": COL_BLUE, "hat_qd": "headphones", "prop_qd": "microphone"},
    {"name": "Merge Archivist", "role": "Steenbeck Vault & Worktree Keeper", "color": COL_FOREST_PALE, "accent": COL_INK_SOFT, "hat_qd": "hat", "prop_qd": "book"},
]

AGENT_OFFICE_SPECS = [
    {
        "id": "office-01-agent-office",
        "num": 1,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "WING A // QUICK, DRAW! 7-PLANE TOWER & 3-FLOOR STUDIO",
        "repo": "AgentSystemLabs/agent-office",
        "url": "https://github.com/AgentSystemLabs/agent-office",
        "category": "3D Multi-Floor Studio + Worktrees (Closest Kohnny Match)",
        "install": "curl -fsSL https://raw.githubusercontent.com/AgentSystemLabs/agent-office/main/install.sh | bash",
        "title": "3D Multi-Floor Soundstage & Worktree Lightboxes",
        "featured_shot": "giraffe-walks-past",
        "blurb": "Full Quick, Draw! multi-floor animation soundstage where the towering 7-level Multiplane Camera Rig stands beside three cutaway floors: the Steenbeck Archive Vault, the 4-Desk Animator Bullpen, and the Storyboard & Espresso Lounge.",
        "stages": ["Vault: 35mm Archive Trace", "Bullpen: 4-Key Lightbox", "Tower: 7-Plane Exposure", "Lounge: Corkboard PR Merge"],
    },
    {
        "id": "office-02-agent-007",
        "num": 2,
        "group": "office",
        "studio_wing": "bg5_bullpen",
        "wing_name": "WING B // QUICK, DRAW! ANIMATOR BULLPEN & ARMATURE WALL",
        "repo": "bill10/agent-007",
        "url": "https://github.com/bill10/agent-007",
        "category": "Kanban Worktrees + Animator Bullpen + Manager Director",
        "install": "git clone https://github.com/bill10/agent-007",
        "title": "Director Billion & 4-Desk Worktree Bullpen",
        "featured_shot": "train",
        "blurb": "Inside the Quick, Draw! isometric bullpen, Director Billion assigns each shot card from the back storyboard wall to an isolated drafting desk where circular lightboxes trace the 4 keyframes in onion-skin ink.",
        "stages": ["Corkboard: Pin Shot Card", "Lightbox: Spawn Worktree", "Projector: Audit 7-Plane", "Steenbeck: Splice to Main"],
    },
    {
        "id": "office-03-claude-office",
        "num": 3,
        "group": "office",
        "studio_wing": "bg2_occupied",
        "wing_name": "WING B2 // SPECIALIST INK & PAINT DRAFTING BULLPEN",
        "repo": "W17ant/Claude-Office",
        "url": "https://github.com/W17ant/Claude-Office",
        "category": "Isometric Studio + Specialist Routing + Footage Meters",
        "install": "git clone https://github.com/W17ant/Claude-Office.git && cd Claude-Office && npm i && npm run install-hooks",
        "title": "Specialist Animator Routing & Cel Footage Meters",
        "featured_shot": "lighthouse",
        "blurb": "Four specialist Quick, Draw! animators (SAM Silhouette Tracer, CLIP Critic, Multiplane Compositor, Gouache Painter) work under Quick, Draw! floor lamps while hook events route cel sheets between their glowing lightboxes.",
        "stages": ["Hook: PreToolUse Slate", "Route to Cel Specialist", "Approve Gouache Keyframe", "Footage & Token Reel Sync"],
    },
    {
        "id": "office-04-pixel-office",
        "num": 4,
        "group": "office",
        "studio_wing": "bg5_bullpen",
        "wing_name": "WING B3 // STORYBOARD ARMATURE WALL & CELEBRATION STAGE",
        "repo": "KalebKE/PixelOffice",
        "url": "https://github.com/KalebKE/PixelOffice",
        "category": "2D Studio (Claude Code + Codex Storyboard Critique)",
        "install": "git clone https://github.com/KalebKE/PixelOffice && cd PixelOffice && python3 hooks/manage_hooks.py install && ./start.sh",
        "title": "Bouleau Storyboard Pitch & 16-Gate Wrap Party",
        "featured_shot": "guitar",
        "blurb": "Doodle animators gather at the Golden Armature storyboard wall during Stage 2-4 planning, return to their Quick, Draw! lightboxes to ink 4-cycle backgrounds, and celebrate when all 16/16 studio gates pass.",
        "stages": ["Pitch at Storyboard Wall", "Ink 4-Cycle BG at Lightbox", "Screen 16-Gate Dailies", "Studio Wrap Celebration"],
    },
    {
        "id": "office-05-pixel-agents",
        "num": 5,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "WING A2 // SPIRAL STAIRCASE & TRANSCRIPT STATE FLOOR",
        "repo": "pixel-agents-hq/pixel-agents",
        "url": "https://github.com/pixel-agents-hq/pixel-agents",
        "category": "A* Studio Floor Pathfinding & Transcript Watcher",
        "install": "git clone https://github.com/pixel-agents-hq/pixel-agents",
        "title": "Studio Floor Pathfinding & Live X-Sheet State Machine",
        "featured_shot": "tractor",
        "blurb": "Watches live JSONL production transcripts and drives Quick, Draw! animators up and down the studio stairs between the Archive Vault, the 2nd-floor Lightbox Desks, and the 7-Level Multiplane Camera Tower.",
        "stages": ["Tail X-Sheet Transcript", "State: ARCHIVE_STEENBECK", "Climb Studio Stairs to Rig", "State: SHOOTING_7_PLANE"],
    },
    {
        "id": "office-06-my-office",
        "num": 6,
        "group": "office",
        "studio_wing": "bg4_optical",
        "wing_name": "WING D // 35MM ARCHIVE VAULT & OPTICAL PRINTER BENCH",
        "repo": "khanh020905/my-office",
        "url": "https://github.com/khanh020905/my-office",
        "category": "3D Archive Film Vault & Rotoscope Bench",
        "install": "git clone https://github.com/khanh020905/my-office",
        "title": "35mm Film Canister Vault & Rotoscope Lightbox Walk",
        "featured_shot": "speedboat",
        "blurb": "Whenever an agent reads archival Quick, Draw! strokes, the doodle animator walks to the archive bookshelves, threads the reel onto the twin-reel rotoscope bench, and traces the multiplane cel layers.",
        "stages": ["Pull Archive Book & Reel", "Thread Twin-Reel Rotoscope", "Trace Quick, Draw! Strokes", "Expose Multiplane Cel"],
    },
    {
        "id": "office-07-workbench",
        "num": 7,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "WING A3 // MULTI-FLOOR WORKTREE SPLICING & VAULT",
        "repo": "proactive-agent/workbench",
        "url": "https://github.com/proactive-agent/workbench",
        "category": "Kanban + Worktrees + AI Semantic Film Splicing",
        "install": "git clone https://github.com/proactive-agent/workbench",
        "title": "Per-Shot Worktree Isolation & Steenbeck Semantic Splice",
        "featured_shot": "bicycle",
        "blurb": "Isolates concurrent animation batches on separate studio floors, swaps model harnesses when rate-limited, and splices completed 7-plane film strips together on the ground-floor Steenbeck editor.",
        "stages": ["Isolate Floor Worktrees", "Swap Animator Crew Pool", "Check Cel Layer Overlap", "Steenbeck Semantic Splice"],
    },
    {
        "id": "office-08-kanban-code",
        "num": 8,
        "group": "office",
        "studio_wing": "bg3_screening",
        "wing_name": "WING C // DAILIES SCREENING ROOM & DIRECTOR ALERTS",
        "repo": "langwatch/kanban-code",
        "url": "https://github.com/langwatch/kanban-code",
        "category": "Dailies Screening Theatre + Mobile Director Review",
        "install": "git clone https://github.com/langwatch/kanban-code",
        "title": "35mm Dailies Screening Theatre & Mobile Director Sign-Off",
        "featured_shot": "carrot",
        "blurb": "Links every shot's worktree to the Quick, Draw! Screening Theatre where the projector beams the 4-pass background redraw sheet onto the big screen and rings the studio alarm clock for 1-tap approval.",
        "stages": ["Thread Worktree Dailies", "Project 4-Pass BG Sheet", "Ring Screening Bell Alert", "Director Chair Sign-Off"],
    },
    {
        "id": "office-09-vibe-kanban",
        "num": 9,
        "group": "office",
        "studio_wing": "bg2_occupied",
        "wing_name": "WING B4 // 10-HARNESS ANIMATOR BULLPEN & CEL DIFFS",
        "repo": "BloopAI/vibe-kanban",
        "url": "https://github.com/BloopAI/vibe-kanban",
        "category": "Multi-Harness Animator Bullpen + Live Stage Preview",
        "install": "npx vibe-kanban",
        "title": "10-Harness Bullpen + Grease-Pencil Cel Diffs & Live Stage",
        "featured_shot": "sandwich",
        "blurb": "Orchestrates 10+ coding harnesses across the Quick, Draw! drafting bullpen with red pencil diff notes written directly on the lightbox glass and live multiplane parallax previews on the studio wall.",
        "stages": ["Brief 10 Bullpen Animators", "Project Live 7-Plane Stage", "Red-Pencil Cel Critique", "FIFO Film Reel Splice"],
    },
    {
        "id": "office-10-claude-code-kanban",
        "num": 10,
        "group": "office",
        "studio_wing": "bg4_optical",
        "wing_name": "WING D2 // CLOCKTOWER 55-MIN CACHE COMPACTOR ATELIER",
        "repo": "NikiforovAll/claude-code-kanban",
        "url": "https://github.com/NikiforovAll/claude-code-kanban",
        "category": "Clocktower Atelier + 55-Min Cache Compactor",
        "install": "npm install -g claude-code-kanban && claude-code-kanban --install && claude-code-kanban --open",
        "title": "Quick, Draw! Clocktower 55-Min Compactor & Before/After Bench",
        "featured_shot": "star",
        "blurb": "Under the giant Quick, Draw! studio clock & hourglass, animators compare Before/After cel mockups on the optical printer bench while the 55-minute chime auto-compacts session handoff notes.",
        "stages": ["Lightbox: Before/After Cel", "Optical Bench: 7-Plane Rig", "55-Min Clocktower Chime", "Handoff Ledger Dispatch"],
    },
]

VIDEO_SKILL_SPECS = [
    {
        "id": "skill-01-huashu-motion",
        "num": 11,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "WING C1 // PREMIERE SCREENING THEATRE & BEAT CONSOLE",
        "timestamp": "00:46",
        "repo": "alchaincyf/huashu-motion",
        "url": "https://github.com/alchaincyf/huashu-motion",
        "category": "Beat-Synced Premiere Films & Dailies Screening Studio",
        "install": "npx skills add alchaincyf/huashu-motion",
        "title": "Beat-Synced Studio Premiere Film & Screening Console",
        "featured_shot": "giraffe-walks-past",
        "blurb": "Inside the Quick, Draw! Dailies Screening Theatre, the projector and stereo mixing console cut between our 346 multiplane shots right on the musical downbeat while projecting live CLIP & 16-gate studio charts.",
        "stages": ["Console: Lock Beat Grid", "Projector: 346-Shot Chart", "Screen: Beat-Synced Cuts", "Premiere Reel Master Out"],
    },
    {
        "id": "skill-02-rossark-motion-skill",
        "num": 12,
        "group": "skill",
        "studio_wing": "bg5_bullpen",
        "wing_name": "WING B5 // 60 FPS ARC-TRACING LIGHTBOX & CLOCKWORK",
        "timestamp": "03:51",
        "repo": "rossark0/rossark-motion-skill",
        "url": "https://github.com/rossark0/rossark-motion-skill",
        "category": "60 FPS Single-Scene-Clock SVG Character Arcs",
        "install": "npx skills@latest add rossark0/rossark-motion-skill",
        "title": "60 FPS Single-Scene-Clock SVG Arcs on the Lightbox",
        "featured_shot": "bicycle",
        "blurb": "On the circular Quick, Draw! lightbox disc, the Lead Animator traces Quick, Draw! SVG strokes along smooth cubic Bezier motion arcs driven by a single deterministic scene clock tau and MOTION.md rules.",
        "stages": ["Pin MOTION.md to Peg Bar", "Lock Single Clock tau", "Trace Cubic Bezier Arcs", "60 FPS Cel Exposure"],
    },
    {
        "id": "skill-03-take-one",
        "num": 13,
        "group": "skill",
        "studio_wing": "bg2_occupied",
        "wing_name": "WING B6 // 4-DESK STORY ROOM (STRATEGIST TO CRITIC)",
        "timestamp": "05:03",
        "repo": "talsiach/take-one",
        "url": "https://github.com/talsiach/take-one",
        "category": "Strategist -> Creative Director -> Storyboard -> Critic",
        "install": "git clone https://github.com/talsiach/take-one ~/.claude/skills/take-one",
        "title": "4-Desk Story Bullpen: Strategist, Director, Boarder & Critic",
        "featured_shot": "lighthouse",
        "blurb": "Each of the 4 Quick, Draw! drafting desks in our bullpen houses one role: Desk 1 writes the visual thesis, Desk 2 locks Bouleau's 14-line armature, Desk 3 inks the 4 keyframes, and Desk 4 runs blind critique.",
        "stages": ["Desk 1: Visual Thesis", "Desk 2: 14-Line Armature", "Desk 3: 4-Key Cel Inking", "Desk 4: Blind Critic Pass"],
    },
    {
        "id": "skill-04-reanimate",
        "num": 14,
        "group": "skill",
        "studio_wing": "bg4_optical",
        "wing_name": "WING D3 // TWIN-REEL ROTOSCOPE & OPTICAL DIFF BENCH",
        "timestamp": "06:55",
        "repo": "NiazMorshed2007/reanimate",
        "url": "https://github.com/NiazMorshed2007/reanimate",
        "category": "Twin-Reel Rotoscope Reverse-Engineering & IoU Match",
        "install": "npx skills add NiazMorshed2007/reanimate -g",
        "title": "Twin-Reel Optical Printer & Onion-Skin Silhouette Match",
        "featured_shot": "train",
        "blurb": "Threads the archival reference trace and our new 7-plane Quick, Draw! cartoon reel through the twin-reel optical printer, superimposing both frames on the glowing rotoscope glass until silhouette IoU >= 0.89.",
        "stages": ["Thread 33 Archival Frames", "Superimpose on Rotoscope", "Refine Keyframe Pose Arc", "Lock Silhouette IoU >= 0.89"],
    },
    {
        "id": "skill-05-argo",
        "num": 15,
        "group": "skill",
        "studio_wing": "bg1_multifloor",
        "wing_name": "WING A4 // 7-LEVEL MULTIPLANE RIG & SPOTLIGHT CALLOUTS",
        "timestamp": "09:02",
        "repo": "shreyaskarnik/argo",
        "url": "https://github.com/shreyaskarnik/argo",
        "category": "Multiplane Camera Rig Walkthrough & Spotlight Callouts",
        "install": "npm i -D @argo-video/cli",
        "title": "7-Level Multiplane Camera Rig Tour & Plane Callouts",
        "featured_shot": "guitar",
        "blurb": "Guides the studio camera down all 7 gouache platens of the towering Quick, Draw! Multiplane Camera Rig—spotlighting the Sky Backdrop (P6), Midground Scenery (P3), Hero Character Cel (P2), and Foreground (P0).",
        "stages": ["Rig Top: Camera Lens Check", "Spotlight Planes P6..P3", "Zoom Callout on Hero P2", "Sync Exposure to Narration"],
    },
    {
        "id": "skill-06-explainroo",
        "num": 16,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "WING C2 // VOICEOVER BOOTH & SUBTITLE SCREENING STAGE",
        "timestamp": "09:50",
        "repo": "vincentsch/explainroo",
        "url": "https://github.com/vincentsch/explainroo",
        "category": "Voiceover Recording Booth + Word-Timed Screening Titles",
        "install": "git clone https://github.com/vincentsch/explainroo.git",
        "title": "Voiceover Boom Booth & Word-Timed Screening Cards",
        "featured_shot": "diamond",
        "blurb": "Records warm studio narration under the hanging Quick, Draw! condenser microphone on the left soundstage while projecting word-accurate title cards onto the dailies projection screen on the right.",
        "stages": ["Script 13-Stage Studio Tour", "Record at Condenser Mic", "Align Whisper Word Cues", "Project Title Cards on Screen"],
    },
    {
        "id": "skill-07-video-director",
        "num": 17,
        "group": "skill",
        "studio_wing": "bg1_multifloor",
        "wing_name": "WING A5 // X-SHEET CONTINUITY & STEENBECK QA LAB",
        "timestamp": "11:20",
        "repo": "win4r/video-director",
        "url": "https://github.com/win4r/video-director",
        "category": "Pre-Shoot X-Sheet Check + Steenbeck Post-Render QA",
        "install": "git clone https://github.com/win4r/video-director.git && cp -R video-director/video-director ~/.claude/skills/",
        "title": "Pre-Shoot X-Sheet Check & Steenbeck Loupe QA",
        "featured_shot": "carrot",
        "blurb": "Before shooting on the 7-level Multiplane Tower, the Director validates the exposure X-sheet (`check_timeline.py`), then inspects the developed film on the ground-floor Steenbeck (`qa_video.py`).",
        "stages": ["Audit Exposure X-Sheet", "Shoot 7-Plane Multiplane", "Inspect Steenbeck Loupe", "Certify LUFS & Print Master"],
    },
    {
        "id": "skill-08-earshot",
        "num": 18,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "WING C3 // FOLEY PIT (WOOD/GRAVEL/WATER) & JERK SYNC",
        "timestamp": "12:57",
        "repo": "Leonxlnx/earshot",
        "url": "https://github.com/Leonxlnx/earshot",
        "category": "Foley Pit + 252-SFX Bank Synced to Worm Jerk Spikes",
        "install": "git clone https://github.com/Leonxlnx/earshot.git",
        "title": "Live Foley Pit & 252-SFX Synced to Worm Jerk ||c'''(tau)||",
        "featured_shot": "zigzag",
        "blurb": "Whenever the 3D Spatiotemporal Worm's Kinetic Jerk ||c'''(tau)|| spikes on the screening room monitor, the ON AIR lamp flashes and our Foley Artist strikes Quick, Draw! props in the floor Foley pit!",
        "stages": ["Track 3D Worm Jerk ||c'''||", "Flash ON AIR at Jerk Peak", "Strike Foley Pit Props", "Mix Console VU & Master"],
    },
]

_QD_CACHE = {}
_WORM_DATA_CACHE = None
_SKETCH_PAIR_CACHE = {}


def chaikin_smooth(pts: list, iterations: int = 1) -> list:
    if len(pts) < 3:
        return pts
    out = [list(p) for p in pts]
    for _ in range(iterations):
        if len(out) < 3:
            break
        nxt = [out[0]]
        for i in range(len(out) - 1):
            p0, p1 = out[i], out[i + 1]
            nxt.append([0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]])
            nxt.append([0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]])
        nxt.append(out[-1])
        out = nxt
    return out


def load_qd_norm(cat: str, idx: int = 0) -> list:
    """Loads a clean, normalized `[0,1]` Quick, Draw! vector doodle from `quickdraw_samples/<cat>.json`."""
    if cat not in _QD_CACHE:
        p = QD_DIR / f"{cat}.json"
        if not p.exists():
            _QD_CACHE[cat] = []
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            recs = data.get("drawings", data.get("records", [])) if isinstance(data, dict) else data
            valid = []
            for r in recs:
                st = r.get("strokes", r.get("drawing", []))
                n_pts = r.get("num_points", sum(len(s[0]) if (len(s) == 2 and isinstance(s[0], list)) else len(s) for s in st))
                if 2 <= len(st) <= 14 and 12 <= n_pts <= 130:
                    valid.append(st)
            if not valid:
                valid = [r.get("strokes", r.get("drawing", [])) for r in recs[:10] if r.get("strokes", r.get("drawing", []))]
            _QD_CACHE[cat] = valid
        except Exception:
            _QD_CACHE[cat] = []
            return []
    items = _QD_CACHE[cat]
    if not items:
        return []
    raw = items[idx % len(items)]
    strokes = []
    xs, ys = [], []
    for s in raw:
        if len(s) == 2 and isinstance(s[0], list):
            pts = [[float(x), float(y)] for x, y in zip(s[0], s[1])]
        else:
            pts = [[float(p[0]), float(p[1])] for p in s]
        if len(pts) >= 2:
            strokes.append(pts)
            for px, py in pts:
                xs.append(px)
                ys.append(py)
    if not xs:
        return []
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(10.0, max(x1 - x0, y1 - y0))
    w_span = max(6.0, x1 - x0)
    h_span = max(6.0, y1 - y0)
    norm = []
    for s in strokes:
        ns = [[(p[0] - x0) / w_span, (p[1] - y0) / h_span] for p in s]
        norm.append(chaikin_smooth(ns, 1))
    return norm


def draw_qd_line(canvas, p0, p1, color=COL_INK, thickness=2, boil_seed=0):
    """Draws a hand-sketched Quick, Draw! ink line with subtle organic midpoint wobble."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(2, min(14, int(dist / 28.0)))
    pts = []
    for i in range(steps + 1):
        t = i / float(steps)
        px = x0 + (x1 - x0) * t
        py = y0 + (y1 - y0) * t
        if 0 < i < steps:
            jx = math.sin(i * 1.9 + boil_seed * 1.3 + x0 * 0.05) * 1.1
            jy = math.cos(i * 2.3 + boil_seed * 1.7 + y0 * 0.05) * 1.1
            px += jx
            py += jy
        pts.append([int(round(px)), int(round(py))])
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_qd_card(canvas, x0, y0, x1, y1, fill_bgr=COL_SHEET, ink_bgr=COL_INK, thickness=2, shadow=True, boil_seed=0, header_bgr=None, title=""):
    """Draws a Quick, Draw! cream paper sheet card with hard offset ink drop-shadow and hand-inked border."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if shadow:
        cv2.rectangle(canvas, (x0 + 3, y0 + 3), (x1 + 3, y1 + 3), ink_bgr, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), fill_bgr, -1)
    if header_bgr is not None:
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + 20), header_bgr, -1)
        draw_qd_line(canvas, (x0, y0 + 20), (x1, y0 + 20), ink_bgr, 1, boil_seed)
    draw_qd_line(canvas, (x0, y0), (x1, y0), ink_bgr, thickness, boil_seed)
    draw_qd_line(canvas, (x1, y0), (x1, y1), ink_bgr, thickness, boil_seed + 1)
    draw_qd_line(canvas, (x1, y1), (x0, y1), ink_bgr, thickness, boil_seed + 2)
    draw_qd_line(canvas, (x0, y1), (x0, y0), ink_bgr, thickness, boil_seed + 3)
    if title:
        cv2.putText(canvas, title, (x0 + 7, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL_INK, 1, cv2.LINE_AA)


def draw_qd_doodle(canvas, cat: str, idx: int, cx: float, cy: float, w: float, h: float,
                   cel_bgr=COL_SHEET, ink_bgr=COL_INK, thickness: int = 2, boil_seed: int = 0, flip_x: bool = False):
    """
    Renders a REAL Quick, Draw! vector doodle from `quickdraw_samples/<cat>.json` at `(cx, cy, w, h)`
    with a watertight gouache paper backing cel (so background lines never bleed through) and 4-keyframe boil.
    """
    norm = load_qd_norm(cat, idx)
    if not norm:
        return
    H_c, W_c = canvas.shape[:2]
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    strokes = []
    all_pts = []
    for s_i, s in enumerate(norm):
        pts = []
        for p_i, (px, py) in enumerate(s):
            nx = (1.0 - px) if flip_x else px
            jx = math.sin(boil_seed * 1.6 + s_i * 1.1 + p_i * 0.7) * 0.85
            jy = math.cos(boil_seed * 1.9 + s_i * 1.3 + p_i * 0.9) * 0.85
            gx = int(round(np.clip(x0 + nx * w + jx, 2, W_c - 3)))
            gy = int(round(np.clip(y0 + py * h + jy, 2, H_c - 3)))
            pts.append([gx, gy])
            all_pts.append([gx, gy])
        if len(pts) >= 2:
            strokes.append(np.array(pts, dtype=np.int32))

    if cel_bgr is not None and len(all_pts) >= 3:
        # Tight morphological gouache backing cel so studio background lines are cleanly occluded
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        pad = 8
        bx0, by0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        bx1, by1 = min(W_c - 1, max(xs) + pad), min(H_c - 1, max(ys) + pad)
        mw, mh = max(6, bx1 - bx0 + 1), max(6, by1 - by0 + 1)
        mask = np.zeros((mh, mw), dtype=np.uint8)
        for st in strokes:
            shifted = st - np.array([[bx0, by0]], dtype=np.int32)
            cv2.polylines(mask, [shifted], False, 255, 6, cv2.LINE_AA)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
        roi = canvas[by0:by0 + mh, bx0:bx0 + mw]
        if roi.shape[:2] == filled.shape[:2]:
            roi[filled > 0] = cel_bgr

    for st in strokes:
        cv2.polylines(canvas, [st], False, ink_bgr, thickness, cv2.LINE_AA)


def draw_qd_studio_wing(W: int, H: int, wing: str, boil_key: int = 0) -> np.ndarray:
    """
    Constructs the full-frame (W x H) Architectural Animation Studio Soundstage in the
    100% Unified Quick, Draw! Paper & Boiling-Ink Aesthetic (`#F6EFE3` paper + real Quick, Draw! doodles).
    """
    canvas = np.full((H, W, 3), COL_PAPER, dtype=np.uint8)

    # Subtle hand-ruled perspective drafting grid on archival paper
    for gx in range(0, W, 48):
        cv2.line(canvas, (gx, 36), (gx, H - 36), (214, 226, 235), 1)
    for gy in range(36, H - 36, 36):
        cv2.line(canvas, (0, gy), (W, gy), (214, 226, 235), 1)

    if wing == "bg1_multifloor":
        # =====================================================================
        # WING A: 7-LEVEL MULTIPLANE CAMERA TOWER (LEFT) + 3-FLOOR STUDIO (RIGHT)
        # =====================================================================
        # Left Multiplane Tower Zone Wash
        draw_qd_card(canvas, 14, 42, 246, 314, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key)
        # Two vertical cast-iron Multiplane Tower pillars in Quick, Draw! ink
        draw_qd_card(canvas, 28, 46, 44, 310, fill_bgr=COL_PAPER_DEEP, shadow=False, boil_seed=boil_key + 1)
        draw_qd_card(canvas, 216, 46, 232, 310, fill_bgr=COL_PAPER_DEEP, shadow=False, boil_seed=boil_key + 2)
        # Top Camera Mount + Real Quick, Draw! `camera` aiming down!
        draw_qd_doodle(canvas, "camera", 0, 130, 64, 52, 34, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        # 6 Glass Multiplane Platens (P6..P1) holding real Quick, Draw! doodles!
        platen_specs = [
            (102, "P6 SKY", "cloud", COL_BLUE_PALE),
            (136, "P5 FAR", "mountain", COL_PAPER),
            (170, "P4 MID", "tree", COL_FOREST_PALE),
            (204, "P3 PROP", "house", COL_TAG_PALE),
            (238, "P2 HERO", "giraffe", COL_WAX_PALE),
            (274, "P1 FORE", "bush", COL_FOREST_PALE),
        ]
        for p_i, (py, p_lbl, p_cat, p_col) in enumerate(platen_specs):
            shift = int(round(math.sin(boil_key * 0.9 + p_i) * (p_i + 1) * 1.5))
            draw_qd_card(canvas, 48 + shift, py - 11, 212 + shift, py + 11, fill_bgr=p_col, shadow=False, boil_seed=boil_key + p_i)
            draw_qd_doodle(canvas, p_cat, p_i, 145 + shift, py, 34, 18, cel_bgr=None, thickness=1, boil_seed=boil_key + p_i)
            cv2.putText(canvas, p_lbl, (53 + shift, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_INK, 1, cv2.LINE_AA)

        # Center Spiral Staircase / Ladder Column (x: 252..308)
        draw_qd_card(canvas, 252, 42, 308, 314, fill_bgr=COL_PAPER_DEEP, shadow=True, boil_seed=boil_key)
        draw_qd_doodle(canvas, "stairs", 0, 280, 110, 44, 65, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "ladder", 0, 280, 235, 38, 95, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)

        # Right 3 Cutaway Studio Floors (x: 316..626)
        # Floor 3 (Top: Storyboard & Espresso Lounge, y: 42..152)
        draw_qd_card(canvas, 316, 42, 626, 152, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key)
        # Floor 2 (Middle: Animator Lightbox Bullpen, y: 158..244)
        draw_qd_card(canvas, 316, 158, 626, 244, fill_bgr=COL_TAG_PALE, shadow=True, boil_seed=boil_key + 1)
        draw_qd_doodle(canvas, "floor_lamp", 0, 338, 198, 28, 52, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "table", 0, 405, 212, 58, 36, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "chair", 0, 368, 214, 28, 36, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        # Floor 1 (Bottom: 35mm Steenbeck Archive Vault, y: 250..314)
        draw_qd_card(canvas, 316, 250, 626, 314, fill_bgr=COL_BLUE_PALE, shadow=True, boil_seed=boil_key + 2)
        for bx in [342, 388, 555, 598]:
            draw_qd_doodle(canvas, "book", bx % 3, bx, 282, 34, 34, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "television", 0, 472, 282, 48, 42, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "coffee_cup", 0, 518, 288, 26, 24, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)

    elif wing in ("bg5_bullpen", "bg2_occupied"):
        # =====================================================================
        # WING B: QUICK, DRAW! ISOMETRIC ANIMATOR BULLPEN & ARMATURE WALL
        # =====================================================================
        # Isometric studio walls & wooden plank floor lines in Quick, Draw! ink
        draw_qd_card(canvas, 14, 42, 626, 314, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key)
        # Left Studio Window Wall with Quick, Draw! `window`, `clock`, `cat`, `floor_lamp`, `coffee_cup`
        draw_qd_card(canvas, 22, 48, 336, 168, fill_bgr=COL_TAG_PALE, shadow=False, boil_seed=boil_key, title="BULLPEN STORYBOARD & MODEL SHEETS")
        # Pinned Quick, Draw! storyboard frames connected by Bouleau's 14-line golden armature!
        for px, py, pcat in [(58, 96, "picture_frame"), (125, 96, "giraffe"), (192, 96, "train"), (260, 96, "lighthouse"), (88, 138, "bicycle"), (165, 138, "guitar"), (245, 138, "star")]:
            draw_qd_card(canvas, px - 26, py - 18, px + 26, py + 18, fill_bgr=COL_SHEET, shadow=False, boil_seed=boil_key)
            draw_qd_doodle(canvas, pcat, 0, px, py, 36, 26, cel_bgr=None, thickness=1, boil_seed=boil_key)
        # Golden Bouleau armature diagonals across the storyboard corkboard
        draw_qd_line(canvas, (28, 66), (330, 162), COL_TAG, 2, boil_key)
        draw_qd_line(canvas, (28, 162), (330, 66), COL_TAG, 2, boil_key + 1)

        # Drafting Desks, Lamps, Chairs, Paintbrush & Coffee Cups on the Bullpen Floor (y: 174..308)
        draw_qd_doodle(canvas, "floor_lamp", 0, 48, 228, 42, 72, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "table", 0, 175, 248, 110, 64, cel_bgr=COL_PAPER_DEEP, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "chair", 0, 102, 258, 44, 54, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "table", 1, 335, 262, 110, 64, cel_bgr=COL_PAPER_DEEP, thickness=2, boil_seed=boil_key + 1)
        draw_qd_doodle(canvas, "chair", 1, 262, 268, 44, 54, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key + 1)
        draw_qd_doodle(canvas, "coffee_cup", 0, 238, 208, 30, 26, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "paintbrush", 0, 82, 195, 34, 32, cel_bgr=COL_WAX_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "cat", 0, 48, 290, 42, 32, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)

    elif wing == "bg3_screening":
        # =====================================================================
        # WING C: QUICK, DRAW! FOLEY SOUNDSTAGE PIT (LEFT) & SCREENING ROOM (RIGHT)
        # =====================================================================
        # Left Half: Foley Recording Stage (x: 14..368, y: 42..314)
        draw_qd_card(canvas, 14, 42, 368, 314, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key)
        # Acoustic Quick, Draw! baffle panels along top wall
        for bx in range(155, 355, 48):
            draw_qd_card(canvas, bx, 48, bx + 38, 82, fill_bgr=COL_PAPER_DEEP, shadow=False, boil_seed=boil_key + bx)
        # Hanging Boom Microphone & Headphones
        draw_qd_line(canvas, (22, 96), (185, 76), COL_INK, 3, boil_key)
        draw_qd_doodle(canvas, "microphone", 0, 185, 112, 36, 48, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        # Foley Prop Table with Quick, Draw! `shoe`, `alarm_clock`, `teapot`, `megaphone`
        draw_qd_doodle(canvas, "table", 0, 190, 196, 145, 58, cel_bgr=COL_PAPER_DEEP, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "shoe", 0, 142, 174, 34, 24, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "teapot", 0, 190, 172, 34, 26, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "alarm_clock", 0, 238, 172, 30, 28, cel_bgr=COL_WAX_PALE, thickness=2, boil_seed=boil_key)
        # Mixing Console (`stereo` / `radio`)
        draw_qd_doodle(canvas, "stereo", 0, 318, 206, 68, 48, cel_bgr=COL_BLUE_PALE, thickness=2, boil_seed=boil_key)
        # 3-Surface Floor Foley Pit (WOOD / GRAVEL / WATER)
        pit_specs = [(32, 112, COL_TAG_PALE, "WOOD PIT"), (118, 198, COL_PAPER_DEEP, "GRAVEL PIT"), (204, 284, COL_BLUE_PALE, "WATER PIT")]
        for px0, px1, pcol, plbl in pit_specs:
            draw_qd_card(canvas, px0, 258, px1, 308, fill_bgr=pcol, shadow=False, boil_seed=boil_key)
            cv2.putText(canvas, plbl, (px0 + 8, 302), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_INK, 1, cv2.LINE_AA)

        # Right Half: 35mm Dailies Screening Theatre (x: 376..626, y: 42..314)
        draw_qd_card(canvas, 376, 42, 626, 314, fill_bgr=COL_PAPER_DEEP, shadow=True, boil_seed=boil_key + 1)
        # Quick, Draw! `camera` Projector on pedestal + Row of 4 Quick, Draw! Director `chair`s
        draw_qd_doodle(canvas, "camera", 1, 398, 128, 42, 34, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        for cx_chair in [418, 472, 526, 580]:
            draw_qd_doodle(canvas, "chair", 0, cx_chair, 272, 42, 52, cel_bgr=COL_WAX_PALE, thickness=2, boil_seed=boil_key, flip_x=True)

    else:
        # =====================================================================
        # WING D: QUICK, DRAW! CLOCKTOWER OPTICAL PRINTER & 3D WORM ATELIER
        # =====================================================================
        draw_qd_card(canvas, 14, 42, 626, 314, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key)
        # Left: Twin-Reel Optical Printer (`clock`/`wheel` reels + `camera` + `scissors` + `film`)
        draw_qd_card(canvas, 24, 52, 256, 222, fill_bgr=COL_TAG_PALE, shadow=False, boil_seed=boil_key, title="TWIN-REEL OPTICAL PRINTER")
        draw_qd_doodle(canvas, "wheel", 0, 82, 112, 62, 62, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "wheel", 1, 182, 112, 62, 62, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key + 1)
        draw_qd_doodle(canvas, "camera", 0, 132, 178, 58, 42, cel_bgr=COL_BLUE_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "scissors", 0, 52, 185, 36, 32, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)

        # Center: 55-Minute Prompt-Cache Clocktower (`clock`, `hourglass`, `alarm_clock`, `paintbrush` shelves)
        draw_qd_card(canvas, 266, 52, 424, 306, fill_bgr=COL_PAPER_DEEP, shadow=False, boil_seed=boil_key, title="55-MIN CACHE CLOCKTOWER")
        draw_qd_doodle(canvas, "clock", 0, 345, 118, 78, 78, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "hourglass", 0, 305, 198, 42, 52, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "alarm_clock", 0, 382, 198, 44, 48, cel_bgr=COL_WAX_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "paintbrush", 0, 298, 268, 38, 38, cel_bgr=COL_FOREST_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "crayon", 0, 388, 268, 38, 38, cel_bgr=COL_BLUE_PALE, thickness=2, boil_seed=boil_key)

        # Right: 3D Spatiotemporal Worm Pedestal flanked by Quick, Draw! `floor_lamp` & `light_bulb`
        draw_qd_doodle(canvas, "light_bulb", 0, 505, 68, 34, 38, cel_bgr=COL_TAG_PALE, thickness=2, boil_seed=boil_key)
        draw_qd_doodle(canvas, "floor_lamp", 0, 602, 245, 36, 78, cel_bgr=COL_SHEET, thickness=2, boil_seed=boil_key)

    return canvas


def archival_to_quickdraw_sketch(arch_bgr: np.ndarray) -> np.ndarray:
    """
    Converts a raw photographic archival frame into the Unified Quick, Draw! Blueprint & Graphite
    Sketch aesthetic on `#F6EFE3` cream paper (preserving the SAM tracker box in forest green ink).
    """
    h, w = arch_bgr.shape[:2]
    gray = cv2.cvtColor(arch_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 45, 125)
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    ink_mask = cv2.bitwise_or(edges, adapt)

    # Start with warm archival paper (#F6EFE3) + subtle tonal paper wash from source luminance
    out = np.full((h, w, 3), COL_PAPER, dtype=np.uint8)
    dark_tone = (gray < 115)
    out[dark_tone] = COL_PAPER_DEEP
    out[ink_mask > 0] = COL_INK_SOFT

    # Preserve green SAM bounding box from comparison.mp4 in crisp forest green ink
    b, g, r = arch_bgr[:, :, 0], arch_bgr[:, :, 1], arch_bgr[:, :, 2]
    green_box = (g > 150) & (g > r + 40) & (g > b + 40)
    out[green_box] = COL_FOREST
    return out


def get_worm_record(slug: str) -> dict:
    """Loads the 3D Spatiotemporal Worm record for `slug` from data/spatiotemporal_worms_346.json."""
    global _WORM_DATA_CACHE
    if _WORM_DATA_CACHE is None:
        wpath = DATA_DIR / "spatiotemporal_worms_346.json"
        if wpath.exists():
            try:
                raw = json.loads(wpath.read_text())
                _WORM_DATA_CACHE = {s["slug"]: s for s in raw.get("shots", [])}
            except Exception:
                _WORM_DATA_CACHE = {}
        else:
            _WORM_DATA_CACHE = {}
    return _WORM_DATA_CACHE.get(slug, {})


def load_shot_pair(slug: str, frame_idx: int):
    """Loads (quickdraw_archival_sketch, cartoon_frame) at frame_idx from shots/<slug>/comparison.mp4."""
    key = (slug, frame_idx % 33)
    if key in _SKETCH_PAIR_CACHE:
        return _SKETCH_PAIR_CACHE[key]
    comp_path = SHOTS_DIR / slug / "comparison.mp4"
    if comp_path.exists():
        cap = cv2.VideoCapture(str(comp_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % 33)
        ok, fr = cap.read()
        cap.release()
        if ok and fr is not None:
            h, w = fr.shape[:2]
            arch_raw = fr[:, :w // 2]
            cart_raw = fr[:, w // 2:]
            arch_qd = archival_to_quickdraw_sketch(arch_raw)
            _SKETCH_PAIR_CACHE[key] = (arch_qd, cart_raw)
            return arch_qd, cart_raw
    f = np.full((240, 320, 3), COL_SHEET, dtype=np.uint8)
    return f, f


def load_shot_keyframe_strip(slug: str):
    """Loads the 4-keyframe sheet or design.png for `slug` so we can project it on the animator's lightbox."""
    for name in ["selected-keys.png", "design.png", "stage_3_keyframes.png"]:
        p = SHOTS_DIR / slug / name
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def draw_animator_lightbox_disc(canvas, cx, cy, radius, slug, f_idx, label="LIGHTBOX"):
    """Draws a circular Quick, Draw! animator lightbox disc on `#FFFBF4` paper with peg bar & live onion-skin keyframe."""
    boil = (f_idx // 2) % 4
    # Offset ink drop-shadow + outer wooden rim in amber tag wash
    cv2.circle(canvas, (cx + 3, cy + 3), radius + 5, COL_INK, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius + 5, COL_TAG_PALE, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius + 5, COL_INK, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, COL_SHEET, -1, cv2.LINE_AA)

    kf_img = load_shot_keyframe_strip(slug)
    if kf_img is not None:
        kh, kw = kf_img.shape[:2]
        k_idx = (f_idx // 4) % 4
        if kw >= kh * 2:
            cw = kw // 4
            crop = kf_img[:, k_idx * cw:(k_idx + 1) * cw]
        else:
            crop = kf_img
        d = radius * 2 - 6
        if d > 12 and crop.size > 0:
            thumb = cv2.resize(crop, (d, d), interpolation=cv2.INTER_AREA)
            y0, x0 = cy - d // 2, cx - d // 2
            roi = canvas[max(0, y0):min(canvas.shape[0], y0 + d), max(0, x0):min(canvas.shape[1], x0 + d)]
            if roi.shape[:2] == (d, d):
                mask = np.zeros((d, d), dtype=np.uint8)
                cv2.circle(mask, (d // 2, d // 2), d // 2 - 2, 255, -1)
                blended = cv2.addWeighted(roi, 0.15, thumb, 0.85, 0)
                roi[mask > 0] = blended[mask > 0]

    cv2.circle(canvas, (cx, cy), radius, COL_INK, 2, cv2.LINE_AA)
    # Brass registration peg bar at top of lightbox disc
    draw_qd_card(canvas, cx - int(radius * 0.65), cy - radius - 4, cx + int(radius * 0.65), cy - radius + 4,
                 fill_bgr=COL_TAG, shadow=False, boil_seed=boil)
    for px in [-int(radius * 0.4), 0, int(radius * 0.4)]:
        cv2.circle(canvas, (cx + px, cy - radius), 2, COL_INK, -1)

    # Real Quick, Draw! `pencil` / `paintbrush` actively sketching across the lightbox!
    pen_x = cx + int(round(math.cos(f_idx * 0.45) * (radius * 0.35)))
    pen_y = cy + int(round(math.sin(f_idx * 0.65) * (radius * 0.30)))
    draw_qd_doodle(canvas, "pencil", 0, pen_x + 10, pen_y - 10, 24, 24, cel_bgr=None, ink_bgr=COL_WAX, thickness=2, boil_seed=boil)

    if label:
        draw_qd_card(canvas, cx - radius, cy + radius - 10, cx + radius, cy + radius + 5, fill_bgr=COL_SHEET, shadow=False, boil_seed=boil)
        cv2.putText(canvas, label, (cx - radius + 4, cy + radius + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_INK, 1, cv2.LINE_AA)


def draw_studio_crew_member(canvas, x, y, agent, frame_idx=0, scale=0.95, bubble_text=""):
    """
    Draws a 100% Quick, Draw! Doodle Studio Crew Member (constructed from real Quick, Draw! `smiley_face`,
    `crown`/`hat`/`headphones`/`eyeglasses`, hand-inked boiling doodle smock, scissoring legs, and Quick, Draw! prop).
    """
    boil = (frame_idx // 2) % 4
    bob = int(round(math.sin(frame_idx * 0.55 + x * 0.04) * 2.5 * scale))
    cy = y + bob

    # Hand-inked floor shadow loop
    cv2.ellipse(canvas, (x, y + int(23 * scale)), (int(12 * scale), int(4 * scale)), 0, 0, 360, COL_PAPER_DEEP, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x, y + int(23 * scale)), (int(12 * scale), int(4 * scale)), 0, 0, 360, COL_INK, 1, cv2.LINE_AA)

    # Walking Quick, Draw! legs
    stride = int(round(math.sin(frame_idx * 0.65 + y * 0.05) * 6.5 * scale))
    draw_qd_line(canvas, (x - int(4 * scale), cy + int(10 * scale)), (x - int(4 * scale) - stride, y + int(22 * scale)), COL_INK, 2, boil)
    draw_qd_line(canvas, (x + int(4 * scale), cy + int(10 * scale)), (x + int(4 * scale) + stride, y + int(22 * scale)), COL_INK, 2, boil + 1)

    # Quick, Draw! doodle smock torso (gouache cel + boiling ink outline)
    draw_qd_card(canvas, x - int(10 * scale), cy - int(2 * scale), x + int(10 * scale), cy + int(12 * scale),
                 fill_bgr=agent["color"], shadow=False, boil_seed=boil)

    # Real Quick, Draw! `smiley_face` head!
    draw_qd_doodle(canvas, "smiley_face", 0, x, cy - int(11 * scale), int(21 * scale), int(21 * scale),
                   cel_bgr=COL_SHEET, ink_bgr=COL_INK, thickness=2, boil_seed=boil)

    # Real Quick, Draw! Hat / Crown / Headphones / Eyeglasses on top!
    hat_cat = agent.get("hat_qd", "hat")
    draw_qd_doodle(canvas, hat_cat, 0, x, cy - int(22 * scale), int(22 * scale), int(13 * scale),
                   cel_bgr=agent["color"], ink_bgr=COL_INK, thickness=1, boil_seed=boil)

    # Real Quick, Draw! Prop (`megaphone`, `paintbrush`, `camera`, `microphone`, `book`) in hand!
    prop_cat = agent.get("prop_qd", "paintbrush")
    draw_qd_doodle(canvas, prop_cat, 0, x + int(17 * scale), cy + int(2 * scale), int(18 * scale), int(18 * scale),
                   cel_bgr=COL_SHEET, ink_bgr=COL_INK, thickness=1, boil_seed=boil)

    # Cream paper nameplate badge under feet
    name_w = len(agent["name"]) * 5 + 10
    draw_qd_card(canvas, x - name_w // 2, y + int(24 * scale), x + name_w // 2, y + int(37 * scale),
                 fill_bgr=COL_SHEET, shadow=False, boil_seed=boil)
    cv2.putText(canvas, agent["name"], (x - name_w // 2 + 4, y + int(33 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_INK, 1, cv2.LINE_AA)

    if bubble_text:
        bw = min(152, max(76, len(bubble_text) * 5 + 12))
        bx0 = max(6, min(canvas.shape[1] - bw - 6, x - bw // 2))
        by0 = max(40, cy - int(42 * scale))
        draw_qd_card(canvas, bx0, by0, bx0 + bw, by0 + 16, fill_bgr=COL_TAG_PALE, shadow=True, boil_seed=boil)
        cv2.putText(canvas, bubble_text[:26], (bx0 + 5, by0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.29, COL_INK, 1, cv2.LINE_AA)


def draw_3d_worm_prism_widget(canvas, x0, y0, x1, y1, slug, f_idx, title="3D WORM PRISM"):
    """Renders the shot's real 3D Spatiotemporal Worm inside a Quick, Draw! cream paper & ink isometric prism card."""
    boil = (f_idx // 2) % 4
    draw_qd_card(canvas, x0, y0, x1, y1, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil, header_bgr=COL_BLUE_PALE)

    pw, ph = x1 - x0, y1 - y0
    # Hand-sketched Bouleau 14-Line Armature & Rule of Thirds in soft drafting rule ink
    draw_qd_line(canvas, (x0 + 8, y0 + 22), (x1 - 8, y1 - 14), COL_RULE, 1, boil)
    draw_qd_line(canvas, (x0 + 8, y1 - 14), (x1 - 8, y0 + 22), COL_RULE, 1, boil + 1)
    for frac in (0.333, 0.667):
        gx = x0 + int(pw * frac)
        draw_qd_line(canvas, (gx, y0 + 22), (gx, y1 - 14), COL_RULE, 1, boil)

    wrec = get_worm_record(slug)
    traj = wrec.get("trajectory", [])
    raw_topo = wrec.get("topology", "helical_corkscrew")
    topo = raw_topo.get("class", "helical_corkscrew") if isinstance(raw_topo, dict) else str(raw_topo)
    pts = []
    for t_i in range(33):
        tx = x0 + 10 + int(round((t_i / 32.0) * (pw - 20)))
        if t_i < len(traj):
            u_val = float(traj[t_i].get("u", 0.5))
            v_val = float(traj[t_i].get("v", 0.5))
            ty = y0 + 24 + int(round((0.5 * v_val + 0.5 * (1.0 - u_val)) * (ph - 40)))
        else:
            ty = (y0 + y1) // 2 + int(round(math.sin(t_i * 0.35) * (ph * 0.22)))
        pts.append([tx, max(y0 + 22, min(y1 - 16, ty))])

    pts_arr = np.array(pts, dtype=np.int32)
    # Quick, Draw! gouache amber ribbon + crisp black ink centerline
    cv2.polylines(canvas, [pts_arr], False, COL_TAG, 5, cv2.LINE_AA)
    cv2.polylines(canvas, [pts_arr], False, COL_INK, 2, cv2.LINE_AA)

    cur_t = f_idx % 33
    cx, cy = pts[cur_t]
    draw_qd_line(canvas, (cx, y0 + 22), (cx, y1 - 14), COL_WAX, 1, boil)
    cv2.circle(canvas, (cx, cy), 5, COL_WAX, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 5, COL_INK, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"{title}: {topo[:18]}", (x0 + 6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.29, COL_INK, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"tau={cur_t:02d}/33f", (x0 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_INK_SOFT, 1, cv2.LINE_AA)


def generate_foley_wav(wav_path: Path, n_frames: int, fps: int = 10, seed: int = 1):
    """Generates a warm studio projector hum + melodic marimba & Earshot Foley transients WAV."""
    sr = 22050
    dur = n_frames / float(fps)
    n_samples = int(sr * dur)
    t = np.linspace(0.0, dur, n_samples, endpoint=False)
    rng = np.random.default_rng(seed)

    notes = [220.0, 277.18, 329.63, 440.0, 554.37, 659.25]
    sig = 0.025 * np.sin(2.0 * math.pi * 110.0 * t)
    for beat_i in range(int(dur * 2) + 1):
        t0 = beat_i * 0.45
        mask = (t >= t0) & (t < t0 + 0.38)
        if not np.any(mask):
            continue
        freq = notes[(beat_i + seed) % len(notes)]
        env = np.exp(-6.5 * (t[mask] - t0))
        sig[mask] += 0.18 * env * np.sin(2.0 * math.pi * freq * (t[mask] - t0))
        click_mask = (t >= t0) & (t < t0 + 0.03)
        if np.any(click_mask):
            sig[click_mask] += 0.12 * rng.standard_normal(np.sum(click_mask)) * np.exp(-45.0 * (t[click_mask] - t0))

    sig = np.clip(sig, -0.95, 0.95)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def render_studio_film_frame(spec: dict, f_idx: int, total_frames: int = 40):
    """
    Renders one 640x360 frame of a Behind-the-Scenes Studio Film in the 100% Unified Quick, Draw!
    Paper, Gouache Cel & Boiling-Ink Aesthetic.
    """
    W, H = 640, 360
    wing = spec.get("studio_wing", "bg1_multifloor")
    boil_key = (f_idx // 2) % 4

    # 1. Construct the Unified Quick, Draw! Studio Wing with 4-keyframe boiling ink
    canvas = draw_qd_studio_wing(W, H, wing, boil_key=boil_key)

    stage_idx = min(3, int((f_idx / float(total_frames)) * 4))
    stage_progress = (f_idx % (total_frames // 4)) / float(max(1, total_frames // 4 - 1))

    # Load unified Quick, Draw! archival sketch & 7-plane cartoon frames for the featured shot
    arch_fr, cart_fr = load_shot_pair(spec["featured_shot"], f_idx)

    # =========================================================================
    # WING-SPECIFIC QUICK, DRAW! STUDIO EQUIPMENT & LIVE ANIMATION
    # =========================================================================
    if wing == "bg1_multifloor":
        # Golden paper camera exposure cone down the 7-plane tower when exposing a keyframe
        if f_idx % 8 < 3:
            cone_overlay = canvas.copy()
            cone_pts = np.array([[130, 78], [52, 286], [208, 286]], dtype=np.int32)
            cv2.fillPoly(cone_overlay, [cone_pts], COL_TAG_PALE, cv2.LINE_AA)
            cv2.addWeighted(cone_overlay, 0.35, canvas, 0.65, 0, canvas)

        # Top-Right Mezzanine Lounge: Quick, Draw! Dailies Card (x: 322..620, y: 46..148)
        draw_qd_card(canvas, 322, 46, 620, 148, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key,
                     header_bgr=COL_TAG_PALE, title=f"STUDIO DAILIES // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (136, 74))
        cart_sm = cv2.resize(cart_fr, (136, 74))
        canvas[70:144, 330:466] = arch_sm
        canvas[70:144, 476:612] = cart_sm
        draw_qd_card(canvas, 330, 70, 466, 144, fill_bgr=None if False else COL_PAPER, shadow=False, boil_seed=boil_key)
        canvas[71:143, 331:465] = arch_sm[1:-1, 1:-1]
        draw_qd_card(canvas, 476, 70, 612, 144, fill_bgr=COL_SHEET, shadow=False, boil_seed=boil_key + 1)
        canvas[71:143, 477:611] = cart_sm[1:-1, 1:-1]
        cv2.putText(canvas, "ARCHIVE PENCIL TRACE", (334, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.27, COL_INK, 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE QUICK, DRAW!", (480, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.27, COL_FOREST, 1, cv2.LINE_AA)

        # Middle Floor Bullpen: Glowing Lightbox Disc + 3D Worm Card
        draw_animator_lightbox_disc(canvas, 405, 198, 24, spec["featured_shot"], f_idx, "KEY TRACE")
        draw_3d_worm_prism_widget(canvas, 456, 160, 620, 240, spec["featured_shot"], f_idx, "3D WORM")

        # Quick, Draw! Crew walking across the 3 floors
        crew_wps = [(130, 278), (280, 205), (362, 205), (442, 282)]
        for c_i in range(3):
            ag = STUDIO_AGENTS[(spec["num"] + c_i) % len(STUDIO_AGENTS)]
            p0 = crew_wps[(c_i + stage_idx) % len(crew_wps)]
            p1 = crew_wps[(c_i + stage_idx + 1) % len(crew_wps)]
            cx = int(round((1 - stage_progress) * p0[0] + stage_progress * p1[0]))
            cy = int(round((1 - stage_progress) * p0[1] + stage_progress * p1[1]))
            bub = spec["stages"][stage_idx] if c_i == 0 else ""
            draw_studio_crew_member(canvas, cx, cy, ag, f_idx + c_i * 4, scale=0.85, bubble_text=bub)

    elif wing in ("bg5_bullpen", "bg2_occupied"):
        # Two Foreground Quick, Draw! Animator Lightboxes with live onion-skin keyframes
        draw_animator_lightbox_disc(canvas, 175, 226, 33, spec["featured_shot"], f_idx, "DESK A // K01-K02")
        draw_animator_lightbox_disc(canvas, 335, 244, 31, spec["featured_shot"], f_idx + 8, "DESK B // K03-K04")

        # Right Wall: Quick, Draw! Storyboard & Dailies Sheet (x: 344..620, y: 48..162)
        draw_qd_card(canvas, 344, 48, 620, 162, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key,
                     header_bgr=COL_TAG_PALE, title=f"BULLPEN DAILIES // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (126, 84))
        cart_sm = cv2.resize(cart_fr, (126, 84))
        canvas[72:156, 352:478] = arch_sm
        canvas[72:156, 486:612] = cart_sm
        cv2.rectangle(canvas, (352, 72), (478, 156), COL_INK, 1)
        cv2.rectangle(canvas, (486, 72), (612, 156), COL_INK, 1)
        cv2.putText(canvas, "ARCHIVE TRACE", (356, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.27, COL_INK, 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE CEL", (490, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.27, COL_FOREST, 1, cv2.LINE_AA)

        # Bottom-Right: Bouleau 14-Line Worm Card
        draw_3d_worm_prism_widget(canvas, 426, 172, 620, 292, spec["featured_shot"], f_idx, "BOULEAU 14-LINE WORM")

        # Quick, Draw! Crew walking between the drafting desks
        crew_wps = [(112, 252), (255, 198), (262, 265), (395, 255)]
        for c_i in range(3):
            ag = STUDIO_AGENTS[(spec["num"] + c_i) % len(STUDIO_AGENTS)]
            p0 = crew_wps[(c_i + stage_idx) % len(crew_wps)]
            p1 = crew_wps[(c_i + stage_idx + 1) % len(crew_wps)]
            cx = int(round((1 - stage_progress) * p0[0] + stage_progress * p1[0]))
            cy = int(round((1 - stage_progress) * p0[1] + stage_progress * p1[1]))
            bub = spec["stages"][stage_idx] if c_i == 0 else ""
            draw_studio_crew_member(canvas, cx, cy, ag, f_idx + c_i * 5, scale=0.88, bubble_text=bub)

    elif wing == "bg3_screening":
        # Volumetric paper projector beam from Quick, Draw! `camera` projector (412, 128) to Cinema Screen
        beam_overlay = canvas.copy()
        beam_pts = np.array([[412, 126], [426, 52], [426, 198]], dtype=np.int32)
        cv2.fillPoly(beam_overlay, [beam_pts], COL_TAG_PALE, cv2.LINE_AA)
        cv2.addWeighted(beam_overlay, 0.45, canvas, 0.55, 0, canvas)
        draw_qd_line(canvas, (412, 126), (426, 52), COL_TAG, 1, boil_key)
        draw_qd_line(canvas, (412, 126), (426, 198), COL_TAG, 1, boil_key + 1)

        # Grand Dailies Cinema Screen Card (x: 422..620, y: 48..204)
        draw_qd_card(canvas, 422, 48, 620, 204, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key,
                     header_bgr=COL_TAG_PALE, title=f"SCREENING THEATRE // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (90, 68))
        cart_sm = cv2.resize(cart_fr, (92, 68))
        canvas[72:140, 428:518] = arch_sm
        canvas[72:140, 522:614] = cart_sm
        cv2.rectangle(canvas, (428, 72), (518, 140), COL_INK, 1)
        cv2.rectangle(canvas, (522, 72), (614, 140), COL_INK, 1)
        cv2.putText(canvas, "ARCHIVE TRACE", (431, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COL_INK, 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE FILM", (525, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COL_FOREST, 1, cv2.LINE_AA)

        # Hand-inked Earshot Jerk & Audio Waveform bars inside bottom of Cinema Screen
        cv2.putText(canvas, "EARSHOT JERK ||c'''(tau)|| & SFX SYNC", (430, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.27, COL_INK, 1, cv2.LINE_AA)
        for w_i in range(34):
            wx = 432 + w_i * 5
            wh = int(round(abs(math.sin(w_i * 0.45 + f_idx * 0.55)) * 22)) + 3
            wcol = COL_WAX if wh > 16 else COL_FOREST
            cv2.line(canvas, (wx, 194), (wx, 194 - wh), wcol, 2, cv2.LINE_AA)

        # Quick, Draw! "ON AIR // FOLEY" Studio Sign on top-left wall
        is_jerk_peak = (f_idx % 7 < 3)
        draw_qd_card(canvas, 24, 48, 148, 76, fill_bgr=COL_WAX_PALE if is_jerk_peak else COL_SHEET, shadow=True, boil_seed=boil_key)
        cv2.circle(canvas, (38, 62), 6, COL_WAX if is_jerk_peak else COL_PAPER_DEEP, -1, cv2.LINE_AA)
        cv2.circle(canvas, (38, 62), 6, COL_INK, 1, cv2.LINE_AA)
        cv2.putText(canvas, "ON AIR // FOLEY", (50, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.32, COL_INK, 1, cv2.LINE_AA)

        # Hand-drawn acoustic ripple circles over the active Foley Pit surface
        pit_centers = [(72, 282), (158, 282), (244, 282)]
        pc = pit_centers[stage_idx % 3]
        for r_i in range(1, 3):
            rad = ((f_idx * 4 + r_i * 10) % 24) + 6
            cv2.circle(canvas, pc, rad, COL_WAX, 1, cv2.LINE_AA)

        draw_animator_lightbox_disc(canvas, 68, 198, 24, spec["featured_shot"], f_idx, "CUE SHEET")
        draw_studio_crew_member(canvas, 168, 232, STUDIO_AGENTS[4], f_idx, scale=0.90, bubble_text=spec["stages"][stage_idx])
        draw_studio_crew_member(canvas, 332, 258, STUDIO_AGENTS[0], f_idx + 6, scale=0.86, bubble_text="Dailies Cut!")

    else:
        # Wing D: `bg4_optical` — Twin-Reel Optical Printer, 55-Min Clocktower & 3D Worm Cube
        # Bottom-Left: Rotoscope Lightbox Bench Card (x: 24..256, y: 226..312)
        draw_qd_card(canvas, 24, 226, 256, 312, fill_bgr=COL_SHEET, shadow=True, boil_seed=boil_key,
                     header_bgr=COL_BLUE_PALE, title=f"ROTOSCOPE BENCH // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (106, 58))
        cart_sm = cv2.resize(cart_fr, (106, 58))
        canvas[250:308, 30:136] = arch_sm
        canvas[250:308, 144:250] = cart_sm
        cv2.rectangle(canvas, (30, 250), (136, 308), COL_INK, 1)
        cv2.rectangle(canvas, (144, 250), (250, 308), COL_INK, 1)
        cv2.putText(canvas, "ARCHIVE TRACE", (34, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COL_INK, 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE CEL", (148, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COL_FOREST, 1, cv2.LINE_AA)

        # Right: 3D Spatiotemporal Worm Prism Pedestal (x: 434..582, y: 112..268)
        draw_3d_worm_prism_widget(canvas, 434, 112, 582, 268, spec["featured_shot"], f_idx, "OPTICAL WORM CUBE")

        draw_studio_crew_member(canvas, 298, 252, STUDIO_AGENTS[2], f_idx, scale=0.88, bubble_text=spec["stages"][stage_idx])
        draw_studio_crew_member(canvas, 392, 258, STUDIO_AGENTS[1], f_idx + 5, scale=0.86, bubble_text="IoU >= 0.89!")

    # =========================================================================
    # TOP QUICK, DRAW! CLAPPERBOARD SLATE HEADER (y = 0..36, Cream & Amber Ink)
    # =========================================================================
    cv2.rectangle(canvas, (0, 0), (W, 36), COL_PAPER_DEEP, -1)
    draw_qd_line(canvas, (0, 36), (W, 36), COL_INK, 2, boil_key)
    # Hand-drawn diagonal clapperboard stripes along top edge (y = 0..6)
    for cl_x in range(0, W, 28):
        pts_cl = np.array([[cl_x, 0], [cl_x + 14, 0], [cl_x + 8, 6], [cl_x - 6, 6]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_cl], COL_INK, cv2.LINE_AA)

    badge_col = COL_TAG if spec["group"] == "office" else COL_FOREST_PALE
    draw_qd_card(canvas, 8, 9, 138, 31, fill_bgr=badge_col, shadow=False, boil_seed=boil_key)
    group_lbl = f"STUDIO #{spec['num']:02d}" if spec["group"] == "office" else f"SKILL #{spec['num']-10:02d} ({spec.get('timestamp','')})"
    cv2.putText(canvas, group_lbl, (13, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_INK, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{spec['repo']} // {spec['title'][:42]}", (148, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.39, COL_INK, 1, cv2.LINE_AA)

    # =========================================================================
    # BOTTOM QUICK, DRAW! 35MM FILMSTRIP & X-SHEET BAR (y = 320..360)
    # =========================================================================
    cv2.rectangle(canvas, (0, H - 40), (W, H), COL_PAPER_DEEP, -1)
    draw_qd_line(canvas, (0, H - 40), (W, H - 40), COL_INK, 2, boil_key + 1)
    # Hand-inked 35mm sprocket holes along top & bottom of the X-sheet bar
    for sp_x in range(8, W, 18):
        cv2.rectangle(canvas, (sp_x, H - 37), (sp_x + 8, H - 33), COL_SHEET, -1)
        cv2.rectangle(canvas, (sp_x, H - 37), (sp_x + 8, H - 33), COL_INK, 1)
        cv2.rectangle(canvas, (sp_x, H - 6), (sp_x + 8, H - 2), COL_SHEET, -1)
        cv2.rectangle(canvas, (sp_x, H - 6), (sp_x + 8, H - 2), COL_INK, 1)

    for s_i, st_name in enumerate(spec["stages"]):
        sx0 = 10 + s_i * 156
        is_act = (s_i == stage_idx)
        bg_c = COL_TAG if is_act else (COL_FOREST_PALE if s_i < stage_idx else COL_SHEET)
        draw_qd_card(canvas, sx0, H - 30, sx0 + 150, H - 9, fill_bgr=bg_c, shadow=is_act, boil_seed=boil_key + s_i)
        cv2.putText(canvas, f"{s_i+1}. {st_name[:22]}", (sx0 + 5, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.31, COL_INK, 1, cv2.LINE_AA)

    return canvas


def save_master_qd_studio_wings():
    """Saves the 5 high-resolution Quick, Draw! Studio Wing plates (`1280x720`) to `bts_films/` and `ARTIFACT_DIR`."""
    wing_files = [
        ("bg1_multifloor", "studio_bg_1_multifloor_soundstage.jpg"),
        ("bg2_occupied",   "studio_bg_2_drafting_bullpen.jpg"),
        ("bg3_screening",  "studio_bg_3_foley_screening_stage.jpg"),
        ("bg4_optical",    "studio_bg_4_clocktower_optical_lab.jpg"),
        ("bg5_bullpen",    "studio_bg_5_bullpen_unoccupied.jpg"),
    ]
    for w_key, fname in wing_files:
        base_640 = draw_qd_studio_wing(640, 360, w_key, boil_key=0)
        hi_res = cv2.resize(base_640, (1280, 720), interpolation=cv2.INTER_CUBIC)
        out_p = BTS_DIR / fname
        cv2.imwrite(str(out_p), hi_res)
        shutil.copy2(out_p, ARTIFACT_DIR / fname)


def build_single_bts_film(spec: dict):
    """Builds MP4 (with Foley audio), animated GIF, and poster PNG for one Behind-the-Scenes Studio Film."""
    slug = spec["id"]
    total_frames = 40
    fps = 10
    frames_bgr = [render_studio_film_frame(spec, f_i, total_frames) for f_i in range(total_frames)]

    poster_path = BTS_DIR / f"{slug}_poster.png"
    cv2.imwrite(str(poster_path), frames_bgr[12])

    gif_path = BTS_DIR / f"{slug}.gif"
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr[::2]]
    pil_frames[0].save(
        str(gif_path), save_all=True, append_images=pil_frames[1:], duration=180, loop=0, optimize=True
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
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
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
        "poster_bgr": frames_bgr[12],
    }


def render_contact_sheet(results, title, subtitle, out_name, cols=2):
    rows = math.ceil(len(results) / cols)
    cw, ch = 640, 360
    header_h = 68
    canvas = np.full((header_h + rows * ch, cols * cw, 3), COL_PAPER, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (cols * cw, header_h - 4), COL_TAG_PALE, -1)
    cv2.line(canvas, (0, header_h - 4), (cols * cw, header_h - 4), COL_INK, 2)
    cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, COL_INK, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COL_INK_SOFT, 1, cv2.LINE_AA)

    for idx, r in enumerate(results):
        ri = idx // cols
        ci = idx % cols
        x0 = ci * cw
        y0 = header_h + ri * ch
        canvas[y0:y0 + ch, x0:x0 + cw] = r["poster_bgr"]
        cv2.rectangle(canvas, (x0, y0), (x0 + cw - 1, y0 + ch - 1), COL_INK, 2)

    out_path = PREVIEW_DIR / out_name
    cv2.imwrite(str(out_path), canvas)
    shutil.copy2(out_path, ARTIFACT_DIR / out_name)
    return out_path


def main():
    save_master_qd_studio_wings()
    print("Saved 5 Master Quick, Draw! Studio Soundstage Wing plates.")

    all_specs = AGENT_OFFICE_SPECS + VIDEO_SKILL_SPECS
    results = []
    for sp in all_specs:
        res = build_single_bts_film(sp)
        results.append(res)
        print(f"  Built Unified Quick, Draw! Studio Film [{sp['num']:02d}/18]: {sp['id']} ({sp['studio_wing']})")

    office_res = [r for r in results if r["group"] == "office"]
    skill_res = [r for r in results if r["group"] == "skill"]

    p_off = render_contact_sheet(
        office_res,
        "BEHIND THE SCENES PART I // 10 UNIFIED QUICK, DRAW! STUDIO & WORKTREE FILMS",
        "100% Quick, Draw! Vector Doodles, Gouache Cels & 4-Key Boiling Ink Across All 4 Studio Wings (All 346 Cartoons)",
        "bts_10_agent_offices_atlas.png",
        cols=2,
    )
    p_sk = render_contact_sheet(
        skill_res,
        "BEHIND THE SCENES PART II // 8 UNIFIED QUICK, DRAW! AI VIDEO DIRECTOR SKILL FILMS",
        "All 8 Video & Motion Skills Staged Inside the Quick, Draw! Screening Room, Lightbox Bullpen, Camera Tower & Foley Pit",
        "bts_8_video_skills_atlas.png",
        cols=2,
    )
    print("Rendered Unified Quick, Draw! BTS contact sheets:", p_off.name, p_sk.name)

    for sample_id in ["office-01-agent-office", "office-02-agent-007", "skill-01-huashu-motion", "skill-04-reanimate", "skill-08-earshot"]:
        src_gif = BTS_DIR / f"{sample_id}.gif"
        if src_gif.exists():
            shutil.copy2(src_gif, ARTIFACT_DIR / f"{sample_id}.gif")

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
    print(f"Built Master Quick, Draw! Behind-the-Scenes Film: {master_mp4.name} ({master_mp4.stat().st_size / 1e6:.2f} MB)")

    clean_results = [{k: v for k, v in r.items() if k != "poster_bgr"} for r in results]
    payload = {
        "total_films": len(clean_results),
        "master_film": "Behind_The_Scenes_Multiplane_Studio_Film.mp4",
        "agents": [{k: v for k, v in a.items() if not isinstance(v, tuple)} for a in STUDIO_AGENTS],
        "office_films": [r for r in clean_results if r["group"] == "office"],
        "skill_films": [r for r in clean_results if r["group"] == "skill"],
    }
    (DATA_DIR / "bts_studio_films.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "bts_studio_data.js").write_text("window.BTS_STUDIO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
