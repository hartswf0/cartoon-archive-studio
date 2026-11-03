#!/usr/bin/env python3
"""
tools/behind_the_scenes_studio_engine.py
========================================
Generates the Behind-the-Scenes Multiplane Animation Studio Documentary Film +
18 Bespoke Hand-Drawn Animation Studio Soundstage Films ("make for each"):
  - Part I (10 Virtual Agent Office & Kanban Worktree Studio Soundstages)
  - Part II (8 AI Video-Editing & Motion-Director Studio Soundstages)

Every film is staged inside our 5 hand-painted architectural Animation Studio Soundstages:
  1. studio_bg_1_multifloor_soundstage.jpg (7-Level Multiplane Camera Tower + 3-Floor Cutaway Studio)
  2. studio_bg_2_drafting_bullpen.jpg      (Isometric Animator Bullpen & Storyboard Wall)
  3. studio_bg_3_foley_screening_stage.jpg (Foley Soundstage Pit, Mixing Console & Dailies Screening Theatre)
  4. studio_bg_4_clocktower_optical_lab.jpg(Brass Twin-Reel Optical Printer, Clockwork & 3D Worm Glass Prism)
  5. studio_bg_5_bullpen_unoccupied.jpg    (4 Glowing Lightbox Tables & Bouleau 14-Line Armature Corkboard)
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

# Named Studio Crew (with expressive studio attire, props, and colors in BGR)
STUDIO_AGENTS = [
    {"name": "Maestro", "role": "Studio Director & Kanban Lead", "color": (42, 78, 185), "hat": "beret", "prop": "megaphone"},
    {"name": "Auditor (Codex)", "role": "16-Gate & CLIP Dailies Critic", "color": (55, 135, 75), "hat": "visor", "prop": "loupe"},
    {"name": "Camera Op", "role": "7-Plane Multiplane Rig Operator", "color": (165, 95, 38), "hat": "cap", "prop": "reel"},
    {"name": "Lead Animator", "role": "4-Key Rotoscope & Cel Inker", "color": (135, 62, 158), "hat": "beret", "prop": "brush"},
    {"name": "Earshot Foley", "role": "252-SFX Jerk-Spike Sound Artist", "color": (38, 138, 178), "hat": "headphones", "prop": "clapper"},
    {"name": "Merge Archivist", "role": "Steenbeck Vault & Worktree Keeper", "color": (75, 82, 95), "hat": "visor", "prop": "reel"},
]

AGENT_OFFICE_SPECS = [
    {
        "id": "office-01-agent-office",
        "num": 1,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "STAGE A // 7-PLANE CAMERA TOWER & 3-FLOOR SOUNDSTAGE",
        "repo": "AgentSystemLabs/agent-office",
        "url": "https://github.com/AgentSystemLabs/agent-office",
        "category": "3D Multi-Floor Studio + Worktrees (Closest Kohnny Match)",
        "install": "curl -fsSL https://raw.githubusercontent.com/AgentSystemLabs/agent-office/main/install.sh | bash",
        "title": "3D Multi-Floor Soundstage & Worktree Lightboxes",
        "featured_shot": "giraffe-walks-past",
        "blurb": "Full 3D multi-floor animation soundstage where the towering 7-level Multiplane Camera Rig stands beside three cutaway floors: the Steenbeck 35mm Archive Vault, the 4-Desk Animator Bullpen, and the Storyboard & Espresso Lounge.",
        "stages": ["Vault: 35mm Archive Scan", "Bullpen: 4-Key Lightbox", "Tower: 7-Plane Exposure", "Lounge: Corkboard PR Merge"],
    },
    {
        "id": "office-02-agent-007",
        "num": 2,
        "group": "office",
        "studio_wing": "bg5_bullpen",
        "wing_name": "STAGE B // ISOMETRIC ANIMATOR BULLPEN & STORYBOARD WALL",
        "repo": "bill10/agent-007",
        "url": "https://github.com/bill10/agent-007",
        "category": "Kanban Worktrees + Animator Bullpen + Manager Director",
        "install": "git clone https://github.com/bill10/agent-007",
        "title": "Director Billion & 4-Desk Worktree Bullpen",
        "featured_shot": "train",
        "blurb": "Inside the sunlit isometric bullpen, Director Billion assigns each shot card from the back storyboard corkboard to an isolated wooden drafting desk where glowing circular lightboxes trace the 4 keyframes in onion-skin pencil.",
        "stages": ["Corkboard: Pin Shot Card", "Lightbox: Spawn Worktree", "Projector: Audit 7-Plane", "Steenbeck: Splice to Main"],
    },
    {
        "id": "office-03-claude-office",
        "num": 3,
        "group": "office",
        "studio_wing": "bg2_occupied",
        "wing_name": "STAGE B2 // SPECIALIST INK & PAINT DRAFTING BULLPEN",
        "repo": "W17ant/Claude-Office",
        "url": "https://github.com/W17ant/Claude-Office",
        "category": "Isometric Studio + Specialist Routing + Footage Meters",
        "install": "git clone https://github.com/W17ant/Claude-Office.git && cd Claude-Office && npm i && npm run install-hooks",
        "title": "Specialist Animator Routing & Cel Footage Meters",
        "featured_shot": "lighthouse",
        "blurb": "Four seated specialist animators (SAM Silhouette Tracer, CLIP Critic, Multiplane Compositor, Gouache Background Painter) work under brass gooseneck lamps while hook events route cel sheets between their glowing lightboxes.",
        "stages": ["Hook: PreToolUse Slate", "Route to Cel Specialist", "Approve Gouache Keyframe", "Footage & Token Reel Sync"],
    },
    {
        "id": "office-04-pixel-office",
        "num": 4,
        "group": "office",
        "studio_wing": "bg5_bullpen",
        "wing_name": "STAGE B3 // STORYBOARD ARMATURE WALL & CELEBRATION STAGE",
        "repo": "KalebKE/PixelOffice",
        "url": "https://github.com/KalebKE/PixelOffice",
        "category": "2D Studio (Claude Code + Codex Storyboard Critique)",
        "install": "git clone https://github.com/KalebKE/PixelOffice && cd PixelOffice && python3 hooks/manage_hooks.py install && ./start.sh",
        "title": "Bouleau Storyboard Pitch & 16-Gate Wrap Party",
        "featured_shot": "guitar",
        "blurb": "Animators gather at the Golden Armature corkboard wall during Stage 2-4 storyboarding, return to their tilted wooden lightboxes to ink 4-cycle backgrounds, and celebrate when all 16/16 studio gates pass.",
        "stages": ["Pitch at Storyboard Wall", "Ink 4-Cycle BG at Lightbox", "Screen 16-Gate Dailies", "Studio Wrap Celebration"],
    },
    {
        "id": "office-05-pixel-agents",
        "num": 5,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "STAGE A2 // SPIRAL STAIRCASE & TRANSCRIPT STATE FLOOR",
        "repo": "pixel-agents-hq/pixel-agents",
        "url": "https://github.com/pixel-agents-hq/pixel-agents",
        "category": "A* Studio Floor Pathfinding & Transcript Watcher",
        "install": "git clone https://github.com/pixel-agents-hq/pixel-agents",
        "title": "Studio Floor Pathfinding & Live X-Sheet State Machine",
        "featured_shot": "tractor",
        "blurb": "Watches live JSONL production transcripts and drives animators up and down the iron spiral staircase between the Steenbeck Archive Vault, the 2nd-floor Lightbox Desks, and the 7-Level Multiplane Camera Tower.",
        "stages": ["Tail X-Sheet Transcript", "State: ARCHIVE_STEENBECK", "Climb Spiral Stair to Rig", "State: SHOOTING_7_PLANE"],
    },
    {
        "id": "office-06-my-office",
        "num": 6,
        "group": "office",
        "studio_wing": "bg4_optical",
        "wing_name": "STAGE D // 35MM ARCHIVE VAULT & OPTICAL PRINTER BENCH",
        "repo": "khanh020905/my-office",
        "url": "https://github.com/khanh020905/my-office",
        "category": "3D Archive Film Vault & Rotoscope Bench",
        "install": "git clone https://github.com/khanh020905/my-office",
        "title": "35mm Film Canister Vault & Rotoscope Lightbox Walk",
        "featured_shot": "speedboat",
        "blurb": "Whenever an agent reads archival Quick, Draw! strokes, the animator walks to the 35mm film canister shelves, threads the reel onto the brass twin-reel rotoscope bench, and traces the multiplane cel layers.",
        "stages": ["Pull 35mm Canister Reel", "Thread Twin-Reel Rotoscope", "Trace Quick, Draw! Strokes", "Expose Multiplane Cel"],
    },
    {
        "id": "office-07-workbench",
        "num": 7,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "STAGE A3 // MULTI-FLOOR WORKTREE SPLICING & VAULT",
        "repo": "proactive-agent/workbench",
        "url": "https://github.com/proactive-agent/workbench",
        "category": "Kanban + Worktrees + AI Semantic Film Splicing",
        "install": "git clone https://github.com/proactive-agent/workbench",
        "title": "Per-Shot Worktree Isolation & Steenbeck Semantic Splice",
        "featured_shot": "bicycle",
        "blurb": "Isolates concurrent animation batches on separate studio floors, swaps model harnesses when rate-limited, and splices completed 7-plane film strips together on the ground-floor Steenbeck flatbed editor.",
        "stages": ["Isolate Floor Worktrees", "Swap Animator Crew Pool", "Check Cel Layer Overlap", "Steenbeck Semantic Splice"],
    },
    {
        "id": "office-08-kanban-code",
        "num": 8,
        "group": "office",
        "studio_wing": "bg3_screening",
        "wing_name": "STAGE C // DAILIES SCREENING ROOM & DIRECTOR ALERTS",
        "repo": "langwatch/kanban-code",
        "url": "https://github.com/langwatch/kanban-code",
        "category": "Dailies Screening Theatre + Mobile Director Review",
        "install": "git clone https://github.com/langwatch/kanban-code",
        "title": "35mm Dailies Screening Theatre & Mobile Director Sign-Off",
        "featured_shot": "carrot",
        "blurb": "Links every shot's worktree to the Dailies Screening Theatre where the 35mm projector beams the 4-pass background redraw sheet onto the big screen and alerts the Director's phone for 1-tap approval.",
        "stages": ["Thread Worktree Dailies", "Project 4-Pass BG Sheet", "Ring Screening Bell Alert", "Director Chair Sign-Off"],
    },
    {
        "id": "office-09-vibe-kanban",
        "num": 9,
        "group": "office",
        "studio_wing": "bg2_occupied",
        "wing_name": "STAGE B4 // 10-HARNESS ANIMATOR BULLPEN & CEL DIFFS",
        "repo": "BloopAI/vibe-kanban",
        "url": "https://github.com/BloopAI/vibe-kanban",
        "category": "Multi-Harness Animator Bullpen + Live Stage Preview",
        "install": "npx vibe-kanban",
        "title": "10-Harness Bullpen + Grease-Pencil Cel Diffs & Live Stage",
        "featured_shot": "sandwich",
        "blurb": "Orchestrates 10+ coding harnesses across the drafting bullpen with grease-pencil diff notes written directly on the lightbox glass and live multiplane parallax previews projected on the studio wall.",
        "stages": ["Brief 10 Bullpen Animators", "Project Live 7-Plane Stage", "Grease-Pencil Cel Critique", "FIFO Film Reel Splice"],
    },
    {
        "id": "office-10-claude-code-kanban",
        "num": 10,
        "group": "office",
        "studio_wing": "bg4_optical",
        "wing_name": "STAGE D2 // CLOCKTOWER 55-MIN CACHE COMPACTOR ATELIER",
        "repo": "NikiforovAll/claude-code-kanban",
        "url": "https://github.com/NikiforovAll/claude-code-kanban",
        "category": "Clocktower Atelier + 55-Min Cache Compactor",
        "install": "npm install -g claude-code-kanban && claude-code-kanban --install && claude-code-kanban --open",
        "title": "Brass Clocktower 55-Min Compactor & Before/After Lightbox",
        "featured_shot": "star",
        "blurb": "Under the giant brass studio clockwork, animators compare Before/After cel mockups on the optical printer lightbox while the 55-minute pendulum chime auto-compacts session handoff notes before cache expiry.",
        "stages": ["Lightbox: Before/After Cel", "Optical Bench: 7-Plane Rig", "55-Min Clocktower Chime", "Pneumatic Tube Handoff"],
    },
]

VIDEO_SKILL_SPECS = [
    {
        "id": "skill-01-huashu-motion",
        "num": 11,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "STAGE C1 // PREMIERE SCREENING THEATRE & BEAT CONSOLE",
        "timestamp": "00:46",
        "repo": "alchaincyf/huashu-motion",
        "url": "https://github.com/alchaincyf/huashu-motion",
        "category": "Beat-Synced Premiere Films & Dailies Screening Studio",
        "install": "npx skills add alchaincyf/huashu-motion",
        "title": "Beat-Synced Studio Premiere Film & Screening Console",
        "featured_shot": "giraffe-walks-past",
        "blurb": "Inside the Dailies Screening Theatre, the 35mm projector and analog mixing console cut between our 346 multiplane shots right on the musical downbeat while projecting live CLIP & 16-gate studio charts.",
        "stages": ["Console: Lock Beat Grid", "Projector: 346-Shot Chart", "Screen: Beat-Synced Cuts", "Premiere Reel Master Out"],
    },
    {
        "id": "skill-02-rossark-motion-skill",
        "num": 12,
        "group": "skill",
        "studio_wing": "bg5_bullpen",
        "wing_name": "STAGE B5 // 60 FPS ARC-TRACING LIGHTBOX & CLOCKWORK",
        "timestamp": "03:51",
        "repo": "rossark0/rossark-motion-skill",
        "url": "https://github.com/rossark0/rossark-motion-skill",
        "category": "60 FPS Single-Scene-Clock SVG Character Arcs",
        "install": "npx skills@latest add rossark0/rossark-motion-skill",
        "title": "60 FPS Single-Scene-Clock SVG Arcs on the Lightbox",
        "featured_shot": "bicycle",
        "blurb": "On the tilted wooden lightbox disc, the Lead Animator traces Quick, Draw! SVG strokes along smooth cubic Bezier motion arcs driven by a single deterministic scene clock tau and MOTION.md choreography rules.",
        "stages": ["Pin MOTION.md to Peg Bar", "Lock Single Clock tau", "Trace Cubic Bezier Arcs", "60 FPS Cel Exposure"],
    },
    {
        "id": "skill-03-take-one",
        "num": 13,
        "group": "skill",
        "studio_wing": "bg2_occupied",
        "wing_name": "STAGE B6 // 4-DESK STORY ROOM (STRATEGIST TO CRITIC)",
        "timestamp": "05:03",
        "repo": "talsiach/take-one",
        "url": "https://github.com/talsiach/take-one",
        "category": "Strategist -> Creative Director -> Storyboard -> Critic",
        "install": "git clone https://github.com/talsiach/take-one ~/.claude/skills/take-one",
        "title": "4-Desk Story Bullpen: Strategist, Director, Boarder & Critic",
        "featured_shot": "lighthouse",
        "blurb": "Each of the 4 wooden drafting desks in our bullpen houses one role: Desk 1 writes the visual thesis, Desk 2 locks Bouleau's 14-line armature, Desk 3 inks the 4 keyframes, and Desk 4 runs blind dailies critique.",
        "stages": ["Desk 1: Visual Thesis", "Desk 2: 14-Line Armature", "Desk 3: 4-Key Cel Inking", "Desk 4: Blind Critic Pass"],
    },
    {
        "id": "skill-04-reanimate",
        "num": 14,
        "group": "skill",
        "studio_wing": "bg4_optical",
        "wing_name": "STAGE D3 // TWIN-REEL ROTOSCOPE & OPTICAL DIFF BENCH",
        "timestamp": "06:55",
        "repo": "NiazMorshed2007/reanimate",
        "url": "https://github.com/NiazMorshed2007/reanimate",
        "category": "Twin-Reel Rotoscope Reverse-Engineering & IoU Match",
        "install": "npx skills add NiazMorshed2007/reanimate -g",
        "title": "Twin-Reel Optical Printer & Onion-Skin Silhouette Match",
        "featured_shot": "train",
        "blurb": "Threads the archival 35mm reference reel and our new 7-plane cartoon reel through the brass twin-reel optical printer, superimposing both frames on the glowing rotoscope glass until silhouette IoU >= 0.89.",
        "stages": ["Thread 33 Archival Frames", "Superimpose on Rotoscope", "Refine Keyframe Pose Arc", "Lock Silhouette IoU >= 0.89"],
    },
    {
        "id": "skill-05-argo",
        "num": 15,
        "group": "skill",
        "studio_wing": "bg1_multifloor",
        "wing_name": "STAGE A4 // 7-LEVEL MULTIPLANE RIG & SPOTLIGHT CALLOUTS",
        "timestamp": "09:02",
        "repo": "shreyaskarnik/argo",
        "url": "https://github.com/shreyaskarnik/argo",
        "category": "Multiplane Camera Rig Walkthrough & Spotlight Callouts",
        "install": "npm i -D @argo-video/cli",
        "title": "7-Level Multiplane Camera Rig Tour & Plane Callouts",
        "featured_shot": "guitar",
        "blurb": "Guides the studio camera down all 7 glass platens of the towering Multiplane Camera Rig—spotlighting the Sky Backdrop (P6), Midground Scenery (P3), Hero Character Cel (P2), and Foreground Repoussoir (P0).",
        "stages": ["Rig Top: Camera Lens Check", "Spotlight Planes P6..P3", "Zoom Callout on Hero P2", "Sync Exposure to Narration"],
    },
    {
        "id": "skill-06-explainroo",
        "num": 16,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "STAGE C2 // VOICEOVER BOOTH & SUBTITLE SCREENING STAGE",
        "timestamp": "09:50",
        "repo": "vincentsch/explainroo",
        "url": "https://github.com/vincentsch/explainroo",
        "category": "Voiceover Recording Booth + Word-Timed Screening Titles",
        "install": "git clone https://github.com/vincentsch/explainroo.git",
        "title": "Voiceover Boom Booth & Word-Timed Screening Cards",
        "featured_shot": "diamond",
        "blurb": "Records warm studio narration under the hanging condenser boom mic on the left soundstage while burning word-accurate title cards onto the 35mm dailies projection screen on the right.",
        "stages": ["Script 13-Stage Studio Tour", "Record at Condenser Mic", "Align Whisper Word Cues", "Project Title Cards on Screen"],
    },
    {
        "id": "skill-07-video-director",
        "num": 17,
        "group": "skill",
        "studio_wing": "bg1_multifloor",
        "wing_name": "STAGE A5 // X-SHEET CONTINUITY & STEENBECK QA LAB",
        "timestamp": "11:20",
        "repo": "win4r/video-director",
        "url": "https://github.com/win4r/video-director",
        "category": "Pre-Shoot X-Sheet Check + Steenbeck Post-Render QA",
        "install": "git clone https://github.com/win4r/video-director.git && cp -R video-director/video-director ~/.claude/skills/",
        "title": "Pre-Shoot X-Sheet Check & Steenbeck Loupe QA",
        "featured_shot": "carrot",
        "blurb": "Before shooting on the 7-level Multiplane Tower, the Director validates the exposure X-sheet (`check_timeline.py`), then inspects the developed film on the ground-floor Steenbeck (`qa_video.py`) for flash frames and loudness.",
        "stages": ["Audit Exposure X-Sheet", "Shoot 7-Plane Multiplane", "Inspect Steenbeck Loupe", "Certify LUFS & Print Master"],
    },
    {
        "id": "skill-08-earshot",
        "num": 18,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "STAGE C3 // FOLEY PIT (WOOD/GRAVEL/WATER) & JERK SYNC",
        "timestamp": "12:57",
        "repo": "Leonxlnx/earshot",
        "url": "https://github.com/Leonxlnx/earshot",
        "category": "Foley Pit + 252-SFX Bank Synced to Worm Jerk Spikes",
        "install": "git clone https://github.com/Leonxlnx/earshot.git",
        "title": "Live Foley Pit & 252-SFX Synced to Worm Jerk ||c'''(tau)||",
        "featured_shot": "zigzag",
        "blurb": "Whenever the 3D Spatiotemporal Worm's Kinetic Jerk ||c'''(tau)|| spikes on the screening room monitor, the red ON AIR lamp flashes and our Foley Artist strikes wood, gravel, or water in the floor Foley pit under the boom mic!",
        "stages": ["Track 3D Worm Jerk ||c'''||", "Flash ON AIR at Jerk Peak", "Strike Foley Pit Props", "Mix Console VU & Master"],
    },
]

# Cache for loaded painted studio backgrounds and worm trajectories
_STUDIO_BG_CACHE = {}
_WORM_DATA_CACHE = None


def get_studio_bg(wing_key: str) -> np.ndarray:
    """Loads and caches the 5 hand-painted architectural Animation Studio plates."""
    if not _STUDIO_BG_CACHE:
        mapping = {
            "bg1_multifloor": BTS_DIR / "studio_bg_1_multifloor_soundstage.jpg",
            "bg2_occupied": BTS_DIR / "studio_bg_2_drafting_bullpen.jpg",
            "bg3_screening": BTS_DIR / "studio_bg_3_foley_screening_stage.jpg",
            "bg4_optical": BTS_DIR / "studio_bg_4_clocktower_optical_lab.jpg",
            "bg5_bullpen": BTS_DIR / "studio_bg_5_bullpen_unoccupied.jpg",
        }
        for k, p in mapping.items():
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    # Resize to slightly larger than 640x360 (704x396) so we can pan/dolly smoothly!
                    _STUDIO_BG_CACHE[k] = cv2.resize(img, (704, 396), interpolation=cv2.INTER_AREA)
            if k not in _STUDIO_BG_CACHE:
                _STUDIO_BG_CACHE[k] = np.full((396, 704, 3), (225, 238, 246), dtype=np.uint8)
    return _STUDIO_BG_CACHE.get(wing_key, _STUDIO_BG_CACHE["bg1_multifloor"])


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


def load_shot_keyframe_strip(slug: str):
    """Loads the 4-keyframe sheet or design.png for `slug` so we can project it on the animator's lightbox."""
    for name in ["selected-keys.png", "design.png", "stage_3_keyframes.png"]:
        p = SHOTS_DIR / slug / name
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def draw_wood_frame(canvas, x0, y0, x1, y1, title="", brass_trim=True):
    """Draws a warm hand-inked mahogany wood & brushed brass studio frame (never a flat tech box)."""
    # Drop shadow
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 + 4, y0 + 4), (x1 + 4, y1 + 4), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
    # Outer mahogany frame
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (36, 52, 78), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (18, 22, 28), 2, cv2.LINE_AA)
    if brass_trim:
        cv2.rectangle(canvas, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (75, 168, 218), 1, cv2.LINE_AA)
        # Corner brass rivets
        for rx, ry in [(x0 + 6, y0 + 6), (x1 - 6, y0 + 6), (x0 + 6, y1 - 6), (x1 - 6, y1 - 6)]:
            cv2.circle(canvas, (rx, ry), 2, (95, 195, 240), -1, cv2.LINE_AA)
    if title:
        cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x1 - 4, y0 + 20), (26, 36, 52), -1)
        cv2.putText(canvas, title, (x0 + 9, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (195, 232, 252), 1, cv2.LINE_AA)


def draw_animator_lightbox_disc(canvas, cx, cy, radius, slug, f_idx, label="LIGHTBOX"):
    """Draws a glowing circular frosted-glass animator lightbox disc with brass peg bar & live onion-skin keyframe."""
    # Warm wooden tilted desk rim around the disc
    cv2.circle(canvas, (cx, cy), radius + 6, (58, 96, 142), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius + 6, (22, 28, 36), 2, cv2.LINE_AA)
    # Frosted backlit glass glow (#FFF9E6)
    cv2.circle(canvas, (cx, cy), radius, (226, 248, 255), -1, cv2.LINE_AA)

    # Project actual keyframe drawing onto the circular lightbox in onion-skin ink!
    kf_img = load_shot_keyframe_strip(slug)
    if kf_img is not None:
        kh, kw = kf_img.shape[:2]
        # Crop one of the 4 keyframes based on f_idx
        k_idx = (f_idx // 4) % 4
        if kw >= kh * 2:
            cw = kw // 4
            crop = kf_img[:, k_idx * cw:(k_idx + 1) * cw]
        else:
            crop = kf_img
        d = radius * 2 - 8
        if d > 12 and crop.size > 0:
            thumb = cv2.resize(crop, (d, d), interpolation=cv2.INTER_AREA)
            # Blend inside circular mask
            y0, x0 = cy - d // 2, cx - d // 2
            roi = canvas[max(0, y0):min(canvas.shape[0], y0 + d), max(0, x0):min(canvas.shape[1], x0 + d)]
            if roi.shape[:2] == (d, d):
                mask = np.zeros((d, d), dtype=np.uint8)
                cv2.circle(mask, (d // 2, d // 2), d // 2 - 2, 255, -1)
                blended = cv2.addWeighted(roi, 0.25, thumb, 0.75, 0)
                roi[mask > 0] = blended[mask > 0]

    cv2.circle(canvas, (cx, cy), radius, (45, 85, 125), 1, cv2.LINE_AA)
    # Brass registration peg bar at top of lightbox disc
    cv2.rectangle(canvas, (cx - int(radius * 0.65), cy - radius - 3), (cx + int(radius * 0.65), cy - radius + 4), (72, 168, 218), -1)
    cv2.rectangle(canvas, (cx - int(radius * 0.65), cy - radius - 3), (cx + int(radius * 0.65), cy - radius + 4), (22, 28, 36), 1)
    for px in [-int(radius * 0.4), 0, int(radius * 0.4)]:
        cv2.circle(canvas, (cx + px, cy - radius + 1), 2, (25, 25, 25), -1)

    # Animated dip-pen nib & hand sketching across the lightbox
    pen_x = cx + int(round(math.cos(f_idx * 0.45) * (radius * 0.42)))
    pen_y = cy + int(round(math.sin(f_idx * 0.65) * (radius * 0.35)))
    cv2.line(canvas, (pen_x, pen_y), (pen_x + 18, pen_y - 16), (25, 35, 145), 3, cv2.LINE_AA)
    cv2.circle(canvas, (pen_x, pen_y), 2, (20, 20, 20), -1, cv2.LINE_AA)

    if label:
        cv2.rectangle(canvas, (cx - radius, cy + radius - 11), (cx + radius, cy + radius + 3), (28, 36, 48), -1)
        cv2.putText(canvas, label, (cx - radius + 4, cy + radius - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (215, 240, 255), 1, cv2.LINE_AA)


def draw_studio_crew_member(canvas, x, y, agent, frame_idx=0, scale=0.95, bubble_text=""):
    """Draws an expressive hand-drawn studio animator/director with beret/visor, coat, legs, prop, and speech bubble."""
    col = agent["color"]
    bob = int(round(math.sin(frame_idx * 0.55 + x * 0.04) * 2.5 * scale))
    cy = y + bob
    r_head = int(round(10 * scale))

    # Warm floor shadow
    overlay = canvas.copy()
    cv2.ellipse(overlay, (x, y + int(24 * scale)), (int(13 * scale), int(5 * scale)), 0, 0, 360, (25, 32, 42), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0, canvas)

    # Walking legs
    stride = int(round(math.sin(frame_idx * 0.65 + y * 0.05) * 6.5 * scale))
    cv2.line(canvas, (x - int(4 * scale), cy + int(10 * scale)), (x - int(4 * scale) - stride, y + int(23 * scale)), (22, 26, 32), 2, cv2.LINE_AA)
    cv2.line(canvas, (x + int(4 * scale), cy + int(10 * scale)), (x + int(4 * scale) + stride, y + int(23 * scale)), (22, 26, 32), 2, cv2.LINE_AA)

    # Studio smock / apron torso
    pts_body = np.array([
        [x - int(9 * scale), cy - int(2 * scale)],
        [x + int(9 * scale), cy - int(2 * scale)],
        [x + int(11 * scale), cy + int(12 * scale)],
        [x - int(11 * scale), cy + int(12 * scale)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts_body], col, cv2.LINE_AA)
    cv2.polylines(canvas, [pts_body], True, (18, 20, 24), 2, cv2.LINE_AA)

    # White collar / tie detail
    cv2.line(canvas, (x, cy - int(2 * scale)), (x, cy + int(7 * scale)), (235, 245, 252), 2, cv2.LINE_AA)

    # Animated arm holding studio prop (brush, megaphone, film reel, loupe, clapper)
    arm_y = cy + int(2 * scale) + int(round(math.sin(frame_idx * 0.7) * 3))
    cv2.line(canvas, (x + int(8 * scale), cy + int(1 * scale)), (x + int(17 * scale), arm_y), (18, 20, 24), 2, cv2.LINE_AA)
    prop = agent.get("prop", "brush")
    px, py = x + int(18 * scale), arm_y
    if prop == "megaphone":
        pts_m = np.array([[px, py], [px + 10, py - 5], [px + 10, py + 5]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_m], (55, 175, 235), cv2.LINE_AA)
        cv2.polylines(canvas, [pts_m], True, (18, 20, 24), 1, cv2.LINE_AA)
    elif prop == "reel":
        cv2.circle(canvas, (px + 3, py), 6, (180, 190, 200), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px + 3, py), 6, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.circle(canvas, (px + 3, py), 2, (40, 40, 40), -1, cv2.LINE_AA)
    elif prop == "loupe":
        cv2.circle(canvas, (px + 3, py - 2), 5, (210, 245, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px + 3, py - 2), 5, (35, 145, 195), 2, cv2.LINE_AA)
    else:
        # Paintbrush / pencil with red tip
        cv2.line(canvas, (px - 2, py + 4), (px + 7, py - 6), (45, 95, 165), 2, cv2.LINE_AA)
        cv2.circle(canvas, (px + 7, py - 6), 2, (45, 65, 225), -1, cv2.LINE_AA)

    # Head
    cv2.circle(canvas, (x, cy - int(11 * scale)), r_head, (215, 235, 250), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, cy - int(11 * scale)), r_head, (18, 20, 24), 2, cv2.LINE_AA)
    # Eyes
    cv2.circle(canvas, (x - int(3 * scale), cy - int(12 * scale)), max(1, int(2 * scale)), (18, 20, 24), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x + int(4 * scale), cy - int(12 * scale)), max(1, int(2 * scale)), (18, 20, 24), -1, cv2.LINE_AA)

    # Animator Beret / Director Cap
    cv2.ellipse(canvas, (x, cy - int(20 * scale)), (int(12 * scale), int(5 * scale)), -8, 0, 360, col, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (x, cy - int(20 * scale)), (int(12 * scale), int(5 * scale)), -8, 0, 360, (18, 20, 24), 1, cv2.LINE_AA)

    # Nameplate badge under feet
    name_w = len(agent["name"]) * 5 + 8
    cv2.rectangle(canvas, (x - name_w // 2, y + int(25 * scale)), (x + name_w // 2, y + int(37 * scale)), (28, 34, 44), -1)
    cv2.rectangle(canvas, (x - name_w // 2, y + int(25 * scale)), (x + name_w // 2, y + int(37 * scale)), (75, 168, 218), 1)
    cv2.putText(canvas, agent["name"], (x - name_w // 2 + 4, y + int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (235, 245, 255), 1, cv2.LINE_AA)

    # Optional hand-lettered cream comic callout bubble
    if bubble_text:
        bw = min(148, max(74, len(bubble_text) * 5 + 12))
        bx0 = max(6, min(canvas.shape[1] - bw - 6, x - bw // 2))
        by0 = max(40, cy - int(44 * scale))
        cv2.rectangle(canvas, (bx0, by0), (bx0 + bw, by0 + 16), (225, 246, 255), -1)
        cv2.rectangle(canvas, (bx0, by0), (bx0 + bw, by0 + 16), (25, 30, 38), 1, cv2.LINE_AA)
        cv2.putText(canvas, bubble_text[:26], (bx0 + 4, by0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (20, 24, 30), 1, cv2.LINE_AA)


def draw_3d_worm_prism_widget(canvas, x0, y0, x1, y1, slug, f_idx, title="3D WORM PRISM"):
    """Renders the shot's real 3D Spatiotemporal Worm inside an illuminated brass-and-glass studio display case."""
    # Translucent warm amber glass interior
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (26, 34, 46), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (78, 175, 228), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0 + 3, y0 + 3), (x1 - 3, y1 - 3), (48, 95, 135), 1, cv2.LINE_AA)

    # Draw 3D Isometric Prism wireframe + Bouleau Golden Armature diagonals
    pw, ph = x1 - x0, y1 - y0
    cv2.line(canvas, (x0 + 8, y0 + 18), (x1 - 8, y1 - 14), (48, 118, 168), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0 + 8, y1 - 14), (x1 - 8, y0 + 18), (48, 118, 168), 1, cv2.LINE_AA)
    # Rule of thirds golden filaments
    for frac in (0.333, 0.667):
        gx = x0 + int(pw * frac)
        cv2.line(canvas, (gx, y0 + 16), (gx, y1 - 14), (42, 96, 138), 1, cv2.LINE_AA)

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
            ty = y0 + 20 + int(round((0.5 * v_val + 0.5 * (1.0 - u_val)) * (ph - 36)))
        else:
            ty = (y0 + y1) // 2 + int(round(math.sin(t_i * 0.35) * (ph * 0.22)))
        pts.append([tx, max(y0 + 18, min(y1 - 16, ty))])

    pts_arr = np.array(pts, dtype=np.int32)
    # Outer core-shell glow + inner worm Worldline
    cv2.polylines(canvas, [pts_arr], False, (55, 155, 245), 4, cv2.LINE_AA)
    cv2.polylines(canvas, [pts_arr], False, (145, 245, 190), 2, cv2.LINE_AA)

    cur_t = f_idx % 33
    cx, cy = pts[cur_t]
    cv2.line(canvas, (cx, y0 + 16), (cx, y1 - 14), (95, 205, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 5, (45, 75, 245), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"{title}: {topo}", (x0 + 6, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (185, 232, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"tau={cur_t:02d}/33", (x0 + 6, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (215, 240, 220), 1, cv2.LINE_AA)


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
    Renders one 640x360 frame of a Behind-the-Scenes Studio Film inside our hand-painted
    Animation Studio Soundstages (`studio_bg_1` .. `studio_bg_5`).
    """
    W, H = 640, 360
    wing = spec.get("studio_wing", "bg1_multifloor")
    bg_large = get_studio_bg(wing)  # 704x396

    # Smooth cinematic camera pan across the painted 704x396 studio room over the 40 frames
    pan_phase = math.sin(2.0 * math.pi * (f_idx / float(total_frames)) + spec["num"] * 0.4)
    ox = int(round(32 + pan_phase * 24))
    oy = int(round(18 + math.cos(2.0 * math.pi * (f_idx / float(total_frames))) * 8))
    canvas = bg_large[oy:oy + H, ox:ox + W].copy()

    stage_idx = min(3, int((f_idx / float(total_frames)) * 4))
    stage_progress = (f_idx % (total_frames // 4)) / float(max(1, total_frames // 4 - 1))

    # Load live archival & 7-plane cartoon frames for the featured shot
    arch_fr, cart_fr = load_shot_pair(spec["featured_shot"], f_idx)

    # =========================================================================
    # WING-SPECIFIC ARCHITECTURAL STUDIO EQUIPMENT & ANIMATION
    # =========================================================================
    if wing == "bg1_multifloor":
        # 1. Left: 7-Level Multiplane Camera Tower (x: 85..235, y: 48..315)
        # Golden camera exposure cone firing down through the 7 glass platens!
        cone_overlay = canvas.copy()
        cone_pts = np.array([[162, 54], [98, 305], [226, 305]], dtype=np.int32)
        cv2.fillPoly(cone_overlay, [cone_pts], (145, 225, 255), cv2.LINE_AA)
        flash_alpha = 0.24 if (f_idx % 8 < 2) else 0.11
        cv2.addWeighted(cone_overlay, flash_alpha, canvas, 1.0 - flash_alpha, 0, canvas)

        # Highlight the active glass platen (P6..P0) oscillating in parallax
        plane_ys = [84, 120, 158, 198, 238, 278]
        for p_i, py in enumerate(plane_ys):
            shift_x = int(round(math.sin(f_idx * 0.35 + p_i) * (p_i + 1) * 1.8))
            col_p = (75, 225, 255) if (p_i == f_idx % 6) else (65, 145, 195)
            cv2.rectangle(canvas, (104 + shift_x, py - 4), (218 + shift_x, py + 4), col_p, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"P{6-p_i}", (82, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (25, 35, 50), 1, cv2.LINE_AA)

        # 2. Top-Right Mezzanine Lounge: Hang the Mahogany Studio Dailies Screen (x: 345..626, y: 44..152)
        draw_wood_frame(canvas, 345, 44, 626, 154, f"STUDIO DAILIES SCREEN // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (130, 82))
        cart_sm = cv2.resize(cart_fr, (130, 82))
        canvas[66:148, 353:483] = arch_sm
        canvas[66:148, 489:619] = cart_sm
        cv2.rectangle(canvas, (353, 66), (483, 148), (40, 55, 75), 1)
        cv2.rectangle(canvas, (489, 66), (619, 148), (55, 185, 115), 1)
        cv2.putText(canvas, "35mm ARCHIVE", (358, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE MASTER", (494, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (20, 85, 45), 1, cv2.LINE_AA)

        # 3. Middle Floor Bullpen: Glowing Lightbox Disc + Walking Crew
        draw_animator_lightbox_disc(canvas, 405, 202, 24, spec["featured_shot"], f_idx, "KEY TRACE")
        draw_3d_worm_prism_widget(canvas, 468, 162, 626, 236, spec["featured_shot"], f_idx, "3D WORM")

        # 4. Crew walking across the 3 floors
        crew_wps = [(160, 290), (295, 210), (365, 208), (445, 290)]
        for c_i in range(3):
            ag = STUDIO_AGENTS[(spec["num"] + c_i) % len(STUDIO_AGENTS)]
            p0 = crew_wps[(c_i + stage_idx) % len(crew_wps)]
            p1 = crew_wps[(c_i + stage_idx + 1) % len(crew_wps)]
            cx = int(round((1 - stage_progress) * p0[0] + stage_progress * p1[0]))
            cy = int(round((1 - stage_progress) * p0[1] + stage_progress * p1[1]))
            bub = spec["stages"][stage_idx] if c_i == 0 else ""
            draw_studio_crew_member(canvas, cx, cy, ag, f_idx + c_i * 4, scale=0.85, bubble_text=bub)

    elif wing in ("bg5_bullpen", "bg2_occupied"):
        # Isometric Animator Bullpen & Storyboard Wall!
        # 1. Project live onion-skin keyframes onto two foreground wooden lightbox discs
        draw_animator_lightbox_disc(canvas, 185, 212, 34, spec["featured_shot"], f_idx, "DESK A // K01-K02")
        draw_animator_lightbox_disc(canvas, 332, 258, 32, spec["featured_shot"], f_idx + 8, "DESK B // K03-K04")

        # 2. Mount the Studio Storyboard & Dailies Projection Screen on the right/back wall
        draw_wood_frame(canvas, 352, 44, 628, 162, f"BULLPEN STORYBOARD & DAILIES // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (128, 88))
        cart_sm = cv2.resize(cart_fr, (128, 88))
        canvas[66:154, 360:488] = arch_sm
        canvas[66:154, 492:620] = cart_sm
        cv2.rectangle(canvas, (360, 66), (488, 154), (40, 55, 75), 1)
        cv2.rectangle(canvas, (492, 66), (620, 154), (55, 185, 115), 1)
        cv2.putText(canvas, "ARCHIVE REF", (365, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE CEL", (497, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (20, 85, 45), 1, cv2.LINE_AA)

        # 3. 3D Spatiotemporal Worm & Bouleau Armature Widget in bottom-right
        draw_3d_worm_prism_widget(canvas, 438, 174, 628, 282, spec["featured_shot"], f_idx, "BOULEAU 14-LINE WORM")

        # 4. Studio Crew walking between the 4 drafting desks & storyboard corkboard
        crew_wps = [(110, 255), (258, 185), (265, 268), (402, 252)]
        for c_i in range(3):
            ag = STUDIO_AGENTS[(spec["num"] + c_i) % len(STUDIO_AGENTS)]
            p0 = crew_wps[(c_i + stage_idx) % len(crew_wps)]
            p1 = crew_wps[(c_i + stage_idx + 1) % len(crew_wps)]
            cx = int(round((1 - stage_progress) * p0[0] + stage_progress * p1[0]))
            cy = int(round((1 - stage_progress) * p0[1] + stage_progress * p1[1]))
            bub = spec["stages"][stage_idx] if c_i == 0 else ""
            draw_studio_crew_member(canvas, cx, cy, ag, f_idx + c_i * 5, scale=0.90, bubble_text=bub)

    elif wing == "bg3_screening":
        # Left: Foley Recording Stage (Boom Mic, Prop Table, 3-Surface Pit, Console)
        # Right: Dailies Screening Theatre (35mm Projector Beam, Cinema Screen, Director Chairs)
        # 1. Volumetric golden 35mm projector beam from projector (370, 128) to the Cinema Screen (x: 418..628, y: 48..196)
        beam_overlay = canvas.copy()
        beam_pts = np.array([[372, 126], [420, 52], [420, 192]], dtype=np.int32)
        cv2.fillPoly(beam_overlay, [beam_pts], (155, 232, 255), cv2.LINE_AA)
        cv2.addWeighted(beam_overlay, 0.28, canvas, 0.72, 0, canvas)

        # 2. Hang the Grand Dailies Cinema Screen on the right theatre wall (x: 416..628, y: 46..196)
        draw_wood_frame(canvas, 416, 46, 628, 196, f"SCREENING ROOM THEATRE // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (96, 72))
        cart_lg = cv2.resize(cart_fr, (100, 72))
        canvas[70:142, 424:520] = arch_sm
        canvas[70:142, 522:622] = cart_lg
        cv2.rectangle(canvas, (424, 70), (520, 142), (45, 60, 80), 1)
        cv2.rectangle(canvas, (522, 70), (622, 142), (65, 205, 125), 1)
        cv2.putText(canvas, "35mm ARCHIVE", (428, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE FILM", (526, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (20, 80, 40), 1, cv2.LINE_AA)
        # Earshot Jerk / Audio Waveform inside bottom of Cinema Screen
        for w_i in range(38):
            wx = 426 + w_i * 5
            wh = int(round(abs(math.sin(w_i * 0.45 + f_idx * 0.55)) * 18)) + 3
            wcol = (55, 85, 245) if wh > 15 else (85, 215, 145)
            cv2.line(canvas, (wx, 184), (wx, 184 - wh), wcol, 2, cv2.LINE_AA)
        cv2.putText(canvas, "EARSHOT JERK & AUDIO SYNC", (428, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 235, 255), 1, cv2.LINE_AA)

        # 3. Glowing red "ON AIR // FOLEY SYNC" studio lamp on left soundstage wall
        is_jerk_peak = (f_idx % 7 < 3)
        lamp_col = (45, 55, 245) if is_jerk_peak else (45, 45, 110)
        cv2.rectangle(canvas, (24, 46), (148, 72), (25, 28, 34), -1)
        cv2.rectangle(canvas, (24, 46), (148, 72), lamp_col, 2)
        cv2.circle(canvas, (38, 59), 6, lamp_col, -1, cv2.LINE_AA)
        cv2.putText(canvas, "ON AIR // FOLEY", (50, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (235, 240, 255), 1, cv2.LINE_AA)

        # Acoustic ripple rings over the 3-Surface Foley Pit (wood/gravel/water) when Jerk spikes!
        pit_centers = [(105, 278), (175, 278), (235, 278)]
        pc = pit_centers[stage_idx % 3]
        for r_i in range(1, 3):
            rad = ((f_idx * 4 + r_i * 10) % 28) + 6
            cv2.circle(canvas, pc, rad, (65, 195, 255), 1, cv2.LINE_AA)

        # Mini Lightbox on the Foley table + Crew in Foley Pit & Screening Room
        draw_animator_lightbox_disc(canvas, 68, 198, 25, spec["featured_shot"], f_idx, "CUE SHEET")
        draw_studio_crew_member(canvas, 172, 238, STUDIO_AGENTS[4], f_idx, scale=0.92, bubble_text=spec["stages"][stage_idx])
        draw_studio_crew_member(canvas, 335, 262, STUDIO_AGENTS[0], f_idx + 6, scale=0.88, bubble_text="Dailies Cut!")

    else:
        # Wing D: `bg4_optical` — Brass Twin-Reel Optical Printer, Clockwork & 3D Worm Glass Cube!
        # 1. Animate spinning spokes on the two giant brass 35mm optical printer reels (78, 102) & (172, 100)
        for rx, ry in [(78, 102), (172, 100)]:
            for sp_i in range(4):
                ang = f_idx * 0.28 + sp_i * (math.pi / 2.0)
                sx = rx + int(round(math.cos(ang) * 24))
                sy = ry + int(round(math.sin(ang) * 24))
                cv2.line(canvas, (rx, ry), (sx, sy), (42, 78, 118), 2, cv2.LINE_AA)

        # 2. Superimpose the live onion-skin Archival + Cartoon comparison inside the bottom-left Rotoscope Lightbox!
        draw_wood_frame(canvas, 22, 226, 256, 316, f"ROTOSCOPE BENCH // {spec['featured_shot'].upper()}")
        arch_sm = cv2.resize(arch_fr, (108, 64))
        cart_sm = cv2.resize(cart_fr, (108, 64))
        canvas[248:312, 28:136] = arch_sm
        canvas[248:312, 142:250] = cart_sm
        cv2.putText(canvas, "ARCHIVE", (32, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(canvas, "7-PLANE", (146, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (20, 80, 40), 1, cv2.LINE_AA)

        # 3. Render the live 3D Spatiotemporal Worm inside the Illuminated Glass Prism Cube on the right pedestal!
        draw_3d_worm_prism_widget(canvas, 442, 136, 568, 256, spec["featured_shot"], f_idx, "OPTICAL WORM CUBE")

        # 4. Studio Crew operating the Optical Printer & Clocktower
        draw_studio_crew_member(canvas, 305, 258, STUDIO_AGENTS[2], f_idx, scale=0.92, bubble_text=spec["stages"][stage_idx])
        draw_studio_crew_member(canvas, 402, 264, STUDIO_AGENTS[1], f_idx + 5, scale=0.88, bubble_text="IoU >= 0.89!")

    # =========================================================================
    # TOP WOODEN CLAPPERBOARD SLATE & STAGE PEG-BAR TABS (y = 0..38)
    # =========================================================================
    cv2.rectangle(canvas, (0, 0), (W, 38), (22, 26, 34), -1)
    cv2.line(canvas, (0, 38), (W, 38), (78, 175, 228), 2)
    # Diagonal black-and-cream clapperboard stripes on top border (y = 0..6)
    for cl_x in range(0, W, 28):
        pts_cl = np.array([[cl_x, 0], [cl_x + 14, 0], [cl_x + 8, 6], [cl_x - 6, 6]], dtype=np.int32)
        cv2.fillPoly(canvas, [pts_cl], (215, 235, 248), cv2.LINE_AA)

    tag_col = (55, 155, 240) if spec["group"] == "office" else (85, 195, 115)
    cv2.rectangle(canvas, (8, 10), (132, 32), tag_col, -1)
    group_lbl = f"STUDIO #{spec['num']:02d}" if spec["group"] == "office" else f"SKILL #{spec['num']-10:02d} ({spec.get('timestamp','')})"
    cv2.putText(canvas, group_lbl, (13, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (15, 15, 15), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{spec['repo']} // {spec['title'][:42]}", (140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (235, 244, 252), 1, cv2.LINE_AA)

    # =========================================================================
    # BOTTOM 35MM FILMSTRIP & ANIMATION X-SHEET EXPOSURE BAR (y = 320..360)
    # =========================================================================
    cv2.rectangle(canvas, (0, H - 40), (W, H), (20, 24, 30), -1)
    cv2.line(canvas, (0, H - 40), (W, H - 40), (78, 175, 228), 1)
    # 35mm film sprocket holes along top & bottom of the X-sheet bar
    for sp_x in range(8, W, 18):
        cv2.rectangle(canvas, (sp_x, H - 37), (sp_x + 8, H - 33), (215, 232, 245), -1)
        cv2.rectangle(canvas, (sp_x, H - 6), (sp_x + 8, H - 2), (215, 232, 245), -1)

    # 4 Brass Stage Tabs along the X-Sheet bar
    for s_i, st_name in enumerate(spec["stages"]):
        sx0 = 10 + s_i * 156
        is_act = (s_i == stage_idx)
        bg_c = (62, 155, 225) if is_act else ((42, 85, 62) if s_i < stage_idx else (36, 42, 52))
        txt_c = (15, 18, 22) if is_act else (220, 232, 240)
        cv2.rectangle(canvas, (sx0, H - 30), (sx0 + 150, H - 10), bg_c, -1)
        cv2.rectangle(canvas, (sx0, H - 30), (sx0 + 150, H - 10), (95, 185, 235) if is_act else (65, 75, 88), 1)
        cv2.putText(canvas, f"{s_i+1}. {st_name[:22]}", (sx0 + 5, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.31, txt_c, 1, cv2.LINE_AA)

    return canvas


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
    canvas = np.full((header_h + rows * ch, cols * cw, 3), (20, 24, 30), dtype=np.uint8)
    cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (215, 238, 252), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (175, 215, 195), 1, cv2.LINE_AA)

    for idx, r in enumerate(results):
        ri = idx // cols
        ci = idx % cols
        x0 = ci * cw
        y0 = header_h + ri * ch
        canvas[y0:y0 + ch, x0:x0 + cw] = r["poster_bgr"]
        cv2.rectangle(canvas, (x0, y0), (x0 + cw - 1, y0 + ch - 1), (75, 165, 215), 2)

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
        print(f"  Built BTS Studio Soundstage Film [{sp['num']:02d}/18]: {sp['id']} ({sp['studio_wing']})")

    office_res = [r for r in results if r["group"] == "office"]
    skill_res = [r for r in results if r["group"] == "skill"]

    p_off = render_contact_sheet(
        office_res,
        "BEHIND THE SCENES PART I // 10 ANIMATION STUDIO & WORKTREE SOUNDSTAGE FILMS",
        "Hand-Painted Multi-Floor Multiplane Tower, Isometric Animator Bullpen, Foley Stage & Optical Lab (All 346 Cartoons)",
        "bts_10_agent_offices_atlas.png",
        cols=2,
    )
    p_sk = render_contact_sheet(
        skill_res,
        "BEHIND THE SCENES PART II // 8 AI VIDEO DIRECTOR & MOTION SKILL SOUNDSTAGE FILMS",
        "All 8 Video & Motion Skills Staged Inside the Screening Room, Lightbox Bullpen, Multiplane Tower & Foley Pit",
        "bts_8_video_skills_atlas.png",
        cols=2,
    )
    print("Rendered BTS contact sheets:", p_off.name, p_sk.name)

    # Copy sample animated GIFs to artifact dir for visual inspection
    for sample_id in ["office-01-agent-office", "office-02-agent-007", "skill-01-huashu-motion", "skill-04-reanimate", "skill-08-earshot"]:
        src_gif = BTS_DIR / f"{sample_id}.gif"
        if src_gif.exists():
            shutil.copy2(src_gif, ARTIFACT_DIR / f"{sample_id}.gif")

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
