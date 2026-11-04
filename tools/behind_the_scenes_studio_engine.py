#!/usr/bin/env python3
"""
Secondhand Studios: The Autonomous Agent Animation Bullpen (4-Act Studio Film Engine)
======================================================================================
Every Behind-the-Scenes episode is an animated 4-Scene Studio Film where we watch our
cast of 5 Agent Animators physically create the cartoon step-by-step inside the studio:

  - SCENE 1 (Act I, 0.0s - 3.0s): THE STORYBOARD & MOVIOLA ROOM
    Lead Agent (`Nib`, `Blue-Line`, `Cel-Inker`, `Sprocket`, or `Chime`) & Co-Worker Agent
    study the live `4:3` Archival Moviola Monitor (`240x180`, exact 4:3) as the raw reference
    plays, while pinning and sketching the 4 keyframe poses (`K1..K4`) onto the Corkboard
    Storyboard Wall!

  - SCENE 2 (Act II, 3.0s - 6.0s): THE 4-DESK WORKTREE DRAFTING BULLPEN
    Four Agent Animators sit at 4 tilted, glowing Lightbox Desks (`Worktree Desk #1..#4`).
    Above each animator's desk is their `4:3` Keyframe Canvas (`192x144`, exact 4:3).
    As their hands draw on the lightboxes, we watch all 4 keyframes transform step-by-step:
      Step A: Blue Non-Photo Pencil Rough -> Step B: Black Speedball Ink -> Step C: Gouache Cel!

  - SCENE 3 (Act III, 6.0s - 9.0s): THE 7-LEVEL MULTIPLANE CAMERA TOWER
    `Cel-Inker` slides the painted glass planes (`P6 Sky 0.2x`, `P4 Midground 0.5x`, `P2 Hero 1.0x`)
    into the towering Multiplane Camera Rig while `Nib` turns the rotating brass hand-crank
    wheel and golden exposure rays flash down the tower. Beside the rig, the `4:3` Multiplane
    Viewfinder Monitor (`340x255`, exact 4:3) shows the layers stacking from background wash
    to hero cel to full 7-plane parallax!

  - SCENE 4 (Act IV, 9.0s - 12.0s): THE 35MM DAILIES SCREENING THEATRE & FOLEY STAGE
    Inside the darkened Screening Theatre, `Sprocket`'s spinning Twin-Reel 35mm Projector
    beams a golden cone of light onto the `4:3` Theatre Silver Screen (`440x330`, exact 4:3)
    playing the finished 33-frame cartoon in sync with the `4:3` Booth Archival Monitor
    (`180x135`, exact 4:3), while `Chime` strikes Foley sound effects on motion peaks and
    `Nib` & `Blue-Line` watch from the front-row director chairs!
"""

import json
import math
import shutil
import subprocess
import textwrap
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hand_animated_code_rig import (
    build_10x_curved_arc_strokes,
    draw_code_hand_animated_agent,
    draw_code_animated_studio_apparatus,
    draw_code_animated_macro_inset,
    enhance_hero_board_with_code_multiplane,
    generate_3100_frame_flipbook_atlases,
)

ROOT = Path(__file__).resolve().parents[1]
BTS_DIR = ROOT / "bts_films"
PREVIEW_DIR = ROOT / "preview"
DATA_DIR = ROOT / "data"
QD_DIR = ROOT / "quickdraw_samples"
SHOTS_DIR = ROOT / "shots"
ARTIFACT_DIR = Path("/Users/gaia/.gemini/antigravity/brain/113f482f-cd73-450c-be83-dd0bcb11307b")

BTS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# UNIFIED PAPER, INK & GOUACHE STUDIO PALETTE (BGR for OpenCV, RGB for PIL)
# =============================================================================
BGR_PAPER = (227, 239, 246)       # #F6EFE3 Warm archival animation bond paper
BGR_GUTTER = (244, 251, 255)      # #FFFBF4 Crisp comic gutter & speech balloon white-cream
BGR_PAPER_DARK = (206, 226, 236)  # #ECE2CE Shaded drafting board & card backing
BGR_STAGE_DARK = (38, 34, 30)     # #1E2226 Darkened screening room / multiplane stage wall
BGR_INK = (18, 20, 22)            # #161412 Bold cartoonist Speedball India ink
BGR_GRAPHITE = (102, 112, 122)    # #7A7066 Pencil construction lines
BGR_NONPHOTO_BLUE = (210, 145, 65)  # Animator's non-photo blue rough pencil (BGR)

BGR_AMBER = (58, 159, 239)        # #EF9F3A Ochre gold highlight
BGR_AMBER_PALE = (194, 229, 251)  # #FBE5C2 Caption bar warm gouache wash
BGR_RED = (41, 74, 201)           # #C94A29 Terracotta red badge / Nib's apron
BGR_CORAL_PALE = (206, 216, 247)  # #F7D8CE Soft coral gouache wash
BGR_BLUE = (132, 91, 43)          # #2B5B84 Prussian blue / Blue-Line's smock
BGR_BLUE_PALE = (240, 228, 214)   # #D6E4F0 Sky plane gouache wash
BGR_GREEN = (79, 107, 47)         # #2F6B4F Forest green / Sprocket's vest
BGR_MINT_PALE = (220, 232, 212)   # #D4E8DC Meadow plane gouache wash
BGR_INK_SOFT = BGR_GRAPHITE
BGR_RED_PALE = BGR_CORAL_PALE
BGR_GREEN_PALE = BGR_MINT_PALE


def load_comic_font(size: int, style: str = "bold") -> ImageFont.FreeTypeFont:
    candidates = {
        "display": [
            "/System/Library/Fonts/Supplemental/Futura.ttc",
            "/System/Library/Fonts/Supplemental/Impact.ttf",
            "/System/Library/Fonts/HelveticaNeue.ttc",
        ],
        "comic": [
            "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
            "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf",
            "/System/Library/Fonts/Supplemental/Futura.ttc",
        ],
        "bold": [
            "/System/Library/Fonts/Supplemental/Futura.ttc",
            "/System/Library/Fonts/HelveticaNeue.ttc",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        ],
    }
    for path in candidates.get(style, candidates["bold"]):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_DISPLAY_XL = load_comic_font(26, "display")
FONT_DISPLAY_LG = load_comic_font(21, "display")
FONT_DISPLAY_MD = load_comic_font(17, "display")
FONT_SPEECH_LG = load_comic_font(19, "comic")
FONT_SPEECH_MD = load_comic_font(16, "comic")
FONT_CAPTION_LG = load_comic_font(18, "bold")
FONT_CAPTION_MD = load_comic_font(15, "bold")
FONT_CAPTION_SM = load_comic_font(13, "comic")
FONT_LABEL_MD = load_comic_font(15, "bold")
FONT_LABEL_SM = load_comic_font(12, "bold")


def fit_text_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, preferred_fonts: list) -> ImageFont.FreeTypeFont:
    for f in preferred_fonts:
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_w:
            return f
    for sz in range(14, 8, -1):
        f = load_comic_font(sz, "bold")
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_w:
            return f
    return load_comic_font(9, "bold")


# =============================================================================
# THE CAST OF 5 ORIGINAL AGENT ANIMATORS
# =============================================================================
STUDIO_AGENTS = {
    "nib": {
        "id": "nib",
        "name": "Nib (Showrunner)",
        "short_name": "NIB",
        "role": "Fountain-Nib Maestro & Showrunner",
        "dept": "MULTIPLANE DIRECTION",
        "color_bgr": BGR_RED,
        "color_rgb": (201, 74, 41),
        "pale_bgr": BGR_CORAL_PALE,
    },
    "blueline": {
        "id": "blueline",
        "name": "Blue-Line (Key Inker)",
        "short_name": "BLUE-LINE",
        "role": "Worktree Keyframe Architect (K1..K4)",
        "dept": "WORKTREE KEY INKING",
        "color_bgr": BGR_BLUE,
        "color_rgb": (43, 91, 132),
        "pale_bgr": BGR_BLUE_PALE,
    },
    "celinker": {
        "id": "celinker",
        "name": "Cel-Inker (Painter)",
        "short_name": "CEL-INKER",
        "role": "7-Plane Gouache Painter & Rigger",
        "dept": "7-PLANE GOUACHE RIG",
        "color_bgr": BGR_AMBER,
        "color_rgb": (239, 159, 58),
        "pale_bgr": BGR_AMBER_PALE,
    },
    "sprocket": {
        "id": "sprocket",
        "name": "Sprocket (Auditor)",
        "short_name": "SPROCKET",
        "role": "35mm Dailies Critic & 16-Gate Auditor",
        "dept": "16-GATE DAILIES AUDIT",
        "color_bgr": BGR_GREEN,
        "color_rgb": (47, 107, 79),
        "pale_bgr": BGR_MINT_PALE,
    },
    "chime": {
        "id": "chime",
        "name": "Chime (Foley & Continuity)",
        "short_name": "CHIME",
        "role": "Foley Percussionist & 55m X-Sheet Keeper",
        "dept": "FOLEY & X-SHEET SYNC",
        "color_bgr": BGR_RED,
        "color_rgb": (201, 74, 41),
        "pale_bgr": BGR_AMBER_PALE,
    },
}

# =============================================================================
# 18 EPISODES: EACH EPISODE IS A 4-SCENE FILM OF THE AGENT STUDIO IN ACTION
# =============================================================================
AGENT_OFFICE_SPECS = [
    {
        "id": "office-01-agent-office",
        "num": 1,
        "group": "office",
        "lead_agent": "nib",
        "cameo_agent": "blueline",
        "studio_wing": "bg1_multifloor",
        "wing_name": "EPISODE 01 // THE AGENT ANIMATOR STUDIO",
        "repo": "Ks1030/Agent-Office",
        "url": "https://github.com/Ks1030/Agent-Office",
        "category": "Multiplane Agent Studio Architecture",
        "install": "git clone https://github.com/Ks1030/Agent-Office.git && cd Agent-Office && npm install && npm run dev",
        "title": "Inside Our Autonomous Agent Animation Studio",
        "featured_shot": "giraffe-walks-past",
        "speech_p1": "WATCH OUR AGENT CREW TURN THIS ARCHIVAL GIRAFFE CLIP INTO A 7-PLANE CARTOON ACROSS FOUR STUDIO ROOMS!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS DRAW KEYFRAMES K1..K4 AT LIGHTBOX DESKS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): CEL-INKER STACKS 7 GLASS PLANES WHILE NIB CRANKS THE CAMERA!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): SPROCKET & CHIME PROJECT & FOLEY-SYNC THE FINISHED 4:3 FILM!",
        "blurb": "Follow Showrunner Nib, Key Inker Blue-Line, Gouache Rigger Cel-Inker, Auditor Sprocket, and Foley Keeper Chime across all 4 studio rooms as they turn archival footage of a walking giraffe into a 7-plane cartoon.",
        "stages": ["Room 1: Moviola & Storyboard", "Room 2: 4-Desk Worktree Bullpen", "Room 3: 7-Plane Camera Tower", "Room 4: 35mm Screening & Foley"],
    },
    {
        "id": "office-02-agent-007",
        "num": 2,
        "group": "office",
        "lead_agent": "blueline",
        "cameo_agent": "nib",
        "studio_wing": "bg5_bullpen",
        "wing_name": "EPISODE 02 // ISOLATED WORKTREE BOOTHS",
        "repo": "Tryilen/agent-007",
        "url": "https://github.com/Tryilen/agent-007",
        "category": "Isolated Git Worktree Drafting Desks",
        "install": "git clone https://github.com/Tryilen/agent-007.git && cd agent-007 && bun install && bun run dev",
        "title": "Four Parallel Inker Agents at Isolated Worktree Desks",
        "featured_shot": "train",
        "speech_p1": "TO ANIMATE THIS STEAM TRAIN WITHOUT SMUDGING EACH OTHER'S INK, WE GIVE EVERY AGENT ITS OWN WORKTREE DESK!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 AGENTS INK POSES K1..K4 IN PARALLEL AT ISOLATED LIGHTBOXES!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): CEL-INKER LOADS THE 4 WORKTREE CELS INTO THE 7-PLANE RIG!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): THE SPLICED 4:3 TRAIN REEL ROLLS WITH ZERO MERGE CONFLICTS!",
        "blurb": "Watch Keyframe Architect Blue-Line and three subagent inkers work simultaneously at 4 isolated git worktree lightbox desks (`DESK K1..K4`), transforming blue roughs into painted train cels.",
        "stages": ["Room 1: Train Moviola Trace", "Room 2: 4 Isolated Worktree Desks", "Room 3: Multiplane Platen Stack", "Room 4: Zero-Conflict Screening"],
    },
    {
        "id": "office-03-claude-office",
        "num": 3,
        "group": "office",
        "lead_agent": "celinker",
        "cameo_agent": "blueline",
        "studio_wing": "bg5_bullpen",
        "wing_name": "EPISODE 03 // THE STORYBOARD ARMATURE WALL",
        "repo": "paulrobello/claude-office",
        "url": "https://github.com/paulrobello/claude-office",
        "category": "Storyboard Pin-Up & Golden Ratio Armature",
        "install": "git clone https://github.com/paulrobello/claude-office.git && cd claude-office && just install && just start",
        "title": "Pinning the Lighthouse Horizon Before Inking",
        "featured_shot": "lighthouse",
        "speech_p1": "LOOK AT THE MOVIOLA! BEFORE WE INK THE LIGHTHOUSE, BLUE-LINE PINS ALL 4 POSES TO OUR GOLDEN RATIO WALL!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS TRACE THE ROTATING BEAM ONTO 4:3 LIGHTBOX SHEETS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): CEL-INKER SEPARATES SKY (0.2x), CLIFF (0.5x), AND TOWER (1.0x)!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): THE LIGHTHOUSE BEAM SWEEPS ACROSS OUR 4:3 SILVER SCREEN!",
        "blurb": "Cel-Inker and Blue-Line show how the storyboard armature wall locks the coastal horizon before the 4 worktree inkers and 7-plane camera rig bring the lighthouse to life.",
        "stages": ["Room 1: Golden Ratio Pin-Up", "Room 2: 4-Desk Beam Inking", "Room 3: Cliff & Sky Parallax Rig", "Room 4: Coastal Dailies Screening"],
    },
    {
        "id": "office-04-pixel-office",
        "num": 4,
        "group": "office",
        "lead_agent": "sprocket",
        "cameo_agent": "chime",
        "studio_wing": "bg3_screening",
        "wing_name": "EPISODE 04 // PEER-AUDITED DAILIES",
        "repo": "feedingantioch/pixel-office",
        "url": "https://github.com/feedingantioch/pixel-office",
        "category": "Cross-Agent Dailies Critique (Sprocket Auditing Blue-Line)",
        "install": "git clone https://github.com/feedingantioch/pixel-office.git && cd pixel-office && npm install && npm run build",
        "title": "Why One Agent Crew Draws and Another Agent Audits",
        "featured_shot": "guitar",
        "speech_p1": "IN OUR STUDIO, THE AGENTS WHO INK A SHOT NEVER GRADE THEIR OWN REEL! WATCH HOW WE AUDIT SHOT #136 GUITAR!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): BLUE-LINE'S CREW INKS & PAINTS THE 4 GUITAR KEYFRAMES AT THEIR DESKS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): CEL-INKER & NIB COMPOSITE THE STAGE FLOOR AND HERO GUITAR PLANES!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): SPROCKET COMPARE-AUDITS BOTH 4:3 SCREENS & STAMPS 16/16 PASS!",
        "blurb": "Auditor Sprocket takes us from the Moviola room through the 4-desk bullpen and Multiplane Tower into the 35mm Screening Theatre to show how cross-agent peer review guarantees 16/16 quality gates.",
        "stages": ["Room 1: Archival Guitar Study", "Room 2: 4-Desk Strum Keyframes", "Room 3: Stage & Hero Platen Rig", "Room 4: Sprocket's 16-Gate Audit"],
    },
    {
        "id": "office-05-pixel-agents",
        "num": 5,
        "group": "office",
        "lead_agent": "blueline",
        "cameo_agent": "celinker",
        "studio_wing": "bg5_bullpen",
        "wing_name": "EPISODE 05 // FOUR INKERS, LIVING LINE BOIL",
        "repo": "pablodelucca/pixel-agents",
        "url": "https://github.com/pablodelucca/pixel-agents",
        "category": "Parallel Subagent Inkers & Hand-Drawn Boil",
        "install": "git clone https://github.com/pablodelucca/pixel-agents.git && cd pixel-agents && npm install",
        "title": "How Four Subagent Inkers Create Living Line Boil",
        "featured_shot": "tractor",
        "speech_p1": "SLIDING ONE STATIC DRAWING LOOKS DEAD! WATCH FOUR SUBAGENT INKERS REDRAW THE TRACTOR AT FOUR DESKS!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): EACH AGENT REDRAWS THE TRACTOR FRESH FROM ROUGH TO INK TO GOUACHE!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): STACKING THE FURROWED FIELD, TREES, AND BOILING TRACTOR CELS!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): THE TRACTOR CHUGS ACROSS THE 4:3 SCREEN WITH LIVING HAND-DRAWN BOIL!",
        "blurb": "Watch 4 subagent inkers at their 4 glowing lightbox desks redraw keyframes K1, K2, K3, and K4 from blue roughs to black ink so the tractor breathes with authentic hand-drawn line boil.",
        "stages": ["Room 1: Tractor Motion Trace", "Room 2: 4 Parallel Subagent Desks", "Room 3: Farm Field Parallax Rig", "Room 4: Living Line-Boil Screening"],
    },
    {
        "id": "office-06-my-office",
        "num": 6,
        "group": "office",
        "lead_agent": "chime",
        "cameo_agent": "nib",
        "studio_wing": "bg4_clocktower",
        "wing_name": "EPISODE 06 // THE 55-MINUTE EXPOSURE SHEET",
        "repo": "m1heng/my-office",
        "url": "https://github.com/m1heng/my-office",
        "category": "55-Minute Cache Continuity & X-Sheet Handoff",
        "install": "git clone https://github.com/m1heng/my-office.git && cd my-office && npm install && npm run build",
        "title": "Passing the Exposure Sheet Across Agent Shifts",
        "featured_shot": "speedboat",
        "speech_p1": "WHEN AN AGENT SHIFT REACHES 55 MINUTES, CHIME LOGS THE EXPOSURE SHEET SO THE NEXT CREW NEVER LOSES THE THREAD!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): THE NEW AGENT SHIFT READS THE X-SHEET & INKS ALL 4 SPEEDBOAT KEYS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): RIGGING SHORELINE PINES (0.4x), RIVER WAKE (0.8x), AND SPEEDBOAT (1.0x)!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): SEAMLESS 4:3 CONTINUITY AS THE SPEEDBOAT PLANES ACROSS THE WATER!",
        "blurb": "Chime and Nib show how logging palette swatches, plane depths, and keyframe arcs onto the studio Exposure Sheet preserves unbroken continuity across 55-minute agent shift handoffs.",
        "stages": ["Room 1: 55m X-Sheet Handoff", "Room 2: Shift-2 Lightbox Inking", "Room 3: River Wake Multiplane Rig", "Room 4: Continuity Screening"],
    },
    {
        "id": "office-07-workbench",
        "num": 7,
        "group": "office",
        "lead_agent": "celinker",
        "cameo_agent": "sprocket",
        "studio_wing": "bg1_multifloor",
        "wing_name": "EPISODE 07 // INSIDE THE MULTIPLANE RIG",
        "repo": "taw0002/Claude-Code-Agent-Monitor-Workbench",
        "url": "https://github.com/taw0002/Claude-Code-Agent-Monitor-Workbench",
        "category": "7-Level Multiplane Camera & Depth Parallax",
        "install": "git clone https://github.com/taw0002/Claude-Code-Agent-Monitor-Workbench",
        "title": "Cranking 7 Glass Levels for True Multiplane Parallax",
        "featured_shot": "bicycle",
        "speech_p1": "HOW DO WE MAKE FLAT PAPER FEEL THREE-DIMENSIONAL? FOLLOW ME FROM THE LIGHTBOXES TO OUR 7-PLANE CAMERA TOWER!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS DRAW THE BICYCLE PEDAL CYCLE ACROSS DESKS K1..K4!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): CEL-INKER SLIDES SKY (0.2x), TREES (0.5x) & BICYCLE (1.0x) UNDER THE LENS!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): ON THE 4:3 SCREEN, CLOUDS, TREES & BICYCLE GLIDE IN REAL DEPTH!",
        "blurb": "Cel-Inker leads a tour from the 4-desk bullpen up the 7-level Multiplane Camera Tower, showing how each glass platen moves at its own parallax speed under the overhead lens.",
        "stages": ["Room 1: Parallax Depth Plan", "Room 2: 4-Desk Bicycle Keys", "Room 3: 7-Glass-Platen Camera Rig", "Room 4: Multiplane Depth Dailies"],
    },
    {
        "id": "office-08-kanban-code",
        "num": 8,
        "group": "office",
        "lead_agent": "nib",
        "cameo_agent": "blueline",
        "studio_wing": "bg5_bullpen",
        "wing_name": "EPISODE 08 // THE 4-ROOM STUDIO KANBAN",
        "repo": "langoustine69/kanban-code",
        "url": "https://github.com/langoustine69/kanban-code",
        "category": "4-Stage Studio Production Kanban",
        "install": "git clone https://github.com/langoustine69/kanban-code && cd kanban-code && npm i && npm run compile",
        "title": "Following Shot #063 Carrot Through All Four Studio Rooms",
        "featured_shot": "carrot",
        "speech_p1": "EVERY CARTOON IN OUR ARCHIVE TRAVELS THROUGH FOUR PHYSICAL STUDIO ROOMS IN SEQUENCE! WATCH SHOT #063 CARROT!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): ROUGH PENCIL -> BLACK INDIA INK -> ORANGE GOUACHE ACROSS 4 DESKS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): COMPOSITING THE GARDEN SOIL, FOLIAGE, AND HERO CARROT PLANES!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): SPROCKET AUDITS & CHIME FOLEY-SYNCS THE FINISHED 4:3 CARROT REEL!",
        "blurb": "Follow Shot #063 Carrot from Room 1 (Moviola & Storyboard) to Room 2 (4-Desk Lightbox Bullpen) to Room 3 (Multiplane Camera Tower) to Room 4 (35mm Screening Theatre).",
        "stages": ["Room 1: Storyboard Pin-Up", "Room 2: Rough -> Ink -> Gouache", "Room 3: Multiplane Compositing", "Room 4: Final Screening & Splice"],
    },
    {
        "id": "office-09-vibe-kanban",
        "num": 9,
        "group": "office",
        "lead_agent": "sprocket",
        "cameo_agent": "blueline",
        "studio_wing": "bg3_screening",
        "wing_name": "EPISODE 09 // TWIN-SCREEN MOTION LOCK",
        "repo": "BloopAI/vibe-kanban",
        "url": "https://github.com/BloopAI/vibe-kanban",
        "category": "Side-by-Side Archival vs. Multiplane Dailies",
        "install": "npx vibe-kanban",
        "title": "Matching Archival Motion from Moviola to Silver Screen",
        "featured_shot": "diamond",
        "speech_p1": "HOW DO WE MAKE SURE OUR CARTOON MATCHES THE ARCHIVAL VIDEO'S MOTION? WE TRACK THE SPINE IN EVERY ROOM!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS LOCK THE DIAMOND'S FACETS TO THE ARCHIVAL MOTION ARC!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): NIB CRANKS THE CAMERA ALONG THE EXACT ARCHIVAL VELOCITY CURVE!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): BOTH 4:3 MONITORS PLAY IN LOCKSTEPInside THE SCREENING ROOM!",
        "blurb": "Sprocket and Blue-Line show how the archival motion spine traced on the Room 1 Moviola guides the 4 worktree inkers, the Multiplane crank, and the dual 4:3 screening monitors.",
        "stages": ["Room 1: Moviola Spine Trace", "Room 2: 4-Desk Facet Inking", "Room 3: Motion-Locked Camera Crank", "Room 4: Dual-4:3 Sync Screening"],
    },
    {
        "id": "office-10-claude-code-kanban",
        "num": 10,
        "group": "office",
        "lead_agent": "nib",
        "cameo_agent": "chime",
        "studio_wing": "bg4_clocktower",
        "wing_name": "EPISODE 10 // THE 346-REEL STUDIO FINALE",
        "repo": "NikkeTryHard/claude-code-kanban",
        "url": "https://github.com/NikkeTryHard/claude-code-kanban",
        "category": "Full-Catalog 346-Shot Master Archive",
        "install": "git clone https://github.com/NikkeTryHard/claude-code-kanban.git && cd claude-code-kanban && npm install && npm start",
        "title": "How Our Five Agent Animators Built All 346 Films",
        "featured_shot": "star",
        "speech_p1": "ACROSS 346 PRODUCTIONS, OUR FIVE AGENT ANIMATORS INKED 1,384 KEYFRAMES AND 2,422 MULTIPLANE GLASS LAYERS!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 1,384 HAND-DRAWN 4:3 KEYFRAMES INKED ACROSS OUR BULLPEN DESKS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): 2,422 GOUACHE DEPTH PLANES COMPOSITED UNDER THE OVERHEAD CAMERA!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): THE ENTIRE AGENT CREW CELEBRATES AS SHOT #285 STAR LIGHTS THE SCREEN!",
        "blurb": "All 5 Agent Animators—Nib, Blue-Line, Cel-Inker, Sprocket, and Chime—gather across the 4 studio rooms to celebrate completing all 346 multiplane cartoons.",
        "stages": ["Room 1: 346 Archival Traces", "Room 2: 1,384 Worktree Keyframes", "Room 3: 2,422 Multiplane Layers", "Room 4: Grand Studio Screening"],
    },
]

VIDEO_SKILL_SPECS = [
    {
        "id": "skill-01-huashu-motion",
        "num": 11,
        "group": "skill",
        "timestamp": "01:41",
        "lead_agent": "nib",
        "cameo_agent": "blueline",
        "studio_wing": "bg5_bullpen",
        "wing_name": "SKILL 01 // DIRECTING CAMERA PACING",
        "repo": "octohash/huashu-skills (motion-graphics)",
        "url": "https://github.com/octohash/huashu-skills",
        "category": "Establishing Wide -> Action Medium -> Hero Close-Up",
        "install": "git clone https://github.com/octohash/huashu-skills.git",
        "title": "How Nib Directs Camera Pacing Across the Four Rooms",
        "featured_shot": "butterfly",
        "speech_p1": "A GREAT ANIMATION STUDIO NEVER RUSHES THE VIEWER! WATCH HOW WE PACE SHOT #050 BUTTERFLY FROM SKETCH TO SCREEN!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS DRAW THE WINGBEAT POSES K1..K4 AT UNHURRIED RHYTHM!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): CEL-INKER COMPOSITES THE GARDEN BLOSSOMS AND FLUTTERING BUTTERFLY!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): ON OUR 4:3 SCREEN, EVERY WINGBEAT READS WITH CRYSTAL CLARITY!",
        "blurb": "Nib demonstrates how deliberate studio pacing across the Storyboard Room, 4-Desk Bullpen, Multiplane Tower, and Screening Theatre lets the viewer savor every wingbeat.",
        "stages": ["Room 1: Pacing the Storyboard", "Room 2: 4-Desk Wingbeat Inking", "Room 3: Garden Multiplane Stack", "Room 4: Unhurried 4:3 Screening"],
    },
    {
        "id": "skill-02-rossark-motion-skill",
        "num": 12,
        "group": "skill",
        "timestamp": "05:37",
        "lead_agent": "celinker",
        "cameo_agent": "nib",
        "studio_wing": "bg1_multifloor",
        "wing_name": "SKILL 02 // MULTIPLANE CAMERA DOLLY",
        "repo": "rossark/motion-skill",
        "url": "https://github.com/rossark/motion-skill",
        "category": "Multiplane Camera Dolly & Parallax Rig",
        "install": "git clone https://github.com/rossark/motion-skill.git",
        "title": "Operating the Multiplane Camera Dolly on Shot #042 Campfire",
        "featured_shot": "campfire",
        "speech_p1": "TO MAKE THE NIGHT WOODS FEEL DEEP AROUND THE CAMPFIRE, CEL-INKER AND I RIG SEVEN SEPARATE GLASS LEVELS!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS PAINT THE FLICKERING FLAME POSES K1..K4 IN WARM GOUACHE!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): NIB CRANKS THE DOLLY WHEEL AS PINE TREES AND FLAMES SHIFT IN PARALLAX!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): THE CAMPFIRE GLOWS WARMLY ON OUR 4:3 SCREENING ROOM THEATRE SCREEN!",
        "blurb": "Watch Cel-Inker and Nib load the night sky, pine woods, riverbank logs, and flickering campfire cels into the Multiplane Tower and crank the camera dolly.",
        "stages": ["Room 1: Night Scene Breakdown", "Room 2: 4-Desk Flame Gouache", "Room 3: Cranking the Dolly Wheel", "Room 4: Campfire Theatre Screening"],
    },
    {
        "id": "skill-03-take-one",
        "num": 13,
        "group": "skill",
        "timestamp": "09:49",
        "lead_agent": "blueline",
        "cameo_agent": "sprocket",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 03 // BLOCKING STAGE TAKE ONE",
        "repo": "nerfels/take-one",
        "url": "https://github.com/nerfels/take-one",
        "category": "Pre-Roll Stage Blocking & Clapperboard Lock",
        "install": "git clone https://github.com/nerfels/take-one.git",
        "title": "How Our Agent Crew Blocks Shot #071 Clock Before Rolling",
        "featured_shot": "clock",
        "speech_p1": "BEFORE WE ROLL TAKE ONE ON SHOT #071 CLOCK, EVERY AGENT STATION LOCKS REGISTRATION PEGS AND MARGINS!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS ALIGN THE CLOCK HANDS & PENDULUM ON 4:3 PEGBAR SHEETS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): LOCKING THE TABLETOP WALLPAPER, SHADOW, AND CLOCK FACE PLATENS!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): CHIME TAPS THE TICK-TOCK FOLEY AS THE 4:3 CLOCK REEL PASSES TAKE ONE!",
        "blurb": "Blue-Line and Sprocket walk through the Take-One pre-roll protocol across all 4 studio rooms, ensuring zero margin clipping and frame-accurate pendulum timing.",
        "stages": ["Room 1: Pegbar Registration Lock", "Room 2: 4-Desk Pendulum Keys", "Room 3: Tabletop Multiplane Rig", "Room 4: Tick-Tock Foley Screening"],
    },
    {
        "id": "skill-04-reanimate",
        "num": 14,
        "group": "skill",
        "timestamp": "13:02",
        "lead_agent": "blueline",
        "cameo_agent": "celinker",
        "studio_wing": "bg4_clocktower",
        "wing_name": "SKILL 04 // ROTOSCOPE REANIMATION",
        "repo": "bytefer/reanimate",
        "url": "https://github.com/bytefer/reanimate",
        "category": "Archival Optical Flow -> 4-Keyframe Rotoscope",
        "install": "git clone https://github.com/bytefer/reanimate.git",
        "title": "From Raw Archival Video to Hand-Inked Multiplane Cel",
        "featured_shot": "giraffe-walks-past",
        "speech_p1": "WE NEVER FILTER PHOTOREAL VIDEO! WATCH HOW OUR AGENT CREW RE-INKS THE ARCHIVAL GIRAFFE FROM SCRATCH!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): TRACING BLUE ROUGHS -> BLACK INDIA INK -> GOUACHE ACROSS 4 DESKS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): COMPOSITING THE SAVANNA SKY, ACACIA TREES, AND RE-INKED GIRAFFE!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): COMPARE THE 4:3 ARCHIVAL MONITOR BESIDE OUR 4:3 REANIMATED CARTOON!",
        "blurb": "See the complete Reanimate pipeline in action: tracing the motion spine on the Room 1 Moviola, redrawing 4 keyframes at the Room 2 lightboxes, stacking 7 planes in Room 3, and comparing both 4:3 screens in Room 4.",
        "stages": ["Room 1: Moviola Optical Trace", "Room 2: Blue Rough -> Ink -> Paint", "Room 3: Savanna Multiplane Stack", "Room 4: Side-by-Side 4:3 Theatre"],
    },
    {
        "id": "skill-05-argo",
        "num": 15,
        "group": "skill",
        "timestamp": "18:54",
        "lead_agent": "sprocket",
        "cameo_agent": "chime",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 05 // THE 16-GATE STUDIO HARNESS",
        "repo": "mylin0/argo",
        "url": "https://github.com/mylin0/argo",
        "category": "Automated 16-Gate Visual Quality Assurance",
        "install": "git clone https://github.com/mylin0/argo.git",
        "title": "How Sprocket Enforces All 16 Quality Gates Across the Studio",
        "featured_shot": "hot-air-balloon",
        "speech_p1": "IF A HOT-AIR BALLOON CLIPS THE FRAME BORDER, I HALT THE LINE! WATCH HOW WE GUARD SHOT #152 AT EVERY STATION!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS KEEP THE RISING BALLOON INSIDE SAFE 4:3 MARGINS!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): VERIFYING PARALLAX SEPARATION BETWEEN HIGH CLOUDS AND THE BALLOON!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): SPROCKET AUDITS THE 4:3 PROJECTION AND STAMPS 16/16 GATES PASS!",
        "blurb": "Follow Auditor Sprocket through all 4 studio rooms to see how margin safety, 4:3 aspect lock, line boil, and parallax separation are verified on Shot #152 Hot-Air Balloon.",
        "stages": ["Room 1: Safe-Margin Storyboard", "Room 2: 4-Desk Margin Check", "Room 3: Cloud Parallax Verification", "Room 4: 16/16 Gate Approval Stamp"],
    },
    {
        "id": "skill-06-explainroo",
        "num": 16,
        "group": "skill",
        "timestamp": "23:18",
        "lead_agent": "nib",
        "cameo_agent": "celinker",
        "studio_wing": "bg5_bullpen",
        "wing_name": "SKILL 06 // SHOWING THE CRAFT CLEARLY",
        "repo": "Explainum/explainroo",
        "url": "https://github.com/Explainum/explainroo",
        "category": "Visual Storytelling by Showing the Studio in Action",
        "install": "git clone https://github.com/Explainum/explainroo.git",
        "title": "Showing Our Agent Animators at Work Instead of Telemetry",
        "featured_shot": "lighthouse",
        "speech_p1": "THE BEST WAY TO EXPLAIN OUR STUDIO IS TO OPEN THE DOORS AND LET YOU WATCH OUR AGENT ANIMATORS MAKE A FILM!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): WATCH THE 4 LIGHTHOUSE KEYFRAMES GROW FROM PENCIL ROUGH TO PAINTED CEL!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): WATCH THE 3 GLASS LEVELS COMBINE INTO THE FULL VIEWFINDER SCENE!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): WATCH THE FINISHED 4:3 FILM ROLL IN OUR STUDIO SCREENING ROOM!",
        "blurb": "Nib opens the doors to all 4 rooms of Secondhand Studios so you can watch the agent animators trace, ink, paint, stack, crank, and project a cartoon from start to finish.",
        "stages": ["Room 1: Open-Door Moviola Room", "Room 2: Live Lightbox Progression", "Room 3: Live Multiplane Viewfinder", "Room 4: Full Theatre Screening"],
    },
    {
        "id": "skill-07-video-director",
        "num": 17,
        "group": "skill",
        "timestamp": "27:45",
        "lead_agent": "sprocket",
        "cameo_agent": "nib",
        "studio_wing": "bg1_multifloor",
        "wing_name": "SKILL 07 // UNIFIED STUDIO PALETTE & INK",
        "repo": "fireflewdev/video-director-skill",
        "url": "https://github.com/fireflewdev/video-director-skill",
        "category": "Archive-Wide Gouache Palette & Rhythm Continuity",
        "install": "git clone https://github.com/fireflewdev/video-director-skill.git",
        "title": "Mixing the Same Five Gouache Colors Across Every Room",
        "featured_shot": "guitar",
        "speech_p1": "WHY DO ALL 346 FILMS FEEL LIKE ONE UNIFIED WORLD? BECAUSE EVERY ROOM IN OUR STUDIO USES THE SAME INK & GOUACHE!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): ALL 4 INKER DESKS SHARE OUR WARM ARCHIVAL PAPER & SPEEDBALL INDIA INK!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): CEL-INKER GLAZES EVERY GLASS PLANE IN OUR 5 SIGNATURE GOUACHE WASHES!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): THE FINISHED 4:3 GUITAR REEL MATCHES ALL 346 FILMS IN THE ARCHIVE!",
        "blurb": "Sprocket and Nib show how standardizing warm bond paper, Speedball India ink, and 5 gouache washes across all 4 studio rooms unifies the entire 346-shot catalog.",
        "stages": ["Room 1: Palette Specification", "Room 2: Standardized Ink Desks", "Room 3: 5-Gouache Glass Platens", "Room 4: Color-Matched Screening"],
    },
    {
        "id": "skill-08-earshot",
        "num": 18,
        "group": "skill",
        "timestamp": "31:10",
        "lead_agent": "chime",
        "cameo_agent": "sprocket",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 08 // LIVE FOLEY SOUNDSTAGE SYNC",
        "repo": "sphere-ai-lab/Earshot",
        "url": "https://github.com/sphere-ai-lab/Earshot",
        "category": "Motion-Triggered Foley & Soundstage Sync",
        "install": "git clone https://github.com/sphere-ai-lab/Earshot.git",
        "title": "How Chime Strikes Foley Sound Effects on Motion Peaks",
        "featured_shot": "train",
        "speech_p1": "HOW DOES SILENT INK GET ITS VOICE? WE TRACK ACCELERATION PEAKS FROM THE MOVIOLA ALL THE WAY TO MY FOLEY STAGE!",
        "caption_p2": "ROOM 2 (WORKTREE BULLPEN): 4 INKER AGENTS MARK RED FOLEY CUE DOTS ON KEYFRAMES K1, K2, K3 & K4!",
        "caption_p3": "ROOM 3 (MULTIPLANE TOWER): NIB LOCKS THE CAMERA SHUTTER PULSES TO THE X-SHEET TIMING MARKS!",
        "caption_p4": "ROOM 4 (SCREENING THEATRE): CHIME STRIKES THE FOLEY PROPS IN SYNC WITH THE 4:3 STEAM TRAIN!",
        "blurb": "Follow Foley Percussionist Chime from the Room 1 Moviola through the bullpen and Multiplane Tower to the Room 4 Foley Stage, striking acoustic cues on the exact frames of peak acceleration.",
        "stages": ["Room 1: Marking Jerk Peaks", "Room 2: X-Sheet Cue Inking", "Room 3: Shutter-Synced Camera Rig", "Room 4: Live Foley Stage Performance"],
    },
]

_QD_CACHE = {}
_SHOT_FRAMES_CACHE = {}


def load_qd_norm(cat: str, idx: int = 0) -> list:
    cat_clean = "giraffe" if cat == "giraffe-walks-past" else cat.replace("-", "_").strip().lower()
    if cat_clean not in _QD_CACHE:
        p = QD_DIR / f"{cat_clean}.json"
        if not p.exists():
            fallbacks = {
                "palette": "paint_can",
                "bell": "alarm_clock",
            }
            p = QD_DIR / f"{fallbacks.get(cat_clean, 'star')}.json"
            if not p.exists():
                p = QD_DIR / "star.json"
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
            _QD_CACHE[cat_clean] = valid
        except Exception:
            _QD_CACHE[cat_clean] = []
            return []
    items = _QD_CACHE[cat_clean]
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
    ox = (span - w_span) * 0.5
    oy = (span - h_span) * 0.5
    norm = []
    for s in strokes:
        norm.append([[(px - x0 + ox) / span, (py - y0 + oy) / span] for px, py in s])
    return norm


def draw_qd_doodle(
    img: np.ndarray,
    cat: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    fill_bgr=BGR_GUTTER,
    ink_bgr=BGR_INK,
    thickness: int = 3,
    boil_phase: int = 0,
    idx: int = 0,
):
    """Strictly preserves 1:1 uniform aspect ratio (`side = min(w, h)`)."""
    norm = load_qd_norm(cat, idx=idx)
    if not norm:
        return
    rng = np.random.RandomState((hash(cat) & 0xFFFF) + boil_phase * 53)
    side = float(min(w, h))
    x0 = cx - side * 0.5
    y0 = cy - side * 0.5
    scaled_strokes = []
    all_pts = []
    for s in norm:
        pts = []
        for nx, ny in s:
            jx = float(rng.uniform(-0.35, 0.35)) if boil_phase > 0 else 0.0
            jy = float(rng.uniform(-0.35, 0.35)) if boil_phase > 0 else 0.0
            px = int(round(x0 + nx * side + jx))
            py = int(round(y0 + ny * side + jy))
            pts.append([px, py])
            all_pts.append([px, py])
        if len(pts) >= 2:
            scaled_strokes.append(np.array(pts, dtype=np.int32))
    if not scaled_strokes:
        return
    if fill_bgr is not None and len(all_pts) >= 3:
        hull = cv2.convexHull(np.array(all_pts, dtype=np.int32))
        cv2.fillConvexPoly(img, hull + np.array([3, 3], dtype=np.int32), BGR_INK, cv2.LINE_AA)
        cv2.fillConvexPoly(img, hull, fill_bgr, cv2.LINE_AA)
    for arr in scaled_strokes:
        cv2.polylines(img, [arr], False, ink_bgr, thickness, cv2.LINE_AA)


def load_shot_frames(slug: str):
    """
    Loads authentic, crystal-clear 4:3 (`480x360`) assets for `slug`:
      1. `arch_frames` (33 frames): Real archival source frames from `shots/<slug>/frames-src/f*.png`
         with a warm 35mm archival grade + clean Surgical SAM tracking box & blue gesture arc (ZERO Canny static!).
      2. `blue_roughs` (33 frames): True Animator's Non-Photo Blue (`#3A7CA5`) construction & rough line sheets
         built from `scene-plate.png` + `selected-keys.png` (`K1..K4`) on warm animation paper with golden-ratio grid.
      3. `ink_cels` (33 frames): Crisp Speedball Black India Ink (`#141414`) line cels over faint non-photo blue underdrawing.
      4. `cart_frames` (33 frames): Full `640x480` (4:3 exact!) 7-plane gouache cartoon frames with old HUD badges
         seamlessly healed out using adjacent sky/ground scanlines (ZERO cropping, ZERO black HUD bars!).
      5. `key_stages` (4 tuples of `(blue_k, ink_k, gouache_k)` for `K1, K2, K3, K4`): Built directly from the
         4 quadrants of `shots/<slug>/selected-keys.png` composited over `shots/<slug>/scene-plate.png`!
      6. `platen_layers` (`(sky_layer, mid_layer, hero_layer)`): True separated multiplane layers for Room 3's Camera Tower!
    """
    if slug in _SHOT_FRAMES_CACHE:
        return _SHOT_FRAMES_CACHE[slug]

    sdir = SHOTS_DIR / slug
    comp_path = sdir / "comparison.mp4"
    f_src_paths = sorted((sdir / "frames-src").glob("f*.png")) if (sdir / "frames-src").exists() else []
    dsa_path = sdir / "depth_sam_analysis.json"
    dsa_frames = []
    if dsa_path.exists():
        try:
            dsa_frames = json.loads(dsa_path.read_text(encoding="utf-8")).get("frames", [])
        except Exception:
            dsa_frames = []

    # 1. Load clean 4:3 Finished Cartoon Frames from right half of comparison.mp4 (640x480 -> 480x360, NO cropping!)
    cart_frames = []
    raw_left_frames = []
    if comp_path.exists():
        cap = cv2.VideoCapture(str(comp_path))
        while len(cart_frames) < 33:
            ok, fr = cap.read()
            if not ok:
                break
            fh, fw = fr.shape[:2]  # 480 x 1280
            half_w = fw // 2       # 640 (4:3 exact!)
            left_640 = fr[:, :half_w].copy()
            right_640 = fr[:, half_w:].copy()

            # Heal top-right badge (y=0..37, x=368..638) using clean sky column 365 (no lamp smear!)
            right_640[0:37, 368:638] = np.repeat(right_640[0:37, 365:366], 270, axis=1)
            # Heal bottom-left badge (y=446..479, x=0..520) using clean bottom scanline 479 (no hoof smear or SAM CEL!)
            right_640[446:479, 0:520] = np.repeat(right_640[479:480, 0:520], 33, axis=0)

            raw_left_frames.append(cv2.resize(left_640, (480, 360), interpolation=cv2.INTER_AREA))
            cart_frames.append(cv2.resize(right_640, (480, 360), interpolation=cv2.INTER_AREA))
        cap.release()

    if not cart_frames:
        blank = np.full((360, 480, 3), BGR_PAPER, dtype=np.uint8)
        draw_qd_doodle(blank, slug, 240, 180, 200, 200, fill_bgr=BGR_AMBER_PALE, thickness=4)
        cart_frames = [blank.copy() for _ in range(33)]
        raw_left_frames = [blank.copy() for _ in range(33)]
    while len(cart_frames) < 33:
        cart_frames.append(cart_frames[-1].copy())
        raw_left_frames.append(raw_left_frames[-1].copy())

    # 2. Load clean 4:3 Archival Source Frames directly from `frames-src/f*.png` (NO HUD bars, NO Canny static!)
    arch_sketches = []
    for f_i in range(33):
        if f_src_paths:
            p_f = f_src_paths[min(f_i, len(f_src_paths) - 1)]
            src_bgr = cv2.imread(str(p_f))
            if src_bgr is not None:
                arch_480 = cv2.resize(src_bgr, (480, 360), interpolation=cv2.INTER_AREA)
            else:
                arch_480 = raw_left_frames[f_i].copy()
        else:
            arch_480 = raw_left_frames[f_i].copy()

        # Subtle warm archival 35mm print grade (100% clear photographic image!)
        warm_tint = np.full_like(arch_480, BGR_PAPER)
        graded = cv2.addWeighted(arch_480, 0.86, warm_tint, 0.14, 0)

        # Overlay clean studio tracking reticle / SAM bounding box if available
        if f_i < len(dsa_frames) and "bbox" in dsa_frames[f_i]:
            bb = dsa_frames[f_i]["bbox"]
            # Scale bbox if in 640x480 vs 480x360
            scale_b = 0.75 if max(bb) > 485 else 1.0
            x0 = int(round(bb[0] * scale_b))
            y0 = int(round(bb[1] * scale_b))
            x1 = int(round(bb[2] * scale_b))
            y1 = int(round(bb[3] * scale_b))
            cv2.rectangle(graded, (x0, y0), (x1, y1), BGR_NONPHOTO_BLUE, 2, cv2.LINE_AA)
            # Corner registration ticks
            cx_b, cy_b = (x0 + x1) // 2, (y0 + y1) // 2
            cv2.drawMarker(graded, (cx_b, cy_b), BGR_RED, cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
        arch_sketches.append(graded)

    # 3. Extract true Non-Photo Blue Roughs & Clean Black India Ink Cels from the cartoon frames + `selected-keys.png`
    blue_roughs = []
    ink_cels = []
    for f_i in range(33):
        raw_c = cart_frames[f_i]
        gray_c = cv2.cvtColor(raw_c, cv2.COLOR_BGR2GRAY)
        # Extract dark ink strokes cleanly without Canny double-edge noise
        ink_mask = gray_c < 165

        # Stage A: Animator's Non-Photo Blue Rough Construction Sheet
        rough = np.full((360, 480, 3), BGR_GUTTER, dtype=np.uint8)
        # Golden Ratio & Perspective Horizon Grid in warm light sepia
        cv2.line(rough, (0, 222), (479, 222), (228, 208, 182), 1)
        cv2.line(rough, (183, 0), (183, 359), (228, 208, 182), 1)
        cv2.line(rough, (296, 0), (296, 359), (228, 208, 182), 1)
        # Soft blue underdrawing
        alpha_b = np.clip((215.0 - gray_c.astype(np.float32)) / 145.0, 0.0, 1.0)[..., None]
        rough_f = rough.astype(np.float32) * (1.0 - alpha_b) + np.array(BGR_NONPHOTO_BLUE, dtype=np.float32) * alpha_b
        blue_roughs.append(np.clip(rough_f, 0, 255).astype(np.uint8))

        # Stage B: Clean Black Speedball India Ink Line Cel over faint non-photo blue construction lines
        ink_cel = rough.copy()
        ink_cel_f = ink_cel.astype(np.float32) * (1.0 - 0.35 * alpha_b) + np.array(BGR_NONPHOTO_BLUE, dtype=np.float32) * (0.35 * alpha_b)
        alpha_k = np.clip((185.0 - gray_c.astype(np.float32)) / 120.0, 0.0, 1.0)[..., None]
        ink_cel_f = ink_cel_f * (1.0 - alpha_k) + np.array(BGR_INK, dtype=np.float32) * alpha_k
        ink_cels.append(np.clip(ink_cel_f, 0, 255).astype(np.uint8))

    _SHOT_FRAMES_CACHE[slug] = (arch_sketches, blue_roughs, ink_cels, cart_frames)
    return arch_sketches, blue_roughs, ink_cels, cart_frames


_SHOT_KEYS_PLATE_CACHE = {}


def load_shot_keys_and_plate(slug: str):
    """
    Loads the actual high-resolution hand-drawn `shots/<slug>/selected-keys.png` (4 quadrants K1..K4)
    and `shots/<slug>/scene-plate.png` so Room 1's Storyboard Wall, Room 2's 4 Lightbox Desks,
    and Room 3's Multiplane Camera Tower Platens display the REAL hand-drawn Quick, Draw! keyframe
    and environment drawings!
    """
    if slug in _SHOT_KEYS_PLATE_CACHE:
        return _SHOT_KEYS_PLATE_CACHE[slug]

    sdir = SHOTS_DIR / slug
    sk_path = sdir / "selected-keys.png"
    sp_path = sdir / "scene-plate.png"

    plate_gray = cv2.imread(str(sp_path), cv2.IMREAD_GRAYSCALE) if sp_path.exists() else np.full((360, 480), 255, dtype=np.uint8)
    if plate_gray is None:
        plate_gray = np.full((360, 480), 255, dtype=np.uint8)
    plate_gray = cv2.resize(plate_gray, (480, 360), interpolation=cv2.INTER_AREA)

    quad_grays = []
    if sk_path.exists():
        sk_gray = cv2.imread(str(sk_path), cv2.IMREAD_GRAYSCALE)
        if sk_gray is not None:
            H, W = sk_gray.shape[:2]
            for q_i in range(4):
                r, c = q_i // 2, q_i % 2
                sub = sk_gray[r * (H // 2) + 8:(r + 1) * (H // 2) - 8, c * (W // 2) + 8:(c + 1) * (W // 2) - 8]
                quad_grays.append(cv2.resize(sub, (480, 360), interpolation=cv2.INTER_AREA))
    while len(quad_grays) < 4:
        quad_grays.append(plate_gray.copy())

    _SHOT_KEYS_PLATE_CACHE[slug] = (plate_gray, quad_grays)
    return plate_gray, quad_grays


# =============================================================================
# VECTOR STROKE-BY-STROKE ENGINE FOR HERO BOARDS, AGENTS, APPARATUS & ROOMS
# =============================================================================
DOODLE_ASSETS_DIR = ROOT / "assets" / "studio_doodles"
STUDIO_TRACED_JSON = DOODLE_ASSETS_DIR / "studio_traced.json"
_STUDIO_TRACED_CATALOG = None


def get_studio_traced_catalog() -> dict:
    global _STUDIO_TRACED_CATALOG
    if _STUDIO_TRACED_CATALOG is None:
        if STUDIO_TRACED_JSON.exists():
            _STUDIO_TRACED_CATALOG = json.loads(STUDIO_TRACED_JSON.read_text(encoding="utf-8"))
        else:
            _STUDIO_TRACED_CATALOG = {"sheets": {}, "hero_shots": {}}
    return _STUDIO_TRACED_CATALOG


def draw_hand_inked_line(
    canvas: np.ndarray,
    pt0: tuple,
    pt1: tuple,
    color_bgr: tuple = BGR_INK,
    thickness: int = 2,
    wobble: float = 1.15,
    boil_phase: int = 0,
    n_segs: int = 10,
):
    """Draws a multi-vertex hand-wobbled ink stroke instead of a computer-straight cv2.line."""
    x0, y0 = float(pt0[0]), float(pt0[1])
    x1, y1 = float(pt1[0]), float(pt1[1])
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(3, min(22, int(round(dist / 28.0)) if n_segs is None else n_segs))
    pts = []
    seed_base = (x0 * 0.07 + y0 * 0.11 + x1 * 0.05 + y1 * 0.09)
    for i in range(steps + 1):
        t = i / float(steps)
        px = x0 + (x1 - x0) * t
        py = y0 + (y1 - y0) * t
        if 0 < i < steps and wobble > 0.0:
            env = math.sin(math.pi * t)
            ang = seed_base + i * 2.1 + boil_phase * 1.5
            px += wobble * env * math.sin(ang)
            py += wobble * env * math.cos(ang * 1.3)
        pts.append([int(round(px)), int(round(py))])
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color_bgr, thickness, cv2.LINE_AA)


def draw_hand_inked_rect(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill_bgr: tuple = None,
    outline_bgr: tuple = BGR_INK,
    thickness: int = 3,
    wobble: float = 1.2,
    boil_phase: int = 0,
    shadow_offset: int = 0,
):
    """Draws a hand-inked box with organic wobbled edges and optional drop shadow & gouache fill."""
    if shadow_offset > 0:
        sx0, sy0, sx1, sy1 = x0 + shadow_offset, y0 + shadow_offset, x1 + shadow_offset, y1 + shadow_offset
        cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), BGR_INK, -1)
    if fill_bgr is not None:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill_bgr, -1)
    if outline_bgr is not None and thickness > 0:
        draw_hand_inked_line(canvas, (x0, y0), (x1, y0), outline_bgr, thickness, wobble, boil_phase)
        draw_hand_inked_line(canvas, (x1, y0), (x1, y1), outline_bgr, thickness, wobble, boil_phase + 1)
        draw_hand_inked_line(canvas, (x1, y1), (x0, y1), outline_bgr, thickness, wobble, boil_phase + 2)
        draw_hand_inked_line(canvas, (x0, y1), (x0, y0), outline_bgr, thickness, wobble, boil_phase + 3)


def draw_active_pen_nib(
    canvas: np.ndarray,
    x_tip: int,
    y_tip: int,
    is_blue_pencil: bool = False,
    is_brush: bool = False,
    scale: float = 1.0,
):
    """
    Draws a hand-inked Speedball Dip-Pen Nib (or Non-Photo Blue Pencil / Gouache Brush)
    right at the active stroke vertex `(x_tip, y_tip)` that is currently laying down ink!
    """
    H, W = canvas.shape[:2]
    if not (2 <= x_tip < W - 2 and 2 <= y_tip < H - 2):
        return
    s = max(0.65, float(scale))
    # Angled pen shaft pointing down-left to (x_tip, y_tip) at ~42 degrees
    ux, uy = 0.72, -0.69  # unit vector along shaft away from tip
    px, py = 0.69, 0.72   # perpendicular unit vector

    def pt(along, perp):
        return [int(round(x_tip + (along * ux + perp * px) * s)), int(round(y_tip + (along * uy + perp * py) * s))]

    nib_col = BGR_NONPHOTO_BLUE if is_blue_pencil else (BGR_AMBER if is_brush else BGR_AMBER_PALE)
    holder_col = BGR_BLUE_PALE if is_blue_pencil else BGR_RED

    # Metallic nib / brush tip triangle (0..13 px along shaft)
    nib_poly = np.array([pt(0, 0), pt(12, -4.2), pt(14, 0), pt(12, 4.2)], dtype=np.int32)
    cv2.fillPoly(canvas, [nib_poly], nib_col, cv2.LINE_AA)
    cv2.polylines(canvas, [nib_poly], True, BGR_INK, max(1, int(round(1.6 * s))), cv2.LINE_AA)

    # Wooden pen holder barrel (12..30 px along shaft)
    barrel_poly = np.array([pt(12, -4.5), pt(30, -5.2), pt(30, 5.2), pt(12, 4.5)], dtype=np.int32)
    cv2.fillPoly(canvas, [barrel_poly], holder_col, cv2.LINE_AA)
    cv2.polylines(canvas, [barrel_poly], True, BGR_INK, max(1, int(round(1.6 * s))), cv2.LINE_AA)

    # Ink breather hole & wet ink droplet right at the tip
    breather = pt(7, 0)
    cv2.circle(canvas, (breather[0], breather[1]), max(1, int(round(1.5 * s))), BGR_INK, -1, cv2.LINE_AA)
    tip_dot_col = BGR_NONPHOTO_BLUE if is_blue_pencil else (BGR_AMBER if is_brush else BGR_INK)
    cv2.circle(canvas, (x_tip, y_tip), max(2, int(round(2.5 * s))), tip_dot_col, -1, cv2.LINE_AA)


def _draw_strokes_progressive(
    canvas: np.ndarray,
    strokes_transformed: list,
    progress: float,
    color_bgr: tuple,
    thickness: int = 2,
    underdraw_bgr: tuple = None,
    underdraw_thickness: int = 1,
):
    """
    Draws a list of transformed vector polylines `strokes_transformed` stroke-by-stroke
    up to `progress` in `[0.0, 1.0]`.
    Returns `(x_tip, y_tip)` of the active vertex currently being drawn (or `None` if finished/0).
    """
    if not strokes_transformed:
        return None

    # Optional faint underdrawing of all strokes first (e.g. Non-Photo Blue rough lines waiting to be inked)
    if underdraw_bgr is not None and progress < 0.999:
        for st in strokes_transformed:
            if len(st) >= 2:
                cv2.polylines(canvas, [np.array(st, dtype=np.int32)], False, underdraw_bgr, underdraw_thickness, cv2.LINE_AA)

    if progress <= 0.001:
        first_pt = strokes_transformed[0][0]
        return int(round(first_pt[0])), int(round(first_pt[1]))

    def _stroke_w(st_pts):
        if thickness <= 1 or len(st_pts) < 2:
            return max(1, thickness)
        span = math.hypot(st_pts[-1][0] - st_pts[0][0], st_pts[-1][1] - st_pts[0][1])
        return thickness if (span >= 14.0 or len(st_pts) >= 7) else max(1, thickness - 1)

    n_st = len(strokes_transformed)
    if progress >= 0.999:
        for st in strokes_transformed:
            if len(st) >= 2:
                cv2.polylines(canvas, [np.array(st, dtype=np.int32)], False, color_bgr, _stroke_w(st), cv2.LINE_AA)
        return None

    exact_idx = float(np.clip(progress, 0.0, 1.0)) * n_st
    full_cnt = int(exact_idx)
    rem_frac = exact_idx - full_cnt

    # 1. Draw all completed strokes `0 .. full_cnt - 1`
    for i in range(min(full_cnt, n_st)):
        st = strokes_transformed[i]
        if len(st) >= 2:
            cv2.polylines(canvas, [np.array(st, dtype=np.int32)], False, color_bgr, _stroke_w(st), cv2.LINE_AA)

    # 2. Draw the currently active stroke `full_cnt` vertex-by-vertex up to `rem_frac`
    active_tip = None
    if full_cnt < n_st:
        cur_st = strokes_transformed[full_cnt]
        if len(cur_st) >= 2:
            # Interpolate along polyline segments
            seg_lens = [math.hypot(cur_st[k + 1][0] - cur_st[k][0], cur_st[k + 1][1] - cur_st[k][1]) for k in range(len(cur_st) - 1)]
            tot_len = sum(seg_lens)
            if tot_len <= 1e-3:
                active_tip = (int(round(cur_st[0][0])), int(round(cur_st[0][1])))
            else:
                target_d = rem_frac * tot_len
                partial_pts = [[int(round(cur_st[0][0])), int(round(cur_st[0][1]))]]
                acc = 0.0
                for k, sl in enumerate(seg_lens):
                    if acc + sl <= target_d:
                        partial_pts.append([int(round(cur_st[k + 1][0])), int(round(cur_st[k + 1][1]))])
                        acc += sl
                    else:
                        f_seg = (target_d - acc) / max(sl, 1e-6)
                        ix = cur_st[k][0] + (cur_st[k + 1][0] - cur_st[k][0]) * f_seg
                        iy = cur_st[k][1] + (cur_st[k + 1][1] - cur_st[k][1]) * f_seg
                        partial_pts.append([int(round(ix)), int(round(iy))])
                        break
                if len(partial_pts) >= 2:
                    cv2.polylines(canvas, [np.array(partial_pts, dtype=np.int32)], False, color_bgr, thickness, cv2.LINE_AA)
                active_tip = (partial_pts[-1][0], partial_pts[-1][1])
        elif len(cur_st) == 1:
            active_tip = (int(round(cur_st[0][0])), int(round(cur_st[0][1])))

    return active_tip


def _morph_hero_strokes(hero_entry: dict, ka: int, kb: int, t_raw: float):
    """
    Constructs a 10x sub-frame Curved-Arc & Squash-Stretch In-Between between hero keyframes
    `ka` and `kb` (`70 x 10 = 700` hero frames) without scrambling unrelated contour topologies!
    """
    ka_i = int(ka) % 4
    kb_i = int(kb) % 4
    keys = hero_entry.get("keys", [])
    if not keys:
        return [], [], []
    k_a = keys[ka_i]
    if ka_i == kb_i or t_raw <= 0.02:
        return k_a.get("strokes", []), k_a.get("cel_polys", []), k_a.get("accent_polys", [])
    k_b = keys[kb_i]
    if t_raw >= 0.98:
        return k_b.get("strokes", []), k_b.get("cel_polys", []), k_b.get("accent_polys", [])

    # 10-step Biomechanical Parabolic Stride / Flight Arc around canvas center (160, 120)
    cx_h, cy_h = 160.0, 120.0
    if t_raw < 0.50:
        u = t_raw / 0.50
        src = k_a
        dx = 6.5 * math.sin(u * math.pi * 0.85)
        dy = -5.5 * math.sin(u * math.pi)
        sy = 1.0 - 0.025 * math.sin(u * math.pi) + 0.032 * (u ** 2)
        sx = 1.0 / sy
        rot_rad = math.radians(3.2 * math.sin(u * math.pi * 0.8))
    else:
        u = (t_raw - 0.50) / 0.50
        damp = 1.0 - u
        src = k_b
        dx = -6.0 * damp * math.cos(u * math.pi * 0.5)
        dy = -5.0 * (damp ** 1.5)
        sy = 1.0 + 0.030 * (damp ** 1.6) - 0.018 * damp * math.sin(u * math.pi)
        sx = 1.0 / sy
        rot_rad = math.radians(-2.8 * damp * math.cos(u * math.pi))

    def _tf(pt):
        du = (pt[0] - cx_h) * sx
        dv = (pt[1] - cy_h) * sy
        rx = du * math.cos(rot_rad) - dv * math.sin(rot_rad)
        ry = du * math.sin(rot_rad) + dv * math.cos(rot_rad)
        return [cx_h + rx + dx, cy_h + ry + dy]

    out_strokes = [[_tf(p) for p in st] for st in src.get("strokes", [])]
    cel_p = [[_tf(p) for p in poly] for poly in src.get("cel_polys", [])]
    acc_p = [[_tf(p) for p in poly] for poly in src.get("accent_polys", [])]
    return out_strokes, cel_p, acc_p


def render_progressive_keyframe_board(
    slug: str,
    k_idx: int,
    local_prog: float,
    out_w: int,
    out_h: int,
    boil_phase: int = 0,
    stage_mode: str = "auto",
    animate_4keys: bool = False,
) -> np.ndarray:
    """
    Renders a Keyframe Drafting Board (`K1..K4`) at exact 4:3 `(out_w, out_h)`
    using 100% TRUE VECTOR STROKE-BY-STROKE ANIMATION from `studio_traced.json["hero_shots"][slug]`.
    Supports:
      - `stage_mode="blue_only"`: Dedicated Non-Photo Blue construction & rough sketching + 4-key rolling!
      - `stage_mode="ink_only"`: Dedicated Speedball Black India Inking over faint blue roughs + 4-key rolling!
      - `stage_mode="gouache_only"`: Dedicated Gouache Cel Wash flooding + 4-key rolling!
      - `stage_mode="auto"`: Full A -> B -> C progression across `local_prog` (`0.0 .. 1.0`).
    """
    cat = get_studio_traced_catalog()
    hero_entry = cat.get("hero_shots", {}).get(slug)
    canvas = np.full((out_h, out_w, 3), BGR_GUTTER, dtype=np.uint8)

    gy_hor = int(round(out_h * 0.62))
    gx_phi1 = int(round(out_w * 0.382))
    gx_phi2 = int(round(out_w * 0.618))
    cv2.line(canvas, (0, gy_hor), (out_w - 1, gy_hor), (225, 208, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (gx_phi1, 0), (gx_phi1, out_h - 1), (225, 208, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (gx_phi2, 0), (gx_phi2, out_h - 1), (225, 208, 184), 1, cv2.LINE_AA)

    if not hero_entry:
        return canvas

    sx = out_w / 320.0
    sy = out_h / 240.0
    plate_strokes = hero_entry.get("plate_strokes", [])

    # Determine active keyframe pose (or smooth 4-keyframe morphing `K0 -> K1 -> K2 -> K3` when `animate_4keys=True`)
    if animate_4keys and local_prog > 0.52:
        u_anim = float(np.clip((local_prog - 0.52) / 0.48, 0.0, 0.999)) * 3.0
        ka = int(u_anim)
        kb = min(3, ka + 1)
        mt = u_anim - ka
        raw_hero_st, raw_cel_p, raw_acc_p = _morph_hero_strokes(hero_entry, ka, kb, mt)
        k_data = hero_entry["keys"][ka]
    else:
        k_data = hero_entry["keys"][int(k_idx) % 4]
        raw_hero_st = k_data.get("strokes", [])
        raw_cel_p = k_data.get("cel_polys", [])
        raw_acc_p = k_data.get("accent_polys", [])

    def map_strokes(st_list, boil_amp=0.0):
        out_st = []
        for s_i, st in enumerate(st_list):
            pts = []
            for p in st:
                px = p[0] * sx
                py = p[1] * sy
                if boil_amp > 0.0:
                    ang = p[0] * 0.11 + p[1] * 0.09 + boil_phase * 1.7 + s_i * 0.5
                    px += boil_amp * math.sin(ang)
                    py += boil_amp * math.cos(ang * 1.3)
                pts.append([px, py])
            if len(pts) >= 2:
                out_st.append(pts)
        return out_st

    boil_amp = 0.55 * min(sx, sy)
    plate_tf = map_strokes(plate_strokes, boil_amp=boil_amp * 0.5)
    const_tf = map_strokes(k_data.get("construction_strokes", []), boil_amp=0.0)
    hero_tf = map_strokes(raw_hero_st, boil_amp=boil_amp)
    cel_tf = map_strokes(raw_cel_p, boil_amp=boil_amp * 0.4)
    acc_tf = map_strokes(raw_acc_p, boil_amp=boil_amp * 0.4)

    nib_scale = max(0.55, min(1.15, out_w / 185.0))
    ink_thick = max(1, int(round(2.2 * min(sx, sy))))

    if stage_mode == "blue_only" or (stage_mode == "auto" and local_prog < 0.28):
        p_blue = float(np.clip(local_prog / (0.52 if stage_mode == "blue_only" else 0.28), 0.05, 1.0))
        all_blue_strokes = const_tf + plate_tf[::2] + hero_tf
        tip = _draw_strokes_progressive(
            canvas, all_blue_strokes, p_blue,
            color_bgr=BGR_NONPHOTO_BLUE, thickness=max(1, ink_thick),
        )
        if tip is not None and p_blue < 0.98 and out_w >= 110:
            draw_active_pen_nib(canvas, tip[0], tip[1], is_blue_pencil=True, scale=nib_scale)

    elif stage_mode == "ink_only" or (stage_mode == "auto" and local_prog < 0.76):
        p_ink = (
            float(np.clip(local_prog / 0.52, 0.04, 1.0))
            if stage_mode == "ink_only"
            else float(np.clip((local_prog - 0.28) / 0.48, 0.0, 1.0))
        )
        for st in plate_tf:
            cv2.polylines(canvas, [np.array(st, dtype=np.int32)], False, (195, 182, 168), 1, cv2.LINE_AA)
        for st in const_tf:
            cv2.polylines(canvas, [np.array(st, dtype=np.int32)], False, (228, 175, 115), 1, cv2.LINE_AA)

        tip = _draw_strokes_progressive(
            canvas, hero_tf, p_ink,
            color_bgr=BGR_INK, thickness=ink_thick,
            underdraw_bgr=BGR_NONPHOTO_BLUE, underdraw_thickness=1,
        )
        if tip is not None and p_ink < 0.98 and out_w >= 110:
            draw_active_pen_nib(canvas, tip[0], tip[1], is_blue_pencil=False, scale=nib_scale)

    else:
        p_cel = (
            float(np.clip(local_prog / 0.50, 0.05, 1.0))
            if stage_mode == "gouache_only"
            else float(np.clip((local_prog - 0.76) / 0.24, 0.0, 1.0))
        )
        sky_overlay = canvas.copy()
        cv2.rectangle(sky_overlay, (0, 0), (out_w, gy_hor), BGR_BLUE_PALE, -1)
        cv2.rectangle(sky_overlay, (0, gy_hor), (out_w, out_h), BGR_AMBER_PALE, -1)
        cv2.addWeighted(sky_overlay, 0.45 * p_cel, canvas, 1.0 - 0.45 * p_cel, 0, canvas)

        for st in plate_tf:
            cv2.polylines(canvas, [np.array(st, dtype=np.int32)], False, (125, 118, 110), 1, cv2.LINE_AA)

        cel_layer = canvas.copy()
        for cp in cel_tf:
            cv2.fillPoly(cel_layer, [np.array(cp, dtype=np.int32)], BGR_PAPER, cv2.LINE_AA)
        for ap in acc_tf:
            cv2.fillPoly(cel_layer, [np.array(ap, dtype=np.int32)], BGR_AMBER_PALE, cv2.LINE_AA)
        cv2.addWeighted(cel_layer, p_cel, canvas, 1.0 - p_cel, 0, canvas)

        for st in hero_tf:
            cv2.polylines(canvas, [np.array(st, dtype=np.int32)], False, BGR_INK, ink_thick, cv2.LINE_AA)

        if p_cel < 0.94 and hero_tf and out_w >= 110:
            idx_b = min(len(hero_tf) - 1, int(p_cel * len(hero_tf)))
            pt_b = hero_tf[idx_b][len(hero_tf[idx_b]) // 2]
            draw_active_pen_nib(canvas, int(round(pt_b[0])), int(round(pt_b[1])), is_brush=True, scale=nib_scale)

    if out_w >= 240:
        enhance_hero_board_with_code_multiplane(
            canvas, slug, local_prog, int(local_prog * 100), stage_mode=stage_mode, boil_phase=boil_phase
        )

    return canvas


_CROSS_MATCH_CACHE = {}


def _get_cross_sheet_matches(sheet_a_name: str, q_a: int, sheet_b_name: str, q_b: int) -> list:
    """
    Computes and caches stroke-to-stroke correspondence between `(sheet_a_name, q_a)`
    and `(sheet_b_name, q_b)` in normalized `[u, v]` bounding-box coordinates so any
    keyframe, breakdown, in-between, turnaround, or pass progression sheet can smoothly
    morph into any other sheet!
    """
    key = (sheet_a_name, int(q_a) % 4, sheet_b_name, int(q_b) % 4)
    if key in _CROSS_MATCH_CACHE:
        return _CROSS_MATCH_CACHE[key]

    cat = get_studio_traced_catalog()
    sa_obj = cat.get("sheets", {}).get(sheet_a_name)
    sb_obj = cat.get("sheets", {}).get(sheet_b_name)
    if not sa_obj or not sb_obj:
        _CROSS_MATCH_CACHE[key] = []
        return []

    if sheet_a_name == sheet_b_name:
        pm = sa_obj.get("pair_matches", {}).get(f"{int(q_a) % 4}_{int(q_b) % 4}")
        matches = pm.get("matches", []) if pm else []
        _CROSS_MATCH_CACHE[key] = matches
        return matches

    qa_data = sa_obj["quadrants"][int(q_a) % 4]
    qb_data = sb_obj["quadrants"][int(q_b) % 4]
    strokes_a = qa_data.get("strokes", [])
    strokes_b = qb_data.get("strokes", [])
    if not strokes_a or not strokes_b:
        _CROSS_MATCH_CACHE[key] = []
        return []

    ax0, ay0, ax1, ay1 = [float(v) for v in qa_data["bbox"]]
    bx0, by0, bx1, by1 = [float(v) for v in qb_data["bbox"]]
    aw, ah = max(16.0, ax1 - ax0), max(16.0, ay1 - ay0)
    bw, bh = max(16.0, bx1 - bx0), max(16.0, by1 - by0)
    acx, acy = 0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1)
    bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)

    def _feat(st, cx_b, cy_b, w_b, h_b):
        p0, pm_pt, p1 = st[0], st[len(st) // 2], st[-1]
        return (
            (p0[0] - cx_b) / w_b, (p0[1] - cy_b) / h_b,
            (pm_pt[0] - cx_b) / w_b, (pm_pt[1] - cy_b) / h_b,
            (p1[0] - cx_b) / w_b, (p1[1] - cy_b) / h_b,
        )

    fa = [_feat(s, acx, acy, aw, ah) for s in strokes_a]
    fb = [_feat(s, bcx, bcy, bw, bh) for s in strokes_b]
    used_b = set()
    matches = []
    for ia, (u0a, v0a, uma, vma, u1a, v1a) in enumerate(fa):
        best_ib = None
        best_d = 0.55
        best_rev = False
        for ib, (u0b, v0b, umb, vmb, u1b, v1b) in enumerate(fb):
            if ib in used_b:
                continue
            d_fwd = math.hypot(uma - umb, vma - vmb) + 0.5 * math.hypot(u0a - u0b, v0a - v0b) + 0.5 * math.hypot(u1a - u1b, v1a - v1b)
            d_rev = math.hypot(uma - umb, vma - vmb) + 0.5 * math.hypot(u0a - u1b, v0a - v1b) + 0.5 * math.hypot(u1a - u0b, v1a - v0b)
            if d_fwd < best_d or d_rev < best_d:
                if d_rev < d_fwd:
                    best_d = d_rev
                    best_ib = ib
                    best_rev = True
                else:
                    best_d = d_fwd
                    best_ib = ib
                    best_rev = False
        if best_ib is not None:
            used_b.add(best_ib)
            matches.append([ia, best_ib, best_rev])

    _CROSS_MATCH_CACHE[key] = matches
    return matches


def draw_studio_doodle_sprite(
    canvas: np.ndarray,
    sheet_name: str,
    quad_idx: int,
    cx: int,
    cy: int,
    max_w: int,
    max_h: int,
    tint_bgr: tuple = None,
    tint_strength: float = 0.28,
    flip_x: bool = False,
    draw_prog: float = 1.0,
    bob_phase: float = 0.0,
    boil_phase: int = 0,
    morph_to_quad: int = None,
    morph_t: float = 0.0,
    morph_to_sheet: str = None,
    onion_ghost_mode: str = None,
):
    """
    100% VECTOR STROKE-BY-STROKE RENDERER for all 60 Studio Sheets (240 Hand-Drawn Studio Frames)
    (`assets/studio_doodles/studio_traced.json`).
    ZERO raster bitmap cutouts! Every character, turnaround, pass progression, breakdown, in-between,
    multiplane layer, and studio machine is drawn dynamically from:
      1. Pass A/B Non-Photo Blue construction curves (`construction_strokes`)
      2. Pass C/E Watertight Gouache Cel polygons (`cel_polys` & `accent_polys`)
      3. Pass D Ordered Speedball India Ink vector strokes (`strokes`) with:
         - Progressive stroke-by-stroke drawing (`draw_prog` in `0.0 .. 1.0`) + live moving Dip-Pen Nib!
         - Biomechanical per-vertex head/arm/hand/tool articulation (`bob_phase`)!
         - Same-sheet AND cross-sheet Hungarian stroke-to-stroke pose morphing (`morph_to_sheet`, `morph_to_quad`, `morph_t`)!
         - Section XII Onion-Skin Flip Ghosting (`onion_ghost_mode="prev"|"next"`) in Non-Photo Blue / Warm Sepia!
         - Organic 6 Hz hand-drawn vector line boil (`boil_phase`)!
    """
    cat = get_studio_traced_catalog()
    sheet = cat.get("sheets", {}).get(sheet_name)
    if not sheet:
        return

    target_sheet_name = morph_to_sheet if morph_to_sheet else sheet_name
    sheet_b = cat.get("sheets", {}).get(target_sheet_name, sheet)

    is_agent = sheet.get("is_agent", False)
    q_i = int(quad_idx) % 4
    quad = sheet["quadrants"][q_i]

    bx0, by0, bx1, by1 = [float(v) for v in quad["bbox"]]
    has_morph = (
        morph_to_quad is not None
        and 0.02 < morph_t < 0.98
        and ((int(morph_to_quad) % 4) != q_i or target_sheet_name != sheet_name)
    )

    bw = max(16.0, bx1 - bx0)
    bh = max(16.0, by1 - by0)
    scx = 0.5 * (bx0 + bx1)
    scy = 0.5 * (by0 + by1)

    s_uni = min(float(max_w) / bw, float(max_h) / bh)
    # Volume-conserving squash-and-stretch breathing for agents
    breath = 1.0 + (0.016 * math.sin(bob_phase) if is_agent else 0.0)
    sx_final = s_uni * breath
    sy_final = s_uni / breath

    def transform_vertex(px: float, py: float, part: str = "torso", s_idx: int = 0) -> list:
        u = (px - scx) / bw  # normalized [-0.5, 0.5]
        v = (py - scy) / bh  # normalized [-0.5, 0.5]

        if is_agent:
            if part == "head" or (v < -0.04 and abs(u) < 0.34):
                piv_u, piv_v = 0.0, -0.04
                ang = math.radians(3.6 * math.sin(bob_phase * 1.25 + 0.4))
                du_p, dv_p = u - piv_u, v - piv_v
                u = piv_u + du_p * math.cos(ang) - dv_p * math.sin(ang)
                v = piv_v + du_p * math.sin(ang) + dv_p * math.cos(ang)
            elif part == "left_arm" or (u < -0.15 and v >= -0.12):
                piv_u, piv_v = -0.15, -0.02
                weight = min(1.0, abs(u - piv_u) / 0.30)
                ang = math.radians(7.2 * math.sin(bob_phase * 2.0))
                du_p, dv_p = u - piv_u, v - piv_v
                u = piv_u + du_p * math.cos(ang) - dv_p * math.sin(ang) + 0.038 * weight * math.cos(bob_phase * 2.0)
                v = piv_v + du_p * math.sin(ang) + dv_p * math.cos(ang) + 0.026 * weight * math.sin(bob_phase * 2.0)
            elif part == "right_arm" or (u > 0.15 and v >= -0.12):
                piv_u, piv_v = 0.15, -0.02
                weight = min(1.0, abs(u - piv_u) / 0.30)
                ang = math.radians(-7.2 * math.sin(bob_phase * 2.0 + 0.7))
                du_p, dv_p = u - piv_u, v - piv_v
                u = piv_u + du_p * math.cos(ang) - dv_p * math.sin(ang) - 0.038 * weight * math.sin(bob_phase * 2.0)
                v = piv_v + du_p * math.sin(ang) + dv_p * math.cos(ang) + 0.028 * weight * math.cos(bob_phase * 2.0)

        if flip_x:
            u = -u

        ox = cx + u * bw * sx_final
        oy = cy + v * bh * sy_final

        # Organic 6 Hz hand-drawn vector line boil
        boil_ang = px * 0.11 + py * 0.09 + boil_phase * 1.85 + s_idx * 0.45
        boil_r = 0.82 * min(1.2, s_uni)
        ox += boil_r * math.sin(boil_ang)
        oy += boil_r * math.cos(boil_ang * 1.27)
        return [ox, oy]

    # Build transformed polygons & strokes (with 10x Curved-Arc Normalized Stroke In-Betweening!)
    strokes_raw = quad["strokes"]
    parts_raw = quad.get("stroke_parts", ["torso"] * len(strokes_raw))
    active_cel_polys = quad.get("cel_polys", [])
    active_acc_polys = quad.get("accent_polys", [])

    if has_morph:
        q_b = int(morph_to_quad) % 4
        strokes_raw, parts_raw, active_cel_polys, active_acc_polys = build_10x_curved_arc_strokes(
            sheet, q_i, sheet_b, q_b, morph_t, key_id=(sheet_name, q_i, target_sheet_name, q_b)
        )

    # Onion-Skin Ghost Mode (`previous → current → next`, Section XII: Flip the Drawings!)
    if onion_ghost_mode is not None:
        ghost_ov = canvas.copy()
        ghost_col = (218, 165, 102) if onion_ghost_mode == "prev" else (182, 198, 224)
        for i, st in enumerate(strokes_raw[::2]):
            pts_g = [transform_vertex(p[0], p[1], parts_raw[i * 2] if (i * 2) < len(parts_raw) else "torso", i) for p in st]
            if len(pts_g) >= 2:
                cv2.polylines(ghost_ov, [np.array(pts_g, dtype=np.int32)], False, ghost_col, 1, cv2.LINE_AA)
        cv2.addWeighted(ghost_ov, 0.32, canvas, 0.68, 0, canvas)
        return

    cel_tf = [[transform_vertex(p[0], p[1], "torso", 0) for p in poly] for poly in active_cel_polys]
    acc_tf = [[transform_vertex(p[0], p[1], "torso", 1) for p in poly] for poly in active_acc_polys]
    const_tf = [[transform_vertex(p[0], p[1], "torso", 2) for p in st] for st in quad.get("construction_strokes", [])]
    strokes_tf = [
        [transform_vertex(p[0], p[1], parts_raw[i] if i < len(parts_raw) else "torso", i) for p in st]
        for i, st in enumerate(strokes_raw)
    ]

    # 1. Pass C/E Watertight Opaque Gouache Cel Fill (fades in smoothly during stroke-by-stroke creation)
    cel_alpha = float(np.clip((draw_prog - 0.10) / 0.45, 0.30, 1.0)) if draw_prog < 0.99 else 1.0
    base_cel_col = BGR_GUTTER
    accent_col = tint_bgr if tint_bgr is not None else BGR_AMBER_PALE

    if cel_tf:
        target_canvas = canvas if cel_alpha >= 0.98 else canvas.copy()
        for cp in cel_tf:
            cv2.fillPoly(target_canvas, [np.array(cp, dtype=np.int32)], base_cel_col, cv2.LINE_AA)
        for ap in acc_tf:
            cv2.fillPoly(target_canvas, [np.array(ap, dtype=np.int32)], accent_col, cv2.LINE_AA)
        if cel_alpha < 0.98:
            cv2.addWeighted(target_canvas, cel_alpha, canvas, 1.0 - cel_alpha, 0, canvas)

    # 2. Pass A/B Non-Photo Blue Construction Curves (visible during stroke-by-stroke sketch & pose breakdowns!)
    is_breakdown = has_morph and 0.12 < morph_t < 0.88
    if draw_prog < 0.82 or is_breakdown or ("breakdown" in sheet_name or "inbetween" in sheet_name or "pass_progression" in sheet_name):
        const_col = BGR_NONPHOTO_BLUE if (draw_prog < 0.55 or is_breakdown) else (222, 188, 148)
        p_const = float(np.clip(draw_prog / 0.22, 0.15, 1.0))
        _draw_strokes_progressive(canvas, const_tf, p_const, color_bgr=const_col, thickness=1)

    # 3. Pass D Ordered Speedball India Ink Vector Strokes (animated stroke-by-stroke with live Dip-Pen Nib!)
    ink_thick = max(2, int(round(2.4 * min(1.15, s_uni))))
    p_ink = float(np.clip((draw_prog - 0.08) / 0.88, 0.0, 1.0)) if draw_prog < 0.999 else 1.0
    tip = _draw_strokes_progressive(
        canvas,
        strokes_tf,
        p_ink,
        color_bgr=BGR_INK,
        thickness=ink_thick,
        underdraw_bgr=BGR_NONPHOTO_BLUE,
        underdraw_thickness=1,
    )
    if tip is not None and draw_prog < 0.98:
        draw_active_pen_nib(canvas, tip[0], tip[1], is_blue_pencil=(draw_prog < 0.18), scale=max(0.7, min(1.1, s_uni)))


def draw_multi_sheet_chain_sprite(
    canvas: np.ndarray,
    frame_chain: list,
    sub_step: float,
    cx: int,
    cy: int,
    max_w: int,
    max_h: int,
    tint_bgr: tuple = None,
    tint_strength: float = 0.34,
    stroke_in_end: float = 0.12,
    onion_skin: bool = False,
    bob_phase: float = 0.0,
    boil_phase: int = 0,
    flip_x: bool = False,
) -> tuple:
    """
    Chains `8 to 20` hand-drawn frames across multiple 2x2 sheets (`frame_chain = [(sheet_name, quad_idx, code_lbl), ...]`)!
      - `0.00 .. stroke_in_end`: draws `frame_chain[0]` stroke-by-stroke with a live moving Dip-Pen Nib (`draw_prog: 0.14 -> 1.00`).
      - `stroke_in_end .. 1.00`: steps through every hand-drawn Turnaround, Pass A/B/D, Story Key (`K01..K08`),
        Breakdown (`B01..B04`), and In-Between (`I01..I04`) in `frame_chain`!
      - Same-sheet transitions (`sh_a == sh_b`) smoothly morph via Hungarian stroke correspondence; cross-sheet
        Key -> Breakdown -> In-Between transitions step cleanly onto the hand-drawn Breakdown/In-Between drawing
        with Section XII Onion-Skin Flip Ghosting (`previous → current → next`)!
    Returns `(active_4stage_idx, active_code_label, n_chain, cur_idx)`.
    """
    n_chain = len(frame_chain)
    if n_chain == 0:
        return 0, "K01", 1, 0

    dx_bob = int(round(1.4 * math.cos(bob_phase)))
    dy_bob = int(round(2.0 * math.sin(bob_phase * 2.0)))

    if n_chain == 1 or sub_step < stroke_in_end:
        sh0, q0, lbl0 = frame_chain[0]
        dp = float(np.clip(0.14 + 0.86 * (sub_step / max(stroke_in_end, 1e-4)), 0.14, 1.0)) if sub_step < stroke_in_end else 1.0
        draw_studio_doodle_sprite(
            canvas,
            sheet_name=sh0,
            quad_idx=q0,
            cx=cx + dx_bob,
            cy=cy + dy_bob,
            max_w=max_w,
            max_h=max_h,
            tint_bgr=tint_bgr,
            tint_strength=tint_strength,
            flip_x=flip_x,
            draw_prog=dp,
            bob_phase=bob_phase,
            boil_phase=boil_phase,
        )
        return 0, lbl0, n_chain, 0

    n_segs = n_chain - 1
    u = float(np.clip((sub_step - stroke_in_end) / max(1.0 - stroke_in_end, 1e-4), 0.0, 0.9999))
    seg_f = u * n_segs
    seg_i = min(n_segs - 1, int(seg_f))
    local_u = seg_f - seg_i

    sh_a, q_a, lbl_a = frame_chain[seg_i]
    sh_b, q_b, lbl_b = frame_chain[seg_i + 1]

    # 10x Sub-Frame Quantized Curved-Arc In-Betweening (`sub_k in 0..9`) across all 240 studio drawings!
    sub_k = min(9, int(local_u * 10.0))
    if sub_k <= 1:
        mt = 0.0
        sh_next, q_next = None, None
        cur_idx = seg_i
        cur_lbl = f"{lbl_a} [{sub_k + 1}/10]"
    elif sub_k >= 9:
        sh_a, q_a, lbl_a = sh_b, q_b, lbl_b
        mt = 0.0
        sh_next, q_next = None, None
        cur_idx = seg_i + 1
        cur_lbl = f"{lbl_b} [10/10]"
    else:
        mt = float((sub_k - 1) / 8.0)
        sh_next, q_next = sh_b, q_b
        cur_idx = seg_i if mt < 0.5 else (seg_i + 1)
        base_l = lbl_a if mt < 0.5 else lbl_b
        cur_lbl = f"{base_l} [INB {sub_k + 1}/10]"

    # Section XII: Onion-Skin Flip Comparison (`previous → current → next`)
    if onion_skin and cur_idx > 0:
        sh_prev, q_prev, _ = frame_chain[cur_idx - 1]
        draw_studio_doodle_sprite(
            canvas,
            sheet_name=sh_prev,
            quad_idx=q_prev,
            cx=cx + dx_bob - 6,
            cy=cy + dy_bob,
            max_w=max_w,
            max_h=max_h,
            flip_x=flip_x,
            bob_phase=bob_phase,
            boil_phase=boil_phase,
            onion_ghost_mode="prev",
        )

    draw_studio_doodle_sprite(
        canvas,
        sheet_name=sh_a,
        quad_idx=q_a,
        cx=cx + dx_bob,
        cy=cy + dy_bob,
        max_w=max_w,
        max_h=max_h,
        tint_bgr=tint_bgr,
        tint_strength=tint_strength,
        flip_x=flip_x,
        draw_prog=1.0,
        bob_phase=bob_phase,
        boil_phase=boil_phase,
        morph_to_quad=q_next,
        morph_t=mt,
        morph_to_sheet=sh_next,
    )
    stage_4_idx = min(3, int((cur_idx / float(max(1, n_chain))) * 4))
    return stage_4_idx, cur_lbl, n_chain, cur_idx


def draw_7plane_multiplane_world(
    canvas: np.ndarray,
    bx0: int,
    by0: int,
    bx1: int,
    by1: int,
    cap_h: int,
    floor_y: int,
    scene_idx: int,
    sub_step: float,
    boil_phase: int = 0,
    layer_pass: str = "bg",
):
    """
    Implements Section V & VI (`BUILD THE MULTIPLANE WORLD` & `MULTIPLANE CAMERA PHYSICS`:
    `screen_motion ∝ camera_motion / layer_depth`) using our 5 dedicated hand-drawn Multiplane sheets:
      - PLANE 6 (`depth = 6.0`): `multiplane_plane6_sky_atmosphere_keys` (Sky gradient & celestial rays)
      - PLANE 5 (`depth = 4.0`): `multiplane_plane5_distant_bg_keys` (Distant studio rafters & water tower)
      - PLANE 4 (`depth = 2.4`): `multiplane_plane4_midground_arch_keys` (Midground studio columns & arched windows)
      - PLANE 1 (`depth = 0.70`): `multiplane_plane1_foreground_props_keys` (Foreground drafting table props & ink bottles)
      - PLANE 0 (`depth = 0.42`): `multiplane_plane0_fg_mask_keys` (Extreme foreground proscenium beam & camera cables)
    """
    cam_motion = (sub_step - 0.5) * 44.0  # Lateral multiplane camera dolly pan
    q_plane = (scene_idx + int(sub_step * 3.99)) % 4

    if layer_pass == "bg":
        bg_layer = canvas.copy()
        # PLANE 6 (Sky / Atmosphere, depth = 6.0 -> slowest parallax!)
        dx_p6 = int(round(cam_motion / 6.0))
        draw_studio_doodle_sprite(
            bg_layer, "multiplane_plane6_sky_atmosphere_keys", q_plane,
            cx=180 + dx_p6, cy=by0 + cap_h + 74, max_w=240, max_h=110,
            tint_bgr=BGR_BLUE_PALE, boil_phase=boil_phase,
        )
        # PLANE 5 (Distant Background Rafters, depth = 4.0)
        dx_p5 = int(round(cam_motion / 4.0))
        draw_studio_doodle_sprite(
            bg_layer, "multiplane_plane5_distant_bg_keys", (q_plane + 1) % 4,
            cx=436 + dx_p5, cy=by0 + cap_h + 76, max_w=210, max_h=114,
            tint_bgr=BGR_AMBER_PALE, boil_phase=boil_phase + 1,
        )
        # PLANE 4 (Midground Studio Columns & Windows, depth = 2.4)
        dx_p4 = int(round(cam_motion / 2.4))
        draw_studio_doodle_sprite(
            bg_layer, "multiplane_plane4_midground_arch_keys", q_plane,
            cx=180 + dx_p4, cy=floor_y - 140, max_w=260, max_h=175,
            tint_bgr=BGR_PAPER_DARK, boil_phase=boil_phase + 2,
        )
        cv2.addWeighted(bg_layer, 0.16, canvas, 0.84, 0, canvas)

    elif layer_pass == "fg":
        fg_layer = canvas.copy()
        # PLANE 1 (Foreground Props Crossing Bench, depth = 0.70 -> strong differential parallax!)
        dx_p1 = int(round(cam_motion / 0.70))
        draw_studio_doodle_sprite(
            fg_layer, "multiplane_plane1_foreground_props_keys", q_plane,
            cx=336 + dx_p1, cy=by1 - 68, max_w=96, max_h=64,
            tint_bgr=BGR_AMBER_PALE, boil_phase=boil_phase,
        )
        # PLANE 0 (Extreme Foreground Camera Mask / Proscenium Post, depth = 0.42 -> fastest parallax!)
        dx_p0 = int(round(cam_motion / 0.42))
        draw_studio_doodle_sprite(
            fg_layer, "multiplane_plane0_fg_mask_keys", q_plane,
            cx=bx0 + 32 + dx_p0, cy=by0 + cap_h + 95, max_w=72, max_h=140,
            tint_bgr=BGR_PAPER_DARK, boil_phase=boil_phase + 1,
        )
        cv2.addWeighted(fg_layer, 0.22, canvas, 0.78, 0, canvas)


def draw_agent_animator(
    img: np.ndarray,
    agent_id: str,
    cx: int,
    cy: int,
    scale: float = 1.0,
    pose: int = 0,
    dark_bg: bool = False,
    flip_x: bool = False,
    bob_phase: float = 0.0,
    boil_phase: int = 0,
    draw_prog: float = 1.0,
    morph_to_pose: int = None,
    morph_t: float = 0.0,
):
    """
    Draws one of our 5 bespoke Agent Animator characters (`nib`, `blueline`, `celinker`,
    `sprocket`, `chime`) 100% from their traced vector stroke rigs (`studio_traced.json`)
    with progressive stroke-by-stroke drawing (`draw_prog`), biomechanical limb articulation
    (`bob_phase`), Hungarian stroke pose morphing (`morph_to_pose`, `morph_t`), and 6 Hz line boil!
    """
    ag = STUDIO_AGENTS.get(agent_id, STUDIO_AGENTS["nib"])
    sheet_map = {
        "nib": "agent_nib_keys",
        "blueline": "agent_blueline_keys",
        "celinker": "agent_celinker_keys",
        "sprocket": "agent_sprocket_keys",
        "chime": "agent_chime_keys",
    }
    sheet_name = sheet_map.get(agent_id, "agent_nib_keys")
    max_w = int(round(176 * scale))
    max_h = int(round(184 * scale))
    dx_bob = int(round(1.5 * math.cos(bob_phase)))
    dy_bob = int(round(2.2 * math.sin(bob_phase * 2.0)))
    draw_studio_doodle_sprite(
        img,
        sheet_name=sheet_name,
        quad_idx=int(pose) % 4,
        cx=cx + dx_bob,
        cy=cy + int(round(12 * scale)) + dy_bob,
        max_w=max_w,
        max_h=max_h,
        tint_bgr=ag["pale_bgr"],
        tint_strength=0.38,
        flip_x=flip_x,
        draw_prog=draw_prog,
        bob_phase=bob_phase,
        boil_phase=boil_phase,
        morph_to_quad=morph_to_pose,
        morph_t=morph_t,
    )


def draw_studio_room_architecture(
    canvas: np.ndarray,
    bx0: int,
    by0: int,
    bx1: int,
    by1: int,
    cap_h: int,
    floor_y: int,
    light_xs: list = None,
    boil_phase: int = 0,
):
    """
    Draws cohesive hand-inked Quick, Draw! studio room architecture using multi-vertex
    wobbled ink strokes (`draw_hand_inked_line` / `draw_hand_inked_rect`):
      - Warm ceiling beam & hanging studio lamp cones (Pass F volumetric light)
      - Hand-wobbled vertical wall wainscoting panels & chair-rail molding
      - Hand-wobbled perspective hardwood floorboards converging toward the room horizon (`floor_y`)
    """
    top_y = by0 + cap_h
    cv2.rectangle(canvas, (bx0, top_y), (bx1, floor_y), BGR_PAPER, -1)
    cv2.rectangle(canvas, (bx0, floor_y), (bx1, by1), BGR_PAPER_DARK, -1)

    # Hand-wobbled vertical wainscoting wall lines
    for i_w, wx in enumerate(range(bx0 + 80, bx1 - 40, 110)):
        draw_hand_inked_line(canvas, (wx, top_y + 4), (wx, floor_y), (212, 222, 232), 1, wobble=0.9, boil_phase=boil_phase + i_w)
    # Hand-wobbled chair-rail molding line slightly above floor
    draw_hand_inked_line(canvas, (bx0, floor_y - 24), (bx1, floor_y - 24), (188, 202, 214), 2, wobble=1.1, boil_phase=boil_phase)

    # Volumetric ceiling light cones (Pass F)
    if light_xs:
        overlay = canvas.copy()
        for lx in light_xs:
            cone = np.array([
                [lx - 22, top_y],
                [lx + 22, top_y],
                [lx + 115, floor_y],
                [lx - 115, floor_y],
            ], dtype=np.int32)
            cv2.fillConvexPoly(overlay, cone, BGR_GUTTER, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.38, canvas, 0.62, 0, canvas)

    # Bold hand-inked floorboard horizon line & perspective plank lines
    draw_hand_inked_line(canvas, (bx0, floor_y), (bx1, floor_y), BGR_INK, 3, wobble=1.3, boil_phase=boil_phase)
    cx_vanish = (bx0 + bx1) // 2
    for i_f, fx in enumerate(range(bx0 - 120, bx1 + 160, 95)):
        x_top = int(round(cx_vanish + (fx - cx_vanish) * 0.62))
        if bx0 <= x_top <= bx1:
            draw_hand_inked_line(canvas, (x_top, floor_y), (max(bx0, min(bx1, fx)), by1), (172, 188, 202), 2, wobble=0.95, boil_phase=boil_phase + i_f)


def draw_studio_workbench(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    fill_bgr: tuple = BGR_AMBER_PALE,
    boil_phase: int = 0,
):
    """
    Draws a solid hand-inked wooden studio workbench / counter in the foreground (`y0..y1`)
    with wobbled ink edges so waist-up Agent Animators stand or sit naturally behind a physical table.
    """
    draw_hand_inked_rect(canvas, x0, y0 + 10, x1, y1, fill_bgr=fill_bgr, outline_bgr=BGR_INK, thickness=3, wobble=1.15, boil_phase=boil_phase)
    draw_hand_inked_rect(canvas, x0 - 4, y0, x1 + 4, y0 + 10, fill_bgr=BGR_PAPER_DARK, outline_bgr=BGR_INK, thickness=3, wobble=1.1, boil_phase=boil_phase + 1)
    mid_y = (y0 + 10 + y1) // 2
    draw_hand_inked_line(canvas, (x0 + 12, mid_y), (x1 - 12, mid_y), BGR_INK_SOFT, 1, wobble=0.85, boil_phase=boil_phase)
    for bx in range(x0 + 24, x1 - 16, 140):
        cv2.circle(canvas, (bx, y0 + 5), 2, BGR_INK, -1, cv2.LINE_AA)


def draw_spinning_wheel(canvas: np.ndarray, cx: int, cy: int, radius: int, angle_deg: float, color=BGR_INK):
    """Draws rotating reel/wheel spokes at `(cx, cy)` to bring film reels & camera cranks to life."""
    for s_i in range(5):
        ang = math.radians(angle_deg + s_i * 72.0)
        x1 = int(round(cx + (radius - 2) * math.cos(ang)))
        y1 = int(round(cy + (radius - 2) * math.sin(ang)))
        cv2.line(canvas, (cx, cy), (x1, y1), color, 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, BGR_AMBER, -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, BGR_INK, 2, cv2.LINE_AA)


def _compute_pose_transition(sub_val: float, p_early: int, p_late: int, split_t: float = 0.55, span: float = 0.18):
    """Computes `(cur_pose, morph_to_pose, morph_t)` for smooth Hungarian vector stroke morphing across a pose change."""
    t0 = split_t - span * 0.5
    t1 = split_t + span * 0.5
    if sub_val <= t0:
        return p_early, None, 0.0
    elif sub_val >= t1:
        return p_late, None, 0.0
    else:
        return p_early, p_late, float((sub_val - t0) / max(span, 1e-4))


def _compute_key_sequence_schedule(
    sub_step: float,
    key_sequence: tuple = (0, 1, 2, 3),
    stroke_in_end: float = 0.20,
):
    """
    Schedules a multi-keyframe hand-drawn sheet (`Q0 -> Q1 -> Q2 -> Q3`) over `sub_step in [0.0, 1.0]`:
      - `0.00 .. stroke_in_end`: draws `key_sequence[0]` stroke-by-stroke with a live moving Dip-Pen Nib (`draw_prog: 0.14 -> 1.00`).
      - `stroke_in_end .. 1.00`: holds each key pose clearly and smoothly morphs between consecutive key poses using Hungarian vector stroke correspondence.
    Returns `(draw_prog, q_cur, q_next, morph_t, active_seq_idx)`.
    """
    if len(key_sequence) <= 1:
        q0 = key_sequence[0] if key_sequence else 0
        dp = float(np.clip(0.14 + 0.86 * (sub_step / max(stroke_in_end, 1e-4)), 0.14, 1.0)) if sub_step < stroke_in_end else 1.0
        return dp, q0, None, 0.0, 0

    if sub_step < stroke_in_end:
        dp = float(np.clip(0.14 + 0.86 * (sub_step / max(stroke_in_end, 1e-4)), 0.14, 1.0))
        return dp, key_sequence[0], None, 0.0, 0

    n_segs = len(key_sequence) - 1
    u = float(np.clip((sub_step - stroke_in_end) / max(1.0 - stroke_in_end, 1e-4), 0.0, 1.0))
    seg_f = u * n_segs
    seg_i = min(n_segs - 1, int(seg_f))
    local_u = seg_f - seg_i
    q_a = key_sequence[seg_i]
    q_b = key_sequence[seg_i + 1]

    if local_u <= 0.20:
        return 1.0, q_a, None, 0.0, seg_i
    elif local_u >= 0.80:
        return 1.0, q_b, None, 0.0, seg_i + 1
    else:
        mt = float((local_u - 0.20) / 0.60)
        act_idx = seg_i if mt < 0.5 else (seg_i + 1)
        return 1.0, q_a, q_b, mt, act_idx


def draw_4key_sequence_sprite(
    canvas: np.ndarray,
    sheet_name: str,
    sub_step: float,
    cx: int,
    cy: int,
    max_w: int,
    max_h: int,
    tint_bgr: tuple = None,
    tint_strength: float = 0.34,
    key_sequence: tuple = (0, 1, 2, 3),
    stroke_in_end: float = 0.20,
    bob_phase: float = 0.0,
    boil_phase: int = 0,
    flip_x: bool = False,
) -> int:
    """
    Draws any 2x2 hand-drawn studio sheet (`sheet_name`) through its full 4-keyframe
    action sequence (`Q0 -> Q1 -> Q2 -> Q3`) with initial stroke-by-stroke creation
    and Hungarian vector stroke morphing. Returns `active_seq_idx` (`0..3`).
    """
    dp, q_cur, q_next, mt, act_idx = _compute_key_sequence_schedule(
        sub_step, key_sequence=key_sequence, stroke_in_end=stroke_in_end
    )
    dx_bob = int(round(1.4 * math.cos(bob_phase)))
    dy_bob = int(round(2.0 * math.sin(bob_phase * 2.0)))
    draw_studio_doodle_sprite(
        canvas,
        sheet_name=sheet_name,
        quad_idx=q_cur,
        cx=cx + dx_bob,
        cy=cy + dy_bob,
        max_w=max_w,
        max_h=max_h,
        tint_bgr=tint_bgr,
        tint_strength=tint_strength,
        flip_x=flip_x,
        draw_prog=dp,
        bob_phase=bob_phase,
        boil_phase=boil_phase,
        morph_to_quad=q_next,
        morph_t=mt,
    )
    return act_idx


def draw_macro_inset_box(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    sheet_name: str,
    sub_step: float,
    key_sequence: tuple = (0, 1, 2, 3),
    tint_bgr: tuple = BGR_AMBER_PALE,
    boil_phase: int = 0,
    frame_chain: list = None,
    onion_skin: bool = False,
) -> int:
    """
    Draws an exact 4:3 (`184x138`) Macro Hand/Mechanism/Thumbnail/Turnaround Close-Up Inset Box
    in the center column, supporting either a single 2x2 `sheet_name` or a multi-sheet `frame_chain`!
    """
    draw_hand_inked_rect(
        canvas, x0, y0, x0 + w, y0 + h,
        fill_bgr=BGR_PAPER, outline_bgr=BGR_INK, thickness=3,
        boil_phase=boil_phase, shadow_offset=4,
    )
    cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + w - 4, y0 + h - 4), BGR_GUTTER, 1)
    if frame_chain:
        act_idx, _, _, _ = draw_multi_sheet_chain_sprite(
            canvas,
            frame_chain=frame_chain,
            sub_step=sub_step,
            cx=x0 + w // 2,
            cy=y0 + h // 2 + 4,
            max_w=w - 14,
            max_h=h - 14,
            tint_bgr=tint_bgr,
            stroke_in_end=0.14,
            onion_skin=onion_skin,
            bob_phase=sub_step * math.pi * 4.0,
            boil_phase=boil_phase + 1,
        )
    else:
        act_idx = draw_4key_sequence_sprite(
            canvas,
            sheet_name=sheet_name,
            sub_step=sub_step,
            cx=x0 + w // 2,
            cy=y0 + h // 2 + 4,
            max_w=w - 14,
            max_h=h - 14,
            tint_bgr=tint_bgr,
            key_sequence=key_sequence,
            stroke_in_end=0.18,
            bob_phase=sub_step * math.pi * 4.0,
            boil_phase=boil_phase + 1,
        )
    draw_hand_inked_rect(
        canvas, x0, y0, x0 + w, y0 + h,
        fill_bgr=None, outline_bgr=BGR_INK, thickness=3,
        boil_phase=boil_phase,
    )
    return act_idx


def draw_4key_progress_strip(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    total_w: int,
    active_idx: int,
    pose_labels: list,
    accent_rgb: tuple,
):
    """
    Draws a 4-stage Multiplane Production Tracker bar below the giant 384x288 (4:3) right-hand stage.
    """
    n_k = len(pose_labels)
    gap = 6
    cell_w = (total_w - gap * (n_k - 1)) // n_k
    cell_h = 34
    for i, lbl in enumerate(pose_labels):
        cx0 = x0 + i * (cell_w + gap)
        cx1 = cx0 + cell_w
        is_act = (i == active_idx)
        fill_c = accent_rgb if is_act else (251, 229, 194)
        txt_c = (255, 251, 244) if is_act else (22, 20, 18)
        draw.rounded_rectangle([cx0 + 2, y0 + 2, cx1 + 2, y0 + cell_h + 2], radius=5, fill=(22, 20, 18))
        draw.rounded_rectangle([cx0, y0, cx1, y0 + cell_h], radius=5, fill=fill_c, outline=(22, 20, 18), width=3 if is_act else 2)
        f_p = fit_text_font(draw, lbl, cell_w - 10, [FONT_LABEL_SM, FONT_CAPTION_SM])
        bb = draw.textbbox((0, 0), lbl, font=f_p)
        tw = bb[2] - bb[0]
        draw.text((cx0 + (cell_w - tw) // 2, y0 + 9), lbl, font=f_p, fill=txt_c)


# =============================================================================
# 8-SCENE SEQUENTIAL MULTIPLANE STUDIO ENGINE (240 STUDIO DRAWINGS CHAINED!)
# =============================================================================
SCENE_HEADERS = {
    1: ("ACT I-A · THUMBNAILS & 35MM MOVIOLA", "nib", ["1. TURN & PASS A/B", "2. KEYS K01..K02", "3. BRK B01 & INB I01", "4. EXT KEYS K05..K08"]),
    2: ("ACT I-B · TURNAROUNDS & STORYBOARD", "blueline", ["1. MODEL TURNS", "2. PIN KEYS K01..04", "3. MOTION ARCS", "4. 4-CARD LOCK"]),
    3: ("ACT II-A · KEYS, BREAKDOWNS & IN-BETWEENS", "blueline", ["1. PASS A/B/D", "2. KEYS K01..K04", "3. BRK B01..04 & INB", "4. FLIP TEST K05..08"]),
    4: ("ACT II-B · INSIDE-OUT PASS & INDIA INKING", "nib", ["1. PASS A→B→D→E", "2. INK KEYS K01..02", "3. INK BRK B01..04", "4. CEL CHECK K03..04"]),
    5: ("ACT II-C · GOUACHE CEL OPAQUING & RACK", "celinker", ["1. CEL PASS A..E", "2. FLOOD KEYS K01..04", "3. BRUSH BRK B01..04", "4. RACK EXT K05..08"]),
    6: ("ACT III · 7-PLANE CAMERA TOWER & PARALLAX", "celinker", ["1. PLANE 6..4 BG", "2. LOAD KEYS & BRK", "3. PARALLAX OPTICS", "4. PLANE 1..0 FG"]),
    7: ("ACT IV-A · 16-GATE AUDIT & FOLEY STAGE", "sprocket", ["1. CALIPER K01..04", "2. GATE BRK B01..04", "3. FOLEY KEYS & BRK", "4. FOLEY EXT K05..08"]),
    8: ("ACT IV-B · PLAYBACK & 35MM PREMIERE", "chime", ["1. SCREEN KEYS K01..02", "2. CREW BRK B01..02", "3. CHEER KEYS K03..04", "4. BOW BRK B03..04"]),
}


def render_single_panel_fullscreen(
    spec: dict,
    panel_num: int,
    sub_step: float = 1.0,
    shot_frame_idx: int = 16,
    boil_phase: int = 0,
) -> np.ndarray:
    """
    Renders Full-Screen Cartoon Scene `panel_num` (`1..8`) at `960x540` (16:9) resolution,
    chaining all 60 hand-drawn 2x2 Studio Sheets (`240` Studio Drawings) + 14 Hero Shots (`70` Drawings)
    = `310` Vector-Traced Hand-Drawn Frames across the 17-Phase Multiplane Animation Pipeline:
      - Phase 2 Thumbnail Staging (`phase02_staging_thumbnails_keys`)
      - Phase 3, 5, 6 7-Plane Multiplane World & Camera Parallax (`PLANE 0..6`, `screen_motion ∝ camera_motion / layer_depth`)
      - Phase 21 Character Turnaround Model Sheets (`model_sheet_*_turns`)
      - Phase 22 Inside-Out Pass Progressions (`*_pass_progression`: Pass A Gesture -> Pass B Volume -> Pass D Ink -> Pass E/F Tone)
      - Phase 4, 6, 8 Story Keys (`K01..K08`), Breakdowns with Dotted Motion Arcs (`B01..B04`), In-Betweens (`I01..I04`),
        and Section XII Onion-Skin Flip Ghosting (`previous → current → next`)!
    """
    W, H = 960, 540
    canvas = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)
    slug = spec["featured_shot"]
    arch_sketches, blue_roughs, ink_cels, cart_frames = load_shot_frames(slug)

    p_idx = ((int(panel_num) - 1) % 8) + 1
    act_title, scene_ag_id, pose_labels = SCENE_HEADERS[p_idx]
    scene_ag = STUDIO_AGENTS[scene_ag_id]

    pad = 16
    cap_h = 52
    bx0, by0, bx1, by1 = pad, pad, W - pad, H - pad
    floor_y = by1 - 88
    bob_rad = sub_step * math.pi * 4.0

    # Studio architecture + 7-Plane Multiplane Background (`PLANE 6 Sky`, `PLANE 5 Distant BG`, `PLANE 4 Midground`)
    light_xs = [180, 438, 736] if p_idx != 8 else None
    draw_studio_room_architecture(canvas, bx0, by0, bx1, by1, cap_h, floor_y, light_xs=light_xs, boil_phase=boil_phase)
    draw_7plane_multiplane_world(canvas, bx0, by0, bx1, by1, cap_h, floor_y, p_idx, sub_step, boil_phase=boil_phase, layer_pass="bg")

    mac_x, mac_y, mac_w, mac_h = 344, by0 + cap_h + 36, 184, 138
    rx0, ry0, rw, rh = 544, by0 + cap_h + 36, 384, 288

    cone_poly = np.array([
        [mac_x + mac_w, mac_y + mac_h // 2],
        [rx0, ry0 + 8],
        [rx0, ry0 + rh - 8],
        [mac_x + mac_w, floor_y - 45],
    ], dtype=np.int32)
    ov_cone = canvas.copy()
    cv2.fillConvexPoly(ov_cone, cone_poly, BGR_AMBER_PALE, cv2.LINE_AA)
    cv2.addWeighted(ov_cone, 0.38, canvas, 0.62, 0, canvas)

    active_pose_idx = 0
    active_code_lbl = "K01"
    chain_len = 16
    chain_cur = 0
    macro_caption = "MACRO HAND CAM (4:3)"
    apparatus_caption = "STUDIO APPARATUS"
    right_board_caption = f"4:3 STAGE — #{slug.upper()}"

    # =========================================================================
    # SCENE 1: ACT I-A — THUMBNAIL STAGING (`T01..T04`) & NIB AT THE 35MM MOVIOLA (16 FRAMES!)
    # =========================================================================
    if p_idx == 1:
        nib_moviola_chain = [
            ("model_sheet_nib_turns", 0, "TURN-FRONT"),
            ("model_sheet_nib_turns", 1, "TURN-3/4"),
            ("nib_moviola_pass_progression", 0, "PASS-A GESTURE"),
            ("nib_moviola_pass_progression", 1, "PASS-B VOLUME"),
            ("nib_moviola_keys", 0, "KEY K01"),
            ("nib_moviola_breakdowns", 0, "BRK B01 (ARC)"),
            ("nib_moviola_inbetweens", 0, "INB I01"),
            ("nib_moviola_keys", 1, "KEY K02"),
            ("nib_moviola_breakdowns", 1, "BRK B02 (ARC)"),
            ("nib_moviola_inbetweens", 1, "INB I02"),
            ("nib_moviola_keys", 2, "KEY K03"),
            ("nib_moviola_keys", 3, "KEY K04"),
            ("nib_moviola_keys_ext", 0, "EXT K05 LOUPE"),
            ("nib_moviola_keys_ext", 1, "EXT K06 SPLICE"),
            ("nib_moviola_keys_ext", 2, "EXT K07 MARK"),
            ("nib_moviola_keys_ext", 3, "EXT K08 LOCK"),
        ]
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, nib_moviola_chain, sub_step,
            cx=128, cy=by1 - 144, max_w=192, max_h=246,
            tint_bgr=STUDIO_AGENTS["nib"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        draw_code_hand_animated_agent(
            canvas, "blueline", cx=272, cy=by1 - 122, scale=0.80,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="moviola_crank", show_pass_ab=True, boil_phase=boil_phase,
        )
        mac_chain_s1 = [
            ("phase02_staging_thumbnails_keys", 0, "THUMB T01"),
            ("phase02_staging_thumbnails_keys", 1, "THUMB T02"),
            ("phase02_staging_thumbnails_keys", 2, "THUMB T03"),
            ("phase02_staging_thumbnails_keys", 3, "THUMB T04"),
            ("closeup_moviola_hands_keys", 0, "SPLICE M01"),
            ("closeup_moviola_hands_keys", 1, "SPLICE M02"),
            ("closeup_moviola_hands_keys", 2, "SPLICE M03"),
            ("closeup_moviola_hands_keys", 3, "SPLICE M04"),
        ]
        draw_macro_inset_box(
            canvas, mac_x, mac_y, mac_w, mac_h,
            sheet_name="phase02_staging_thumbnails_keys", sub_step=sub_step,
            tint_bgr=BGR_AMBER_PALE, boil_phase=boil_phase, frame_chain=mac_chain_s1,
        )
        draw_code_animated_macro_inset(canvas, mac_x, mac_y, mac_w, mac_h, 1, sub_step, shot_frame_idx, boil_phase=boil_phase)
        draw_4key_sequence_sprite(
            canvas, "apparatus_moviola_anim_keys", sub_step,
            cx=436, cy=by1 - 128, max_w=176, max_h=186,
            tint_bgr=BGR_BLUE_PALE, bob_phase=bob_rad * 0.4, boil_phase=boil_phase + 1,
        )
        draw_code_animated_studio_apparatus(canvas, 1, 436, by1 - 128, 176, 186, sub_step, shot_frame_idx, boil_phase=boil_phase)
        if sub_step > 0.20:
            reel_ang = sub_step * 720.0
            draw_spinning_wheel(canvas, 436 - 42, by1 - 182, 16, reel_ang)
            draw_spinning_wheel(canvas, 436 + 42, by1 - 182, 16, reel_ang + 40.0)

        draw_hand_inked_rect(canvas, rx0 - 6, ry0 - 6, rx0 + rw + 6, ry0 + rh + 6, fill_bgr=BGR_BLUE_PALE, outline_bgr=BGR_INK, thickness=3, boil_phase=boil_phase, shadow_offset=5)
        arch_fr = arch_sketches[shot_frame_idx % len(arch_sketches)]
        arch_resized = cv2.resize(arch_fr, (rw, rh), interpolation=cv2.INTER_AREA)
        enhance_hero_board_with_code_multiplane(arch_resized, slug, sub_step, shot_frame_idx, stage_mode="blue_only", boil_phase=boil_phase)
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = arch_resized
        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=None, outline_bgr=BGR_INK, thickness=4, boil_phase=boil_phase)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_AMBER_PALE, boil_phase=boil_phase)
        macro_caption = "THUMBNAILS T01..T04 & SPLICER"
        apparatus_caption = "NIB // ARCHIVAL MOVIOLA"
        right_board_caption = f"ARCHIVAL 35MM MOVIOLA SCREEN (4:3) — #{slug.upper()}"

    # =========================================================================
    # SCENE 2: ACT I-B — CHARACTER TURNAROUND MODEL SHEETS & 4-KEY STORYBOARD WALL (12+12 X 10 FRAMES!)
    # =========================================================================
    elif p_idx == 2:
        blueline_sb_chain = [
            ("model_sheet_blueline_turns", 0, "TURN-FRONT"),
            ("model_sheet_blueline_turns", 1, "TURN-3/4"),
            ("model_sheet_blueline_turns", 2, "TURN-PROFILE"),
            ("model_sheet_blueline_turns", 3, "TURN-SILHOUETTE"),
            ("blueline_storyboard_keys", 0, "KEY K01 SKETCH"),
            ("blueline_storyboard_keys", 1, "KEY K02 PIN"),
            ("blueline_storyboard_keys", 2, "KEY K03 ARCS"),
            ("blueline_storyboard_keys", 3, "KEY K04 LOCK"),
            ("agent_blueline_keys", 0, "EXT K05 POSE"),
            ("agent_blueline_keys", 1, "EXT K06 POSE"),
            ("agent_blueline_keys", 2, "EXT K07 POSE"),
            ("agent_blueline_keys", 3, "EXT K08 POSE"),
        ]
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, blueline_sb_chain, sub_step,
            cx=128, cy=by1 - 148, max_w=192, max_h=250,
            tint_bgr=STUDIO_AGENTS["blueline"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        draw_code_hand_animated_agent(
            canvas, "nib", cx=272, cy=by1 - 122, scale=0.80,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="storyboard_pin", show_pass_ab=True, boil_phase=boil_phase,
        )
        mac_turns_chain = [
            ("model_sheet_celinker_turns", 0, "CEL-TURN 1"),
            ("model_sheet_celinker_turns", 1, "CEL-TURN 2"),
            ("model_sheet_celinker_turns", 2, "CEL-TURN 3"),
            ("model_sheet_celinker_turns", 3, "CEL-TURN 4"),
            ("model_sheet_sprocket_turns", 0, "SPR-TURN 1"),
            ("model_sheet_sprocket_turns", 1, "SPR-TURN 2"),
            ("model_sheet_sprocket_turns", 2, "SPR-TURN 3"),
            ("model_sheet_sprocket_turns", 3, "SPR-TURN 4"),
            ("model_sheet_chime_turns", 0, "CHM-TURN 1"),
            ("model_sheet_chime_turns", 1, "CHM-TURN 2"),
            ("model_sheet_chime_turns", 2, "CHM-TURN 3"),
            ("model_sheet_chime_turns", 3, "CHM-TURN 4"),
        ]
        draw_macro_inset_box(
            canvas, mac_x, mac_y, mac_w, mac_h,
            sheet_name="model_sheet_celinker_turns", sub_step=sub_step,
            tint_bgr=BGR_BLUE_PALE, boil_phase=boil_phase, frame_chain=mac_turns_chain,
        )
        draw_code_animated_macro_inset(canvas, mac_x, mac_y, mac_w, mac_h, 2, sub_step, shot_frame_idx, boil_phase=boil_phase)
        draw_4key_sequence_sprite(
            canvas, "prop_storyboard_wall_anim_keys", sub_step,
            cx=436, cy=by1 - 130, max_w=178, max_h=186,
            tint_bgr=BGR_AMBER_PALE, bob_phase=bob_rad * 0.3, boil_phase=boil_phase + 1,
        )
        draw_code_animated_studio_apparatus(canvas, 2, 436, by1 - 130, 178, 186, sub_step, shot_frame_idx, boil_phase=boil_phase)

        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=BGR_AMBER_PALE, outline_bgr=BGR_INK, thickness=4, boil_phase=boil_phase, shadow_offset=5)
        cw_card, ch_card = 172, 129
        for pk_i in range(4):
            pr, pc = pk_i // 2, pk_i % 2
            cx_k = rx0 + 12 + pc * (cw_card + 16)
            cy_k = ry0 + 10 + pr * (ch_card + 10)
            card_prog = float(np.clip((sub_step - pk_i * 0.16) * 1.55, 0.10, 1.0))
            k_im = render_progressive_keyframe_board(slug, pk_i, card_prog, cw_card, ch_card, boil_phase=boil_phase + pk_i, stage_mode="blue_only")
            canvas[cy_k:cy_k + ch_card, cx_k:cx_k + cw_card] = k_im
            draw_hand_inked_rect(canvas, cx_k, cy_k, cx_k + cw_card, cy_k + ch_card, fill_bgr=None, outline_bgr=BGR_INK, thickness=2, wobble=0.8, boil_phase=boil_phase + pk_i)
            cv2.circle(canvas, (cx_k + cw_card // 2, cy_k + 5), 5, BGR_RED, -1, cv2.LINE_AA)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_BLUE_PALE, boil_phase=boil_phase)
        macro_caption = "360° TURNAROUND & MODEL SHEETS"
        apparatus_caption = "PIN-UP CORKBOARD (Q1..Q4)"
        right_board_caption = "4-CARD STORYBOARD WALL (EACH CARD EXACT 4:3)"

    # =========================================================================
    # SCENE 3: ACT II-A — LIGHTBOX KEYS, BREAKDOWNS (`B01..B04`), IN-BETWEENS (`I01..I04`) & ONION-SKIN FLIP (20 X 10 FRAMES!)
    # =========================================================================
    elif p_idx == 3:
        blueline_lb_chain = [
            ("blueline_lightbox_pass_progression", 0, "PASS-A GESTURE"),
            ("blueline_lightbox_pass_progression", 1, "PASS-B VOLUME"),
            ("blueline_lightbox_pass_progression", 2, "PASS-D CLEAN"),
            ("blueline_lightbox_pass_progression", 3, "PASS-E SHADE"),
            ("blueline_lightbox_keys", 0, "KEY K01 PEGS"),
            ("blueline_lightbox_breakdowns", 0, "BRK B01 (ARC)"),
            ("blueline_lightbox_inbetweens", 0, "INB I01 SLOW-IN"),
            ("blueline_lightbox_keys", 1, "KEY K02 ROUGH"),
            ("blueline_lightbox_breakdowns", 1, "BRK B02 (ARC)"),
            ("blueline_lightbox_inbetweens", 1, "INB I02 ROLL"),
            ("blueline_lightbox_keys", 2, "KEY K03 FLIP"),
            ("blueline_lightbox_breakdowns", 2, "BRK B03 (ARC)"),
            ("blueline_lightbox_inbetweens", 2, "INB I03CUSHION"),
            ("blueline_lightbox_keys", 3, "KEY K04 STACK"),
            ("blueline_lightbox_breakdowns", 3, "BRK B04 (ARC)"),
            ("blueline_lightbox_inbetweens", 3, "INB I04 SETTLE"),
            ("blueline_lightbox_keys_ext", 0, "EXT K05 DISC"),
            ("blueline_lightbox_keys_ext", 1, "EXT K06 3-FLIP"),
            ("blueline_lightbox_keys_ext", 2, "EXT K07 DIVIDERS"),
            ("blueline_lightbox_keys_ext", 3, "EXT K08 DOPE"),
        ]
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, blueline_lb_chain, sub_step,
            cx=128, cy=by1 - 150, max_w=194, max_h=252,
            tint_bgr=STUDIO_AGENTS["blueline"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        draw_code_hand_animated_agent(
            canvas, "blueline", cx=272, cy=by1 - 122, scale=0.80,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="paper_flip", show_pass_ab=True, boil_phase=boil_phase,
        )
        mac_lb_chain = [
            ("closeup_hand_drawing_keys", 0, "PEGBAR LOCK"),
            ("closeup_hand_drawing_keys", 1, "BLUE PENCIL"),
            ("blueline_lightbox_inbetweens", 0, "INB I01"),
            ("blueline_lightbox_inbetweens", 1, "INB I02"),
            ("blueline_lightbox_inbetweens", 2, "INB I03"),
            ("blueline_lightbox_inbetweens", 3, "INB I04"),
        ]
        draw_macro_inset_box(
            canvas, mac_x, mac_y, mac_w, mac_h,
            sheet_name="closeup_hand_drawing_keys", sub_step=sub_step,
            tint_bgr=BGR_BLUE_PALE, boil_phase=boil_phase, frame_chain=mac_lb_chain, onion_skin=True,
        )
        draw_code_animated_macro_inset(canvas, mac_x, mac_y, mac_w, mac_h, 3, sub_step, shot_frame_idx, boil_phase=boil_phase)
        draw_4key_sequence_sprite(
            canvas, "apparatus_lightbox_anim_keys", sub_step,
            cx=436, cy=by1 - 130, max_w=178, max_h=186,
            tint_bgr=BGR_AMBER_PALE, bob_phase=bob_rad * 0.35, boil_phase=boil_phase + 1,
        )
        draw_code_animated_studio_apparatus(canvas, 3, 436, by1 - 130, 178, 186, sub_step, shot_frame_idx, boil_phase=boil_phase)

        cv2.rectangle(canvas, (rx0 + 5, ry0 + 5), (rx0 + rw + 5, ry0 + rh + 5), BGR_INK, -1)
        board_im = render_progressive_keyframe_board(
            slug, 0, sub_step, rw, rh, boil_phase=boil_phase,
            stage_mode="blue_only", animate_4keys=True,
        )
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = board_im
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + 12), BGR_AMBER_PALE, -1)
        for px_off in (96, 192, 288):
            cv2.circle(canvas, (rx0 + px_off, ry0 + 6), 4, BGR_INK, -1, cv2.LINE_AA)
        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=None, outline_bgr=BGR_NONPHOTO_BLUE, thickness=4, boil_phase=boil_phase)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_BLUE_PALE, boil_phase=boil_phase)
        macro_caption = "3-SHEET FLIP & IN-BETWEENS I01..I04"
        apparatus_caption = "BACKLIT LIGHTBOX DISC (Q1..Q4)"
        right_board_caption = f"PASS A/B: ROUGH KEYS, BREAKDOWNS & IN-BETWEENS (4:3) — #{slug.upper()}"

    # =========================================================================
    # SCENE 4: ACT II-B — INSIDE-OUT PASS PROGRESSION & SPEEDBALL INDIA INKING (12 X 10 FRAMES!)
    # =========================================================================
    elif p_idx == 4:
        nib_ink_chain = [
            ("nib_moviola_pass_progression", 0, "PASS-A GESTURE"),
            ("nib_moviola_pass_progression", 1, "PASS-B VOLUME"),
            ("nib_moviola_pass_progression", 2, "PASS-D CLEANUP"),
            ("nib_moviola_pass_progression", 3, "PASS-E TONE"),
            ("nib_inking_keys", 0, "KEY K01 DIP NIB"),
            ("nib_inking_breakdowns", 0, "BRK B01 (ARC)"),
            ("nib_inking_keys", 1, "KEY K02 TAPER"),
            ("nib_inking_breakdowns", 1, "BRK B02 (ARC)"),
            ("nib_inking_keys", 2, "KEY K03 BLOT"),
            ("nib_inking_breakdowns", 2, "BRK B03 (ARC)"),
            ("nib_inking_keys", 3, "KEY K04 CEL INSPECT"),
            ("nib_inking_breakdowns", 3, "BRK B04 (ARC)"),
        ]
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, nib_ink_chain, sub_step,
            cx=128, cy=by1 - 150, max_w=194, max_h=252,
            tint_bgr=STUDIO_AGENTS["nib"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        draw_code_hand_animated_agent(
            canvas, "nib", cx=272, cy=by1 - 122, scale=0.80,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="draw_lightbox", show_pass_ab=True, boil_phase=boil_phase,
        )
        mac_ink_chain = [
            ("closeup_hand_drawing_keys", 1, "BLUE UNDERDRAW"),
            ("closeup_hand_drawing_keys", 2, "SPEEDBALL NIB"),
            ("nib_moviola_inbetweens", 0, "INB I01"),
            ("nib_moviola_inbetweens", 1, "INB I02"),
            ("nib_moviola_inbetweens", 2, "INB I03"),
            ("nib_moviola_inbetweens", 3, "INB I04"),
        ]
        draw_macro_inset_box(
            canvas, mac_x, mac_y, mac_w, mac_h,
            sheet_name="closeup_hand_drawing_keys", sub_step=sub_step,
            tint_bgr=BGR_RED_PALE, boil_phase=boil_phase, frame_chain=mac_ink_chain, onion_skin=True,
        )
        draw_code_animated_macro_inset(canvas, mac_x, mac_y, mac_w, mac_h, 4, sub_step, shot_frame_idx, boil_phase=boil_phase)
        draw_4key_sequence_sprite(
            canvas, "agent_nib_keys", sub_step,
            cx=436, cy=by1 - 130, max_w=172, max_h=182,
            tint_bgr=BGR_AMBER_PALE, key_sequence=(1, 0, 2, 3),
            bob_phase=bob_rad * 0.4, boil_phase=boil_phase + 1,
        )
        draw_code_animated_studio_apparatus(canvas, 4, 436, by1 - 130, 172, 182, sub_step, shot_frame_idx, boil_phase=boil_phase)

        cv2.rectangle(canvas, (rx0 + 5, ry0 + 5), (rx0 + rw + 5, ry0 + rh + 5), BGR_INK, -1)
        board_im = render_progressive_keyframe_board(
            slug, 1, sub_step, rw, rh, boil_phase=boil_phase,
            stage_mode="ink_only", animate_4keys=True,
        )
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = board_im
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + 12), BGR_AMBER_PALE, -1)
        for px_off in (96, 192, 288):
            cv2.circle(canvas, (rx0 + px_off, ry0 + 6), 4, BGR_INK, -1, cv2.LINE_AA)
        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=None, outline_bgr=BGR_INK, thickness=4, boil_phase=boil_phase)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_RED_PALE, boil_phase=boil_phase)
        macro_caption = "MACRO: SPEEDBALL NIB & IN-BETWEENS"
        apparatus_caption = "NIB // CLEAN-UP INKING DESK"
        right_board_caption = f"PASS D: SPEEDBALL INDIA INK CEL (4:3) — #{slug.upper()}"

    # =========================================================================
    # SCENE 5: ACT II-C — CEL-INKER PASS PROGRESSION, GOUACHE OPAQUING & BREAKDOWNS (16 X 10 FRAMES!)
    # =========================================================================
    elif p_idx == 5:
        celinker_paint_chain = [
            ("celinker_pass_progression", 0, "PASS-A GESTURE"),
            ("celinker_pass_progression", 1, "PASS-B VOLUME"),
            ("celinker_pass_progression", 2, "PASS-D CLEANUP"),
            ("celinker_pass_progression", 3, "PASS-E GOUACHE"),
            ("celinker_painting_keys", 0, "KEY K01 MIX"),
            ("celinker_painting_breakdowns", 0, "BRK B01 (ARC)"),
            ("celinker_painting_keys", 1, "KEY K02 FLOOD"),
            ("celinker_painting_breakdowns", 1, "BRK B02 (ARC)"),
            ("celinker_painting_keys", 2, "KEY K03 PEEL"),
            ("celinker_painting_breakdowns", 2, "BRK B03 (ARC)"),
            ("celinker_painting_keys", 3, "KEY K04 HANG"),
            ("celinker_painting_breakdowns", 3, "BRK B04 (ARC)"),
            ("celinker_painting_keys_ext", 0, "EXT K05 GRIND"),
            ("celinker_painting_keys_ext", 1, "EXT K06 SWATCH"),
            ("celinker_painting_keys_ext", 2, "EXT K07 BACK-PAINT"),
            ("celinker_painting_keys_ext", 3, "EXT K08 RACK LOCK"),
        ]
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, celinker_paint_chain, sub_step,
            cx=128, cy=by1 - 150, max_w=194, max_h=252,
            tint_bgr=STUDIO_AGENTS["celinker"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        draw_code_hand_animated_agent(
            canvas, "celinker", cx=272, cy=by1 - 122, scale=0.80,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="draw_lightbox", show_pass_ab=True, boil_phase=boil_phase,
        )
        mac_paint_chain = [
            ("closeup_hand_drawing_keys", 2, "INK CONTOUR"),
            ("closeup_hand_drawing_keys", 3, "GOUACHE BRUSH"),
            ("celinker_pass_progression", 0, "PASS-A"),
            ("celinker_pass_progression", 1, "PASS-B"),
            ("celinker_pass_progression", 2, "PASS-D"),
            ("celinker_pass_progression", 3, "PASS-E"),
        ]
        draw_macro_inset_box(
            canvas, mac_x, mac_y, mac_w, mac_h,
            sheet_name="closeup_hand_drawing_keys", sub_step=sub_step,
            tint_bgr=BGR_AMBER_PALE, boil_phase=boil_phase, frame_chain=mac_paint_chain,
        )
        draw_code_animated_macro_inset(canvas, mac_x, mac_y, mac_w, mac_h, 5, sub_step, shot_frame_idx, boil_phase=boil_phase)
        draw_4key_sequence_sprite(
            canvas, "prop_drying_rack_anim_keys", sub_step,
            cx=436, cy=by1 - 130, max_w=178, max_h=186,
            tint_bgr=BGR_GREEN_PALE, bob_phase=bob_rad * 0.35, boil_phase=boil_phase + 1,
        )
        draw_code_animated_studio_apparatus(canvas, 5, 436, by1 - 130, 178, 186, sub_step, shot_frame_idx, boil_phase=boil_phase)

        cv2.rectangle(canvas, (rx0 + 5, ry0 + 5), (rx0 + rw + 5, ry0 + rh + 5), BGR_INK, -1)
        board_im = render_progressive_keyframe_board(
            slug, 2, sub_step, rw, rh, boil_phase=boil_phase,
            stage_mode="gouache_only", animate_4keys=True,
        )
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = board_im
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + 12), BGR_AMBER_PALE, -1)
        for px_off in (96, 192, 288):
            cv2.circle(canvas, (rx0 + px_off, ry0 + 6), 4, BGR_INK, -1, cv2.LINE_AA)
        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=None, outline_bgr=BGR_AMBER, thickness=4, boil_phase=boil_phase)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_AMBER_PALE, boil_phase=boil_phase)
        macro_caption = "SABLE GOUACHE & PASS A→B→D→E"
        apparatus_caption = "OVERHEAD CEL DRYING RACK (Q1..Q4)"
        right_board_caption = f"PASS C/E: WATERTIGHT GOUACHE CEL (4:3) — #{slug.upper()}"

    # =========================================================================
    # SCENE 6: ACT III — 7-LEVEL MULTIPLANE CAMERA TOWER & PARALLAX PHYSICS (`PLANE 0..6`, 28 X 10 FRAMES!)
    # =========================================================================
    elif p_idx == 6:
        cel_mp_chain = [
            ("celinker_multiplane_keys", 0, "MP-K01 SLIDE P2"),
            ("celinker_multiplane_breakdowns", 0, "MP-B01 (ARC)"),
            ("celinker_multiplane_keys", 1, "MP-K02 LOCK P4"),
            ("celinker_multiplane_breakdowns", 1, "MP-B02 (ARC)"),
            ("celinker_multiplane_keys", 2, "MP-K03 LAMPS"),
            ("celinker_multiplane_breakdowns", 2, "MP-B03 (ARC)"),
            ("celinker_multiplane_keys", 3, "MP-K04 READY"),
            ("celinker_multiplane_breakdowns", 3, "MP-B04 (ARC)"),
            ("multiplane_crew_keys_ext", 0, "MP-K05 DIAL"),
            ("multiplane_crew_keys_ext", 1, "MP-K06WIPE"),
            ("multiplane_crew_keys_ext", 2, "MP-K07 FOCUS"),
            ("multiplane_crew_keys_ext", 3, "MP-K08 EXPOSE"),
        ]
        nib_crank_chain = [
            ("nib_cameracrank_keys", 0, "CRK-K01 GRIP"),
            ("nib_cameracrank_breakdowns", 0, "CRK-B01"),
            ("nib_cameracrank_keys", 1, "CRK-K02 TURN"),
            ("nib_cameracrank_breakdowns", 1, "CRK-B02"),
            ("nib_cameracrank_keys", 2, "CRK-K03 SHUTTER"),
            ("nib_cameracrank_breakdowns", 2, "CRK-B03"),
            ("nib_cameracrank_keys", 3, "CRK-K04 LOG"),
            ("nib_cameracrank_breakdowns", 3, "CRK-B04"),
        ]
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, cel_mp_chain, sub_step,
            cx=96, cy=by1 - 146, max_w=148, max_h=224,
            tint_bgr=STUDIO_AGENTS["celinker"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        draw_multi_sheet_chain_sprite(
            canvas, nib_crank_chain, sub_step,
            cx=212, cy=by1 - 146, max_w=148, max_h=224,
            tint_bgr=STUDIO_AGENTS["nib"]["pale_bgr"],
            bob_phase=bob_rad + 0.9, boil_phase=boil_phase + 1,
        )
        draw_code_hand_animated_agent(
            canvas, "blueline", cx=304, cy=by1 - 118, scale=0.68,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="multiplane_crank", show_pass_ab=True, boil_phase=boil_phase,
        )
        mac_mp_chain = [
            ("phase06_camera_parallax_diagram_keys", 0, "OPTICS CAM-1"),
            ("phase06_camera_parallax_diagram_keys", 1, "PARALLAX CAM-2"),
            ("phase06_camera_parallax_diagram_keys", 2, "OCCLUSION CAM-3"),
            ("phase06_camera_parallax_diagram_keys", 3, "DOLLY CAM-4"),
            ("closeup_multiplane_platen_keys", 0, "PLATEN PL-1"),
            ("closeup_multiplane_platen_keys", 1, "PLATEN PL-2"),
            ("closeup_multiplane_platen_keys", 2, "PLATEN PL-3"),
            ("closeup_multiplane_platen_keys", 3, "PLATEN PL-4"),
        ]
        draw_macro_inset_box(
            canvas, mac_x, mac_y, mac_w, mac_h,
            sheet_name="phase06_camera_parallax_diagram_keys", sub_step=sub_step,
            tint_bgr=BGR_AMBER_PALE, boil_phase=boil_phase, frame_chain=mac_mp_chain,
        )
        draw_code_animated_macro_inset(canvas, mac_x, mac_y, mac_w, mac_h, 6, sub_step, shot_frame_idx, boil_phase=boil_phase)
        app_mp_chain = [
            ("apparatus_multiplane_anim_keys", 0, "RIG K01"),
            ("apparatus_multiplane_breakdowns", 0, "RIG B01"),
            ("apparatus_multiplane_anim_keys", 1, "RIG K02"),
            ("apparatus_multiplane_breakdowns", 1, "RIG B02"),
            ("apparatus_multiplane_anim_keys", 2, "RIG K03"),
            ("apparatus_multiplane_breakdowns", 2, "RIG B03"),
            ("apparatus_multiplane_anim_keys", 3, "RIG K04"),
            ("apparatus_multiplane_breakdowns", 3, "RIG B04"),
        ]
        draw_multi_sheet_chain_sprite(
            canvas, app_mp_chain, sub_step,
            cx=436, cy=by1 - 134, max_w=182, max_h=196,
            tint_bgr=BGR_AMBER_PALE, bob_phase=bob_rad * 0.3, boil_phase=boil_phase + 2,
        )
        draw_code_animated_studio_apparatus(canvas, 6, 436, by1 - 134, 182, 196, sub_step, shot_frame_idx, boil_phase=boil_phase)

        # Giant 384x288 (4:3) Live Multiplane Camera Viewfinder on Right with Differential Parallax (`PLANE 6..0`)!
        draw_hand_inked_rect(canvas, rx0 - 6, ry0 - 6, rx0 + rw + 6, ry0 + rh + 6, fill_bgr=BGR_INK, outline_bgr=BGR_INK, thickness=3, boil_phase=boil_phase, shadow_offset=5)
        f_cur = shot_frame_idx % len(cart_frames)
        if sub_step < 0.36:
            vf_img = render_progressive_keyframe_board(
                slug, 1, sub_step / 0.35, rw, rh, boil_phase=boil_phase,
                stage_mode="gouache_only", animate_4keys=True,
            )
        else:
            vf_img = cv2.resize(cart_frames[f_cur], (rw, rh), interpolation=cv2.INTER_LINEAR)
            enhance_hero_board_with_code_multiplane(vf_img, slug, sub_step, shot_frame_idx, stage_mode="gouache_only", boil_phase=boil_phase)
        # Overlay live differential multiplane parallax planes (`P6 Sky`, `P4 Midground`, `P0 FG Mask`) inside the 4:3 viewfinder
        cam_vf = (sub_step - 0.5) * 36.0
        vf_over = vf_img.copy()
        draw_studio_doodle_sprite(vf_over, "multiplane_plane6_sky_atmosphere_keys", int(sub_step * 3.99) % 4, cx=rw // 2 + int(cam_vf / 6.0), cy=54, max_w=240, max_h=95, tint_bgr=BGR_BLUE_PALE, boil_phase=boil_phase)
        draw_studio_doodle_sprite(vf_over, "multiplane_plane0_fg_mask_keys", int(sub_step * 3.99) % 4, cx=42 + int(cam_vf / 0.45), cy=rh // 2, max_w=78, max_h=240, tint_bgr=BGR_PAPER_DARK, boil_phase=boil_phase)
        cv2.addWeighted(vf_over, 0.28, vf_img, 0.72, 0, vf_img)
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = vf_img
        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=None, outline_bgr=BGR_AMBER, thickness=4, boil_phase=boil_phase)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_AMBER_PALE, boil_phase=boil_phase)
        macro_caption = "CAMERA PARALLAX & PLATEN MICROMETER"
        apparatus_caption = "7-PLANE TOWER (KEYS & BREAKDOWNS)"
        right_board_caption = f"7-PLANE MULTIPLANE VIEWFINDER (P0..P6 PARALLAX) — #{slug.upper()}"

    # =========================================================================
    # SCENE 7: ACT IV-A — SPROCKET'S 35MM CALIPER AUDIT & CHIME'S FOLEY SOUNDSTAGE (20 X 10 FRAMES!)
    # =========================================================================
    elif p_idx == 7:
        sprocket_audit_chain = [
            ("sprocket_projector_keys", 0, "AUD-K01 THREAD"),
            ("sprocket_projector_breakdowns", 0, "AUD-B01 (ARC)"),
            ("sprocket_projector_keys", 1, "AUD-K02 CALIPER"),
            ("sprocket_projector_breakdowns", 1, "AUD-B02 (ARC)"),
            ("sprocket_projector_keys", 2, "AUD-K03 FOCUS"),
            ("sprocket_projector_breakdowns", 2, "AUD-B03 (ARC)"),
            ("sprocket_projector_keys", 3, "AUD-K04 16/16 PASS"),
            ("sprocket_projector_breakdowns", 3, "AUD-B04 (ARC)"),
        ]
        chime_foley_chain = [
            ("chime_foley_keys", 0, "FOL-K01 CLAPPER"),
            ("chime_foley_breakdowns", 0, "FOL-B01 (ARC)"),
            ("chime_foley_keys", 1, "FOL-K02 HOOVES"),
            ("chime_foley_breakdowns", 1, "FOL-B02 (ARC)"),
            ("chime_foley_keys", 2, "FOL-K03 WIND"),
            ("chime_foley_breakdowns", 2, "FOL-B03 (ARC)"),
            ("chime_foley_keys", 3, "FOL-K04 BELL"),
            ("chime_foley_breakdowns", 3, "FOL-B04 (ARC)"),
            ("chime_foley_keys_ext", 0, "FOL-K05 THUNDER"),
            ("chime_foley_keys_ext", 1, "FOL-K06 WHISTLE"),
            ("chime_foley_keys_ext", 2, "FOL-K07 GRAVEL"),
            ("chime_foley_keys_ext", 3, "FOL-K08 X-SHEET"),
        ]
        _, _, _, _ = draw_multi_sheet_chain_sprite(
            canvas, sprocket_audit_chain, sub_step,
            cx=96, cy=by1 - 146, max_w=148, max_h=224,
            tint_bgr=STUDIO_AGENTS["sprocket"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, chime_foley_chain, sub_step,
            cx=212, cy=by1 - 146, max_w=148, max_h=224,
            tint_bgr=STUDIO_AGENTS["chime"]["pale_bgr"], onion_skin=True,
            bob_phase=bob_rad + 1.1, boil_phase=boil_phase + 1,
        )
        draw_code_hand_animated_agent(
            canvas, "chime", cx=304, cy=by1 - 118, scale=0.68,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="foley_strike", show_pass_ab=True, boil_phase=boil_phase,
        )
        draw_macro_inset_box(
            canvas, mac_x, mac_y, mac_w, mac_h,
            sheet_name="closeup_projector_threading_keys", sub_step=sub_step,
            tint_bgr=BGR_GREEN_PALE, boil_phase=boil_phase,
        )
        draw_code_animated_macro_inset(canvas, mac_x, mac_y, mac_w, mac_h, 7, sub_step, shot_frame_idx, boil_phase=boil_phase)
        draw_4key_sequence_sprite(
            canvas, "prop_foley_table_anim_keys", sub_step,
            cx=436, cy=by1 - 130, max_w=178, max_h=186,
            tint_bgr=BGR_AMBER_PALE, bob_phase=bob_rad * 0.4, boil_phase=boil_phase + 2,
        )
        draw_code_animated_studio_apparatus(canvas, 7, 436, by1 - 130, 178, 186, sub_step, shot_frame_idx, boil_phase=boil_phase)

        draw_hand_inked_rect(canvas, rx0 - 6, ry0 - 6, rx0 + rw + 6, ry0 + rh + 6, fill_bgr=BGR_GREEN_PALE, outline_bgr=BGR_INK, thickness=3, boil_phase=boil_phase, shadow_offset=5)
        f_cur = shot_frame_idx % len(cart_frames)
        cel_im = cv2.resize(ink_cels[f_cur % len(ink_cels)], (rw, rh), interpolation=cv2.INTER_AREA)
        fin_im = cv2.resize(cart_frames[f_cur], (rw, rh), interpolation=cv2.INTER_LINEAR)
        split_x = int(round(rw * (0.25 + 0.55 * math.sin(sub_step * math.pi) ** 2)))
        comp_im = fin_im.copy()
        comp_im[:, :split_x] = cel_im[:, :split_x]
        cv2.line(comp_im, (split_x, 0), (split_x, rh), BGR_RED, 3)
        enhance_hero_board_with_code_multiplane(comp_im, slug, sub_step, shot_frame_idx, stage_mode="gouache_only", boil_phase=boil_phase)
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = comp_im
        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=None, outline_bgr=BGR_INK, thickness=4, boil_phase=boil_phase)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_GREEN_PALE, boil_phase=boil_phase)
        macro_caption = "MACRO: 35MM GATE & 4:3 CALIPER"
        apparatus_caption = "FOLEY SOUND RIG (Q1..Q4)"
        right_board_caption = "16/16 GATES AUDIT & FOLEY SYNC MONITOR (4:3)"

    # =========================================================================
    # SCENE 8: ACT IV-B — GRAND 35MM DAILIES PREMIERE & CREW CELEBRATION BREAKDOWNS (12 X 10 FRAMES!)
    # =========================================================================
    else:
        crew_premiere_chain = [
            ("studio_crew_celebration_keys", 0, "PREM-K01 AWE"),
            ("studio_crew_celebration_breakdowns", 0, "PREM-B01 (ARC)"),
            ("studio_crew_celebration_keys", 1, "PREM-K02 TOAST"),
            ("studio_crew_celebration_breakdowns", 1, "PREM-B02 (ARC)"),
            ("studio_crew_celebration_keys", 2, "PREM-K03 LEAP"),
            ("studio_crew_celebration_breakdowns", 2, "PREM-B03 (ARC)"),
            ("studio_crew_celebration_keys", 3, "PREM-K04 BOW"),
            ("studio_crew_celebration_breakdowns", 3, "PREM-B04 (ARC)"),
            ("agent_nib_keys", 3, "NIB ENCORE"),
            ("agent_blueline_keys", 3, "BLUELINE ENCORE"),
            ("agent_sprocket_keys", 3, "SPROCKET 16/16"),
            ("agent_chime_keys", 3, "CHIME FINALE"),
        ]
        active_pose_idx, active_code_lbl, chain_len, chain_cur = draw_multi_sheet_chain_sprite(
            canvas, crew_premiere_chain, sub_step,
            cx=134, cy=by1 - 148, max_w=204, max_h=252,
            tint_bgr=BGR_AMBER_PALE, onion_skin=False,
            bob_phase=bob_rad, boil_phase=boil_phase,
        )
        draw_code_hand_animated_agent(
            canvas, "sprocket", cx=276, cy=by1 - 122, scale=0.78,
            sub_step=sub_step, frame_idx=shot_frame_idx + boil_phase,
            action_mode="celebrate", show_pass_ab=True, boil_phase=boil_phase,
        )
        cur_idx = shot_frame_idx % len(cart_frames)
        arch_im = cv2.resize(arch_sketches[cur_idx % len(arch_sketches)], (mac_w, mac_h), interpolation=cv2.INTER_AREA)
        draw_hand_inked_rect(canvas, mac_x, mac_y, mac_x + mac_w, mac_y + mac_h, fill_bgr=BGR_BLUE_PALE, outline_bgr=BGR_INK, thickness=3, boil_phase=boil_phase, shadow_offset=4)
        canvas[mac_y:mac_y + mac_h, mac_x:mac_x + mac_w] = arch_im
        draw_hand_inked_rect(canvas, mac_x, mac_y, mac_x + mac_w, mac_y + mac_h, fill_bgr=None, outline_bgr=BGR_INK, thickness=3, boil_phase=boil_phase)

        draw_4key_sequence_sprite(
            canvas, "apparatus_projector_anim_keys", sub_step,
            cx=436, cy=by1 - 130, max_w=178, max_h=186,
            tint_bgr=BGR_BLUE_PALE, bob_phase=bob_rad * 0.4, boil_phase=boil_phase + 1,
        )
        reel_ang = sub_step * 900.0
        draw_spinning_wheel(canvas, 436 - 34, by1 - 188, 15, reel_ang)
        draw_spinning_wheel(canvas, 436 + 28, by1 - 188, 15, reel_ang + 45.0)

        draw_hand_inked_rect(canvas, rx0 - 10, ry0 - 8, rx0 + rw + 10, ry0 + rh + 8, fill_bgr=BGR_RED, outline_bgr=BGR_INK, thickness=4, boil_phase=boil_phase, shadow_offset=5)
        hero_im = cv2.resize(cart_frames[cur_idx], (rw, rh), interpolation=cv2.INTER_LINEAR)
        enhance_hero_board_with_code_multiplane(hero_im, slug, sub_step, shot_frame_idx, stage_mode="gouache_only", boil_phase=boil_phase)
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = hero_im
        draw_hand_inked_rect(canvas, rx0, ry0, rx0 + rw, ry0 + rh, fill_bgr=None, outline_bgr=BGR_INK, thickness=4, boil_phase=boil_phase)

        draw_studio_workbench(canvas, bx0 + 12, by1 - 48, 528, by1 - 4, fill_bgr=BGR_AMBER_PALE, boil_phase=boil_phase)
        macro_caption = "ARCHIVAL SYNC MONITOR (4:3)"
        apparatus_caption = "35MM BOOTH PROJECTOR (Q1..Q4)"
        right_board_caption = f"35MM DAILIES THEATRE SCREEN (4:3) — #{slug.upper()}"

    # 7-Plane Multiplane Foreground Occlusion (`PLANE 1 Foreground Props` & `PLANE 0 Camera Mask`)
    draw_7plane_multiplane_world(canvas, bx0, by0, bx1, by1, cap_h, floor_y, p_idx, sub_step, boil_phase=boil_phase, layer_pass="fg")

    # =========================================================================
    # PIL HIGH-RESOLUTION COMIC LETTERING, 4-STAGE TRACKER & LIVE X-SHEET
    # =========================================================================
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    draw.rectangle([bx0, by0, bx1, by0 + cap_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
    draw.rectangle([bx0, by0, bx0 + 52, by0 + cap_h], fill=scene_ag["color_rgb"], outline=(22, 20, 18), width=4)
    draw.text((bx0 + 17, by0 + 9), str(p_idx), font=FONT_DISPLAY_XL, fill=(255, 251, 244))
    hdr_txt = f"SCENE {p_idx}/8 ({act_title}) — {spec['title'].upper()}"
    f_hdr = fit_text_font(draw, hdr_txt, (bx1 - bx0) - 70, [FONT_DISPLAY_LG, FONT_DISPLAY_MD, FONT_CAPTION_LG, FONT_CAPTION_MD])
    draw.text((bx0 + 64, by0 + 14), hdr_txt, font=f_hdr, fill=(22, 20, 18))

    s_box = (bx0 + 12, by0 + cap_h + 12, 332, by0 + cap_h + 114)
    if p_idx == 1:
        bubble_txt = spec["speech_p1"].upper()
    elif p_idx in (2, 3):
        bubble_txt = spec["caption_p2"].upper()
    elif p_idx in (4, 5):
        bubble_txt = f"{scene_ag['name'].upper()} ANIMATE KEYS, BREAKDOWNS & IN-BETWEENS ({chain_len} DRAWINGS)!"
    elif p_idx == 6:
        bubble_txt = spec["caption_p3"].upper()
    else:
        bubble_txt = spec["caption_p4"].upper()

    draw.rounded_rectangle([s_box[0] + 3, s_box[1] + 3, s_box[2] + 3, s_box[3] + 3], radius=16, fill=(22, 20, 18))
    draw.rounded_rectangle(s_box, radius=16, fill=(255, 255, 255), outline=(22, 20, 18), width=3)
    lines = textwrap.wrap(bubble_txt, width=28)[:3]
    lh = 24
    sy_txt = s_box[1] + ((s_box[3] - s_box[1]) - len(lines) * lh) // 2 - 2
    for i, ln in enumerate(lines):
        f_sp = fit_text_font(draw, ln, (s_box[2] - s_box[0]) - 18, [FONT_SPEECH_MD, FONT_CAPTION_MD, FONT_CAPTION_SM])
        bb = draw.textbbox((0, 0), ln, font=f_sp)
        tw = bb[2] - bb[0]
        draw.text((s_box[0] + ((s_box[2] - s_box[0]) - tw) // 2, sy_txt + i * lh), ln, font=f_sp, fill=(22, 20, 18))

    draw.rounded_rectangle([mac_x, mac_y - 26, mac_x + mac_w, mac_y - 4], radius=4, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
    f_mc = fit_text_font(draw, macro_caption, mac_w - 10, [FONT_LABEL_SM, FONT_CAPTION_SM])
    draw.text((mac_x + 6, mac_y - 22), macro_caption, font=f_mc, fill=(22, 20, 18))

    draw.rounded_rectangle([rx0, ry0 - 28, rx0 + rw, ry0 - 4], radius=5, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
    f_rb = fit_text_font(draw, right_board_caption, rw - 14, [FONT_LABEL_SM, FONT_CAPTION_SM])
    draw.text((rx0 + 8, ry0 - 23), right_board_caption, font=f_rb, fill=(22, 20, 18))

    draw_4key_progress_strip(draw, rx0, ry0 + rh + 14, rw, active_pose_idx, pose_labels, scene_ag["color_rgb"])

    # Bottom-Right Live Exposure Sheet (`X-SHEET`) Bar (`y: 462..512`)
    tx0, ty0, tx1, ty1 = rx0, by1 - 46, rx0 + rw, by1 - 8
    draw.rounded_rectangle([tx0, ty0, tx1, ty1], radius=5, fill=(246, 236, 218), outline=(22, 20, 18), width=3)
    bar_w = (tx1 - tx0) - 20
    prog_px = int(round(sub_step * bar_w))
    draw.rectangle([tx0 + 10, ty0 + 10, tx0 + 10 + bar_w, ty1 - 10], fill=(222, 212, 196), outline=(22, 20, 18), width=2)
    if prog_px > 2:
        draw.rectangle([tx0 + 10, ty0 + 10, tx0 + 10 + prog_px, ty1 - 10], fill=scene_ag["color_rgb"], outline=(22, 20, 18), width=2)
    sub_frame_10 = (chain_cur * 10) + min(9, int(sub_step * 10.0)) + 1
    tot_frame_10 = chain_len * 10
    sc_lbl = f"X-SHEET · FRAME {sub_frame_10}/{tot_frame_10} (3,100 TOTAL) [{active_code_lbl}] · 24 FPS ON ONES"
    f_sc = fit_text_font(draw, sc_lbl, bar_w - 12, [FONT_LABEL_SM])
    draw.text((tx0 + 16, ty0 + 11), sc_lbl, font=f_sc, fill=(22, 20, 18))

    ag_pill = f"{scene_ag['name'].upper()} // {active_code_lbl}"
    draw.rounded_rectangle([bx0 + 20, by1 - 36, 326, by1 - 10], radius=5, fill=(255, 251, 244), outline=(22, 20, 18), width=2)
    f_ap = fit_text_font(draw, ag_pill, 292, [FONT_LABEL_SM, FONT_CAPTION_SM])
    draw.text((bx0 + 28, by1 - 31), ag_pill, font=f_ap, fill=(22, 20, 18))

    draw.rounded_rectangle([338, by1 - 36, 522, by1 - 10], radius=5, fill=(214, 228, 240), outline=(22, 20, 18), width=2)
    f_app = fit_text_font(draw, apparatus_caption, 172, [FONT_LABEL_SM, FONT_CAPTION_SM])
    draw.text((345, by1 - 31), apparatus_caption, font=f_app, fill=(22, 20, 18))

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_bgr, (bx0, by0), (bx1, by1), BGR_INK, 5)
    return out_bgr


def render_fullscreen_hero_poster(spec: dict, panel_num: int = None) -> np.ndarray:
    """
    Renders a spacious Single-Scene Full-Screen `1280x720` (16:9) Hero Poster so no poster
    ever crowds 4 rooms onto one image at the same time!
    """
    if panel_num is None:
        ag_to_scene = {"nib": 4, "blueline": 3, "celinker": 5, "sprocket": 7, "chime": 8}
        panel_num = ag_to_scene.get(spec.get("lead_agent", "nib"), 1)
    frame_960 = render_single_panel_fullscreen(spec, panel_num=panel_num, sub_step=0.72, shot_frame_idx=16, boil_phase=0)
    return cv2.resize(frame_960, (1280, 720), interpolation=cv2.INTER_LINEAR)


def assemble_8scene_storyboard_sheet(spec: dict) -> np.ndarray:
    """
    Assembles all 8 Sequential Studio Scenes (`Scene 1..8`) into a clean `2x4` overview sheet
    for documentation & walkthrough inspection.
    """
    rows = []
    for r_i in range(4):
        p_left = render_single_panel_fullscreen(spec, r_i * 2 + 1, sub_step=0.72, shot_frame_idx=16, boil_phase=r_i)
        p_right = render_single_panel_fullscreen(spec, r_i * 2 + 2, sub_step=0.72, shot_frame_idx=16, boil_phase=r_i + 1)
        rows.append(np.hstack([p_left, p_right]))
    grid = np.vstack(rows)
    return cv2.resize(grid, (1600, 1800), interpolation=cv2.INTER_AREA)


def synthesize_studio_soundtrack(wav_path: Path, duration_sec: float, seed: int = 42):
    sr = 22050
    n_samples = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    rng = np.random.RandomState(seed)

    audio = (
        0.04 * np.sin(2 * np.pi * 220.0 * t)
        + 0.03 * np.sin(2 * np.pi * 277.18 * t)
        + 0.025 * np.sin(2 * np.pi * 329.63 * t)
    )
    for beat_s in np.arange(0.05, duration_sec, 4.0):
        idx = int(beat_s * sr)
        blen = min(int(0.12 * sr), n_samples - idx)
        if blen > 0:
            env = np.exp(-np.linspace(0, 6, blen))
            audio[idx:idx + blen] += 0.16 * env * rng.normal(0, 1, blen)

    audio = np.clip(audio, -0.95, 0.95)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# =============================================================================
# 5 STANDALONE FULL-SCREEN DEPARTMENT / AGENT CARTOON FILMS
# =============================================================================
STANDALONE_AGENT_CARTOON_SPECS = [
    {
        "id": "cartoon-01-nib-moviola",
        "num": 101,
        "title": "Nib & Blue-Line: The Archival Moviola & 4-Key Storyboard Wall",
        "lead_agent": "nib",
        "cameo_agent": "blueline",
        "featured_shot": "giraffe-walks-past",
        "scenes": [1, 2],
        "poster_scene": 1,
        "speech_p1": "Hold the 35mm reel! Watch me thread the archival print and mark every extreme on the X-sheet!",
        "caption_p2": "Blue-Line sketches all 4 key poses in non-photo blue and pins each 4:3 index card to the storyboard wall.",
        "caption_p3": "Multiplane camera check.",
        "caption_p4": "Dailies screening check.",
    },
    {
        "id": "cartoon-02-blueline-rough-lightbox",
        "num": 102,
        "title": "Blue-Line at the Lightbox: Non-Photo Blue Rough Construction",
        "lead_agent": "blueline",
        "cameo_agent": "nib",
        "featured_shot": "lighthouse",
        "scenes": [2, 3],
        "poster_scene": 3,
        "speech_p1": "Every great cartoon starts on the backlit glass disc with sweeping non-photo blue lines of action!",
        "caption_p2": "Blue-Line locks the punched animation paper onto the brass Acme peg bar and flips between Key 1 and Key 2.",
        "caption_p3": "Multiplane camera check.",
        "caption_p4": "Dailies screening check.",
    },
    {
        "id": "cartoon-03-nib-celinker-ink-gouache",
        "num": 103,
        "title": "Nib & Cel-Inker: Speedball India Ink & Gouache Cel Opaquing",
        "lead_agent": "nib",
        "cameo_agent": "celinker",
        "featured_shot": "campfire",
        "scenes": [4, 5],
        "poster_scene": 4,
        "speech_p1": "Dip the Speedball nib in India ink! Pull one continuous tapered line right over the blue construction rough!",
        "caption_p2": "Cel-Inker flips the acetate cel and flood-fills opaque gouache pigment before clipping it to the drying line.",
        "caption_p3": "Multiplane camera check.",
        "caption_p4": "Dailies screening check.",
    },
    {
        "id": "cartoon-04-multiplane-camera-tower",
        "num": 104,
        "title": "Cel-Inker & Nib: The 7-Level Multiplane Camera Tower",
        "lead_agent": "celinker",
        "cameo_agent": "nib",
        "featured_shot": "train",
        "scenes": [5, 6],
        "poster_scene": 6,
        "speech_p1": "Slide the inked hero cel onto Plane P2 and dial the brass micrometer! Watch the 4:3 viewfinder lock in depth!",
        "caption_p2": "Cel-Inker loads the glass platens while Nib cranks the overhead camera wheel one 35mm exposure at a time.",
        "caption_p3": "All 7 gouache planes locked in live multiplane parallax inside the 4:3 camera viewfinder!",
        "caption_p4": "Dailies screening check.",
    },
    {
        "id": "cartoon-05-sprocket-chime-screening",
        "num": 105,
        "title": "Sprocket & Chime: 35mm Caliper Audit, Foley & Grand Premiere",
        "lead_agent": "sprocket",
        "cameo_agent": "chime",
        "featured_shot": "butterfly",
        "scenes": [7, 8],
        "poster_scene": 8,
        "speech_p1": "Brass caliper reads exact 4:3! 16 out of 16 gates pass! Hit the coconut hooves and roll the 35mm projector!",
        "caption_p2": "Sprocket audits the aperture gate while Chime syncs the clapperboard, brass bell, and wind machine.",
        "caption_p3": "Multiplane camera check.",
        "caption_p4": "The entire 5-agent crew cheers in the screening theatre as the finished 4:3 cartoon plays on the silver screen!",
    },
]


def render_standalone_agent_cartoons() -> list:
    """
    Renders the 5 Standalone Full-Screen Department/Agent Cartoons (`cartoon-01..05`),
    giving each 2-scene department film `20.0 seconds` (`480 frames @ 24fps`, `240 frames = 10.0s` per scene!)
    so every turnaround, pass progression, keyframe (`K01..K08`), breakdown (`B01..B04`),
    in-between (`I01..I04`), and 10x code-animated sub-frame gets ample unhurried screen time!
    """
    results = []
    fps = 24
    frames_per_scene = 240  # 10.0 full seconds per scene @ 24fps On Ones = 480 frames per standalone cartoon!
    W, H = 960, 540

    for c_spec in STANDALONE_AGENT_CARTOON_SPECS:
        cid = c_spec["id"]
        mp4_path = BTS_DIR / f"{cid}.mp4"
        gif_path = BTS_DIR / f"{cid}.gif"
        poster_path = BTS_DIR / f"{cid}_poster.png"
        raw_mp4 = BTS_DIR / f"_{cid}_raw.mp4"
        wav_path = BTS_DIR / f"_{cid}.wav"

        scenes = c_spec["scenes"]
        n_frames = frames_per_scene * len(scenes)

        poster_bgr = render_fullscreen_hero_poster(c_spec, panel_num=c_spec["poster_scene"])
        cv2.imwrite(str(poster_path), poster_bgr)
        shutil.copy2(poster_path, ARTIFACT_DIR / f"{cid}_poster.png")

        vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        gif_pil_frames = []

        for f_i in range(n_frames):
            boil_phase = f_i // 4
            sc_idx = min(len(scenes) - 1, f_i // frames_per_scene)
            p_num = scenes[sc_idx]
            local_f = f_i - sc_idx * frames_per_scene
            sub = local_f / float(frames_per_scene - 1)
            shot_f = int(round(sub * 32))

            frame_bgr = render_single_panel_fullscreen(
                c_spec,
                panel_num=p_num,
                sub_step=sub,
                shot_frame_idx=shot_f,
                boil_phase=boil_phase,
            )
            vw.write(frame_bgr)
            if f_i % 6 == 0:
                rgb_small = cv2.cvtColor(cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                gif_pil_frames.append(Image.fromarray(rgb_small))

        vw.release()
        gif_pil_frames[0].save(
            str(gif_path),
            save_all=True,
            append_images=gif_pil_frames[1:],
            duration=115,
            loop=0,
            optimize=True,
        )
        shutil.copy2(gif_path, ARTIFACT_DIR / f"{cid}.gif")

        synthesize_studio_soundtrack(wav_path, duration_sec=n_frames / fps, seed=c_spec["num"] * 97)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(raw_mp4),
                "-i", str(wav_path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "96k", "-shortest",
                str(mp4_path),
            ],
            check=True,
        )
        raw_mp4.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

        out_rec = dict(c_spec)
        out_rec["mp4"] = f"bts_films/{cid}.mp4"
        out_rec["gif"] = f"bts_films/{cid}.gif"
        out_rec["poster"] = f"bts_films/{cid}_poster.png"
        results.append(out_rec)
        print(f"   [Standalone Cartoon {len(results)}/5] {cid} -> {out_rec['mp4']} (20.0s, 480 frames @ 24fps)")

    return results


def render_single_bts_film(spec: dict) -> dict:
    """
    Renders a 20.0-second (480 frames @ 24fps On Ones) Sequential 8-Scene Full-Screen Studio Film!
    Each of the 8 Studio Scenes gets 60 frames (2.5 seconds @ 24fps) at full `960x540` resolution
    so the viewer watches one stage at a time (`Scene 1 -> Scene 2 -> ... -> Scene 8`)
    with 10x code-animated frame density!
    """
    fid = spec["id"]
    mp4_path = BTS_DIR / f"{fid}.mp4"
    gif_path = BTS_DIR / f"{fid}.gif"
    poster_path = BTS_DIR / f"{fid}_poster.png"
    raw_mp4 = BTS_DIR / f"_{fid}_raw.mp4"
    wav_path = BTS_DIR / f"_{fid}.wav"

    fps = 24
    frames_per_scene = 60  # 60 frames per full-screen scene * 8 scenes = 480 frames @ 24fps!
    n_frames = frames_per_scene * 8
    W, H = 960, 540

    hero_scene = ((spec["num"] - 1) % 8) + 1
    poster_bgr = render_fullscreen_hero_poster(spec, panel_num=hero_scene)
    cv2.imwrite(str(poster_path), poster_bgr)

    if fid == "office-01-agent-office":
        sb8_bgr = assemble_8scene_storyboard_sheet(spec)
        sb8_path = BTS_DIR / "studio_8scenes_overview_atlas.png"
        cv2.imwrite(str(sb8_path), sb8_bgr)
        shutil.copy2(sb8_path, ARTIFACT_DIR / "studio_8scenes_overview_atlas.png")

    vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gif_pil_frames = []

    for f_i in range(n_frames):
        boil_phase = f_i // 4
        p_num = (f_i // frames_per_scene) + 1
        local_f = f_i % frames_per_scene
        sub = local_f / float(frames_per_scene - 1)
        shot_f = int(round(sub * 32))

        frame_bgr = render_single_panel_fullscreen(
            spec,
            panel_num=p_num,
            sub_step=sub,
            shot_frame_idx=shot_f,
            boil_phase=boil_phase,
        )
        vw.write(frame_bgr)
        if f_i % 5 == 0:
            rgb_small = cv2.cvtColor(cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            gif_pil_frames.append(Image.fromarray(rgb_small))

    vw.release()

    gif_pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=gif_pil_frames[1:],
        duration=110,
        loop=0,
        optimize=True,
    )

    synthesize_studio_soundtrack(wav_path, duration_sec=n_frames / fps, seed=spec["num"] * 101)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw_mp4),
            "-i", str(wav_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "96k", "-shortest",
            str(mp4_path),
        ],
        check=True,
    )
    raw_mp4.unlink(missing_ok=True)
    wav_path.unlink(missing_ok=True)

    out_rec = dict(spec)
    out_rec["mp4"] = f"bts_films/{fid}.mp4"
    out_rec["gif"] = f"bts_films/{fid}.gif"
    out_rec["poster"] = f"bts_films/{fid}_poster.png"
    return out_rec


def generate_studio_wing_comic_pages():
    """
    Renders 5 spacious Full-Screen `1280x720` Single-Stage Studio Soundstage Hero Plates
    (no crowded 2x2 split-screens!).
    """
    wing_specs = [
        ("studio_bg_1_multifloor_soundstage.jpg", AGENT_OFFICE_SPECS[0], 1),
        ("studio_bg_2_drafting_bullpen.jpg", AGENT_OFFICE_SPECS[2], 3),
        ("studio_bg_3_foley_screening_stage.jpg", VIDEO_SKILL_SPECS[7], 8),
        ("studio_bg_4_clocktower_optical_lab.jpg", VIDEO_SKILL_SPECS[3], 6),
        ("studio_bg_5_bullpen_unoccupied.jpg", AGENT_OFFICE_SPECS[1], 4),
    ]
    for fname, sp, sc_num in wing_specs:
        page_bgr = render_fullscreen_hero_poster(sp, panel_num=sc_num)
        out_p = BTS_DIR / fname
        cv2.imwrite(str(out_p), page_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        shutil.copy2(out_p, ARTIFACT_DIR / fname)


def render_contact_sheet(records: list, title: str, subtitle: str, out_name: str) -> Path:
    n = len(records)
    cols = 2
    rows = int(math.ceil(n / cols))
    cw, ch = 640, 360
    pad = 16
    header_h = 68
    W = pad + cols * (cw + pad)
    H = header_h + pad + rows * (ch + pad)

    canvas = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([0, 0, W, header_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
    f_t = fit_text_font(draw, title.upper(), W - 36, [FONT_DISPLAY_LG, FONT_DISPLAY_MD])
    draw.text((18, 10), title.upper(), font=f_t, fill=(22, 20, 18))
    draw.text((18, 40), subtitle.upper(), font=FONT_CAPTION_MD, fill=(60, 55, 50))
    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    for i, r in enumerate(records):
        ri = i // cols
        ci = i % cols
        x0 = pad + ci * (cw + pad)
        y0 = header_h + pad + ri * (ch + pad)
        p_img = cv2.imread(str(ROOT / r["poster"]))
        if p_img is not None:
            thumb = cv2.resize(p_img, (cw, ch), interpolation=cv2.INTER_AREA)
            cv2.rectangle(canvas, (x0 + 4, y0 + 4), (x0 + cw + 4, y0 + ch + 4), BGR_INK, -1)
            canvas[y0:y0 + ch, x0:x0 + cw] = thumb
            cv2.rectangle(canvas, (x0, y0), (x0 + cw, y0 + ch), BGR_INK, 3)

    out_path = PREVIEW_DIR / out_name
    cv2.imwrite(str(out_path), canvas)
    shutil.copy2(out_path, ARTIFACT_DIR / out_name)
    return out_path


def main():
    print("0. Generating 8 Multi-Frame Flipbook Sheets + Master 3,100-Frame Code-Animated Contact Atlas...")
    fb_paths = generate_3100_frame_flipbook_atlases(
        render_single_panel_fullscreen, AGENT_OFFICE_SPECS[0], PREVIEW_DIR, ARTIFACT_DIR
    )
    print(f"   Rendered {len(fb_paths)} flipbook sheets (including studio_3100_hand_animated_code_atlas.png)")

    print("1. Rendering 5 Full-Screen Studio Soundstage Hero Plates...")
    generate_studio_wing_comic_pages()

    print("2. Rendering 5 Standalone Full-Screen Department/Agent Cartoons (20.0s, 480 frames @ 24fps each)...")
    standalone_results = render_standalone_agent_cartoons()
    p_std = render_contact_sheet(
        standalone_results,
        "5 STANDALONE AGENT ANIMATOR DEPARTMENT CARTOONS (3,100 CODE-ANIMATED FRAMES: 2,400 STUDIO + 700 HERO @ 24 FPS)",
        "18-Joint 2-Bone IK Rig · Calligraphic Speedball Splines · 3-Sheet Rolling Paper Flip · 10x Curved-Arc In-Betweens · 7-Plane Parallax",
        "bts_5_standalone_agent_cartoons_atlas.png",
    )
    print("   Rendered Standalone Cartoons Atlas:", p_std.name)

    office_results = []
    skill_results = []

    print("3. Rendering 10 Virtual Agent Office 8-Scene Sequential Full-Screen Films (480 frames @ 24fps each)...")
    for sp in AGENT_OFFICE_SPECS:
        res = render_single_bts_film(sp)
        office_results.append(res)
        print(f"   [Office {sp['num']:02d}/10] {sp['id']} -> {res['mp4']}")

    print("4. Rendering 8 AI Video Director Skill 8-Scene Sequential Full-Screen Films (480 frames @ 24fps each)...")
    for sp in VIDEO_SKILL_SPECS:
        res = render_single_bts_film(sp)
        skill_results.append(res)
        print(f"   [Skill  {sp['num']-10:02d}/08] {sp['id']} -> {res['mp4']}")

    p_off = render_contact_sheet(
        office_results,
        "PART I: 10 AGENT ANIMATOR 8-SCENE SEQUENTIAL STUDIO FILMS (3,100 HAND-ANIMATED CODE FRAMES @ 24 FPS ON ONES)",
        "Act I Thumbnails, Moviola & Turnarounds -> Act II Keys, Breakdowns, In-Betweens & Ink -> Act III 7-Plane Tower -> Act IV Audit & Premiere",
        "bts_10_agent_offices_atlas.png",
    )
    p_skl = render_contact_sheet(
        skill_results,
        "PART II: 8 AGENT ANIMATOR CRAFT & MOTION SKILL 8-SCENE STUDIO FILMS (3,100 HAND-ANIMATED CODE FRAMES @ 24 FPS)",
        "18-Joint 2-Bone IK Rig · 10x Curved-Arc In-Betweens · 3-Sheet Rolling Paper Flip · Differential Camera Parallax (P0..P6) · Live X-Sheet",
        "bts_8_video_skills_atlas.png",
    )
    print("5. Rendered Contact Sheets:", p_off.name, p_skl.name)

    for sample_id in ["office-01-agent-office", "office-02-agent-007", "skill-01-huashu-motion", "skill-04-reanimate", "skill-08-earshot"]:
        shutil.copy2(BTS_DIR / f"{sample_id}.gif", ARTIFACT_DIR / f"{sample_id}.gif")
        shutil.copy2(BTS_DIR / f"{sample_id}_poster.png", ARTIFACT_DIR / f"{sample_id}_poster.png")

    concat_txt = BTS_DIR / "_master_concat.txt"
    all_mp4s = [ROOT / r["mp4"] for r in (standalone_results + office_results + skill_results)]
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in all_mp4s) + "\n")
    master_mp4 = ROOT / "Behind_The_Scenes_Multiplane_Studio_Film.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "29",
            "-c:a", "copy", str(master_mp4),
        ],
        check=True,
    )
    concat_txt.unlink(missing_ok=True)
    print(f"6. Assembled Master Film: {master_mp4.name} ({master_mp4.stat().st_size / 1e6:.2f} MB)")

    payload = {
        "studio_title": "Secondhand Studios — The Autonomous Agent Animation Bullpen (10x Code-Animated 3,100-Frame Multiplane Edition)",
        "total_hand_drawn_studio_keyframes": 240,
        "total_hero_shot_drawings": 70,
        "total_vector_traced_frames": 310,
        "total_10x_code_animated_studio_frames": 2400,
        "total_10x_code_animated_hero_frames": 700,
        "total_hand_animated_code_frames": 3100,
        "fps": 24,
        "total_studio_sheets": 60,
        "agents": [
            {"id": k, "name": v["name"], "role": v["role"], "dept": v["dept"]}
            for k, v in STUDIO_AGENTS.items()
        ],
        "standalone_agent_cartoons": standalone_results,
        "office_films": office_results,
        "skill_films": skill_results,
        "master_film": "Behind_The_Scenes_Multiplane_Studio_Film.mp4",
    }
    (DATA_DIR / "bts_studio_films.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "bts_studio_data.js").write_text("window.BTS_STUDIO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
