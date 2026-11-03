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
    Loads 33 archival pencil sketches, 33 blue-pencil roughs, 33 black-ink line cels,
    and 33 full gouache multiplane frames at their EXACT 4:3 aspect ratio (`480x360`),
    with old baked-in top/bottom HUD bars cleanly removed!
    """
    if slug in _SHOT_FRAMES_CACHE:
        return _SHOT_FRAMES_CACHE[slug]
    comp_path = SHOTS_DIR / slug / "comparison.mp4"
    arch_sketches, blue_roughs, ink_cels, cart_frames = [], [], [], []
    if comp_path.exists():
        cap = cv2.VideoCapture(str(comp_path))
        while len(cart_frames) < 33:
            ok, fr = cap.read()
            if not ok:
                break
            fh, fw = fr.shape[:2]  # 480 x 1280
            half_w = fw // 2       # 640
            y_top = 30
            y_bot = fh - 26        # 454 (height = 424)
            crop_h = y_bot - y_top
            crop_w = int(round(crop_h * 4.0 / 3.0))  # 565 (exact 4:3!)
            x_pad = max(0, (half_w - crop_w) // 2)   # 37

            crop_a = fr[y_top:y_bot, x_pad:x_pad + crop_w]
            crop_c = fr[y_top:y_bot, half_w + x_pad:half_w + x_pad + x_pad + crop_w - x_pad]
            raw_a = cv2.resize(crop_a, (480, 360), interpolation=cv2.INTER_AREA)
            raw_c = cv2.resize(crop_c, (480, 360), interpolation=cv2.INTER_AREA)

            # 1. Archival Graphite Sketch on Cream Paper
            gray_a = cv2.cvtColor(raw_a, cv2.COLOR_BGR2GRAY)
            blur_a = cv2.GaussianBlur(gray_a, (3, 3), 0)
            edges_a = cv2.Canny(blur_a, 45, 125)
            adapt_a = cv2.adaptiveThreshold(blur_a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
            ink_a = cv2.bitwise_or(edges_a, adapt_a)
            sk = np.full((360, 480, 3), BGR_PAPER, dtype=np.uint8)
            sk[blur_a < 115] = BGR_PAPER_DARK
            sk[ink_a > 0] = BGR_INK
            arch_sketches.append(sk)

            # 2. Animator's Non-Photo Blue Rough Construction Sheet (from cartoon edges)
            gray_c = cv2.cvtColor(raw_c, cv2.COLOR_BGR2GRAY)
            edges_c = cv2.Canny(gray_c, 40, 120)
            dil_c = cv2.dilate(edges_c, np.ones((2, 2), np.uint8), iterations=1)
            rough = np.full((360, 480, 3), BGR_GUTTER, dtype=np.uint8)
            # Subtle golden ratio construction lines on rough sheet
            cv2.line(rough, (0, 222), (479, 222), (235, 215, 190), 1)
            cv2.line(rough, (183, 0), (183, 359), (235, 215, 190), 1)
            cv2.line(rough, (296, 0), (296, 359), (235, 215, 190), 1)
            rough[dil_c > 0] = BGR_NONPHOTO_BLUE
            blue_roughs.append(rough)

            # 3. Clean Black Speedball India Ink Line Cel (before gouache color fill)
            ink_cel = np.full((360, 480, 3), BGR_GUTTER, dtype=np.uint8)
            ink_cel[gray_c < 90] = BGR_INK
            ink_cel[dil_c > 0] = BGR_INK
            ink_cels.append(ink_cel)

            # 4. Finished 7-Plane Gouache Cartoon Frame
            cart_frames.append(raw_c)
        cap.release()
    if not cart_frames:
        blank = np.full((360, 480, 3), BGR_PAPER, dtype=np.uint8)
        draw_qd_doodle(blank, slug, 240, 180, 200, 200, fill_bgr=BGR_AMBER_PALE, thickness=4)
        arch_sketches = [blank.copy() for _ in range(33)]
        blue_roughs = [blank.copy() for _ in range(33)]
        ink_cels = [blank.copy() for _ in range(33)]
        cart_frames = [blank.copy() for _ in range(33)]
    while len(cart_frames) < 33:
        arch_sketches.append(arch_sketches[-1])
        blue_roughs.append(blue_roughs[-1])
        ink_cels.append(ink_cels[-1])
        cart_frames.append(cart_frames[-1])
    _SHOT_FRAMES_CACHE[slug] = (arch_sketches, blue_roughs, ink_cels, cart_frames)
    return arch_sketches, blue_roughs, ink_cels, cart_frames


# =============================================================================
# HAND-DRAWN QUICK, DRAW! DOODLE SPRITE ENGINE FOR AGENTS & STUDIO APPARATUS
# =============================================================================
DOODLE_ASSETS_DIR = ROOT / "assets" / "studio_doodles"
_DOODLE_QUAD_CACHE = {}


def load_doodle_quadrant(sheet_name: str, quad_idx: int) -> tuple:
    """
    Loads one of the 4 quadrants (`0: Top-Left`, `1: Top-Right`, `2: Bottom-Left`,
    `3: Bottom-Right`) from `assets/studio_doodles/<sheet_name>.jpg`, crops tightly
    to its ink bounding box, and computes the flood-filled opaque cel silhouette mask
    so background lines never bleed through the hand-drawn character or machine.
    """
    key = (sheet_name, int(quad_idx) % 4)
    if key in _DOODLE_QUAD_CACHE:
        return _DOODLE_QUAD_CACHE[key]

    p = DOODLE_ASSETS_DIR / f"{sheet_name}.jpg"
    if not p.exists():
        blank_g = np.full((120, 120), 255, dtype=np.uint8)
        blank_m = np.zeros((120, 120), dtype=np.uint8)
        _DOODLE_QUAD_CACHE[key] = (blank_g, blank_m)
        return blank_g, blank_m

    gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape[:2]
    r = (int(quad_idx) % 4) // 2
    c = (int(quad_idx) % 4) % 2
    pad_seam = 10
    y0 = r * (H // 2) + (pad_seam if r == 1 else 6)
    y1 = (r + 1) * (H // 2) - (pad_seam if r == 0 else 6)
    x0 = c * (W // 2) + (pad_seam if c == 1 else 6)
    x1 = (c + 1) * (W // 2) - (pad_seam if c == 0 else 6)

    sub = gray[y0:y1, x0:x1]
    ys, xs = np.where(sub < 200)
    if len(xs) > 10:
        bx0 = max(0, int(xs.min()) - 10)
        by0 = max(0, int(ys.min()) - 10)
        bx1 = min(sub.shape[1], int(xs.max()) + 11)
        by1 = min(sub.shape[0], int(ys.max()) + 11)
        q_gray = sub[by0:by1, bx0:bx1].copy()
    else:
        q_gray = sub.copy()

    # Flood-fill from outer border on a binary threshold (`q_gray > 210`) to find exterior background
    qh, qw = q_gray.shape[:2]
    # Close tiny 1-2px gaps in outer ink contour before flood-filling exterior
    ink_bin = (q_gray < 210).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink_closed = cv2.morphologyEx(ink_bin, cv2.MORPH_CLOSE, kernel)
    ff_img = (ink_closed == 0).astype(np.uint8) * 255
    mask = np.zeros((qh + 2, qw + 2), dtype=np.uint8)
    for sx in range(0, qw, 4):
        if ff_img[0, sx] == 255:
            cv2.floodFill(ff_img, mask, (sx, 0), 128)
        if ff_img[qh - 1, sx] == 255:
            cv2.floodFill(ff_img, mask, (sx, qh - 1), 128)
    for sy in range(0, qh, 4):
        if ff_img[sy, 0] == 255:
            cv2.floodFill(ff_img, mask, (0, sy), 128)
        if ff_img[sy, qw - 1] == 255:
            cv2.floodFill(ff_img, mask, (qw - 1, sy), 128)

    # Interior opaque cel mask is everything NOT reached by exterior flood-fill (128)
    cel_mask = (ff_img != 128).astype(np.uint8) * 255
    _DOODLE_QUAD_CACHE[key] = (q_gray, cel_mask)
    return q_gray, cel_mask


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
):
    """
    Renders a high-craft hand-drawn Quick, Draw! doodle quadrant (`sheet_name`, `quad_idx`)
    centered at `(cx, cy)` inside `(max_w, max_h)` with strict aspect-ratio lock,
    opaque warm gouache cel fill, and anti-aliased Speedball India ink line strokes.
    """
    q_gray, cel_mask = load_doodle_quadrant(sheet_name, quad_idx)
    if flip_x:
        q_gray = cv2.flip(q_gray, 1)
        cel_mask = cv2.flip(cel_mask, 1)

    qh, qw = q_gray.shape[:2]
    if qw < 4 or qh < 4:
        return
    scale = min(float(max_w) / float(qw), float(max_h) / float(qh))
    tw = max(8, int(round(qw * scale)))
    th = max(8, int(round(qh * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    g_res = cv2.resize(q_gray, (tw, th), interpolation=interp)
    m_res = cv2.resize(cel_mask, (tw, th), interpolation=interp)

    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - th / 2.0))
    x1 = x0 + tw
    y1 = y0 + th

    H, W = canvas.shape[:2]
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return

    sx0, sy0 = cx0 - x0, cy0 - y0
    sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)

    roi = canvas[cy0:cy1, cx0:cx1].astype(np.float32)
    g_sub = g_res[sy0:sy1, sx0:sx1].astype(np.float32)
    m_sub = (m_res[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0)[..., None]

    # 1. Opaque Warm Cel Interior Fill (prevents background lines bleeding through character/prop)
    base_cel = np.array(BGR_GUTTER, dtype=np.float32)
    if tint_bgr is not None:
        t_arr = np.array(tint_bgr, dtype=np.float32)
        fill_col = base_cel * (1.0 - tint_strength) + t_arr * tint_strength
    else:
        fill_col = base_cel
    roi = roi * (1.0 - m_sub) + fill_col * m_sub

    # 2. Crisp Anti-Aliased Hand-Drawn Black Marker Ink Strokes (`BGR_INK`)
    ink_alpha = np.clip((235.0 - g_sub) / 185.0, 0.0, 1.0) ** 0.85
    ink_alpha = ink_alpha[..., None]
    ink_col = np.array(BGR_INK, dtype=np.float32)
    roi = roi * (1.0 - ink_alpha) + ink_col * ink_alpha

    canvas[cy0:cy1, cx0:cx1] = np.clip(roi, 0, 255).astype(np.uint8)


def draw_agent_animator(
    img: np.ndarray,
    agent_id: str,
    cx: int,
    cy: int,
    scale: float = 1.0,
    pose: int = 0,
    dark_bg: bool = False,
    flip_x: bool = False,
):
    """
    Draws one of our 5 bespoke Agent Animator characters (`nib`, `blueline`, `celinker`,
    `sprocket`, `chime`) using their genuine hand-drawn 4-keyframe Quick, Draw! doodle
    sheets (`assets/studio_doodles/agent_<id>_keys.jpg`), cycling through keyframes K1..K4!
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
    draw_studio_doodle_sprite(
        img,
        sheet_name=sheet_name,
        quad_idx=int(pose) % 4,
        cx=cx,
        cy=cy + int(round(12 * scale)),
        max_w=max_w,
        max_h=max_h,
        tint_bgr=ag["pale_bgr"],
        tint_strength=0.34,
        flip_x=flip_x,
    )


# =============================================================================
# 4-SCENE ANIMATED STUDIO FILM ENGINE (EVERY SCENE SHOWS THE AGENTS WORKING!)
# =============================================================================
def render_single_panel_fullscreen(
    spec: dict,
    panel_num: int,
    sub_step: float = 1.0,
    shot_frame_idx: int = 16,
    boil_phase: int = 0,
) -> np.ndarray:
    """
    Renders Scene `panel_num` (`1..4`) at FULL `960x540` (16:9) resolution.
    In EVERY SINGLE SCENE, we see our Agent Animators inside the studio physically
    performing that stage of making the cartoon, with all embedded monitors/canvases
    locked to their exact `4:3` (`1.3333:1`) aspect ratio!
    """
    W, H = 960, 540
    canvas = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)
    slug = spec["featured_shot"]
    lead_id = spec.get("lead_agent", "nib")
    cameo_id = spec.get("cameo_agent", "blueline")
    lead_ag = STUDIO_AGENTS[lead_id]
    cameo_ag = STUDIO_AGENTS[cameo_id]
    arch_sketches, blue_roughs, ink_cels, cart_frames = load_shot_frames(slug)

    pad = 16
    cap_h = 52
    bx0, by0, bx1, by1 = pad, pad, W - pad, H - pad

    # =========================================================================
    # SCENE 1 (ROOM 1): THE STORYBOARD & ARCHIVAL MOVIOLA ROOM
    # Lead Agent + Co-Worker Agent study the live 4:3 Archival Moviola Monitor
    # (`220x165`, exact 4:3) beside the hand-drawn Moviola Console & Film Rack
    # while pinning the 4 keyframes onto the Corkboard Storyboard Wall!
    # =========================================================================
    if panel_num == 1:
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        # Studio Wainscoting / Floorboard Horizon Line
        cv2.rectangle(canvas, (bx0, by1 - 82), (bx1, by1), BGR_PAPER_DARK, -1)
        cv2.line(canvas, (bx0, by1 - 82), (bx1, by1 - 82), BGR_INK, 3)

        # Left Studio Wall Prop: Hand-Drawn 35mm Film Canister Vault Shelf & Reels (`studio_room_props_keys` Q0)
        draw_studio_doodle_sprite(
            canvas, "studio_room_props_keys", 0,
            cx=bx0 + 78, cy=by1 - 155, max_w=118, max_h=142,
            tint_bgr=BGR_AMBER_PALE, tint_strength=0.28,
        )

        # Center-Left Prop: Hand-Drawn Wheeled Twin-Reel Moviola Console (`studio_apparatus_keys` Q0)
        draw_studio_doodle_sprite(
            canvas, "studio_apparatus_keys", 0,
            cx=435, cy=by1 - 135, max_w=145, max_h=175,
            tint_bgr=BGR_BLUE_PALE, tint_strength=0.30,
        )

        # Center-Right: Live 4:3 Archival Moviola Screen (`220x165`, exact 4:3!)
        mx0, my0 = 492, by0 + cap_h + 40
        mw, mh = 220, 165  # 220 / 165 = 4:3 exact!
        cv2.rectangle(canvas, (mx0 - 8, my0 - 8), (mx0 + mw + 8, my0 + mh + 26), BGR_BLUE_PALE, -1)
        cv2.rectangle(canvas, (mx0 - 8, my0 - 8), (mx0 + mw + 8, my0 + mh + 26), BGR_INK, 4)

        arch_fr = arch_sketches[shot_frame_idx % len(arch_sketches)]
        mov_thumb = cv2.resize(arch_fr, (mw, mh), interpolation=cv2.INTER_AREA)
        canvas[my0:my0 + mh, mx0:mx0 + mw] = mov_thumb
        cv2.rectangle(canvas, (mx0, my0), (mx0 + mw, my0 + mh), BGR_INK, 3)

        # Right Wall: Corkboard Storyboard Pin-Up Board (4 mini 4:3 cards being pinned!)
        sbx0, sby0 = 732, by0 + cap_h + 26
        sbx1, sby1 = bx1 - 18, by0 + cap_h + 206
        cv2.rectangle(canvas, (sbx0 + 4, sby0 + 4), (sbx1 + 4, sby1 + 4), BGR_INK, -1)
        cv2.rectangle(canvas, (sbx0, sby0), (sbx1, sby1), BGR_AMBER_PALE, -1)
        cv2.rectangle(canvas, (sbx0, sby0), (sbx1, sby1), BGR_INK, 3)

        n_pinned = max(1, min(4, int(sub_step * 4) + 1)) if sub_step < 0.98 else 4
        k_frames = [0, 10, 21, 32]
        for pk_i in range(n_pinned):
            pr = pk_i // 2
            pc = pk_i % 2
            cx_k = sbx0 + 12 + pc * 86
            cy_k = sby0 + 30 + pr * 72
            kw_s, kh_s = 76, 57  # 76 / 57 = 4:3 exact!
            k_im = cv2.resize(blue_roughs[k_frames[pk_i]], (kw_s, kh_s), interpolation=cv2.INTER_AREA)
            canvas[cy_k:cy_k + kh_s, cx_k:cx_k + kw_s] = k_im
            cv2.rectangle(canvas, (cx_k, cy_k), (cx_k + kw_s, cy_k + kh_s), BGR_INK, 2)
            cv2.circle(canvas, (cx_k + kw_s // 2, cy_k + 4), 4, BGR_RED, -1, cv2.LINE_AA)

        # Lead Agent Animator on Left + Co-Worker Agent Animator at Storyboard/Moviola on Right!
        draw_agent_animator(canvas, lead_id, bx0 + 228, by1 - 142, scale=1.18, pose=boil_phase, dark_bg=False)
        draw_agent_animator(canvas, cameo_id, 785, by1 - 135, scale=1.02, pose=(boil_phase + 2) % 4, dark_bg=False)

    # =========================================================================
    # SCENE 2 (ROOM 2): THE 4-DESK WORKTREE DRAFTING BULLPEN
    # Four Agent Animators sit at 4 hand-drawn Tilted Lightbox Desks (`DESK K1..K4`).
    # Above each desk is their `4:3` Keyframe Sheet (`192x144`, exact 4:3!)
    # progressing live from Blue Pencil Rough -> Black India Ink -> Gouache Cel!
    # =========================================================================
    elif panel_num == 2:
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        cv2.rectangle(canvas, (bx0, by1 - 135), (bx1, by1), BGR_PAPER_DARK, -1)
        cv2.line(canvas, (bx0, by1 - 135), (bx1, by1 - 135), BGR_INK, 3)

        desk_agents = ["blueline", "celinker", "nib", "sprocket"]
        desk_poses = [0, 1, 0, 2]
        key_indices = [0, 10, 21, 32]
        kw, kh = 192, 144  # 192 / 144 = 4:3 exact!

        for d_i in range(4):
            dx0 = bx0 + 22 + d_i * 224
            dy0 = by0 + cap_h + 38
            f_idx = key_indices[d_i]

            # Progressive Inking Stage across Scene 2 (`sub_step` 0.0 -> 1.0):
            # Each desk progresses from Blue Non-Photo Rough -> Black Ink Line -> Full Gouache Cel!
            local_prog = np.clip(sub_step * 1.35 - d_i * 0.10, 0.0, 1.0)
            if local_prog < 0.36:
                src_sheet = blue_roughs[f_idx]
                stage_col = BGR_BLUE
            elif local_prog < 0.72:
                src_sheet = ink_cels[f_idx]
                stage_col = BGR_INK
            else:
                src_sheet = cart_frames[f_idx]
                stage_col = BGR_AMBER

            # 1. Exact 4:3 Keyframe Drawing Sheet above the desk (`192x144`)
            cv2.rectangle(canvas, (dx0 + 4, dy0 + 4), (dx0 + kw + 4, dy0 + kh + 4), BGR_INK, -1)
            sheet_thumb = cv2.resize(src_sheet, (kw, kh), interpolation=cv2.INTER_AREA)
            canvas[dy0:dy0 + kh, dx0:dx0 + kw] = sheet_thumb
            cv2.rectangle(canvas, (dx0, dy0), (dx0 + kw, dy0 + kh), stage_col, 4 if local_prog >= 0.72 else 3)
            # Top brass registration pegbar pins on the drawing sheet
            for px_off in (48, 96, 144):
                cv2.circle(canvas, (dx0 + px_off, dy0 + 6), 4, BGR_INK, -1, cv2.LINE_AA)

            # Warm projection lines connecting the lightbox desk up to the 4:3 keyframe sheet
            cv2.line(canvas, (dx0 + 124, by1 - 152), (dx0 + 16, dy0 + kh + 2), BGR_AMBER, 2, cv2.LINE_AA)
            cv2.line(canvas, (dx0 + 152, by1 - 152), (dx0 + kw - 12, dy0 + kh + 2), BGR_AMBER, 2, cv2.LINE_AA)

            # 2. Hand-Drawn Tilted Animator Lightbox Drafting Desk (`studio_apparatus_keys` Q1)
            draw_studio_doodle_sprite(
                canvas, "studio_apparatus_keys", 1,
                cx=dx0 + 138, cy=by1 - 134, max_w=128, max_h=148,
                tint_bgr=BGR_AMBER_PALE, tint_strength=0.35,
            )

            # 3. Hand-Drawn Agent Animator actively drawing at this Worktree Lightbox Desk!
            ag_id = desk_agents[d_i]
            draw_agent_animator(
                canvas, ag_id,
                cx=dx0 + 56, cy=by1 - 118,
                scale=0.82, pose=(desk_poses[d_i] + boil_phase) % 4, dark_bg=False,
            )

    # =========================================================================
    # SCENE 3 (ROOM 3): THE 7-LEVEL MULTIPLANE CAMERA TOWER
    # `Cel-Inker` loads the 3 separated glass depth platens (`P6 Sky`, `P4 Mid`, `P2 Hero`)
    # into the hand-drawn Multiplane Camera Tower while `Nib` cranks the camera wheel,
    # and the `4:3` Camera Viewfinder Monitor (`340x255`, exact 4:3!) composites them live!
    # =========================================================================
    elif panel_num == 3:
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        cv2.rectangle(canvas, (bx0, by1 - 92), (bx1, by1), BGR_PAPER_DARK, -1)
        cv2.line(canvas, (bx0, by1 - 92), (bx1, by1 - 92), BGR_INK, 3)

        # Center-Left: Hand-Drawn 7-Level Multiplane Camera Tower Rig (`studio_apparatus_keys` Q2)
        cam_cx = 298
        cam_y = by0 + cap_h + 38
        draw_studio_doodle_sprite(
            canvas, "studio_apparatus_keys", 2,
            cx=cam_cx, cy=by0 + cap_h + 215, max_w=265, max_h=390,
            tint_bgr=BGR_AMBER_PALE, tint_strength=0.25,
        )

        # Composite the 3 separated Quick, Draw! depth doodles onto the 3 glass platens of the tower!
        shift_px = int(round(math.sin(sub_step * math.pi * 2) * 12))
        platen_doodles = [
            (by0 + cap_h + 195, slug, int(shift_px * 1.0)),
            (by0 + cap_h + 266, "tree", int(shift_px * 0.5)),
            (by0 + cap_h + 338, "cloud", int(shift_px * 0.2)),
        ]
        for py, pcat, dx_s in platen_doodles:
            draw_qd_doodle(canvas, pcat, cam_cx - 6 + dx_s, py, 44, 44, fill_bgr=BGR_GUTTER, thickness=2, boil_phase=boil_phase)

        # `Cel-Inker` on Left loading the glass platens + `Nib` on Right turning the Crank Wheel!
        draw_agent_animator(canvas, "celinker", bx0 + 96, by1 - 135, scale=1.02, pose=2 if (boil_phase % 2 == 0) else 0, dark_bg=False)
        draw_agent_animator(canvas, "nib", 472, by1 - 135, scale=1.02, pose=2 if (boil_phase % 2 == 0) else 1, dark_bg=False)

        # Right Side: Exact 4:3 Multiplane Camera Viewfinder Monitor (`340x255` -> 340/255 = 4:3!)
        vw, vh = 340, 255  # EXACT 4:3 aspect ratio!
        vx0 = bx1 - vw - 22
        vy0 = by0 + cap_h + 56
        # Braided cable from Overhead Camera Unit to Viewfinder Monitor
        cv2.line(canvas, (cam_cx + 32, cam_y + 10), (vx0 + vw // 2, cam_y + 10), BGR_INK, 4, cv2.LINE_AA)
        cv2.line(canvas, (vx0 + vw // 2, cam_y + 10), (vx0 + vw // 2, vy0), BGR_INK, 4, cv2.LINE_AA)

        cv2.rectangle(canvas, (vx0 - 8, vy0 - 8), (vx0 + vw + 8, vy0 + vh + 8), BGR_INK, -1)
        f_cur = shot_frame_idx % len(cart_frames)
        if sub_step < 0.30:
            vf_src = blue_roughs[f_cur]
        elif sub_step < 0.62:
            vf_src = cv2.addWeighted(cart_frames[f_cur], 0.55, ink_cels[f_cur], 0.45, 0)
        else:
            vf_src = cart_frames[f_cur]
        vf_img = cv2.resize(vf_src, (vw, vh), interpolation=cv2.INTER_LINEAR)
        canvas[vy0:vy0 + vh, vx0:vx0 + vw] = vf_img
        cv2.rectangle(canvas, (vx0, vy0), (vx0 + vw, vy0 + vh), BGR_AMBER, 3)

    # =========================================================================
    # SCENE 4 (ROOM 4): THE 35MM DAILIES SCREENING THEATRE & FOLEY STAGE
    # Inside the Studio Screening Theatre:
    #   - Left Booth: Hand-Drawn Tripod Twin-Reel 35mm Projector casting a golden beam
    #     + Exact 4:3 Archival Reference Monitor (`180x135`, exact 4:3!)
    #   - Center-Right Silver Screen: Exact 4:3 Grand Theatre Screen (`440x330`, exact 4:3!)
    #     playing the finished 7-plane cartoon!
    #   - Foreground Stage: `Sprocket`, Hand-Drawn Foley Prop Table & Boom Mic, `Chime`, & `Nib`!
    # =========================================================================
    elif panel_num == 4:
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        # Theatre Floor & Stage Apron
        cv2.rectangle(canvas, (bx0, by1 - 108), (bx1, by1), BGR_PAPER_DARK, -1)
        cv2.line(canvas, (bx0, by1 - 108), (bx1, by1 - 108), BGR_INK, 3)

        # Right: Grand Theatre Silver Screen at EXACT 4:3 (`440x330` -> 440/330 = 4:3!)
        sw, sh = 440, 330  # EXACT 4:3 aspect ratio!
        sx0 = bx1 - sw - 24
        sy0 = by0 + cap_h + 36

        # Golden Projector Light Cone across theatre onto the 4:3 Silver Screen!
        px_proj, py_proj = 315, by0 + cap_h + 125
        beam_poly = np.array([
            [px_proj + 48, py_proj - 12],
            [sx0, sy0 + 6],
            [sx0, sy0 + sh - 6],
            [px_proj + 48, py_proj + 18],
        ], dtype=np.int32)
        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, beam_poly, BGR_AMBER_PALE, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        # Center-Left: Hand-Drawn Tripod-Mounted Twin-Reel 35mm Film Projector (`studio_apparatus_keys` Q3)
        draw_studio_doodle_sprite(
            canvas, "studio_apparatus_keys", 3,
            cx=px_proj, cy=py_proj + 12, max_w=165, max_h=190,
            tint_bgr=BGR_BLUE_PALE, tint_strength=0.28,
        )

        # Top-Left: Exact 4:3 Booth Archival Reference Monitor (`180x135` -> 180/135 = 4:3!)
        aw, ah = 180, 135  # EXACT 4:3 aspect ratio!
        ax0, ay0 = bx0 + 20, by0 + cap_h + 36
        cur_idx = shot_frame_idx % len(cart_frames)
        arch_im = cv2.resize(arch_sketches[cur_idx], (aw, ah), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (ax0 + 4, ay0 + 4), (ax0 + aw + 4, ay0 + ah + 4), BGR_INK, -1)
        canvas[ay0:ay0 + ah, ax0:ax0 + aw] = arch_im
        cv2.rectangle(canvas, (ax0, ay0), (ax0 + aw, ay0 + ah), BGR_INK, 3)

        # Right: The 440x330 Exact 4:3 Silver Screen with Terracotta Proscenium Curtain Frame!
        cv2.rectangle(canvas, (sx0 - 12, sy0 - 10), (sx0 + sw + 12, sy0 + sh + 10), BGR_RED, -1)
        cv2.rectangle(canvas, (sx0 - 12, sy0 - 10), (sx0 + sw + 12, sy0 + sh + 10), BGR_INK, 4)
        hero_im = cv2.resize(cart_frames[cur_idx], (sw, sh), interpolation=cv2.INTER_LINEAR)
        canvas[sy0:sy0 + sh, sx0:sx0 + sw] = hero_im
        cv2.rectangle(canvas, (sx0, sy0), (sx0 + sw, sy0 + sh), BGR_INK, 4)

        # Bottom-Left Foreground: `Sprocket`, Hand-Drawn Foley Prop Table & Boom Mic, `Chime`, & `Nib`!
        draw_agent_animator(canvas, "sprocket", bx0 + 58, by1 - 120, scale=0.80, pose=2 if (boil_phase % 2 == 0) else 3, dark_bg=False)
        draw_studio_doodle_sprite(
            canvas, "studio_room_props_keys", 2,
            cx=bx0 + 166, cy=by1 - 110, max_w=116, max_h=128,
            tint_bgr=BGR_AMBER_PALE, tint_strength=0.30,
        )
        draw_agent_animator(canvas, "chime", bx0 + 274, by1 - 120, scale=0.80, pose=boil_phase % 4, dark_bg=False)
        draw_agent_animator(canvas, "nib", bx0 + 378, by1 - 120, scale=0.78, pose=3 if (boil_phase % 2 == 0) else 0, dark_bg=False)

    # =========================================================================
    # PIL HIGH-RESOLUTION COMIC LETTERING (0% TRUNCATION GUARANTEED)
    # =========================================================================
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    if panel_num == 1:
        draw.rectangle([bx0, by0, bx1, by0 + cap_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
        draw.rectangle([bx0, by0, bx0 + 52, by0 + cap_h], fill=lead_ag["color_rgb"], outline=(22, 20, 18), width=4)
        draw.text((bx0 + 17, by0 + 9), "1", font=FONT_DISPLAY_XL, fill=(255, 251, 244))
        hdr_txt = f"ROOM 1 (STORYBOARD & MOVIOLA) — {spec['title'].upper()}"
        f_hdr = fit_text_font(draw, hdr_txt, (bx1 - bx0) - 70, [FONT_DISPLAY_LG, FONT_DISPLAY_MD, FONT_CAPTION_MD])
        draw.text((bx0 + 66, by0 + 13), hdr_txt, font=f_hdr, fill=(22, 20, 18))

        # Speech Balloon above Lead Agent Animator on left (`x: 42..468`)
        s_box = (bx0 + 24, by0 + cap_h + 12, 472, by0 + cap_h + 150)
        tx, ty = bx0 + 160, by1 - 205
        bx_mid = (s_box[0] + s_box[2]) // 2
        tail_pts = [(bx_mid - 16, s_box[3] - 4), (bx_mid + 14, s_box[3] - 4), (tx, ty)]
        draw.polygon([(p[0] + 4, p[1] + 4) for p in tail_pts], fill=(22, 20, 18))
        draw.rounded_rectangle([s_box[0] + 4, s_box[1] + 4, s_box[2] + 4, s_box[3] + 4], radius=24, fill=(22, 20, 18))
        draw.rounded_rectangle(s_box, radius=24, fill=(255, 255, 255), outline=(22, 20, 18), width=4)
        draw.polygon(tail_pts, fill=(255, 255, 255))
        draw.line([tail_pts[0], tail_pts[2]], fill=(22, 20, 18), width=4)
        draw.line([tail_pts[1], tail_pts[2]], fill=(22, 20, 18), width=4)

        lines = textwrap.wrap(spec["speech_p1"].upper(), width=30)[:4]
        lh = 24 if len(lines) == 4 else 27
        sy_txt = s_box[1] + ((s_box[3] - s_box[1]) - len(lines) * lh) // 2 - 2
        for i, ln in enumerate(lines):
            f_sp = fit_text_font(draw, ln, (s_box[2] - s_box[0]) - 24, [FONT_SPEECH_MD if len(lines) == 4 else FONT_SPEECH_LG, FONT_SPEECH_MD, FONT_CAPTION_MD])
            bb = draw.textbbox((0, 0), ln, font=f_sp)
            tw = bb[2] - bb[0]
            draw.text((s_box[0] + ((s_box[2] - s_box[0]) - tw) // 2, sy_txt + i * lh), ln, font=f_sp, fill=(22, 20, 18))

        # Labels on the 4:3 Moviola & Corkboard Storyboard Wall
        draw.rounded_rectangle([488, by0 + cap_h + 244, 712, by0 + cap_h + 272], radius=5, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        draw.text((502, by0 + cap_h + 249), "ARCHIVAL MOVIOLA (4:3)", font=FONT_LABEL_SM, fill=(22, 20, 18))

        draw.text((740, by0 + cap_h + 30), "STORYBOARD WALL (K1..K4)", font=FONT_LABEL_SM, fill=(22, 20, 18))

        # Crew pills at bottom
        lead_pill = f"{lead_ag['short_name']} // {lead_ag['dept']}"
        draw.rounded_rectangle([bx0 + 24, by1 - 36, bx0 + 305, by1 - 10], radius=6, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        f_lp = fit_text_font(draw, lead_pill, 265, [FONT_CAPTION_MD, FONT_CAPTION_SM])
        draw.text((bx0 + 34, by1 - 31), lead_pill, font=f_lp, fill=(22, 20, 18))

        cameo_pill = f"{cameo_ag['short_name']} // STORYBOARD PIN-UP"
        draw.rounded_rectangle([660, by1 - 36, bx1 - 18, by1 - 10], radius=6, fill=(214, 228, 240), outline=(22, 20, 18), width=2)
        f_cp = fit_text_font(draw, cameo_pill, 248, [FONT_CAPTION_MD, FONT_CAPTION_SM])
        draw.text((670, by1 - 31), cameo_pill, font=f_cp, fill=(22, 20, 18))

    else:
        cap_map = {2: spec["caption_p2"], 3: spec["caption_p3"], 4: spec["caption_p4"]}
        cap_txt = cap_map[panel_num].upper()
        draw.rectangle([bx0, by0, bx1, by0 + cap_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
        draw.rectangle([bx0, by0, bx0 + 52, by0 + cap_h], fill=lead_ag["color_rgb"], outline=(22, 20, 18), width=4)
        draw.text((bx0 + 17, by0 + 9), str(panel_num), font=FONT_DISPLAY_XL, fill=(255, 251, 244))
        f_cap = fit_text_font(draw, cap_txt, (bx1 - bx0) - 70, [FONT_CAPTION_LG, FONT_CAPTION_MD])
        draw.text((bx0 + 66, by0 + 14), cap_txt, font=f_cap, fill=(22, 20, 18))

    if panel_num == 2:
        desk_titles = [
            "DESK 1: BLUE-LINE (K1 4:3)",
            "DESK 2: CEL-INKER (K2 4:3)",
            "DESK 3: NIB (K3 4:3)",
            "DESK 4: SPROCKET (K4 4:3)",
        ]
        for d_i, d_lbl in enumerate(desk_titles):
            dx0 = bx0 + 22 + d_i * 224
            dy0 = by0 + cap_h + 10
            draw.rounded_rectangle([dx0, dy0, dx0 + 192, dy0 + 24], radius=4, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
            f_dl = fit_text_font(draw, d_lbl, 180, [FONT_LABEL_SM])
            draw.text((dx0 + 6, dy0 + 4), d_lbl, font=f_dl, fill=(22, 20, 18))

            local_prog = np.clip(sub_step * 1.35 - d_i * 0.10, 0.0, 1.0)
            if local_prog < 0.36:
                st_txt = "STEP A: BLUE PENCIL ROUGH"
                st_fill = (214, 228, 240)
            elif local_prog < 0.72:
                st_txt = "STEP B: SPEEDBALL INDIA INK"
                st_fill = (255, 251, 244)
            else:
                st_txt = "STEP C: GOUACHE CEL LOCKED"
                st_fill = (212, 232, 220)
            draw.rounded_rectangle([dx0, by1 - 30, dx0 + 192, by1 - 8], radius=4, fill=st_fill, outline=(22, 20, 18), width=2)
            f_st = fit_text_font(draw, st_txt, 180, [FONT_LABEL_SM])
            draw.text((dx0 + 8, by1 - 26), st_txt, font=f_st, fill=(22, 20, 18))

    elif panel_num == 3:
        # Labels on the Multiplane Tower & 4:3 Viewfinder Monitor
        draw.rounded_rectangle([bx0 + 16, by0 + cap_h + 16, bx0 + 236, by0 + cap_h + 42], radius=4, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        draw.text((bx0 + 24, by0 + cap_h + 21), "7-LEVEL MULTIPLANE TOWER", font=FONT_LABEL_SM, fill=(22, 20, 18))

        vx0 = bx1 - 340 - 22
        vy0 = by0 + cap_h + 56
        draw.rounded_rectangle([vx0, vy0 - 34, vx0 + 340, vy0 - 8], radius=5, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        draw.text((vx0 + 10, vy0 - 29), "LIVE CAMERA VIEWFINDER (4:3 ASPECT)", font=FONT_LABEL_MD, fill=(22, 20, 18))

        if sub_step < 0.30:
            comp_txt = "COMPOSITE STEP 1/3: LOADING BLUE ROUGH & HORIZON GRID"
        elif sub_step < 0.62:
            comp_txt = "COMPOSITE STEP 2/3: DROPPING HERO INK CEL ON PLANE P2"
        else:
            comp_txt = "COMPOSITE STEP 3/3: ALL 7 GOUACHE PLANES LOCKED IN PARALLAX!"
        draw.rounded_rectangle([vx0, vy0 + 255 + 14, vx0 + 340, vy0 + 255 + 44], radius=5, fill=(212, 232, 220), outline=(22, 20, 18), width=2)
        f_ct = fit_text_font(draw, comp_txt, 324, [FONT_LABEL_SM])
        draw.text((vx0 + 8, vy0 + 255 + 21), comp_txt, font=f_ct, fill=(22, 20, 18))

        draw.rounded_rectangle([bx0 + 16, by1 - 32, bx0 + 205, by1 - 8], radius=5, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        draw.text((bx0 + 24, by1 - 27), "CEL-INKER (PLATENS)", font=FONT_LABEL_SM, fill=(22, 20, 18))
        draw.rounded_rectangle([395, by1 - 32, 555, by1 - 8], radius=5, fill=(247, 216, 206), outline=(22, 20, 18), width=2)
        draw.text((405, by1 - 27), "NIB (CAMERA CRANK)", font=FONT_LABEL_SM, fill=(22, 20, 18))

    elif panel_num == 4:
        ax0, ay0 = bx0 + 20, by0 + cap_h + 36
        draw.rounded_rectangle([ax0, ay0 - 26, ax0 + 180, ay0 - 4], radius=4, fill=(214, 228, 240), outline=(22, 20, 18), width=2)
        draw.text((ax0 + 6, ay0 - 22), "ARCHIVAL MONITOR (4:3)", font=FONT_LABEL_SM, fill=(22, 20, 18))

        sx0 = bx1 - 440 - 24
        sy0 = by0 + cap_h + 36
        draw.rounded_rectangle([sx0, sy0 - 28, sx0 + 440, sy0 - 4], radius=4, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        scr_lbl = f"35MM DAILIES THEATRE SCREEN (4:3) — #{slug.upper()} (16/16 GATES PASS)"
        f_sl = fit_text_font(draw, scr_lbl, 424, [FONT_LABEL_SM])
        draw.text((sx0 + 8, sy0 - 23), scr_lbl, font=f_sl, fill=(22, 20, 18))

        # Foley cue badge above Chime & Sprocket in the Screening Theatre
        draw.rounded_rectangle([bx0 + 18, by1 - 30, bx0 + 412, by1 - 8], radius=5, fill=(212, 232, 220), outline=(22, 20, 18), width=2)
        draw.text((bx0 + 26, by1 - 26), "SPROCKET (16/16 AUDIT) · CHIME (FOLEY) · NIB (DIRECTOR)", font=FONT_LABEL_SM, fill=(22, 20, 18))

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_bgr, (bx0, by0), (bx1, by1), BGR_INK, 5)
    return out_bgr


def assemble_4panel_comic_page(spec: dict) -> np.ndarray:
    """
    Assembles all 4 Studio Rooms (`Room 1: Storyboard & Moviola` -> `Room 2: 4-Desk Worktree Bullpen`
    -> `Room 3: 7-Level Multiplane Tower` -> `Room 4: 35mm Screening Theatre & Foley Stage`)
    into a `1280x720` (16:9) 2x2 Comic Page where the Agent Animators are visibly working in every panel!
    """
    p1 = render_single_panel_fullscreen(spec, 1, sub_step=1.0, shot_frame_idx=16, boil_phase=0)
    p2 = render_single_panel_fullscreen(spec, 2, sub_step=0.65, shot_frame_idx=16, boil_phase=1)
    p3 = render_single_panel_fullscreen(spec, 3, sub_step=1.0, shot_frame_idx=16, boil_phase=2)
    p4 = render_single_panel_fullscreen(spec, 4, sub_step=1.0, shot_frame_idx=16, boil_phase=3)

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])
    return cv2.resize(grid, (1280, 720), interpolation=cv2.INTER_AREA)


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
    for beat_s in [0.05, 3.0, 6.0, 9.0, 10.2, 11.1]:
        if beat_s < duration_sec:
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


def render_single_bts_film(spec: dict) -> dict:
    """
    Renders a 12.0-second (144 frames @ 12fps) Behind-the-Scenes Animated Studio Film
    taking the viewer through all 4 Studio Rooms (3.0 seconds per room) as the Agent
    Animators create and screen the cartoon!
    """
    fid = spec["id"]
    mp4_path = BTS_DIR / f"{fid}.mp4"
    gif_path = BTS_DIR / f"{fid}.gif"
    poster_path = BTS_DIR / f"{fid}_poster.png"
    raw_mp4 = BTS_DIR / f"_{fid}_raw.mp4"
    wav_path = BTS_DIR / f"_{fid}.wav"

    fps = 12
    n_frames = 144  # 12.0 seconds (3.0s per Studio Room!)
    W, H = 960, 540

    poster_bgr = assemble_4panel_comic_page(spec)
    cv2.imwrite(str(poster_path), poster_bgr)

    vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gif_pil_frames = []

    for f_i in range(n_frames):
        boil_phase = (f_i // 4) % 4
        if f_i < 36:
            p_num = 1
            sub = f_i / 35.0
            shot_f = int(round(sub * 32))
        elif f_i < 72:
            p_num = 2
            sub = (f_i - 36) / 35.0
            shot_f = int(round(sub * 32))
        elif f_i < 108:
            p_num = 3
            sub = (f_i - 72) / 35.0
            shot_f = int(round(sub * 32))
        else:
            p_num = 4
            sub = (f_i - 108) / 35.0
            shot_f = int(round(sub * 32))

        frame_bgr = render_single_panel_fullscreen(
            spec,
            panel_num=p_num,
            sub_step=sub,
            shot_frame_idx=shot_f,
            boil_phase=boil_phase,
        )
        vw.write(frame_bgr)
        if f_i % 4 == 0:
            rgb_small = cv2.cvtColor(cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            gif_pil_frames.append(Image.fromarray(rgb_small))

    vw.release()

    gif_pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=gif_pil_frames[1:],
        duration=330,
        loop=0,
        optimize=True,
    )

    synthesize_studio_soundtrack(wav_path, duration_sec=n_frames / fps, seed=spec["num"] * 101)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(raw_mp4),
            "-i", str(wav_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "22",
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
    wing_specs = [
        ("studio_bg_1_multifloor_soundstage.jpg", AGENT_OFFICE_SPECS[0]),
        ("studio_bg_2_drafting_bullpen.jpg", AGENT_OFFICE_SPECS[2]),
        ("studio_bg_3_foley_screening_stage.jpg", VIDEO_SKILL_SPECS[7]),
        ("studio_bg_4_clocktower_optical_lab.jpg", VIDEO_SKILL_SPECS[3]),
        ("studio_bg_5_bullpen_unoccupied.jpg", AGENT_OFFICE_SPECS[1]),
    ]
    for fname, sp in wing_specs:
        page_bgr = assemble_4panel_comic_page(sp)
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
    print("1. Rendering 5 Agent Studio 4-Room Comic Pages (Agents Working in Every Scene)...")
    generate_studio_wing_comic_pages()

    office_results = []
    skill_results = []

    print("2. Rendering 10 Virtual Agent Office 4-Room Studio Films (12.0s each)...")
    for sp in AGENT_OFFICE_SPECS:
        res = render_single_bts_film(sp)
        office_results.append(res)
        print(f"   [Office {sp['num']:02d}/10] {sp['id']} -> {res['mp4']}")

    print("3. Rendering 8 AI Video Director Skill 4-Room Studio Films (12.0s each)...")
    for sp in VIDEO_SKILL_SPECS:
        res = render_single_bts_film(sp)
        skill_results.append(res)
        print(f"   [Skill  {sp['num']-10:02d}/08] {sp['id']} -> {res['mp4']}")

    p_off = render_contact_sheet(
        office_results,
        "PART I: 10 AGENT ANIMATOR STUDIO FILMS (ROOM 1 MOVIOLA -> ROOM 2 BULLPEN -> ROOM 3 TOWER -> ROOM 4 THEATRE)",
        "Watch Nib, Blue-Line, Cel-Inker, Sprocket & Chime Create Each 4:3 Cartoon Step-by-Step Across All 4 Studio Rooms",
        "bts_10_agent_offices_atlas.png",
    )
    p_skl = render_contact_sheet(
        skill_results,
        "PART II: 8 AGENT ANIMATOR CRAFT & MOTION SKILL STUDIO FILMS (SECONDHAND STUDIOS)",
        "Blue Rough -> Black Ink -> Gouache Cel · 7-Plane Multiplane Camera Rig · 35mm Dailies Screening & Foley Stage",
        "bts_8_video_skills_atlas.png",
    )
    print("4. Rendered Contact Sheets:", p_off.name, p_skl.name)

    for sample_id in ["office-01-agent-office", "office-02-agent-007", "skill-01-huashu-motion", "skill-04-reanimate", "skill-08-earshot"]:
        shutil.copy2(BTS_DIR / f"{sample_id}.gif", ARTIFACT_DIR / f"{sample_id}.gif")
        shutil.copy2(BTS_DIR / f"{sample_id}_poster.png", ARTIFACT_DIR / f"{sample_id}_poster.png")

    concat_txt = BTS_DIR / "_master_concat.txt"
    all_mp4s = [ROOT / r["mp4"] for r in (office_results + skill_results)]
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in all_mp4s) + "\n")
    master_mp4 = ROOT / "Behind_The_Scenes_Multiplane_Studio_Film.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-c", "copy", str(master_mp4),
        ],
        check=True,
    )
    concat_txt.unlink(missing_ok=True)
    print(f"5. Assembled Master Film: {master_mp4.name} ({master_mp4.stat().st_size / 1e6:.2f} MB)")

    payload = {
        "studio_title": "Secondhand Studios — The Autonomous Agent Animation Bullpen",
        "agents": [
            {"id": k, "name": v["name"], "role": v["role"], "dept": v["dept"]}
            for k, v in STUDIO_AGENTS.items()
        ],
        "office_films": office_results,
        "skill_films": skill_results,
        "master_film": "Behind_The_Scenes_Multiplane_Studio_Film.mp4",
    }
    (DATA_DIR / "bts_studio_films.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "bts_studio_data.js").write_text("window.BTS_STUDIO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
