#!/usr/bin/env python3
"""
Secondhand Studios: The Autonomous Agent Animation House
=========================================================
Synthesizes the 18 Behind-the-Scenes Studio Chapters (10 Virtual Agent Office
Architectures + 8 AI Video Director Skills) as a unified graphic-novel and
full-screen sequential film series starring our original cast of **Agent Animators**:

  1. NIB (The Fountain-Nib Showrunner & Multiplane Maestro)
  2. BLUE-LINE (The Keyframe Architect & Isolated Worktree Inker)
  3. CEL-INKER (The 7-Plane Gouache Painter & Parallax Rigger)
  4. SPROCKET (The Codex Dailies Auditor & 16-Gate Projectionist)
  5. CHIME (The Foley Percussionist & 55-Minute X-Sheet Continuity Keeper)

Strict Aspect Ratio & Visual Clarity Guarantees:
  - Native Shot Aspect Ratio (4:3 = 1.3333:1):
    * Cropped cleanly from `comparison.mp4` at `576x432` (exact 4:3, removing old HUD bars)
    * Stored at `480x360` (exact 4:3)
    * Displayed in Panel 3's 35mm Keyframe Filmstrip at `204x153` (exact 4:3)
    * Displayed in Panel 4's Screening Room at `280x210` (Archival Trace, exact 4:3)
      and `560x420` (7-Plane Multiplane Cartoon, exact 4:3)
  - Quick, Draw! Vector Doodles (1:1 Square Aspect Ratio):
    * Always scaled uniformly (`s = min(w, h)`) so doodles never stretch or squish.
  - Static 2x2 Comic Page (`1280x720`, 16:9) vs. Animated Full-Screen Sequential Film (`960x540`, 16:9, 10.0s).
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
BGR_INK = (18, 20, 22)            # #161412 Bold cartoonist Speedball India ink
BGR_GRAPHITE = (102, 112, 122)    # #7A7066 Pencil construction lines

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


FONT_DISPLAY_XL = load_comic_font(28, "display")
FONT_DISPLAY_LG = load_comic_font(22, "display")
FONT_DISPLAY_MD = load_comic_font(18, "display")
FONT_SPEECH_LG = load_comic_font(21, "comic")
FONT_SPEECH_MD = load_comic_font(18, "comic")
FONT_CAPTION_LG = load_comic_font(19, "bold")
FONT_CAPTION_MD = load_comic_font(16, "bold")
FONT_CAPTION_SM = load_comic_font(14, "comic")
FONT_LABEL_MD = load_comic_font(17, "bold")


def fit_text_font(draw: ImageDraw.ImageDraw, text: str, max_w: int, preferred_fonts: list) -> ImageFont.FreeTypeFont:
    for f in preferred_fonts:
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_w:
            return f
    for sz in range(15, 9, -1):
        f = load_comic_font(sz, "bold")
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_w:
            return f
    return load_comic_font(10, "bold")


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
        "dept": "ISOLATED WORKTREE INKING",
        "color_bgr": BGR_BLUE,
        "color_rgb": (43, 91, 132),
        "pale_bgr": BGR_BLUE_PALE,
    },
    "celinker": {
        "id": "celinker",
        "name": "Cel-Inker (Painter)",
        "short_name": "CEL-INKER",
        "role": "7-Plane Gouache Painter & Rigger",
        "dept": "7-PLANE GOUACHE & PARALLAX",
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
        "dept": "FOLEY & 55M CONTINUITY",
        "color_bgr": BGR_RED,
        "color_rgb": (201, 74, 41),
        "pale_bgr": BGR_AMBER_PALE,
    },
}

# =============================================================================
# 18 EPISODES: THE NARRATIVE OF THE AGENT ANIMATOR STUDIO
# =============================================================================
AGENT_OFFICE_SPECS = [
    {
        "id": "office-01-agent-office",
        "num": 1,
        "group": "office",
        "lead_agent": "nib",
        "cameo_agent": "celinker",
        "studio_wing": "bg1_multifloor",
        "wing_name": "EPISODE 01 // THE AGENT ANIMATOR STUDIO",
        "repo": "Ks1030/Agent-Office",
        "url": "https://github.com/Ks1030/Agent-Office",
        "category": "Multiplane Agent Studio Architecture",
        "install": "git clone https://github.com/Ks1030/Agent-Office.git && cd Agent-Office && npm install && npm run dev",
        "title": "Welcome to the Agent Animation Bullpen",
        "featured_shot": "giraffe-walks-past",
        "speech_p1": "WELCOME TO SECONDHAND STUDIOS! HERE, EVERY AI AGENT IS AN ANIMATOR WITH A DEDICATED DESK ON OUR 7-PLANE RIG!",
        "caption_p2": "OUR AGENT CREW SPLITS EVERY CARTOON INTO 3 CLEAR ROLES:",
        "diag_steps": [("pencil", "1. KEY INKER"), ("paintbrush", "2. 7-PLANE CEL"), ("camera", "3. DAILIES AUDIT")],
        "caption_p3": "FOUR HAND-DRAWN KEYFRAMES LOCK THE GIRAFFE'S WALK CYCLE:",
        "gutter_note": "BLUE-LINE INKS POSES K1, K2, K3 & K4 IN TRUE 4:3 ASPECT RATIO SO THE WALK CYCLE NEVER DISTORTS!",
        "caption_p4": "SCREENING ROOM: ARCHIVAL TRACE VS. 7-PLANE MULTIPLANE CARTOON!",
        "blurb": "Fountain-Nib Showrunner 'Nib' introduces our autonomous Agent Animation Studio, where coding agents operate as specialized keyframe inkers, 7-plane gouache painters, and 35mm dailies auditors.",
        "stages": ["Panel 1: Nib's Studio Welcome", "Panel 2: 3-Agent Crew Pipeline", "Panel 3: 4:3 Giraffe Keyframes", "Panel 4: 4:3 Multiplane Screening"],
    },
    {
        "id": "office-02-agent-007",
        "num": 2,
        "group": "office",
        "lead_agent": "blueline",
        "cameo_agent": "nib",
        "studio_wing": "bg5_bullpen",
        "wing_name": "EPISODE 02 // ISOLATED DRAFTING BOOTHS",
        "repo": "Tryilen/agent-007",
        "url": "https://github.com/Tryilen/agent-007",
        "category": "Isolated Git Worktree Drafting Desks",
        "install": "git clone https://github.com/Tryilen/agent-007.git && cd agent-007 && bun install && bun run dev",
        "title": "Why Every Agent Animator Gets a Private Desk",
        "featured_shot": "train",
        "speech_p1": "IF FOUR AGENT ANIMATORS DRAW ON ONE SHEET AT ONCE, OUR INK SMUDGES! SO EACH AGENT WORKS IN AN ISOLATED WORKTREE BOOTH!",
        "caption_p2": "HOW ISOLATED WORKTREE DRAFTING BOOTHS PREVENT INK COLLISIONS:",
        "diag_steps": [("door", "1. PRIVATE DESK"), ("pencil", "2. INK 4 KEYS"), ("train", "3. CLEAN MERGE")],
        "caption_p3": "INSIDE BOOTH #324, I INK FOUR 4:3 KEYFRAMES OF THE STEAM TRAIN:",
        "gutter_note": "EACH KEYFRAME IS DRAWN IN AN ISOLATED GIT WORKTREE BEFORE SPLICING ONTO THE MASTER REEL!",
        "caption_p4": "THE COMPLETED TRAIN REEL PASSES ALL 16 GATES WITH ZERO CONFLICTS!",
        "blurb": "Keyframe Architect 'Blue-Line' explains why parallel AI agents need isolated git worktree drafting desks so their drawings never overwrite one another.",
        "stages": ["Panel 1: Blue-Line's Worktree Booth", "Panel 2: Isolate -> Ink -> Merge", "Panel 3: 4:3 Train Keyframes", "Panel 4: Clean Master Reel"],
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
        "title": "Pinning Keyframes to the 14-Line Armature",
        "featured_shot": "lighthouse",
        "speech_p1": "BEFORE PAINTING A SINGLE CEL, OUR AGENT CREW PINS ALL FOUR KEYFRAMES TO THE STORYBOARD WALL TO LOCK THE HORIZON!",
        "caption_p2": "EVERY SHOT FOLLOWS OUR 3-STAGE STORYBOARD CHECKLIST:",
        "diag_steps": [("picture_frame", "1. PIN 4 KEYS"), ("compass", "2. GOLDEN GRID"), ("lighthouse", "3. PAINT CELS")],
        "caption_p3": "WATCH THE LIGHTHOUSE BEAM SWEEP ACROSS THE FOUR 4:3 KEYFRAMES:",
        "gutter_note": "CEL-INKER ALIGNS THE COASTAL CLIFF AND LIGHTHOUSE TOWER TO THE GOLDEN RATIO HORIZON!",
        "caption_p4": "ALL 7 GOUACHE PLANES LOCK INTO PLACE AROUND THE ROTATING BEAM!",
        "blurb": "7-Plane Painter 'Cel-Inker' demonstrates how the agent crew pins rough sketches to the storyboard wall and aligns them to Bouleau's 14-line composition grid before painting.",
        "stages": ["Panel 1: Cel-Inker at the Lightbox", "Panel 2: Pin -> Grid -> Paint", "Panel 3: 4:3 Lighthouse Keys", "Panel 4: 7-Plane Coastal Scene"],
    },
    {
        "id": "office-04-pixel-office",
        "num": 4,
        "group": "office",
        "lead_agent": "sprocket",
        "cameo_agent": "chime",
        "studio_wing": "bg3_screening",
        "wing_name": "EPISODE 04 // THE DAILIES AUDITOR",
        "repo": "feedingantioch/pixel-office",
        "url": "https://github.com/feedingantioch/pixel-office",
        "category": "Cross-Agent Dailies Critique (Sprocket Reviewing Blue-Line)",
        "install": "git clone https://github.com/feedingantioch/pixel-office.git && cd pixel-office && npm install && npm run build",
        "title": "Why One Agent Draws and Another Agent Audits",
        "featured_shot": "guitar",
        "speech_p1": "IN OUR STUDIO, NO AGENT GRADES ITS OWN HOMEWORK! I PROJECT EVERY REEL IN THE SCREENING ROOM TO AUDIT ALL 16 GATES!",
        "caption_p2": "OUR TWO-AGENT PEER REVIEW LOOP IN THE SCREENING THEATRE:",
        "diag_steps": [("paintbrush", "1. CLAUDE INKS"), ("eyeglasses", "2. CODEX AUDITS"), ("guitar", "3. 16/16 STAMP")],
        "caption_p3": "I INSPECT ALL FOUR 4:3 KEYFRAMES OF SHOT #136 GUITAR:",
        "gutter_note": "SPROCKET CHECKS LINE WEIGHT, 4:3 FRAMING, AND PARALLAX SEPARATION ACROSS EVERY KEYFRAME!",
        "caption_p4": "STAMPED AND APPROVED: THE GUITAR STRUMS IN FULL MULTIPLANE DEPTH!",
        "blurb": "Projectionist & Critic 'Sprocket' explains why separating the Animator Agent (Claude) from the Dailies Auditor Agent (Codex) prevents hallucinations and guarantees 16/16 studio gates.",
        "stages": ["Panel 1: Sprocket's Screening Room", "Panel 2: Draw -> Audit -> Stamp", "Panel 3: 4:3 Guitar Keyframes", "Panel 4: Audited Master Reel"],
    },
    {
        "id": "office-05-pixel-agents",
        "num": 5,
        "group": "office",
        "lead_agent": "blueline",
        "cameo_agent": "celinker",
        "studio_wing": "bg5_bullpen",
        "wing_name": "EPISODE 05 // LIVING LINE BOIL VS. SLIDING",
        "repo": "pablodelucca/pixel-agents",
        "url": "https://github.com/pablodelucca/pixel-agents",
        "category": "Parallel Subagent Inkers & Hand-Drawn Boil",
        "install": "git clone https://github.com/pablodelucca/pixel-agents.git && cd pixel-agents && npm install",
        "title": "Four Parallel Inkers Create Living Line Boil",
        "featured_shot": "tractor",
        "speech_p1": "SLIDING ONE DEAD CUTOUT ACROSS THE SCREEN LOOKS ROBOTIC! FOUR INKER AGENTS REDRAW FOUR DISTINCT POSES SO THE LINE BREATHES!",
        "caption_p2": "FOUR SUBAGENT INKERS REDRAW KEYFRAMES K1, K2, K3 & K4:",
        "diag_steps": [("pencil", "1. KEYS K1 & K2"), ("paintbrush", "2. KEYS K3 & K4"), ("tractor", "3. LIVING BOIL")],
        "caption_p3": "COMPARE THE FOUR HAND-DRAWN 4:3 TRACTOR POSES ON THE STRIP:",
        "gutter_note": "BECAUSE EACH KEYFRAME IS FRESHLY INKED, THE TRACTOR CHUGS FORWARD WITH ORGANIC HAND-DRAWN BOIL!",
        "caption_p4": "THE TRACTOR ROLLS ACROSS THE FURROWS WITH GENUINE WEIGHT!",
        "blurb": "Blue-Line shows how spawning 4 subagent inkers in parallel to draw 4 distinct keyframes eliminates robotic affine sliding and produces authentic hand-drawn line boil.",
        "stages": ["Panel 1: Why Sliding Looks Dead", "Panel 2: 4 Parallel Key Inkers", "Panel 3: 4:3 Tractor Keyframes", "Panel 4: Living Line Boil"],
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
        "title": "Keeping Studio Continuity Across Shift Changes",
        "featured_shot": "speedboat",
        "speech_p1": "WHEN AN AGENT'S 55-MINUTE MEMORY WINDOW EXPIRES, DO WE LOSE THE STORY? NEVER! WE SEAL THE MASTER EXPOSURE SHEET!",
        "caption_p2": "BEFORE THE 55-MINUTE CLOCK CHIMES, WE LOG THE X-SHEET:",
        "diag_steps": [("clock", "1. 55M CLOCK"), ("book", "2. X-SHEET LOG"), ("speedboat", "3. NEXT SHIFT")],
        "caption_p3": "EVERY 4:3 SPEEDBOAT KEYFRAME REMEMBERS THE EXACT CAMERA ARC:",
        "gutter_note": "CHIME RECORDS PALETTE SWATCHES, PLANE DEPTHS, AND VELOCITY VECTORS ON THE EXPOSURE SHEET!",
        "caption_p4": "THE NEXT AGENT SHIFT PICKS UP SEAMLESSLY AS THE SPEEDBOAT PLANES!",
        "blurb": "Continuity Keeper 'Chime' explains how sealing the studio Exposure Sheet (X-Sheet) before the 55-minute prompt-cache chime keeps visual continuity unbroken across shift changes.",
        "stages": ["Panel 1: Chime's 55m Clocktower", "Panel 2: Clock -> X-Sheet -> Handoff", "Panel 3: 4:3 Speedboat Keys", "Panel 4: Seamless Shift Handoff"],
    },
    {
        "id": "office-07-workbench",
        "num": 7,
        "group": "office",
        "lead_agent": "celinker",
        "cameo_agent": "sprocket",
        "studio_wing": "bg1_multifloor",
        "wing_name": "EPISODE 07 // THE 7-PLANE MULTIPLANE TOWER",
        "repo": "taw0002/Claude-Code-Agent-Monitor-Workbench",
        "url": "https://github.com/taw0002/Claude-Code-Agent-Monitor-Workbench",
        "category": "7-Level Multiplane Camera & Depth Parallax",
        "install": "git clone https://github.com/taw0002/Claude-Code-Agent-Monitor-Workbench",
        "title": "Separating Sky, Trees, and Hero Across 7 Glass Planes",
        "featured_shot": "bicycle",
        "speech_p1": "HOW DO CLOUDS, TREES, AND THE BICYCLE GLIDE AT DIFFERENT SPEEDS? WE STACK THEM ON SEVEN SEPARATE GLASS PLANES!",
        "caption_p2": "THE MULTIPLANE TOWER ASSIGNS EACH GLASS LEVEL ITS PARALLAX:",
        "diag_steps": [("cloud", "1. SKY (0.2x)"), ("tree", "2. TREES (0.5x)"), ("bicycle", "3. HERO (1.0x)")],
        "caption_p3": "WATCH THE BICYCLE PEDAL FORWARD ACROSS FOUR 4:3 KEYFRAMES:",
        "gutter_note": "DISTANT SKY DRIFTS SLOWLY AT 0.2x WHILE MIDGROUND TREES PASS AT 0.5x AND THE BICYCLE AT 1.0x!",
        "caption_p4": "PROJECTED TOGETHER, FLAT PAPER GAINS TRUE MULTIPLANE DEPTH!",
        "blurb": "Cel-Inker takes us up the 7-level Multiplane Camera Tower to show how separating Sky (0.2x), Midground Trees (0.5x), and Hero (1.0x) creates dimensional parallax.",
        "stages": ["Panel 1: The 7-Level Glass Tower", "Panel 2: 0.2x / 0.5x / 1.0x Planes", "Panel 3: 4:3 Bicycle Keyframes", "Panel 4: Multiplane Parallax"],
    },
    {
        "id": "office-08-kanban-code",
        "num": 8,
        "group": "office",
        "lead_agent": "nib",
        "cameo_agent": "blueline",
        "studio_wing": "bg5_bullpen",
        "wing_name": "EPISODE 08 // THE 4-DESK PRODUCTION LINE",
        "repo": "langoustine69/kanban-code",
        "url": "https://github.com/langoustine69/kanban-code",
        "category": "4-Stage Studio Production Kanban",
        "install": "git clone https://github.com/langoustine69/kanban-code && cd kanban-code && npm i && npm run compile",
        "title": "How a Shot Travels Across Our Four Studio Desks",
        "featured_shot": "carrot",
        "speech_p1": "NO SINGLE AGENT TRIES TO DO EVERYTHING AT ONCE! EVERY SHOT TRAVELS ACROSS FOUR SPECIALIST DESKS IN ORDER!",
        "caption_p2": "FROM ARCHIVAL TRACE TO KEYFRAMES TO GOUACHE TO MASTER REEL:",
        "diag_steps": [("light_bulb", "1. STORYBOARD"), ("pencil", "2. 4-KEY INKING"), ("carrot", "3. GOUACHE CEL")],
        "caption_p3": "HERE ARE THE FOUR STRUCTURAL 4:3 KEYFRAMES OF SHOT #063 CARROT:",
        "gutter_note": "EACH DEPARTMENT AGENT SIGNS OFF ON THE EXPOSURE SHEET BEFORE PASSING THE FOLDER TO THE NEXT DESK!",
        "caption_p4": "CRAFT AND GOUACHE TEXTURE BRING THE SIMPLE CARROT DOODLE TO LIFE!",
        "blurb": "Showrunner Nib walks through the 4-desk studio Kanban pipeline: Storyboard Pin-Up -> 4-Keyframe Inking -> 7-Plane Gouache Painting -> Steenbeck Vault Splice.",
        "stages": ["Panel 1: Nib's 4-Desk Pipeline", "Panel 2: Storyboard -> Ink -> Paint", "Panel 3: 4:3 Carrot Keyframes", "Panel 4: Finished Gouache Cel"],
    },
    {
        "id": "office-09-vibe-kanban",
        "num": 9,
        "group": "office",
        "lead_agent": "sprocket",
        "cameo_agent": "blueline",
        "studio_wing": "bg3_screening",
        "wing_name": "EPISODE 09 // TWIN-REEL DAILIES COMPARISON",
        "repo": "BloopAI/vibe-kanban",
        "url": "https://github.com/BloopAI/vibe-kanban",
        "category": "Side-by-Side Archival vs. Multiplane Dailies",
        "install": "npx vibe-kanban",
        "title": "Synchronizing Archival Footage with Multiplane Ink",
        "featured_shot": "diamond",
        "speech_p1": "HOW DO WE PROVE OUR CARTOON HONORS THE ORIGINAL VIDEO'S MOTION? WE LOCK BOTH REELS IN OUR TWIN PROJECTOR!",
        "caption_p2": "JUXTAPOSING ARCHIVAL PENCIL TRACE AND 7-PLANE CARTOON CEL:",
        "diag_steps": [("television", "1. ARCHIVAL CLIP"), ("eyeglasses", "2. MOTION LOCK"), ("diamond", "3. TWIN DAILIES")],
        "caption_p3": "FRAME 1 SHOWS THE ARCHIVAL TRACE; FRAMES 2-4 SHOW OUR 4:3 KEYS:",
        "gutter_note": "SPROCKET VERIFIES THAT THE DIAMOND'S CENTER OF GRAVITY FOLLOWS THE EXACT ARCHIVAL TRAJECTORY!",
        "caption_p4": "LEFT: 4:3 ARCHIVAL PENCIL TRACE — RIGHT: 4:3 MULTIPLANE CARTOON!",
        "blurb": "Sprocket demonstrates the Twin-Reel Dailies Projector, locking the archival reference trace beside the newly inked 7-plane cartoon in true 4:3 aspect ratio.",
        "stages": ["Panel 1: Twin-Reel Projector", "Panel 2: Reference -> Lock -> Dailies", "Panel 3: 4:3 Diamond Keyframes", "Panel 4: Side-by-Side 4:3 Proof"],
    },
    {
        "id": "office-10-claude-code-kanban",
        "num": 10,
        "group": "office",
        "lead_agent": "nib",
        "cameo_agent": "chime",
        "studio_wing": "bg4_clocktower",
        "wing_name": "EPISODE 10 // THE 346-REEL MASTER VAULT",
        "repo": "NikkeTryHard/claude-code-kanban",
        "url": "https://github.com/NikkeTryHard/claude-code-kanban",
        "category": "Full-Catalog 346-Shot Master Archive",
        "install": "git clone https://github.com/NikkeTryHard/claude-code-kanban.git && cd claude-code-kanban && npm install && npm start",
        "title": "How Our Agent Crew Completed All 346 Cartoons",
        "featured_shot": "star",
        "speech_p1": "WORKING TOGETHER ACROSS 346 WORKTREES, OUR AGENT ANIMATOR CREW TRANSFORMED EVERY ARCHIVAL CLIP INTO A LIVING CARTOON!",
        "caption_p2": "EVERY ONE OF THE 346 REELS PASSED THROUGH THREE STUDIO GATES:",
        "diag_steps": [("pencil", "1. 346 TRACES"), ("paintbrush", "2. 2,422 PLANES"), ("star", "3. 346 REELS")],
        "caption_p3": "FOUR RADIANT 4:3 KEYFRAMES OF SHOT #285 STAR ON 35MM FILM:",
        "gutter_note": "346 SHOTS x 7 MULTIPLANE LAYERS = 2,422 HAND-INKED GOUACHE PLANES IN THE ARCHIVE VAULT!",
        "caption_p4": "ALL 346 MULTIPLANE CARTOONS ARE SPLICED AND READY FOR SCREENING!",
        "blurb": "Nib and Chime celebrate the completion of all 346 multiplane cartoons (2,422 painted depth planes) in the Secondhand Studios Master Archive Vault.",
        "stages": ["Panel 1: The 346-Reel Vault", "Panel 2: 346 Traces -> 2,422 Planes", "Panel 3: 4:3 Star Keyframes", "Panel 4: Master Archive Finale"],
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
        "wing_name": "SKILL 01 // CAMERA PACING & FRAMING",
        "repo": "octohash/huashu-skills (motion-graphics)",
        "url": "https://github.com/octohash/huashu-skills",
        "category": "Establishing Wide -> Action Medium -> Hero Close-Up",
        "install": "git clone https://github.com/octohash/huashu-skills.git",
        "title": "Directing the Camera: Wide, Medium, and Hero Framing",
        "featured_shot": "butterfly",
        "speech_p1": "A GREAT AGENT ANIMATOR DOESN'T JUST DRAW—WE DIRECT THE CAMERA! WIDE ESTABLISHING, MEDIUM ACTION, THEN HERO FOCUS!",
        "caption_p2": "THREE CLASSIC CAMERA FRAMINGS USED BY OUR AGENT DIRECTORS:",
        "diag_steps": [("picture_frame", "1. WIDE SCENE"), ("camera", "2. ACTION ARC"), ("butterfly", "3. HERO FOCUS")],
        "caption_p3": "WATCH THE BUTTERFLY FLUTTER ACROSS FOUR 4:3 KEYFRAMES:",
        "gutter_note": "EACH 4:3 FRAME GIVES THE BUTTERFLY BREATHING ROOM INSIDE THE GARDEN ENVIRONMENT!",
        "caption_p4": "UNHURRIED PACING LETS YOUR EYE FOLLOW EVERY WINGBEAT!",
        "blurb": "Nib demonstrates how the agent director stages camera framing and unhurried sequential pacing so the butterfly's flight reads effortlessly.",
        "stages": ["Panel 1: Nib's Camera Lesson", "Panel 2: Wide -> Medium -> Hero", "Panel 3: 4:3 Butterfly Keyframes", "Panel 4: Garden Multiplane Flight"],
    },
    {
        "id": "skill-02-rossark-motion-skill",
        "num": 12,
        "group": "skill",
        "timestamp": "05:37",
        "lead_agent": "celinker",
        "cameo_agent": "nib",
        "studio_wing": "bg1_multifloor",
        "wing_name": "SKILL 02 // MULTIPLANE CAMERA SWEEP",
        "repo": "rossark/motion-skill",
        "url": "https://github.com/rossark/motion-skill",
        "category": "Multiplane Camera Dolly & Parallax Rig",
        "install": "git clone https://github.com/rossark/motion-skill.git",
        "title": "Cranking the Multiplane Camera Across 7 Glass Planes",
        "featured_shot": "campfire",
        "speech_p1": "WHEN WE CRANK THE MULTIPLANE CAMERA ACROSS THE CAMPFIRE, OUR 7 GLASS PLANES SHIFT IN PARALLAX SO FLAT PAPER FEELS DEEP!",
        "caption_p2": "HOW OUR CAMERA AGENT COORDINATES FOREGROUND AND BACKGROUND:",
        "diag_steps": [("mountain", "1. FAR HILLS"), ("tree", "2. PINE WOODS"), ("campfire", "3. WARM FLAME")],
        "caption_p3": "FOUR FLICKERING 4:3 KEYFRAMES OF SHOT #042 CAMPFIRE:",
        "gutter_note": "THE RIVERBANK LOGS FRAME THE FOREGROUND WHILE THE FLAMES DANCE ON THE HERO PLANE!",
        "caption_p4": "WARM GOUACHE GLOW AND MULTIPLANE DEPTH AROUND THE CAMPFIRE!",
        "blurb": "Cel-Inker shows how the Rossark Motion Skill cranks the virtual multiplane camera across distant hills, pine woods, and the flickering campfire.",
        "stages": ["Panel 1: Cranking the Multiplane Rig", "Panel 2: Hills -> Woods -> Flame", "Panel 3: 4:3 Campfire Keyframes", "Panel 4: Glowing Night Scene"],
    },
    {
        "id": "skill-03-take-one",
        "num": 13,
        "group": "skill",
        "timestamp": "09:49",
        "lead_agent": "blueline",
        "cameo_agent": "sprocket",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 03 // STAGING TAKE ONE",
        "repo": "nerfels/take-one",
        "url": "https://github.com/nerfels/take-one",
        "category": "Pre-Roll Stage Blocking & Clapperboard Lock",
        "install": "git clone https://github.com/nerfels/take-one.git",
        "title": "Blocking the Stage Floor Before Calling 'Action!'",
        "featured_shot": "clock",
        "speech_p1": "BEFORE THE CAMERA ROLLS ON TAKE ONE, EVERY PROP, SHADOW, AND KEYFRAME MUST HIT ITS MARK ON THE STAGE FLOOR!",
        "caption_p2": "THREE CHECKS BEFORE OUR AGENT DIRECTOR SNAPS THE CLAPPER:",
        "diag_steps": [("compass", "1. BLOCK STAGE"), ("megaphone", "2. SNAP SLATE"), ("clock", "3. ROLL TAKE 1")],
        "caption_p3": "FOUR PRECISION 4:3 KEYFRAMES OF SHOT #071 CLOCK:",
        "gutter_note": "THE PENDULUM AND HANDS ADVANCE CLEANLY ACROSS ALL FOUR KEYFRAMES WITHOUT CLIPPING THE BORDER!",
        "caption_p4": "CRISP TABLETOP STAGING AND TIMING ON TAKE ONE!",
        "blurb": "Blue-Line and Sprocket demonstrate the Take-One pre-roll checklist: stage blocking, clapperboard sync, and 4-keyframe timing verification.",
        "stages": ["Panel 1: Pre-Roll Stage Blocking", "Panel 2: Block -> Slate -> Roll", "Panel 3: 4:3 Clock Keyframes", "Panel 4: Tabletop Clock Scene"],
    },
    {
        "id": "skill-04-reanimate",
        "num": 14,
        "group": "skill",
        "timestamp": "13:02",
        "lead_agent": "blueline",
        "cameo_agent": "celinker",
        "studio_wing": "bg4_clocktower",
        "wing_name": "SKILL 04 // REANIMATING ARCHIVAL VIDEO",
        "repo": "bytefer/reanimate",
        "url": "https://github.com/bytefer/reanimate",
        "category": "Archival Optical Flow -> 4-Keyframe Rotoscope",
        "install": "git clone https://github.com/bytefer/reanimate.git",
        "title": "Turning Archival Video into Living Hand-Drawn Ink",
        "featured_shot": "giraffe-walks-past",
        "speech_p1": "HOW DO WE REANIMATE RAW ARCHIVAL VIDEO INTO HAND-DRAWN INK? WE TRACE THE MOTION SPINE AND DRAW FOUR FRESH KEYFRAMES!",
        "caption_p2": "OUR 3-STEP ROTOSCOPE & MULTIPLANE REANIMATION PIPELINE:",
        "diag_steps": [("television", "1. TRACE MOTION"), ("pencil", "2. INK 4 KEYS"), ("giraffe", "3. 7-PLANE CEL")],
        "caption_p3": "FROM PENCIL TRACE (FRAME 1) TO PAINTED 4:3 KEYFRAMES (2-4):",
        "gutter_note": "WE NEVER BLUR PHOTOREAL PIXELS—WE RE-INK THE MOTION ENTIRELY IN QUICK, DRAW! LINE & GOUACHE!",
        "caption_p4": "THE ARCHIVAL GIRAFFE REBORN AS A 7-PLANE MULTIPLANE CARTOON!",
        "blurb": "Blue-Line and Cel-Inker reveal the core Reanimate workflow: extracting the motion trajectory from archival video and re-inking it as a 7-plane cartoon.",
        "stages": ["Panel 1: The Rotoscope Lightbox", "Panel 2: Trace -> Ink -> Paint", "Panel 3: 4:3 Giraffe Keyframes", "Panel 4: Reanimated 4:3 Masterpiece"],
    },
    {
        "id": "skill-05-argo",
        "num": 15,
        "group": "skill",
        "timestamp": "18:54",
        "lead_agent": "sprocket",
        "cameo_agent": "chime",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 05 // THE 16 QUALITY GATES",
        "repo": "mylin0/argo",
        "url": "https://github.com/mylin0/argo",
        "category": "Automated 16-Gate Visual Quality Assurance",
        "install": "git clone https://github.com/mylin0/argo.git",
        "title": "Guarding Every Frame Against Edge Clipping & Drift",
        "featured_shot": "hot-air-balloon",
        "speech_p1": "IF A HOT-AIR BALLOON DRIFTS OFF THE PAPER EDGE, I STOP THE PRESSES! EVERY REEL MUST PASS ALL 16 STUDIO QUALITY GATES!",
        "caption_p2": "THREE CRITICAL QUALITY GATES CHECKED BY SPROCKET:",
        "diag_steps": [("picture_frame", "1. ZERO CLIP"), ("eyeglasses", "2. 4:3 ASPECT"), ("hot_air_balloon", "3. 16/16 PASS")],
        "caption_p3": "FOUR CENTERED 4:3 KEYFRAMES OF SHOT #152 HOT-AIR BALLOON:",
        "gutter_note": "SPROCKET CONFIRMS SAFE MARGINS, TRUE 4:3 ASPECT RATIO, AND FULL SKY-TO-CLOUD PARALLAX!",
        "caption_p4": "THE HOT-AIR BALLOON RISES SMOOTHLY THROUGH THE PAINTED SKY!",
        "blurb": "Auditor Sprocket demonstrates the Argo 16-gate inspection harness, checking margin safety, true 4:3 aspect ratios, and multiplane parallax.",
        "stages": ["Panel 1: Sprocket's 16-Gate Audit", "Panel 2: Margins -> Aspect -> Pass", "Panel 3: 4:3 Balloon Keyframes", "Panel 4: Approved Sky Ascension"],
    },
    {
        "id": "skill-06-explainroo",
        "num": 16,
        "group": "skill",
        "timestamp": "23:18",
        "lead_agent": "nib",
        "cameo_agent": "celinker",
        "studio_wing": "bg5_bullpen",
        "wing_name": "SKILL 06 // VISUAL STORYTELLING CLARITY",
        "repo": "Explainum/explainroo",
        "url": "https://github.com/Explainum/explainroo",
        "category": "Amplification Through Simplification & Clear Diagrams",
        "install": "git clone https://github.com/Explainum/explainroo.git",
        "title": "Why Simple Cartoons Explain Complex Systems Best",
        "featured_shot": "lighthouse",
        "speech_p1": "NEVER BURY THE VIEWER IN FLOATING TELEMETRY NUMBERS! LET ONE CLEAR DIAGRAM AND ONE LIVING CARTOON TELL THE STORY!",
        "caption_p2": "HOW OUR STUDIO TURNS COMPLEX DATA INTO CLEAR SEQUENTIAL ART:",
        "diag_steps": [("scissors", "1. CUT CLUTTER"), ("smiley_face", "2. BOLD ICON"), ("lighthouse", "3. CLEAR STORY")],
        "caption_p3": "FOUR UNCLUTTERED 4:3 KEYFRAMES OF THE COASTAL LIGHTHOUSE:",
        "gutter_note": "BY STRIPPING AWAY TECHNICAL OVERLAYS, YOUR EYE READS THE BEAM AND WAVES INSTANTLY!",
        "caption_p4": "CLEAN PAPER, BOLD INDIA INK, AND UNOBSTRUCTED 4:3 MOTION!",
        "blurb": "Nib explains the studio's visual storytelling philosophy: stripping away noisy telemetry labels so clear icons and sequential panels communicate immediately.",
        "stages": ["Panel 1: Cutting Visual Clutter", "Panel 2: Simplify -> Icon -> Story", "Panel 3: 4:3 Lighthouse Keys", "Panel 4: Unobstructed Clarity"],
    },
    {
        "id": "skill-07-video-director",
        "num": 17,
        "group": "skill",
        "timestamp": "27:45",
        "lead_agent": "sprocket",
        "cameo_agent": "nib",
        "studio_wing": "bg1_multifloor",
        "wing_name": "SKILL 07 // UNIFIED STUDIO COLOR & RHYTHM",
        "repo": "fireflewdev/video-director-skill",
        "url": "https://github.com/fireflewdev/video-director-skill",
        "category": "Archive-Wide Gouache Palette & Rhythm Continuity",
        "install": "git clone https://github.com/fireflewdev/video-director-skill.git",
        "title": "Harmonizing Paper, Ink, and Gouache Across 346 Films",
        "featured_shot": "guitar",
        "speech_p1": "ACROSS ALL 346 SHOTS, OUR AGENT STUDIO BALANCES CREAM PAPER, INDIA INK, AND GOUACHE SO EVERY REEL FEELS UNIFIED!",
        "caption_p2": "THREE VISUAL PILLARS OF OUR UNIFIED STUDIO AESTHETIC:",
        "diag_steps": [("paintbrush", "1. WARM PAPER"), ("palette", "2. 5 GOUACHES"), ("guitar", "3. BOLD INK")],
        "caption_p3": "FOUR HARMONIZED 4:3 KEYFRAMES OF SHOT #136 GUITAR:",
        "gutter_note": "EVERY SHOT IN THE ARCHIVE SHARES THE EXACT SAME WARM PAPER AND WATERCOLOR GOUACHE PALETTE!",
        "caption_p4": "UNIFIED COLOR, LINE WEIGHT, AND 4:3 FRAMING IN EVERY REEL!",
        "blurb": "Sprocket and Nib show how the Video Director Skill enforces a single cohesive warm-paper and gouache palette across all 346 films.",
        "stages": ["Panel 1: Archive-Wide Direction", "Panel 2: Paper -> Gouache -> Ink", "Panel 3: 4:3 Guitar Keyframes", "Panel 4: Unified Studio Aesthetic"],
    },
    {
        "id": "skill-08-earshot",
        "num": 18,
        "group": "skill",
        "timestamp": "31:10",
        "lead_agent": "chime",
        "cameo_agent": "sprocket",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 08 // FOLEY SOUNDSTAGE & JERK PEAKS",
        "repo": "sphere-ai-lab/Earshot",
        "url": "https://github.com/sphere-ai-lab/Earshot",
        "category": "Motion-Triggered Foley & Soundstage Sync",
        "install": "git clone https://github.com/sphere-ai-lab/Earshot.git",
        "title": "Triggering Foley Sound Effects on Motion Peaks",
        "featured_shot": "train",
        "speech_p1": "SILENT INK COMES ALIVE WITH SOUND! I WATCH THE 3D MOTION CURVE FOR SHARP ACCELERATION PEAKS TO TRIGGER EVERY FOLEY CUE!",
        "caption_p2": "HOW CHIME SYNCS FOLEY SOUND EFFECTS TO THE ANIMATION:",
        "diag_steps": [("lightning", "1. JERK PEAK"), ("Bell", "2. FOLEY CUE"), ("train", "3. SYNCED REEL")],
        "caption_p3": "FOUR 4:3 KEYFRAMES OF SHOT #324 TRAIN WITH ACOUSTIC CUES:",
        "gutter_note": "WHENEVER THE TRAIN CHUGS OR WHISTLES AT PEAK ACCELERATION, CHIME DROPS A FRAME-ACCURATE FOLEY MARKER!",
        "caption_p4": "PICTURE AND FOLEY LOCKED TOGETHER ON THE MASTER SOUNDTRACK!",
        "blurb": "Foley Engineer 'Chime' demonstrates how Earshot detects acceleration peaks along the 3D motion curve to trigger frame-accurate Foley sound effects.",
        "stages": ["Panel 1: Chime's Foley Soundstage", "Panel 2: Jerk Peak -> Foley -> Sync", "Panel 3: 4:3 Train Keyframes", "Panel 4: Synced Sound Master"],
    },
]

_QD_CACHE = {}
_SHOT_FRAMES_CACHE = {}


def load_qd_norm(cat: str, idx: int = 0) -> list:
    cat_clean = "giraffe" if cat == "giraffe-walks-past" else cat.replace("-", "_").strip().lower()
    if cat_clean not in _QD_CACHE:
        p = QD_DIR / f"{cat_clean}.json"
        if not p.exists():
            # Fallback to a visually related category if needed
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
    """
    Draws a genuine Google Quick, Draw! vector doodle centered at (cx, cy).
    Strictly preserves 1:1 uniform aspect ratio (`side = min(w, h)`) so doodles
    never stretch or squish!
    """
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
    Loads 33 archival pencil sketches and 33 Quick, Draw! cartoon frames from `comparison.mp4`
    at their EXACT native 4:3 aspect ratio (`576x432` crop -> `480x360` 4:3 frames),
    with old baked-in top/bottom HUD bars cleanly removed!
    """
    if slug in _SHOT_FRAMES_CACHE:
        return _SHOT_FRAMES_CACHE[slug]
    comp_path = SHOTS_DIR / slug / "comparison.mp4"
    arch_sketches, cart_frames = [], []
    if comp_path.exists():
        cap = cv2.VideoCapture(str(comp_path))
        while len(cart_frames) < 33:
            ok, fr = cap.read()
            if not ok:
                break
            fh, fw = fr.shape[:2]  # 480 x 1280 (each half is 640x480 = 4:3)
            half_w = fw // 2       # 640
            # Crop 30px top and 26px bottom -> height = 424px.
            # For exact 4:3 aspect ratio, width = 424 * 4 / 3 = 565px!
            y_top = 30
            y_bot = fh - 26        # 454 (height = 424)
            crop_h = y_bot - y_top
            crop_w = int(round(crop_h * 4.0 / 3.0))  # 565
            x_pad = max(0, (half_w - crop_w) // 2)   # 37

            crop_a = fr[y_top:y_bot, x_pad:x_pad + crop_w]
            crop_c = fr[y_top:y_bot, half_w + x_pad:half_w + x_pad + crop_w]
            # Resize to 480x360 (EXACT 4:3 aspect ratio!)
            raw_a = cv2.resize(crop_a, (480, 360), interpolation=cv2.INTER_AREA)
            raw_c = cv2.resize(crop_c, (480, 360), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(raw_a, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 45, 125)
            adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
            ink = cv2.bitwise_or(edges, adapt)
            sk = np.full((360, 480, 3), BGR_PAPER, dtype=np.uint8)
            sk[blur < 115] = BGR_PAPER_DARK
            sk[ink > 0] = BGR_INK
            arch_sketches.append(sk)
            cart_frames.append(raw_c)
        cap.release()
    if not cart_frames:
        blank = np.full((360, 480, 3), BGR_PAPER, dtype=np.uint8)
        draw_qd_doodle(blank, slug, 240, 180, 200, 200, fill_bgr=BGR_AMBER_PALE, thickness=4)
        arch_sketches = [blank.copy() for _ in range(33)]
        cart_frames = [blank.copy() for _ in range(33)]
    while len(cart_frames) < 33:
        arch_sketches.append(arch_sketches[-1])
        cart_frames.append(cart_frames[-1])
    _SHOT_FRAMES_CACHE[slug] = (arch_sketches, cart_frames)
    return arch_sketches, cart_frames


# =============================================================================
# ORIGINAL CHARACTER ART: THE 5 AGENT ANIMATORS OF SECONDHAND STUDIOS
# =============================================================================
def draw_agent_animator(
    img: np.ndarray,
    agent_id: str,
    cx: int,
    cy: int,
    scale: float = 1.0,
    pose: int = 0,
    dark_bg: bool = False,
):
    """
    Draws one of our 5 bespoke Agent Animator characters in a classic 1930s
    Fleischer / Inkwell-Toon + Quick, Draw! aesthetic (with pie-cut cartoon eyes,
    white four-fingered animator gloves, studio apron/smock, and iconic headgear):
      - `nib`: Gold-and-Ink Fountain-Pen Nib Crown + Terracotta Director Apron + Stylus Baton
      - `blueline`: Prussian Blue Draftsman Visor + Ear Pencil + 4-Key Pegbar Slate
      - `celinker`: Ochre Beret + Flip-Down Magnifier Loupe + Gouache Brush & Palette
      - `sprocket`: Twin-Reel 35mm Projector Cap + Brass Monocle + 16-Gate Stamp
      - `chime`: Studio Headphones + Clockwork Gear Crest + Brass Tuning Fork & Stopwatch
    """
    ag = STUDIO_AGENTS.get(agent_id, STUDIO_AGENTS["nib"])
    col_bgr = ag["color_bgr"]
    pale_bgr = ag["pale_bgr"]
    outline_bgr = BGR_GUTTER if dark_bg else BGR_INK

    def S(val: float) -> int:
        return int(round(val * scale))

    # 1. Cartoon Studio Smock / Apron Torso
    tx0, tx1 = cx - S(48), cx + S(48)
    ty0, ty1 = cy + S(26), cy + S(112)
    # White rolled-sleeve animator shirt underneath
    shirt_pts = np.array([
        [cx - S(42), ty0],
        [cx + S(42), ty0],
        [tx1 + S(4), ty1],
        [tx0 - S(4), ty1],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, shirt_pts, BGR_GUTTER, cv2.LINE_AA)
    cv2.polylines(img, [shirt_pts], True, BGR_INK, max(2, S(3)), cv2.LINE_AA)

    # Colored Studio Apron / Vest over the shirt
    apron_pts = np.array([
        [cx - S(32), ty0 + S(4)],
        [cx + S(32), ty0 + S(4)],
        [tx1 - S(4), ty1],
        [tx0 + S(4), ty1],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, apron_pts, col_bgr, cv2.LINE_AA)
    cv2.polylines(img, [apron_pts], True, BGR_INK, max(2, S(3)), cv2.LINE_AA)

    # Apron shoulder straps & chest badge
    cv2.line(img, (cx - S(22), ty0), (cx - S(18), ty0 + S(26)), BGR_INK, max(2, S(3)), cv2.LINE_AA)
    cv2.line(img, (cx + S(22), ty0), (cx + S(18), ty0 + S(26)), BGR_INK, max(2, S(3)), cv2.LINE_AA)
    cv2.rectangle(img, (cx - S(18), ty0 + S(26)), (cx + S(18), ty0 + S(56)), pale_bgr, -1)
    cv2.rectangle(img, (cx - S(18), ty0 + S(26)), (cx + S(18), ty0 + S(56)), BGR_INK, max(2, S(2)))

    # Chest pocket emblem by role
    if agent_id == "nib":
        # 35mm film reel emblem
        cv2.circle(img, (cx, ty0 + S(41)), S(10), BGR_AMBER, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, ty0 + S(41)), S(10), BGR_INK, max(1, S(2)), cv2.LINE_AA)
        cv2.circle(img, (cx, ty0 + S(41)), S(3), BGR_INK, -1, cv2.LINE_AA)
    elif agent_id == "blueline":
        # 4-key registration peg marks
        for kx in (-10, -3, 4, 11):
            cv2.line(img, (cx + S(kx), ty0 + S(32)), (cx + S(kx), ty0 + S(50)), BGR_BLUE, max(1, S(2)))
    elif agent_id == "celinker":
        # 3 gouache color dots
        for dx, dc in [(-9, BGR_RED), (0, BGR_AMBER), (9, BGR_GREEN)]:
            cv2.circle(img, (cx + S(dx), ty0 + S(41)), S(4), dc, -1, cv2.LINE_AA)
            cv2.circle(img, (cx + S(dx), ty0 + S(41)), S(4), BGR_INK, 1, cv2.LINE_AA)
    elif agent_id == "sprocket":
        # Checkmark gate emblem
        chk = np.array([[cx - S(8), ty0 + S(41)], [cx - S(2), ty0 + S(48)], [cx + S(10), ty0 + S(33)]], dtype=np.int32)
        cv2.polylines(img, [chk], False, BGR_GREEN, max(2, S(3)), cv2.LINE_AA)
    else:
        # 55m stopwatch dial
        cv2.circle(img, (cx, ty0 + S(41)), S(10), BGR_GUTTER, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, ty0 + S(41)), S(10), BGR_INK, max(1, S(2)), cv2.LINE_AA)
        cv2.line(img, (cx, ty0 + S(41)), (cx + S(5), ty0 + S(35)), BGR_RED, max(1, S(2)), cv2.LINE_AA)

    # 2. Expressive Gloved Cartoon Arm & Role Tool Prop
    arm_lift = S(-8) if (pose % 2 == 1) else 0
    hx = cx + S(64)
    hy = cy + S(32) + arm_lift
    cv2.line(img, (cx + S(36), ty0 + S(18)), (hx, hy), BGR_GUTTER, max(6, S(11)), cv2.LINE_AA)
    cv2.line(img, (cx + S(36), ty0 + S(18)), (hx, hy), BGR_INK, max(2, S(3)), cv2.LINE_AA)
    # Classic 4-finger white cartoon glove
    cv2.circle(img, (hx + S(4), hy - S(2)), S(11), BGR_GUTTER, -1, cv2.LINE_AA)
    cv2.circle(img, (hx + S(4), hy - S(2)), S(11), BGR_INK, max(2, S(2)), cv2.LINE_AA)

    # Prop in hand
    if agent_id == "nib":
        # Fountain-Tip Director's Baton
        cv2.line(img, (hx + S(6), hy - S(4)), (hx + S(32), hy - S(34)), BGR_INK, max(2, S(4)), cv2.LINE_AA)
        nib_tip = np.array([
            [hx + S(28), hy - S(30)],
            [hx + S(40), hy - S(44)],
            [hx + S(34), hy - S(26)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, nib_tip, BGR_AMBER, cv2.LINE_AA)
        cv2.polylines(img, [nib_tip], True, BGR_INK, max(1, S(2)), cv2.LINE_AA)
    elif agent_id == "blueline":
        # Blue Non-Photo Animator's Pencil
        cv2.line(img, (hx + S(4), hy - S(4)), (hx + S(30), hy - S(32)), BGR_BLUE, max(3, S(5)), cv2.LINE_AA)
        cv2.circle(img, (hx + S(32), hy - S(34)), S(3), BGR_RED, -1, cv2.LINE_AA)
    elif agent_id == "celinker":
        # Gouache Paintbrush with amber paint tip
        cv2.line(img, (hx + S(4), hy - S(4)), (hx + S(28), hy - S(30)), BGR_INK, max(2, S(4)), cv2.LINE_AA)
        cv2.circle(img, (hx + S(31), hy - S(33)), S(7), BGR_AMBER, -1, cv2.LINE_AA)
        cv2.circle(img, (hx + S(31), hy - S(33)), S(7), BGR_INK, max(1, S(2)), cv2.LINE_AA)
    elif agent_id == "sprocket":
        # 16-Gate Auditor Stamp
        cv2.rectangle(img, (hx + S(8), hy - S(24)), (hx + S(30), hy - S(10)), BGR_GREEN, -1)
        cv2.rectangle(img, (hx + S(8), hy - S(24)), (hx + S(30), hy - S(10)), BGR_INK, max(1, S(2)))
    else:
        # Brass Tuning Fork
        cv2.line(img, (hx + S(6), hy - S(4)), (hx + S(18), hy - S(18)), BGR_AMBER, max(2, S(3)), cv2.LINE_AA)
        cv2.line(img, (hx + S(14), hy - S(16)), (hx + S(24), hy - S(30)), BGR_AMBER, max(2, S(2)), cv2.LINE_AA)
        cv2.line(img, (hx + S(20), hy - S(12)), (hx + S(30), hy - S(26)), BGR_AMBER, max(2, S(2)), cv2.LINE_AA)

    # 3. Cartoon Neck & Round Inkwell-Toon Head
    cv2.rectangle(img, (cx - S(9), cy + S(14)), (cx + S(9), cy + S(28)), BGR_GUTTER, -1)
    cv2.rectangle(img, (cx - S(9), cy + S(14)), (cx + S(9), cy + S(28)), BGR_INK, max(2, S(2)))

    cv2.ellipse(img, (cx, cy - S(6)), (S(30), S(32)), 0, 0, 360, BGR_GUTTER, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy - S(6)), (S(30), S(32)), 0, 0, 360, BGR_INK, max(2, S(3)), cv2.LINE_AA)

    # 4. Classic 1930s Pie-Cut Cartoon Eyes & Expressive Smile
    for ex in (cx - S(11), cx + S(12)):
        ey = cy - S(7)
        # Outer eye oval
        cv2.ellipse(img, (ex, ey), (S(7), S(9)), 0, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.ellipse(img, (ex, ey), (S(7), S(9)), 0, 0, 360, BGR_INK, max(1, S(2)), cv2.LINE_AA)
        # Ink pupil with classic vintage white pie-cut wedge
        cv2.ellipse(img, (ex + S(1), ey), (S(4), S(6)), 0, 0, 360, BGR_INK, -1, cv2.LINE_AA)
        pie = np.array([[ex + S(1), ey], [ex + S(6), ey - S(4)], [ex + S(6), ey - S(1)]], dtype=np.int32)
        cv2.fillConvexPoly(img, pie, (255, 255, 255), cv2.LINE_AA)

    # Cartoon nose button & warm grin
    cv2.ellipse(img, (cx + S(1), cy + S(3)), (S(4), S(3)), 0, 0, 360, col_bgr, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx + S(1), cy + S(3)), (S(4), S(3)), 0, 0, 360, BGR_INK, max(1, S(2)), cv2.LINE_AA)
    cv2.ellipse(img, (cx + S(1), cy + S(13)), (S(10), S(6)), 0, 10, 170, BGR_INK, max(2, S(3)), cv2.LINE_AA)

    # 5. Iconic Agent Animator Headgear / Silhouette
    if agent_id == "nib":
        # Sculpted Gold-and-Ink Fountain Pen Nib Crown rising from head!
        band = np.array([
            [cx - S(31), cy - S(24)],
            [cx + S(31), cy - S(24)],
            [cx + S(27), cy - S(34)],
            [cx - S(27), cy - S(34)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, band, BGR_RED, cv2.LINE_AA)
        cv2.polylines(img, [band], True, outline_bgr, max(2, S(2)), cv2.LINE_AA)

        nib_crown = np.array([
            [cx - S(22), cy - S(34)],
            [cx - S(26), cy - S(52)],
            [cx, cy - S(76)],
            [cx + S(26), cy - S(52)],
            [cx + S(22), cy - S(34)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, nib_crown, BGR_AMBER, cv2.LINE_AA)
        cv2.polylines(img, [nib_crown], True, outline_bgr, max(2, S(3)), cv2.LINE_AA)
        # Fountain-pen slit & breather hole
        cv2.line(img, (cx, cy - S(75)), (cx, cy - S(48)), BGR_INK, max(2, S(3)), cv2.LINE_AA)
        cv2.circle(img, (cx, cy - S(48)), S(4), BGR_INK, -1, cv2.LINE_AA)

    elif agent_id == "blueline":
        # Prussian Blue Animator's Eye-Shade Visor + Tucked Non-Photo Blue Pencil
        visor = np.array([
            [cx - S(32), cy - S(22)],
            [cx + S(38), cy - S(18)],
            [cx + S(28), cy - S(34)],
            [cx - S(28), cy - S(34)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, visor, BGR_BLUE, cv2.LINE_AA)
        cv2.polylines(img, [visor], True, outline_bgr, max(2, S(3)), cv2.LINE_AA)
        # Crown dome
        cv2.ellipse(img, (cx, cy - S(32)), (S(25), S(14)), 0, 180, 360, BGR_BLUE_PALE, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy - S(32)), (S(25), S(14)), 0, 180, 360, outline_bgr, max(2, S(2)), cv2.LINE_AA)
        # Pencil behind ear
        cv2.line(img, (cx - S(34), cy - S(10)), (cx - S(18), cy - S(42)), BGR_AMBER, max(2, S(4)), cv2.LINE_AA)

    elif agent_id == "celinker":
        # Artist's Ochre Beret + Flip-Down Magnifier Loupe over right eye
        beret = np.array([
            [cx - S(32), cy - S(24)],
            [cx - S(38), cy - S(42)],
            [cx - S(6), cy - S(50)],
            [cx + S(34), cy - S(38)],
            [cx + S(28), cy - S(24)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, beret, BGR_AMBER, cv2.LINE_AA)
        cv2.polylines(img, [beret], True, outline_bgr, max(2, S(3)), cv2.LINE_AA)
        # Flip-down Magnifier Loupe
        cv2.circle(img, (cx + S(13), cy - S(7)), S(12), BGR_BLUE_PALE, 2, cv2.LINE_AA)
        cv2.circle(img, (cx + S(13), cy - S(7)), S(12), BGR_INK, max(2, S(2)), cv2.LINE_AA)
        cv2.line(img, (cx + S(22), cy - S(14)), (cx + S(26), cy - S(26)), BGR_INK, max(2, S(2)), cv2.LINE_AA)

    elif agent_id == "sprocket":
        # Twin-Reel 35mm Projector Cap + Brass Monocle
        for rx in (cx - S(18), cx + S(18)):
            ry = cy - S(44)
            cv2.circle(img, (rx, ry), S(14), BGR_PAPER_DARK, -1, cv2.LINE_AA)
            cv2.circle(img, (rx, ry), S(14), outline_bgr, max(2, S(3)), cv2.LINE_AA)
            cv2.circle(img, (rx, ry), S(4), BGR_INK, -1, cv2.LINE_AA)
        cap_band = np.array([
            [cx - S(28), cy - S(24)],
            [cx + S(28), cy - S(24)],
            [cx + S(24), cy - S(36)],
            [cx - S(24), cy - S(36)],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, cap_band, BGR_GREEN, cv2.LINE_AA)
        cv2.polylines(img, [cap_band], True, outline_bgr, max(2, S(2)), cv2.LINE_AA)
        # Auditor Monocle
        cv2.circle(img, (cx - S(11), cy - S(7)), S(12), BGR_AMBER, max(2, S(2)), cv2.LINE_AA)

    else:
        # Chime: Oversized Studio Headphones + Clockwork Antenna
        cv2.ellipse(img, (cx, cy - S(10)), (S(35), S(32)), 0, 180, 360, outline_bgr, max(3, S(4)), cv2.LINE_AA)
        for hx_cup in (cx - S(33), cx + S(33)):
            cv2.ellipse(img, (hx_cup, cy - S(4)), (S(8), S(14)), 0, 0, 360, BGR_RED, -1, cv2.LINE_AA)
            cv2.ellipse(img, (hx_cup, cy - S(4)), (S(8), S(14)), 0, 0, 360, outline_bgr, max(2, S(2)), cv2.LINE_AA)
        cv2.line(img, (cx, cy - S(42)), (cx, cy - S(58)), outline_bgr, max(2, S(3)), cv2.LINE_AA)
        cv2.circle(img, (cx, cy - S(60)), S(6), BGR_AMBER, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - S(60)), S(6), outline_bgr, max(1, S(2)), cv2.LINE_AA)


def draw_panel1_studio_environment(canvas: np.ndarray, spec: dict, boil_phase: int = 0):
    """
    Draws the Agent Animation Studio apparatus on the right half of Panel 1 (x: 470..920, y: 185..505),
    complete with a Co-Worker Agent Animator (`cameo_agent`) actively operating the studio machine!
    All embedded doodles and frames preserve exact 1:1 or 4:3 aspect ratios.
    """
    wing = spec["studio_wing"]
    slug = spec["featured_shot"]
    cameo_id = spec.get("cameo_agent", "celinker")
    dark_p1 = (spec["num"] % 2 == 1)

    if wing == "bg1_multifloor":
        # 3-Tier Multiplane Camera Stand (x: 495..815, y: 195..500) + Cameo Agent operating crank on right
        cv2.line(canvas, (510, 195), (510, 505), BGR_GUTTER if dark_p1 else BGR_INK, 6, cv2.LINE_AA)
        cv2.line(canvas, (790, 195), (790, 505), BGR_GUTTER if dark_p1 else BGR_INK, 6, cv2.LINE_AA)
        draw_qd_doodle(canvas, "camera", 650, 222, 64, 64, fill_bgr=BGR_AMBER_PALE, thickness=3, boil_phase=boil_phase)
        plane_specs = [
            (295, BGR_CORAL_PALE, slug),
            (375, BGR_MINT_PALE, "tree"),
            (455, BGR_BLUE_PALE, "cloud"),
        ]
        for py, pcol, pcat in plane_specs:
            pts = np.array([[495, py + 12], [530, py - 14], [805, py - 14], [770, py + 12]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts + np.array([3, 3]), BGR_INK, cv2.LINE_AA)
            cv2.fillConvexPoly(canvas, pts, pcol, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, BGR_INK, 3, cv2.LINE_AA)
            draw_qd_doodle(canvas, pcat, 650, py - 2, 46, 46, fill_bgr=BGR_GUTTER, thickness=2, boil_phase=boil_phase)
        # Co-worker Agent Animator operating the multiplane rig on the right!
        draw_agent_animator(canvas, cameo_id, 868, 405, scale=0.76, pose=boil_phase, dark_bg=dark_p1)

    elif wing in ("bg5_bullpen", "bg2_occupied"):
        # Storyboard Pin-Up Wall + Tilted Lightbox Desk + Co-worker Agent Animator
        cv2.rectangle(canvas, (545, 192), (835, 302), BGR_INK, -1)
        cv2.rectangle(canvas, (540, 187), (830, 297), BGR_PAPER_DARK, -1)
        cv2.rectangle(canvas, (540, 187), (830, 297), BGR_INK, 3)
        for ci, ccat in enumerate(["picture_frame", slug, "star"]):
            cx_c = 592 + ci * 92
            cv2.rectangle(canvas, (cx_c - 36, 200), (cx_c + 36, 284), BGR_GUTTER, -1)
            cv2.rectangle(canvas, (cx_c - 36, 200), (cx_c + 36, 284), BGR_INK, 2)
            draw_qd_doodle(canvas, ccat, cx_c, 242, 52, 52, fill_bgr=BGR_AMBER_PALE, thickness=2, boil_phase=boil_phase)

        desk_pts = np.array([[475, 430], [785, 365], [835, 450], [525, 505]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, desk_pts + np.array([4, 4]), BGR_INK, cv2.LINE_AA)
        cv2.fillConvexPoly(canvas, desk_pts, BGR_AMBER_PALE, cv2.LINE_AA)
        cv2.polylines(canvas, [desk_pts], True, BGR_INK, 4, cv2.LINE_AA)
        cv2.ellipse(canvas, (655, 434), (72, 40), -10, 0, 360, BGR_GUTTER, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (655, 434), (72, 40), -10, 0, 360, BGR_INK, 3, cv2.LINE_AA)
        draw_qd_doodle(canvas, slug, 655, 432, 66, 66, fill_bgr=None, thickness=3, boil_phase=boil_phase)
        draw_agent_animator(canvas, cameo_id, 872, 402, scale=0.76, pose=boil_phase, dark_bg=dark_p1)

    elif wing == "bg3_screening":
        # 35mm Dailies Film Projector casting beam onto EXACT 4:3 Theater Screen (`240x180`) + Co-worker Agent
        cv2.circle(canvas, (485, 338), 24, BGR_PAPER_DARK, -1, cv2.LINE_AA)
        cv2.circle(canvas, (485, 338), 24, BGR_INK, 3, cv2.LINE_AA)
        cv2.circle(canvas, (538, 338), 24, BGR_PAPER_DARK, -1, cv2.LINE_AA)
        cv2.circle(canvas, (538, 338), 24, BGR_INK, 3, cv2.LINE_AA)
        cv2.rectangle(canvas, (476, 364), (550, 428), BGR_BLUE_PALE, -1)
        cv2.rectangle(canvas, (476, 364), (550, 428), BGR_INK, 3)

        # Exact 4:3 Dailies Screen (`240x180`, 240/180 = 4:3!)
        sx0, sy0 = 600, 218
        sw, sh = 224, 168  # 224 / 168 = 4:3 exact!
        beam_pts = np.array([[552, 382], [sx0, sy0], [sx0, sy0 + sh], [552, 402]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, beam_pts, BGR_AMBER_PALE, cv2.LINE_AA)
        cv2.rectangle(canvas, (sx0 + 4, sy0 + 4), (sx0 + sw + 4, sy0 + sh + 4), BGR_INK, -1)
        _, cart_frs = load_shot_frames(slug)
        scr_thumb = cv2.resize(cart_frs[16], (sw, sh), interpolation=cv2.INTER_AREA)
        canvas[sy0:sy0 + sh, sx0:sx0 + sw] = scr_thumb
        cv2.rectangle(canvas, (sx0, sy0), (sx0 + sw, sy0 + sh), BGR_INK, 4)
        draw_agent_animator(canvas, cameo_id, 875, 405, scale=0.76, pose=boil_phase, dark_bg=dark_p1)

    else:
        # 55-Minute Clocktower & 3 Exact 4:3 Keyframe Cards (`96x72` each = 4:3!) + Co-worker Agent
        draw_qd_doodle(canvas, "clock", 522, 265, 86, 86, fill_bgr=BGR_AMBER_PALE, thickness=3, boil_phase=boil_phase)
        _, cart_frs = load_shot_frames(slug)
        for fi, f_idx in enumerate([0, 16, 32]):
            fx0 = 590 + fi * 76
            fy0 = 225 + fi * 32
            fw_c, fh_c = 96, 72  # 96 / 72 = 4:3 exact!
            cv2.rectangle(canvas, (fx0 + 4, fy0 + 4), (fx0 + fw_c + 4, fy0 + fh_c + 4), BGR_INK, -1)
            k_thumb = cv2.resize(cart_frs[f_idx], (fw_c, fh_c), interpolation=cv2.INTER_AREA)
            canvas[fy0:fy0 + fh_c, fx0:fx0 + fw_c] = k_thumb
            cv2.rectangle(canvas, (fx0, fy0), (fx0 + fw_c, fy0 + fh_c), BGR_INK, 3)
        draw_agent_animator(canvas, cameo_id, 872, 405, scale=0.76, pose=boil_phase, dark_bg=dark_p1)


def render_single_panel_fullscreen(
    spec: dict,
    panel_num: int,
    sub_step: float = 1.0,
    shot_frame_idx: int = 16,
    boil_phase: int = 0,
) -> np.ndarray:
    """
    Renders ONE single panel (`panel_num` in 1..4) at FULL `960x540` (16:9) resolution.
    Every embedded shot keyframe, archival sketch, and cartoon projection inside Panels 1, 3, and 4
    is locked to its exact `4:3` (`1.3333:1`) aspect ratio, and every Quick, Draw! doodle is locked to `1:1`.
    """
    W, H = 960, 540
    canvas = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)
    slug = spec["featured_shot"]
    lead_id = spec.get("lead_agent", "nib")
    cameo_id = spec.get("cameo_agent", "celinker")
    lead_ag = STUDIO_AGENTS[lead_id]
    cameo_ag = STUDIO_AGENTS[cameo_id]
    arch_sketches, cart_frames = load_shot_frames(slug)

    pad = 16
    cap_h = 54
    bx0, by0, bx1, by1 = pad, pad, W - pad, H - pad

    if panel_num == 1:
        dark_p1 = (spec["num"] % 2 == 1)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_INK if dark_p1 else BGR_PAPER, -1)
        if dark_p1:
            rng_s = np.random.RandomState(spec["num"] * 31)
            for _ in range(24):
                sx = int(rng_s.randint(bx0 + 20, bx1 - 20))
                sy = int(rng_s.randint(by0 + 70, by1 - 20))
                cv2.circle(canvas, (sx, sy), 2, BGR_GUTTER, -1)

        # Draw Studio Apparatus + Co-Worker Cameo Agent on the right
        draw_panel1_studio_environment(canvas, spec, boil_phase=boil_phase)

        # Draw Lead Agent Animator on the bottom-left (stopping cleanly inside panel border)
        pres_pose = (boil_phase // 2) % 2
        draw_agent_animator(canvas, lead_id, bx0 + 158, by1 - 150, scale=1.22, pose=pres_pose, dark_bg=dark_p1)

    elif panel_num == 2:
        # 3-Step Agent Animator Workflow (`[1] --> [2] --> [3]`, 1:1 square doodle icons)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        steps = spec["diag_steps"]
        n_s = len(steps)
        vis_steps = max(1, min(n_s, int(sub_step * n_s) + 1)) if sub_step < 0.99 else n_s
        g_fills = [BGR_BLUE_PALE, BGR_AMBER_PALE, BGR_MINT_PALE]
        box_w, box_h = 216, 216  # 1:1 square cards!
        gap = 68
        total_w = n_s * box_w + (n_s - 1) * gap
        start_x = (W - total_w) // 2
        cy_box = by0 + cap_h + (by1 - by0 - cap_h) // 2 - 18

        for s_i in range(vis_steps):
            cat_name, _ = steps[s_i]
            sx0 = start_x + s_i * (box_w + gap)
            sy0 = cy_box - box_h // 2
            sx1, sy1 = sx0 + box_w, sy0 + box_h
            cv2.rectangle(canvas, (sx0 + 5, sy0 + 5), (sx1 + 5, sy1 + 5), BGR_INK, -1)
            cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), BGR_GUTTER, -1)
            cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), BGR_INK, 4)
            draw_qd_doodle(
                canvas, cat_name,
                cx=sx0 + box_w // 2, cy=sy0 + box_h // 2,
                w=148, h=148,
                fill_bgr=g_fills[s_i % len(g_fills)], ink_bgr=BGR_INK, thickness=4,
                boil_phase=boil_phase,
            )
            if s_i < vis_steps - 1:
                ax0 = sx1 + 12
                ax1 = sx1 + gap - 12
                ay = cy_box
                cv2.line(canvas, (ax0, ay), (ax1, ay), BGR_INK, 6, cv2.LINE_AA)
                cv2.polylines(
                    canvas,
                    [np.array([[ax1 - 14, ay - 12], [ax1 + 2, ay], [ax1 - 14, ay + 12]], dtype=np.int32)],
                    False, BGR_INK, 6, cv2.LINE_AA,
                )

    elif panel_num == 3:
        # 35mm 4-Keyframe Filmstrip with EXACT 4:3 Aspect Ratio (`kw=204, kh=153` -> 204/153 = 4:3!)
        # + Agent Animator Exposure-Sheet Note Card below!
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        kw, kh = 204, 153  # EXACT 4:3 aspect ratio!
        strip_y0 = by0 + cap_h + 18
        strip_y1 = strip_y0 + kh + 82  # 235px tall 35mm filmstrip
        cv2.rectangle(canvas, (bx0 + 20, strip_y0), (bx1 - 20, strip_y1), BGR_INK, -1)
        for sx in range(bx0 + 34, bx1 - 30, 28):
            cv2.rectangle(canvas, (sx, strip_y0 + 8), (sx + 14, strip_y0 + 19), BGR_GUTTER, -1)
            cv2.rectangle(canvas, (sx, strip_y1 - 19), (sx + 14, strip_y1 - 8), BGR_GUTTER, -1)

        key_indices = [0, 10, 21, 32]
        active_k = min(3, int(sub_step * 4)) if sub_step < 0.99 else -1
        for k_i, f_idx in enumerate(key_indices):
            kx0 = bx0 + 35 + k_i * (kw + 14)
            ky0 = strip_y0 + 52
            src_fr = arch_sketches[f_idx] if k_i == 0 else cart_frames[f_idx]
            thumb = cv2.resize(src_fr, (kw, kh), interpolation=cv2.INTER_AREA)  # 480x360 -> 204x153 (EXACT 4:3!)
            canvas[ky0:ky0 + kh, kx0:kx0 + kw] = thumb
            is_hi = (k_i == active_k or active_k == -1)
            cv2.rectangle(canvas, (kx0, ky0), (kx0 + kw, ky0 + kh), BGR_AMBER if is_hi else BGR_GUTTER, 4 if is_hi else 2)

        # Bottom Exposure-Sheet Note Card + Mini Agent Animator Cameo
        nx0, ny0 = bx0 + 20, strip_y1 + 14
        nx1, ny1 = bx1 - 20, by1 - 14
        cv2.rectangle(canvas, (nx0 + 4, ny0 + 4), (nx1 + 4, ny1 + 4), BGR_INK, -1)
        cv2.rectangle(canvas, (nx0, ny0), (nx1, ny1), BGR_GUTTER, -1)
        cv2.rectangle(canvas, (nx0, ny0), (nx1, ny1), BGR_INK, 3)
        draw_agent_animator(canvas, lead_id, nx0 + 50, ny0 + 56, scale=0.46, pose=0, dark_bg=False)

    elif panel_num == 4:
        # Left Crew & Archival Reference Column (`280x210` EXACT 4:3!) +
        # Right Hero 7-Plane Multiplane Cartoon (`560x420` EXACT 4:3!)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)

        # Right: Hero 7-Plane Cartoon at EXACT 4:3 (`560x420`)
        vw, vh = 560, 420  # 560 / 420 = 4:3 exact!
        vx0 = bx1 - vw - 18
        vy0 = by0 + cap_h + 16
        cur_fr = cart_frames[shot_frame_idx % len(cart_frames)]
        big_cart = cv2.resize(cur_fr, (vw, vh), interpolation=cv2.INTER_LINEAR)  # 480x360 -> 560x420 (EXACT 4:3!)
        cv2.rectangle(canvas, (vx0 + 5, vy0 + 5), (vx0 + vw + 5, vy0 + vh + 5), BGR_INK, -1)
        canvas[vy0:vy0 + vh, vx0:vx0 + vw] = big_cart
        cv2.rectangle(canvas, (vx0, vy0), (vx0 + vw, vy0 + vh), BGR_INK, 4)

        # Left Top: Archival Pencil Sketch at EXACT 4:3 (`280x210`)
        aw, ah = 280, 210  # 280 / 210 = 4:3 exact!
        ax0 = bx0 + 20
        ay0 = vy0 + 32
        cur_arch = arch_sketches[shot_frame_idx % len(arch_sketches)]
        arch_thumb = cv2.resize(cur_arch, (aw, ah), interpolation=cv2.INTER_AREA)  # 480x360 -> 280x210 (EXACT 4:3!)
        cv2.rectangle(canvas, (ax0 + 4, ay0 + 4), (ax0 + aw + 4, ay0 + ah + 4), BGR_INK, -1)
        canvas[ay0:ay0 + ah, ax0:ax0 + aw] = arch_thumb
        cv2.rectangle(canvas, (ax0, ay0), (ax0 + aw, ay0 + ah), BGR_INK, 3)

        # Left Bottom: Agent Animator Crew Sign-Off Card (`280x162`)
        cx0, cy0 = ax0, ay0 + ah + 16
        cx1, cy1 = ax0 + aw, vy0 + vh
        cv2.rectangle(canvas, (cx0 + 4, cy0 + 4), (cx1 + 4, cy1 + 4), BGR_INK, -1)
        cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), BGR_GUTTER, -1)
        cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), BGR_INK, 3)
        draw_agent_animator(canvas, lead_id, cx0 + 52, cy0 + 86, scale=0.62, pose=boil_phase % 2, dark_bg=False)

    # =========================================================================
    # PIL HIGH-RESOLUTION COMIC LETTERING (0% TRUNCATION GUARANTEED)
    # =========================================================================
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    if panel_num == 1:
        draw.rectangle([bx0, by0, bx1, by0 + cap_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
        draw.rectangle([bx0, by0, bx0 + 54, by0 + cap_h], fill=lead_ag["color_rgb"], outline=(22, 20, 18), width=4)
        draw.text((bx0 + 18, by0 + 10), "1", font=FONT_DISPLAY_XL, fill=(255, 251, 244))
        hdr_txt = f"{spec['wing_name']} — {spec['title'].upper()}"
        f_hdr = fit_text_font(draw, hdr_txt, (bx1 - bx0) - 74, [FONT_DISPLAY_LG, FONT_DISPLAY_MD, FONT_CAPTION_MD])
        draw.text((bx0 + 68, by0 + 14), hdr_txt, font=f_hdr, fill=(22, 20, 18))

        # Large Oval Speech Balloon above the Lead Agent Animator (0% line cut-off!)
        s_box = (bx0 + 30, by0 + cap_h + 14, bx0 + 548, by0 + cap_h + 158)
        tx, ty = bx0 + 172, by1 - 216
        bx_mid = (s_box[0] + s_box[2]) // 2
        tail_pts = [(bx_mid - 18, s_box[3] - 6), (bx_mid + 12, s_box[3] - 6), (tx, ty)]
        draw.polygon([(p[0] + 4, p[1] + 4) for p in tail_pts], fill=(22, 20, 18))
        draw.rounded_rectangle([s_box[0] + 4, s_box[1] + 4, s_box[2] + 4, s_box[3] + 4], radius=28, fill=(22, 20, 18))
        draw.polygon(tail_pts, fill=(255, 255, 255), outline=(22, 20, 18))
        draw.rounded_rectangle(s_box, radius=28, fill=(255, 255, 255), outline=(22, 20, 18), width=4)
        draw.polygon([(bx_mid - 15, s_box[3] - 8), (bx_mid + 9, s_box[3] - 8), (bx_mid - 3, s_box[3] - 1)], fill=(255, 255, 255))

        lines = textwrap.wrap(spec["speech_p1"].upper(), width=34)[:4]
        lh = 25 if len(lines) == 4 else 28
        sy_txt = s_box[1] + ((s_box[3] - s_box[1]) - len(lines) * lh) // 2 - 2
        for i, ln in enumerate(lines):
            f_sp = fit_text_font(draw, ln, (s_box[2] - s_box[0]) - 28, [FONT_SPEECH_MD if len(lines) == 4 else FONT_SPEECH_LG, FONT_SPEECH_MD, FONT_CAPTION_MD])
            bb = draw.textbbox((0, 0), ln, font=f_sp)
            tw = bb[2] - bb[0]
            draw.text((s_box[0] + ((s_box[2] - s_box[0]) - tw) // 2, sy_txt + i * lh), ln, font=f_sp, fill=(22, 20, 18))

        # Studio Role Pill below the Lead Agent Animator & Cameo Agent Animator
        lead_pill = f"{lead_ag['short_name']} // {lead_ag['dept']}"
        draw.rounded_rectangle([bx0 + 28, by1 - 38, bx0 + 318, by1 - 10], radius=6, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        f_lp = fit_text_font(draw, lead_pill, 276, [FONT_CAPTION_MD, FONT_CAPTION_SM])
        draw.text((bx0 + 38, by1 - 33), lead_pill, font=f_lp, fill=(22, 20, 18))

        cameo_pill = f"CREW: {cameo_ag['short_name']}"
        draw.rounded_rectangle([W - 188, by1 - 38, W - 26, by1 - 10], radius=6, fill=(214, 228, 240), outline=(22, 20, 18), width=2)
        f_cp = fit_text_font(draw, cameo_pill, 148, [FONT_CAPTION_MD, FONT_CAPTION_SM])
        draw.text((W - 178, by1 - 33), cameo_pill, font=f_cp, fill=(22, 20, 18))

    else:
        cap_map = {2: spec["caption_p2"], 3: spec["caption_p3"], 4: spec["caption_p4"]}
        cap_txt = cap_map[panel_num].upper()
        draw.rectangle([bx0, by0, bx1, by0 + cap_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
        draw.rectangle([bx0, by0, bx0 + 54, by0 + cap_h], fill=lead_ag["color_rgb"], outline=(22, 20, 18), width=4)
        draw.text((bx0 + 18, by0 + 10), str(panel_num), font=FONT_DISPLAY_XL, fill=(255, 251, 244))
        f_cap = fit_text_font(draw, cap_txt, (bx1 - bx0) - 74, [FONT_CAPTION_LG, FONT_CAPTION_MD])
        draw.text((bx0 + 68, by0 + 15), cap_txt, font=f_cap, fill=(22, 20, 18))

    if panel_num == 2:
        steps = spec["diag_steps"]
        n_s = len(steps)
        vis_steps = max(1, min(n_s, int(sub_step * n_s) + 1)) if sub_step < 0.99 else n_s
        box_w, box_h = 216, 216
        gap = 68
        total_w = n_s * box_w + (n_s - 1) * gap
        start_x = (W - total_w) // 2
        cy_box = by0 + cap_h + (by1 - by0 - cap_h) // 2 - 18
        for s_i in range(vis_steps):
            _, lbl = steps[s_i]
            scx = start_x + s_i * (box_w + gap) + box_w // 2
            sy1 = cy_box + box_h // 2
            bb = draw.textbbox((0, 0), lbl, font=FONT_LABEL_MD)
            tw = bb[2] - bb[0]
            draw.rounded_rectangle([scx - tw // 2 - 12, sy1 + 14, scx + tw // 2 + 12, sy1 + 46], radius=8, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
            draw.text((scx - tw // 2, sy1 + 18), lbl, font=FONT_LABEL_MD, fill=(22, 20, 18))

    elif panel_num == 3:
        kw, kh = 204, 153
        strip_y0 = by0 + cap_h + 18
        strip_y1 = strip_y0 + kh + 82
        k_labels = ["1. ARCHIVAL (4:3)", "2. KEY K1 (4:3)", "3. KEY K2 (4:3)", "4. KEY K4 (4:3)"]
        for k_i, kl in enumerate(k_labels):
            kx0 = bx0 + 35 + k_i * (kw + 14)
            ky0 = strip_y0 + 23
            draw.rounded_rectangle([kx0, ky0, kx0 + kw, ky0 + 25], radius=4, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
            f_kl = fit_text_font(draw, kl, kw - 12, [FONT_CAPTION_MD, FONT_CAPTION_SM])
            draw.text((kx0 + 8, ky0 + 3), kl, font=f_kl, fill=(22, 20, 18))

        nx0, ny0 = bx0 + 20, strip_y1 + 14
        nx1 = bx1 - 20
        note_hdr = f"{lead_ag['short_name']}'S EXPOSURE SHEET NOTE (CLOSURE ACROSS THE GUTTER):"
        draw.text((nx0 + 124, ny0 + 12), note_hdr, font=FONT_CAPTION_MD, fill=lead_ag["color_rgb"])
        note_body = spec.get("gutter_note", "FOUR HAND-DRAWN 4:3 KEYFRAMES LOCK THE MOTION PATH ACROSS THE GUTTER!")
        f_nb = fit_text_font(draw, note_body, (nx1 - nx0) - 140, [FONT_CAPTION_MD, FONT_CAPTION_SM])
        draw.text((nx0 + 124, ny0 + 40), note_body, font=f_nb, fill=(22, 20, 18))

    elif panel_num == 4:
        vy0 = by0 + cap_h + 16
        ax0 = bx0 + 20
        aw, ah = 280, 210
        # Label above 4:3 Archival Pencil Reference
        draw.rounded_rectangle([ax0, vy0, ax0 + aw, vy0 + 28], radius=5, fill=(214, 228, 240), outline=(22, 20, 18), width=2)
        draw.text((ax0 + 10, vy0 + 4), "ARCHIVAL TRACE (4:3 ASPECT)", font=FONT_CAPTION_MD, fill=(22, 20, 18))

        # Crew Sign-Off Card text on bottom-left
        cx0 = ax0
        cy0 = vy0 + 32 + ah + 16
        draw.text((cx0 + 104, cy0 + 14), f"LEAD: {lead_ag['short_name']}", font=FONT_CAPTION_MD, fill=lead_ag["color_rgb"])
        f_rl = fit_text_font(draw, lead_ag["dept"], 166, [FONT_CAPTION_SM])
        draw.text((cx0 + 104, cy0 + 38), lead_ag["dept"], font=f_rl, fill=(22, 20, 18))
        draw.text((cx0 + 104, cy0 + 62), f"CREW: {cameo_ag['short_name']}", font=FONT_CAPTION_SM, fill=(43, 91, 132))
        draw.rounded_rectangle([cx0 + 104, cy0 + 90, cx0 + aw - 10, cy0 + 122], radius=5, fill=(212, 232, 220), outline=(22, 20, 18), width=2)
        draw.text((cx0 + 112, cy0 + 97), "4:3 · 16/16 PASS", font=FONT_CAPTION_SM, fill=(47, 107, 79))

        # Badge on top-left of the 560x420 4:3 Hero Cartoon
        vx0 = bx1 - 560 - 18
        draw.rounded_rectangle([vx0 + 10, vy0 + 10, vx0 + 275, vy0 + 38], radius=5, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
        draw.text((vx0 + 18, vy0 + 15), f"7-PLANE CARTOON (4:3) // #{slug.upper()}", font=FONT_CAPTION_SM, fill=(22, 20, 18))

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_bgr, (bx0, by0), (bx1, by1), BGR_INK, 5)
    return out_bgr


def assemble_4panel_comic_page(spec: dict) -> np.ndarray:
    """
    Assembles the 4 full-resolution panels (`Panel 1..4`, each `960x540` = 16:9)
    into a crisp `1280x720` (16:9) 2x2 Comic Page with zero aspect distortion!
    """
    p1 = render_single_panel_fullscreen(spec, 1, sub_step=1.0, shot_frame_idx=16, boil_phase=0)
    p2 = render_single_panel_fullscreen(spec, 2, sub_step=1.0, shot_frame_idx=16, boil_phase=0)
    p3 = render_single_panel_fullscreen(spec, 3, sub_step=1.0, shot_frame_idx=16, boil_phase=0)
    p4 = render_single_panel_fullscreen(spec, 4, sub_step=1.0, shot_frame_idx=16, boil_phase=0)

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
    for beat_s in [0.05, 2.3, 4.6, 7.0]:
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
    fid = spec["id"]
    mp4_path = BTS_DIR / f"{fid}.mp4"
    gif_path = BTS_DIR / f"{fid}.gif"
    poster_path = BTS_DIR / f"{fid}_poster.png"
    raw_mp4 = BTS_DIR / f"_{fid}_raw.mp4"
    wav_path = BTS_DIR / f"_{fid}.wav"

    fps = 12
    n_frames = 120  # 10.0 calm seconds per episode
    W, H = 960, 540

    poster_bgr = assemble_4panel_comic_page(spec)
    cv2.imwrite(str(poster_path), poster_bgr)

    vw = cv2.VideoWriter(str(raw_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gif_pil_frames = []

    for f_i in range(n_frames):
        boil_phase = (f_i // 6) % 4
        if f_i < 28:
            p_num = 1
            sub = f_i / 27.0
            shot_f = 16
        elif f_i < 56:
            p_num = 2
            sub = (f_i - 28) / 27.0
            shot_f = 16
        elif f_i < 84:
            p_num = 3
            sub = (f_i - 56) / 27.0
            shot_f = 16
        else:
            p_num = 4
            sub = (f_i - 84) / 35.0
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
    cw, ch = 640, 360  # Exact 16:9 matching 1280x720 poster!
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
    print("1. Rendering 5 Agent Animator Studio Wing Comic Pages (Exact 4:3 & 1:1 Aspect Ratios)...")
    generate_studio_wing_comic_pages()

    office_results = []
    skill_results = []

    print("2. Rendering 10 Virtual Agent Office Episodes (Starring Nib, Blue-Line, Cel-Inker, Sprocket & Chime)...")
    for sp in AGENT_OFFICE_SPECS:
        res = render_single_bts_film(sp)
        office_results.append(res)
        print(f"   [Office {sp['num']:02d}/10] {sp['id']} ({sp['lead_agent']}) -> {res['mp4']}")

    print("3. Rendering 8 AI Video Director Skill Episodes...")
    for sp in VIDEO_SKILL_SPECS:
        res = render_single_bts_film(sp)
        skill_results.append(res)
        print(f"   [Skill  {sp['num']-10:02d}/08] {sp['id']} ({sp['lead_agent']}) -> {res['mp4']}")

    p_off = render_contact_sheet(
        office_results,
        "PART I: 10 AGENT ANIMATOR STUDIO EPISODES (STARRING NIB, BLUE-LINE, CEL-INKER, SPROCKET & CHIME)",
        "Exact 4:3 Shot Aspect Ratio (204x153 Keyframes & 560x420 Hero Screenings) · Zero Stretched Frames",
        "bts_10_agent_offices_atlas.png",
    )
    p_skl = render_contact_sheet(
        skill_results,
        "PART II: 8 AGENT ANIMATOR CRAFT & MOTION SKILL EPISODES (SECONDHAND STUDIOS)",
        "Camera Pacing · 7-Plane Parallax · Rotoscope Reanimation · 16-Gate Dailies Audit · Earshot Foley Sync",
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
