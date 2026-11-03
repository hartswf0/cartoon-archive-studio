#!/usr/bin/env python3
"""
tools/behind_the_scenes_studio_engine.py
========================================
Scott McCloud "Understanding Comics" + Quick, Draw! Sequential Art Studio Engine.

Directly implements Scott McCloud's core distinction between COMICS (Space) and FILM (Time)
from *Understanding Comics* (Chapter 1, Page 7 / PDF Page 14):
  "Each successive frame of a movie is projected on exactly the SAME space — the screen —
   while each frame of COMICS must occupy a DIFFERENT space.
   SPACE does for COMICS what TIME does for FILM!"

1. IN STATIC COMIC PAGES (`*_poster.png`, `studio_bg_1..5.jpg`, and `bts_*_atlas.png`):
   All 4 panels (`1920x1080` downsampled cleanly to `1280x720`) are juxtaposed in SPACE as a
   2x2 Scott McCloud Comic Page with crisp cream gutters, 4px black ink borders, auto-fitted
   captions (0% text truncation), and rich comic scenes modeled directly on *Understanding Comics*:
     - Scene A (McCloud p. 14): Multiplane Camera Rig with 3 separated glass depth planes
     - Scene B (McCloud p. 9):  Tilted animation drafting table, swing-arm lamp & storyboard wall
     - Scene C (McCloud p. 14): 35mm Dailies Film Projector casting a light beam onto a theater screen
     - Scene D (McCloud p. 24): Sequential Time-Frames gallery wall & 55-minute studio clock

2. IN ANIMATED FILMS (`bts_films/*.mp4`, `*.gif`, and `Behind_The_Scenes_Multiplane_Studio_Film.mp4`):
   Each of the 4 panels is projected FULL-SCREEN (`960x540`) ONE AT A TIME in TIME on the screen
   (10.0 seconds total @ 12fps = 120 frames):
     - Act 1 (0.0s - 2.3s): FULL-SCREEN PANEL 1 (Scott in the studio + huge 24pt speech balloon)
     - Act 2 (2.3s - 4.6s): FULL-SCREEN PANEL 2 (Step-by-step `1 --> 2 --> 3` visual diagram)
     - Act 3 (4.6s - 7.0s): FULL-SCREEN PANEL 3 (35mm 4-Keyframe Filmstrip: K1 -> K2 -> K3 -> K4)
     - Act 4 (7.0s - 10.0s): FULL-SCREEN PANEL 4 (Unobstructed 900x430 7-plane Quick, Draw! cartoon!)
"""

import json
import math
import os
import shutil
import subprocess
import textwrap
import wave
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# Unified McCloud + Quick, Draw! Palette (BGR for OpenCV)
BGR_GUTTER      = (244, 251, 255)  # #FFFBF4 crisp comic page gutter
BGR_PAPER       = (227, 239, 246)  # #F6EFE3 warm panel paper
BGR_PAPER_DARK  = (206, 226, 236)  # #ECE2CE shaded paper tone
BGR_INK         = (18, 20, 22)     # #161412 solid comic ink black
BGR_AMBER_PALE  = (194, 229, 251)  # #FBE5C2 narrative caption box fill
BGR_AMBER       = (58, 159, 239)   # #EF9F3A warm golden accent
BGR_RED         = (41, 74, 201)    # #C94A29 comic title red / wax accent
BGR_CORAL_PALE  = (206, 216, 247)  # #F7D8CE soft coral gouache cel
BGR_MINT_PALE   = (220, 232, 212)  # #D4E8DC soft sage gouache cel
BGR_BLUE_PALE   = (240, 228, 214)  # #D6E4F0 soft sky gouache cel
BGR_BLUE        = (132, 91, 43)    # #2B5B84 blueprint / McCloud jacket blue
BGR_GREEN       = (79, 107, 47)    # #2F6B4F forest green accent


def _load_font(paths, size: int):
    for p, idx in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size, index=idx)
            except Exception:
                try:
                    return ImageFont.truetype(p, size=size)
                except Exception:
                    pass
    return ImageFont.load_default()


FONT_DISPLAY_XL = _load_font([
    ("/System/Library/Fonts/Supplemental/Futura.ttc", 2),
    ("/System/Library/Fonts/Supplemental/Arial Black.ttf", 0),
], 28)

FONT_DISPLAY_LG = _load_font([
    ("/System/Library/Fonts/Supplemental/Futura.ttc", 2),
    ("/System/Library/Fonts/Supplemental/Arial Black.ttf", 0),
], 22)

FONT_DISPLAY_MD = _load_font([
    ("/System/Library/Fonts/Supplemental/Futura.ttc", 2),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
], 17)

FONT_SPEECH_LG = _load_font([
    ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf", 0),
], 22)

FONT_CAPTION_LG = _load_font([
    ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
], 19)

FONT_CAPTION_MD = _load_font([
    ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
], 16)

FONT_LABEL_MD = _load_font([
    ("/System/Library/Fonts/Supplemental/ChalkboardSE.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
], 17)


def fit_text_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, preferred_fonts: list):
    """Returns the largest font from `preferred_fonts` that fits `text` within `max_width`."""
    for f in preferred_fonts:
        bbox = draw.textbbox((0, 0), text, font=f)
        if (bbox[2] - bbox[0]) <= max_width:
            return f
    return preferred_fonts[-1]


STUDIO_AGENTS = [
    {"name": "Scott (Director)", "role": "Sequential Art Narrator & Guide", "color": BGR_BLUE_PALE, "accent": BGR_BLUE},
    {"name": "Lead Animator", "role": "4-Key Quick, Draw! Cel Inker", "color": BGR_CORAL_PALE, "accent": BGR_RED},
    {"name": "Camera Op", "role": "7-Plane Multiplane Rig Operator", "color": BGR_AMBER_PALE, "accent": BGR_AMBER},
    {"name": "Auditor (Codex)", "role": "16-Gate & Closure Dailies Critic", "color": BGR_MINT_PALE, "accent": BGR_GREEN},
]

# 10 Virtual Agent Office & Worktree Comic Chapters
AGENT_OFFICE_SPECS = [
    {
        "id": "office-01-agent-office",
        "num": 1,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "CHAPTER 01 // 3D MULTI-FLOOR STUDIO",
        "repo": "AgentSystemLabs/agent-office",
        "url": "https://github.com/AgentSystemLabs/agent-office",
        "category": "3D Multi-Floor Studio + Worktrees",
        "install": "curl -fsSL https://raw.githubusercontent.com/AgentSystemLabs/agent-office/main/install.sh | bash",
        "title": "Separating Depth Into 3 Studio Floors",
        "featured_shot": "giraffe-walks-past",
        "speech_p1": "HOW DO WE TURN A FLAT DOODLE INTO A DEEP MULTIPLANE WORLD?",
        "caption_p2": "WE SPLIT THE STUDIO INTO 3 FLOORS: VAULT, BULLPEN & TOWER!",
        "diag_steps": [("book", "1. ARCHIVE VAULT"), ("paintbrush", "2. 4-KEY BULLPEN"), ("camera", "3. 7-PLANE TOWER")],
        "caption_p3": "BEFORE IT'S PROJECTED, FILM IS 4 STATIC KEYFRAMES IN SPACE:",
        "caption_p4": "PROJECTED IN TIME, CLOSURE CREATES LIVING PARALLAX!",
        "blurb": "Explains in 4 sequential comic panels how the studio separates archive tracing (Floor 1), 4-keyframe cel inking (Floor 2), and 7-plane camera exposure (Tower) so every shot has true multiplane depth.",
        "stages": ["Panel 1: The Depth Question", "Panel 2: 3 Studio Floors", "Panel 3: 4 Keyframes in Space", "Panel 4: Living Parallax"],
    },
    {
        "id": "office-02-agent-007",
        "num": 2,
        "group": "office",
        "studio_wing": "bg5_bullpen",
        "wing_name": "CHAPTER 02 // KANBAN WORKTREE BULLPEN",
        "repo": "bill10/agent-007",
        "url": "https://github.com/bill10/agent-007",
        "category": "Kanban Worktrees + Isolated Desks",
        "install": "git clone https://github.com/bill10/agent-007",
        "title": "One Isolated Drafting Desk Per Shot",
        "featured_shot": "train",
        "speech_p1": "IF 4 ANIMATORS DRAW ON ONE CEL, IT'S AN UNREADABLE MESS!",
        "caption_p2": "EACH SHOT CARD SPAWNS ITS OWN ISOLATED DRAFTING DESK:",
        "diag_steps": [("picture_frame", "1. SHOT CARD"), ("table", "2. ISOLATED DESK"), ("train", "3. CLEAN MERGE")],
        "caption_p3": "EACH DESK INKS 4 CLEAN QUICK, DRAW! KEYFRAMES IN SEQUENCE:",
        "caption_p4": "AND SPLICES A CRISP STEAM TRAIN INTO THE MASTER REEL!",
        "blurb": "Shows why piling multiple agents onto one canvas fails—and how giving each shot card its own isolated Git worktree drafting desk produces clean, conflict-free 4-keyframe animation.",
        "stages": ["Panel 1: Why Shared Cels Fail", "Panel 2: Isolated Worktree Desk", "Panel 3: 4 Train Keyframes", "Panel 4: Clean Master Splice"],
    },
    {
        "id": "office-03-claude-office",
        "num": 3,
        "group": "office",
        "studio_wing": "bg2_occupied",
        "wing_name": "CHAPTER 03 // SPECIALIST INK & PAINT",
        "repo": "W17ant/Claude-Office",
        "url": "https://github.com/W17ant/Claude-Office",
        "category": "Specialist Routing + Cel Handoff",
        "install": "git clone https://github.com/W17ant/Claude-Office.git && cd Claude-Office && npm i && npm run install-hooks",
        "title": "Passing the Cel From Pencil to Gouache",
        "featured_shot": "lighthouse",
        "speech_p1": "ONE ARTIST SHOULDN'T DO EVERYTHING AT ONCE! WE DIVIDE THE CRAFT.",
        "caption_p2": "THE CEL PASSES FROM PENCIL TRACE TO WATERTIGHT GOUACHE:",
        "diag_steps": [("pencil", "1. PENCIL TRACE"), ("paintbrush", "2. GOUACHE CEL"), ("lighthouse", "3. 7-PLANE SCENE")],
        "caption_p3": "SPACE DOES FOR COMICS WHAT TIME DOES FOR FILM (4 KEYS):",
        "caption_p4": "THE WATERTIGHT CEL BLOCKS BACKGROUND BLEED IN MOTION!",
        "blurb": "Follows a single lighthouse cel as it moves from archival pencil silhouette to watertight gouache backing cel to 7-plane coastal composition.",
        "stages": ["Panel 1: Dividing the Craft", "Panel 2: Pencil -> Ink -> Paint", "Panel 3: 4 Lighthouse Keys", "Panel 4: Watertight Projection"],
    },
    {
        "id": "office-04-pixel-office",
        "num": 4,
        "group": "office",
        "studio_wing": "bg5_bullpen",
        "wing_name": "CHAPTER 04 // STORYBOARD & CRITIQUE",
        "repo": "KalebKE/PixelOffice",
        "url": "https://github.com/KalebKE/PixelOffice",
        "category": "Storyboard Pitch + Codex Critique",
        "install": "git clone https://github.com/KalebKE/PixelOffice && cd PixelOffice && python3 hooks/manage_hooks.py install && ./start.sh",
        "title": "Separating Form From Content at the Storyboard",
        "featured_shot": "guitar",
        "speech_p1": "BEFORE INKING A SINGLE FRAME, WE COMPOSE THE STAGE ARMATURE!",
        "caption_p2": "COMPOSER DRAFTS THE POSE -> CRITIC AUDITS THE SILHOUETTE:",
        "diag_steps": [("picture_frame", "1. ARMATURE"), ("eyeglasses", "2. CRITIC AUDIT"), ("guitar", "3. APPROVED CEL")],
        "caption_p3": "FOUR DISTINCT POSES CREATE ORGANIC LINE BOIL ACROSS FRAMES:",
        "caption_p4": "RESULT: A CLEAR, UNCLUTTERED GUITAR PERFORMANCE!",
        "blurb": "Demonstrates how planning composition on the storyboard wall before inking prevents visual clutter and locks the subject onto clean harmonic lines.",
        "stages": ["Panel 1: Plan Before Inking", "Panel 2: Armature & Audit", "Panel 3: 4 Guitar Keyframes", "Panel 4: Uncluttered Stage"],
    },
    {
        "id": "office-05-pixel-agents",
        "num": 5,
        "group": "office",
        "studio_wing": "bg2_occupied",
        "wing_name": "CHAPTER 05 // LIVING IN LINE",
        "repo": "pablodelucca/pixel-agents",
        "url": "https://github.com/pablodelucca/pixel-agents",
        "category": "Parallel Subagent Inkers",
        "install": "code --install-extension pablodelucca.pixel-agents",
        "title": "Why 4 Hand-Drawn Keys Beat Mechanical Tweening",
        "featured_shot": "tractor",
        "speech_p1": "SLIDING ONE STATIC DRAWING ACROSS THE SCREEN FEELS ROBOTIC!",
        "caption_p2": "FOUR SUBAGENT INKERS REDRAW KEYFRAMES K1, K2, K3 & K4:",
        "diag_steps": [("pencil", "1. KEYS K1 & K2"), ("paintbrush", "2. KEYS K3 & K4"), ("tractor", "3. LIVING BOIL")],
        "caption_p3": "COMPARE THE 4 HAND-DRAWN TRACTOR KEYS SIDE-BY-SIDE:",
        "caption_p4": "YOUR EYE CLOSES THE GUTTER INTO BREATHING HAND-DRAWN MOTION!",
        "blurb": "Explains Scott McCloud's 'Living in Line' (Ch. 5) and 'Blood in the Gutter' (Ch. 3): why 4 distinct hand-drawn Quick, Draw! keyframes feel alive while affine sliding feels robotic.",
        "stages": ["Panel 1: Why Sliding Feels Dead", "Panel 2: 4 Parallel Key Inkers", "Panel 3: K1..K4 Side-by-Side", "Panel 4: Living Line Boil"],
    },
    {
        "id": "office-06-my-office",
        "num": 6,
        "group": "office",
        "studio_wing": "bg4_clocktower",
        "wing_name": "CHAPTER 06 // TIME FRAMES & CONTINUITY",
        "repo": "m1heng/my-office",
        "url": "https://github.com/m1heng/my-office",
        "category": "Session Continuity & Time Frames",
        "install": "git clone https://github.com/m1heng/my-office.git && cd my-office && npm install && npm run build",
        "title": "Keeping Continuity Across Time Frames",
        "featured_shot": "speedboat",
        "speech_p1": "HOW DO WE KEEP CONTINUITY ACROSS 346 SHOTS WITHOUT LOSING THE THREAD?",
        "caption_p2": "BEFORE THE 55-MINUTE CLOCK EXPIRES, WE SEAL THE X-SHEET:",
        "diag_steps": [("clock", "1. 55M CLOCK"), ("book", "2. X-SHEET LOG"), ("speedboat", "3. NEXT REEL")],
        "caption_p3": "EVERY KEYFRAME IN THE FILMSTRIP REMEMBERS THE PREVIOUS POSE:",
        "caption_p4": "SO THE SPEEDBOAT CUTS ACROSS THE WATER WITH ZERO JUMPS!",
        "blurb": "Uses McCloud's Chapter 4 ('Time Frames') to show how logging exposure sheets before the 55-minute cache window preserves visual continuity across shots.",
        "stages": ["Panel 1: The Continuity Problem", "Panel 2: 55-Minute X-Sheet Log", "Panel 3: 4 Speedboat Keys", "Panel 4: Seamless Motion"],
    },
    {
        "id": "office-07-workbench",
        "num": 7,
        "group": "office",
        "studio_wing": "bg1_multifloor",
        "wing_name": "CHAPTER 07 // 7-PLANE EXPOSURE SHEET",
        "repo": "taw0002/Claude-Code-Agent-Monitor-Workbench",
        "url": "https://github.com/taw0002/Claude-Code-Agent-Monitor-Workbench",
        "category": "X-Sheet & Multiplane Platen Check",
        "install": "git clone https://github.com/taw0002/Claude-Code-Agent-Monitor-Workbench",
        "title": "Reading the 7-Plane Multiplane X-Sheet",
        "featured_shot": "bicycle",
        "speech_p1": "HOW DO CLOUDS, TREES, AND THE HERO MOVE AT DIFFERENT SPEEDS?",
        "caption_p2": "THE X-SHEET ASSIGNS EACH LAYER ITS OWN PARALLAX DEPTH:",
        "diag_steps": [("cloud", "1. SKY (0.2x)"), ("tree", "2. TREES (0.5x)"), ("bicycle", "3. HERO (1.0x)")],
        "caption_p3": "SIDE-BY-SIDE IN SPACE, YOU SEE THE BICYCLE PEDAL FORWARD:",
        "caption_p4": "PROJECTED IN TIME, THE 7 PLANES GLIDE IN DEPTH!",
        "blurb": "Breaks down multiplane parallax into a simple 3-step diagram: Sky moves at 0.2x, Midground Trees at 0.5x, and the Hero Bicycle at 1.0x.",
        "stages": ["Panel 1: How Parallax Works", "Panel 2: 0.2x / 0.5x / 1.0x Planes", "Panel 3: 4 Bicycle Keyframes", "Panel 4: Multiplane Depth"],
    },
    {
        "id": "office-08-kanban-code",
        "num": 8,
        "group": "office",
        "studio_wing": "bg5_bullpen",
        "wing_name": "CHAPTER 08 // THE 6 STEPS OF CREATION",
        "repo": "langoustine69/kanban-code",
        "url": "https://github.com/langoustine69/kanban-code",
        "category": "McCloud's 6 Steps: Idea -> Form -> Craft",
        "install": "git clone https://github.com/langoustine69/kanban-code && cd kanban-code && npm i && npm run compile",
        "title": "From Idea to Form to Idiom to Craft",
        "featured_shot": "carrot",
        "speech_p1": "IN CHAPTER 7, EVERY WORK OF ART TRAVELS THROUGH CLEAR STAGES!",
        "caption_p2": "IDEA -> FORM (4 KEYS) -> IDIOM (QUICK, DRAW!) -> CRAFT:",
        "diag_steps": [("light_bulb", "1. CORE IDEA"), ("pencil", "2. 4-KEY FORM"), ("carrot", "3. FINISHED CRAFT")],
        "caption_p3": "HERE ARE THE 4 STRUCTURAL KEYFRAMES OF SHOT #063 CARROT:",
        "caption_p4": "CRAFT AND SURFACE BRING THE SIMPLE DOODLE TO LIFE!",
        "blurb": "Applies Scott McCloud's Chapter 7 ('The Six Steps': Idea/Purpose -> Form -> Idiom -> Structure -> Craft -> Surface) to our 4-column studio Kanban pipeline.",
        "stages": ["Panel 1: McCloud's Six Steps", "Panel 2: Idea -> Form -> Craft", "Panel 3: 4 Carrot Keyframes", "Panel 4: Finished Surface"],
    },
    {
        "id": "office-09-vibe-kanban",
        "num": 9,
        "group": "office",
        "studio_wing": "bg3_screening",
        "wing_name": "CHAPTER 09 // SIDE-BY-SIDE DAILIES",
        "repo": "BloopAI/vibe-kanban",
        "url": "https://github.com/BloopAI/vibe-kanban",
        "category": "Side-by-Side Archival vs. Cartoon Review",
        "install": "npx vibe-kanban",
        "title": "Juxtaposing Archival Reference & Cartoon Cel",
        "featured_shot": "diamond",
        "speech_p1": "HOW DO WE KNOW OUR CARTOON MATCHES THE ARCHIVAL VIDEO'S MOTION?",
        "caption_p2": "WE JUXTAPOSE ARCHIVAL TRACE & CARTOON CEL SIDE-BY-SIDE:",
        "diag_steps": [("television", "1. ARCHIVAL"), ("eyeglasses", "2. COMPARE ARC"), ("diamond", "3. CARTOON CEL")],
        "caption_p3": "TRACE THE MOTION ARC ACROSS ALL 4 KEYFRAMES IN SPACE:",
        "caption_p4": "SAME MOTION TRAJECTORY — AMPLIFIED THROUGH SIMPLIFICATION!",
        "blurb": "Shows how placing the archival reference and the Quick, Draw! cartoon side-by-side lets the director verify motion fidelity while amplifying clarity through simplification.",
        "stages": ["Panel 1: Verifying the Motion Arc", "Panel 2: Archival vs. Cartoon", "Panel 3: 4 Diamond Keyframes", "Panel 4: Amplified Clarity"],
    },
    {
        "id": "office-10-claude-code-kanban",
        "num": 10,
        "group": "office",
        "studio_wing": "bg3_screening",
        "wing_name": "CHAPTER 10 // PUTTING IT ALL TOGETHER",
        "repo": "NikkeTryHard/claude-code-kanban",
        "url": "https://github.com/NikkeTryHard/claude-code-kanban",
        "category": "346-Shot Master Anthology Assembly",
        "install": "git clone https://github.com/NikkeTryHard/claude-code-kanban && cd claude-code-kanban && npm i && npm start",
        "title": "Assembling All 346 Shots Into One Anthology",
        "featured_shot": "star",
        "speech_p1": "WHEN ALL 346 SHOTS PASS REVIEW, WE SPLICE THEM INTO ONE REEL!",
        "caption_p2": "STORYBOARD -> 4-KEY LIGHTBOX -> 7-PLANE RIG -> MASTER REEL:",
        "diag_steps": [("picture_frame", "1. STORYBOARD"), ("camera", "2. MULTIPLANE"), ("star", "3. ANTHOLOGY")],
        "caption_p3": "FOUR KEYFRAMES OF SHOT #285 STAR IN DELIBERATE SEQUENCE:",
        "caption_p4": "SPACE AND TIME UNITE IN THE FINISHED CARTOON ARCHIVE!",
        "blurb": "Mirrors Scott McCloud's Chapter 9 ('Putting It All Together'), showing how 346 individual 4-keyframe cartoons combine into a single unified visual language.",
        "stages": ["Panel 1: The 346-Shot Goal", "Panel 2: Full Studio Pipeline", "Panel 3: 4 Star Keyframes", "Panel 4: The Living Anthology"],
    },
]

# 8 AI Video Director & Motion Skill Comic Chapters
VIDEO_SKILL_SPECS = [
    {
        "id": "skill-01-huashu-motion",
        "num": 11,
        "group": "skill",
        "studio_wing": "bg1_multifloor",
        "wing_name": "SKILL 01 // CAMERA BLOCKING & PAN",
        "timestamp": "01:12",
        "repo": "ceilf6/Huashu-lead-universal-to-motion",
        "url": "https://github.com/ceilf6/Huashu-lead-universal-to-motion",
        "category": "Multiplane Camera Pan & Pacing",
        "install": "git clone https://github.com/ceilf6/Huashu-lead-universal-to-motion.git",
        "title": "How Camera Pan Creates Depth Across Planes",
        "featured_shot": "giraffe-walks-past",
        "speech_p1": "WHY DOES THE CAMERA PAN SLOWLY INSTEAD OF JUMPING AROUND?",
        "caption_p2": "A SMOOTH PAN LETS YOUR EYE READ FOREGROUND vs. HORIZON:",
        "diag_steps": [("mountain", "1. HORIZON"), ("bush", "2. MID BUSHES"), ("giraffe", "3. HERO WALK")],
        "caption_p3": "FOUR KEYFRAMES OF THE GIRAFFE STRIDING PAST THE CAMERA:",
        "caption_p4": "CALM CAMERA PACING MAKES THE MOTION EFFORTLESS TO WATCH!",
        "blurb": "Explains why calm, deliberate camera pans across separated depth planes let the viewer perceive spatial composition instead of chaotic motion blur.",
        "stages": ["Panel 1: Why Calm Pacing Matters", "Panel 2: Horizon / Mid / Hero", "Panel 3: 4 Giraffe Keyframes", "Panel 4: Effortless Parallax"],
    },
    {
        "id": "skill-02-rossark-motion-skill",
        "num": 12,
        "group": "skill",
        "studio_wing": "bg5_bullpen",
        "wing_name": "SKILL 02 // FIGURE-GROUND CONTRAST",
        "timestamp": "02:35",
        "repo": "rossark/motion-skill",
        "url": "https://github.com/rossark/motion-skill",
        "category": "Figure-Ground Separation & Typography",
        "install": "git clone https://github.com/rossark/motion-skill.git ~/.claude/skills/motion-skill",
        "title": "Amplification Through Simplification",
        "featured_shot": "lighthouse",
        "speech_p1": "WHEN A FRAME IS FULL OF TINY LABELS, YOU CAN'T SEE THE STORY!",
        "caption_p2": "WE STRIP AWAY CLUTTER SO THE ICON READS IN ONE GLANCE:",
        "diag_steps": [("cloud", "1. SOFT SKY"), ("tree", "2. SIMPLE SHORE"), ("lighthouse", "3. BOLD HERO")],
        "caption_p3": "BOLD BLACK INK + WARM GOUACHE = 100% FIGURE-GROUND CLARITY:",
        "caption_p4": "BY SIMPLIFYING THE FORM, WE AMPLIFY THE MEANING!",
        "blurb": "Applies Scott McCloud's 'Amplification Through Simplification' (Ch. 2, p. 30): stripping away visual noise and floating labels so the bold Quick, Draw! icon commands the frame.",
        "stages": ["Panel 1: Eliminate Visual Noise", "Panel 2: Sky / Shore / Hero", "Panel 3: 4 Lighthouse Keys", "Panel 4: Instant Readability"],
    },
    {
        "id": "skill-03-take-one",
        "num": 13,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 03 // ONE IDEA PER SHOT",
        "timestamp": "04:55",
        "repo": "puravparab/take-one",
        "url": "https://github.com/puravparab/take-one",
        "category": "Single-Focus Shot Directing",
        "install": "npx skills add puravparab/take-one",
        "title": "One Clear Action Per Panel, One Beat at a Time",
        "featured_shot": "bicycle",
        "speech_p1": "NEVER MAKE THE VIEWER READ FOUR MOVING THINGS AT ONCE!",
        "caption_p2": "GIVE EACH PANEL ONE CLEAR SUBJECT AND ONE CLEAR ACTION:",
        "diag_steps": [("megaphone", "1. CALL ACTION"), ("camera", "2. FRAME HERO"), ("bicycle", "3. HOLD POSE")],
        "caption_p3": "FOUR DELIBERATE KEYFRAMES OF THE BICYCLE IN SEQUENCE:",
        "caption_p4": "ONE CLEAR SUBJECT IN MOTION — NOTHING EXTRA TO DISTRACT YOU!",
        "blurb": "Demonstrates the golden rule of both comics and film direction: one clear focal action per panel so the viewer's eye always knows exactly where to look.",
        "stages": ["Panel 1: Stop Simultaneous Clutter", "Panel 2: One Subject Per Frame", "Panel 3: 4 Bicycle Keyframes", "Panel 4: Pure Focal Motion"],
    },
    {
        "id": "skill-04-reanimate",
        "num": 14,
        "group": "skill",
        "studio_wing": "bg4_clocktower",
        "wing_name": "SKILL 04 // ROTOSCOPE & CLOSURE",
        "timestamp": "06:34",
        "repo": "NiazMorshed2007/reanimate",
        "url": "https://github.com/NiazMorshed2007/reanimate",
        "category": "Archival Trace -> Iconic Doodle",
        "install": "npx skills add NiazMorshed2007/reanimate -g",
        "title": "From Photographic Video to Iconic Cartoon",
        "featured_shot": "train",
        "speech_p1": "A PHOTOGRAPH SHOWS ONE OBJECT. A CARTOON SPEAKS UNIVERSALLY!",
        "caption_p2": "WE TRACE THE ARCHIVAL MOTION ARC INTO AN ICONIC DOODLE:",
        "diag_steps": [("television", "1. RAW VIDEO"), ("pencil", "2. MOTION ARC"), ("train", "3. ICONIC TRAIN")],
        "caption_p3": "THE 4 ICONIC TRAIN KEYFRAMES PRESERVE THE EXACT TRAJECTORY:",
        "caption_p4": "NOW THE ARCHIVAL MOTION LIVES AS A UNIVERSAL CARTOON!",
        "blurb": "Illustrates Scott McCloud's Picture Plane triangle (Ch. 2, pp. 28-31): moving from realistic video toward iconic cartooning while preserving the authentic physical motion arc.",
        "stages": ["Panel 1: Photo vs. Cartoon Icon", "Panel 2: Trace the Motion Arc", "Panel 3: 4 Iconic Train Keys", "Panel 4: Universal Motion"],
    },
    {
        "id": "skill-05-argo",
        "num": 15,
        "group": "skill",
        "studio_wing": "bg1_multifloor",
        "wing_name": "SKILL 05 // FULL-SCREEN SEQUENCING",
        "timestamp": "09:02",
        "repo": "shreyaskarnik/argo",
        "url": "https://github.com/shreyaskarnik/argo",
        "category": "Full-Screen Sequential Projection",
        "install": "npm i -D @argo-video/cli",
        "title": "Space in Comics vs. Time in Film",
        "featured_shot": "guitar",
        "speech_p1": "IN COMICS, PANELS SHARE THE PAGE. IN FILM, EACH FILLS THE SCREEN!",
        "caption_p2": "WE PROJECT ONE FULL-SCREEN BEAT AT A TIME (2.5s EACH):",
        "diag_steps": [("flashlight", "1. SETUP BEAT"), ("eyeglasses", "2. DIAGRAM BEAT"), ("guitar", "3. FULL FINALE")],
        "caption_p3": "FIRST YOU SEE THE 4 KEYFRAMES LAID OUT SIDE-BY-SIDE...",
        "caption_p4": "...THEN THE FINISHED ANIMATION FILLS THE SCREEN IN TIME!",
        "blurb": "Implements Scott McCloud's Page 14 law ('Each successive frame of a movie is projected on the SAME space — the screen!'): projecting each panel full-screen in the video while juxtaposing all 4 on the comic page.",
        "stages": ["Panel 1: Space vs. Time (McCloud p. 14)", "Panel 2: Full-Screen Beats", "Panel 3: 4 Guitar Keyframes", "Panel 4: Full-Screen Finale"],
    },
    {
        "id": "skill-06-explainroo",
        "num": 16,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 06 // SHOW AND TELL (CH. 6)",
        "timestamp": "09:50",
        "repo": "vincentsch/explainroo",
        "url": "https://github.com/vincentsch/explainroo",
        "category": "Interdependent Words & Pictures",
        "install": "git clone https://github.com/vincentsch/explainroo.git",
        "title": "Show and Tell: Words & Pictures Together",
        "featured_shot": "diamond",
        "speech_p1": "IN CHAPTER 6, WORDS AND PICTURES ARE PARTNERS IN A DANCE!",
        "caption_p2": "WORDS SET THE QUESTION -> PICTURES SHOW WHAT WORDS CANNOT:",
        "diag_steps": [("microphone", "1. SHORT CAPTION"), ("picture_frame", "2. CLEAR IMAGE"), ("diamond", "3. MEANING")],
        "caption_p3": "NO TINY TELEMETRY NUMBERS — JUST 4 CLEAR VISUAL KEYFRAMES:",
        "caption_p4": "THE CAPTION SETS THE STAGE; THE CARTOON DELIVERS THE MOTION!",
        "blurb": "Applies Scott McCloud's Chapter 6 ('Show and Tell'): replacing redundant technical text tags with short, punchy captions that let the drawings do the visual heavy lifting.",
        "stages": ["Panel 1: Partners in a Dance", "Panel 2: Words + Pictures", "Panel 3: 4 Visual Keyframes", "Panel 4: Show, Don't Clutter"],
    },
    {
        "id": "skill-07-video-director",
        "num": 17,
        "group": "skill",
        "studio_wing": "bg1_multifloor",
        "wing_name": "SKILL 07 // COMPOSITION & BALANCE",
        "timestamp": "11:20",
        "repo": "win4r/video-director",
        "url": "https://github.com/win4r/video-director",
        "category": "Rule of Thirds & Visual Balance",
        "install": "git clone https://github.com/win4r/video-director.git && cp -R video-director/video-director ~/.claude/skills/",
        "title": "Anchoring the Hero in Clean Space",
        "featured_shot": "carrot",
        "speech_p1": "GOOD COMPOSITION MEANS AMPLE BREATHING ROOM AROUND THE HERO!",
        "caption_p2": "WE ANCHOR THE SUBJECT ON THE HARMONIC GRID WITH CLEAN MARGINS:",
        "diag_steps": [("picture_frame", "1. AMPLE MARGIN"), ("scissors", "2. TRIM CLUTTER"), ("carrot", "3. BALANCED SHOT")],
        "caption_p3": "NOTICE THE GENEROUS CREAM PAPER SPACE AROUND ALL 4 KEYFRAMES:",
        "caption_p4": "NEGATIVE SPACE MAKES THE CARTOON SILHOUETTE POP!",
        "blurb": "Demonstrates why negative space (warm cream paper around the subject) is essential for readable composition across all 346 shots.",
        "stages": ["Panel 1: The Power of Negative Space", "Panel 2: Trim the Clutter", "Panel 3: 4 Balanced Keyframes", "Panel 4: Crisp Silhouette Pop"],
    },
    {
        "id": "skill-08-earshot",
        "num": 18,
        "group": "skill",
        "studio_wing": "bg3_screening",
        "wing_name": "SKILL 08 // SOUND IN THE GUTTER",
        "timestamp": "12:57",
        "repo": "Leonxlnx/earshot",
        "url": "https://github.com/Leonxlnx/earshot",
        "category": "Comic Sound Effects & Foley Sync",
        "install": "git clone https://github.com/Leonxlnx/earshot.git",
        "title": "Seeing Sound & Hearing Motion",
        "featured_shot": "zigzag",
        "speech_p1": "IN COMICS, A BOLD 'BANG!' OR 'WHOOSH!' LETS YOUR EYES HEAR SOUND!",
        "caption_p2": "IN FILM, WE TRIGGER FOLEY SFX AT THE EXACT PEAK OF MOTION:",
        "diag_steps": [("headphones", "1. PEAK TURN"), ("megaphone", "2. WHOOSH! SFX"), ("zigzag", "3. SYNCED REEL")],
        "caption_p3": "WHEREVER THE ZIGZAG SHARPLY CHANGES DIRECTION (K2 & K3)...",
        "caption_p4": "...EARSHOT FIRES A CRISP FOLEY ACCENT RIGHT ON THE BEAT!",
        "blurb": "Connects comic-book onomatopoeia ('PTU!', 'BANG!', 'WHOOSH!' from McCloud pp. 5-6) to our automated Earshot Foley cues triggered at sharp turns in the motion path.",
        "stages": ["Panel 1: Seeing Sound in Comics", "Panel 2: Peak Turn -> Foley Cue", "Panel 3: 4 Zigzag Keyframes", "Panel 4: Synced Sound & Motion"],
    },
]

_QD_CACHE = {}
_SHOT_FRAMES_CACHE = {}


def load_qd_norm(cat: str, idx: int = 0) -> list:
    cat_clean = "giraffe" if cat == "giraffe-walks-past" else cat.replace("-", "_").strip()
    if cat_clean not in _QD_CACHE:
        p = QD_DIR / f"{cat_clean}.json"
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
    norm = load_qd_norm(cat, idx=idx)
    if not norm:
        return
    rng = np.random.RandomState((hash(cat) & 0xFFFF) + boil_phase * 53)
    x0 = cx - w * 0.5
    y0 = cy - h * 0.5
    scaled_strokes = []
    all_pts = []
    for s in norm:
        pts = []
        for nx, ny in s:
            jx = float(rng.uniform(-0.4, 0.4)) if boil_phase > 0 else 0.0
            jy = float(rng.uniform(-0.4, 0.4)) if boil_phase > 0 else 0.0
            px = int(round(x0 + nx * w + jx))
            py = int(round(y0 + ny * h + jy))
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
    """Loads 33 archival pencil sketches and 33 Quick, Draw! cartoon frames with old HUD bars cleanly cropped out."""
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
            fh, fw = fr.shape[:2]
            half_w = fw // 2
            # Crop top 22px and bottom 24px so old baked-in HUD bars in comparison.mp4 are 100% gone!
            y_top = min(22, fh // 8)
            y_bot = max(y_top + 40, fh - 24)
            crop_a = fr[y_top:y_bot, 4:half_w - 4]
            crop_c = fr[y_top:y_bot, half_w + 4:fw - 4]
            raw_a = cv2.resize(crop_a, (480, 320), interpolation=cv2.INTER_AREA)
            raw_c = cv2.resize(crop_c, (480, 320), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(raw_a, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 45, 125)
            adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
            ink = cv2.bitwise_or(edges, adapt)
            sk = np.full((320, 480, 3), BGR_PAPER, dtype=np.uint8)
            sk[blur < 115] = BGR_PAPER_DARK
            sk[ink > 0] = BGR_INK
            arch_sketches.append(sk)
            cart_frames.append(raw_c)
        cap.release()
    if not cart_frames:
        blank = np.full((320, 480, 3), BGR_PAPER, dtype=np.uint8)
        draw_qd_doodle(blank, slug, 240, 160, 180, 180, fill_bgr=BGR_AMBER_PALE, thickness=4)
        arch_sketches = [blank.copy() for _ in range(33)]
        cart_frames = [blank.copy() for _ in range(33)]
    while len(cart_frames) < 33:
        arch_sketches.append(arch_sketches[-1])
        cart_frames.append(cart_frames[-1])
    _SHOT_FRAMES_CACHE[slug] = (arch_sketches, cart_frames)
    return arch_sketches, cart_frames


def draw_mccloud_presenter(img: np.ndarray, cx: int, cy: int, scale: float = 1.0, pose: int = 0, dark_bg: bool = False):
    """Draws our iconic Scott McCloud-style Cartoon Narrator (round white glasses, plaid blazer, lightning tee)."""
    def S(val):
        return int(round(val * scale))

    tx0, tx1 = cx - S(52), cx + S(52)
    ty0, ty1 = cy + S(28), cy + S(115)
    torso_pts = np.array([
        [cx - S(40), ty0],
        [cx + S(40), ty0],
        [tx1, ty1],
        [tx0, ty1],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, torso_pts, BGR_BLUE, cv2.LINE_AA)
    cv2.polylines(img, [torso_pts], True, BGR_INK if not dark_bg else BGR_GUTTER, 3, cv2.LINE_AA)

    for gx in range(cx - S(42), cx + S(44), max(8, S(14))):
        cv2.line(img, (gx, ty0 + S(4)), (gx, ty1 - S(2)), BGR_INK, 2, cv2.LINE_AA)
    for gy in range(ty0 + S(14), ty1, max(8, S(16))):
        cv2.line(img, (tx0 + S(6), gy), (tx1 - S(6), gy), BGR_INK, 2, cv2.LINE_AA)

    shirt_pts = np.array([
        [cx - S(16), ty0],
        [cx + S(16), ty0],
        [cx + S(11), ty1],
        [cx - S(11), ty1],
    ], dtype=np.int32)
    cv2.fillConvexPoly(img, shirt_pts, BGR_RED, cv2.LINE_AA)
    cv2.polylines(img, [shirt_pts], True, BGR_INK, 2, cv2.LINE_AA)
    bolt_pts = np.array([
        [cx - S(6), ty0 + S(20)],
        [cx + S(8), ty0 + S(32)],
        [cx - S(2), ty0 + S(38)],
        [cx + S(9), ty0 + S(54)],
    ], dtype=np.int32)
    cv2.polylines(img, [bolt_pts], False, BGR_AMBER, 3, cv2.LINE_AA)

    arm_lift = S(-10) if (pose % 2 == 1) else 0
    hx = cx + S(68)
    hy = cy + S(34) + arm_lift
    cv2.line(img, (cx + S(38), ty0 + S(20)), (hx, hy), BGR_BLUE, max(8, S(13)), cv2.LINE_AA)
    cv2.line(img, (cx + S(38), ty0 + S(20)), (hx, hy), BGR_INK if not dark_bg else BGR_GUTTER, 3, cv2.LINE_AA)
    cv2.circle(img, (hx + S(5), hy - S(3)), S(10), BGR_GUTTER, -1, cv2.LINE_AA)
    cv2.circle(img, (hx + S(5), hy - S(3)), S(10), BGR_INK, 2, cv2.LINE_AA)

    cv2.rectangle(img, (cx - S(10), cy + S(15)), (cx + S(10), cy + S(30)), BGR_GUTTER, -1)
    cv2.rectangle(img, (cx - S(10), cy + S(15)), (cx + S(10), cy + S(30)), BGR_INK, 2)

    cv2.ellipse(img, (cx, cy - S(6)), (S(29), S(34)), 0, 0, 360, BGR_GUTTER, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy - S(6)), (S(29), S(34)), 0, 0, 360, BGR_INK, 3, cv2.LINE_AA)

    hair_pts = np.array([
        [cx - S(30), cy - S(8)],
        [cx - S(28), cy - S(32)],
        [cx - S(8), cy - S(43)],
        [cx + S(15), cy - S(42)],
        [cx + S(31), cy - S(28)],
        [cx + S(30), cy - S(8)],
        [cx + S(19), cy - S(19)],
        [cx + S(2), cy - S(16)],
        [cx - S(14), cy - S(24)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [hair_pts], BGR_INK, cv2.LINE_AA)
    if dark_bg:
        cv2.polylines(img, [hair_pts], True, BGR_GUTTER, 1, cv2.LINE_AA)

    lx, rx = cx - S(11), cx + S(13)
    gy = cy - S(6)
    gr = S(11)
    cv2.circle(img, (lx, gy), gr, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (rx, gy), gr, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (lx, gy), gr, BGR_INK, 3, cv2.LINE_AA)
    cv2.circle(img, (rx, gy), gr, BGR_INK, 3, cv2.LINE_AA)
    cv2.line(img, (lx + gr, gy), (rx - gr, gy), BGR_INK, 3, cv2.LINE_AA)
    cv2.line(img, (cx - S(29), gy - S(2)), (lx - gr, gy), BGR_INK, 3, cv2.LINE_AA)

    brow_y = gy - gr - S(5)
    cv2.line(img, (lx - S(8), brow_y + S(2)), (lx + S(7), brow_y - S(2)), BGR_INK, 2, cv2.LINE_AA)
    cv2.line(img, (rx - S(7), brow_y - S(2)), (rx + S(8), brow_y + S(2)), BGR_INK, 2, cv2.LINE_AA)

    nose_pts = np.array([[cx + S(1), gy + S(3)], [cx + S(7), gy + S(11)], [cx + S(1), gy + S(13)]], dtype=np.int32)
    cv2.polylines(img, [nose_pts], False, BGR_INK, 2, cv2.LINE_AA)
    if pose % 2 == 0:
        cv2.ellipse(img, (cx + S(1), cy + S(16)), (S(8), S(5)), 0, 0, 180, BGR_INK, -1, cv2.LINE_AA)
    else:
        cv2.ellipse(img, (cx + S(1), cy + S(16)), (S(7), S(3)), 0, 0, 180, BGR_INK, 2, cv2.LINE_AA)


def draw_panel1_studio_environment(canvas: np.ndarray, spec: dict, boil_phase: int = 0):
    """
    Draws a rich, uncluttered Scott McCloud studio environment on the right side of Panel 1 (x: 430..920, y: 190..505)
    modeled directly on *Understanding Comics*:
      - `bg1_multifloor`: 3-Plane Multiplane Camera Tower (Sky / Midground / Hero glass platens)
      - `bg5_bullpen` / `bg2_occupied`: Tilted Animation Drafting Table & Swing-Arm Lamp (McCloud p. 9, Panel 1)
      - `bg3_screening`: 35mm Dailies Film Projector & Theater Screen (McCloud p. 14, Panel 7)
      - `bg4_clocktower`: Sequential Gallery Wall & 55-Minute Studio Clock (McCloud p. 15 & p. 24)
    """
    wing = spec["studio_wing"]
    slug = spec["featured_shot"]

    if wing == "bg1_multifloor":
        # 3-Tier Multiplane Camera Stand (x: 500..880, y: 195..500)
        cv2.line(canvas, (520, 200), (520, 505), BGR_INK, 6, cv2.LINE_AA)
        cv2.line(canvas, (860, 200), (860, 505), BGR_INK, 6, cv2.LINE_AA)
        # Top Camera Unit pointing down
        draw_qd_doodle(canvas, "camera", 690, 222, 74, 54, fill_bgr=BGR_AMBER_PALE, thickness=3, boil_phase=boil_phase)
        # 3 Separated Horizontal Glass Planes (`P2 HERO`, `P4 MID`, `P6 SKY`)
        plane_specs = [
            (295, BGR_CORAL_PALE, slug, "HERO PLANE (1.0x)"),
            (375, BGR_MINT_PALE, "tree", "MID PLANE (0.5x)"),
            (455, BGR_BLUE_PALE, "cloud", "SKY PLANE (0.2x)"),
        ]
        for py, pcol, pcat, _ in plane_specs:
            pts = np.array([[505, py + 12], [545, py - 14], [875, py - 14], [835, py + 12]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts + np.array([3, 3]), BGR_INK, cv2.LINE_AA)
            cv2.fillConvexPoly(canvas, pts, pcol, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, BGR_INK, 3, cv2.LINE_AA)
            draw_qd_doodle(canvas, pcat, 690, py - 4, 58, 44, fill_bgr=BGR_GUTTER, thickness=2, boil_phase=boil_phase)

    elif wing in ("bg5_bullpen", "bg2_occupied"):
        # Scott's Tilted Animation Drafting Table & Architect Lamp (McCloud p. 9, Panel 1!)
        # Back Storyboard Card Frame on wall
        cv2.rectangle(canvas, (565, 198), (885, 312), BGR_INK, -1)
        cv2.rectangle(canvas, (560, 193), (880, 307), BGR_PAPER_DARK, -1)
        cv2.rectangle(canvas, (560, 193), (880, 307), BGR_INK, 3)
        for ci, ccat in enumerate(["picture_frame", slug, "star"]):
            cx_c = 618 + ci * 102
            cv2.rectangle(canvas, (cx_c - 40, 206), (cx_c + 40, 294), BGR_GUTTER, -1)
            cv2.rectangle(canvas, (cx_c - 40, 206), (cx_c + 40, 294), BGR_INK, 2)
            draw_qd_doodle(canvas, ccat, cx_c, 250, 56, 56, fill_bgr=BGR_AMBER_PALE, thickness=2, boil_phase=boil_phase)

        # Tilted Drafting Desk surface + glowing circular animation lightbox disc
        desk_pts = np.array([[475, 425], [840, 355], [900, 445], [535, 505]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, desk_pts + np.array([4, 4]), BGR_INK, cv2.LINE_AA)
        cv2.fillConvexPoly(canvas, desk_pts, BGR_AMBER_PALE, cv2.LINE_AA)
        cv2.polylines(canvas, [desk_pts], True, BGR_INK, 4, cv2.LINE_AA)
        cv2.ellipse(canvas, (690, 428), (78, 42), -10, 0, 360, BGR_GUTTER, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (690, 428), (78, 42), -10, 0, 360, BGR_INK, 3, cv2.LINE_AA)
        draw_qd_doodle(canvas, slug, 690, 426, 86, 64, fill_bgr=None, thickness=3, boil_phase=boil_phase)

    elif wing == "bg3_screening":
        # McCloud Page 14 Panel 7: 35mm Film Projector on table casting light beam onto Theater Screen!
        # Projector Reels & Body (x: 470..590, y: 310..480)
        cv2.circle(canvas, (495, 335), 28, BGR_PAPER_DARK, -1, cv2.LINE_AA)
        cv2.circle(canvas, (495, 335), 28, BGR_INK, 3, cv2.LINE_AA)
        cv2.circle(canvas, (555, 335), 28, BGR_PAPER_DARK, -1, cv2.LINE_AA)
        cv2.circle(canvas, (555, 335), 28, BGR_INK, 3, cv2.LINE_AA)
        cv2.rectangle(canvas, (485, 365), (568, 435), BGR_BLUE_PALE, -1)
        cv2.rectangle(canvas, (485, 365), (568, 435), BGR_INK, 3)
        # Golden projection beam cone
        beam_pts = np.array([[570, 385], [890, 220], [890, 475], [570, 405]], dtype=np.int32)
        cv2.fillConvexPoly(canvas, beam_pts, BGR_AMBER_PALE, cv2.LINE_AA)
        cv2.line(canvas, (570, 385), (890, 220), BGR_INK, 2, cv2.LINE_AA)
        cv2.line(canvas, (570, 405), (890, 475), BGR_INK, 2, cv2.LINE_AA)
        # Projection Screen on right (x: 685..905, y: 215..475)
        cv2.rectangle(canvas, (690, 220), (905, 475), BGR_INK, -1)
        cv2.rectangle(canvas, (684, 214), (899, 469), BGR_GUTTER, -1)
        cv2.rectangle(canvas, (684, 214), (899, 469), BGR_INK, 4)
        draw_qd_doodle(canvas, slug, 792, 342, 145, 145, fill_bgr=BGR_AMBER_PALE, thickness=3, boil_phase=boil_phase)

    else:
        # McCloud Page 15 & 24: Time-Frames Gallery Wall + 55-Minute Clock
        draw_qd_doodle(canvas, "clock", 535, 275, 95, 95, fill_bgr=BGR_AMBER_PALE, thickness=3, boil_phase=boil_phase)
        for fi in range(3):
            fx0 = 615 + fi * 98
            fy0 = 235 + fi * 18
            cv2.rectangle(canvas, (fx0 + 4, fy0 + 4), (fx0 + 84, fy0 + 134), BGR_INK, -1)
            cv2.rectangle(canvas, (fx0, fy0), (fx0 + 80, fy0 + 130), BGR_GUTTER, -1)
            cv2.rectangle(canvas, (fx0, fy0), (fx0 + 80, fy0 + 130), BGR_INK, 3)
            draw_qd_doodle(canvas, slug, fx0 + 40, fy0 + 65, 58, 58, fill_bgr=BGR_BLUE_PALE, thickness=2, idx=fi)


def render_single_panel_fullscreen(
    spec: dict,
    panel_num: int,
    sub_step: float = 1.0,
    shot_frame_idx: int = 16,
    boil_phase: int = 0,
) -> np.ndarray:
    """
    Renders ONE single panel (`panel_num` in 1..4) at FULL 960x540 screen size!
    Because each panel is designed at 960x540 with large 20pt-28pt typography and auto-fitted
    headers, it is 100% readable both when projected full-screen in the video AND when assembled
    into a 2x2 comic page!
    """
    W, H = 960, 540
    canvas = np.full((H, W, 3), BGR_PAPER, dtype=np.uint8)
    slug = spec["featured_shot"]
    arch_sketches, cart_frames = load_shot_frames(slug)

    pad = 16
    cap_h = 54
    bx0, by0, bx1, by1 = pad, pad, W - pad, H - pad

    if panel_num == 1:
        # Dramatic solid black backdrop for odd chapters (like McCloud pp. 9, 10, 13, 14), warm paper for even
        dark_p1 = (spec["num"] % 2 == 1)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_INK if dark_p1 else BGR_PAPER, -1)
        if dark_p1:
            rng_s = np.random.RandomState(spec["num"] * 31)
            for _ in range(28):
                sx = int(rng_s.randint(bx0 + 20, bx1 - 20))
                sy = int(rng_s.randint(by0 + 70, by1 - 20))
                cv2.circle(canvas, (sx, sy), 2, BGR_GUTTER, -1)

        # Draw the McCloud Studio Environment on the right half
        draw_panel1_studio_environment(canvas, spec, boil_phase=boil_phase)

        # Draw Scott McCloud Presenter on the bottom-left (large & expressive)
        pres_pose = (boil_phase // 2) % 2
        draw_mccloud_presenter(canvas, bx0 + 165, by1 - 135, scale=1.35, pose=pres_pose, dark_bg=dark_p1)

    elif panel_num == 2:
        # 3-Step Visual Diagram (`[1] --> [2] --> [3]`)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        steps = spec["diag_steps"]
        n_s = len(steps)
        vis_steps = max(1, min(n_s, int(sub_step * n_s) + 1)) if sub_step < 0.99 else n_s
        g_fills = [BGR_BLUE_PALE, BGR_AMBER_PALE, BGR_MINT_PALE]
        box_w, box_h = 216, 230
        gap = 68
        total_w = n_s * box_w + (n_s - 1) * gap
        start_x = (W - total_w) // 2
        cy_box = by0 + cap_h + (by1 - by0 - cap_h) // 2 - 16

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
                w=145, h=145,
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
        # McCloud Page 14-15: "Before it's projected, film is just a very slow comic!" (35mm 4-Keyframe Strip)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        strip_y0 = by0 + cap_h + 36
        strip_y1 = by1 - 36
        cv2.rectangle(canvas, (bx0 + 20, strip_y0), (bx1 - 20, strip_y1), BGR_INK, -1)
        for sx in range(bx0 + 34, bx1 - 30, 28):
            cv2.rectangle(canvas, (sx, strip_y0 + 8), (sx + 14, strip_y0 + 20), BGR_GUTTER, -1)
            cv2.rectangle(canvas, (sx, strip_y1 - 20), (sx + 14, strip_y1 - 8), BGR_GUTTER, -1)

        key_indices = [0, 10, 21, 32]
        kw = (bx1 - bx0 - 80) // 4
        kh = (strip_y1 - strip_y0) - 68
        active_k = min(3, int(sub_step * 4)) if sub_step < 0.99 else -1

        for k_i, f_idx in enumerate(key_indices):
            kx0 = bx0 + 32 + k_i * (kw + 6)
            ky0 = strip_y0 + 34
            src_fr = arch_sketches[f_idx] if k_i == 0 else cart_frames[f_idx]
            thumb = cv2.resize(src_fr, (kw, kh), interpolation=cv2.INTER_AREA)
            canvas[ky0:ky0 + kh, kx0:kx0 + kw] = thumb
            is_hi = (k_i == active_k or active_k == -1)
            cv2.rectangle(canvas, (kx0, ky0), (kx0 + kw, ky0 + kh), BGR_AMBER if is_hi else BGR_GUTTER, 4 if is_hi else 2)

    elif panel_num == 4:
        # Full-screen unobstructed 7-Plane Quick, Draw! Cartoon
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), BGR_PAPER, -1)
        vx0, vy0 = bx0 + 24, by0 + cap_h + 14
        vw, vh = (bx1 - bx0) - 48, (by1 - by0 - cap_h) - 28
        cur_fr = cart_frames[shot_frame_idx % len(cart_frames)]
        big_cart = cv2.resize(cur_fr, (vw, vh), interpolation=cv2.INTER_LINEAR)
        canvas[vy0:vy0 + vh, vx0:vx0 + vw] = big_cart
        cv2.rectangle(canvas, (vx0, vy0), (vx0 + vw, vy0 + vh), BGR_INK, 4)

    # =========================================================================
    # PIL HIGH-RESOLUTION COMIC LETTERING (0% TRUNCATION GUARANTEED)
    # =========================================================================
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    if panel_num == 1:
        # Top chapter pill inside Panel 1
        draw.rectangle([bx0, by0, bx1, by0 + cap_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
        draw.rectangle([bx0, by0, bx0 + 54, by0 + cap_h], fill=(201, 74, 41), outline=(22, 20, 18), width=4)
        draw.text((bx0 + 18, by0 + 10), "1", font=FONT_DISPLAY_XL, fill=(255, 251, 244))
        hdr_txt = f"{spec['wing_name']} — {spec['title'].upper()}"
        f_hdr = fit_text_font(draw, hdr_txt, (bx1 - bx0) - 74, [FONT_DISPLAY_LG, FONT_DISPLAY_MD, FONT_CAPTION_MD])
        draw.text((bx0 + 68, by0 + 14), hdr_txt, font=f_hdr, fill=(22, 20, 18))

        # Large Oval Speech Balloon above Scott
        s_box = (bx0 + 44, by0 + cap_h + 18, bx0 + 540, by0 + cap_h + 148)
        tx, ty = bx0 + 185, by1 - 190
        bx_mid = (s_box[0] + s_box[2]) // 2
        tail_pts = [(bx_mid - 18, s_box[3] - 6), (bx_mid + 12, s_box[3] - 6), (tx, ty)]
        draw.polygon([(p[0] + 4, p[1] + 4) for p in tail_pts], fill=(22, 20, 18))
        draw.rounded_rectangle([s_box[0] + 4, s_box[1] + 4, s_box[2] + 4, s_box[3] + 4], radius=28, fill=(22, 20, 18))
        draw.polygon(tail_pts, fill=(255, 255, 255), outline=(22, 20, 18))
        draw.rounded_rectangle(s_box, radius=28, fill=(255, 255, 255), outline=(22, 20, 18), width=4)
        draw.polygon([(bx_mid - 15, s_box[3] - 8), (bx_mid + 9, s_box[3] - 8), (bx_mid - 3, s_box[3] - 1)], fill=(255, 255, 255))

        lines = textwrap.wrap(spec["speech_p1"].upper(), width=28)[:3]
        lh = 28
        sy_txt = s_box[1] + ((s_box[3] - s_box[1]) - len(lines) * lh) // 2 - 2
        for i, ln in enumerate(lines):
            bb = draw.textbbox((0, 0), ln, font=FONT_SPEECH_LG)
            tw = bb[2] - bb[0]
            draw.text((s_box[0] + ((s_box[2] - s_box[0]) - tw) // 2, sy_txt + i * lh), ln, font=FONT_SPEECH_LG, fill=(22, 20, 18))

    else:
        cap_map = {2: spec["caption_p2"], 3: spec["caption_p3"], 4: spec["caption_p4"]}
        cap_txt = cap_map[panel_num].upper()
        draw.rectangle([bx0, by0, bx1, by0 + cap_h], fill=(251, 229, 194), outline=(22, 20, 18), width=4)
        draw.rectangle([bx0, by0, bx0 + 54, by0 + cap_h], fill=(201, 74, 41), outline=(22, 20, 18), width=4)
        draw.text((bx0 + 18, by0 + 10), str(panel_num), font=FONT_DISPLAY_XL, fill=(255, 251, 244))
        f_cap = fit_text_font(draw, cap_txt, (bx1 - bx0) - 74, [FONT_CAPTION_LG, FONT_CAPTION_MD])
        draw.text((bx0 + 68, by0 + 15), cap_txt, font=f_cap, fill=(22, 20, 18))

    if panel_num == 2:
        steps = spec["diag_steps"]
        n_s = len(steps)
        vis_steps = max(1, min(n_s, int(sub_step * n_s) + 1)) if sub_step < 0.99 else n_s
        box_w, box_h = 216, 230
        gap = 68
        total_w = n_s * box_w + (n_s - 1) * gap
        start_x = (W - total_w) // 2
        cy_box = by0 + cap_h + (by1 - by0 - cap_h) // 2 - 16
        for s_i in range(vis_steps):
            _, lbl = steps[s_i]
            scx = start_x + s_i * (box_w + gap) + box_w // 2
            sy1 = cy_box + box_h // 2
            bb = draw.textbbox((0, 0), lbl, font=FONT_LABEL_MD)
            tw = bb[2] - bb[0]
            draw.rounded_rectangle([scx - tw // 2 - 12, sy1 + 14, scx + tw // 2 + 12, sy1 + 46], radius=8, fill=(251, 229, 194), outline=(22, 20, 18), width=2)
            draw.text((scx - tw // 2, sy1 + 18), lbl, font=FONT_LABEL_MD, fill=(22, 20, 18))

    elif panel_num == 3:
        strip_y0 = by0 + cap_h + 36
        kw = (bx1 - bx0 - 80) // 4
        k_labels = ["1. ARCHIVAL TRACE", "2. KEYFRAME K1", "3. KEYFRAME K2", "4. KEYFRAME K4"]
        for k_i, kl in enumerate(k_labels):
            kx0 = bx0 + 32 + k_i * (kw + 6)
            ky0 = strip_y0 + 34
            draw.rectangle([kx0 + 6, ky0 + 6, kx0 + 168, ky0 + 32], fill=(255, 251, 244), outline=(22, 20, 18), width=2)
            draw.text((kx0 + 12, ky0 + 9), kl, font=FONT_CAPTION_MD, fill=(22, 20, 18))

    out_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out_bgr, (bx0, by0), (bx1, by1), BGR_INK, 5)
    return out_bgr


def assemble_4panel_comic_page(spec: dict) -> np.ndarray:
    """
    Assembles the 4 full-resolution panels (`Panel 1..4`) into a crisp 1280x720 2x2 Scott McCloud Comic Page
    (Juxtaposed in SPACE for static reading on posters, studio wing plates, and contact sheets).
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
    """
    Implements McCloud Page 14 ("Each successive frame of a movie is projected on the SAME space — the screen!"):
    Projects each of the 4 panels FULL-SCREEN (`960x540`) in deliberate sequence over 10.0 seconds (120 frames @ 12fps):
      - Frames 00..27 (0.0s - 2.3s): FULL-SCREEN PANEL 1 (Scott + Studio Environment + Big Speech Balloon)
      - Frames 28..55 (2.3s - 4.6s): FULL-SCREEN PANEL 2 (`1 --> 2 --> 3` Diagram)
      - Frames 56..83 (4.6s - 7.0s): FULL-SCREEN PANEL 3 (35mm 4-Keyframe Filmstrip stepping K1 -> K4)
      - Frames 84..119 (7.0s - 10.0s): FULL-SCREEN PANEL 4 (Full-screen 7-plane Quick, Draw! cartoon animation!)
    """
    fid = spec["id"]
    mp4_path = BTS_DIR / f"{fid}.mp4"
    gif_path = BTS_DIR / f"{fid}.gif"
    poster_path = BTS_DIR / f"{fid}_poster.png"
    raw_mp4 = BTS_DIR / f"_{fid}_raw.mp4"
    wav_path = BTS_DIR / f"_{fid}.wav"

    fps = 12
    n_frames = 120  # 10.0 calm, unhurried seconds per film
    W, H = 960, 540

    # Static poster shows the 2x2 Juxtaposed Comic Page (`1280x720`)
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
        duration=330,  # ~3 fps GIF -> 30 frames = 10.0 calm, readable seconds
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
    print("1. Rendering 5 Scott McCloud + Quick, Draw! Studio Chapter Comic Pages...")
    generate_studio_wing_comic_pages()

    office_results = []
    skill_results = []

    print("2. Rendering 10 Virtual Agent Office Full-Screen Sequential Comic Films (10.0s each)...")
    for sp in AGENT_OFFICE_SPECS:
        res = render_single_bts_film(sp)
        office_results.append(res)
        print(f"   [Office {sp['num']:02d}/10] {sp['id']} -> {res['mp4']}")

    print("3. Rendering 8 AI Video Director Skill Full-Screen Sequential Comic Films (10.0s each)...")
    for sp in VIDEO_SKILL_SPECS:
        res = render_single_bts_film(sp)
        skill_results.append(res)
        print(f"   [Skill  {sp['num']-10:02d}/08] {sp['id']} -> {res['mp4']}")

    p_off = render_contact_sheet(
        office_results,
        "PART I: 10 VIRTUAL AGENT OFFICE & WORKTREE COMIC CHAPTERS (UNDERSTANDING COMICS + QUICK, DRAW!)",
        "Static Page = Juxtaposed in Space (4 Panels) · Video = Projected Full-Screen One Panel at a Time in Time (McCloud p. 14)",
        "bts_10_agent_offices_atlas.png",
    )
    p_skl = render_contact_sheet(
        skill_results,
        "PART II: 8 AI VIDEO DIRECTOR & MOTION SKILL COMIC CHAPTERS (UNDERSTANDING COMICS + QUICK, DRAW!)",
        "Amplification Through Simplification · Blood in the Gutter · Show and Tell · Full-Screen Sequential Pacing",
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
        "studio_title": "Understanding Secondhand Motion — A Scott McCloud & Quick, Draw! Sequential Art Guide",
        "agents": [
            {"name": a["name"], "role": a["role"]}
            for a in STUDIO_AGENTS
        ],
        "office_films": office_results,
        "skill_films": skill_results,
        "master_film": "Behind_The_Scenes_Multiplane_Studio_Film.mp4",
    }
    (DATA_DIR / "bts_studio_films.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "bts_studio_data.js").write_text("window.BTS_STUDIO_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
