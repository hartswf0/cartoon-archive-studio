#!/usr/bin/env python3
"""
SCION / MULTIPLANE ANIMATION STUDIO — Stage 1 & 2 Neural Depth + SAM Pass
Runs:
  1. `depth-anything/Depth-Anything-V2-Small-hf` (loaded with strict=True from local safetensors)
     on every shot's archival frames (`frames-src/f00..f32.png`) and environment plates
     to compute dense monocular depth maps Z(x,y,t) and stratify the scene into 7 physical multiplane depth layers:
       PLANE 0 — EXTREME FOREGROUND (z=0.85)
       PLANE 1 — FOREGROUND (z=1.20)
       PLANE 2 — CHARACTER ACTION (z=2.00)
       PLANE 3 — SECONDARY ACTION (z=2.50)
       PLANE 4 — MIDGROUND (z=3.50)
       PLANE 5 — DISTANT BACKGROUND (z=6.00)
       PLANE 6 — SKY / ATMOSPHERE (z=10.00)
  2. `IDEA-Research/grounding-dino-base` + `facebook/sam-vit-base` on both:
       (a) the archival video frames (`frames-src/f00..f32.png`) with shot-specific anatomical & volume-conservation
           priors so every shot tracks the true primary subject without bbox popping or shrinking, and
       (b) the 4 generated character keyframe drawings (`keys/art-key-1..4.png`) to extract clean SAM silhouettes,
           verify silhouette readability (Section IX), and fix stroke extraction on thin interior lines.
  3. Saves inspectable artifacts per shot:
       - `shots/<slug>/depth-pass.png`
       - `shots/<slug>/sam-pass.png`
       - `shots/<slug>/depth_sam_analysis.json`
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

ROOT = Path("/Users/gaia/WORLDBUILDING-TERMS/cartoon_archive_studio")
SHOTS_DIR = ROOT / "shots"
DEPTH_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/5426e4f0f36572d16453bbda7a8389317b1bef99"
)


class PatchEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 384, 14, stride=14)


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.patch_embeddings = PatchEmbeddings()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1370, 384))

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_embeddings.projection(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        pos_cls = self.position_embeddings[:, :1, :]
        pos_p = self.position_embeddings[:, 1:, :].reshape(1, 37, 37, 384).permute(0, 3, 1, 2)
        pos_p = (
            F.interpolate(pos_p, size=(H // 14, W // 14), mode="bicubic", align_corners=False)
            .permute(0, 2, 3, 1)
            .reshape(1, -1, 384)
        )
        return torch.cat([cls + pos_cls, p + pos_p], dim=1)


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(384, 384)
        self.key = nn.Linear(384, 384)
        self.value = nn.Linear(384, 384)


class AttentionOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(384, 384)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.output = AttentionOut()

    def forward(self, x):
        B, N, C = x.shape
        q = self.attention.query(x).reshape(B, N, 6, 64).transpose(1, 2)
        k = self.attention.key(x).reshape(B, N, 6, 64).transpose(1, 2)
        v = self.attention.value(x).reshape(B, N, 6, 64).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.output.dense(a)


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(384))

    def forward(self, x):
        return x * self.lambda1


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DinoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attention = Attention()
        self.layer_scale1 = LayerScale()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = MLP()
        self.layer_scale2 = LayerScale()

    def forward(self, x):
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([DinoBlock() for _ in range(12)])


class DinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.encoder = DinoEncoder()
        self.layernorm = nn.LayerNorm(384, eps=1e-6)


class ResidualBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.convolution1 = nn.Conv2d(c, c, 3, padding=1, bias=True)
        self.convolution2 = nn.Conv2d(c, c, 3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(x, inplace=False)
        out = F.relu(self.convolution1(out), inplace=False)
        out = self.convolution2(out)
        return x + out


class FusionLayer(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.projection = nn.Conv2d(c, c, 1, bias=True)
        self.residual_layer1 = ResidualBlock(c)
        self.residual_layer2 = ResidualBlock(c)

    def forward(self, x, res=None, size=None):
        if res is not None:
            x = x + self.residual_layer1(res)
        x = self.residual_layer2(x)
        x = (
            F.interpolate(x, size=size, mode="bilinear", align_corners=True)
            if size
            else F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        )
        return self.projection(x)


class ReassembleLayer(nn.Module):
    def __init__(self, in_c, out_c, factor):
        super().__init__()
        self.projection = nn.Conv2d(in_c, out_c, 1, bias=True)
        if factor == 4:
            self.resize = nn.ConvTranspose2d(out_c, out_c, 4, stride=4, padding=0)
        elif factor == 2:
            self.resize = nn.ConvTranspose2d(out_c, out_c, 2, stride=2, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor == 0.5:
            self.resize = nn.Conv2d(out_c, out_c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.resize(self.projection(x))


class ReassembleStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ReassembleLayer(384, 48, 4),
                ReassembleLayer(384, 96, 2),
                ReassembleLayer(384, 192, 1),
                ReassembleLayer(384, 384, 0.5),
            ]
        )


class FusionStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FusionLayer(64) for _ in range(4)])


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.reassemble_stage = ReassembleStage()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(48, 64, 3, padding=1, bias=False),
                nn.Conv2d(96, 64, 3, padding=1, bias=False),
                nn.Conv2d(192, 64, 3, padding=1, bias=False),
                nn.Conv2d(384, 64, 3, padding=1, bias=False),
            ]
        )
        self.fusion_stage = FusionStage()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)


class DepthAnythingV2Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DinoBackbone()
        self.neck = Neck()
        self.head = Head()

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        ph, pw = H // 14, W // 14
        x = self.backbone.embeddings(pixel_values)
        hs = []
        for i, blk in enumerate(self.backbone.encoder.layer):
            x = blk(x)
            if (i + 1) in (3, 6, 9, 12):
                hs.append(self.backbone.layernorm(x))
        feats = []
        for idx, h in enumerate(hs):
            patch_t = h[:, 1:, :].transpose(1, 2).reshape(B, 384, ph, pw)
            r = self.neck.reassemble_stage.layers[idx](patch_t)
            feats.append(self.neck.convs[idx](r))
        u = self.neck.fusion_stage.layers[0](feats[3], None, size=feats[2].shape[-2:])
        u = self.neck.fusion_stage.layers[1](u, feats[2], size=feats[1].shape[-2:])
        u = self.neck.fusion_stage.layers[2](u, feats[1], size=feats[0].shape[-2:])
        u = self.neck.fusion_stage.layers[3](u, feats[0], size=(ph * 14, pw * 14))
        u = self.head.conv1(u)
        u = F.interpolate(u, size=(H, W), mode="bilinear", align_corners=True)
        u = F.relu(self.head.conv2(u), inplace=False)
        u = F.relu(self.head.conv3(u), inplace=False)
        return u.squeeze(1)


SHOT_CONFIGS = {
    "giraffe-walks-past": {
        "category": "giraffe",
        "dino_prompt": "giraffe . tall walking giraffe .",
        "key_frames": [0, 10, 20, 32],
        "canonical_trajectory": [
            [45.0, 38.0, 295.0, 448.0],
            [125.0, 36.0, 375.0, 448.0],
            [215.0, 38.0, 465.0, 446.0],
            [305.0, 42.0, 555.0, 444.0],
        ],
        "action_verbs": "Tall adult giraffe perceives camera -> shifts weight onto left foreleg -> neck swings in figure-8 arc -> strides across savannah behind foreground grass while two juveniles watch.",
        "emotion": "Calm vigilance -> rhythmic poise -> curious glance",
    },
    "dog": {
        "category": "dog",
        "dino_prompt": "dog . small terrier dog .",
        "key_frames": [0, 10, 21, 32],
        "canonical_trajectory": [
            [155.0, 215.0, 335.0, 375.0],
            [185.0, 225.0, 368.0, 385.0],
            [230.0, 250.0, 420.0, 408.0],
            [260.0, 272.0, 455.0, 425.0],
        ],
        "action_verbs": "Terrier looks up at owner's coat -> ears perk at water basin -> pivots forequarters with lifted paw -> trots down diagonal path -> lowers muzzle into basin to drink.",
        "emotion": "Attentive loyalty -> sudden curiosity -> eager thirst -> satisfied drinking",
    },
    "horse": {
        "category": "horse",
        "dino_prompt": "horse . horse head and neck .",
        "key_frames": [0, 11, 21, 32],
        "canonical_trajectory": [
            [45.0, 75.0, 345.0, 430.0],
            [75.0, 95.0, 395.0, 438.0],
            [110.0, 130.0, 455.0, 448.0],
            [140.0, 165.0, 495.0, 455.0],
        ],
        "action_verbs": "Horse enters corral frame with high head -> spots fresh hay mound -> arches muscular neck over rail -> lowers long muzzle -> pulls hay tuft through rails.",
        "emotion": "Alert inspection -> purposeful appetite -> contented grazing",
    },
    "rabbit": {
        "category": "rabbit",
        "dino_prompt": "rabbit . white bunny .",
        "key_frames": [0, 10, 20, 32],
        "canonical_trajectory": [
            [135.0, 225.0, 315.0, 435.0],
            [195.0, 235.0, 385.0, 442.0],
            [245.0, 255.0, 445.0, 445.0],
            [335.0, 265.0, 565.0, 442.0],
        ],
        "action_verbs": "Rabbit peeks past doorway jamb -> tentative forepaw step -> settles into compressed crouch -> ears swivel -> explosive bound across floorboards toward right exit.",
        "emotion": "Timid hesitation -> sensory check -> coiled decision -> sudden bolt",
    },
    "bicycle": {
        "category": "bicycle",
        "dino_prompt": "bicycle . cyclist legs and wheels .",
        "key_frames": [0, 11, 21, 32],
        "canonical_trajectory": [
            [225.0, 175.0, 610.0, 458.0],
            [175.0, 178.0, 560.0, 458.0],
            [130.0, 176.0, 515.0, 458.0],
            [85.0, 175.0, 470.0, 458.0],
        ],
        "action_verbs": "Rider's knee crests top dead center -> drives downward power stroke -> diamond frame glides left across street -> ankle flicks through bottom dead center -> closes gap on lead wheel.",
        "emotion": "Focused cadence -> mechanical momentum -> effortless cruise",
    },
    "helicopter": {
        "category": "helicopter",
        "dino_prompt": "helicopter . aircraft .",
        "key_frames": [0, 11, 21, 32],
        "canonical_trajectory": [
            [165.0, 115.0, 515.0, 375.0],
            [145.0, 98.0, 495.0, 352.0],
            [125.0, 78.0, 475.0, 330.0],
            [105.0, 58.0, 455.0, 308.0],
        ],
        "action_verbs": "Helicopter banks into turn with nose dipped -> main rotor bites air -> fuselage levels against cloud deck -> pitches nose upward -> climbs briskly above distant hangar.",
        "emotion": "Heavy mechanical torque -> stabilized lift -> soaring ascent",
    },
    "sailboat": {
        "category": "sailboat",
        "dino_prompt": "sailboat . sailing boat .",
        "key_frames": [0, 11, 21, 32],
        "canonical_trajectory": [
            [195.0, 135.0, 395.0, 392.0],
            [235.0, 122.0, 448.0, 400.0],
            [275.0, 110.0, 502.0, 408.0],
            [315.0, 98.0, 555.0, 416.0],
        ],
        "action_verbs": "Sailboat catches harbor gust -> heels slightly as mainsail bellies out -> cuts across tugboat's distant wake -> glides forward into right foreground with rippling mast reflection.",
        "emotion": "Quiet anticipation of wind -> buoyant catch -> serene maritime glide",
    },
    "windmill": {
        "category": "windmill",
        "dino_prompt": "windmill . windmill tower and blades .",
        "key_frames": [0, 11, 21, 32],
        "canonical_trajectory": [
            [135.0, 42.0, 505.0, 448.0],
            [135.0, 42.0, 505.0, 448.0],
            [135.0, 42.0, 505.0, 448.0],
            [135.0, 42.0, 505.0, 448.0],
        ],
        "action_verbs": "Gust rustles framing pine boughs -> top lattice sail catches wind -> heavy wooden hub creaks into torque -> four sails sweep a full 90-degree quadrant against drifting clouds.",
        "emotion": "Grounded architectural permanence -> rhythmic wind-driven pulse",
    },
}


def compute_depth_map(bgr_img, depth_model, device, W=640, H=480):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    inp_rs = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (inp_rs - mean) / std
    t = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = depth_model(t)
        pred_hw = F.interpolate(pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False).squeeze()
    d = pred_hw.cpu().numpy()
    d_min, d_max = float(np.percentile(d, 2)), float(np.percentile(d, 98))
    disp = np.clip((d - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    z_map = 0.85 + (1.0 - disp) ** 1.35 * 9.15
    return disp, z_map


def to_dev(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, dtype=torch.float32) if v.is_floating_point() else v.to(device)
        else:
            out[k] = v
    return out


def run_dino_sam(pil_img, prompt, canonical_box, gd_proc, gd_model, sam_proc, sam_model, device, W=640, H=480):
    with torch.no_grad():
        inp = to_dev(gd_proc(images=pil_img, text=prompt, return_tensors="pt"), device)
        out = gd_model(**inp)
        res = gd_proc.post_process_grounded_object_detection(
            out, inp["input_ids"], threshold=0.12, text_threshold=0.12, target_sizes=[(H, W)]
        )[0]

    cx_can = 0.5 * (canonical_box[0] + canonical_box[2])
    cy_can = 0.5 * (canonical_box[1] + canonical_box[3])
    w_can = canonical_box[2] - canonical_box[0]
    h_can = canonical_box[3] - canonical_box[1]

    best_box = list(canonical_box)
    best_score = 0.0
    all_boxes = []

    if len(res["boxes"]) > 0:
        boxes_np = res["boxes"].cpu().numpy()
        scores_np = res["scores"].cpu().numpy()
        for b, sc in zip(boxes_np, scores_np):
            all_boxes.append({"box": [round(float(x), 1) for x in b], "score": round(float(sc), 3)})
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if bw * bh < 0.04 * (W * H) or bw * bh > 0.92 * (W * H):
                continue
            bcx = 0.5 * (b[0] + b[2])
            bcy = 0.5 * (b[1] + b[3])
            dist_pen = np.hypot(bcx - cx_can, bcy - cy_can) / 300.0
            size_ratio = min(bw / max(w_can, 1.0), max(w_can, 1.0) / max(bw, 1.0)) * min(
                bh / max(h_can, 1.0), max(h_can, 1.0) / max(bh, 1.0)
            )
            combined = float(sc) * (0.5 + 0.5 * size_ratio) - 0.25 * dist_pen
            if combined > best_score:
                best_score = combined
                alpha = 0.35
                best_box = [
                    float(alpha * b[0] + (1.0 - alpha) * canonical_box[0]),
                    float(alpha * b[1] + (1.0 - alpha) * canonical_box[1]),
                    float(alpha * b[2] + (1.0 - alpha) * canonical_box[2]),
                    float(alpha * b[3] + (1.0 - alpha) * canonical_box[3]),
                ]

    with torch.no_grad():
        sam_raw = sam_proc(pil_img, input_boxes=[[best_box]], return_tensors="pt")
        orig_sz = sam_raw["original_sizes"].cpu()
        resh_sz = sam_raw["reshaped_input_sizes"].cpu()
        sam_inp = to_dev(sam_raw, device)
        sam_out = sam_model(**sam_inp)
        masks_t = sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            orig_sz,
            resh_sz,
        )[0]
        iou_scores = sam_out.iou_scores.cpu().numpy()[0, 0]
        best_m_idx = int(np.argmax(iou_scores))
        sam_mask = masks_t[0, best_m_idx].numpy().astype(np.uint8)
        sam_iou = float(iou_scores[best_m_idx])

    return best_box, sam_mask, sam_iou, all_boxes


def segment_keyframe_drawing_with_sam(art_bgr, sam_proc, sam_model, device, slug=""):
    H, W = art_bgr.shape[:2]
    gray = cv2.cvtColor(art_bgr, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    ink = cv2.bitwise_or(otsu, (gray < 212).astype(np.uint8) * 255)
    ink = cv2.bitwise_or(ink, adapt & (gray < 232).astype(np.uint8) * 255)

    ink[:6, :] = 0
    ink[-6:, :] = 0
    ink[:, :6] = 0
    ink[:, -6:] = 0

    if slug == "horse":
        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        clean_ink = np.zeros_like(ink)
        for i in range(1, num_l):
            x, y, w, h, area = stats[i]
            if area < 8:
                continue
            if w > 0.85 * W and h < 0.18 * H:
                continue
            clean_ink[labels == i] = 255
        ink = clean_ink

    ys, xs = np.where(ink > 0)
    if len(xs) > 10:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        box = [float(max(0, x0 - 8)), float(max(0, y0 - 8)), float(min(W - 1, x1 + 8)), float(min(H - 1, y1 + 8))]
    else:
        box = [float(0.15 * W), float(0.15 * H), float(0.85 * W), float(0.85 * H)]

    pil_art = Image.fromarray(cv2.cvtColor(art_bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        sam_raw = sam_proc(pil_art, input_boxes=[[box]], return_tensors="pt")
        orig_sz = sam_raw["original_sizes"].cpu()
        resh_sz = sam_raw["reshaped_input_sizes"].cpu()
        sam_inp = to_dev(sam_raw, device)
        sam_out = sam_model(**sam_inp)
        masks_t = sam_proc.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(),
            orig_sz,
            resh_sz,
        )[0]
        iou_scores = sam_out.iou_scores.cpu().numpy()[0, 0]
        best_idx = int(np.argmax(iou_scores))
        sam_mask = masks_t[0, best_idx].numpy().astype(np.uint8)
        sam_iou = float(iou_scores[best_idx])

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_close, iterations=2)
    ff = closed.copy()
    mask_ff = np.zeros((H + 2, W + 2), dtype=np.uint8)
    cv2.floodFill(ff, mask_ff, (0, 0), 128)
    cel_mask = ((ff != 128) | (ink > 0)).astype(np.uint8)

    sam_area_ratio = float(sam_mask.sum()) / max(float((box[2] - box[0]) * (box[3] - box[1])), 1.0)
    if 0.12 <= sam_area_ratio <= 0.85:
        cel_mask = np.clip(cel_mask | (sam_mask & (cv2.dilate(cel_mask, k_close, iterations=2))), 0, 1)

    return ink, cel_mask, [int(round(v)) for v in box], sam_iou


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[SCION Depth + SAM Pass] Loading Depth-Anything-V2, GroundingDINO, and SAM on {device}...", flush=True)

    depth_model = DepthAnythingV2Small()
    sd = load_file(os.path.join(DEPTH_SNAP, "model.safetensors"))
    depth_model.load_state_dict(sd, strict=True)
    depth_model = depth_model.to(device).eval()

    gd_proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base", local_files_only=True)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base", local_files_only=True
    ).to(device).eval()

    sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-base", local_files_only=True)
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base", local_files_only=True).to(device).eval()

    for slug, cfg in SHOT_CONFIGS.items():
        sdir = SHOTS_DIR / slug
        src_info = json.loads((sdir / "source.json").read_text(encoding="utf-8"))
        vpath = (ROOT.parent / src_info["source"]["local_video_path"]).resolve()
        fdir = sdir / "frames-src"
        kdir = sdir / "keys"
        pdir = sdir / "planes"
        fdir.mkdir(exist_ok=True)
        pdir.mkdir(exist_ok=True)

        N = 33
        W, H = 640, 480
        t0, t1 = src_info["interval"]
        sample_times = np.linspace(t0, t1, N)
        cap = cv2.VideoCapture(str(vpath))
        v_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        frame_paths = []
        for idx_f, ts in enumerate(sample_times):
            f_pos = int(min(max(round(ts * v_fps), 0), max(total_f - 1, 0)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_pos)
            ok, bgr_raw = cap.read()
            fp = fdir / f"f{idx_f:02d}.png"
            if ok and bgr_raw is not None:
                bgr_rs = cv2.resize(bgr_raw, (W, H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(fp), bgr_rs)
            frame_paths.append(fp)
        cap.release()

        key_indices = cfg["key_frames"]
        can_traj = cfg["canonical_trajectory"]

        can_boxes = []
        for f in range(N):
            k_pos = 0
            for idx in range(len(key_indices) - 1):
                if key_indices[idx] <= f <= key_indices[idx + 1]:
                    k_pos = idx
                    break
            f0, f1 = key_indices[k_pos], key_indices[k_pos + 1]
            t = 0.0 if f1 == f0 else (f - f0) / float(f1 - f0)
            te = t * t * (3.0 - 2.0 * t)
            b0, b1 = can_traj[k_pos], can_traj[k_pos + 1]
            can_boxes.append([(1.0 - te) * b0[c] + te * b1[c] for c in range(4)])

        depth_rows = []
        sam_rows = []
        frame_records = []
        plane_depth_hist = {
            "P0_extreme_fore": 0.0,
            "P1_fore": 0.0,
            "P2_char": 0.0,
            "P3_secondary": 0.0,
            "P4_mid": 0.0,
            "P5_far": 0.0,
            "P6_sky": 0.0,
        }

        sam_ious = []
        char_depths = []

        for f_idx, fp in enumerate(frame_paths):
            bgr = cv2.imread(str(fp))
            bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            disp, z_map = compute_depth_map(bgr, depth_model, device, W, H)

            if f_idx in key_indices or f_idx % 4 == 0:
                box, sam_mask, sam_iou, _ = run_dino_sam(
                    pil_img,
                    cfg["dino_prompt"],
                    can_boxes[f_idx],
                    gd_proc,
                    gd_model,
                    sam_proc,
                    sam_model,
                    device,
                    W,
                    H,
                )
            else:
                box = can_boxes[f_idx]
                with torch.no_grad():
                    sam_raw = sam_proc(pil_img, input_boxes=[[box]], return_tensors="pt")
                    orig_sz = sam_raw["original_sizes"].cpu()
                    resh_sz = sam_raw["reshaped_input_sizes"].cpu()
                    sam_inp = to_dev(sam_raw, device)
                    sam_out = sam_model(**sam_inp)
                    masks_t = sam_proc.image_processor.post_process_masks(
                        sam_out.pred_masks.cpu(),
                        orig_sz,
                        resh_sz,
                    )[0]
                    iou_scores = sam_out.iou_scores.cpu().numpy()[0, 0]
                    best_m_idx = int(np.argmax(iou_scores))
                    sam_mask = masks_t[0, best_m_idx].numpy().astype(np.uint8)
                    sam_iou = float(iou_scores[best_m_idx])

            sam_ious.append(sam_iou)

            if sam_mask.sum() > 20:
                z_char = float(np.median(z_map[sam_mask > 0]))
            else:
                z_char = 2.0
            char_depths.append(z_char)

            plane_depth_hist["P0_extreme_fore"] += float((z_map < 1.05).mean()) / N
            plane_depth_hist["P1_fore"] += float(((z_map >= 1.05) & (z_map < 1.65)).mean()) / N
            plane_depth_hist["P2_char"] += float((sam_mask > 0).mean()) / N
            plane_depth_hist["P3_secondary"] += float(((z_map >= 1.65) & (z_map < 2.80) & (sam_mask == 0)).mean()) / N
            plane_depth_hist["P4_mid"] += float(((z_map >= 2.80) & (z_map < 4.80)).mean()) / N
            plane_depth_hist["P5_far"] += float(((z_map >= 4.80) & (z_map < 7.50)).mean()) / N
            plane_depth_hist["P6_sky"] += float((z_map >= 7.50).mean()) / N

            ys_m, xs_m = np.where(sam_mask > 0)
            if len(xs_m) > 20:
                cx_m = float(xs_m.mean())
                cy_m = float(ys_m.mean())
                y_bot = float(np.percentile(ys_m, 96))
                bot_xs = xs_m[ys_m >= np.percentile(ys_m, 90)]
                x_contact = float(np.median(bot_xs)) if len(bot_xs) > 0 else cx_m
            else:
                cx_m = 0.5 * (box[0] + box[2])
                cy_m = 0.5 * (box[1] + box[3])
                x_contact = cx_m
                y_bot = box[3]

            frame_records.append(
                {
                    "frame": f_idx,
                    "bbox": [round(x, 1) for x in box],
                    "sam_iou": round(sam_iou, 3),
                    "z_char": round(z_char, 3),
                    "centroid": [round(cx_m, 1), round(cy_m, 1)],
                    "contact": [round(x_contact, 1), round(y_bot, 1)],
                    "mask_area_pct": round(float((sam_mask > 0).mean() * 100.0), 2),
                }
            )

            if f_idx in key_indices:
                k_slot = key_indices.index(f_idx) + 1
                disp_u8 = (disp * 255).astype(np.uint8)
                depth_col = cv2.applyColorMap(disp_u8, cv2.COLORMAP_INFERNO)
                cv2.putText(
                    depth_col,
                    f"f{f_idx:02d} Depth-Anything-V2 (Z_char={z_char:.2f})",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                strat = np.zeros((H, W, 3), dtype=np.uint8)
                strat[z_map >= 7.50] = (232, 218, 198)
                strat[(z_map >= 4.80) & (z_map < 7.50)] = (195, 178, 156)
                strat[(z_map >= 2.80) & (z_map < 4.80)] = (150, 138, 118)
                strat[(z_map >= 1.65) & (z_map < 2.80)] = (112, 134, 102)
                strat[(z_map >= 1.05) & (z_map < 1.65)] = (72, 95, 118)
                strat[z_map < 1.05] = (45, 58, 82)
                strat[sam_mask > 0] = (48, 92, 215)
                cv2.putText(
                    strat,
                    "7-Plane Stratification (P0..P6 + SAM P2 Char)",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                sam_vis = bgr.copy()
                overlay = sam_vis.copy()
                overlay[sam_mask > 0] = (36, 195, 95)
                sam_vis = cv2.addWeighted(sam_vis, 0.62, overlay, 0.38, 0)
                cnts, _ = cv2.findContours(sam_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(sam_vis, cnts, -1, (20, 255, 100), 2, lineType=cv2.LINE_AA)
                bx0, by0, bx1, by1 = [int(round(v)) for v in box]
                cv2.rectangle(sam_vis, (bx0, by0), (bx1, by1), (0, 210, 255), 2)
                cv2.circle(sam_vis, (int(round(x_contact)), int(round(y_bot))), 6, (0, 60, 255), -1)
                cv2.putText(
                    sam_vis,
                    f"K0{k_slot} (f{f_idx:02d}) DINO+SAM IoU={sam_iou:.3f}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                art_p = kdir / f"art-key-{k_slot}.png"
                art_bgr = cv2.imread(str(art_p))
                art_bgr = cv2.resize(art_bgr, (W, H), interpolation=cv2.INTER_AREA)
                ink_k, cel_k, box_k, iou_k = segment_keyframe_drawing_with_sam(
                    art_bgr, sam_proc, sam_model, device, slug=slug
                )
                cv2.imwrite(str(kdir / f"sam-cel-{k_slot}.png"), cel_k * 255)
                cv2.imwrite(str(kdir / f"sam-ink-{k_slot}.png"), ink_k)

                art_vis = np.full((H, W, 3), (235, 242, 247), dtype=np.uint8)
                art_vis[cel_k > 0] = (195, 232, 210)
                cnts_k, _ = cv2.findContours(cel_k, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(art_vis, cnts_k, -1, (40, 150, 75), 2, lineType=cv2.LINE_AA)
                art_vis[ink_k > 0] = (18, 20, 22)
                cv2.rectangle(art_vis, (box_k[0], box_k[1]), (box_k[2], box_k[3]), (180, 70, 30), 2)
                cv2.putText(
                    art_vis,
                    f"Drawn K0{k_slot} SAM Cel Check (IoU={iou_k:.3f})",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )

                depth_row = np.hstack(
                    [
                        cv2.resize(bgr, (320, 240)),
                        cv2.resize(depth_col, (320, 240)),
                        cv2.resize(strat, (320, 240)),
                    ]
                )
                sam_row = np.hstack(
                    [
                        cv2.resize(sam_vis, (320, 240)),
                        cv2.resize(art_vis, (320, 240)),
                        cv2.resize(strat, (320, 240)),
                    ]
                )
                depth_rows.append(depth_row)
                sam_rows.append(sam_row)

        plate_p = sdir / "scene-plate.png"
        plate_bgr = cv2.imread(str(plate_p))
        plate_bgr = cv2.resize(plate_bgr, (W, H), interpolation=cv2.INTER_AREA)
        _, plate_z = compute_depth_map(plate_bgr, depth_model, device, W, H)
        np.save(str(sdir / "scene_plate_z.npy"), plate_z.astype(np.float32))

        depth_grid = np.vstack(depth_rows)
        sam_grid = np.vstack(sam_rows)
        cv2.imwrite(str(sdir / "depth-pass.png"), depth_grid)
        cv2.imwrite(str(sdir / "sam-pass.png"), sam_grid)

        analysis = {
            "shot_id": slug,
            "category": cfg["category"],
            "action_verbs": cfg["action_verbs"],
            "emotion_arc": cfg["emotion"],
            "depth_model": "depth-anything/Depth-Anything-V2-Small-hf (strict=True)",
            "sam_model": "facebook/sam-vit-base + IDEA-Research/grounding-dino-base",
            "mean_sam_iou": round(float(np.mean(sam_ious)), 4),
            "mean_char_depth_z": round(float(np.mean(char_depths)), 3),
            "plane_depth_distribution": {k: round(v, 4) for k, v in plane_depth_hist.items()},
            "frames": frame_records,
        }
        (sdir / "depth_sam_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        print(
            f"  [OK] {slug:20s}: mean_SAM_IoU={analysis['mean_sam_iou']:.3f}, "
            f"mean_Z_char={analysis['mean_char_depth_z']:.2f}, saved depth-pass.png & sam-pass.png",
            flush=True,
        )


if __name__ == "__main__":
    main()
