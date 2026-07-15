
"""
FREUID Challenge 2026 — Docker sandbox entrypoint
==================================================
Sandbox contract (organizers run this):
  Input  : /data/   — flat directory of images, NO CSV
  Output : /submissions/submission.csv

Architecture is identical to train.py:
  - DINOv3Model  (not DINOv2Model)
  - apply_lora(backbone, cfg)  — takes cfg object
  - same LoRALinear, same forward()

All weights baked into the image. --network none compatible.
"""

import argparse, gc, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModel
from tqdm import tqdm

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

BACKBONE_DIR = Path("/app/backbone")
WEIGHTS_DIR  = Path("/app/weights")
FOLD_CKPTS   = [WEIGHTS_DIR / f"best_fold{i}.pt" for i in range(5)]


# ── Config (mirrors train.py Config) ──────────────────────────────────────────
class Config:
    BACKBONE     = str(BACKBONE_DIR)
    EMBED_DIM    = 1024
    IMG_SIZE     = 448
    LORA_R       = 16
    LORA_ALPHA   = 32
    LORA_DROPOUT = 0.05
    USE_DUAL     = False
    BATCH_SIZE   = 32
    AMP_DTYPE    = torch.bfloat16
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS  = 4
    PIN_MEMORY   = True
    PREFETCH     = 2

cfg = Config()


# ── LoRALinear — identical to train.py ────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, original, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original = original
        for p in self.original.parameters():
            p.requires_grad = False
        dtype = original.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, original.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, r, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self):
        return self.original.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        return self.original(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


# ── apply_lora — identical signature to train.py ──────────────────────────────
def apply_lora(model, cfg):
    replaced = 0
    for block in model.timm_model.blocks:
        for name in ["qkv", "proj"]:
            orig = getattr(block.attn, name, None)
            if isinstance(orig, nn.Linear):
                setattr(block.attn, name, LoRALinear(orig, cfg.LORA_R, cfg.LORA_ALPHA, cfg.LORA_DROPOUT))
                replaced += 1
        for name in ["fc1", "fc2"]:
            orig = getattr(block.mlp, name, None)
            if isinstance(orig, nn.Linear):
                setattr(block.mlp, name, LoRALinear(orig, cfg.LORA_R, cfg.LORA_ALPHA, cfg.LORA_DROPOUT))
                replaced += 1
    print(f"  LoRA: {replaced} layers replaced")
    return replaced


# ── DINOv3Model — identical to train.py ───────────────────────────────────────
class DINOv3Model(nn.Module):
    def __init__(self, cfg, num_types):
        super().__init__()
        self.use_dual = cfg.USE_DUAL
        self.backbone = AutoModel.from_pretrained(cfg.BACKBONE, local_files_only=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        apply_lora(self.backbone, cfg)
        self.head_fraud = nn.Linear(cfg.EMBED_DIM, 1)
        if self.use_dual:
            self.head_type = nn.Linear(cfg.EMBED_DIM, num_types)

    def forward(self, x):
        features = self.backbone(x).last_hidden_state[:, 0]
        logit_fraud = self.head_fraud(features).squeeze(1)
        if self.use_dual:
            return logit_fraud, self.head_type(features)
        return logit_fraud


# ── Dataset — flat directory, no CSV ──────────────────────────────────────────
class FlatImageDataset(Dataset):
    """id = filename stem (no extension), as required by sandbox contract."""
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = np.array(Image.open(p).convert("RGB"))
        img = self.transform(image=img)["image"]
        return img, p.stem


# ── TTA transforms — identical to inference.py ────────────────────────────────
def get_tta_transforms(img_size):
    base = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
    hflip = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
    bright = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])
    return [base, hflip, bright]


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    all_scores, all_ids = [], []
    for imgs, img_ids in tqdm(loader, leave=False, desc="  inference"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype):
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
        all_scores.append(torch.sigmoid(logits).cpu().float().numpy())
        all_ids.extend(img_ids)
    return np.concatenate(all_scores), all_ids


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data",
                        help="Flat directory of test images (no CSV)")
    parser.add_argument("--output",   default="/submissions/submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    device    = cfg.DEVICE
    amp_dtype = cfg.AMP_DTYPE if device == "cuda" else torch.float32
    print(f"Device    : {device}")
    print(f"AMP dtype : {amp_dtype}")

    # Collect images
    data_dir = Path(args.data_dir)
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXT)
    print(f"Images    : {len(paths)} found in {data_dir}")
    if not paths:
        raise RuntimeError(f"No images found in {data_dir}")

    tta_transforms = get_tta_transforms(cfg.IMG_SIZE)
    accumulated = np.zeros(len(paths))
    id_order    = None
    n_total     = 0

    for fold_idx, ckpt_path in enumerate(FOLD_CKPTS):
        if not ckpt_path.exists():
            print(f"⚠  Fold {fold_idx}: {ckpt_path} not found — skipped")
            continue

        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
        _num_types = ckpt.get("num_types", 5)
        cfg.USE_DUAL = ckpt.get("use_dual", False)

        model = DINOv3Model(cfg, _num_types).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        print(f"\nFold {fold_idx} | val FREUID={ckpt.get('freuid', 0):.4f} | dual={cfg.USE_DUAL}")

        for ti, tf in enumerate(tta_transforms):
            ds = FlatImageDataset(paths, tf)
            loader = DataLoader(
                ds,
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.NUM_WORKERS,
                pin_memory=cfg.PIN_MEMORY,
                prefetch_factor=cfg.PREFETCH,
                persistent_workers=True,
            )
            scores, ids = predict(model, loader, device, amp_dtype)
            accumulated += scores
            n_total += 1
            if id_order is None:
                id_order = ids
            print(f"  TTA {ti+1}/3 | range [{scores.min():.4f}, {scores.max():.4f}]")

        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if n_total == 0:
        raise RuntimeError("No checkpoints loaded — check /app/weights/")

    final = accumulated / n_total
    print(f"\nEnsemble : {n_total} passes | range [{final.min():.4f}, {final.max():.4f}] | mean {final.mean():.4f}")

    # Write output — one row per image, id = stem
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": id_order, "label": final}).to_csv(out_path, index=False)
    print(f"✅  Saved → {out_path}  ({len(final)} rows)")


if __name__ == "__main__":
    main()
