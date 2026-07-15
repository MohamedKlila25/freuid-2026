"""
FREUID Challenge 2026 — DINOv3-Base + LoRA + Focal Loss (v2 ROBUST)
====================================================================
Améliorations vs v1 :
  - LoraLinear vide supprimée
  - LOTO (LeaveOneGroupOut) pour validation réaliste
  - Dual Supervision réactivée (head_type améliore OOD)
  # this encourages overfitting 
  - TTA étendu (original + hflip + brightness)
  # this envourages diversity 
  - Sauvegarde complète du state_dict pour prédiction standalone
  - num_types hardcodé dans la cell prédiction
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — IMPORTS + CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

from transformers import Dinov2Model
import gc, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import LeaveOneGroupOut

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import roc_curve, auc as sk_auc
from collections import Counter
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import AutoModel
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark     = True
torch.set_float32_matmul_precision('high')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]







class Config:
    DATA_DIR   = Path("/kaggle/input/datasets/coreclock/the-freuid-challenge-2026-ijcai-ecai-mine")
    TRAIN_CSV  = DATA_DIR / 'train_labels.csv'
    TRAIN_DIR  = DATA_DIR / 'train'
    TEST_DIR   = DATA_DIR / 'public_test' / 'public_test'
    OUTPUT_DIR = Path('/kaggle/working/kaggle_dino')

    # ── DINOv2-Base ───────────────────────────────────────────────────
    BACKBONE  = "/kaggle/input/datasets/lizhiyaya/dinov3-large"
    EMBED_DIM = 1024   # Large = 1024
    IMG_SIZE  = 448

    # ── LoRA ──────────────────────────────────────────────────────────
    LORA_R       = 16
    LORA_ALPHA   = 32
    LORA_DROPOUT = 0.05

    # ── Dual Supervision ──────────────────────────────────────────────
    USE_DUAL =False # True     # True = head_type activé (meilleure OOD)
    AUX_WEIGHT = 0.3

    # ── Focal Loss ────────────────────────────────────────────────────
    FOCAL_ALPHA = 0.2
    FOCAL_GAMMA = 2.0

    # ── Anti-overfitting ──────────────────────────────────────────────
    LABEL_SMOOTH = 0.05
    MIXUP_ALPHA  = 0.4
    MIXUP_PROB   = 0.3

    # ── Entraînement ──────────────────────────────────────────────────
    EPOCHS        = 20
    BATCH_SIZE    = 16
    GRAD_ACCUM    = 1
    LR            = 2e-5
    WEIGHT_DECAY  = 0.01
    WARMUP_EPOCHS = 1
    CLIP_GRAD     = 1.0
    AMP_DTYPE     = torch.bfloat16
    PATIENCE      = 4

    SEED        = 342
    DEVICE      = 'cuda'
    NUM_WORKERS = 12
    PIN_MEMORY  = True
    PREFETCH    = 4


cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED); random.seed(cfg.SEED)

print(f'Backbone  : {cfg.BACKBONE}')
print(f'IMG       : {cfg.IMG_SIZE} | Batch: {cfg.BATCH_SIZE}×{cfg.GRAD_ACCUM}={cfg.BATCH_SIZE*cfg.GRAD_ACCUM}')
print(f'LoRA      : r={cfg.LORA_R} alpha={cfg.LORA_ALPHA}')
print(f'Dual sup  : {cfg.USE_DUAL} (AUX_WEIGHT={cfg.AUX_WEIGHT})')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — LoRA
# ═══════════════════════════════════════════════════════════════════════════════

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

    # ← ajoute ces deux propriétés
    @property
    def weight(self):
        return self.original.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        return self.original(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


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

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — FOCAL LOSS
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt     = torch.exp(-bce)
        alpha_t = torch.where(targets >= 0.5, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()

print('✓ Focal Loss')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — MODÈLE : DINOv2 + LoRA + Dual Head (optionnel)
# ═══════════════════════════════════════════════════════════════════════════════
from transformers import AutoModel

class DINOv3Model(nn.Module):
    def __init__(self, cfg, num_types):
        super().__init__()
        self.use_dual = cfg.USE_DUAL
        self.backbone = AutoModel.from_pretrained(
            cfg.BACKBONE, local_files_only=True
        )
        for p in self.backbone.parameters():
            p.requires_grad = False
        apply_lora(self.backbone, cfg)
        self.head_fraud = nn.Linear(cfg.EMBED_DIM, 1)
        if self.use_dual:
            self.head_type = nn.Linear(cfg.EMBED_DIM, num_types)

    def forward(self, x):
        # TimmWrapperModel retourne last_hidden_state
        # Le CLS token est à l'index 0
        features = self.backbone(x).last_hidden_state[:, 0]
        logit_fraud = self.head_fraud(features).squeeze(1)
        if self.use_dual:
            return logit_fraud, self.head_type(features)
        return logit_fraud
print('✓ DINOv2Model')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — DATASET + TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def get_transforms(split):
    if split == 'train':
        return A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.ImageCompression(quality_range=(40, 90), p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GaussNoise(std_range=(0.01, 0.06), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02, p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Affine(rotate=(-6, 6), scale=(0.94, 1.06), shear=(-2, 2), p=0.4),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


class FreuidDS(Dataset):
    def __init__(self, df, data_dir, transform, type2idx=None, split='train'):
        self.df       = df.reset_index(drop=True)
        self.dir      = Path(data_dir)
        self.transform = transform
        self.type2idx  = type2idx or {}
        self.split     = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p   = self.dir / (row['filename'] if self.split == 'test' else row['image_path'])
        img = np.array(Image.open(p).convert('RGB'))
        img = self.transform(image=img)['image']
        label    = float(row['label']) if 'label' in self.df.columns else -1.0
        type_idx = self.type2idx.get(row.get('type', ''), 0)
        return img, torch.tensor(label, dtype=torch.float32), torch.tensor(type_idx, dtype=torch.long)

print('✓ Dataset')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_freuid(scores, labels, bpcer_budget=0.01):
    bad = np.isnan(scores).sum() + np.isinf(scores).sum()
    if bad > 0:
        print(f'  ⚠️ {bad} invalid scores → 0.5')
        scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    scores = np.clip(scores, 0.0, 1.0)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr   = 1 - tpr
    audet = sk_auc(fpr, fnr)
    valid = np.where(fpr <= bpcer_budget)[0]
    apcer = fnr[valid[-1]] if len(valid) else 1.0
    ga, gp = 1 - audet, 1 - apcer
    d = ga + gp
    freuid = 1.0 if d == 0 else 1 - 2 * ga * gp / d
    return dict(audet=round(audet, 4), apcer=round(apcer, 4), freuid=round(freuid, 4))

print('✓ Metrics')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def mixup_data(x, y_fraud, y_type, alpha=0.4):
    if x.size(0) < 2:
        return x, y_fraud, y_type
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return (lam * x + (1 - lam) * x[idx],
            lam * y_fraud + (1 - lam) * y_fraud[idx],
            y_type if lam > 0.5 else y_type[idx])


def train_epoch(model, loader, opt, sched, focal, cfg):
    model.train()
    tot_loss, n = 0.0, 0
    opt.zero_grad(set_to_none=True)

    for step, (imgs, labels, types) in enumerate(tqdm(loader, desc='Train', leave=False)):
        imgs   = imgs.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)
        types  = types.to(cfg.DEVICE, non_blocking=True)

        if np.random.random() < cfg.MIXUP_PROB:
            imgs, labels, types = mixup_data(imgs, labels, types, cfg.MIXUP_ALPHA)

        s = cfg.LABEL_SMOOTH
        labels_s = labels * (1 - 2 * s) + s

        with torch.autocast(device_type='cuda', dtype=cfg.AMP_DTYPE):
            if cfg.USE_DUAL:
                logit_fraud, logit_type = model(imgs)
                loss = (focal(logit_fraud, labels_s)
                        + cfg.AUX_WEIGHT * F.cross_entropy(logit_type, types, label_smoothing=0.1))
            else:
                logit_fraud = model(imgs)
                loss = focal(logit_fraud, labels_s)
            loss = loss / cfg.GRAD_ACCUM

        if torch.isnan(loss):
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()
        if (step + 1) % cfg.GRAD_ACCUM == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
            opt.step(); sched.step()
            opt.zero_grad(set_to_none=True)

        tot_loss += loss.item() * cfg.GRAD_ACCUM * imgs.size(0)
        n += imgs.size(0)

    return tot_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, focal, cfg):
    model.eval()
    tot_loss = 0.0
    all_s, all_l = [], []

    for imgs, labels, types in tqdm(loader, desc='Val', leave=False):
        imgs   = imgs.to(cfg.DEVICE, non_blocking=True)
        labels = labels.to(cfg.DEVICE, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=cfg.AMP_DTYPE):
            if cfg.USE_DUAL:
                logit_fraud, _ = model(imgs)
            else:
                logit_fraud = model(imgs)
            loss = focal(logit_fraud, labels)

        tot_loss += loss.item() * imgs.size(0)
        all_s.append(torch.sigmoid(logit_fraud).cpu().float().numpy())
        all_l.append(labels.cpu().float().numpy())

    return tot_loss / len(loader.dataset), np.concatenate(all_s), np.concatenate(all_l)

print('✓ Train/Eval')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — TRAIN FOLD
# ═══════════════════════════════════════════════════════════════════════════════

def train_fold(cfg, train_df, val_df, fold_idx, type2idx, num_types):
    print(f'\n{"="*60}')
    print(f'FOLD {fold_idx+1} | Train: {len(train_df)} | Val: {len(val_df)}')
    print(f'{"="*60}')

    _kw = dict(num_workers=cfg.NUM_WORKERS, pin_memory=cfg.PIN_MEMORY,
               persistent_workers=True, prefetch_factor=cfg.PREFETCH)
    train_loader = DataLoader(
        FreuidDS(train_df, cfg.TRAIN_DIR, get_transforms('train'), type2idx, 'train'),
        batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True, **_kw)
    val_loader = DataLoader(
        FreuidDS(val_df, cfg.TRAIN_DIR, get_transforms('val'), type2idx, 'val'),
        batch_size=cfg.BATCH_SIZE * 2, shuffle=False, **_kw)

    steps = len(train_loader)
    print(f'Steps/epoch: {steps}')

    model = DINOv2Model(cfg, num_types).to(cfg.DEVICE)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Params: {total/1e6:.0f}M total | {trainable/1e6:.2f}M trainable ({trainable/total*100:.2f}%)')

    focal = FocalLoss(cfg.FOCAL_ALPHA, cfg.FOCAL_GAMMA)
    opt   = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = OneCycleLR(opt, max_lr=cfg.LR,
                       total_steps=cfg.EPOCHS * steps // cfg.GRAD_ACCUM + 10,
                       pct_start=cfg.WARMUP_EPOCHS / cfg.EPOCHS,
                       anneal_strategy='cos', div_factor=10, final_div_factor=1e3)

    best, no_impr = float('inf'), 0
    ckpt_path = cfg.OUTPUT_DIR / f'best_fold{fold_idx}.pt'

    for ep in range(1, cfg.EPOCHS + 1):
        t_loss = train_epoch(model, train_loader, opt, sched, focal, cfg)
        v_loss, scores, labels = evaluate(model, val_loader, focal, cfg)
        m  = compute_freuid(scores, labels)
        lr = opt.param_groups[0]['lr']

        print(f'  E{ep:02d} | t={t_loss:.4f} v={v_loss:.4f} | '
              f'FREUID={m["freuid"]:.4f} AuDET={m["audet"]:.4f} '
              f'APCER={m["apcer"]:.4f} | lr={lr:.1e}')

        if m['freuid'] < best:
            best, no_impr = m['freuid'], 0
            # Sauvegarde : LoRA + heads + config nécessaire pour reload standalone
            torch.save({
                'state_dict': {k: v for k, v in model.state_dict().items()
                               if 'lora_' in k or 'head_' in k},
                'freuid': best, 'epoch': ep, 'metrics': m,
                'num_types': num_types,
                'use_dual': cfg.USE_DUAL,
            }, ckpt_path)
            print(f'    ✅ Saved (FREUID={best:.4f})')
        else:
            no_impr += 1
            if no_impr >= cfg.PATIENCE:
                print(f'  🛑 Early stop E{ep}')
                break

    del model, opt, sched, focal
    gc.collect(); torch.cuda.empty_cache()
    return ckpt_path, best

print('✓ train_fold')

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — LANCER (LOTO)
# ═══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(cfg.TRAIN_CSV)
print(f'Full dataset: {len(df)} | Fraud: {df.label.mean()*100:.1f}%')

all_types = sorted(df['type'].unique())
type2idx  = {t: i for i, t in enumerate(all_types)}
num_types = len(type2idx)
print(f'Doc types ({num_types}): {all_types}')

logo   = LeaveOneGroupOut()
groups = df['type']
fold_results = []

for fold_idx, (tr, va) in enumerate(logo.split(df, df['label'], groups)):
    if fold_idx in [0,1]:
        continue
    
    val_type    = df.iloc[va]['type'].iloc[0]
    train_types = sorted(df.iloc[tr]['type'].unique())
    print(f'\nFold {fold_idx+1}/{groups.nunique()}')
    print(f'  Train : {train_types} ({len(tr)} images)')
    print(f'  Val   : {val_type} ({len(va)} images)')

    ckpt, score = train_fold(cfg, df.iloc[tr], df.iloc[va], fold_idx, type2idx, num_types)
    fold_results.append({'fold': fold_idx, 'ckpt': ckpt, 'freuid': score, 'type': val_type})
    torch.cuda.empty_cache()

print('\n' + '='*60)
print('LOTO Results')
print('='*60)
for r in fold_results:
    print(f'  {r["type"]:20s} → FREUID = {r["freuid"]:.4f}')
print(f'\n  Moyenne LOTO : {np.mean([r["freuid"] for r in fold_results]):.4f}')
