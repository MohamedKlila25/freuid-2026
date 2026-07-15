"""
FREUID Challenge 2026 — Inférence complète (public + private)
=============================================================
Prédit TOUTES les images trouvées dans public_test ET private_test.
Aucun fillna(0) — pas de zéros artificiels.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — IMPORTS + CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

import gc, random, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModel

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    DATA_DIR    = Path("/kaggle/input/datasets/coreclock/the-freuid-challenge-2026-ijcai-ecai-mine")
    BACKBONE    = "/kaggle/input/datasets/lizhiyaya/dinov3-large"
    WEIGHTS_DIR = Path("/kaggle/input/datasets/mohamedklila8klila/data-fold")

    # Dossiers d'images — on scanne LES DEUX
    PUBLIC_TEST_DIR  = DATA_DIR / 'public_test' / 'public_test'
    PRIVATE_TEST_DIR = Path("/kaggle/input/datasets/mohamedklila8klila/private-data") / 'private_test'
    # ↑ Vérifie le chemin exact dans ton explorateur Kaggle si erreur "dossier introuvable"

    SAMPLE_SUB  = DATA_DIR / 'sample_submission.csv'
    OUTPUT_DIR  = Path('/kaggle/working/kaggle_dino')

    EMBED_DIM   = 1024   # DINOv3-Large
    IMG_SIZE    = 448
    LORA_R      = 16
    LORA_ALPHA  = 32
    LORA_DROPOUT = 0.05
    USE_DUAL    = False

    BATCH_SIZE  = 128 #32
    AMP_DTYPE   = torch.bfloat16
    DEVICE      = 'cuda'
    NUM_WORKERS = 4
    PIN_MEMORY  = True
    PREFETCH    = 2

cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

FOLD_CKPTS = [
    cfg.WEIGHTS_DIR / 'best_fold0.pt',
    cfg.WEIGHTS_DIR / 'best_fold1.pt',
    cfg.WEIGHTS_DIR / 'best_fold2.pt',
    cfg.WEIGHTS_DIR / 'best_fold3.pt',
    cfg.WEIGHTS_DIR / 'best_fold4.pt',
]

# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLE — LoRA + DINOv2
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

    @property
    def weight(self):
        return self.original.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self):
        return self.original.bias

    def forward(self, x):
        return self.original(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


def apply_lora(model, cfg):
    """Signature identique au code de train."""
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


class DINOv3Model(nn.Module):
    """Nom identique au code de train (DINOv3Model, pas DINOv2Model)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET — lit directement les fichiers image (pas de CSV)
# ═══════════════════════════════════════════════════════════════════════════════

class ImageFileDataset(Dataset):
    """Dataset qui lit une liste de Path. id = stem (sans extension)."""
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = np.array(Image.open(p).convert('RGB'))
        img = self.transform(image=img)['image']
        return img, p.stem   # (tensor, id_string)


def get_transform_base():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

def get_transform_hflip():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

def get_transform_bright():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

TTA_TRANSFORMS = [get_transform_base(), get_transform_hflip(), get_transform_bright()]

# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTE DE TOUTES LES IMAGES (public + private)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_images(*dirs):
    """Scanne plusieurs dossiers et retourne une liste de Path triée."""
    paths = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f'  ⚠️  Dossier introuvable : {d}')
            continue
        found = [p for p in d.iterdir() if p.suffix.lower() in SUPPORTED_EXT]
        print(f'  {d.name}: {len(found)} images')
        paths.extend(found)
    # Dédupliquer par stem (au cas où même fichier dans deux dossiers)
    seen, unique = set(), []
    for p in sorted(paths):
        if p.stem not in seen:
            seen.add(p.stem)
            unique.append(p)
    return unique

print('Collecte des images...')
all_image_paths = collect_images(cfg.PUBLIC_TEST_DIR, cfg.PRIVATE_TEST_DIR)
print(f'  → Total : {len(all_image_paths)} images uniques')

# ═══════════════════════════════════════════════════════════════════════════════
# INFÉRENCE — 5 folds × 3 TTA
# ═══════════════════════════════════════════════════════════════════════════════

accumulated = np.zeros(len(all_image_paths))
id_order    = None   # on fixe l'ordre à la première passe
n_total     = 0

for fold_idx, ckpt_path in enumerate(FOLD_CKPTS):
    if not ckpt_path.exists():
        print(f'⚠️  {ckpt_path} introuvable — fold {fold_idx} ignoré')
        continue

    ckpt_data  = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
    _num_types = ckpt_data.get('num_types', 5)
    _use_dual  = ckpt_data.get('use_dual', False)
    cfg.USE_DUAL = _use_dual

    model = DINOv3Model(cfg, _num_types).to(cfg.DEVICE)
    model.load_state_dict(ckpt_data['state_dict'], strict=False)
    model.eval()
    print(f'\nFold {fold_idx} | val FREUID={ckpt_data.get("freuid", 0):.4f} | dual={_use_dual}')

    for ti, tf in enumerate(TTA_TRANSFORMS):
        ds = ImageFileDataset(all_image_paths, transform=tf)
        loader = DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            prefetch_factor=cfg.PREFETCH,
            persistent_workers=True,
        )

        preds, ids_this_pass = [], []
        with torch.no_grad():
            for imgs, img_ids in tqdm(loader, desc=f'  TTA {ti+1}/3', leave=False):
                imgs = imgs.to(cfg.DEVICE, non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=cfg.AMP_DTYPE):
                    out = model(imgs)
                    logits = out[0] if isinstance(out, tuple) else out
                preds.append(torch.sigmoid(logits).cpu().float().numpy())
                ids_this_pass.extend(img_ids)

        preds = np.concatenate(preds)
        accumulated += preds
        n_total += 1

        if id_order is None:
            id_order = ids_this_pass   # ordre fixé à la première passe

        print(f'  TTA {ti+1}/3 done | range [{preds.min():.4f}, {preds.max():.4f}]')

    del model
    gc.collect()
    torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# MOYENNE & SOUMISSION
# ═══════════════════════════════════════════════════════════════════════════════

assert n_total > 0, "Aucun checkpoint chargé — vérifie WEIGHTS_DIR"
final_scores = accumulated / n_total

print(f'\nEnsemble: {n_total} passes ({len([c for c in FOLD_CKPTS if c.exists()])} folds × 3 TTA)')
print(f'Range: [{final_scores.min():.4f}, {final_scores.max():.4f}] | Mean: {final_scores.mean():.4f}')

# ── Construire le DataFrame de soumission ────────────────────────────────────
sub = pd.DataFrame({'id': id_order, 'label': final_scores})

# ── Vérifier la cohérence avec sample_submission (optionnel mais recommandé) ─
if cfg.SAMPLE_SUB.exists():
    sample = pd.read_csv(cfg.SAMPLE_SUB)
    expected_ids = set(sample['id'].astype(str))
    predicted_ids = set(sub['id'].astype(str))

    missing  = expected_ids - predicted_ids
    extra    = predicted_ids - expected_ids

    if missing:
        print(f'\n⚠️  {len(missing)} IDs du sample_submission sans prédiction :')
        print('   ', list(missing)[:10])
        # Ajouter ces IDs avec score 0.5 (neutre) plutôt que 0
        missing_df = pd.DataFrame({'id': list(missing), 'label': 0.5})
        sub = pd.concat([sub, missing_df], ignore_index=True)
        print(f'   → Ajoutés avec score 0.5 (neutre)')

    if extra:
        print(f'\nℹ️  {len(extra)} IDs prédits non présents dans sample_submission (images extra ignorées)')
        sub = sub[sub['id'].isin(expected_ids)]

    # Réordonner selon sample_submission
    sub['id'] = sub['id'].astype(str)
    sample['id'] = sample['id'].astype(str)
    sub = sample[['id']].merge(sub, on='id', how='left')
    # Dernier filet de sécurité — ne devrait plus arriver
    sub['label'] = sub['label'].fillna(0.5)

    print(f'\n✅ Aligné avec sample_submission : {len(sub)} lignes')
    print(f'   Zéros : {(sub.label == 0).sum()} | Score neutre (0.5) : {(sub.label == 0.5).sum()}')
else:
    print('\nℹ️  sample_submission.csv non trouvé — soumission sans réordonnancement')

# ── Sauvegarde ────────────────────────────────────────────────────────────────
sub_path = cfg.OUTPUT_DIR / 'submission_full.csv'
sub.to_csv(sub_path, index=False)
print(f'\n✅ Soumission sauvegardée → {sub_path}')
print(f'   {len(sub)} lignes | range [{sub.label.min():.4f}, {sub.label.max():.4f}]')