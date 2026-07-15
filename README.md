# FREUID Challenge 2026 — DINOv3-Large + LoRA

**Team:** MohamedKlila25  
**Competition:** [The FREUID Challenge 2026 (IJCAI-ECAI)](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai)  
**Public FREUID Score:** 0.04087  
**Method:** DINOv3-Large · LoRA (r=16) · Focal Loss · 5-fold LOTO · 3×TTA ensemble  
**License:** MIT

---

## Repository structure

```
freuid-2026/
├── train.py                        # Full training pipeline (LOTO cross-validation)
├── inference.py                    # Full inference — public + private test images
├── requirements.txt                # Python dependencies
├── FREUID_Technical_Report.pdf     # Technical report
├── LICENSE                         # MIT
├── README.md
├── checkpoints/
│   ├── best_fold0.pt               # LOTO fold 0 checkpoint
│   ├── best_fold1.pt               # LOTO fold 1 checkpoint
│   ├── best_fold2.pt               # LOTO fold 2 checkpoint
│   ├── best_fold3.pt               # LOTO fold 3 checkpoint
│   └── best_fold4.pt               # LOTO fold 4 checkpoint
└── docker/
    ├── Dockerfile                  # No-network sandbox (organizer contract)
    ├── prepare_submission.py       # Docker entrypoint → reads /data/, writes /submissions/
    └── requirements.txt            # Pinned dependencies for Docker
```

---

## Method

### Backbone
**DINOv3-Large** (1024-d CLS token, 307M parameters) — frozen during training, loaded offline from HuggingFace.

### LoRA fine-tuning
All attention (`qkv`, `proj`) and MLP (`fc1`, `fc2`) layers augmented with low-rank adapters:
- Rank `r=16`, alpha `α=32`, dropout `0.05`
- ~25M trainable parameters (~8% of total)

### Loss
Focal Loss (`α=0.2`, `γ=2.0`) + label smoothing (`ε=0.05`) + MixUp (`α=0.4`, `p=0.3`)

### Validation
**Leave-One-Type-Out (LOTO)** — each fold holds out one complete document type, directly simulating the private test set where unseen document types appear.

### Inference ensemble
**5 checkpoints × 3 TTA passes = 15 forward passes** averaged per image.  
TTA: original / horizontal flip / brightness+contrast jitter.

---

## Training

```bash
pip install -r requirements.txt
python train.py
```

Checkpoints saved to `/kaggle/working/kaggle_dino/best_fold{N}.pt`.

### Key hyperparameters

| Parameter | Value |
|---|---|
| Backbone | DINOv3-Large (embed_dim=1024) |
| Image size | 448 × 448 |
| LoRA r / α | 16 / 32 |
| Epochs | 20 (early stop patience=4) |
| Batch size | 16 (train) / 128 (inference) |
| Learning rate | 2e-5 (OneCycleLR cosine) |
| Weight decay | 0.01 |
| Focal α / γ | 0.2 / 2.0 |
| Label smoothing | 0.05 |
| MixUp α / p | 0.4 / 0.3 |
| Precision | bfloat16 (AMP) |
| Seed | 342 |

### Kaggle data paths
```
DATA_DIR  = /kaggle/input/datasets/coreclock/the-freuid-challenge-2026-ijcai-ecai-mine
BACKBONE  = /kaggle/input/datasets/lizhiyaya/dinov3-large
WEIGHTS   = /kaggle/input/datasets/mohamedklila8klila/data-fold
```

---

## Inference (Kaggle)

Predicts all images from both `public_test/` and `private_test/` — no artificial zeros.

```bash
python inference.py
```

Output: `/kaggle/working/kaggle_dino/submission_full.csv`

---

## Docker — organizer sandbox

### Prerequisites

Before building, place the following files locally:

```
docker/
├── Dockerfile
├── prepare_submission.py
├── requirements.txt
├── backbone/          ← DINOv3-Large files from https://www.kaggle.com/datasets/lizhiyaya/dinov3-large
└── weights/           ← best_fold0.pt … best_fold4.pt from checkpoints/ above
```

### Build

```bash
cd docker
docker build -t freuid-repro:local .
```

### Run — no-network sandbox (exactly as organizers will execute)

```bash
docker run --rm \
  --network none \
  --gpus all \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

### Output format

```
id,label
3f49e6921f3f4b10910738f7e9b476f3,0.8712
c39d1c24d46a44a088991ddc7a072b7c,0.0234
```

`label` ∈ [0, 1] — higher = more likely fraudulent.  
One row per image. No missing or extra IDs.

### Sandbox contract

| Mount | Container path | Mode |
|---|---|---|
| Test images (flat dir, no CSV) | `/data/` | read-only |
| Output CSV | `/submissions/` | read-write |

Supported extensions: `.jpg` `.jpeg` `.png` `.webp` `.bmp` `.tif` `.tiff`

---

## Hardware

| Stage | Hardware |
|---|---|
| Training | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |
| Inference / Docker | NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) |

---

## External resources

- DINOv3-Large pretrained weights — Meta AI (Apache 2.0)
- No external fraud or identity document datasets
- No synthetic training data or pseudo-labels

---

## License

MIT — see `LICENSE`.  
DINOv3-Large backbone: Apache 2.0 (Meta AI).
