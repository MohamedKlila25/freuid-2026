# ─────────────────────────────────────────────────────────────────────────────
# FREUID Challenge 2026 — DINOv3-Large + LoRA inference container
# Sandbox : --network none | /data/ read-only | /submissions/ output
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── DINOv3-Large backbone (offline, no internet needed) ───────────────────────
# Download from: https://www.kaggle.com/datasets/lizhiyaya/dinov3-large
# Place all files in docker/backbone/ before building
COPY backbone/ /app/backbone/

# ── Fold checkpoints (best_fold0.pt … best_fold4.pt) ─────────────────────────
# Download from: https://www.kaggle.com/datasets/mohamedklila8klila/data-fold
# Place all .pt files in docker/weights/ before building
COPY weights/ /app/weights/

COPY prepare_submission.py .

ENTRYPOINT ["python", "prepare_submission.py", \
            "--data_dir", "/data", \
            "--output",   "/submissions/submission.csv"]
