#!/usr/bin/env bash
# =============================================================================
# train_gcp.sh -- One-shot training on a GCP GPU VM
#
# Usage:
#   1. Create a spot T4 VM:
#      gcloud compute instances create proust-train \
#        --zone=us-central1-a \
#        --machine-type=n1-standard-4 \
#        --accelerator=type=nvidia-tesla-t4,count=1 \
#        --image-family=pytorch-latest-gpu \
#        --image-project=deeplearning-platform-release \
#        --boot-disk-size=50GB \
#        --provisioning-model=SPOT \
#        --maintenance-policy=TERMINATE
#
#   2. SSH in and run:
#      gcloud compute ssh proust-train --zone=us-central1-a
#      bash train_gcp.sh
#
#   3. After training, copy the checkpoint:
#      gcloud compute scp proust-train:~/proust-attention/checkpoints/best.pt . --zone=us-central1-a
#
#   4. Delete the VM:
#      gcloud compute instances delete proust-train --zone=us-central1-a
#
# Estimated cost: < $1 (spot T4 ~$0.17/hr, training ~10 min)
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/GonorAndres/proust-attention.git"
BRANCH="main"
REPO_DIR="$HOME/proust-attention"

# -- Training hyperparameters --
EPOCHS=50
BATCH_SIZE=64
CONTEXT_LENGTH=256
STRIDE=128

echo "============================================================"
echo "  Proust Attention Machine -- GCP Training"
echo "============================================================"

# Clone or update repo
if [ -d "$REPO_DIR" ]; then
    echo "Repo exists, pulling latest..."
    cd "$REPO_DIR"
    git fetch origin && git checkout "$BRANCH" && git pull origin "$BRANCH"
else
    echo "Cloning repo..."
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q numpy torch matplotlib seaborn tqdm

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Verify corpus
python -c "
from pathlib import Path
p = Path('data/processed/proust_corpus.txt')
assert p.exists(), f'Corpus not found at {p}'
print(f'Corpus: {p.stat().st_size / 1e6:.1f} MB')
"

# Train
echo ""
echo "Starting training: epochs=$EPOCHS, batch=$BATCH_SIZE, stride=$STRIDE"
echo "============================================================"

python src/train.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --context-length "$CONTEXT_LENGTH" \
    --stride "$STRIDE" \
    --device cuda

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Checkpoint: $REPO_DIR/checkpoints/best.pt"
echo ""
echo "  To download:"
echo "    gcloud compute scp $(hostname):$REPO_DIR/checkpoints/best.pt . --zone=us-central1-a"
echo "============================================================"
