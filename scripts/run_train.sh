#!/usr/bin/env bash
#
# Train the weather probability model (Phase 1).
#
# Usage:
#   bash scripts/run_train.sh                  # Use config defaults
#   bash scripts/run_train.sh --epochs 50      # Override epochs
#   bash scripts/run_train.sh --device cpu     # Force CPU training

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default arguments
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-1e-3}
DEVICE=${DEVICE:-cuda}

echo "=== Training Weather Probability Model ==="
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Device:     $DEVICE"
echo ""

python -m training.train \
    --config configs/config.yaml \
    --data_dir ./data/raw/era5_monthly \
    --output_dir ./checkpoints \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    "$@"

echo ""
echo "Training complete!"
echo "Next step: python -m evaluation.metrics (evaluate on test set)"
