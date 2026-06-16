#!/usr/bin/env bash
#
# Evaluate the trained model on test data.
#
# Usage: bash scripts/run_eval.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Evaluating Weather Probability Model ==="

python -m evaluation.metrics \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/raw/era5_monthly \
    --norm_stats ./data/processed/norm_stats.json \
    --years 2018 2019 \
    --output ./results \
    "$@"

echo ""
echo "Evaluation complete! Results saved to ./results/"
