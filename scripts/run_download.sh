#!/usr/bin/env bash
#
# Phase 1: Download ERA5 monthly data and climate indices.
#
# Prerequisites:
#   1. Register at https://cds.climate.copernicus.eu/
#   2. pip install cdsapi
#   3. Create ~/.cdsapirc with your credentials
#
# Usage: bash scripts/run_download.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Downloading ERA5 Monthly Surface Data ==="
python -m data.download_era5_monthly \
    --years 1995 2019 \
    --output ./data/raw/era5_monthly

echo ""
echo "=== Downloading Climate Indices ==="
python -m data.download_climate_indices \
    --output ./data/raw/climate_indices

echo ""
echo "=== Computing Normalization Statistics ==="
python -m data.normalization \
    --data_dir ./data/raw/era5_monthly \
    --output ./data/processed/norm_stats.json \
    --years 1995 2014

echo ""
echo "Downloads complete!"
echo "Next step: bash scripts/run_train.sh"
