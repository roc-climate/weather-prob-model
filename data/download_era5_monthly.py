#!/usr/bin/env python3
"""
Download ERA5 monthly averaged surface variables via CDS API.

Phase 1: Minimal data (~500 MB compressed)
  - 12 surface variables
  - 1995-2019 (25 years)
  - Monthly means (pre-computed by ECMWF)
  - 0.25° native resolution

CDS API setup:
  1. Register at https://cds.climate.copernicus.eu/
  2. Install: pip install cdsapi
  3. Create ~/.cdsapirc with your key:
     url: https://cds.climate.copernicus.eu/api
     key: <your-uid>:<your-api-key>

Usage:
  python download_era5_monthly.py --years 1995 2019 --output ./data/raw/era5_monthly
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import cdsapi
except ImportError:
    print("Please install cdsapi: pip install cdsapi")
    sys.exit(1)

# Variable definitions matching our project's variable list
VARIABLES = [
    # Prediction targets
    "2m_temperature",              # t2m
    "total_precipitation",         # tp
    # Slow boundary forcing
    "sea_surface_temperature",     # sst
    "volumetric_soil_water_layer_1",  # swvl1
    "snow_depth",                  # sd
    "sea_ice_cover",               # siconc
    # Atmospheric state
    "mean_sea_level_pressure",     # msl
    "10m_u_component_of_wind",     # u10
    "10m_v_component_of_wind",     # v10
    # Radiative / energy
    "toa_incident_solar_radiation",  # tisr
    "surface_net_solar_radiation",   # ssr
    "surface_net_thermal_radiation", # str
]

# ERA5 monthly averaged data is available in these datasets:
# - reanalysis-era5-single-levels-monthly-means (1979-present, pre-computed)
DATASET = "reanalysis-era5-single-levels-monthly-means"


def download_year(client, year: int, output_dir: Path, dry_run: bool = False):
    """Download one year of monthly data."""
    output_file = output_dir / f"era5_monthly_surface_{year}.nc"

    if output_file.exists():
        print(f"  [{year}] Already exists, skipping: {output_file}")
        return

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    if dry_run:
        print(f"  [{year}] Dry run — would download to {output_file}")
        return

    print(f"  [{year}] Downloading...")
    client.retrieve(DATASET, request, str(output_file))
    print(f"  [{year}] Done → {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download ERA5 monthly averaged surface data"
    )
    parser.add_argument(
        "--years", type=int, nargs=2, default=[1995, 2019],
        help="Start and end year (inclusive), default: 1995 2019"
    )
    parser.add_argument(
        "--output", type=str, default="./data/raw/era5_monthly",
        help="Output directory, default: ./data/raw/era5_monthly"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    start_year, end_year = args.years
    n_years = end_year - start_year + 1
    n_vars = len(VARIABLES)
    est_size_per_year = 20  # MB, roughly for monthly surface at 0.25°
    total_est = n_years * est_size_per_year

    print(f"ERA5 Monthly Surface Data Download")
    print(f"  Dataset: {DATASET}")
    print(f"  Variables: {n_vars}")
    print(f"  Years: {start_year}–{end_year} ({n_years} years)")
    print(f"  Estimated total: ~{total_est} MB")
    print(f"  Output: {output_dir.resolve()}")
    print()

    for year in range(start_year, end_year + 1):
        download_year(client, year, output_dir, dry_run=args.dry_run)

    print()
    print("All downloads complete.")


if __name__ == "__main__":
    main()
