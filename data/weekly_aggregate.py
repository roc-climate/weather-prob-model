#!/usr/bin/env python3
"""
Aggregate ERA5 daily data into weekly means (for Phase 2).

Usage:
  python weekly_aggregate.py --input ./data/raw/era5_daily --output ./data/processed/era5_weekly

The script reads daily .nc files, groups by ISO week, and saves weekly means.
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr


def week_id_from_date(date):
    """Return (year, week_number) for a given date."""
    iso = pd.Timestamp(date).isocalendar()
    return (iso.year, iso.week)


def aggregate_to_weekly(
    input_dir: str,
    output_dir: str,
    variables: list = None,
    years: tuple = None,
):
    """
    Aggregate daily ERA5 data to weekly means.

    Args:
        input_dir: Directory containing daily .nc files
        output_dir: Directory to save weekly .nc files
        variables: List of variables to process
        years: (start_year, end_year) to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover daily files
    daily_files = sorted(input_dir.glob("*.nc"))
    if not daily_files:
        print(f"No .nc files found in {input_dir}")
        return

    print(f"Found {len(daily_files)} daily files")

    # Process year by year to manage memory
    for fpath in daily_files:
        print(f"Processing {fpath.name}...")

        ds = xr.open_dataset(fpath)

        # Determine available variables
        if variables is None:
            variables = [v for v in ds.data_vars if v not in ["latitude", "longitude", "time"]
                         and "time" not in ds[v].dims]

        # Get time coordinate
        time_var = ds["valid_time"] if "valid_time" in ds else ds["time"]
        times = pd.to_datetime(time_var.values)

        # Group by week
        week_groups = defaultdict(list)
        for i, t in enumerate(times):
            wid = week_id_from_date(t)
            week_groups[wid].append(i)

        # Compute weekly means
        for (year, week), indices in sorted(week_groups.items()):
            if years and (year < years[0] or year > years[1]):
                continue

            weekly_ds = ds.isel(**{time_var.name: indices})
            weekly_mean = weekly_ds[variables].mean(dim=time_var.name)

            # Save
            out_fname = f"era5_weekly_{year}_w{week:02d}.nc"
            weekly_mean.to_netcdf(output_dir / out_fname)

        ds.close()

    print(f"Weekly aggregates saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ERA5 daily data to weekly means"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Directory containing daily .nc files")
    parser.add_argument("--output", type=str, default="./data/processed/era5_weekly")
    parser.add_argument("--vars", nargs="*", default=None,
                        help="Variables to aggregate (default: all)")
    parser.add_argument("--years", type=int, nargs=2, default=None,
                        help="Year range to process (default: all)")
    args = parser.parse_args()

    aggregate_to_weekly(args.input, args.output, args.vars,
                        tuple(args.years) if args.years else None)


if __name__ == "__main__":
    main()
