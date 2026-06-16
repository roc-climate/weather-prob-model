"""
Compute and save normalization statistics from ERA5 monthly data.

Normalization is computed per-variable as z-score:
  x_norm = (x - mean) / std

The mean and std are computed over the training period only, and
over all spatial grid points.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import xarray as xr

from .dataset import WeatherDataset


def compute_statistics(
    data_dir: str,
    variable_list: list,
    years: tuple = (1995, 2014),
    time_dim: str = "valid_time",
) -> Dict[str, torch.Tensor]:
    """
    Compute per-variable mean and std from training data.

    Args:
        data_dir: Directory containing .nc files
        variable_list: List of variable names
        years: (start_year, end_year) training years
        time_dim: Time dimension name

    Returns:
        dict with "mean" and "std" tensors of shape (n_vars, 1, 1)
    """
    print(f"Computing normalization statistics ({years[0]}-{years[1]})...")

    sums = {}
    sums_sq = {}
    counts = {}

    data_dir = Path(data_dir)

    for year in range(years[0], years[1] + 1):
        fname = f"era5_monthly_surface_{year}.nc"
        fpath = data_dir / fname

        if not fpath.exists():
            candidates = list(data_dir.glob(f"*{year}*.nc"))
            if candidates:
                fpath = candidates[0]
            else:
                print(f"  Skipping year {year}: file not found")
                continue

        ds = xr.open_dataset(fpath)

        for var_name in variable_list:
            if var_name not in ds:
                continue

            arr = ds[var_name].values.astype(np.float32)
            # arr: (time, lat, lon)
            if var_name not in sums:
                sums[var_name] = 0.0
                sums_sq[var_name] = 0.0
                counts[var_name] = 0

            # Handle NaN (e.g., SST over land, soil moisture over ocean)
            mask = ~np.isnan(arr)
            arr_filled = np.nan_to_num(arr, nan=0.0)

            sums[var_name] += arr_filled.sum(axis=0)
            sums_sq[var_name] += (arr_filled ** 2).sum(axis=0)
            counts[var_name] += mask.sum(axis=0).astype(np.float32)

        ds.close()
        print(f"  Processed year {year}")

    # Compute mean and std
    means = []
    stds = []

    for var_name in variable_list:
        if var_name in sums and counts[var_name].max() > 0:
            safe_count = np.maximum(counts[var_name], 1.0)
            mean = sums[var_name] / safe_count
            variance = sums_sq[var_name] / safe_count - mean ** 2
            variance = np.maximum(variance, 0.0)
            std = np.sqrt(variance)

            # Global scalar mean/std (average over spatial dims)
            global_mean = float(mean[mean > 0].mean()) if (mean > 0).any() else float(mean.mean())
            global_std = float(std[std > 0].mean()) if (std > 0).any() else float(std.mean() + 1e-6)
        else:
            global_mean = 0.0
            global_std = 1.0

        means.append(global_mean)
        stds.append(global_std)
        print(f"  {var_name}: mean={global_mean:.4f}, std={global_std:.4f}")

    mean_tensor = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
    std_tensor = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1)

    return {"mean": mean_tensor, "std": std_tensor}


def save_statistics(stats: Dict, output_path: str):
    """Save normalization statistics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "mean": stats["mean"].squeeze().tolist(),
        "std": stats["std"].squeeze().tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Saved statistics to {output_path}")


def load_statistics(path: str) -> Dict[str, torch.Tensor]:
    """Load normalization statistics from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    mean = torch.tensor(data["mean"], dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(data["std"], dtype=torch.float32).view(-1, 1, 1)

    return {"mean": mean, "std": std}


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--data_dir", type=str, default="./data/raw/era5_monthly")
    parser.add_argument("--output", type=str, default="./data/processed/norm_stats.json")
    parser.add_argument("--vars", nargs="*", default=[
        "t2m", "tp", "sst", "swvl1", "sd", "siconc",
        "msl", "u10", "v10", "tisr", "ssr", "str",
    ])
    parser.add_argument("--years", type=int, nargs=2, default=[1995, 2014])
    args = parser.parse_args()

    stats = compute_statistics(args.data_dir, args.vars, tuple(args.years))
    save_statistics(stats, args.output)


if __name__ == "__main__":
    main()
