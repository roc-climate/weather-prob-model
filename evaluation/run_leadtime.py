"""
Evaluate model skill (CRPS / CRPSS) at multiple lead times.

Usage:
  python -m evaluation.run_leadtime --checkpoint ./checkpoints/best_model.pt \
      --data_dir ./data/raw/era5_monthly --years 2015 2016 --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WeatherDataset
from data.normalization import load_statistics
from evaluation.metrics import (
    compute_crps_gaussian,
    compute_crpss,
    compute_rmse,
    compute_latitude_weights_np,
)
from models.weather_model import WeatherProbModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/raw/era5_monthly")
    parser.add_argument("--norm_stats", type=str, default="./data/processed/norm_stats.json")
    parser.add_argument("--years", type=int, nargs=2, default=[2015, 2016])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lead_times", type=int, nargs="+", default=[1, 2, 3])
    return parser.parse_args()


def evaluate_one_lead_time(
    model, lead_time, dataset_kwargs, norm_stats, lat_weights, device
):
    """Evaluate model CRPS and climatology CRPS at a given lead time."""
    dataset = WeatherDataset(lead_time=lead_time, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    t2m_mean = norm_stats["mean"][0].item()
    t2m_std = norm_stats["std"][0].item()

    all_mu, all_sigma, all_y = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_atmos = batch["x_atmos"].to(device)
            x_slow = batch["x_slow"].to(device)
            y_t2m = batch["y_t2m"].to(device)
            pred = model(x_atmos, x_slow, torch.zeros(x_atmos.shape[0], 6, device=device))
            all_mu.append(pred["t2m"]["mu"].cpu().numpy())
            all_sigma.append(pred["t2m"]["sigma"].cpu().numpy())
            all_y.append(y_t2m.cpu().numpy())

    mu = np.concatenate(all_mu, axis=0)      # (N, 1, H, W)
    sigma = np.concatenate(all_sigma, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Denormalize to physical K
    mu_phys = mu[:, 0] * t2m_std + t2m_mean
    sigma_phys = sigma[:, 0] * t2m_std
    y_phys = y[:, 0] * t2m_std + t2m_mean

    # Model CRPS
    crps_vals = []
    rmse_vals = []
    for i in range(len(mu_phys)):
        crps_vals.append(
            compute_crps_gaussian(mu_phys[i], sigma_phys[i], y_phys[i], lat_weights)
        )
        rmse_vals.append(compute_rmse(mu_phys[i], y_phys[i], lat_weights))

    # Climatology: mean of training data in physical space
    train_ds = WeatherDataset(lead_time=1, **dataset_kwargs)
    # Override years for climatology
    train_ds = WeatherDataset(
        lead_time=lead_time,
        data_dir=dataset_kwargs["data_dir"],
        variable_list=dataset_kwargs["variable_list"],
        years=(1995, 2014),
        norm_stats=norm_stats,
    )
    train_t2m = train_ds.data[:, 0].numpy()  # raw physical (T, H, W)
    clim = np.mean(train_t2m, axis=0)
    clim_std = np.std(train_t2m, axis=0) + 1e-6

    clim_crps_vals = []
    clim_rmse_vals = []
    for i in range(len(y_phys)):
        clim_crps_vals.append(
            compute_crps_gaussian(
                np.broadcast_to(clim, y_phys[i].shape),
                np.broadcast_to(clim_std, y_phys[i].shape),
                y_phys[i], lat_weights,
            )
        )
        clim_rmse_vals.append(compute_rmse(clim, y_phys[i], lat_weights))

    model_crps = float(np.mean(crps_vals))
    clim_crps = float(np.mean(clim_crps_vals))
    crpss = compute_crpss(model_crps, clim_crps)
    model_rmse = float(np.mean(rmse_vals))
    clim_rmse = float(np.mean(clim_rmse_vals))

    return {
        "lead_time": lead_time,
        "n_samples": len(mu_phys),
        "model_crps": model_crps,
        "clim_crps": clim_crps,
        "crpss": crpss,
        "model_rmse": model_rmse,
        "clim_rmse": clim_rmse,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = WeatherProbModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Shared dataset kwargs (no norm_stats needed for evaluate — we denormalize manually)
    norm_stats = load_statistics(args.norm_stats)
    variable_list = [
        "t2m", "tp", "sst", "swvl1", "sd", "siconc",
        "msl", "u10", "v10", "tisr", "ssr", "str",
    ]
    dataset_kwargs = dict(
        data_dir=args.data_dir,
        variable_list=variable_list,
        years=tuple(args.years),
        norm_stats=norm_stats,
    )
    lat_weights = compute_latitude_weights_np()

    # Evaluate at each lead time
    print(f"\n{'='*70}")
    print(f"Lead-Time Skill Evaluation (on {args.years[0]}-{args.years[1]} data)")
    print(f"{'='*70}")
    print()
    print(f"{'Lead':>6s}  {'N':>4s}  {'Model CRPS':>10s}  {'Clim CRPS':>10s}  "
          f"{'CRPSS':>8s}  {'Model RMSE':>10s}  {'Clim RMSE':>10s}")
    print(f"{'-'*6}  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    results = []
    for lt in args.lead_times:
        r = evaluate_one_lead_time(
            model, lt, dataset_kwargs, norm_stats, lat_weights, device
        )
        results.append(r)

        # CRPSS sign indicator
        sign = "+" if r["crpss"] > 0 else ""
        print(f"{r['lead_time']:>6d}  {r['n_samples']:>4d}  "
              f"{r['model_crps']:>10.4f}  {r['clim_crps']:>10.4f}  "
              f"{sign}{r['crpss']:>7.4f}  "
              f"{r['model_rmse']:>10.4f}  {r['clim_rmse']:>10.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Lead time 1m ≈ 4 weeks (model was trained for this)")
    print(f"  Lead time 2m ≈ 8 weeks")
    print(f"  Lead time 3m ≈ 12 weeks")
    print()
    print("  CRPSS > 0  →  model beats climatology at that lead time")
    print("  CRPSS decay →  measures how fast skill drops with lead time")

    # Check if using mock data
    if all(r["crpss"] > 0.9 for r in results if r["lead_time"] == 1):
        print("\n  ⚠  Note: using mock data — the simulated teleconnection is ")
        print("     artificially strong. Real data will show more decay.")


if __name__ == "__main__":
    main()
