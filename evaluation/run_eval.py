#!/usr/bin/env python3
"""
Quick evaluation: load trained model, compute CRPS/RMSE, compare against baselines.

Usage:
  python -m evaluation.run_eval --checkpoint ./checkpoints/best_model.pt \
      --data_dir ./data/raw/era5_monthly --years 2015 2016
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import WeatherDataset
from data.normalization import load_statistics
from evaluation.baselines import ClimatologyBaseline
from evaluation.metrics import (
    compute_crps_gaussian,
    compute_crpss,
    compute_rmse,
    compute_acc,
    compute_latitude_weights_np,
)
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/raw/era5_monthly")
    parser.add_argument("--norm_stats", type=str, default="./data/processed/norm_stats.json")
    parser.add_argument("--years", type=int, nargs=2, default=[2015, 2016])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    variable_list = [
        "t2m", "tp", "sst", "swvl1", "sd", "siconc",
        "msl", "u10", "v10", "tisr", "ssr", "str",
    ]

    norm_stats = load_statistics(args.norm_stats)
    # Keep norm stats on CPU — dataset ops happen on CPU before moving to GPU

    # Test dataset
    dataset = WeatherDataset(
        data_dir=args.data_dir,
        variable_list=variable_list,
        lead_time=1,
        years=tuple(args.years),
        norm_stats=norm_stats,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    print(f"Test samples: {len(dataset)}")

    # Build model
    from models.weather_model import WeatherProbModel
    model_cfg = config.get("model", {})
    model = WeatherProbModel(
        n_atmos_vars=6,
        n_slow_vars=4,
        n_indices=6,
        atmos_encoder_kwargs=model_cfg.get("atmos_encoder", {}),
        slow_encoder_kwargs=model_cfg.get("slow_encoder", {}),
        cross_attn_kwargs=model_cfg.get("cross_attention", {}),
        index_embed_kwargs=model_cfg.get("index_embedding", {}),
        gaussian_head_kwargs=model_cfg.get("prob_heads", {}).get("t2m", {}),
        quantile_head_kwargs=model_cfg.get("prob_heads", {}).get("tp", {}),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {checkpoint['epoch']})")

    # Latitude weights
    lat_weights = compute_latitude_weights_np()  # (121, 1)

    # Get denormalization stats for targets
    t2m_idx = variable_list.index("t2m")
    tp_idx = variable_list.index("tp")
    t2m_mean = norm_stats["mean"][t2m_idx].item()
    t2m_std = norm_stats["std"][t2m_idx].item()
    tp_mean = norm_stats["mean"][tp_idx].item()
    tp_std = norm_stats["std"][tp_idx].item()

    # --- Model predictions ---
    all_mu_t2m = []
    all_sigma_t2m = []
    all_quantiles_tp = []
    all_y_t2m = []
    all_y_tp = []

    with torch.no_grad():
        for batch in loader:
            x_atmos = batch["x_atmos"].to(device)
            x_slow = batch["x_slow"].to(device)
            y_t2m = batch["y_t2m"].to(device)
            y_tp = batch["y_tp"].to(device)
            B = x_atmos.shape[0]
            x_index = torch.zeros(B, 6, device=device)

            pred = model(x_atmos, x_slow, x_index)

            # t2m — denormalize to physical units
            all_mu_t2m.append((pred["t2m"]["mu"] * t2m_std + t2m_mean).cpu().numpy())
            all_sigma_t2m.append((pred["t2m"]["sigma"] * t2m_std).cpu().numpy())
            all_y_t2m.append((y_t2m * t2m_std + t2m_mean).cpu().numpy())

            # tp — denormalize
            all_quantiles_tp.append((pred["tp"]["quantiles"] * tp_std + tp_mean).cpu().numpy())
            all_y_tp.append((y_tp * tp_std + tp_mean).cpu().numpy())

    # Concatenate
    mu_t2m = np.concatenate(all_mu_t2m, axis=0)
    sigma_t2m = np.concatenate(all_sigma_t2m, axis=0)
    y_t2m = np.concatenate(all_y_t2m, axis=0)
    quantiles_tp = np.concatenate(all_quantiles_tp, axis=0)
    y_tp = np.concatenate(all_y_tp, axis=0)

    # --- Compute metrics ---
    n_samples = mu_t2m.shape[0]
    print(f"\n{'='*50}")
    print(f"Evaluation on {n_samples} samples ({args.years[0]}-{args.years[1]})")
    print(f"{'='*50}")

    # t2m metrics (per-sample then average)
    t2m_crps_list = []
    t2m_rmse_list = []
    t2m_acc_list = []
    tp_rmse_list = []

    for i in range(n_samples):
        # t2m
        crps = compute_crps_gaussian(
            mu_t2m[i, 0], sigma_t2m[i, 0], y_t2m[i, 0], lat_weights
        )
        t2m_crps_list.append(crps)

        rmse = compute_rmse(mu_t2m[i, 0], y_t2m[i, 0], lat_weights)
        t2m_rmse_list.append(rmse)

        # tp: use median (quantile 0.5) as best estimate
        tp_median = quantiles_tp[i, 2]  # index 2 = 0.5 quantile (from [0.1, 0.25, 0.5, 0.75, 0.9])
        rmse_tp = compute_rmse(tp_median, y_tp[i, 0], lat_weights)
        tp_rmse_list.append(rmse_tp)

    t2m_crps_model = float(np.mean(t2m_crps_list))
    t2m_rmse_model = float(np.mean(t2m_rmse_list))
    tp_rmse_model = float(np.mean(tp_rmse_list))

    print(f"\n--- Model ---")
    print(f"  t2m CRPS:  {t2m_crps_model:.4f} K")
    print(f"  t2m RMSE:  {t2m_rmse_model:.4f} K")
    print(f"  tp  RMSE:  {tp_rmse_model:.6f} m")

    # --- Climatology baseline ---
    # Build climatology from training data
    climate = ClimatologyBaseline()
    train_data = WeatherDataset(
        data_dir=args.data_dir,
        variable_list=variable_list,
        lead_time=1,
        years=(1995, 2014),
        norm_stats=norm_stats,
    )
    # Fit climatology for t2m
    t2m_idx = variable_list.index("t2m")
    train_t2m = train_data.data[:, t2m_idx].numpy()  # (T, H, W)
    train_times = train_data.time_index

    # Simple climatology: mean of all training data (not month-specific for simplicity)
    t2m_clim = np.nanmean(train_t2m, axis=0)  # (H, W)
    # Use std as climatological uncertainty
    t2m_clim_std = np.nanstd(train_t2m, axis=0) + 1e-6

    t2m_crps_clim = 0.0
    t2m_rmse_clim = 0.0
    for i in range(n_samples):
        crps_c = compute_crps_gaussian(
            np.broadcast_to(t2m_clim, y_t2m[i, 0].shape),
            np.broadcast_to(t2m_clim_std, y_t2m[i, 0].shape),
            y_t2m[i, 0], lat_weights
        )
        t2m_crps_clim += crps_c
        t2m_rmse_clim += compute_rmse(
            np.broadcast_to(t2m_clim, y_t2m[i, 0].shape),
            y_t2m[i, 0], lat_weights
        )
    t2m_crps_clim /= n_samples
    t2m_rmse_clim /= n_samples

    crpss = compute_crpss(t2m_crps_model, t2m_crps_clim)

    print(f"\n--- Climatology Baseline ---")
    print(f"  t2m CRPS:  {t2m_crps_clim:.4f} K")
    print(f"  t2m RMSE:  {t2m_rmse_clim:.4f} K")
    print(f"\n--- Skill ---")
    print(f"  t2m CRPSS: {crpss:.4f}  (>0 = better than climatology)")

    if crpss > 0:
        print("  ✓ Model beats climatology!")
    else:
        print("  ✗ Model does not beat climatology yet (need more training / better data)")


if __name__ == "__main__":
    main()
