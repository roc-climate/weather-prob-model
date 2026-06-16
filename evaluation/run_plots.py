#!/usr/bin/env python3
"""
Generate evaluation plots: spatial skill map, actual-vs-predicted scatter,
rank histogram, and lead-time decay.

Usage:
  python -m evaluation.run_plots --checkpoint ./checkpoints/best_model.pt \
      --data_dir ./data/raw/era5_monthly --years 2015 2016 --output ./results
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
    compute_latitude_weights_np,
    compute_rmse,
    compute_acc,
)
from evaluation.plot_results import (
    set_style,
    plot_rank_histogram,
    plot_spatial_skill,
    plot_actual_vs_predicted,
)
from evaluation.calibration import rank_histogram as compute_rank_hist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/raw/era5_monthly")
    parser.add_argument("--norm_stats", type=str, default="./data/processed/norm_stats.json")
    parser.add_argument("--years", type=int, nargs=2, default=[2015, 2016])
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_style()
    lat_weights = compute_latitude_weights_np()
    lats = np.linspace(-90.0, 90.0, 121)
    lons = np.linspace(0.0, 358.5, 240)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"].get("model", {})

    from models.weather_model import WeatherProbModel
    model = WeatherProbModel(
        n_atmos_vars=6, n_slow_vars=4, n_indices=6,
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

    variable_list = [
        "t2m", "tp", "sst", "swvl1", "sd", "siconc",
        "msl", "u10", "v10", "tisr", "ssr", "str",
    ]
    norm_stats = load_statistics(args.norm_stats)

    t2m_idx = variable_list.index("t2m")
    tp_idx = variable_list.index("tp")
    t2m_mean = norm_stats["mean"][t2m_idx].item()
    t2m_std = norm_stats["std"][t2m_idx].item()
    tp_mean = norm_stats["mean"][tp_idx].item()
    tp_std = norm_stats["std"][tp_idx].item()

    dataset = WeatherDataset(
        data_dir=args.data_dir, variable_list=variable_list,
        lead_time=1, years=tuple(args.years), norm_stats=norm_stats,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- Gather predictions ---
    all_mu_t2m = []
    all_sigma_t2m = []
    all_y_t2m = []
    all_quantiles_tp = []
    all_y_tp = []

    with torch.no_grad():
        for batch in loader:
            x_atmos = batch["x_atmos"].to(device)
            x_slow = batch["x_slow"].to(device)
            y_t2m = batch["y_t2m"].to(device)
            y_tp = batch["y_tp"].to(device)
            x_index = torch.zeros(1, 6, device=device)

            pred = model(x_atmos, x_slow, x_index)

            all_mu_t2m.append(pred["t2m"]["mu"].cpu().numpy())
            all_sigma_t2m.append(pred["t2m"]["sigma"].cpu().numpy())
            all_y_t2m.append(y_t2m.cpu().numpy())
            all_quantiles_tp.append(pred["tp"]["quantiles"].cpu().numpy())
            all_y_tp.append(y_tp.cpu().numpy())

    mu_t2m_norm = np.concatenate(all_mu_t2m, axis=0)      # (N, 1, H, W)
    sigma_t2m_norm = np.concatenate(all_sigma_t2m, axis=0)
    y_t2m_norm = np.concatenate(all_y_t2m, axis=0)
    quantiles_tp_norm = np.concatenate(all_quantiles_tp, axis=0)
    y_tp_norm = np.concatenate(all_y_tp, axis=0)

    # Denormalize
    mu_t2m = mu_t2m_norm * t2m_std + t2m_mean
    sigma_t2m = sigma_t2m_norm * t2m_std
    y_t2m = y_t2m_norm * t2m_std + t2m_mean
    quantiles_tp = quantiles_tp_norm * tp_std + tp_mean
    y_tp = y_tp_norm * tp_std + tp_mean

    n_samples = mu_t2m.shape[0]
    print(f"Generating plots from {n_samples} samples...")

    # ============================================
    # 1. Spatial skill map: model error per grid point (t2m)
    # ============================================
    # Average squared error over all samples
    t2m_error = np.zeros((121, 240), dtype=np.float32)
    for i in range(n_samples):
        t2m_error += (mu_t2m[i, 0] - y_t2m[i, 0]) ** 2
    t2m_error = np.sqrt(t2m_error / n_samples)

    # Compute climatology error for comparison
    train_dataset = WeatherDataset(
        data_dir=args.data_dir, variable_list=variable_list,
        lead_time=1, years=(1995, 2014), norm_stats=norm_stats,
    )
    train_t2m = train_dataset.data[:, t2m_idx].numpy()  # already in physical space
    t2m_clim = np.mean(train_t2m, axis=0)  # (H, W)
    clim_error = np.zeros((121, 240), dtype=np.float32)
    for i in range(n_samples):
        clim_error += (t2m_clim - y_t2m[i, 0]) ** 2
    clim_error = np.sqrt(clim_error / n_samples)

    # Skill improvement: negative = better than climatology
    skill_map = t2m_error - clim_error  # (H, W) — negative values = model better

    plot_spatial_skill(
        skill_map, lats, lons,
        str(output_dir / "spatial_skill_t2m.png"),
        title="t2m RMSE Improvement over Climatology (negative = model better)",
        cmap="RdBu_r",
        label="RMSE difference (K)",
    )

    # Also plot the model's raw RMSE map
    plot_spatial_skill(
        t2m_error, lats, lons,
        str(output_dir / "spatial_rmse_t2m.png"),
        title="t2m RMSE by Location",
        cmap="Reds",
        vmin=0,
        label="RMSE (K)",
    )

    # ============================================
    # 2. Actual vs Predicted scatter
    # ============================================
    # Flatten spatial and use a subset for clarity
    flat_mu = mu_t2m.flatten()
    flat_y = y_t2m.flatten()
    # Subsample to 5000 points
    idx = np.random.default_rng(42).choice(len(flat_mu), min(5000, len(flat_mu)), replace=False)
    plot_actual_vs_predicted(
        flat_y[idx], flat_mu[idx],
        str(output_dir / "actual_vs_predicted_t2m.png"),
        title="t2m: Actual vs Predicted",
        xlabel="Predicted (K)",
        ylabel="Actual (K)",
    )

    # ============================================
    # 3. Rank Histogram (calibration)
    # ============================================
    # Generate ensemble by sampling from predicted Gaussian distribution
    n_ensemble = 50
    all_ensemble = []
    for i in range(n_samples):
        ensemble_i = np.random.normal(
            mu_t2m[i, 0], sigma_t2m[i, 0], size=(n_ensemble, 121, 240)
        )
        all_ensemble.append(ensemble_i)
    ensemble = np.concatenate(all_ensemble, axis=0)  # (N*n_ens, H, W) — but we want ensemble over samples

    # Actually, create ensemble per spatial point
    # Sample points spatially for efficiency
    n_points = min(50000, 121 * 240)
    rng_spatial = np.random.default_rng(42)
    flat_idx = rng_spatial.choice(121 * 240, n_points, replace=False)
    hi, wi = np.unravel_index(flat_idx, (121, 240))

    ensemble_pts = np.zeros((n_ensemble, n_points * n_samples), dtype=np.float32)
    obs_pts = np.zeros(n_points * n_samples, dtype=np.float32)

    for i in range(n_samples):
        for j, (h, w) in enumerate(zip(hi, wi)):
            mu_val = mu_t2m[i, 0, h, w]
            sigma_val = sigma_t2m[i, 0, h, w]
            ensemble_pts[:, i * n_points + j] = np.random.normal(mu_val, sigma_val, n_ensemble)
            obs_pts[i * n_points + j] = y_t2m[i, 0, h, w]

    hist, bin_edges = compute_rank_hist(ensemble_pts, obs_pts)
    plot_rank_histogram(
        hist, bin_edges,
        str(output_dir / "rank_histogram_t2m.png"),
        title="t2m Rank Histogram (Gaussian forecast)",
    )

    # ============================================
    # 4. Precipitation: quantile comparison
    # ============================================
    # Flatten and subsample
    flat_q50 = quantiles_tp[:, 2].flatten()  # median
    flat_ytp = y_tp.flatten()
    idx_tp = np.random.default_rng(42).choice(len(flat_q50), min(5000, len(flat_q50)), replace=False)
    plot_actual_vs_predicted(
        flat_ytp[idx_tp], flat_q50[idx_tp],
        str(output_dir / "actual_vs_predicted_tp.png"),
        title="Precipitation: Actual vs Predicted (median)",
        xlabel="Predicted (m)",
        ylabel="Actual (m)",
    )

    print(f"\nAll plots saved to {output_dir.resolve()}/")
    print("  - spatial_skill_t2m.png")
    print("  - spatial_rmse_t2m.png")
    print("  - actual_vs_predicted_t2m.png")
    print("  - rank_histogram_t2m.png")
    print("  - actual_vs_predicted_tp.png")


if __name__ == "__main__":
    main()
