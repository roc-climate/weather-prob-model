"""
Plotting utilities for model evaluation.

Generates:
  - CRPS/RMSE lead-time decay curves
  - Rank histograms
  - Spatial maps of skill (CRPSS, ACC)
  - Actual vs predicted scatter plots
"""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def set_style():
    """Apply consistent plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def plot_lead_time_decay(
    lead_times: list,
    crps_values: list,
    crps_baseline: list,
    rmse_values: list,
    acc_values: list,
    output_path: str,
    title: str = "Skill vs Lead Time",
):
    """
    Plot skill metrics as a function of lead time.

    Args:
        lead_times: List of lead time labels (e.g., ["1w", "2w", "3w", "4w"])
        crps_values: Model CRPS at each lead time
        crps_baseline: Climatology CRPS at each lead time
        rmse_values: Model RMSE at each lead time
        acc_values: Model ACC at each lead time
        output_path: Path to save figure
    """
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # CRPS
    ax = axes[0]
    ax.plot(lead_times, crps_values, "o-", label="Model", color="tab:blue", linewidth=2)
    ax.plot(lead_times, crps_baseline, "s--", label="Climatology", color="gray", linewidth=1.5)
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("CRPS")
    ax.set_title("CRPS")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE
    ax = axes[1]
    ax.plot(lead_times, rmse_values, "o-", color="tab:orange", linewidth=2)
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE (Ensemble Mean)")
    ax.grid(True, alpha=0.3)

    # ACC
    ax = axes[2]
    ax.plot(lead_times, acc_values, "o-", color="tab:green", linewidth=2)
    ax.axhline(y=0.6, color="gray", linestyle="--", alpha=0.5, label="Useful (ACC=0.6)")
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("ACC")
    ax.set_title("Anomaly Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved lead-time plot to {output_path}")


def plot_rank_histogram(
    hist: np.ndarray,
    bin_edges: np.ndarray,
    output_path: str,
    title: str = "Rank Histogram",
):
    """
    Plot rank histogram for calibration assessment.

    Args:
        hist: (n_bins,) normalized histogram values
        bin_edges: (n_bins + 1,) bin edges
        output_path: Path to save figure
        title: Plot title
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    n_bins = len(hist)
    uniform_val = 1.0 / n_bins
    bw = bin_edges[1] - bin_edges[0]

    ax.bar(bin_edges[:-1], hist, width=bw, alpha=0.7, color="tab:blue", edgecolor="white")
    ax.axhline(y=uniform_val, color="red", linestyle="--", linewidth=2,
               label=f"Uniform ({uniform_val:.3f})")

    ax.set_xlabel("Rank of Observation")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate deviation
    max_dev = np.max(np.abs(hist - uniform_val))
    ax.text(0.95, 0.95, f"Max dev: {max_dev:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", alpha=0.1))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved rank histogram to {output_path}")


def plot_spatial_skill(
    values: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    output_path: str,
    title: str = "Spatial Skill",
    cmap: str = "RdBu_r",
    vmin: float = None,
    vmax: float = None,
    label: str = "CRPSS",
):
    """
    Plot a global map of skill values.

    Args:
        values: (H, W) skill values
        lats: (H,) latitude array
        lons: (W,) longitude array
        output_path: Path to save figure
        title: Plot title
        cmap: Colormap
        vmin, vmax: Color scale limits
        label: Colorbar label
    """
    set_style()
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.set_global()

    if vmin is None:
        vmax_abs = np.nanmax(np.abs(values))
        vmin, vmax = -vmax_abs, vmax_abs

    mesh = ax.pcolormesh(
        lons, lats, values,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto",
    )

    cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.7)
    cbar.set_label(label)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved spatial map to {output_path}")


def plot_actual_vs_predicted(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: str,
    title: str = "Actual vs Predicted",
    xlabel: str = "Predicted",
    ylabel: str = "Actual",
):
    """
    Scatter plot of actual vs predicted values.

    Args:
        actual: (N,) observed values
        predicted: (N,) predicted values (ensemble mean)
        output_path: Path to save figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    # Hexbin for large datasets, scatter for small
    if len(actual) > 10000:
        hb = ax.hexbin(predicted, actual, gridsize=50, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(predicted, actual, alpha=0.3, s=2, color="tab:blue")

    # Identity line
    lims = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="Perfect fit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # R² annotation
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
            va="top", fontsize=11, bbox=dict(boxstyle="round", alpha=0.1))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved scatter plot to {output_path}")
