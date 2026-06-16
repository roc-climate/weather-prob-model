"""
Calibration diagnostics for probabilistic forecasts.

Functions:
  - rank_histogram: Diagnostic for ensemble calibration
  - spread_skill_ratio: Ensemble dispersion vs error
  - temperature_scaling: Post-hoc calibration of predicted variances
  - pit_histogram: Probability Integral Transform histogram for continuous forecasts
"""

import numpy as np
from typing import Tuple


def rank_histogram(
    ensemble: np.ndarray,
    observation: np.ndarray,
    n_bins: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rank histogram.

    Args:
        ensemble: (M, N) or (M, H, W) — ensemble members × spatial points
        observation: (N,) or (H, W) — observed values
        n_bins: Number of bins (default: M + 1)

    Returns:
        hist: (n_bins,) normalized histogram
        bin_edges: (n_bins + 1,) bin edges
    """
    if ensemble.ndim > 1:
        # Flatten spatial dimensions
        shape = ensemble.shape
        M = shape[0]
        ensemble = ensemble.reshape(M, -1)
        observation = observation.reshape(-1)

    M = ensemble.shape[0]
    if n_bins is None:
        n_bins = M + 1

    # Rank: count ensemble members < observation at each point
    ranks = np.sum(ensemble < observation[np.newaxis, :], axis=0)

    hist, bin_edges = np.histogram(
        ranks, bins=n_bins, range=(-0.5, M + 0.5), density=True
    )

    return hist, bin_edges


def spread_skill_ratio(
    ensemble: np.ndarray,
    observation: np.ndarray,
    lat_weights: np.ndarray = None,
) -> float:
    """
    Compute spread-skill ratio.

    Spread = sqrt(mean(ensemble variance))
    Skill = RMSE of ensemble mean

    Ideal ratio ≈ 1.0
    """
    M = ensemble.shape[0]
    ens_mean = ensemble.mean(axis=0)  # (H, W)
    variance = np.mean((ensemble - ens_mean[np.newaxis, :, :]) ** 2, axis=0)
    spread_sq = np.mean(variance) if lat_weights is None else \
        np.average(variance.flatten(), weights=lat_weights.flatten())

    error_sq = (ens_mean - observation) ** 2
    skill_sq = np.mean(error_sq) if lat_weights is None else \
        np.average(error_sq.flatten(), weights=lat_weights.flatten())

    spread = np.sqrt(spread_sq)
    skill = np.sqrt(skill_sq)

    if skill < 1e-10:
        return 1.0
    return float(spread / skill)


def temperature_scale(
    mu: np.ndarray,
    sigma: np.ndarray,
    observation: np.ndarray,
    validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Apply temperature scaling to calibrate predicted variances.

    Finds scaling factor T such that:
      sigma_calibrated = sigma * sqrt(T)

    by minimizing the CRPS on validation data.

    Args:
        mu: (N, H, W) or (T, H, W) predicted means
        sigma: (N, H, W) predicted std
        observation: (N, H, W) observations
        validation_data: (mu_val, sigma_val, obs_val) for T optimization

    Returns:
        calibrated_sigma: sigma * sqrt(T)
        T: optimal temperature
    """
    mu_val, sigma_val, obs_val = validation_data

    # Grid search for optimal T
    T_candidates = np.logspace(-1, 1, 21)  # 0.1 to 10
    best_T = 1.0
    best_loss = float("inf")

    from losses.crps_loss import gaussian_crps
    import torch

    for T in T_candidates:
        # Compute CRPS with scaled sigma
        scaled_sigma = sigma_val * np.sqrt(T)
        loss = gaussian_crps(
            torch.from_numpy(mu_val),
            torch.from_numpy(scaled_sigma),
            torch.from_numpy(obs_val),
        ).mean().item()

        if loss < best_loss:
            best_loss = loss
            best_T = T

    calibrated_sigma = sigma * np.sqrt(best_T)
    return calibrated_sigma, best_T


def pit_histogram(
    mu: np.ndarray,
    sigma: np.ndarray,
    observation: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """
    Probability Integral Transform (PIT) histogram.

    For a Gaussian forecast, compute:
      PIT_i = Φ((y_i - μ_i) / σ_i)

    For a perfectly calibrated forecast, PIT ~ Uniform(0, 1).

    Args:
        mu: (N,) predicted means
        sigma: (N,) predicted std
        observation: (N,) observations
        n_bins: Number of histogram bins

    Returns:
        (n_bins,) normalized histogram
    """
    from scipy.stats import norm

    z = (observation - mu) / np.maximum(sigma, 1e-6)
    pit = norm.cdf(z)

    hist, _ = np.histogram(pit, bins=n_bins, range=(0, 1), density=True)
    # Normalize so that uniform = 1.0
    hist = hist / np.mean(hist)
    return hist
