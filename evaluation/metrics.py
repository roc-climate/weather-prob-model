"""
Evaluation metrics for probabilistic weather predictions.

Metrics:
  - CRPS: Continuous Ranked Probability Score (for Gaussian and ensemble)
  - CRPSS: CRPS Skill Score (relative to climatology baseline)
  - RMSE: Root Mean Square Error (deterministic, on ensemble mean)
  - ACC: Anomaly Correlation Coefficient
  - Rank Histogram: Calibration diagnostic
  - Spread-Skill Ratio: Ensemble dispersion vs error
"""

import math
import numpy as np
import torch

from losses.crps_loss import gaussian_crps


SQRT_PI = math.sqrt(math.pi)
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


# ---- CRPS ----

def compute_crps_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    lat_weights: np.ndarray = None,
) -> float:
    """
    Compute global CRPS for Gaussian predictive distribution.

    Args:
        mu: (H, W) predicted mean
        sigma: (H, W) predicted standard deviation
        y: (H, W) observed value
        lat_weights: (H, 1) latitude weights (cos(lat))

    Returns:
        scalar CRPS
    """
    z = (y - mu) / np.maximum(sigma, 1e-6)
    phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z - LOG_SQRT_2PI)
    crps = sigma * (z * (2.0 * phi - 1.0) + 2.0 * pdf - 1.0 / SQRT_PI)

    if lat_weights is not None:
        crps = crps * lat_weights

    return float(np.mean(crps))


def compute_crps_ensemble(
    ensemble: np.ndarray,
    y: np.ndarray,
    lat_weights: np.ndarray = None,
) -> float:
    """
    Compute CRPS from ensemble samples (no distribution assumed).

    Uses the fair CRPS formula for ensemble predictions:

      CRPS = (1/M) Σ|X_i - y| - (1/(2M²)) Σ Σ|X_i - X_j|

    Args:
        ensemble: (M, H, W) ensemble member predictions
        y: (H, W) observed value
        lat_weights: (H, 1) latitude weights

    Returns:
        scalar CRPS
    """
    M = ensemble.shape[0]

    # Term 1: (1/M) Σ|X_i - y|
    abs_error = np.abs(ensemble - y[np.newaxis, :, :])  # (M, H, W)
    term1 = abs_error.mean(axis=0)  # (H, W)

    # Term 2: (1/(2M²)) Σ Σ|X_i - X_j|
    # Efficient O(M) computation using sorted ensemble
    sorted_ens = np.sort(ensemble, axis=0)  # (M, H, W)
    # For sorted values: Σ Σ|x_i - x_j| = 2 Σ i*(2i - M - 1) * x_i
    weights = 2.0 * np.arange(M) - M + 1  # (M,)
    # Double sum = 2 / M^2 * Σ w_i * sorted_x_i
    term2_inner = (weights[:, np.newaxis, np.newaxis] * sorted_ens).sum(axis=0)
    term2 = term2_inner / (M * M)  # (H, W)

    crps = term1 - term2

    if lat_weights is not None:
        crps = crps * lat_weights

    return float(np.mean(crps))


# ---- CRPSS (Skill Score) ----

def compute_crpss(crps_model: float, crps_baseline: float) -> float:
    """
    CRPS Skill Score relative to a baseline.

    CRPSS = 1 - (CRPS_model / CRPS_baseline)
    > 0 means better than baseline.
    """
    if crps_baseline < 1e-10:
        return 0.0
    return 1.0 - crps_model / crps_baseline


# ---- RMSE ----

def compute_rmse(
    pred: np.ndarray,
    y: np.ndarray,
    lat_weights: np.ndarray = None,
) -> float:
    """
    Compute area-weighted RMSE.

    Args:
        pred: (H, W) predicted (can be ensemble mean)
        y: (H, W) observed
        lat_weights: (H, 1) latitude weights

    Returns:
        scalar RMSE
    """
    sq_error = (pred - y) ** 2

    if lat_weights is not None:
        sq_error = sq_error * lat_weights

    return float(np.sqrt(np.mean(sq_error)))


# ---- ACC (Anomaly Correlation Coefficient) ----

def compute_acc(
    pred: np.ndarray,
    y: np.ndarray,
    climatology: np.ndarray,
    lat_weights: np.ndarray = None,
) -> float:
    """
    Compute Anomaly Correlation Coefficient.

    ACC = <(pred - clim)(y - clim)> / sqrt(<(pred - clim)^2> <(y - clim)^2>)

    Args:
        pred: (H, W) predicted field
        y: (H, W) observed field
        climatology: (H, W) climatological mean
        lat_weights: (H, 1) latitude weights

    Returns:
        scalar ACC
    """
    pred_anom = pred - climatology
    y_anom = y - climatology

    if lat_weights is not None:
        w = lat_weights
        num = np.sum(pred_anom * y_anom * w)
        denom = np.sqrt(np.sum(pred_anom ** 2 * w) * np.sum(y_anom ** 2 * w))
    else:
        num = np.sum(pred_anom * y_anom)
        denom = np.sqrt(np.sum(pred_anom ** 2) * np.sum(y_anom ** 2))

    if denom < 1e-10:
        return 0.0
    return float(num / denom)


# ---- Rank Histogram ----

def compute_rank_histogram(
    ensemble: np.ndarray,
    y: np.ndarray,
    n_bins: int = None,
) -> np.ndarray:
    """
    Compute rank histogram for ensemble calibration assessment.

    For a perfectly calibrated ensemble, the rank of the observation
    within the pooled ensemble+observation set should be uniform.

    Args:
        ensemble: (M, H, W) ensemble members
        y: (H, W) observation
        n_bins: Number of bins (default: M + 1)

    Returns:
        (n_bins,) histogram counts normalized to probability
    """
    M = ensemble.shape[0]
    if n_bins is None:
        n_bins = M + 1

    # For each grid point, count how many ensemble members are < observation
    ranks = np.sum(ensemble < y[np.newaxis, :, :], axis=0)  # (H, W)
    ranks = ranks.flatten()

    hist, _ = np.histogram(ranks, bins=n_bins, range=(-0.5, M + 0.5), density=False)
    return hist / hist.sum()


# ---- Spread-Skill Ratio ----

def compute_spread_skill_ratio(
    ensemble: np.ndarray,
    y: np.ndarray,
    lat_weights: np.ndarray = None,
) -> float:
    """
    Compute ensemble spread-skill ratio.

    Spread = sqrt(mean(ensemble variance over members))
    Skill = RMSE of ensemble mean

    Ratio ≈ 1.0 for well-calibrated ensemble
    Ratio < 1.0 → overconfident (spread too small)
    Ratio > 1.0 → underconfident (spread too large)

    Args:
        ensemble: (M, H, W) ensemble members
        y: (H, W) observed
        lat_weights: (H, 1) latitude weights

    Returns:
        spread_skill_ratio
    """
    M = ensemble.shape[0]
    ens_mean = ensemble.mean(axis=0)  # (H, W)

    # Spread: sqrt of mean variance
    variance = np.mean((ensemble - ens_mean[np.newaxis, :, :]) ** 2, axis=0)  # (H, W)
    spread = np.sqrt(variance)

    # Skill: RMSE of ensemble mean
    skill = np.abs(ens_mean - y)

    if lat_weights is not None:
        w = lat_weights
        mean_spread = np.sqrt(np.sum(spread ** 2 * w) / np.sum(w))
        mean_skill = np.sqrt(np.sum(skill ** 2 * w) / np.sum(w))
    else:
        mean_spread = np.sqrt(np.mean(spread ** 2))
        mean_skill = np.sqrt(np.mean(skill ** 2))

    if mean_skill < 1e-10:
        return 1.0
    return float(mean_spread / mean_skill)


# ---- Utility ----

def compute_latitude_weights_np(
    n_lat: int = 121,
    lat_range: tuple = (-89.25, 89.25),
) -> np.ndarray:
    """Compute cos(lat) weights as numpy array."""
    latitudes = np.linspace(lat_range[0], lat_range[1], n_lat)
    weights = np.cos(np.deg2rad(latitudes))
    weights = weights / weights.mean()
    return weights[:, np.newaxis].astype(np.float32)
