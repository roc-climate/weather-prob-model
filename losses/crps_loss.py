"""
Continuous Ranked Probability Score (CRPS) for Gaussian distributions.

CRPS is a strictly proper scoring rule for probabilistic forecasts.
For a Gaussian predictive distribution N(μ, σ²) and observation y:

  CRPS(N(μ,σ), y) = σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]

where z = (y - μ) / σ, Φ is the Gaussian CDF, φ is the Gaussian PDF.

This is the analytic form — no sampling needed.

Reference: Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"
"""

import math
import torch
import torch.nn as nn


SQRT_PI = math.sqrt(math.pi)
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute CRPS for Gaussian predictive distribution.

    Args:
        mu:    (...,) predicted mean
        sigma: (...,) predicted standard deviation (must be > 0)
        y:     (...,) observed value

    Returns:
        (...,) CRPS at each grid point, averaged over batch/spatial dims
    """
    z = (y - mu) / sigma

    # Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    # Standard normal PDF: φ(z) = exp(-z²/2) / sqrt(2π)
    pdf = torch.exp(-0.5 * z * z - LOG_SQRT_2PI)

    crps = sigma * (z * (2.0 * phi - 1.0) + 2.0 * pdf - 1.0 / SQRT_PI)
    return crps


def compute_latitude_weights(
    n_lat: int,
    device: torch.device = None,
    lat_range: tuple = (-89.25, 89.25),
) -> torch.Tensor:
    """
    Compute cos(latitude) weights for area-weighted averaging.

    At 1.5° resolution: n_lat = 121, ranging from -90 to 90.
    Weights are proportional to cos(lat) to account for converging meridians.

    Args:
        n_lat: Number of latitude grid points
        device: Target device
        lat_range: (lat_min, lat_max) in degrees

    Returns:
        (1, 1, n_lat, 1) weight tensor
    """
    lat_min, lat_max = lat_range
    latitudes = torch.linspace(lat_min, lat_max, n_lat, device=device)
    weights = torch.cos(torch.deg2rad(latitudes))
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights.view(1, 1, -1, 1)


class GaussianCRPSLoss(nn.Module):
    """
    CRPS loss for Gaussian predictive distributions with latitude weighting.

    Args:
        use_latitude_weight: If True, weight by cos(latitude) for area-weighting
        n_lat: Number of latitude grid points
    """
    def __init__(self, use_latitude_weight: bool = True, n_lat: int = 121):
        super().__init__()
        self.use_latitude_weight = use_latitude_weight
        self.n_lat = n_lat
        self.register_buffer("lat_weights", compute_latitude_weights(n_lat))

    def forward(self, pred: dict, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: dict with keys "mu" (B,1,H,W) and "sigma" (B,1,H,W)
            y:    (B, 1, H, W) observed values

        Returns:
            scalar CRPS loss
        """
        mu = pred["mu"]
        sigma = pred["sigma"]
        crps = gaussian_crps(mu, sigma, y)  # (B, 1, H, W)

        if self.use_latitude_weight:
            weights = self.lat_weights.to(crps.device)
            # Interpolate weights if n_lat doesn't match
            if weights.shape[2] != crps.shape[2]:
                weights = torch.nn.functional.interpolate(
                    weights, size=(crps.shape[2], 1), mode="nearest"
                )[:, :, :crps.shape[2], :]
            crps = crps * weights

        return crps.mean()
