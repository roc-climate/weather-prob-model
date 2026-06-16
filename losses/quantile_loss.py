"""
Quantile (pinball) loss for probabilistic prediction of skewed variables.

For precipitation, the Gaussian assumption is poor: the distribution is
zero-inflated and heavily right-skewed. Quantile regression provides a
distribution-free probabilistic prediction by directly predicting quantiles.

The quantile loss (pinball loss) for quantile level τ is:

  L_τ(q_τ, y) = max(τ * (y - q_τ), (τ - 1) * (y - q_τ))

where q_τ is the predicted τ-quantile and y is the observation.

Reference: Koenker & Bassett (1978) "Regression Quantiles"
"""

import torch
import torch.nn as nn

from .crps_loss import compute_latitude_weights


def quantile_loss_single(
    q: torch.Tensor,
    tau: float,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Pinball loss for a single quantile level.

    Args:
        q:   (...,) predicted τ-quantile
        tau: quantile level in (0, 1)
        y:   (...,) observed value

    Returns:
        (...,) pinball loss at each point
    """
    error = y - q
    return torch.maximum(tau * error, (tau - 1.0) * error)


def quantile_loss(
    quantiles: torch.Tensor,
    quantile_levels: list,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Pinball loss for multiple quantile levels.

    Args:
        quantiles:      (B, num_quantiles, H, W) predicted quantiles
        quantile_levels: list of τ values, e.g., [0.1, 0.25, 0.5, 0.75, 0.9]
        y:              (B, 1, H, W) observed values

    Returns:
        (B, 1, H, W) average pinball loss across quantile levels
    """
    B, Nq, H, W = quantiles.shape
    y_expanded = y.expand(-1, Nq, -1, -1)

    total_loss = 0.0
    for i, tau in enumerate(quantile_levels):
        q_i = quantiles[:, i:i+1, :, :]  # (B, 1, H, W)
        y_i = y_expanded[:, i:i+1, :, :]
        loss_i = quantile_loss_single(q_i, tau, y_i)
        total_loss = total_loss + loss_i

    return total_loss / Nq


class QuantilePinballLoss(nn.Module):
    """
    Quantile (pinball) loss with latitude weighting.

    Suitable for precipitation prediction where the distribution is skewed.

    Args:
        quantile_levels: List of quantile levels
        use_latitude_weight: If True, apply cos(lat) area-weighting
        n_lat: Number of latitude grid points
    """
    def __init__(
        self,
        quantile_levels: list = None,
        use_latitude_weight: bool = True,
        n_lat: int = 121,
    ):
        super().__init__()
        if quantile_levels is None:
            quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantile_levels = quantile_levels
        self.use_latitude_weight = use_latitude_weight
        self.register_buffer("lat_weights", compute_latitude_weights(n_lat))

    def forward(self, pred: dict, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: dict with keys "quantiles" (B, Nq, H, W) and "quantile_levels"
            y:    (B, 1, H, W) observed precipitation

        Returns:
            scalar pinball loss
        """
        q = pred["quantiles"]
        levels = pred.get("quantile_levels", self.quantile_levels)

        loss = quantile_loss(q, levels, y)

        if self.use_latitude_weight:
            weights = self.lat_weights.to(loss.device)
            if weights.shape[2] != loss.shape[2]:
                weights = torch.nn.functional.interpolate(
                    weights, size=(loss.shape[2], 1), mode="nearest"
                )[:, :, :loss.shape[2], :]
            loss = loss * weights

        return loss.mean()
