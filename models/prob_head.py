"""
Probabilistic prediction heads.

Two head types for Phase 1:
  - GaussianHead: Outputs μ and log(σ) for CRPS-based training (suitable for t2m)
  - QuantileHead: Outputs multiple quantiles for quantile loss (suitable for tp)

Both heads decode from the fused feature representation and produce
per-grid-point, per-variable distribution parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianHead(nn.Module):
    """
    Predicts Gaussian distribution parameters (μ, log σ) for each grid point.

    The head upsamples from the encoded feature map back to the full resolution
    and outputs two channels: mu and logvar.

    Args:
        in_dim: Input feature dimension from fusion module
        hidden_dims: Dimensions of hidden layers in the decoder
        out_h: Target output height (default: 121 for 1.5°)
        out_w: Target output width (default: 240 for 1.5°)
    """
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dims: list = None,
        out_h: int = 121,
        out_w: int = 240,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Decoder: progressively upsample back to native resolution
        layers = []
        curr_h, curr_w = out_h // 16, out_w // 16  # After 4+2+2 = 16x downsampling (adjust as needed)

        # We receive features at ~1/16 resolution (after stem + 2 downsamples in fusion)
        # Upsample to ~1/8, then ~1/4, then native
        for i, hdim in enumerate(hidden_dims):
            # Upsample block
            layers.append(nn.ConvTranspose2d(
                in_dim if i == 0 else hidden_dims[i - 1],
                hdim,
                kernel_size=4, stride=2, padding=1
            ))
            layers.append(nn.GroupNorm(min(32, hdim), hdim))
            layers.append(nn.GELU())
            curr_h *= 2
            curr_w *= 2

        self.decoder = nn.Sequential(*layers)
        last_dim = hidden_dims[-1] if hidden_dims else in_dim

        # Additional upsample to native resolution if needed
        # Native H=121, W=240. After 3 upsamples from ~7×15: 56×120 → still need more
        # We'll do adaptive pooling to exact dimensions
        self.mu_head = nn.Conv2d(last_dim, 1, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(last_dim, 1, kernel_size=3, padding=1)

        self.out_h = out_h
        self.out_w = out_w

        # Softplus for variance stability
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: (B, C, H_z, W_z) fused features
        Returns:
            dict with:
              mu:     (B, 1, H, W) predicted mean
              logvar: (B, 1, H, W) predicted log variance
              sigma:  (B, 1, H, W) predicted std (softplus of logvar)
        """
        x = self.decoder(z)

        # Upsample to target resolution
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)

        mu = self.mu_head(x)                    # (B, 1, H, W)
        logvar_raw = self.logvar_head(x)        # (B, 1, H, W)

        # Stabilize: clip logvar and convert to sigma
        logvar = torch.clamp(logvar_raw, min=-10.0, max=10.0)
        sigma = self.softplus(logvar_raw) + 1e-6

        return {"mu": mu, "logvar": logvar, "sigma": sigma}


class QuantileHead(nn.Module):
    """
    Predicts multiple quantiles for each grid point.

    Suitable for skewed distributions like precipitation.
    Uses quantile loss (pinball loss) for training.

    Args:
        in_dim: Input feature dimension
        quantiles: List of quantile levels (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
        hidden_dims: Decoder hidden dimensions
        out_h, out_w: Target output resolution
    """
    def __init__(
        self,
        in_dim: int = 256,
        quantiles: list = None,
        hidden_dims: list = None,
        out_h: int = 121,
        out_w: int = 240,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Shared decoder
        layers = []
        for i, hdim in enumerate(hidden_dims):
            layers.append(nn.ConvTranspose2d(
                in_dim if i == 0 else hidden_dims[i - 1],
                hdim,
                kernel_size=4, stride=2, padding=1
            ))
            layers.append(nn.GroupNorm(min(32, hdim), hdim))
            layers.append(nn.GELU())

        self.decoder = nn.Sequential(*layers)
        last_dim = hidden_dims[-1] if hidden_dims else in_dim

        # Per-quantile output heads (lightweight conv for each quantile)
        self.quantile_heads = nn.ModuleList([
            nn.Conv2d(last_dim, 1, kernel_size=3, padding=1)
            for _ in range(self.num_quantiles)
        ])

        self.out_h = out_h
        self.out_w = out_w

    def forward(self, z: torch.Tensor) -> dict:
        """
        Args:
            z: (B, C, H_z, W_z) fused features
        Returns:
            dict with:
              quantiles: (B, num_quantiles, H, W) predicted quantile values
              quantile_levels: list of tau values
        """
        x = self.decoder(z)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)

        quantile_outputs = []
        for head in self.quantile_heads:
            q = head(x)  # (B, 1, H, W)
            quantile_outputs.append(q)

        # Stack: (B, num_quantiles, H, W)
        quantiles = torch.cat(quantile_outputs, dim=1)

        # Enforce monotonicity: soft sort across quantile dimension
        quantiles = torch.sort(quantiles, dim=1).values

        return {
            "quantiles": quantiles,
            "quantile_levels": self.quantiles,
        }
