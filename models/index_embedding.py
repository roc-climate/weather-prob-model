"""
Climate index embedding module.

Encodes scalar climate indices (MJO, ENSO, NAO, AO) and calendar features
(day-of-year) into a fixed-dimensional embedding vector that conditions the
probabilistic decoder.

For week-3+ prediction, these indices carry critical predictability signals
that are hard to extract from gridded data alone.
"""

import math
import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """1D sinusoidal encoding for scalar features."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1) scalar feature
        Returns:
            (B, embed_dim) encoded feature
        """
        half_dim = self.embed_dim // 2
        freq = torch.exp(
            torch.arange(0, half_dim, device=x.device) *
            (-math.log(10000.0) / half_dim)
        )
        angles = x * freq.unsqueeze(0)  # (B, half_dim)
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.embed_dim % 2 == 1:
            encoded = torch.cat([encoded, torch.zeros_like(encoded[:, :1])], dim=-1)
        return encoded


class IndexEmbedding(nn.Module):
    """
    Embeds climate indices and calendar features into a condition vector.

    Input indices (configurable, but typically):
      - RMM1 (MJO phase 1 component)
      - RMM2 (MJO phase 2 component)
      - Nino3.4 anomaly
      - NAO index
      - AO index
      - Day-of-year (continuous, 0-365)

    This ensures the model can explicitly condition on known S2S predictability
    sources rather than having to infer them from gridded data.

    Args:
        num_indices: Number of scalar index inputs
        embed_dim: Output embedding dimension
    """
    def __init__(self, num_indices: int = 6, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_indices = num_indices

        # Per-index sinusoidal encoding + small MLP
        self.encoders = nn.ModuleList([
            SinusoidalEncoding(embed_dim) for _ in range(num_indices)
        ])

        # Combine individual encodings
        self.combine = nn.Sequential(
            nn.Linear(num_indices * embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: (B, num_indices) tensor of scalar indices
        Returns:
            (B, embed_dim) condition embedding vector
        """
        B = indices.shape[0]
        encoded_list = []

        for i, encoder in enumerate(self.encoders):
            feat = indices[:, i:i+1]  # (B, 1)
            encoded = encoder(feat)    # (B, embed_dim)
            encoded_list.append(encoded)

        combined = torch.cat(encoded_list, dim=-1)  # (B, num_indices * embed_dim)
        output = self.combine(combined)             # (B, embed_dim)
        return output
