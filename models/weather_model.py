"""
Full weather probability model.

Assembles the encoders, fusion module, index embedding, and probabilistic heads
into a single end-to-end model.

Architecture:
  Input:  x_atmos (atmos_state + energy_budget + external_forcing vars)
          x_slow  (slow boundary forcing vars: sst, swvl1, sd, siconc)
          x_index (climate indices: MJO, ENSO, NAO, AO, doy)

  Processing:
    x_atmos → AtmosEncoder → z_atmos
    x_slow  → SlowEncoder  → z_slow
    x_index → IndexEmbedding → z_idx

    z_slow + z_atmos → CrossAttentionFusion → z_fused

    z_fused + z_idx → GaussianHead → (mu, sigma) for t2m
    z_fused + z_idx → QuantileHead → quantiles for tp

  Output: dict with t2m and tp distribution parameters
"""

import torch
import torch.nn as nn

from .encoder import ConvNeXtEncoder
from .slow_encoder import SlowVarEncoder
from .cross_attention import CrossAttentionFusion
from .index_embedding import IndexEmbedding
from .prob_head import GaussianHead, QuantileHead


class WeatherProbModel(nn.Module):
    """
    Global 1.5° weekly probabilistic weather prediction model.

    Designed for 3-week+ lead time prediction of t2m and precipitation.

    Args:
        n_atmos_vars: Number of atmospheric/radiative input channels
        n_slow_vars: Number of slow boundary-forcing input channels
        n_indices: Number of scalar climate indices
        atmos_encoder_kwargs: Kwargs for ConvNeXtEncoder
        slow_encoder_kwargs: Kwargs for SlowVarEncoder
        cross_attn_kwargs: Kwargs for CrossAttentionFusion
        index_embed_kwargs: Kwargs for IndexEmbedding
        gaussian_head_kwargs: Kwargs for GaussianHead (t2m)
        quantile_head_kwargs: Kwargs for QuantileHead (tp)
    """
    def __init__(
        self,
        n_atmos_vars: int = 6,
        n_slow_vars: int = 4,
        n_indices: int = 6,
        atmos_encoder_kwargs: dict = None,
        slow_encoder_kwargs: dict = None,
        cross_attn_kwargs: dict = None,
        index_embed_kwargs: dict = None,
        gaussian_head_kwargs: dict = None,
        quantile_head_kwargs: dict = None,
    ):
        super().__init__()

        # --- Encoders ---
        atmos_enc_cfg = atmos_encoder_kwargs or {}
        atmos_enc_cfg.setdefault("in_channels", n_atmos_vars)
        self.atmos_encoder = ConvNeXtEncoder(**atmos_enc_cfg)

        slow_enc_cfg = slow_encoder_kwargs or {}
        slow_enc_cfg.setdefault("in_channels", n_slow_vars)
        self.slow_encoder = SlowVarEncoder(**slow_enc_cfg)

        # --- Spatial alignment ---
        # Both encoders must output the same spatial resolution.
        # Atmos encoder uses 4 stem + 2x downsample per stage (3 stages) = stride 32? -> depends.
        # We'll use a shared projector to align channel dims if needed.
        cross_cfg = cross_attn_kwargs or {}
        fusion_dim = cross_cfg.get("dim", 256)

        self.atmos_proj = nn.Conv2d(self.atmos_encoder.out_dim, fusion_dim, kernel_size=1)
        self.slow_proj = nn.Conv2d(self.slow_encoder.out_dim, fusion_dim, kernel_size=1)

        # --- Cross-attention fusion ---
        self.cross_attention = CrossAttentionFusion(**cross_cfg)

        # --- Index embedding ---
        idx_cfg = index_embed_kwargs or {}
        idx_cfg.setdefault("num_indices", n_indices)
        self.index_embedding = IndexEmbedding(**idx_cfg)
        self.index_dim = idx_cfg.get("embed_dim", 64)

        # --- Probabilistic heads ---
        gauss_cfg = gaussian_head_kwargs or {}
        gauss_cfg.setdefault("in_dim", fusion_dim + self.index_dim)
        self.t2m_head = GaussianHead(**gauss_cfg)

        quant_cfg = quantile_head_kwargs or {}
        quant_cfg.setdefault("in_dim", fusion_dim + self.index_dim)
        self.tp_head = QuantileHead(**quant_cfg)

    def _broadcast_index(self, z_idx: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Broadcast index embedding to spatial dimensions."""
        # z_idx: (B, D_idx)
        # Returns: (B, D_idx, h, w)
        z_idx_spatial = z_idx.unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        z_idx_spatial = z_idx_spatial.expand(-1, -1, h, w)  # (B, D, h, w)
        return z_idx_spatial

    def forward(
        self,
        x_atmos: torch.Tensor,
        x_slow: torch.Tensor,
        x_index: torch.Tensor,
    ) -> dict:
        """
        Args:
            x_atmos: (B, C_a, H, W) atmospheric/radiative variables
            x_slow:  (B, C_s, H, W) slow boundary-forcing variables
            x_index: (B, N_idx) scalar climate indices

        Returns:
            dict:
              t2m: {"mu": (B,1,H,W), "sigma": (B,1,H,W), "logvar": (B,1,H,W)}
              tp:  {"quantiles": (B,N_q,H,W), "quantile_levels": list}
        """
        # Encode
        z_atmos = self.atmos_encoder(x_atmos)      # (B, C_a', H', W')
        z_slow = self.slow_encoder(x_slow)           # (B, C_s', H', W')

        # Project to common dimension
        z_atmos = self.atmos_proj(z_atmos)           # (B, D, H', W')
        z_slow = self.slow_proj(z_slow)              # (B, D, H', W')

        # Fuse via cross-attention
        z_fused = self.cross_attention(z_slow, z_atmos)  # (B, D, H', W')

        # Index embedding
        z_idx = self.index_embedding(x_index)        # (B, D_idx)

        # Concatenate index to fused features (spatially tiled)
        _, _, Hf, Wf = z_fused.shape
        z_idx_spatial = self._broadcast_index(z_idx, Hf, Wf)
        z_decoder = torch.cat([z_fused, z_idx_spatial], dim=1)  # (B, D+D_idx, H', W')

        # Predict
        t2m_pred = self.t2m_head(z_decoder)
        tp_pred = self.tp_head(z_decoder)

        return {"t2m": t2m_pred, "tp": tp_pred}
