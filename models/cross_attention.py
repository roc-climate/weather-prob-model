"""
Cross-attention module for non-local teleconnection fusion.

At 3-week+ lead times, teleconnections (e.g., tropical Pacific SST →
North American temperature) are the primary source of predictability.
Standard CNNs have limited receptive fields and may struggle to learn
these long-range dependencies.

This module uses cross-attention to explicitly model:
  "How does a feature at location (i,j) in the slow variables
   influence a prediction at location (i',j')?"

Architecture: Deformable / Perceiver-style cross-attention
  - slow features act as queries (what we want to know about)
  - atmos features act as keys/values (context to draw from)
  - Output is a fused feature map attending to global context
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """2D sinusoidal position embedding for lat-lon grid features."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """
        Returns (1, dim, h, w) position embeddings.
        Uses sine/cosine encoding for both y (latitude) and x (longitude).
        """
        half_dim = self.dim // 4

        # Coordinate grids: normalized to [-1, 1]
        y_grid = torch.linspace(-1.0, 1.0, h, device=device)  # (H,)
        x_grid = torch.linspace(-1.0, 1.0, w, device=device)  # (W,)

        y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing="ij")  # each (H, W)

        # Frequency bands
        freqs = torch.exp(
            torch.arange(0, half_dim, device=device)
            * (-math.log(10000.0) / half_dim)
        )  # (half_dim,)

        freqs_xy = freqs.view(1, 1, -1) * math.pi  # (1, 1, half_dim)

        # Encode y: (H, W) → (H, W, 1) * (1, 1, half_dim) → (H, W, half_dim)
        y_enc = y_mesh.unsqueeze(-1) * freqs_xy
        y_sin = torch.sin(y_enc)
        y_cos = torch.cos(y_enc)

        # Encode x
        x_enc = x_mesh.unsqueeze(-1) * freqs_xy
        x_sin = torch.sin(x_enc)
        x_cos = torch.cos(x_enc)

        # Concat → (H, W, 4 * half_dim) = (H, W, dim)
        pe = torch.cat([y_sin, y_cos, x_sin, x_cos], dim=-1)

        return pe.permute(2, 0, 1).unsqueeze(0)  # (1, dim, H, W)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention where slow_var features attend to atmos features.

    This explicitly models teleconnections: for each spatial position in
    the slow-variable feature map, the block extracts relevant information
    from all positions in the atmospheric feature map.
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.norm_slow = nn.LayerNorm(dim)
        self.norm_atmos = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)   # Query from slow vars
        self.k_proj = nn.Linear(dim, dim)   # Key from atmos vars
        self.v_proj = nn.Linear(dim, dim)   # Value from atmos vars

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_slow: torch.Tensor,
        x_atmos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_slow:  (B, C, H, W) slow variable features (queries)
            x_atmos: (B, C, H, W) atmos variable features (keys/values)
        Returns:
            (B, C, H, W) fused features
        """
        B, C, H, W = x_slow.shape

        # Reshape to sequences: (B, H*W, C)
        slow_seq = x_slow.flatten(2).transpose(1, 2)   # (B, L_q, C)
        atmos_seq = x_atmos.flatten(2).transpose(1, 2)  # (B, L_kv, C)

        # Normalize
        slow_norm = self.norm_slow(slow_seq)
        atmos_norm = self.norm_atmos(atmos_seq)

        # Project to Q, K, V
        Q = self.q_proj(slow_norm)
        K = self.k_proj(atmos_norm)
        V = self.v_proj(atmos_norm)

        # Reshape for multi-head: (B, L, num_heads, head_dim) → (B, num_heads, L, head_dim)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (Q @ K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V  # (B, num_heads, L_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)

        # Residual connection + reshape back to 2D
        out = out.transpose(1, 2).view(B, C, H, W)
        return x_slow + out


class CrossAttentionFusion(nn.Module):
    """
    Multi-layer cross-attention fusion module.

    Stacks multiple CrossAttentionBlocks with feed-forward networks
    to iteratively refine the fused representation.

    Args:
        dim: Feature dimension (must match slow_encoder.out_dim == atmos_encoder.out_dim)
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout rate
    """
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_embed = SinusoidalPositionEmbedding(dim)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        z_slow: torch.Tensor,
        z_atmos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_slow:  (B, C, H, W) slow variable features
            z_atmos: (B, C, H, W) atmos variable features
        Returns:
            (B, C, H, W) fused features
        """
        # Add position embeddings (separate for each, sizes may differ)
        _, _, Hs, Ws = z_slow.shape
        _, _, Ha, Wa = z_atmos.shape
        z_slow = z_slow + self.pos_embed(Hs, Ws, z_slow.device)
        z_atmos = z_atmos + self.pos_embed(Ha, Wa, z_atmos.device)

        # Cross-attention layers
        for layer in self.layers:
            z_slow = layer(z_slow, z_atmos)

        # Feed-forward
        B, C, H_s, W_s = z_slow.shape
        z_flat = z_slow.flatten(2).transpose(1, 2)  # (B, L, C)
        z_flat = self.ffn(z_flat) + z_flat
        z_out = self.norm_out(z_flat)
        z_out = z_out.transpose(1, 2).view(B, C, H_s, W_s)

        return z_out
