"""
ConvNeXt-based atmospheric encoder.

Lightweight encoder for processing the 7 atmospheric/radiative surface variables
(msl, u10, v10, tisr, ssr, str + t2m/tp as input state) on a 1.5° lat-lon grid.

Input:  [B, C_atmos, H=121, W=240]
Output: [B, D_embed, H_s, W_s] feature map for cross-attention fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block (Liu et al., 2022).
    Uses depthwise conv + pointwise conv with inverted bottleneck.
    """
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = nn.Identity()  # Simplified: no stochastic depth for MVP

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return x + shortcut


class DownsampleBlock(nn.Module):
    """Downsample with LayerNorm + 2x2 conv stride=2."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.conv(x)


class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt encoder for atmospheric variables.

    Args:
        in_channels: Number of input variable channels
        embed_dims: Channel dimensions for each stage
        depths: Number of ConvNeXt blocks per stage
    """
    def __init__(
        self,
        in_channels: int = 7,
        embed_dims: list = None,
        depths: list = None,
    ):
        super().__init__()
        if embed_dims is None:
            embed_dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], kernel_size=4, stride=4),
            LayerNorm2d(embed_dims[0]),
        )

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(len(embed_dims) - 1):
            stage = nn.Sequential(
                *[ConvNeXtBlock(embed_dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

            downsample = DownsampleBlock(embed_dims[i], embed_dims[i + 1])
            self.downsamples.append(downsample)

        # Final stage (no downsample after)
        self.final_stage = nn.Sequential(
            *[ConvNeXtBlock(embed_dims[-1]) for _ in range(depths[-1])]
        )

        self.out_dim = embed_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input tensor
        Returns:
            (B, out_dim, H_s, W_s) feature map
        """
        x = self.stem(x)  # Downsample 4x

        for stage, downsample in zip(self.stages, self.downsamples):
            x = stage(x)
            x = downsample(x)

        x = self.final_stage(x)
        return x
