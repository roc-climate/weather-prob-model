"""
Encoder for slow boundary-forcing variables.

Processes SST, soil moisture, snow depth, and sea ice cover.
These variables have high memory and are the primary source of
predictability at 3-week+ lead times.

The slow encoder uses a lighter architecture than the atmos encoder
because slow variables have smoother spatial patterns.

Input:  [B, C_slow=4, H=121, W=240]  (sst, swvl1, sd, siconc)
Output: [B, D_embed, H_s, W_s] feature map for cross-attention fusion.
"""

import torch
import torch.nn as nn

from .encoder import ConvNeXtBlock, DownsampleBlock, LayerNorm2d


class SlowVarEncoder(nn.Module):
    """
    Lightweight ConvNeXt encoder for slow boundary-forcing variables.

    Uses fewer layers than the atmos encoder because:
    - SST/soil moisture/snow/ice have spatially smooth patterns
    - We want the encoder to preserve large-scale features
    - Fewer parameters reduce overfitting on the small training set

    Args:
        in_channels: Number of slow variable channels (default: 4)
        embed_dims: Channel dimensions for each stage
        depths: Number of ConvNeXt blocks per stage
    """
    def __init__(
        self,
        in_channels: int = 4,
        embed_dims: list = None,
        depths: list = None,
    ):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 256]
        if depths is None:
            depths = [2, 2, 4]

        # Stem: initial downsampling
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
            self.downsamples.append(DownsampleBlock(embed_dims[i], embed_dims[i + 1]))

        # Final stage
        self.final_stage = nn.Sequential(
            *[ConvNeXtBlock(embed_dims[-1]) for _ in range(depths[-1])]
        )

        self.out_dim = embed_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) with slow variables [sst, swvl1, sd, siconc]
        Returns:
            (B, out_dim, H_s, W_s) feature map
        """
        x = self.stem(x)

        for stage, downsample in zip(self.stages, self.downsamples):
            x = stage(x)
            x = downsample(x)

        x = self.final_stage(x)
        return x
