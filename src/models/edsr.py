"""
EDSR super-resolution model for the 32x32 -> 256x256 baseline.

Architecture follows EDSR-baseline (Lim et al., CVPRW 2017):
  - No batch normalisation (removed in EDSR — stabilises SR training)
  - Residual scaling (res_scale=0.1) for deep network stability
  - Sub-pixel convolution (PixelShuffle) for upscaling: three x2 stages = x8
  - L1 loss at training time

Key difference from the original paper:
  - All Conv2d layers use padding_mode='circular' because the vorticity domain
    is doubly periodic. Using zero-padding creates boundary artefacts on a
    torus — this makes the comparison fair against the iterative baseline.
  - Input: (B, 1, 32, 32)  raw coarse vorticity observation at time t
  - Output: (B, 1, 256, 256) super-resolved vorticity at time t
  - NO temporal context — each frame is upscaled independently.

Parameter count with defaults (n_feats=64, n_resblocks=16): ~1.5M
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Circular-padded Conv2d helper
# ---------------------------------------------------------------------------

def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding_mode: str = "circular",
    bias: bool = True,
) -> nn.Conv2d:
    """Conv2d with symmetric padding and configurable padding mode."""
    padding = kernel_size // 2
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=padding, padding_mode=padding_mode, bias=bias,
    )


# ---------------------------------------------------------------------------
# Residual block (no BN, optional residual scaling)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """EDSR residual block: Conv -> ReLU -> Conv, with optional residual scale.

    Removing BN is the key EDSR insight: BN normalises features and discards
    range flexibility, which hurts high-frequency SR reconstruction.

    Args:
        n_feats:      Number of feature channels.
        res_scale:    Scalar multiplied to the residual branch (default 0.1).
        padding_mode: Padding mode for Conv2d layers.
    """

    def __init__(
        self,
        n_feats: int,
        res_scale: float = 0.1,
        padding_mode: str = "circular",
    ) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            _conv(n_feats, n_feats, 3, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            _conv(n_feats, n_feats, 3, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


# ---------------------------------------------------------------------------
# Upsampler: stack of x2 pixel-shuffle blocks
# ---------------------------------------------------------------------------

class _UpsampleBlock(nn.Module):
    """Single x2 upsampling stage via sub-pixel convolution (PixelShuffle).

    Conv2d maps n_feats -> 4*n_feats, then PixelShuffle rearranges the
    extra channels into spatial extent (x2 in each dimension).

    Args:
        n_feats:      Feature channels in and out.
        padding_mode: Padding mode for the Conv2d layer.
    """

    def __init__(self, n_feats: int, padding_mode: str = "circular") -> None:
        super().__init__()
        self.conv = _conv(n_feats, n_feats * 4, 3, padding_mode=padding_mode)
        self.shuffle = nn.PixelShuffle(2)   # (B, 4C, H, W) -> (B, C, 2H, 2W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class Upsampler(nn.Sequential):
    """Stack of x2 upsampling blocks to reach the target scale factor.

    Args:
        scale:        Total upscale factor (must be a power of 2).
        n_feats:      Feature channels.
        padding_mode: Padding mode passed to each block.
    """

    def __init__(self, scale: int, n_feats: int, padding_mode: str = "circular") -> None:
        assert scale & (scale - 1) == 0 and scale >= 2, \
            f"scale must be a power of 2, got {scale}"
        blocks: list[nn.Module] = []
        s = scale
        while s > 1:
            blocks.append(_UpsampleBlock(n_feats, padding_mode=padding_mode))
            s //= 2
        super().__init__(*blocks)


# ---------------------------------------------------------------------------
# EDSR main model
# ---------------------------------------------------------------------------

class EDSR(nn.Module):
    """EDSR-baseline super-resolution network.

    Upscales a (B, 1, H_in, W_in) vorticity field to
    (B, 1, H_in*scale, W_in*scale).

    Architecture:
        head  : Conv2d(1, n_feats, 3)
        body  : n_resblocks x ResBlock(n_feats)
                + Conv2d(n_feats, n_feats, 3)  [long skip add target]
        upsample: Upsampler(scale, n_feats)
        tail  : Conv2d(n_feats, 1, 3)

    Args:
        n_resblocks:  Number of residual blocks (default 16).
        n_feats:      Feature channels (default 64).
        scale:        Upscaling factor, must be power of 2 (default 8).
        res_scale:    Residual branch scaling factor (default 0.1).
        padding_mode: Padding mode for all Conv2d layers (default "circular").
    """

    def __init__(
        self,
        n_resblocks:  int   = 16,
        n_feats:      int   = 64,
        scale:        int   = 8,
        res_scale:    float = 0.1,
        padding_mode: str   = "circular",
    ) -> None:
        super().__init__()

        # Head: lift 1 -> n_feats channels
        self.head = _conv(1, n_feats, 3, padding_mode=padding_mode)

        # Body: residual blocks + final conv (for long residual)
        body: list[nn.Module] = [
            ResBlock(n_feats, res_scale=res_scale, padding_mode=padding_mode)
            for _ in range(n_resblocks)
        ]
        body.append(_conv(n_feats, n_feats, 3, padding_mode=padding_mode))
        self.body = nn.Sequential(*body)

        # Upsampler: x2 x2 x2 = x8
        self.upsample = Upsampler(scale, n_feats, padding_mode=padding_mode)

        # Tail: project back to 1 channel
        self.tail = _conv(n_feats, 1, 3, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) low-resolution vorticity

        Returns:
            (B, 1, H*scale, W*scale) super-resolved vorticity
        """
        x = self.head(x)
        x = self.body(x) + x
        x = self.upsample(x)
        return self.tail(x)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = EDSR(n_resblocks=16, n_feats=64, scale=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(2, 1, 32, 32, device=device)
    with torch.no_grad():
        y = model(x)

    expected = (2, 1, 256, 256)
    ok = tuple(y.shape) == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] input={tuple(x.shape)}  output={tuple(y.shape)}")
    sys.exit(0 if ok else 1)
