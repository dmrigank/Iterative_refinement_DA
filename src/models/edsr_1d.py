"""
EDSR-1D super-resolution model for the 64-pt -> 512-pt Burgers baseline.

1D analogue of the 2D EDSR baseline (edsr.py).  The architecture mirrors
EDSR-baseline (Lim et al., CVPRW 2017) adapted for 1D sequences:
  - No batch normalisation.
  - Residual scaling (res_scale=0.1) for stability.
  - Upsampling via nearest-neighbour interpolation + Conv1d, applied three
    times (×2 each) to achieve ×8 total.  This is the standard 1D alternative
    to PixelShuffle (which is 2D-only in PyTorch).
  - Input:  (B, 1, 64)   raw coarse Burgers field at time t
  - Output: (B, 1, 512)  super-resolved field at time t
  - NO temporal context — each frame is upscaled independently.

Parameter count with defaults (n_feats=64, n_resblocks=16): ~0.9M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv1d helper (circular padding for periodic domain)
# ---------------------------------------------------------------------------

def _conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = True,
) -> nn.Conv1d:
    """Conv1d with circular padding (periodic Burgers domain)."""
    return nn.Conv1d(
        in_channels, out_channels, kernel_size,
        padding=kernel_size // 2,
        padding_mode="circular",
        bias=bias,
    )


# ---------------------------------------------------------------------------
# 1D Residual block (no BN)
# ---------------------------------------------------------------------------

class ResBlock1d(nn.Module):
    """EDSR 1D residual block: Conv → ReLU → Conv, with residual scaling.

    Args:
        n_feats:   Number of feature channels.
        res_scale: Scalar on the residual branch (default 0.1).
    """

    def __init__(self, n_feats: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            _conv1d(n_feats, n_feats, 3),
            nn.ReLU(inplace=True),
            _conv1d(n_feats, n_feats, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


# ---------------------------------------------------------------------------
# 1D Upsampling block: nearest-neighbour ×2 + Conv1d
# ---------------------------------------------------------------------------

class _UpsampleBlock1d(nn.Module):
    """Single ×2 upsampling stage: nearest-neighbour interpolation + conv.

    This is the standard 1D alternative to PixelShuffle (which is 2D-only).
    Circular padding is applied in the conv to preserve periodicity.

    Args:
        n_feats: Feature channels.
    """

    def __init__(self, n_feats: int) -> None:
        super().__init__()
        self.conv = _conv1d(n_feats, n_feats, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> (B, C, 2L)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Upsampler1d(nn.Sequential):
    """Stack of ×2 upsampling blocks.

    Args:
        scale: Total upscale factor (must be a power of 2).
        n_feats: Feature channels.
    """

    def __init__(self, scale: int, n_feats: int) -> None:
        assert scale & (scale - 1) == 0 and scale >= 2, \
            f"scale must be a power of 2, got {scale}"
        blocks: list[nn.Module] = []
        s = scale
        while s > 1:
            blocks.append(_UpsampleBlock1d(n_feats))
            s //= 2
        super().__init__(*blocks)


# ---------------------------------------------------------------------------
# EDSR-1D main model
# ---------------------------------------------------------------------------

class EDSR1d(nn.Module):
    """EDSR-1D super-resolution network: 64-pt → 512-pt Burgers field.

    Architecture:
        head     : Conv1d(1, n_feats, 3)
        body     : n_resblocks × ResBlock1d(n_feats)
                   + Conv1d(n_feats, n_feats, 3)   [long skip]
        upsample : Upsampler1d(scale, n_feats)
        tail     : Conv1d(n_feats, 1, 3)

    Args:
        n_resblocks: Number of residual blocks (default 16).
        n_feats:     Feature channels (default 64).
        scale:       Upscaling factor, must be power of 2 (default 8).
        res_scale:   Residual branch scaling in each block (default 0.1).
    """

    def __init__(
        self,
        n_resblocks: int   = 16,
        n_feats:     int   = 64,
        scale:       int   = 8,
        res_scale:   float = 0.1,
    ) -> None:
        super().__init__()

        self.head = _conv1d(1, n_feats, 3)

        body: list[nn.Module] = [
            ResBlock1d(n_feats, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        body.append(_conv1d(n_feats, n_feats, 3))
        self.body = nn.Sequential(*body)

        self.upsample = Upsampler1d(scale, n_feats)
        self.tail = _conv1d(n_feats, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N_in) low-resolution Burgers field

        Returns:
            (B, 1, N_in * scale) super-resolved field
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

    model = EDSR1d(n_resblocks=16, n_feats=64, scale=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(4, 1, 64, device=device)
    with torch.no_grad():
        y = model(x)

    ok = tuple(y.shape) == (4, 1, 512)
    print(f"  [{'PASS' if ok else 'FAIL'}] input={tuple(x.shape)}  output={tuple(y.shape)}")
    sys.exit(0 if ok else 1)
