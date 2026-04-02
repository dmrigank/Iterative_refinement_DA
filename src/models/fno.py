"""
1D Fourier Neural Operator (FNO) forecaster.

Architecture per resolution level:
  Lifting:  Conv1d(1, width, kernel_size=1)
  x n_fourier_layers:
      FNOBlock = SpectralConv1d(width, width, k_max)  [low-freq path]
              + Conv1d(width, width, kernel_size=1)    [local bypass]
              + GELU
  Projection: Conv1d(width, width, kernel_size=1) -> GELU -> Conv1d(width, 1, kernel_size=1)

SpectralConv1d:
  - rfft input along spatial dim
  - multiply first k_max modes by a learned complex weight matrix W of shape (C_in, C_out, k_max)
  - irfft back to physical space
  - only the low k_max modes are transformed; the irfft reconstructs a full N-length output

k_max = resolution // 4  (set per-resolution at construction)

One FNO1d instance per resolution in the hierarchy: 128, 256, 512.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SpectralConv1d(nn.Module):
    """1D spectral convolution operating on the lowest k_max Fourier modes.

    Learnable parameter: complex weight tensor of shape (in_channels, out_channels, k_max).
    Stored as two real tensors (real and imag parts) for compatibility with
    standard optimisers and checkpointing.

    Args:
        in_channels: Number of input channels C_in
        out_channels: Number of output channels C_out
        k_max: Number of Fourier modes to keep (truncation threshold)
    """

    def __init__(self, in_channels: int, out_channels: int, k_max: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_max = k_max

        # Initialise weights with Xavier-uniform scaled by 1/sqrt(in_channels)
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        self.weight_real = nn.Parameter(
            torch.empty(in_channels, out_channels, k_max).uniform_(-scale, scale)
        )
        self.weight_imag = nn.Parameter(
            torch.empty(in_channels, out_channels, k_max).uniform_(-scale, scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, N)

        Returns:
            (B, C_out, N)
        """
        B, C_in, N = x.shape

        # Transform to Fourier space: (B, C_in, N//2+1) complex
        x_hat = torch.fft.rfft(x, norm="ortho")

        # Truncate to k_max modes: (B, C_in, k_max)
        x_hat_trunc = x_hat[..., : self.k_max]

        # Complex weight matrix: (C_in, C_out, k_max)
        W = torch.complex(self.weight_real, self.weight_imag)

        # Batched matrix multiply over modes:
        # x_hat_trunc: (B, C_in, k_max)  ->  want (B, C_out, k_max)
        # einsum 'bik,iok->bok'  (i=C_in, o=C_out, k=k_max)
        out_hat_trunc = torch.einsum("bik,iok->bok", x_hat_trunc, W)  # (B, C_out, k_max)

        # Pad back to N//2+1 modes with zeros
        out_hat = torch.zeros(
            B, self.out_channels, N // 2 + 1,
            dtype=x_hat.dtype, device=x.device,
        )
        out_hat[..., : self.k_max] = out_hat_trunc

        # Transform back to physical space
        return torch.fft.irfft(out_hat, n=N, norm="ortho")  # (B, C_out, N)


class FNOBlock(nn.Module):
    """Single FNO layer: spectral conv (low-freq) + bypass conv (local) + activation.

    out = GELU( SpectralConv1d(x) + Conv1d(x) )

    Args:
        width: Number of channels (same in and out)
        k_max: Fourier mode truncation for SpectralConv1d
    """

    def __init__(self, width: int, k_max: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, k_max)
        self.bypass = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, width, N)

        Returns:
            (B, width, N)
        """
        return self.act(self.spectral(x) + self.bypass(x))


class FNO1d(nn.Module):
    """Full 1D FNO forecaster for one resolution level.

    Predicts u_{t+1} from u_t.  Input and output are both (B, 1, N_r).

    Args:
        cfg: Full config; reads cfg.fno.{n_fourier_layers, width}
        resolution: Spatial resolution N_r; sets k_max = N_r // 4
    """

    def __init__(self, cfg: DictConfig, resolution: int) -> None:
        super().__init__()
        width: int = int(cfg.fno.width)
        n_layers: int = int(cfg.fno.n_fourier_layers)
        k_max: int = resolution // 4

        self.resolution = resolution
        self.k_max = k_max
        self.width = width

        # Lifting: 1 -> width  (pointwise)
        self.lift = nn.Conv1d(1, width, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock(width, k_max) for _ in range(n_layers)]
        )

        # Projection: width -> width -> 1
        self.proj = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(width, 1, kernel_size=1),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: Input field, shape (B, 1, N_r)

        Returns:
            Predicted next field, shape (B, 1, N_r)
        """
        x = self.lift(u)          # (B, width, N_r)
        for block in self.blocks:
            x = block(x)          # (B, width, N_r)
        return self.proj(x)       # (B, 1, N_r)
