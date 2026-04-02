"""
2D Fourier Neural Operator (FNO) forecaster for Kraichnan turbulence.

Architecture per resolution level:
  Lifting:  Conv2d(1, width, 1)
  x n_fourier_layers:
      FNOBlock2d = SpectralConv2d(width, width, k_max_x, k_max_y)  [low-freq path]
               + Conv2d(width, width, 1)                            [local bypass]
               + GELU
  Projection: Conv2d(width, width//2, 1) -> GELU -> Conv2d(width//2, 1, 1)

SpectralConv2d:
  rfft2 input -> multiply lowest k_max modes (both positive and negative ky)
  by learned complex weights -> irfft2 back to physical space.

  Positive and negative ky modes each have their own weight tensor.
  Output shape equals input spatial shape (ny, nx) via irfft2(..., s=(ny, nx)).

k_max = resolution // 4 in each dimension (passed as constructor argument).

One FNO2d instance per resolution in the hierarchy: 64×64, 128×128, 256×256.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SpectralConv2d(nn.Module):
    """2D spectral convolution over the lowest k_max Fourier modes.

    Uses rfft2/irfft2.  The ky dimension (row axis) of the rfft2 output
    contains both positive (rows 0..k_max_y-1) and negative (rows
    -k_max_y..-1) frequencies; both are multiplied by separate learned
    weight tensors and accumulated into the output spectrum.

    Weight tensors (stored as real/imag pairs):
      weight_{pos,neg}_{real,imag}: (in_channels, out_channels, k_max_y, k_max_x)

    Args:
        in_channels:  C_in
        out_channels: C_out
        k_max_x:      Number of kx modes to retain (rightmost rfft2 axis)
        k_max_y:      Number of ky modes to retain per sign (row axis)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_max_x: int,
        k_max_y: int,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.k_max_x      = k_max_x
        self.k_max_y      = k_max_y

        # Xavier-style initialisation: scale by 1/sqrt(in_ch * out_ch)
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        shape = (in_channels, out_channels, k_max_y, k_max_x)

        # Positive ky weights
        self.weight_pos_real = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))
        self.weight_pos_imag = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))
        # Negative ky weights
        self.weight_neg_real = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))
        self.weight_neg_imag = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))

    def _cmul(
        self,
        x_hat: torch.Tensor,   # (B, C_in,  k_y, k_x) complex
        w_real: nn.Parameter,  # (C_in, C_out, k_y, k_x)
        w_imag: nn.Parameter,  # (C_in, C_out, k_y, k_x)
    ) -> torch.Tensor:
        """Complex multiply x_hat by weight and sum over C_in.

        einsum: 'biyx, ioyx -> boyx'
        Returns (B, C_out, k_y, k_x) complex.
        """
        W = torch.complex(w_real, w_imag)  # (C_in, C_out, k_y, k_x)
        return torch.einsum("biyx,ioyx->boyx", x_hat, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, ny, nx)

        Returns:
            (B, C_out, ny, nx)
        """
        B, C_in, ny, nx = x.shape

        # rfft2: (B, C_in, ny, nx//2+1)  complex
        x_hat = torch.fft.rfft2(x, norm="ortho")

        # Allocate zero output spectrum
        out_hat = torch.zeros(
            B, self.out_channels, ny, nx // 2 + 1,
            dtype=x_hat.dtype, device=x.device,
        )

        # ── Positive ky modes: rows 0 .. k_max_y-1 ──────────────────────────
        x_pos = x_hat[:, :, :self.k_max_y, :self.k_max_x]         # (B, C_in, k_y, k_x)
        out_hat[:, :, :self.k_max_y, :self.k_max_x] = self._cmul(
            x_pos, self.weight_pos_real, self.weight_pos_imag
        )

        # ── Negative ky modes: rows -k_max_y .. -1 ──────────────────────────
        x_neg = x_hat[:, :, -self.k_max_y:, :self.k_max_x]        # (B, C_in, k_y, k_x)
        out_hat[:, :, -self.k_max_y:, :self.k_max_x] = self._cmul(
            x_neg, self.weight_neg_real, self.weight_neg_imag
        )

        # irfft2 back to physical space: (B, C_out, ny, nx)
        return torch.fft.irfft2(out_hat, s=(ny, nx), norm="ortho")


class FNOBlock2d(nn.Module):
    """Single 2D FNO layer: spectral conv (low-freq) + bypass conv (local) + GELU.

    out = GELU( SpectralConv2d(x) + Conv2d(x, 1×1) )

    Args:
        width:   Number of channels (same in and out)
        k_max_x: kx mode truncation for SpectralConv2d
        k_max_y: ky mode truncation for SpectralConv2d
    """

    def __init__(self, width: int, k_max_x: int, k_max_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, k_max_x, k_max_y)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, width, ny, nx)

        Returns:
            (B, width, ny, nx)
        """
        return self.act(self.spectral(x) + self.bypass(x))


class FNO2d(nn.Module):
    """Full 2D FNO forecaster for one resolution level.

    Predicts w_{t+1} from w_t.  Input and output are both (B, 1, ny, nx).

    Args:
        cfg:        Full config; reads cfg.fno.{n_fourier_layers, width}
        resolution: Spatial resolution N (square grid N×N); sets k_max = N // 4
    """

    def __init__(self, cfg: DictConfig, resolution: int) -> None:
        super().__init__()
        width:    int = int(cfg.fno.width)           # 32
        n_layers: int = int(cfg.fno.n_fourier_layers) # 4
        k_max:    int = resolution // 4

        self.resolution = resolution
        self.k_max      = k_max
        self.width      = width

        # Lifting: 1 -> width  (pointwise, no spatial mixing)
        self.lift = nn.Conv2d(1, width, kernel_size=1)

        # FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, k_max, k_max) for _ in range(n_layers)]
        )

        # Projection: width -> width//2 -> 1
        self.proj = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, kernel_size=1),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w: Input vorticity field, shape (B, 1, ny, nx)

        Returns:
            Predicted next vorticity field, shape (B, 1, ny, nx)
        """
        x = self.lift(w)           # (B, width, ny, nx)
        for block in self.blocks:
            x = block(x)           # (B, width, ny, nx)
        return self.proj(x)        # (B, 1, ny, nx)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "fno": {"n_fourier_layers": 4, "width": 32, "activation": "gelu"},
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_passed = True
    for res in [64, 128, 256]:
        model = FNO2d(cfg, resolution=res).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        x = torch.randn(2, 1, res, res, device=device)
        with torch.no_grad():
            y = model(x)

        ok = (y.shape == x.shape)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(
            f"  [{status}] FNO2d  res={res:3d}×{res}  "
            f"k_max={res//4}  params={n_params:,}  "
            f"input={tuple(x.shape)}  output={tuple(y.shape)}"
        )

    sys.exit(0 if all_passed else 1)
