"""
Shared 2D Fourier Neural Operator forecaster — one model for all resolutions.

Unlike FNO2d (src/models/fno_2d.py), which instantiates a separate model per
resolution because k_max = resolution // 4 directly sizes the SpectralConv2d
weight tensors, SharedFNO2d uses a single FIXED k_max across all resolutions.
The spectral weight tensors therefore have one shape, used identically at
64×64, 128×128, and 256×256.

At a given input resolution (ny, nx), only the modes that actually exist are
populated — SharedSpectralConv2d clamps k_max to min(k_max, ny//2, nx//2+1)
at forward time and zero-pads the rest. Lower resolutions simply see fewer
of the weight tensor's slots receive gradient on any given forward pass at
that resolution; the full tensor is exercised whenever a 256×256 batch (or
any resolution with ny//2 >= k_max) passes through.

No resolution embedding / FiLM conditioning is used. The architecture relies
on the input tensor's own spatial shape (i.e. how many Fourier modes are
nonzero) to carry the resolution signal — the operator's physical task
(predict w_{t+1} from w_t under local advection-diffusion) does not change
qualitatively across scale, unlike the diffusion corrector's stage-dependent
trust policy between forecast and coarse-observation channels.

Architecture mirrors FNO2d exactly (same lift/block/projection structure);
only the spectral conv's mode-truncation behaviour differs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


class SharedSpectralConv2d(nn.Module):
    """2D spectral convolution with a fixed k_max, clamped at forward time.

    Weight tensors are always shaped for k_max_shared modes. At forward time,
    if the input resolution offers fewer than k_max_shared modes in either
    axis (ny//2 or nx//2+1), only the available leading sub-block of the
    weight tensor is used; the rest is unused for that call.

    Args:
        in_channels:    C_in
        out_channels:   C_out
        k_max_shared:   Fixed number of modes to allocate weights for
                        (clamped down to whatever a given input resolution
                        actually has available)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_max_shared: int,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.k_max_shared = k_max_shared

        scale = 1.0 / (in_channels * out_channels) ** 0.5
        shape = (in_channels, out_channels, k_max_shared, k_max_shared)

        self.weight_pos_real = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))
        self.weight_pos_imag = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))
        self.weight_neg_real = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))
        self.weight_neg_imag = nn.Parameter(torch.empty(*shape).uniform_(-scale, scale))

    def _cmul(
        self,
        x_hat: torch.Tensor,
        w_real: torch.Tensor,
        w_imag: torch.Tensor,
    ) -> torch.Tensor:
        """einsum 'biyx,ioyx->boyx' complex multiply, summed over C_in."""
        W = torch.complex(w_real, w_imag)
        return torch.einsum("biyx,ioyx->boyx", x_hat, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, ny, nx) — ny, nx may be any resolution in the hierarchy

        Returns:
            (B, C_out, ny, nx)
        """
        B, C_in, ny, nx = x.shape

        # Clamp to the modes actually available at this resolution.
        k_eff_y = min(self.k_max_shared, ny // 2)
        k_eff_x = min(self.k_max_shared, nx // 2 + 1)

        x_hat = torch.fft.rfft2(x, norm="ortho")   # (B, C_in, ny, nx//2+1)

        out_hat = torch.zeros(
            B, self.out_channels, ny, nx // 2 + 1,
            dtype=x_hat.dtype, device=x.device,
        )

        # Positive ky modes: rows 0 .. k_eff_y-1, leading k_eff_x weight slice
        x_pos = x_hat[:, :, :k_eff_y, :k_eff_x]
        out_hat[:, :, :k_eff_y, :k_eff_x] = self._cmul(
            x_pos,
            self.weight_pos_real[:, :, :k_eff_y, :k_eff_x],
            self.weight_pos_imag[:, :, :k_eff_y, :k_eff_x],
        )

        # Negative ky modes: rows -k_eff_y .. -1, leading k_eff_x weight slice
        x_neg = x_hat[:, :, -k_eff_y:, :k_eff_x]
        out_hat[:, :, -k_eff_y:, :k_eff_x] = self._cmul(
            x_neg,
            self.weight_neg_real[:, :, :k_eff_y, :k_eff_x],
            self.weight_neg_imag[:, :, :k_eff_y, :k_eff_x],
        )

        return torch.fft.irfft2(out_hat, s=(ny, nx), norm="ortho")


class SharedFNOBlock2d(nn.Module):
    """Single shared FNO layer: spectral conv (fixed k_max) + bypass conv + GELU."""

    def __init__(self, width: int, k_max_shared: int) -> None:
        super().__init__()
        self.spectral = SharedSpectralConv2d(width, width, k_max_shared)
        self.bypass   = nn.Conv2d(width, width, kernel_size=1)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.bypass(x))


class SharedFNO2d(nn.Module):
    """One FNO forecaster used across all resolution levels (64, 128, 256).

    Architecture identical to FNO2d (lift -> n_fourier_layers blocks ->
    project), but every SharedFNOBlock2d uses the same fixed-shape spectral
    weight tensors regardless of the input's spatial resolution.

    Args:
        cfg:          Full config; reads cfg.fno.{n_fourier_layers, width}
        k_max_shared: Fixed mode truncation, shared across all resolutions
                      (default 64, matching the existing 256×256 FNO's cutoff)
    """

    def __init__(self, cfg: DictConfig, k_max_shared: int = 64) -> None:
        super().__init__()
        width:    int = int(cfg.fno.width)
        n_layers: int = int(cfg.fno.n_fourier_layers)

        self.k_max_shared = k_max_shared
        self.width         = width

        self.lift = nn.Conv2d(1, width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [SharedFNOBlock2d(width, k_max_shared) for _ in range(n_layers)]
        )

        self.proj = nn.Sequential(
            nn.Conv2d(width, width // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, kernel_size=1),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            w: Input vorticity field, shape (B, 1, ny, nx) — any resolution

        Returns:
            Predicted next vorticity field, shape (B, 1, ny, nx)
        """
        x = self.lift(w)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


class SharedFNOResolutionAdapter:
    """dict-like adapter so SharedFNO2d can be dropped into code expecting
    a {resolution: FNO2d} mapping (e.g. IterativeRefinementPipeline2d).

    adapter[64], adapter[128], adapter[256] all return the SAME underlying
    SharedFNO2d instance — only the call-site's input tensor shape differs.
    """

    def __init__(self, model: SharedFNO2d, resolutions: tuple[int, ...] = (64, 128, 256)) -> None:
        self._model = model
        self._resolutions = set(resolutions)

    def __getitem__(self, resolution: int) -> SharedFNO2d:
        if resolution not in self._resolutions:
            raise KeyError(f"SharedFNOResolutionAdapter: unsupported resolution {resolution}")
        return self._model

    def __contains__(self, resolution: int) -> bool:
        return resolution in self._resolutions

    def keys(self):
        return iter(self._resolutions)

    def values(self):
        return (self._model for _ in self._resolutions)

    def items(self):
        return ((r, self._model) for r in self._resolutions)


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

    model = SharedFNO2d(cfg, k_max_shared=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SharedFNO2d  k_max_shared=64  params={n_params:,}")

    all_passed = True
    for res in [64, 128, 256]:
        x = torch.randn(2, 1, res, res, device=device)
        with torch.no_grad():
            y = model(x)
        ok = (y.shape == x.shape)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        k_eff = min(64, res // 2)
        print(
            f"  [{status}] res={res:3d}×{res}  k_eff={k_eff:3d}  "
            f"input={tuple(x.shape)}  output={tuple(y.shape)}"
        )

    # Gradient sanity check: a 256x256 forward should populate gradients
    # across the FULL weight tensor; a 64x64 forward should only touch
    # the leading 32x32 sub-block.
    model.zero_grad()
    x64 = torch.randn(1, 1, 64, 64, device=device)
    model(x64).sum().backward()
    g = model.blocks[0].spectral.weight_pos_real.grad
    touched_64 = (g.abs() > 0).float().mean().item()
    print(f"  Fraction of weight tensor touched by 64×64 forward: {touched_64:.4f} "
          f"(expect ~{(32*32)/(64*64):.4f})")

    sys.exit(0 if all_passed else 1)
