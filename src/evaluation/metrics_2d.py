"""
Evaluation metrics for 2D Kraichnan turbulence DA pipeline.

All functions accept PyTorch tensors and run on CPU or GPU.

Conventions:
  - Fields have shape (ny, nx) or (n_time, ny, nx).
  - Domain is doubly periodic: [0, Lx] × [0, Ly], default Lx=Ly=2π.
  - Wavenumbers are physical (cycles per domain), matching the convention
    used in the solver.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse_2d(
    pred: torch.Tensor,
    truth: torch.Tensor,
    per_timestep: bool = False,
) -> torch.Tensor:
    """RMSE over the spatial dimensions (ny, nx).

    Args:
        pred:         (n_time, ny, nx) or (ny, nx)
        truth:        same shape as pred
        per_timestep: if True, return (n_time,); if False, return scalar

    Returns:
        (n_time,) if per_timestep else scalar tensor
    """
    assert pred.shape == truth.shape, f"Shape mismatch: {pred.shape} vs {truth.shape}"
    sq_err = (pred - truth).pow(2)

    if pred.dim() == 2:
        return sq_err.mean().sqrt()

    # (n_time, ny, nx)
    per_t = sq_err.mean(dim=(-2, -1)).sqrt()   # (n_time,)
    return per_t if per_timestep else per_t.mean()


def rmse_over_time_2d(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """RMSE at each time step, averaged over spatial dimensions.

    Args:
        pred:  (n_time, ny, nx)
        truth: (n_time, ny, nx)

    Returns:
        (n_time,)
    """
    assert pred.dim() == 3, "Expected (n_time, ny, nx)"
    assert pred.shape == truth.shape
    return (pred - truth).pow(2).mean(dim=(-2, -1)).sqrt()   # (n_time,)


# ---------------------------------------------------------------------------
# Radial energy spectrum
# ---------------------------------------------------------------------------

def radial_energy_spectrum(
    w: torch.Tensor,
    lx: float = 2.0 * math.pi,
    ly: float = 2.0 * math.pi,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Radial energy spectrum E(k) from vorticity field(s).

    Computes |ω̂(kx, ky)|² via rfft2, then bins into integer radial shells
    k_r = round(sqrt(kx² + ky²)).  If input is (n_time, ny, nx), averages
    the spectrum over time.

    The normalisation uses rfft2 with norm='forward' so |ω̂|² has units of
    physical energy density (independent of grid size).

    Args:
        w:  (ny, nx) or (n_time, ny, nx) vorticity field(s)
        lx: Domain length in x (default 2π)
        ly: Domain length in y (default 2π)

    Returns:
        E_k:      (k_max+1,) tensor — energy in each radial shell
        k_bins:   (k_max+1,) tensor — integer wavenumber of each shell center
    """
    if w.dim() == 2:
        w = w.unsqueeze(0)    # (1, ny, nx)

    n_time, ny, nx = w.shape
    device = w.device

    # 2D power spectrum, averaged over time
    # norm='forward' divides by ny*nx so amplitudes are physical
    w_hat = torch.fft.rfft2(w.float(), norm="forward")   # (n_time, ny, nx//2+1)
    power = w_hat.abs().pow(2).mean(dim=0)               # (ny, nx//2+1)

    # Integer mode-index wavenumber grids (0, 1, 2, ..., N/2).
    # d=1/N gives frequencies in cycles/sample; multiplying back by N gives
    # integer mode indices directly.  This is independent of domain length.
    kx = torch.fft.fftfreq(ny,  d=1.0 / ny, device=device)   # (ny,)   range: -N/2..N/2-1
    ky = torch.fft.rfftfreq(nx, d=1.0 / nx, device=device)   # (nx//2+1,)  range: 0..N/2

    # Physical wavenumber magnitude at each grid point
    kx_grid = kx.unsqueeze(1).expand(ny, nx // 2 + 1)   # (ny, nx//2+1)
    ky_grid = ky.unsqueeze(0).expand(ny, nx // 2 + 1)   # (ny, nx//2+1)
    k_mag   = (kx_grid.pow(2) + ky_grid.pow(2)).sqrt()  # (ny, nx//2+1)

    # Bin into integer radial shells
    k_int = k_mag.round().long()                         # (ny, nx//2+1)
    k_max = int(k_int.max().item())

    E_k = torch.zeros(k_max + 1, device=device, dtype=power.dtype)
    E_k.scatter_add_(0, k_int.reshape(-1), power.reshape(-1))

    k_bins = torch.arange(k_max + 1, device=device, dtype=torch.float32)

    return E_k, k_bins


# ---------------------------------------------------------------------------
# Enstrophy
# ---------------------------------------------------------------------------

def enstrophy(w: torch.Tensor) -> torch.Tensor:
    """Enstrophy Z = 0.5 * mean(ω²) over spatial dims at each time step.

    Args:
        w: (n_time, ny, nx) vorticity

    Returns:
        (n_time,) enstrophy per frame
    """
    assert w.dim() == 3, "Expected (n_time, ny, nx)"
    return 0.5 * w.pow(2).mean(dim=(-2, -1))   # (n_time,)


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

def temporal_consistency_2d(w_sequence: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame L2 displacement of a 2D vorticity time sequence.

    Args:
        w_sequence: (n_time, ny, nx)

    Returns:
        (n_time-1,) — ||w_t - w_{t-1}||_2 for t = 1 .. n_time-1
    """
    assert w_sequence.dim() == 3, "Expected (n_time, ny, nx)"
    diff = w_sequence[1:] - w_sequence[:-1]              # (n_time-1, ny, nx)
    return diff.pow(2).sum(dim=(-2, -1)).sqrt()          # (n_time-1,)


# ---------------------------------------------------------------------------
# Structural similarity (SSIM)
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """1D Gaussian -> outer product -> 2D kernel, normalised to sum=1."""
    coords = torch.arange(window_size, device=device, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)   # (window_size, window_size)


def structural_similarity_2d(
    pred: torch.Tensor,
    truth: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """SSIM computed per frame then averaged over the time sequence.

    Uses a Gaussian-weighted sliding window matching the standard Wang et al.
    (2004) formulation.  The dynamic range for C1/C2 is assumed to be 1.0
    (C1 and C2 can be passed as absolute values if the field range differs).

    Args:
        pred:        (n_time, ny, nx) or (ny, nx)
        truth:       same shape as pred
        window_size: Gaussian window side length (default 11)
        sigma:       Gaussian std dev (default 1.5)
        C1, C2:      Stability constants (default SSIM paper values for L=1)

    Returns:
        Scalar mean SSIM across frames (range approximately [-1, 1], 1=perfect)
    """
    assert pred.shape == truth.shape

    scalar_input = pred.dim() == 2
    if scalar_input:
        pred  = pred.unsqueeze(0)
        truth = truth.unsqueeze(0)

    n_time, ny, nx = pred.shape
    device = pred.device

    # Build conv kernel: (1, 1, window_size, window_size)
    kernel_2d = _gaussian_kernel_2d(window_size, sigma, device)
    kernel    = kernel_2d.unsqueeze(0).unsqueeze(0)   # (1, 1, W, W)
    pad       = window_size // 2

    # Process all frames in one batched conv by treating time as batch dim
    x = pred.float().unsqueeze(1)    # (n_time, 1, ny, nx)
    y = truth.float().unsqueeze(1)   # (n_time, 1, ny, nx)

    def _filt(t: torch.Tensor) -> torch.Tensor:
        # reflect pad to handle boundaries, then convolve
        t_pad = F.pad(t, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(t_pad, kernel)   # (n_time, 1, ny, nx)

    mu_x    = _filt(x)
    mu_y    = _filt(y)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy   = mu_x * mu_y

    sigma_x_sq = _filt(x.pow(2)) - mu_x_sq
    sigma_y_sq = _filt(y.pow(2)) - mu_y_sq
    sigma_xy   = _filt(x * y)    - mu_xy

    numerator   = (2.0 * mu_xy   + C1) * (2.0 * sigma_xy   + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator.clamp(min=1e-8)   # (n_time, 1, ny, nx)
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def compute_all_metrics_2d(
    pred: torch.Tensor,
    truth: torch.Tensor,
    fno_pred: torch.Tensor | None = None,
    lx: float = 2.0 * math.pi,
    ly: float = 2.0 * math.pi,
) -> dict[str, torch.Tensor]:
    """Compute all scalar 2D metrics.

    Args:
        pred:     Diffusion posterior, shape (n_time, ny, nx)
        truth:    Ground truth,        shape (n_time, ny, nx)
        fno_pred: FNO-only forecast,   shape (n_time, ny, nx) — optional
        lx, ly:   Domain lengths (default 2π)

    Returns:
        Dict of metric_name -> scalar tensor
    """
    assert pred.shape == truth.shape

    metrics: dict[str, torch.Tensor] = {
        "rmse":                 rmse_2d(pred, truth),
        "enstrophy_error":      (enstrophy(pred) - enstrophy(truth)).abs().mean(),
        "temporal_consistency": temporal_consistency_2d(pred).mean(),
        "ssim":                 structural_similarity_2d(pred, truth),
    }

    if fno_pred is not None:
        assert fno_pred.shape == truth.shape
        metrics["fno_rmse"]         = rmse_2d(fno_pred, truth)
        metrics["rmse_improvement"] = metrics["fno_rmse"] - metrics["rmse"]
        metrics["fno_ssim"]         = structural_similarity_2d(fno_pred, truth)

    return metrics
