"""
Evaluation metrics for the iterative refinement DA pipeline.

All functions operate on PyTorch tensors and work on GPU.

Conventions:
  - Spatial grid indices run 0..N-1 on a periodic [0, 2π] domain.
  - du/dx is computed spectrally via rfft (exact for band-limited fields).
  - "shock" is defined as a location where |du/dx| exceeds a high quantile.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Scalar RMSE averaged over all elements.

    Args:
        pred:  (..., N)
        truth: (..., N)

    Returns:
        Scalar tensor: sqrt(mean((pred - truth)^2))
    """
    return (pred - truth).pow(2).mean().sqrt()


def rmse_over_time(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """RMSE at each time step, averaged over the spatial dimension.

    Args:
        pred:  (T, N)
        truth: (T, N)

    Returns:
        (T,) — one RMSE value per time step
    """
    assert pred.shape == truth.shape, f"Shape mismatch: {pred.shape} vs {truth.shape}"
    assert pred.dim() == 2, "Expected (T, N) input"
    return (pred - truth).pow(2).mean(dim=-1).sqrt()   # (T,)


# ---------------------------------------------------------------------------
# Energy spectrum
# ---------------------------------------------------------------------------

def energy_spectrum(u: torch.Tensor) -> torch.Tensor:
    """Energy spectrum E(k) = |û_k|^2, averaged over all leading dimensions.

    For a single field (N,) returns the per-mode energy.
    For (T, N) returns the time-averaged spectrum.
    For (B, T, N) returns the batch-and-time-averaged spectrum.

    Args:
        u: (..., N)

    Returns:
        (N//2 + 1,) — energy per Fourier mode, averaged over leading dims
    """
    u_hat = torch.fft.rfft(u, norm="forward")      # (..., N//2+1)  complex
    E     = u_hat.abs().pow(2)                      # (..., N//2+1)  real

    # Average over all leading dimensions
    n_leading = E.dim() - 1
    for _ in range(n_leading):
        E = E.mean(dim=0)

    return E   # (N//2+1,)


# ---------------------------------------------------------------------------
# Shock detection and position error
# ---------------------------------------------------------------------------

def _spectral_gradient(u: torch.Tensor) -> torch.Tensor:
    """du/dx computed spectrally via rfft, on a periodic [0, 2π] domain.

    Args:
        u: (..., N)

    Returns:
        du/dx, same shape as u
    """
    N     = u.shape[-1]
    u_hat = torch.fft.rfft(u)                       # (..., N//2+1) complex
    # Wavenumbers: k = 0, 1, ..., N/2  (physical, not angular-frequency)
    k     = torch.fft.rfftfreq(N, d=1.0 / N).to(u.device)  # (N//2+1,)
    ik    = 1j * k
    # Multiply and transform back
    du_hat = u_hat * ik
    return torch.fft.irfft(du_hat, n=N)             # (..., N)


def shock_positions(
    u: torch.Tensor,
    threshold_quantile: float = 0.95,
) -> list[torch.Tensor]:
    """Detect shock locations where |du/dx| exceeds a high quantile.

    Args:
        u: Field(s), shape (N,) or (T, N).
           If (T, N), returns a list of T tensors.
        threshold_quantile: Quantile of |du/dx| used as the detection threshold.

    Returns:
        List of 1-D tensors, each containing the grid-index positions of
        detected shocks.  One tensor per leading dimension (or a single
        tensor for a 1-D input).
    """
    scalar_input = u.dim() == 1
    if scalar_input:
        u = u.unsqueeze(0)   # (1, N)

    du = _spectral_gradient(u).abs()   # (T, N)

    positions: list[torch.Tensor] = []
    for t in range(u.shape[0]):
        row  = du[t]
        thresh = torch.quantile(row, threshold_quantile)
        idx    = torch.where(row >= thresh)[0]
        positions.append(idx)

    return positions if not scalar_input else positions   # always a list


def shock_position_error(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Mean minimum-distance shock position error over all time steps.

    For each time step, detects shocks in both pred and truth, then matches
    each predicted shock to its nearest truth shock (greedy nearest-neighbour)
    and accumulates the absolute index distance.  Unmatched shocks are
    penalised by N (one full period).

    Args:
        pred:  (T, N)
        truth: (T, N)

    Returns:
        Scalar mean shock position error in grid-index units.
    """
    assert pred.shape == truth.shape, f"Shape mismatch: {pred.shape} vs {truth.shape}"
    T, N = pred.shape

    pred_shocks  = shock_positions(pred)
    truth_shocks = shock_positions(truth)

    total_error = 0.0
    total_count = 0

    for t in range(T):
        ps = pred_shocks[t].float()    # predicted shock indices
        ts = truth_shocks[t].float()   # truth shock indices

        if ps.numel() == 0 and ts.numel() == 0:
            continue
        if ps.numel() == 0 or ts.numel() == 0:
            # One set empty: penalise by N per unmatched shock
            total_error += N * max(ps.numel(), ts.numel())
            total_count += max(ps.numel(), ts.numel())
            continue

        # Pairwise distances (periodic domain)
        # |i - j| vs N - |i - j|  -> take min
        diff = (ps.unsqueeze(1) - ts.unsqueeze(0)).abs()   # (n_pred, n_truth)
        dist = torch.minimum(diff, N - diff)                # wrap-around distance

        # Each predicted shock -> nearest truth shock
        min_dist, _ = dist.min(dim=1)                       # (n_pred,)
        total_error += min_dist.sum().item()
        total_count += ps.numel()

    if total_count == 0:
        return torch.tensor(0.0, device=pred.device)
    return torch.tensor(total_error / total_count, device=pred.device)


# ---------------------------------------------------------------------------
# Temporal consistency
# ---------------------------------------------------------------------------

def temporal_consistency(u_sequence: torch.Tensor) -> torch.Tensor:
    """Frame-to-frame L2 displacement of a time sequence.

    Args:
        u_sequence: (T, N)

    Returns:
        (T-1,) — ||u_t - u_{t-1}||_2 for t = 1 .. T-1
    """
    assert u_sequence.dim() == 2, "Expected (T, N)"
    diff = u_sequence[1:] - u_sequence[:-1]          # (T-1, N)
    return diff.pow(2).sum(dim=-1).sqrt()             # (T-1,)


# ---------------------------------------------------------------------------
# Spectral RMSE
# ---------------------------------------------------------------------------

def spectral_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """RMSE per Fourier mode, averaged over leading dimensions.

    Args:
        pred:  (..., N)
        truth: (..., N)

    Returns:
        (N//2 + 1,) — per-mode RMSE
    """
    assert pred.shape == truth.shape
    err_hat  = torch.fft.rfft(pred - truth, norm="forward")    # (..., N//2+1)
    sq_err   = err_hat.abs().pow(2)                             # (..., N//2+1)
    # Average over leading dims
    n_leading = sq_err.dim() - 1
    for _ in range(n_leading):
        sq_err = sq_err.mean(dim=0)
    return sq_err.sqrt()    # (N//2+1,)


# ---------------------------------------------------------------------------
# CRPS (for ensemble forecasts)
# ---------------------------------------------------------------------------

def crps(ensemble: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Continuous Ranked Probability Score (energy form).

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent draws from the ensemble.

    This is computed spatially per grid point and then averaged.

    Args:
        ensemble: (B, M, N) — B samples, M ensemble members, N grid points
        truth:    (B, N)

    Returns:
        Scalar mean CRPS
    """
    assert ensemble.dim() == 3, "Expected (B, M, N)"
    assert truth.dim() == 2,    "Expected (B, N)"
    B, M, N = ensemble.shape

    # E[|X - y|]: mean over members of |member - truth| (B, N)
    y_exp = truth.unsqueeze(1)                              # (B, 1, N)
    term1 = (ensemble - y_exp).abs().mean(dim=1)            # (B, N)

    # E[|X - X'|]: mean over all pairs; use the identity
    #   E[|X-X'|] = 2 * sum_i sum_{j>i} |x_i - x_j| / M^2
    # Efficient: E[|X-X'|] = 2*(M * E[|X|_pairs]) via broadcasting
    xi = ensemble.unsqueeze(2)   # (B, M, 1, N)
    xj = ensemble.unsqueeze(1)   # (B, 1, M, N)
    term2 = (xi - xj).abs().mean(dim=(1, 2))               # (B, N)

    crps_field = term1 - 0.5 * term2                        # (B, N)
    return crps_field.mean()


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: torch.Tensor,
    truth: torch.Tensor,
    fno_pred: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all scalar metrics for a set of predictions.

    Args:
        pred:     Diffusion posterior, shape (T, N)
        truth:    Ground truth,        shape (T, N)
        fno_pred: FNO-only forecast,   shape (T, N) — optional

    Returns:
        Dict of metric name -> scalar tensor
    """
    assert pred.shape == truth.shape

    metrics: dict[str, torch.Tensor] = {
        "rmse":                  rmse(pred, truth),
        "shock_position_error":  shock_position_error(pred, truth),
        "temporal_consistency":  temporal_consistency(pred).mean(),
    }

    if fno_pred is not None:
        assert fno_pred.shape == truth.shape
        metrics["fno_rmse"]                 = rmse(fno_pred, truth)
        metrics["fno_shock_position_error"] = shock_position_error(fno_pred, truth)
        metrics["rmse_improvement"]         = metrics["fno_rmse"] - metrics["rmse"]

    return metrics
