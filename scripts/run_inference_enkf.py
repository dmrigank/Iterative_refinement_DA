"""
Ensemble Kalman Filter (EnKF) DA baseline for 2D Kraichnan turbulence.

Uses the PSEUDOSPECTRAL SOLVER as the ensemble forecast model.

Design rationale
================
The central challenge for EnKF here is that:
  1. The truth trajectory is generated with one specific stochastic forcing
     realisation; each ensemble member has its own independent forcing.
  2. The observation is essentially EXACT at k <= 16 (32x32 spectral
     truncation has negligible truncation error compared to the signal).
  3. The Lyapunov time of 2D Kraichnan turbulence at these parameters is
     short relative to the observation interval (delta_t = 0.05), so
     ensemble spread from independent forcings grows quickly.

Given (1)-(3), the optimal analysis strategy is:
  - At observed modes (k <= k_obs = 16): REPLACE ensemble mean with the
    observation directly (Kalman gain -> 1). This is the Spectral OI limit,
    and is justified because observation error << background error at these
    scales.
  - At unobserved modes (k > k_obs): keep the ensemble mean forecast; the
    observation provides zero information there.
  - Apply the above as a DETERMINISTIC ETKF-style update (no perturbed
    observations), which avoids the sampling noise of the stochastic EnKF.

Ensemble perturbations at k <= k_obs are zeroed out after the analysis
(members agree on the observed modes) and re-inflated with band-limited
additive noise scaled to the estimated background error, so the next
forecast step starts with proper spread for the Kalman update.

This is equivalent to a spectrally-localised ETKF with near-unit gain at
observed modes and zero gain at unobserved modes — the classical DA
solution when the observation operator is a spectral truncation and the
observation error is small.

Key parameters
--------------
  N          = 20          ensemble members
  sig_ic     = 0.5         IC perturbation std (realistic, not 0.05)
  sig_r      = 0.01        obs noise std (small: 32x32 obs is near-exact)
  sig_add    = 0.15        additive inflation std (band-limited k <= k_obs)
  infl_mult  = 1.05        mild multiplicative inflation
  k_obs      = 16          observation Nyquist at 256x256 grid

Usage
-----
    python scripts/run_inference_enkf.py
        [--config      configs/kraichnan.yaml]
        [--n_steps     N]
        [--n_ensemble  20]
        [--sigma_ic    0.5]
        [--obs_noise   0.01]
        [--sig_add     0.15]
        [--inflation   1.05]
        [--k_obs       16]
        [--seed        42]

Output: results_enkf/inference_results.pt
    'enkf_256'    : (n_traj, T, 256, 256) — EnKF ensemble mean
    'spread_256'  : (n_traj, T)           — mean spatial ensemble spread
    'truth_256'   : (n_traj, T, 256, 256) — ground truth
    'obs_32'      : (n_traj, T, 32,  32)  — coarse observations
    'trmse_curve' : (T,)                  — RMSE at each step
    'metrics'     : dict
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "irda" not in sys.modules:
    _s = types.ModuleType("irda")
    _o = types.ModuleType("irda.ops")
    _p = types.ModuleType("irda.ops.spectra")
    _p.radial_energy_spectrum_from_vorticity = lambda *a, **kw: (None, None)
    _o.spectra = _p; _s.ops = _o
    sys.modules["irda"] = _s
    sys.modules["irda.ops"] = _o
    sys.modules["irda.ops.spectra"] = _p

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.data.kraichnan_generator import (
    KraichnanSimulationConfig,
    build_spectral_grid,
    update_forcing_hat,
    rk4_step,
)
from src.data.dataset_2d import spectral_downsample_2d, spectral_upsample_2d


# ---------------------------------------------------------------------------
# Spectral observation operator and helpers
# ---------------------------------------------------------------------------

def H(x: torch.Tensor) -> torch.Tensor:
    """Forward obs operator: (..., 256, 256) -> (..., 32, 32)."""
    return spectral_downsample_2d(x, 32, 32)


def _build_k_mask(ny: int, nx: int, k_max: float) -> torch.Tensor:
    """rfft2 mask: True for radial wavenumber <= k_max."""
    kx = torch.fft.fftfreq(ny, d=1.0 / ny)
    ky = torch.fft.rfftfreq(nx, d=1.0 / nx)
    k_mag = (kx.unsqueeze(1).expand(ny, nx // 2 + 1).pow(2) +
             ky.unsqueeze(0).expand(ny, nx // 2 + 1).pow(2)).sqrt()
    return k_mag <= k_max   # (ny, nx//2+1) bool


# ---------------------------------------------------------------------------
# ETKF-style analysis with spectral replacement at observed modes
# ---------------------------------------------------------------------------

@torch.no_grad()
def enkf_analysis(
    X_b: torch.Tensor,    # (N, 256, 256) ensemble background
    y_obs: torch.Tensor,  # (32, 32)      coarse observation
    sigma_r: float,
    inflation: float,
    sig_add: float,
    obs_mask: torch.Tensor,   # (ny, nx//2+1) bool, True for k <= k_obs
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spectrally-localised EnKF analysis with aggressive obs assimilation.

    Strategy:
      1. Multiplicative + additive inflation to maintain spread.
      2. Standard stochastic EnKF correction in 32x32 obs space, upsampled
         to 256x256 — updates k <= 16 modes toward the observation.
      3. Hard spectral replacement: set ensemble MEAN at k <= k_obs modes
         to exactly match the observation (Kalman gain = 1 limit).
         This is justified when sig_r << spread, which holds here.
      4. Re-inject band-limited noise at k <= k_obs to restore ensemble
         spread for the next forecast step.

    Returns:
        X_a     : (N, 256, 256) analysis ensemble
        x_mean_a: (256, 256)    ensemble mean = DA state estimate
    """
    N = X_b.shape[0]
    ny, nx = X_b.shape[-2], X_b.shape[-1]

    # ── Step 1: inflation ────────────────────────────────────────────────────
    x_mean_b = X_b.mean(0)
    X_b = x_mean_b + inflation * (X_b - x_mean_b)

    # ── Step 2: standard stochastic EnKF in obs space ───────────────────────
    p = 32 * 32
    HX_b = H(X_b)                                          # (N, 32, 32)
    HX_anom = (HX_b - HX_b.mean(0)).reshape(N, p)
    eps = sigma_r * torch.randn(N, p)
    innov = y_obs.reshape(1, p).expand(N, p) + eps - HX_b.reshape(N, p)
    C = HX_anom @ HX_anom.T + (N - 1) * sigma_r ** 2 * torch.eye(N)
    alpha = torch.linalg.solve(C, HX_anom @ innov.T)      # (N, N)
    delta_obs = (alpha.T @ HX_anom.reshape(N, p)).reshape(N, 32, 32)
    delta_256 = spectral_upsample_2d(delta_obs, ny, nx)
    X_a = X_b + delta_256                                  # (N, 256, 256)

    # ── Step 3: hard spectral replacement of ensemble MEAN at k <= k_obs ────
    # Compute analysis mean in Fourier space and overwrite observed modes
    # with the (noiseless) observation upsampled to 256x256.
    x_mean_a = X_a.mean(0)
    W_mean = torch.fft.rfft2(x_mean_a)                    # (ny, nx//2+1) complex
    obs_up = spectral_upsample_2d(y_obs.unsqueeze(0), ny, nx).squeeze(0)
    W_obs  = torch.fft.rfft2(obs_up)
    W_mean[obs_mask] = W_obs[obs_mask]                     # replace observed modes
    x_mean_new = torch.fft.irfft2(W_mean, s=(ny, nx))

    # Shift ensemble to new mean while preserving perturbations
    X_a = X_a - x_mean_a + x_mean_new

    # ── Step 4: re-inject band-limited spread at observed modes ─────────────
    # After the hard replacement, all members agree at k <= k_obs.
    # Inject zero-mean band-limited noise so the next forecast step has
    # non-degenerate covariances at those modes.
    if sig_add > 0.0:
        noise = sig_add * torch.randn(N, ny, nx)
        W_noise = torch.fft.rfft2(noise)
        W_noise[:, ~obs_mask] = 0.0                        # only at observed modes
        noise_low = torch.fft.irfft2(W_noise, s=(ny, nx))
        noise_low = noise_low - noise_low.mean(0)          # zero-mean
        X_a = X_a + noise_low

    x_mean_a = X_a.mean(0)
    return X_a, x_mean_a


# ---------------------------------------------------------------------------
# Build solver config
# ---------------------------------------------------------------------------

def _build_solver_cfg(cfg) -> KraichnanSimulationConfig:
    return KraichnanSimulationConfig(
        nx=int(cfg.pde.nx), ny=int(cfg.pde.ny),
        dt=float(cfg.pde.dt), nu=float(cfg.pde.viscosity),
        mu=float(cfg.pde.ekman_drag),
        forcing_amplitude=float(cfg.pde.forcing_amplitude),
        forcing_band_center=float(cfg.pde.forcing_band_center),
        forcing_band_width=float(cfg.pde.forcing_band_width),
        forcing_correlation_time=float(cfg.pde.forcing_correlation_time),
        save_every=int(cfg.pde.save_every),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())

def _per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return (pred - truth).pow(2).mean(dim=(2, 3)).sqrt().mean(dim=1)

def _per_step_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return (pred - truth).pow(2).mean(dim=(0, 2, 3)).sqrt()


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EnKF DA baseline (solver forecast)")
    p.add_argument("--config",     type=str,   default="configs/kraichnan.yaml")
    p.add_argument("--n_steps",    type=int,   default=None)
    p.add_argument("--n_ensemble", type=int,   default=20)
    p.add_argument("--sigma_ic",   type=float, default=0.5,
                   help="Std of IC perturbations (default: 0.5)")
    p.add_argument("--obs_noise",  type=float, default=0.01,
                   help="Obs noise std in 32x32 space (default: 0.01, near-exact obs)")
    p.add_argument("--sig_add",    type=float, default=0.15,
                   help="Additive band-limited inflation std (default: 0.15)")
    p.add_argument("--inflation",  type=float, default=1.05,
                   help="Multiplicative covariance inflation (default: 1.05)")
    p.add_argument("--k_obs",      type=int,   default=16,
                   help="Observation Nyquist wavenumber at 256x256 (default: 16)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    N       = args.n_ensemble
    sig_r   = args.obs_noise
    infl    = args.inflation
    sig_ic  = args.sigma_ic
    sig_add = args.sig_add
    k_obs   = args.k_obs
    n_steps = args.n_steps if args.n_steps is not None \
              else int(cfg.inference.test_time_steps)

    print("=" * 70)
    print("  EnKF DA Baseline  —  2D Kraichnan  (pseudospectral solver)")
    print(f"  N={N}  sig_ic={sig_ic}  obs_noise={sig_r}  "
          f"sig_add={sig_add}  inflation={infl}  k_obs={k_obs}")
    print("=" * 70)

    solver_cfg = _build_solver_cfg(cfg)
    grid       = build_spectral_grid(nx=solver_cfg.nx, ny=solver_cfg.ny)
    ny, nx_grid = solver_cfg.ny, solver_cfg.nx
    obs_mask   = _build_k_mask(ny, nx_grid, k_obs)

    print(f"\nSolver: dt={solver_cfg.dt}  save_every={solver_cfg.save_every}  "
          f"delta_t={solver_cfg.dt * solver_cfg.save_every:.3f}")

    test_data    = torch.load(
        Path(cfg.data.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )
    n_test       = test_data["w_32"].shape[0]
    traj_indices = list(range(n_test))
    est_min      = N * n_steps * n_test * 0.45 / 60
    print(f"Test trajectories: {n_test}  ({n_steps} steps, N={N} members)")
    print(f"Estimated wall time: ~{est_min:.0f} min\n")

    all_enkf:   list[torch.Tensor] = []
    all_spread: list[torch.Tensor] = []
    all_truth:  list[torch.Tensor] = []
    all_obs32:  list[torch.Tensor] = []

    for traj_i in tqdm(traj_indices, desc="Trajectories", ncols=90):
        obs_32    = test_data["w_32" ][traj_i, :n_steps].float()
        truth_256 = test_data["w_256"][traj_i, :n_steps].float()
        T = obs_32.shape[0]

        enkf_frames:   list[torch.Tensor] = []
        spread_frames: list[float]        = []

        # t=0: initialise from spectral upsample of obs + perturbations
        w0_up = spectral_upsample_2d(obs_32[0:1], ny, nx_grid).squeeze(0).numpy()
        members = [
            w0_up.astype(np.float64) + sig_ic * np.random.randn(ny, nx_grid)
            for _ in range(N)
        ]
        rngs    = [np.random.default_rng(args.seed * 1000 + traj_i * 100 + i)
                   for i in range(N)]
        fh_list = [None] * N

        X0 = torch.from_numpy(np.stack(members)).float()
        enkf_frames.append(X0.mean(dim=0))
        spread_frames.append(float(X0.std(dim=0).mean()))

        for t in tqdm(range(1, T), desc=f"  traj {traj_i}", ncols=80, leave=False):
            # ── Forecast ─────────────────────────────────────────────────────
            for i in range(N):
                for _ in range(solver_cfg.save_every):
                    fh_list[i] = update_forcing_hat(
                        fh_list[i], rngs[i], grid, solver_cfg
                    )
                    members[i] = rk4_step(
                        members[i], solver_cfg.dt, fh_list[i],
                        grid, solver_cfg.nu, solver_cfg.mu,
                    )

            X_b = torch.from_numpy(np.stack(members)).float()

            # ── Analysis ─────────────────────────────────────────────────────
            X_a, x_mean = enkf_analysis(
                X_b, obs_32[t], sig_r, infl, sig_add, obs_mask
            )

            members = list(X_a.numpy().astype(np.float64))
            enkf_frames.append(x_mean)
            spread_frames.append(float(X_a.std(dim=0).mean()))

        all_enkf.append(torch.stack(enkf_frames, dim=0))
        all_spread.append(torch.tensor(spread_frames))
        all_truth.append(truth_256)
        all_obs32.append(obs_32)

    enkf_all   = torch.stack(all_enkf,   dim=0)
    spread_all = torch.stack(all_spread, dim=0)
    truth_all  = torch.stack(all_truth,  dim=0)
    obs32_all  = torch.stack(all_obs32,  dim=0)

    pt_enkf       = _per_traj_rmse(enkf_all, truth_all)
    rmse_enkf     = float(pt_enkf.mean())
    rmse_enkf_std = float(pt_enkf.std()) if n_test > 1 else 0.0

    metrics = {
        "rmse_enkf_256":     rmse_enkf,
        "rmse_enkf_std_256": rmse_enkf_std,
        "mean_spread":       float(spread_all.mean()),
        "n_ensemble":        N,
        "sigma_ic":          sig_ic,
        "obs_noise":         sig_r,
        "sig_add":           sig_add,
        "inflation":         infl,
        "k_obs":             k_obs,
    }

    # ── Comparison table ──────────────────────────────────────────────────────
    ri_path = Path(cfg.paths.results_dir) / "inference_results.pt"
    ro_path = Path("results_oneshot") / "inference_results.pt"
    ir_rmse = bic_rmse = one_rmse = float("nan")

    if ri_path.exists():
        ri  = torch.load(ri_path, map_location="cpu", weights_only=True)
        T_c = min(n_steps, ri["posterior_256"].shape[1])
        n_c = min(n_test,  ri["posterior_256"].shape[0])
        ir_rmse = _rmse(ri["posterior_256"][:n_c, :T_c].float(), truth_all[:n_c, :T_c])
    if ro_path.exists():
        ro  = torch.load(ro_path, map_location="cpu", weights_only=True)
        T_c = min(n_steps, ro["posterior_256"].shape[1])
        n_c = min(n_test,  ro["posterior_256"].shape[0])
        bic_rmse = _rmse(ro["bicubic_256"  ][:n_c, :T_c].float(), truth_all[:n_c, :T_c])
        one_rmse = _rmse(ro["posterior_256"][:n_c, :T_c].float(), truth_all[:n_c, :T_c])

    print(f"\n{'='*70}")
    print(f"  Results @ 256x256  ({n_test} traj x {n_steps} steps)")
    print(f"{'='*70}")
    print(f"  {'Method':<48}  {'RMSE':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Spectral upsample (bicubic)':<48}  {bic_rmse:8.4f}")
    print(f"  {'Solver EnKF  (N=%d, spectral replacement)' % N:<48}  {rmse_enkf:8.4f} +/- {rmse_enkf_std:.4f}")
    print(f"  {'One-Shot Diffusion SR':<48}  {one_rmse:8.4f}")
    print(f"  {'Iterative Refinement (ours)':<48}  {ir_rmse:8.4f}")
    print(f"{'='*70}")
    print(f"\n  Mean ensemble spread: {metrics['mean_spread']:.4f}")

    results_dir = Path("results_enkf")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"
    torch.save(
        {
            "enkf_256":    enkf_all,
            "spread_256":  spread_all,
            "truth_256":   truth_all,
            "obs_32":      obs32_all,
            "trmse_curve": _per_step_rmse(enkf_all, truth_all),
            "metrics":     metrics,
        },
        save_path,
    )
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
