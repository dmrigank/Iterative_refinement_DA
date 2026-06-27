"""
Learned EnKF DA baseline for 2D Kraichnan turbulence.

Uses the trained FNO2d (256x256) as the ensemble forecast model instead of
the pseudospectral solver. Since the FNO is deterministic, ensemble spread
is generated entirely via band-limited additive inflation injected at the
analysis step — equivalent to an implicit model-error covariance.

Design rationale
================
The solver EnKF has natural spread from independent stochastic forcings.
The FNO is a deterministic map: N identical inputs → N identical outputs.
The only way to maintain ensemble diversity is to inject perturbations.
We use the same strategy as the solver EnKF analysis (additive band-limited
noise + spectral replacement), but sig_add must be larger to compensate for
the absence of any dynamical spread source.

Calibration: FNO forecast error ≈ 0.24 (256x256), ≈ 0.19 in obs space.
We set sig_add so that ensemble spread in obs space ≈ forecast error, i.e.
the Kalman gain is in a sensible regime (not collapsed, not overblown).

Analysis step (same as solver EnKF):
  1. Multiplicative inflation
  2. Standard stochastic EnKF correction in 32x32 obs space
  3. Hard spectral replacement of ensemble mean at k <= k_obs with observation
  4. Re-inject band-limited noise at k <= k_obs to restore spread

Key parameters
--------------
  N          = 20          ensemble members
  sig_ic     = 0.5         IC perturbation std
  sig_r      = 0.01        obs noise std (near-exact 32x32 obs)
  sig_add    = 0.20        additive inflation (larger than solver EnKF=0.15
                           since FNO has no dynamical spread source)
  infl_mult  = 1.05        mild multiplicative inflation
  k_obs      = 16          observation Nyquist at 256x256 grid

Usage
-----
    python scripts/run_inference_enkf_fno.py
        [--config      configs/kraichnan.yaml]
        [--n_steps     N]
        [--n_ensemble  20]
        [--sigma_ic    0.5]
        [--obs_noise   0.01]
        [--sig_add     0.20]
        [--inflation   1.05]
        [--k_obs       16]
        [--seed        42]

Output: results_enkf_fno/inference_results.pt
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
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.fno_2d import FNO2d
from src.data.dataset_2d import spectral_downsample_2d, spectral_upsample_2d
from scripts.run_inference_enkf import enkf_analysis, _build_k_mask


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
    p = argparse.ArgumentParser(description="Learned EnKF DA baseline (FNO forecast)")
    p.add_argument("--config",     type=str,   default="configs/kraichnan.yaml")
    p.add_argument("--n_steps",    type=int,   default=None)
    p.add_argument("--n_ensemble", type=int,   default=20)
    p.add_argument("--sigma_ic",   type=float, default=0.5)
    p.add_argument("--obs_noise",  type=float, default=0.01)
    p.add_argument("--sig_add",    type=float, default=0.20,
                   help="Additive band-limited inflation std (default: 0.20). "
                        "Larger than solver EnKF since FNO has no dynamical spread.")
    p.add_argument("--inflation",  type=float, default=1.05)
    p.add_argument("--k_obs",      type=int,   default=16)
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

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N       = args.n_ensemble
    sig_r   = args.obs_noise
    infl    = args.inflation
    sig_ic  = args.sigma_ic
    sig_add = args.sig_add
    k_obs   = args.k_obs
    n_steps = args.n_steps if args.n_steps is not None \
              else int(cfg.inference.test_time_steps)

    print("=" * 70)
    print("  Learned EnKF DA Baseline  —  2D Kraichnan  (FNO forecast)")
    print(f"  N={N}  sig_ic={sig_ic}  obs_noise={sig_r}  "
          f"sig_add={sig_add}  inflation={infl}  k_obs={k_obs}")
    print("=" * 70)

    # ── Load FNO (256x256 only — single-stage forecast) ──────────────────────
    print("\nLoading FNO2d (256×256)...")
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    fno = FNO2d(cfg, resolution=256).to(device)

    # Patch type hint (FNO2d references ConditionalUNet1d in diffusion import)
    import src.models.diffusion as _dm
    _dm.ConditionalUNet1d = nn.Module

    ckpt = torch.load(ckpt_dir / "fno_256.pt", map_location=device, weights_only=True)
    fno.load_state_dict(ckpt["model"])
    fno.eval()
    print(f"  Loaded FNO2d 256×256  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4e})")

    obs_mask = _build_k_mask(256, 256, k_obs).to(device)

    test_data = torch.load(
        Path(cfg.data.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )
    n_test       = test_data["w_32"].shape[0]
    traj_indices = list(range(n_test))
    print(f"\nTest trajectories: {n_test}  ({n_steps} steps, N={N} members)")

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

        # t=0: initialise ensemble from spectral upsample of first obs
        w0_up = spectral_upsample_2d(obs_32[0:1], 256, 256).squeeze(0)  # (256,256)
        # perturb on CPU then move to device per-batch during forecast
        X = w0_up.unsqueeze(0).expand(N, -1, -1).clone()                # (N,256,256)
        X = X + sig_ic * torch.randn_like(X)

        enkf_frames.append(X.mean(0).cpu())
        spread_frames.append(float(X.std(0).mean()))

        for t in tqdm(range(1, T), desc=f"  traj {traj_i}", ncols=80, leave=False):
            # ── Forecast: all N members through FNO in one batch ─────────────
            X_dev = X.to(device)
            with torch.no_grad():
                X_fc = fno(X_dev.unsqueeze(1)).squeeze(1)   # (N,256,256)

            X_b = X_fc.cpu()

            # ── Analysis ─────────────────────────────────────────────────────
            X_a, x_mean = enkf_analysis(
                X_b, obs_32[t], sig_r, infl, sig_add,
                obs_mask.cpu()
            )

            X = X_a
            enkf_frames.append(x_mean.cpu())
            spread_frames.append(float(X_a.std(0).mean()))

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
        "rmse_enkf_fno_256":     rmse_enkf,
        "rmse_enkf_fno_std_256": rmse_enkf_std,
        "mean_spread":           float(spread_all.mean()),
        "n_ensemble":            N,
        "sigma_ic":              sig_ic,
        "obs_noise":             sig_r,
        "sig_add":               sig_add,
        "inflation":             infl,
        "k_obs":                 k_obs,
    }

    # ── Comparison table ──────────────────────────────────────────────────────
    ri_path  = Path(cfg.paths.results_dir) / "inference_results.pt"
    re_path  = Path("results_enkf") / "inference_results.pt"
    ir_rmse  = solver_rmse = bic_rmse = float("nan")

    if ri_path.exists():
        ri = torch.load(ri_path, map_location="cpu", weights_only=True)
        T_c = min(n_steps, ri["posterior_256"].shape[1])
        n_c = min(n_test,  ri["posterior_256"].shape[0])
        ir_rmse = _rmse(ri["posterior_256"][:n_c, :T_c].float(), truth_all[:n_c, :T_c])
    if re_path.exists():
        re = torch.load(re_path, map_location="cpu", weights_only=True)
        T_c = min(n_steps, re["enkf_256"].shape[1])
        n_c = min(n_test,  re["enkf_256"].shape[0])
        solver_rmse = _rmse(re["enkf_256"][:n_c, :T_c].float(), truth_all[:n_c, :T_c])

    # bicubic from obs
    flat_obs = obs32_all.reshape(-1, 1, 32, 32).float()
    bic_256  = torch.nn.functional.interpolate(flat_obs, size=(256, 256),
                                               mode="bicubic", align_corners=False)
    bic_256  = bic_256.squeeze(1).reshape(n_test, n_steps, 256, 256)
    bic_rmse = _rmse(bic_256, truth_all)

    print(f"\n{'='*70}")
    print(f"  Results @ 256×256  ({n_test} traj × {n_steps} steps)")
    print(f"{'='*70}")
    print(f"  {'Method':<48}  {'RMSE':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Bicubic':<48}  {bic_rmse:8.4f}")
    print(f"  {'Learned EnKF  (FNO forecast, N=%d)' % N:<48}  {rmse_enkf:8.4f} ± {rmse_enkf_std:.4f}")
    print(f"  {'Solver EnKF  (pseudospectral, N=%d)' % N:<48}  {solver_rmse:8.4f}")
    print(f"  {'Iterative Refinement (ours)':<48}  {ir_rmse:8.4f}")
    print(f"{'='*70}")
    print(f"\n  Mean ensemble spread: {metrics['mean_spread']:.4f}")

    results_dir = Path("results_enkf_fno")
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
