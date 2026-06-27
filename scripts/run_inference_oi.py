"""
Optimal Interpolation (OI) DA baseline for 2D Kraichnan turbulence.

Implements spectral OI as the classical linear DA baseline to compare against
the iterative diffusion refinement approach.  The OI analysis blends the FNO
forecast (background) with the coarse 32x32 observation using mode-dependent
Kalman gains estimated from empirical error statistics.

Method
------
The OI correction is applied to the SAME FNO forecasts that the iterative
refinement pipeline already produced (loaded from results_2d/).  This is the
correct apples-to-apples comparison: it answers "if you replace the diffusion
corrector with a linear OI step, but keep the same FNO prior, how does
performance compare?"

Running OI autoregressively (OI analysis feeds back into next FNO call) is
WRONG for this comparison because:
  1. The OI analysis is spectrally truncated at k=16 (no energy at k>16 from
     the observation side, only from the forecast).  The FNO sees an unphysical
     spectral discontinuity and produces worse forecasts than when fed the
     smooth full-spectrum IR posterior.
  2. This error compounds: worse OI analysis → worse FNO forecast → worse OI
     analysis → ..., making the OI pipeline degrade to 0.45 RMSE vs IR's 0.21.

By reusing the IR forecasts, both methods receive the same FNO prior at every
timestep, and only the correction step differs (linear OI vs nonlinear diffusion).

In spectral space:
    w_a(k) = (1 - K(k)) * w_b(k) + K(k) * w_obs(k)

where K(k) = σ_b²(k) / (σ_b²(k) + σ_r²(k)) is estimated per radial shell
from the background error (FNO forecast - truth) and observation error
(spectral_upsample(obs_32) - truth) on the test data.

  For k ≤ 16 (within 32x32 Nyquist): gain ≈ 1 → analysis ≈ observation
  For k > 16 (beyond obs resolution):  gain = 0 → analysis = forecast

Usage
-----
    python scripts/run_inference_oi.py [--config configs/kraichnan.yaml]
                                       [--ir_results results_2d/inference_results.pt]
                                       [--gain_source {empirical,unit}]

    --gain_source empirical  (default) estimates K(k) from the IR forecast
                             and observation errors on the test set
    --gain_source unit       sets K(k)=1 for k<=16, K(k)=0 for k>16

Output: results_oi/inference_results.pt
    'oi_256'       : (n_traj, T, 256, 256) — OI analysis at 256x256
    'forecast_256' : (n_traj, T, 256, 256) — FNO forecast (same as IR)
    'truth_256'    : (n_traj, T, 256, 256) — ground truth
    'obs_32'       : (n_traj, T, 32, 32)   — coarse observations
    'gains'        : (129,)                 — per-shell Kalman gains K(k)
    'metrics'      : dict of metric_name -> float
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.fno_2d import FNO2d
from src.models.unet_2d import ConditionalUNet2d
from src.models.diffusion import GaussianDiffusion
from src.data.dataset_2d import spectral_upsample_2d
from src.evaluation.metrics_2d import radial_energy_spectrum


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fnos(cfg, device: torch.device) -> dict[int, FNO2d]:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    fnos: dict[int, FNO2d] = {}
    for res in [64, 128, 256]:
        model = FNO2d(cfg, res).to(device)
        ckpt  = torch.load(ckpt_dir / f"fno_{res}.pt",
                           map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        fnos[res] = model
        print(f"  Loaded FNO2d {res}x{res}  (epoch {ckpt['epoch']}, "
              f"val_loss={ckpt['val_loss']:.4e})")
    return fnos


# ---------------------------------------------------------------------------
# Kalman gain estimation
# ---------------------------------------------------------------------------

def estimate_gains(
    cfg,
    fnos: dict[int, FNO2d],
    device: torch.device,
    gain_source: str = "empirical",
) -> torch.Tensor:
    """Estimate per-shell OI Kalman gains K(k) from the training set.

    Uses teacher-forced FNO forecasts on the training split to estimate
    the background error covariance σ_b²(k), and spectral_upsample(obs_32)
    vs truth to estimate the observation error covariance σ_r²(k).

    Args:
        gain_source: "empirical" (default) or "unit" (binary mask at obs Nyquist)

    Returns:
        gains: (129,) tensor of K(k) for each radial shell k=0..128
    """
    k_max_obs = 16   # 32x32 Nyquist frequency in 256x256 domain

    if gain_source == "unit":
        gains = torch.zeros(129)
        for ki in range(1, 17):   # k=1..16 inclusive
            gains[ki] = 1.0
        print(f"  Gains: unit step at k_max_obs={k_max_obs}  "
              f"(K=1 for k<={k_max_obs}, K=0 beyond)")
        return gains

    # Empirical: load training data and compute error spectra entirely on CPU.
    # The FNO-256 is 67M params so we keep it on CPU here to avoid double-loading
    # onto the GPU that will be needed during inference.
    data_dir = Path(cfg.data.data_dir)
    print("  Estimating gains from training data (CPU) ...")
    raw = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True)

    # Use every 10th frame to keep memory manageable
    # (44 traj × ~20 frames = ~880 frames at 256×256 float32 ≈ 880 MB)
    w32_all  = raw["w_32" ][:, ::10].float()   # (n_traj, T', 32, 32)
    w256_all = raw["w_256"][:, ::10].float()   # (n_traj, T', 256, 256)

    w256_in  = raw["w_256"][:, :-1:10].float()   # input  frames for teacher-forced forecast
    w256_out = raw["w_256"][:,  1::10].float()   # target frames (truth at t+1)
    T2 = min(w256_in.shape[1], w256_out.shape[1])
    w256_in  = w256_in[:, :T2]
    w256_out = w256_out[:, :T2]
    del raw   # free raw dict immediately

    # Run FNO-256 on CPU to avoid consuming GPU memory before inference
    fno256_cpu = fnos[256].cpu().eval()
    bg_errors: list[torch.Tensor] = []
    flat_in  = w256_in.reshape(-1, 256, 256)
    flat_out = w256_out.reshape(-1, 256, 256)
    batch_cpu = 8   # small batch on CPU
    for i in range(0, flat_in.shape[0], batch_cpu):
        x = flat_in[i:i+batch_cpu].unsqueeze(1)   # (B, 1, 256, 256) on CPU
        with torch.no_grad():
            fc = fno256_cpu(x).squeeze(1)          # (B, 256, 256) on CPU
        bg_errors.append(fc - flat_out[i:i+batch_cpu])
    bg_err_flat = torch.cat(bg_errors, dim=0)      # (N, 256, 256) on CPU
    del flat_in, flat_out, w256_in, w256_out, bg_errors

    # Move FNO-256 back to device for inference
    fnos[256] = fnos[256].to(device)

    # Observation error on CPU — compute in small chunks to avoid one huge allocation
    w32_flat  = w32_all.reshape(-1, 32, 32)
    w256_flat = w256_all.reshape(-1, 256, 256)
    del w32_all, w256_all

    chunk = 64   # process this many frames at a time for spectral upsample
    obs_err_chunks: list[torch.Tensor] = []
    for i in range(0, w32_flat.shape[0], chunk):
        up = spectral_upsample_2d(w32_flat[i:i+chunk], 256, 256)
        obs_err_chunks.append(up - w256_flat[i:i+chunk])
    obs_err_flat = torch.cat(obs_err_chunks, dim=0)   # (N, 256, 256) on CPU
    del w32_flat, w256_flat, obs_err_chunks

    # Accumulate radial energy spectra in chunks to avoid one giant rfft2 call
    def _accum_spectrum(err_flat: torch.Tensor, chunk_size: int = 64
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate E(k) by averaging over chunks of frames."""
        E_sum: torch.Tensor | None = None
        k_bins_out: torch.Tensor | None = None
        n_chunks = 0
        for i in range(0, err_flat.shape[0], chunk_size):
            E_c, k_c = radial_energy_spectrum(err_flat[i:i+chunk_size])
            E_sum = E_c if E_sum is None else E_sum + E_c
            k_bins_out = k_c
            n_chunks += 1
        return (E_sum / n_chunks), k_bins_out

    print("  Computing background error spectrum ...")
    E_bg,  k_bins = _accum_spectrum(bg_err_flat)
    del bg_err_flat

    print("  Computing observation error spectrum ...")
    E_obs, _      = _accum_spectrum(obs_err_flat)
    del obs_err_flat

    # Kalman gains: K(k) = σ_b²(k) / (σ_b²(k) + σ_r²(k))
    gains = torch.zeros(len(E_bg))
    for ki in range(1, len(gains)):
        if k_bins[ki].item() <= k_max_obs:
            b = E_bg[ki].clamp(min=1e-30)
            o = E_obs[ki].clamp(min=1e-30)
            gains[ki] = b / (b + o)
        # else: gain stays 0 (obs has no info beyond its Nyquist)

    print(f"  Gains (k=1..18): "
          + "  ".join(f"k{int(k_bins[ki].item())}={gains[ki].item():.3f}"
                      for ki in range(1, min(19, len(gains)))))
    return gains


# ---------------------------------------------------------------------------
# Spectral OI analysis (vectorised over batch)
# ---------------------------------------------------------------------------

def spectral_oi(
    w_forecast: torch.Tensor,   # (..., 256, 256)
    w_obs_up:   torch.Tensor,   # (..., 256, 256)
    gain_map:   torch.Tensor,   # (256, 129) — per-pixel gain in rfft2 space
) -> torch.Tensor:
    """Apply spectral OI analysis.

    w_a(k) = (1 - K(k)) * w_b(k) + K(k) * w_obs(k)

    Args:
        w_forecast: FNO forecast field(s), shape (..., 256, 256)
        w_obs_up:   Spectrally upsampled 32x32 obs, shape (..., 256, 256)
        gain_map:   Per-Fourier-bin Kalman gain, shape (256, 129)

    Returns:
        OI analysis, same shape as inputs
    """
    W_b   = torch.fft.rfft2(w_forecast)    # (..., 256, 129)
    W_obs = torch.fft.rfft2(w_obs_up)      # (..., 256, 129)
    # gain_map is (256, 129); broadcast over any leading batch dims
    gm = gain_map
    while gm.dim() < W_b.dim():
        gm = gm.unsqueeze(0)
    W_a = (1.0 - gm) * W_b + gm * W_obs
    return torch.fft.irfft2(W_a, s=(256, 256))


def build_gain_map(gains: torch.Tensor, ny: int = 256, nx: int = 256) -> torch.Tensor:
    """Build the (ny, nx//2+1) gain map from radial gains.

    Maps each rfft2 output pixel to its radial shell index and looks up the gain.

    Args:
        gains: (n_shells,) per-shell Kalman gains
        ny, nx: spatial dimensions of the full field

    Returns:
        (ny, nx//2+1) float tensor of per-pixel gains
    """
    kx = torch.fft.fftfreq(ny,  d=1.0 / ny)   # (ny,)   integer mode indices
    ky = torch.fft.rfftfreq(nx, d=1.0 / nx)   # (nx//2+1,)
    kx_g = kx.unsqueeze(1).expand(ny, nx // 2 + 1)
    ky_g = ky.unsqueeze(0).expand(ny, nx // 2 + 1)
    k_mag = (kx_g.pow(2) + ky_g.pow(2)).sqrt()
    k_int = k_mag.round().long().clamp(0, len(gains) - 1)
    return gains[k_int]   # (ny, nx//2+1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


def _per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(n_traj, T, H, W) -> (n_traj,)"""
    return (pred - truth).pow(2).mean(dim=(2, 3)).sqrt().mean(dim=1)


def _per_step_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(n_traj, T, H, W) -> (T,)"""
    return (pred - truth).pow(2).mean(dim=(0, 2, 3)).sqrt()


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OI DA baseline for 2D Kraichnan")
    p.add_argument("--config",      type=str, default="configs/kraichnan.yaml")
    p.add_argument("--ir_results",  type=str, default="results_2d/inference_results.pt",
                   help="Path to IR inference results — OI reuses its FNO forecasts")
    p.add_argument("--gain_source", type=str, default="empirical",
                   choices=["empirical", "unit"],
                   help="'empirical': estimate K(k) from IR forecast and obs errors; "
                        "'unit': binary step at obs Nyquist (K=1 for k<=16, 0 beyond)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("=" * 65)
    print("  Spectral OI DA Baseline  (2D Kraichnan)")
    print(f"  gain_source={args.gain_source}")
    print(f"  Reusing FNO forecasts from: {args.ir_results}")
    print("=" * 65)

    # ── Load IR results — reuse its FNO forecasts as the OI background ───────
    print(f"\nLoading IR results from {args.ir_results} ...")
    ri = torch.load(args.ir_results, map_location="cpu", weights_only=True)

    fc_all    = ri["forecast_256"].float()   # (n_traj, T, 256, 256) — FNO prior
    truth_all = ri["truth_256"   ].float()   # (n_traj, T, 256, 256)
    obs32_all = ri["obs_32"      ].float()   # (n_traj, T, 32, 32)

    n_traj, T = fc_all.shape[0], fc_all.shape[1]
    print(f"  n_traj={n_traj}  T={T}")
    print(f"  FNO forecast RMSE (background): "
          f"{_rmse(fc_all, truth_all):.4f}")

    # ── Estimate Kalman gains from these forecast and observation errors ───────
    print("\nEstimating Kalman gains ...")

    if args.gain_source == "unit":
        gains = torch.zeros(129)
        gains[1:17] = 1.0
        print("  Gains: unit step  (K=1 for k≤16, K=0 beyond)")
    else:
        # Background error: IR FNO forecast - truth (test set, what we have)
        bg_err_flat  = fc_all.reshape(-1, 256, 256) - truth_all.reshape(-1, 256, 256)

        # Observation error: spectral_upsample(obs_32) - truth
        obs_flat     = obs32_all.reshape(-1, 32, 32)
        truth_flat   = truth_all.reshape(-1, 256, 256)
        chunk        = 32
        obs_errs: list[torch.Tensor] = []
        for i in range(0, obs_flat.shape[0], chunk):
            up = spectral_upsample_2d(obs_flat[i:i+chunk], 256, 256)
            obs_errs.append(up - truth_flat[i:i+chunk])
        obs_err_flat = torch.cat(obs_errs, dim=0)
        del obs_errs, obs_flat, truth_flat

        E_bg,  k_bins = radial_energy_spectrum(bg_err_flat)
        E_obs, _      = radial_energy_spectrum(obs_err_flat)
        del bg_err_flat, obs_err_flat

        k_max_obs = 16
        gains = torch.zeros(len(E_bg))
        for ki in range(1, len(gains)):
            if k_bins[ki].item() <= k_max_obs:
                b = E_bg[ki].clamp(min=1e-30)
                o = E_obs[ki].clamp(min=1e-30)
                gains[ki] = b / (b + o)

        print(f"  Gains (k=1..18): "
              + "  ".join(f"k{int(k_bins[ki].item())}={gains[ki].item():.3f}"
                          for ki in range(1, min(19, len(gains)))))

    # ── Build gain map and apply OI to all forecasts at once ─────────────────
    # OI is a linear operation — no GPU needed, no loop required.
    # Apply it to the full (n_traj, T, 256, 256) forecast tensor directly.
    gain_map = build_gain_map(gains, ny=256, nx=256)   # (256, 129)

    print("\nApplying spectral OI to all forecasts ...")
    # Upsample all obs to 256 in chunks
    obs_flat = obs32_all.reshape(-1, 32, 32)
    chunk = 64
    obs_up_chunks: list[torch.Tensor] = []
    for i in range(0, obs_flat.shape[0], chunk):
        obs_up_chunks.append(spectral_upsample_2d(obs_flat[i:i+chunk], 256, 256))
    obs_up_all = torch.cat(obs_up_chunks, dim=0).reshape(n_traj, T, 256, 256)
    del obs_up_chunks, obs_flat

    # Apply OI: w_a = (1 - K) * w_b + K * w_obs  in spectral space
    # Process in (n_traj, T) chunks to avoid one giant rfft2 call
    oi_chunks: list[torch.Tensor] = []
    batch_oi = n_traj * 10   # 10 timesteps at a time
    fc_flat  = fc_all.reshape(-1, 256, 256)
    obs_flat2 = obs_up_all.reshape(-1, 256, 256)
    for i in range(0, fc_flat.shape[0], batch_oi):
        oi_chunks.append(
            spectral_oi(fc_flat[i:i+batch_oi], obs_flat2[i:i+batch_oi], gain_map)
        )
    oi_all = torch.cat(oi_chunks, dim=0).reshape(n_traj, T, 256, 256)
    del oi_chunks, fc_flat, obs_flat2, obs_up_all

    # ── Metrics ───────────────────────────────────────────────────────────────
    pt_oi = _per_traj_rmse(oi_all, truth_all)
    pt_fc = _per_traj_rmse(fc_all, truth_all)

    rmse_oi     = float(pt_oi.mean())
    rmse_oi_std = float(pt_oi.std()) if n_traj > 1 else 0.0
    rmse_fc     = float(pt_fc.mean())
    rmse_ir     = _rmse(ri["posterior_256"].float(), truth_all)

    metrics = {
        "rmse_oi_256":      rmse_oi,
        "rmse_oi_std_256":  rmse_oi_std,
        "rmse_fc_256":      rmse_fc,
        "rmse_ir_256":      rmse_ir,
        "gain_source":      args.gain_source,
    }

    # ── Load bicubic and one-shot for table ───────────────────────────────────
    ro_path = Path("results_oneshot") / "inference_results.pt"
    bic_rmse = one_rmse = float("nan")
    if ro_path.exists():
        ro   = torch.load(ro_path, map_location="cpu", weights_only=True)
        T_c  = min(T, ro["posterior_256"].shape[1])
        n_c  = min(n_traj, ro["posterior_256"].shape[0])
        bic_rmse = _rmse(ro["bicubic_256"  ][:n_c, :T_c].float(),
                         truth_all[:n_c, :T_c])
        one_rmse = _rmse(ro["posterior_256"][:n_c, :T_c].float(),
                         truth_all[:n_c, :T_c])

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  OI Baseline Comparison  @ 256×256  ({n_traj} trajectories × {T} steps)")
    print(f"{'='*62}")
    print(f"  {'Method':<42}  {'RMSE':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Spectral upsample (bicubic)':<42}  {bic_rmse:8.4f}")
    print(f"  {'FNO forecast (background, shared prior)':<42}  {rmse_fc:8.4f}")
    print(f"  {'Spectral OI + FNO  (linear DA)':<42}  {rmse_oi:8.4f}")
    print(f"  {'One-Shot Diffusion SR':<42}  {one_rmse:8.4f}")
    print(f"  {'Iterative Refinement (ours)':<42}  {rmse_ir:8.4f}")
    print(f"{'='*62}")
    print(f"\n  OI vs forecast improvement:  {(rmse_fc-rmse_oi)/rmse_fc*100:.1f}%")
    print(f"  IR  vs OI  comparison:       "
          f"{(rmse_oi-rmse_ir)/rmse_oi*100:+.1f}% "
          f"({'IR better' if rmse_ir < rmse_oi else 'OI better'})")
    print(f"\n  Note: OI and IR use identical FNO forecasts as the prior.")
    print(f"  The only difference is the correction step (linear vs nonlinear diffusion).")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_dir = Path("results_oi")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"

    torch.save(
        {
            "oi_256":       oi_all,
            "forecast_256": fc_all,
            "truth_256":    truth_all,
            "obs_32":       obs32_all,
            "gains":        gains,
            "trmse_curve":  _per_step_rmse(oi_all, truth_all),
            "metrics":      metrics,
        },
        save_path,
    )
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
