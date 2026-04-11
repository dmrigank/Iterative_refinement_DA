"""
Run one-shot diffusion SR inference on the 2D Kraichnan test set.

Loads the trained one-shot EMA checkpoint, runs:
  1. One-shot autoregressive pipeline (diffusion posterior)
  2. Spectral upsample bicubic baseline (no model)

Also loads iterative refinement results from results_2d/ for comparison.

Saves results to results_oneshot/inference_results.pt and prints a 4-row
RMSE table:
  Bicubic (spectral) | FNO-only (autoreg.) | One-Shot SR | Iterative Refinement

Usage:
    python scripts/run_inference_oneshot.py [--config configs/oneshot_sr.yaml]
                                            [--n_steps N]
                                            [--ddim_steps S]
                                            [--eta ETA]
                                            [--iterative results_2d/inference_results.pt]

Output: results_oneshot/inference_results.pt
  Dict keys:
    'posterior_256': (n_traj, T, 256, 256) — one-shot diffusion posterior
    'bicubic_256':   (n_traj, T, 256, 256) — spectral upsample baseline
    'truth_256':     (n_traj, T, 256, 256) — ground truth
    'obs_32':        (n_traj, T, 32,  32)  — coarse observations
    'metrics':       dict of metric_name -> float
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

from src.models.unet_oneshot import OneShotUNet2d
from src.models.diffusion import GaussianDiffusion
from src.inference.pipeline_oneshot import OneShotPipeline


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_oneshot_diffusion(cfg, device: torch.device) -> GaussianDiffusion:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)   # "checkpoints_oneshot"
    ckpt_path = ckpt_dir / "oneshot_ema.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Loosen type so GaussianDiffusion accepts OneShotUNet2d
    import src.models.diffusion as _dm
    _dm.ConditionalUNet1d = nn.Module

    unet = OneShotUNet2d(
        base_channels  = int(cfg.unet.base_channels),
        channel_mults  = list(cfg.unet.channel_mults),
        cond_embed_dim = int(cfg.unet.cond_embed_dim),
        n_groups       = int(cfg.unet.group_norm_groups),
    ).to(device)
    unet.load_state_dict(ckpt["model"])   # EMA weights
    unet.eval()

    diffusion = GaussianDiffusion(unet, cfg).to(device)
    print(f"  Loaded one-shot EMA  (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4e})")
    return diffusion


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


def per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-trajectory mean-over-time RMSE. (n_traj, T, ny, nx) -> (n_traj,)"""
    return (pred - truth).pow(2).mean(dim=(2, 3)).sqrt().mean(dim=1)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-shot diffusion SR inference")
    parser.add_argument("--config",     type=str, default="configs/oneshot_sr.yaml")
    parser.add_argument("--n_steps",    type=int, default=None,
                        help="Time steps per trajectory (default: cfg.inference.test_time_steps)")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="Override DDIM steps")
    parser.add_argument("--eta",        type=float, default=None,
                        help="Override DDIM eta (0=deterministic, 1=DDPM)")
    parser.add_argument("--iterative",  type=str,
                        default="results_2d/inference_results.pt",
                        help="Path to iterative refinement results for comparison table")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = str(cfg.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    n_steps = args.n_steps if args.n_steps is not None else int(cfg.inference.test_time_steps)
    print(f"Time steps per trajectory: {n_steps}")

    if args.ddim_steps is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"inference": {"ddim_steps": args.ddim_steps}}))
        print(f"DDIM steps: {args.ddim_steps}")

    if args.eta is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"inference": {"eta": args.eta}}))
        print(f"Eta: {args.eta}")

    # ── Load one-shot model ───────────────────────────────────────────────────
    print("\nLoading one-shot model...")
    diffusion = load_oneshot_diffusion(cfg, device)
    pipeline  = OneShotPipeline(diffusion, cfg, device)

    # ── Load test data ────────────────────────────────────────────────────────
    data_dir  = Path(cfg.data.data_dir)   # "data_2d"
    test_data = torch.load(data_dir / "test.pt", map_location="cpu", weights_only=True)
    n_test    = test_data["w_32"].shape[0]
    print(f"\nTest trajectories: {n_test}  ({n_steps} steps each)")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_posterior: list[torch.Tensor] = []
    all_bicubic:   list[torch.Tensor] = []
    all_truth:     list[torch.Tensor] = []
    all_obs32:     list[torch.Tensor] = []

    for traj_i in tqdm(range(n_test), desc="Trajectories", ncols=90):
        obs_32    = test_data["w_32" ][traj_i, :n_steps]   # (T, 32, 32)
        truth_256 = test_data["w_256"][traj_i, :n_steps]   # (T, 256, 256)

        res_diff = pipeline.run(obs_32, n_steps=n_steps)
        res_bic  = pipeline.run_bicubic_baseline(obs_32, n_steps=n_steps)

        all_posterior.append(res_diff["posterior_256"])  # (T, 256, 256)
        all_bicubic.append(res_bic["bicubic_256"])       # (T, 256, 256)
        all_truth.append(truth_256.cpu())
        all_obs32.append(res_diff["obs_32"])

    # ── Assemble tensors ──────────────────────────────────────────────────────
    post_all  = torch.stack(all_posterior, dim=0)   # (n_traj, T, 256, 256)
    bic_all   = torch.stack(all_bicubic,   dim=0)
    truth_all = torch.stack(all_truth,     dim=0)
    obs32_all = torch.stack(all_obs32,     dim=0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    pt_post = per_traj_rmse(post_all, truth_all)
    pt_bic  = per_traj_rmse(bic_all,  truth_all)

    rmse_post     = float(pt_post.mean())
    rmse_post_std = float(pt_post.std()) if n_test > 1 else 0.0
    rmse_bic      = float(pt_bic.mean())
    rmse_bic_std  = float(pt_bic.std())  if n_test > 1 else 0.0

    metrics = {
        "rmse_posterior_256":     rmse_post,
        "rmse_post_std_256":      rmse_post_std,
        "rmse_bicubic_256":       rmse_bic,
        "rmse_bicubic_std_256":   rmse_bic_std,
    }

    # ── Load iterative results for comparison ─────────────────────────────────
    iter_path = Path(args.iterative)
    ri = None
    if iter_path.exists():
        ri = torch.load(iter_path, map_location="cpu", weights_only=True)
        pt_iter = per_traj_rmse(ri["posterior_256"], ri["truth_256"])
        pt_fno  = per_traj_rmse(ri["fno_only_256"],  ri["truth_256"])
        rmse_iter     = float(pt_iter.mean())
        rmse_iter_std = float(pt_iter.std()) if pt_iter.shape[0] > 1 else 0.0
        rmse_fno      = float(pt_fno.mean())
        rmse_fno_std  = float(pt_fno.std())  if pt_fno.shape[0] > 1 else 0.0
        metrics["rmse_iterative_256"]     = rmse_iter
        metrics["rmse_iterative_std_256"] = rmse_iter_std
        metrics["rmse_fno_only_256"]      = rmse_fno
        metrics["rmse_fno_only_std_256"]  = rmse_fno_std
    else:
        print(f"\n  [WARN] Iterative results not found at {iter_path} — skipping comparison")

    # ── Print comparison table ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RMSE Comparison  (256×256, mean ± std across {n_test} test trajectories)")
    print(f"{'='*65}")
    print(f"  {'Method':<35}  {'RMSE':>8}  {'Std':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Spectral upsample (bicubic)':<35}  {rmse_bic:8.4f}  {rmse_bic_std:8.4f}")
    if ri is not None:
        print(f"  {'FNO-only (autoregressive)':<35}  {rmse_fno:8.4f}  {rmse_fno_std:8.4f}")
    print(f"  {'One-Shot Diffusion SR':<35}  {rmse_post:8.4f}  {rmse_post_std:8.4f}")
    if ri is not None:
        print(f"  {'Iterative Refinement (ours)':<35}  {rmse_iter:8.4f}  {rmse_iter_std:8.4f}")
    print(f"{'='*65}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "posterior_256": post_all,
        "bicubic_256":   bic_all,
        "truth_256":     truth_all,
        "obs_32":        obs32_all,
        "metrics":       metrics,
    }

    results_dir = Path(cfg.paths.results_dir)   # "results_oneshot"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"
    torch.save(results, save_path)
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
