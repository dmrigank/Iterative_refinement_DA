"""
Run iterative refinement inference on the test set.

Loads trained FNO and diffusion (EMA) checkpoints, runs both the full
iterative refinement pipeline and the FNO-only baseline on all test
trajectories, computes RMSE metrics, and saves everything to results/.

Usage:
    python scripts/run_inference.py [--config configs/default.yaml]
                                    [--n_steps N]   (default: cfg.inference.test_time_steps)
                                    [--traj_idx I]  (default: all test trajectories)

Output: results/inference_results.pt
  Dict keys:
    'posterior_{N}' : (n_traj, T, N)  — diffusion posterior at resolution N
    'forecast_{N}'  : (n_traj, T, N)  — FNO forecast (prior) at resolution N
    'fno_only_{N}'  : (n_traj, T, N)  — FNO-only autoregressive baseline
    'truth_{N}'     : (n_traj, T, N)  — ground truth
    'obs_64'        : (n_traj, T, 64) — coarse observations
    'metrics'       : dict of metric_name -> float
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.fno import FNO1d
from src.models.unet import ConditionalUNet1d
from src.models.diffusion import GaussianDiffusion
from src.training.train_diffusion import EMAModel
from src.inference.pipeline import IterativeRefinementPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fnos(cfg, device: torch.device) -> dict[int, FNO1d]:
    """Load all FNO checkpoints."""
    fnos: dict[int, FNO1d] = {}
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    for res in [128, 256, 512]:
        model = FNO1d(cfg, res).to(device)
        ckpt  = torch.load(ckpt_dir / f"fno_{res}.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        fnos[res] = model
        print(f"  Loaded FNO {res}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4e})")
    return fnos


def load_diffusion(cfg, device: torch.device) -> GaussianDiffusion:
    """Load diffusion EMA checkpoint."""
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt     = torch.load(ckpt_dir / "diffusion_ema.pt", map_location=device, weights_only=True)

    unet = ConditionalUNet1d(cfg).to(device)
    unet.load_state_dict(ckpt["model"])   # 'model' key already holds EMA weights
    unet.eval()

    diffusion = GaussianDiffusion(unet, cfg).to(device)
    print(f"  Loaded diffusion EMA  (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4e})")
    return diffusion


def rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """RMSE over all elements."""
    return float((pred - truth).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run iterative refinement inference")
    parser.add_argument("--config",      type=str,   default="configs/default.yaml")
    parser.add_argument("--n_steps",     type=int,   default=None,
                        help="Number of time steps per trajectory (default: cfg.inference.test_time_steps)")
    parser.add_argument("--traj_idx",    type=int,   default=None,
                        help="Run only this trajectory index (default: all)")
    parser.add_argument("--ddim_steps",  type=int,   default=None,
                        help="Override DDIM steps (default: cfg.inference.ddim_steps)")
    parser.add_argument("--eta",         type=float, default=None,
                        help="Override DDIM eta (0=deterministic, 1=DDPM; default: cfg.inference.eta)")
    parser.add_argument("--results_dir", type=str,   default=None,
                        help="Override results directory (default: cfg.paths.results_dir)")
    return parser.parse_args()


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
        print(f"DDIM steps overridden to: {args.ddim_steps}")

    if args.eta is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"inference": {"eta": args.eta}}))
        print(f"Eta overridden to: {args.eta}")

    if args.results_dir is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"paths": {"results_dir": args.results_dir}}))
        print(f"Results dir overridden to: {args.results_dir}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading models...")
    fnos      = load_fnos(cfg, device)
    diffusion = load_diffusion(cfg, device)

    pipeline  = IterativeRefinementPipeline(fnos, diffusion, cfg, device)

    # ── Load test data ───────────────────────────────────────────────────────
    test_data = torch.load(
        Path(cfg.data.data_dir) / "test.pt",
        map_location="cpu",
        weights_only=True,
    )
    n_test = test_data["u_64"].shape[0]
    traj_indices = [args.traj_idx] if args.traj_idx is not None else list(range(n_test))
    print(f"\nTest trajectories: {traj_indices}  ({n_steps} steps each)")

    # ── Run inference ─────────────────────────────────────────────────────────
    target_resolutions = [128, 256, 512]

    # Collectors: list over trajectories
    all_posterior: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_forecast:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_fno_only:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_truth:     dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_obs64:     list[torch.Tensor] = []

    for traj_i in tqdm(traj_indices, desc="Trajectories", ncols=80):
        obs_64  = test_data["u_64" ][traj_i, :n_steps, :]   # (T, 64)

        # Full iterative refinement
        res_full = pipeline.run(obs_64, n_steps=n_steps)
        # FNO-only baseline
        res_fno  = pipeline.run_fno_only(obs_64, n_steps=n_steps)

        all_obs64.append(res_full["obs_64"])  # (T, 64)

        for res in target_resolutions:
            all_posterior[res].append(res_full[f"posterior_{res}"])   # (T, res)
            all_forecast[res].append(res_full[f"forecast_{res}"]  )   # (T, res)
            all_fno_only[res].append(res_fno[ f"fno_only_{res}"]  )   # (T, res)
            all_truth[res].append(
                test_data[f"u_{res}"][traj_i, :n_steps, :].cpu()      # (T, res)
            )

    # ── Assemble & compute metrics ────────────────────────────────────────────
    results: dict[str, object] = {}

    results["obs_64"] = torch.stack(all_obs64, dim=0)   # (n_traj, T, 64)

    print("\n── RMSE Summary ─────────────────────────────────────────────────")
    print(f"{'Resolution':>12}  {'Posterior':>10}  {'FNO-only':>10}  {'FNO forecast':>12}")

    metrics: dict[str, float] = {}
    for res in target_resolutions:
        post_all  = torch.stack(all_posterior[res], dim=0)   # (n_traj, T, res)
        fc_all    = torch.stack(all_forecast[res],  dim=0)
        fno_all   = torch.stack(all_fno_only[res],  dim=0)
        truth_all = torch.stack(all_truth[res],     dim=0)

        results[f"posterior_{res}"] = post_all
        results[f"forecast_{res}" ] = fc_all
        results[f"fno_only_{res}" ] = fno_all
        results[f"truth_{res}"    ] = truth_all

        rmse_post = rmse(post_all,  truth_all)
        rmse_fno  = rmse(fno_all,   truth_all)
        rmse_fc   = rmse(fc_all,    truth_all)

        metrics[f"rmse_posterior_{res}"] = rmse_post
        metrics[f"rmse_fno_only_{res}" ] = rmse_fno
        metrics[f"rmse_forecast_{res}" ] = rmse_fc

        print(f"  N={res:4d}          {rmse_post:10.4f}  {rmse_fno:10.4f}  {rmse_fc:12.4f}")

    results["metrics"] = metrics

    # ── Save ─────────────────────────────────────────────────────────────────
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"
    torch.save(results, save_path)
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
