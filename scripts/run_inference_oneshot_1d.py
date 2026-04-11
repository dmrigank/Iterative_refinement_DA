"""
Run 1D one-shot diffusion SR inference on the test set.

Loads the trained EMA checkpoint, runs the autoregressive one-shot pipeline
and the spectral-upsample (bicubic) baseline on all test trajectories,
then loads iterative-refinement results from results/ for comparison.

Output: results_oneshot_1d/inference_results.pt
  Dict keys:
    'posterior_512' : (n_traj, T, 512) — one-shot diffusion posterior
    'bicubic_512'   : (n_traj, T, 512) — spectral upsample baseline
    'truth_512'     : (n_traj, T, 512) — ground truth
    'obs_64'        : (n_traj, T, 64)  — coarse observations

Usage:
    python scripts/run_inference_oneshot_1d.py [--config configs/oneshot_sr_1d.yaml]
                                               [--n_steps N]
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

from src.models.unet_oneshot_1d import OneShotUNet1d
from src.models.diffusion import GaussianDiffusion
from src.inference.pipeline_oneshot_1d import OneShotPipeline1d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_oneshot_model(cfg, device: torch.device) -> GaussianDiffusion:
    ckpt_path = Path(cfg.paths.checkpoint_dir) / "oneshot_ema.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    unet = OneShotUNet1d(
        base_channels  = int(cfg.unet.base_channels),
        channel_mults  = list(cfg.unet.channel_mults),
        n_res_blocks   = int(cfg.unet.n_res_blocks),
        n_groups       = int(cfg.unet.group_norm_groups),
        cond_embed_dim = int(cfg.unet.cond_embed_dim),
    ).to(device)
    unet.load_state_dict(ckpt["model"])
    unet.eval()

    diffusion = GaussianDiffusion(unet, cfg).to(device)
    print(f"  Loaded one-shot EMA  (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4e})")
    return diffusion


def per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """RMSE per trajectory.  pred, truth: (n_traj, T, N) -> (n_traj,)."""
    return (pred - truth).pow(2).mean(dim=(1, 2)).sqrt()


def rmse_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1D one-shot SR inference")
    p.add_argument("--config",  type=str, default="configs/oneshot_sr_1d.yaml")
    p.add_argument("--n_steps", type=int, default=None,
                   help="Time steps per trajectory (default: cfg.inference.test_time_steps)")
    return p.parse_args()


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

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading one-shot model ...")
    diffusion = load_oneshot_model(cfg, device)
    pipeline  = OneShotPipeline1d(diffusion, cfg, device)

    # ── Load test data ───────────────────────────────────────────────────────
    data_dir  = Path(cfg.data.data_dir)
    test_data = torch.load(data_dir / "test.pt", map_location="cpu", weights_only=True)
    n_test    = test_data["u_64"].shape[0]
    print(f"Test trajectories: {n_test}  ({n_steps} steps each)")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_posterior: list[torch.Tensor] = []
    all_bicubic:   list[torch.Tensor] = []
    all_truth:     list[torch.Tensor] = []
    all_obs64:     list[torch.Tensor] = []

    for traj_i in tqdm(range(n_test), desc="Trajectories", ncols=80):
        obs_64 = test_data["u_64"][traj_i, :n_steps, :]   # (T, 64)

        res_full = pipeline.run(obs_64, n_steps=n_steps)
        res_bic  = pipeline.run_bicubic(obs_64, n_steps=n_steps)

        all_posterior.append(res_full["posterior_512"])    # (T, 512)
        all_bicubic.append(  res_bic["bicubic_512"])       # (T, 512)
        all_truth.append(test_data["u_512"][traj_i, :n_steps, :].cpu())  # (T, 512)
        all_obs64.append(obs_64.cpu())

    posterior_all = torch.stack(all_posterior, dim=0)   # (n_traj, T, 512)
    bicubic_all   = torch.stack(all_bicubic,   dim=0)
    truth_all     = torch.stack(all_truth,     dim=0)
    obs64_all     = torch.stack(all_obs64,     dim=0)

    # ── Load iterative results for comparison ────────────────────────────────
    iter_path = Path("results") / "inference_results.pt"
    iter_avail = iter_path.exists()
    if iter_avail:
        iter_res = torch.load(iter_path, map_location="cpu", weights_only=True)
        T_min = min(posterior_all.shape[1], iter_res["posterior_512"].shape[1])
        iter_post = iter_res["posterior_512"][:n_test, :T_min]
        iter_fno  = iter_res.get("fno_only_512",
                        iter_res.get("forecast_512"))[:n_test, :T_min]
        iter_truth = iter_res["truth_512"][:n_test, :T_min].cpu()
        # Align all to same T
        T_min = min(T_min, posterior_all.shape[1])
        posterior_cmp = posterior_all[:, :T_min]
        bicubic_cmp   = bicubic_all[:, :T_min]
        truth_cmp     = truth_all[:, :T_min]
    else:
        print(f"  [WARNING] Iterative results not found at {iter_path} — skipping comparison.")
        iter_post  = None
        iter_fno   = None
        posterior_cmp = posterior_all
        bicubic_cmp   = bicubic_all
        truth_cmp     = truth_all

    # ── Print comparison table ────────────────────────────────────────────────
    bic_rmse  = rmse_scalar(bicubic_cmp,   truth_cmp)
    one_rmse  = rmse_scalar(posterior_cmp, truth_cmp)
    col_w = 18
    print(f"\n{'─'*70}")
    print(f"  {'Method':<28} {'RMSE @ 512':>{col_w}}")
    print(f"{'─'*70}")
    print(f"  {'Bicubic (spectral up)':<28} {bic_rmse:>{col_w}.4f}")
    if iter_avail and iter_fno is not None:
        fno_rmse  = rmse_scalar(iter_fno,   truth_cmp)
        print(f"  {'FNO-only (autoreg.)':<28} {fno_rmse:>{col_w}.4f}")
    print(f"  {'One-Shot SR':<28} {one_rmse:>{col_w}.4f}")
    if iter_avail and iter_post is not None:
        it_rmse   = rmse_scalar(iter_post,  truth_cmp)
        print(f"  {'Iterative Refinement':<28} {it_rmse:>{col_w}.4f}")
    print(f"{'─'*70}")

    # ── Save ─────────────────────────────────────────────────────────────────
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"
    torch.save(
        {
            "posterior_512": posterior_all,
            "bicubic_512":   bicubic_all,
            "truth_512":     truth_all,
            "obs_64":        obs64_all,
        },
        save_path,
    )
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
