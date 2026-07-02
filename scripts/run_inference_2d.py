"""
Run iterative refinement inference on the 2D Kraichnan test set.

Loads trained 2D FNO and diffusion (EMA) checkpoints, runs both the full
iterative refinement pipeline and the FNO-only autoregressive baseline on
all test trajectories, computes RMSE metrics, and saves everything to
results_2d/.

Usage:
    python scripts/run_inference_2d.py [--config configs/kraichnan.yaml]
                                       [--n_steps N]
                                       [--traj_idx I]
                                       [--ddim_steps S]

Output: results_2d/inference_results.pt
  Dict keys:
    'posterior_{N}'  : (n_traj, T, N, N)  — diffusion posterior
    'forecast_{N}'   : (n_traj, T, N, N)  — FNO one-step forecast (prior)
    'fno_only_{N}'   : (n_traj, T, N, N)  — FNO-only autoregressive baseline
    'truth_{N}'      : (n_traj, T, N, N)  — ground truth
    'obs_32'         : (n_traj, T, 32, 32) — coarse observations
    'metrics'        : dict of metric_name -> float
  where N ∈ {64, 128, 256}.
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
from src.models.fno_2d_shared import SharedFNO2d, SharedFNOResolutionAdapter
from src.models.unet_2d import ConditionalUNet2d
from src.models.diffusion import GaussianDiffusion
from src.inference.pipeline_2d import IterativeRefinementPipeline2d


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_fnos_2d(cfg, device: torch.device) -> dict[int, FNO2d]:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    fnos: dict[int, FNO2d] = {}
    for res in [64, 128, 256]:
        model = FNO2d(cfg, res).to(device)
        ckpt  = torch.load(ckpt_dir / f"fno_{res}.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        fnos[res] = model
        print(f"  Loaded FNO2d {res}×{res}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4e})")
    return fnos


def load_shared_fno_2d(
    cfg, device: torch.device, k_max_shared: int = 64,
) -> SharedFNOResolutionAdapter:
    """Load the shared FNO checkpoint and wrap it in a dict-like adapter so
    it drops into IterativeRefinementPipeline2d (which indexes fnos[res])
    without any pipeline changes.
    """
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    model    = SharedFNO2d(cfg, k_max_shared=k_max_shared).to(device)
    ckpt     = torch.load(ckpt_dir / "fno_shared.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded SharedFNO2d  k_max_shared={k_max_shared}  "
          f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4e})")
    return SharedFNOResolutionAdapter(model, resolutions=(64, 128, 256))


def load_diffusion_2d(cfg, device: torch.device,
                      diffusion_ckpt: str | None = None) -> GaussianDiffusion:
    ckpt_path = Path(diffusion_ckpt) if diffusion_ckpt else \
                Path(cfg.paths.checkpoint_dir) / "diffusion_ema.pt"
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Patch type hint so GaussianDiffusion accepts ConditionalUNet2d
    import src.models.diffusion as _dm
    _dm.ConditionalUNet1d = nn.Module

    unet = ConditionalUNet2d(cfg).to(device)
    unet.load_state_dict(ckpt["model"])   # 'model' key already holds EMA weights
    unet.eval()

    diffusion = GaussianDiffusion(unet, cfg).to(device)
    print(f"  Loaded diffusion EMA  (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4e})")
    return diffusion


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


def per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Per-trajectory RMSE. pred/truth: (n_traj, T, ny, nx) -> (n_traj,)"""
    return (pred - truth).pow(2).mean(dim=(1, 2, 3)).sqrt()


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2D iterative refinement inference")
    parser.add_argument("--config",     type=str, default="configs/kraichnan.yaml")
    parser.add_argument("--n_steps",    type=int, default=None,
                        help="Time steps per trajectory (default: cfg.inference.test_time_steps)")
    parser.add_argument("--traj_idx",   type=int, default=None,
                        help="Run only this trajectory index (default: all)")
    parser.add_argument("--ddim_steps",  type=int,   default=None,
                        help="Override DDIM steps (default: cfg.inference.ddim_steps)")
    parser.add_argument("--eta",         type=float, default=None,
                        help="Override DDIM eta (0=deterministic, 1=DDPM; default: cfg.inference.eta)")
    parser.add_argument("--results_dir", type=str,   default=None,
                        help="Override results directory (default: cfg.paths.results_dir)")
    parser.add_argument("--diffusion_ckpt", type=str, default=None,
                        help="Override path to diffusion_ema.pt (default: cfg.paths.checkpoint_dir/diffusion_ema.pt)")
    parser.add_argument("--fno_variant", type=str, default="separate",
                        choices=["separate", "shared"],
                        help="'separate' (default): three independent FNO2d checkpoints "
                             "(fno_64.pt/fno_128.pt/fno_256.pt). 'shared': one SharedFNO2d "
                             "checkpoint (fno_shared.pt). When using 'shared', also pass "
                             "--diffusion_ckpt pointing at the diffusion model retrained "
                             "against the shared FNO's forecasts (e.g. diffusion_sharedfno_ema.pt) "
                             "— the default diffusion_ema.pt was trained on the separate FNOs' "
                             "forecast distribution and is NOT valid for the shared FNO.")
    parser.add_argument("--k_max_shared", type=int, default=64,
                        help="Fixed spectral mode truncation used by the shared FNO "
                             "(only relevant when --fno_variant shared; must match training)")
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
        print(f"DDIM steps overridden to: {args.ddim_steps}")

    if args.eta is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"inference": {"eta": args.eta}}))
        print(f"Eta overridden to: {args.eta}")

    if args.results_dir is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"paths": {"results_dir": args.results_dir}}))
        print(f"Results dir overridden to: {args.results_dir}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading models...")
    if args.fno_variant == "shared":
        fnos = load_shared_fno_2d(cfg, device, k_max_shared=args.k_max_shared)
    else:
        fnos = load_fnos_2d(cfg, device)
    diffusion = load_diffusion_2d(cfg, device, diffusion_ckpt=args.diffusion_ckpt)
    pipeline  = IterativeRefinementPipeline2d(fnos, diffusion, cfg, device)

    # ── Load test data ───────────────────────────────────────────────────────
    test_data = torch.load(
        Path(cfg.data.data_dir) / "test.pt",
        map_location="cpu",
        weights_only=True,
    )
    n_test       = test_data["w_32"].shape[0]
    traj_indices = [args.traj_idx] if args.traj_idx is not None else list(range(n_test))
    print(f"\nTest trajectories: {traj_indices}  ({n_steps} steps each)")

    # ── Run inference ─────────────────────────────────────────────────────────
    target_resolutions = [64, 128, 256]

    all_posterior: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_forecast:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_fno_only:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_truth:     dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    all_obs32:     list[torch.Tensor] = []

    for traj_i in tqdm(traj_indices, desc="Trajectories", ncols=90):
        obs_32 = test_data["w_32"][traj_i, :n_steps]   # (T, 32, 32)

        res_full = pipeline.run(obs_32, n_steps=n_steps)
        res_fno  = pipeline.run_fno_only(obs_32, n_steps=n_steps)

        all_obs32.append(res_full["obs_32"])  # (T, 32, 32)

        for res in target_resolutions:
            all_posterior[res].append(res_full[f"posterior_{res}"])
            all_forecast[res].append(res_full[f"forecast_{res}"])
            all_fno_only[res].append(res_fno[f"fno_only_{res}"])
            all_truth[res].append(test_data[f"w_{res}"][traj_i, :n_steps].cpu())

    # ── Assemble results & metrics ────────────────────────────────────────────
    results: dict[str, object] = {}
    results["obs_32"] = torch.stack(all_obs32, dim=0)   # (n_traj, T, 32, 32)

    print("\n── RMSE Summary ──────────────────────────────────────────────────────")
    print(f"{'Resolution':>14}  {'Posterior':>12}  {'FNO-only':>12}  {'FNO forecast':>12}")

    metrics: dict[str, float] = {}

    for res in target_resolutions:
        post_all  = torch.stack(all_posterior[res], dim=0)   # (n_traj, T, res, res)
        fc_all    = torch.stack(all_forecast[res],  dim=0)
        fno_all   = torch.stack(all_fno_only[res],  dim=0)
        truth_all = torch.stack(all_truth[res],     dim=0)

        results[f"posterior_{res}"] = post_all
        results[f"forecast_{res}"]  = fc_all
        results[f"fno_only_{res}"]  = fno_all
        results[f"truth_{res}"]     = truth_all

        rmse_post = rmse(post_all,  truth_all)
        rmse_fno  = rmse(fno_all,   truth_all)
        rmse_fc   = rmse(fc_all,    truth_all)

        pt_post = per_traj_rmse(post_all, truth_all)
        pt_fno  = per_traj_rmse(fno_all,  truth_all)

        metrics[f"rmse_posterior_{res}"]    = rmse_post
        metrics[f"rmse_fno_only_{res}"]     = rmse_fno
        metrics[f"rmse_forecast_{res}"]     = rmse_fc
        metrics[f"rmse_post_std_{res}"]     = float(pt_post.std())
        metrics[f"rmse_fno_only_std_{res}"] = float(pt_fno.std())

        print(
            f"  N={res:3d}×{res:<3d}      "
            f"{rmse_post:12.4f}  {rmse_fno:12.4f}  {rmse_fc:12.4f}"
        )

    results["metrics"] = metrics

    # ── Save ─────────────────────────────────────────────────────────────────
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"
    torch.save(results, save_path)
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
