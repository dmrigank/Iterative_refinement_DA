"""
Ablation: Observation frequency sweep (sparse observations).

Tests how the 2D iterative refinement model degrades when the coarse 32×32
observation is NOT available at every timestep.  Instead, the FNO forecaster
rolls out autoregressively for K steps between each assimilation update:

  For timesteps where an observation IS available (t % K == 0):
    w_forecast  = FNO^1(prev_posterior)     # single-step FNO from last posterior
    w_coarse_up = spectral_upsample(obs_t)
    posterior_t = diffusion_correct(w_forecast, w_coarse_up)

  For timesteps where NO observation is available (t % K != 0):
    w_forecast  = FNO(prev_posterior)
    posterior_t = w_forecast               # pure FNO rollout, no correction

  K=1 recovers the standard pipeline (observe at every step).
  K=5 means one correction every 5 steps, 4 pure-FNO steps between.

Key question: at what K does the diffusion corrector fail to reconcile the
accumulated FNO forecast error with the coarse observation?  The corrector
was trained on single-step FNO errors (std ~0.22 at 256-pt) — multi-step
rollout errors are larger and structurally different (phase-shifted vortices),
placing the correction task out-of-distribution.

K values swept: [1, 2, 3, 5, 10]
  K=1  : baseline (standard pipeline, observe every step)
  K=2,3: mild sparsity — FNO errors haven't compounded much
  K=5  : moderate sparsity — ~0.25 physical time units between corrections
  K=10 : aggressive sparsity — approaching decorrelation timescale

Output: results_2d/ablation_obs_freq/
  obs_freq_K{k}/inference_results.pt  for each K

Each results file contains:
  'posterior_256' : (n_traj, T, 256, 256)
  'truth_256'     : (n_traj, T, 256, 256)
  'obs_32'        : (n_traj, T, 32, 32)
  'obs_mask'      : (T,) bool tensor — True at timesteps where obs was used
  'metrics'       : dict with RMSE at full sequence and at obs/non-obs steps

Usage:
    python scripts/ablation_obs_frequency.py
        [--config   configs/kraichnan.yaml]
        [--n_steps  N]
        [--k_values 1,2,3,5,10]
        [--skip_k1]   skip K=1 (use existing results_2d/inference_results.pt)
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
from src.inference.pipeline_2d import IterativeRefinementPipeline2d
from src.data.dataset_2d import spectral_upsample_2d


# ---------------------------------------------------------------------------
# Sparse-observation pipeline
# ---------------------------------------------------------------------------

class SparseObsPipeline(IterativeRefinementPipeline2d):
    """IR pipeline with observations available only every K timesteps.

    Between observation timesteps, the pipeline runs the FNO autoregressively
    with no diffusion correction.  At observation timesteps, the standard
    three-stage cascade (FNO forecast + diffusion correct) is applied.

    Args:
        k: Observation interval.  k=1 → standard dense-obs pipeline.
    """

    def __init__(
        self,
        fnos: dict[int, FNO2d],
        diffusion: GaussianDiffusion,
        cfg,
        device: torch.device,
        k: int = 1,
    ) -> None:
        super().__init__(fnos, diffusion, cfg, device)
        self.k = k

    @torch.no_grad()
    def run(
        self,
        observations_32: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run with sparse observations, correcting only every k steps.

        At non-observation timesteps, posteriors are replaced by the FNO
        forecast (pure autoregressive rollout) with no DDIM correction.

        Returns same structure as IterativeRefinementPipeline2d.run() plus:
          'obs_mask': (T,) bool tensor, True at observation timesteps
        """
        if observations_32.dim() == 4:
            observations_32 = observations_32.squeeze(1)
        obs = observations_32.to(self.device)   # (T, 32, 32)
        T   = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        target_resolutions = [64, 128, 256]
        posteriors: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
        forecasts:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
        obs_mask: list[bool] = []

        # t=0: always initialise from spectral upsample of first obs
        prev_post: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            w_init = self._upsample(obs[0:1], res)
            prev_post[res] = w_init
            forecasts[res].append(w_init.squeeze().cpu())
            posteriors[res].append(w_init.squeeze().cpu())
        obs_mask.append(True)   # t=0 always uses obs

        for t in tqdm(range(1, T),
                      desc=f"  K={self.k}", ncols=90, leave=False):

            # Observation is available when t is a multiple of k
            # (t=0 already processed; first correction at t=k)
            use_obs = (t % self.k == 0)
            obs_mask.append(use_obs)

            if use_obs:
                # ── Assimilation step: FNO forecast + diffusion correction ──
                obs_t = obs[t:t+1]              # (1, 32, 32)
                stage_posts: dict[int, torch.Tensor] = {}

                for coarse_res, target_res, res_idx in self._STAGES:
                    w_fc = self._fno_forward(self.fnos[target_res], prev_post[target_res])

                    if coarse_res == 32:
                        coarse_field = obs_t
                    else:
                        coarse_field = stage_posts[coarse_res]

                    w_co      = self._upsample(coarse_field, target_res)
                    posterior = self._ddim(w_fc, w_co, res_idx)

                    stage_posts[target_res] = posterior
                    forecasts[target_res].append(w_fc.squeeze().cpu())
                    posteriors[target_res].append(posterior.squeeze().cpu())

                prev_post = {r: stage_posts[r] for r in target_resolutions}

            else:
                # ── Free-forecast step: pure FNO rollout, no correction ──
                # Each resolution is advanced independently with its FNO.
                # The posterior equals the FNO forecast (no obs to correct with).
                for res in target_resolutions:
                    w_fc = self._fno_forward(self.fnos[res], prev_post[res])
                    prev_post[res] = w_fc
                    forecasts[res].append(w_fc.squeeze().cpu())
                    posteriors[res].append(w_fc.squeeze().cpu())

        result: dict[str, torch.Tensor] = {"obs_32": obs.cpu()}
        for res in target_resolutions:
            result[f"posterior_{res}"] = torch.stack(posteriors[res], dim=0)
            result[f"forecast_{res}"]  = torch.stack(forecasts[res],  dim=0)
        result["obs_mask"] = torch.tensor(obs_mask, dtype=torch.bool)
        return result


# ---------------------------------------------------------------------------
# Model loading (reused from run_inference_2d.py)
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
    print(f"  Loaded FNO2d: {list(fnos.keys())}")
    return fnos


def _load_diffusion(cfg, device: torch.device) -> GaussianDiffusion:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt     = torch.load(ckpt_dir / "diffusion_ema.pt",
                          map_location=device, weights_only=True)
    import src.models.diffusion as _dm
    _dm.ConditionalUNet1d = nn.Module
    unet = ConditionalUNet2d(cfg).to(device)
    unet.load_state_dict(ckpt["model"])
    unet.eval()
    diffusion = GaussianDiffusion(unet, cfg).to(device)
    print(f"  Loaded diffusion EMA (step {ckpt['step']})")
    return diffusion


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


def _per_step_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(n_traj, T, H, W) -> (T,) spatial RMSE at each step, averaged over traj."""
    return (pred - truth).pow(2).mean(dim=(0, 2, 3)).sqrt()


# ---------------------------------------------------------------------------
# Run one K variant and save
# ---------------------------------------------------------------------------

def _run_k(
    k: int,
    fnos: dict,
    diffusion: GaussianDiffusion,
    cfg,
    test_data: dict,
    traj_indices: list[int],
    n_steps: int,
    out_root: Path,
    device: torch.device,
) -> dict:
    """Run sparse-obs pipeline for one K value, save and return results."""
    pipeline = SparseObsPipeline(fnos, diffusion, cfg, device, k=k)

    all_post:  list[torch.Tensor] = []
    all_truth: list[torch.Tensor] = []
    all_obs32: list[torch.Tensor] = []
    obs_mask: torch.Tensor | None = None

    for traj_i in tqdm(traj_indices, desc=f"Trajectories (K={k})", ncols=90):
        obs_32    = test_data["w_32" ][traj_i, :n_steps]
        truth_256 = test_data["w_256"][traj_i, :n_steps].float()

        res = pipeline.run(obs_32.float(), n_steps=n_steps)
        all_post.append(res["posterior_256"])
        all_truth.append(truth_256.cpu())
        all_obs32.append(res["obs_32"])
        if obs_mask is None:
            obs_mask = res["obs_mask"]   # same for every trajectory

    post_all  = torch.stack(all_post,  dim=0)   # (n_traj, T, 256, 256)
    truth_all = torch.stack(all_truth, dim=0)
    obs32_all = torch.stack(all_obs32, dim=0)

    rmse_full  = _rmse(post_all, truth_all)
    rmse_curve = _per_step_rmse(post_all, truth_all)   # (T,)

    # RMSE split: assimilation steps vs free-forecast steps
    obs_idx  = obs_mask.nonzero(as_tuple=True)[0]
    free_idx = (~obs_mask).nonzero(as_tuple=True)[0]
    rmse_at_obs  = float((post_all[:, obs_idx]  - truth_all[:, obs_idx] ).pow(2).mean().sqrt()) \
                   if len(obs_idx)  > 0 else float("nan")
    rmse_at_free = float((post_all[:, free_idx] - truth_all[:, free_idx]).pow(2).mean().sqrt()) \
                   if len(free_idx) > 0 else float("nan")

    metrics = {
        "k":              k,
        "rmse_full":      rmse_full,
        "rmse_at_obs":    rmse_at_obs,
        "rmse_at_free":   rmse_at_free,
        "obs_fraction":   float(obs_mask.float().mean()),
    }
    print(f"    K={k}  RMSE={rmse_full:.4f}  "
          f"at_obs={rmse_at_obs:.4f}  at_free={rmse_at_free:.4f}  "
          f"obs_fraction={metrics['obs_fraction']:.2f}")

    results = {
        "posterior_256": post_all,
        "truth_256":     truth_all,
        "obs_32":        obs32_all,
        "obs_mask":      obs_mask,
        "rmse_curve":    rmse_curve,
        "metrics":       metrics,
    }

    out_dir = out_root / f"obs_freq_K{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "inference_results.pt")
    print(f"    Saved {out_dir / 'inference_results.pt'}")
    return results


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation: observation frequency sweep"
    )
    p.add_argument("--config",   type=str, default="configs/kraichnan.yaml")
    p.add_argument("--n_steps",  type=int, default=None)
    p.add_argument("--k_values", type=str, default="1,2,3,5,10",
                   help="Comma-separated K values to sweep (default: 1,2,3,5,10)")
    p.add_argument("--skip_k1",  action="store_true",
                   help="Skip K=1 and copy from results_2d/inference_results.pt")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args     = parse_args()
    cfg      = OmegaConf.load(args.config)
    k_values = [int(x) for x in args.k_values.split(",")]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if str(cfg.device) != "auto":
        device_str = str(cfg.device)
    device = torch.device(device_str)

    n_steps  = args.n_steps if args.n_steps is not None \
               else int(cfg.inference.test_time_steps)
    out_root = Path(cfg.paths.results_dir) / "ablation_obs_freq"

    print("=" * 65)
    print("  Ablation: Observation frequency sweep")
    print(f"  K values: {k_values}  |  n_steps={n_steps}  |  device={device}")
    print(f"  dt per step = {cfg.pde.dt * cfg.pde.save_every:.3f} physical time units")
    print(f"  K=5 → {5 * cfg.pde.dt * cfg.pde.save_every:.3f} time units between corrections")
    print("=" * 65)

    # ── Handle K=1 skip ───────────────────────────────────────────────────────
    if args.skip_k1 and 1 in k_values:
        src = Path(cfg.paths.results_dir) / "inference_results.pt"
        dst_dir = out_root / "obs_freq_K1"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "inference_results.pt"
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy(src, dst)
            print(f"\n  Copied K=1 results from {src}")
        elif dst.exists():
            print(f"\n  K=1 results already at {dst} — skipping")
        k_values = [k for k in k_values if k != 1]

    if not k_values:
        print("\nAll K values handled. Done.")
        _print_summary(out_root, [int(x) for x in args.k_values.split(",")])
        return

    # ── Load models once ──────────────────────────────────────────────────────
    print("\nLoading models...")
    fnos      = _load_fnos(cfg, device)
    diffusion = _load_diffusion(cfg, device)

    # ── Load test data ────────────────────────────────────────────────────────
    test_data    = torch.load(
        Path(cfg.data.data_dir) / "test.pt",
        map_location="cpu", weights_only=True,
    )
    n_test       = test_data["w_32"].shape[0]
    traj_indices = list(range(n_test))
    print(f"  Test trajectories: {n_test}  ({n_steps} steps each)")

    # ── Sweep K ───────────────────────────────────────────────────────────────
    all_k = [int(x) for x in args.k_values.split(",")]
    for k in k_values:
        print(f"\n── K = {k}  ({k * cfg.pde.dt * cfg.pde.save_every:.3f} time units / correction) ──")
        _run_k(k, fnos, diffusion, cfg, test_data,
               traj_indices, n_steps, out_root, device)

    _print_summary(out_root, all_k)


def _print_summary(out_root: Path, k_values: list[int]) -> None:
    print("\n── Obs-frequency Summary ───────────────────────────────────────")
    print(f"  {'K':>4}  {'RMSE (full)':>12}  {'RMSE @ obs':>12}  {'RMSE @ free':>12}")
    print(f"  {'─'*46}")
    for k in k_values:
        path = out_root / f"obs_freq_K{k}" / "inference_results.pt"
        if path.exists():
            r = torch.load(path, map_location="cpu", weights_only=True)
            m = r["metrics"] if isinstance(r.get("metrics"), dict) else {}
            # K=1 from main results may have different key structure
            if "rmse_full" in m:
                rf  = m["rmse_full"]
                ro  = m.get("rmse_at_obs",  float("nan"))
                rfr = m.get("rmse_at_free", float("nan"))
            elif "posterior_256" in r and "truth_256" in r:
                # Fallback: compute from tensors
                post  = r["posterior_256"]
                truth = r["truth_256"]
                rf    = float((post - truth).pow(2).mean().sqrt())
                ro    = float("nan")
                rfr   = float("nan")
            else:
                rf = ro = rfr = float("nan")
            print(f"  {k:>4}  {rf:>12.4f}  {ro:>12.4f}  {rfr:>12.4f}")
        else:
            print(f"  {k:>4}  {'not found':>12}")
    print()


if __name__ == "__main__":
    main()
