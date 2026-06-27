"""
Ablation A1: Cascade depth sweep.

Tests whether each additional refinement stage meaningfully reduces error.
Three variants — all using the SAME trained FNOs and diffusion model (no
retraining), so only the cascade structure differs:

  1-stage  — direct 64→512 in one step
             Uses FNO_512 (res_idx=2) from the shared diffusion model.
             Coarse input: spectral_upsample(obs_64, 512) directly.
  2-stage  — 64→256→512 (skips the 64→128 intermediate)
             Uses FNO_256 (res_idx=1) and FNO_512 (res_idx=2).
  3-stage  — 64→128→256→512 — current default (baseline)

All three variants reuse the trained weights unchanged; only _STAGES differs.
This is a true apples-to-apples comparison of cascade depth.

Hypothesis: each additional stage reduces RMSE and spectral error at high k;
  the 1-stage variant misses fine-scale structure most severely since it
  tries to bridge the full 64→512 gap in a single diffusion pass.

Output directories (tagged under results/):
  results/ablation_a1_1stage/inference_results.pt
  results/ablation_a1_2stage/inference_results.pt
  results/ablation_a1_3stage/inference_results.pt   (re-run of baseline)

Usage:
    python scripts/ablation_a1_cascade_depth.py
        [--config      configs/default.yaml]
        [--n_steps     N]
        [--skip_1stage]   skip 1-stage (use existing results/ablation_a1_1stage/)
        [--skip_2stage]   skip 2-stage
        [--skip_3stage]   skip 3-stage baseline (falls back to results/)
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
from src.inference.pipeline import IterativeRefinementPipeline
from src.data.solver import spectral_upsample


# ---------------------------------------------------------------------------
# 1-stage pipeline: 64 → 512 in a single diffusion step
# ---------------------------------------------------------------------------

class OneStagePipeline(IterativeRefinementPipeline):
    """1-stage cascade: 64→512 directly.

    Uses FNO_512 as the forecaster and res_idx=2 to condition the shared
    diffusion model — the same res_idx the 3-stage pipeline uses for its
    final (256→512) transition.  The coarse input is spectral_upsample(obs_64, 512)
    directly, with no intermediate posteriors.

    This is a strict ablation of cascade depth: same model weights, same
    FNO prior, just a single correction step spanning the full 8× gap.
    """

    _STAGES = [
        # (coarse_res, target_res, res_idx)
        (64, 512, 2),   # direct: 64-pt obs → 512-pt target in one step
    ]

    def run(
        self,
        observations_64: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if observations_64.dim() == 3:
            observations_64 = observations_64.squeeze(1)
        obs = observations_64.to(self.device)
        T   = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        posteriors: list[torch.Tensor] = []
        forecasts:  list[torch.Tensor] = []

        # t=0: initialise from spectral upsample of first obs
        u_init = spectral_upsample(obs[0:1], 512).unsqueeze(1)  # (1, 1, 512)
        prev_post_512 = u_init
        forecasts.append(u_init.squeeze(0).squeeze(0).cpu())
        posteriors.append(u_init.squeeze(0).squeeze(0).cpu())

        for t in tqdm(range(1, T), desc="1-stage inference", ncols=80, leave=False):
            obs_t = obs[t:t+1]   # (1, 64)

            # FNO forecast at 512 from previous 512-pt posterior
            u_fc = self._fno_forward(self.fnos[512], prev_post_512)  # (1, 1, 512)

            # Coarse input: upsample current 64-pt obs directly to 512
            u_co = spectral_upsample(obs_t, 512).unsqueeze(1)        # (1, 1, 512)

            # Single DDIM correction at res_idx=2
            posterior = self._ddim(u_fc, u_co, res_idx=2)            # (1, 1, 512)

            prev_post_512 = posterior
            forecasts.append(u_fc.squeeze().cpu())
            posteriors.append(posterior.squeeze().cpu())

        return {
            "obs_64":        obs.cpu(),
            "posterior_512": torch.stack(posteriors, dim=0),   # (T, 512)
            "forecast_512":  torch.stack(forecasts,  dim=0),   # (T, 512)
        }


# ---------------------------------------------------------------------------
# 2-stage pipeline: 64 → 256 → 512
# ---------------------------------------------------------------------------

class TwoStagePipeline(IterativeRefinementPipeline):
    """2-stage cascade: 64→256 (res_idx=1) then 256→512 (res_idx=2).

    Skips the 64→128 intermediate stage.  Coarse input to the first stage
    is spectral_upsample(obs_64, 256) directly from the observation.
    """

    _STAGES = [
        (64,  256, 1),   # 64-pt obs → 256-pt target
        (256, 512, 2),   # 256-pt posterior → 512-pt target
    ]

    def run(
        self,
        observations_64: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if observations_64.dim() == 3:
            observations_64 = observations_64.squeeze(1)
        obs = observations_64.to(self.device)
        T   = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        target_resolutions = [256, 512]
        posteriors: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
        forecasts:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}

        prev_post: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            u_init = spectral_upsample(obs[0:1], res).unsqueeze(1)
            prev_post[res] = u_init
            forecasts[res].append(u_init.squeeze(0).squeeze(0).cpu())
            posteriors[res].append(u_init.squeeze(0).squeeze(0).cpu())

        for t in tqdm(range(1, T), desc="2-stage inference", ncols=80, leave=False):
            obs_t = obs[t:t+1]
            stage_posts: dict[int, torch.Tensor] = {}

            for coarse_res, target_res, res_idx in self._STAGES:
                u_fc = self._fno_forward(self.fnos[target_res], prev_post[target_res])

                if coarse_res == 64:
                    coarse_field = obs_t
                else:
                    coarse_field = stage_posts[coarse_res].squeeze(1)

                u_co = spectral_upsample(coarse_field, target_res).unsqueeze(1)
                posterior = self._ddim(u_fc, u_co, res_idx)

                stage_posts[target_res] = posterior
                forecasts[target_res].append(u_fc.squeeze().cpu())
                posteriors[target_res].append(posterior.squeeze().cpu())

            prev_post = {r: stage_posts[r] for r in target_resolutions}

        result: dict[str, torch.Tensor] = {"obs_64": obs.cpu()}
        for res in target_resolutions:
            result[f"posterior_{res}"] = torch.stack(posteriors[res], dim=0)
            result[f"forecast_{res}"]  = torch.stack(forecasts[res],  dim=0)
        return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fnos(cfg, device: torch.device) -> dict[int, FNO1d]:
    fnos: dict[int, FNO1d] = {}
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    for res in [128, 256, 512]:
        model = FNO1d(cfg, res).to(device)
        ckpt  = torch.load(ckpt_dir / f"fno_{res}.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        fnos[res] = model
    print(f"  Loaded FNOs: {list(fnos.keys())}")
    return fnos


def _load_diffusion(cfg, device: torch.device) -> GaussianDiffusion:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt     = torch.load(ckpt_dir / "diffusion_ema.pt", map_location=device, weights_only=True)
    unet = ConditionalUNet1d(cfg).to(device)
    unet.load_state_dict(ckpt["model"])
    unet.eval()
    diffusion = GaussianDiffusion(unet, cfg).to(device)
    print(f"  Loaded diffusion EMA (step {ckpt['step']})")
    return diffusion


def _rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Run a pipeline variant and save
# ---------------------------------------------------------------------------

def _run_and_save(
    pipeline,
    test_data: dict,
    traj_indices: list[int],
    n_steps: int,
    target_resolutions: list[int],
    out_dir: Path,
) -> dict:
    collectors: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    truth_coll: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
    obs_coll:   list[torch.Tensor] = []

    for traj_i in tqdm(traj_indices, desc="  Trajectories", ncols=80):
        obs_64 = test_data["u_64"][traj_i, :n_steps, :]
        res    = pipeline.run(obs_64, n_steps=n_steps)
        obs_coll.append(res["obs_64"])
        for r in target_resolutions:
            key = f"posterior_{r}"
            if key in res:
                collectors[r].append(res[key])
                truth_coll[r].append(test_data[f"u_{r}"][traj_i, :n_steps, :].cpu())

    results: dict = {"obs_64": torch.stack(obs_coll, dim=0)}
    metrics: dict[str, float] = {}
    for r in target_resolutions:
        if collectors[r]:
            post  = torch.stack(collectors[r], dim=0)
            truth = torch.stack(truth_coll[r], dim=0)
            results[f"posterior_{r}"] = post
            results[f"truth_{r}"]     = truth
            rmse_val = _rmse(post, truth)
            metrics[f"rmse_posterior_{r}"] = rmse_val
            print(f"    N={r:4d}  RMSE={rmse_val:.4f}")
    results["metrics"] = metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "inference_results.pt")
    print(f"  Saved {out_dir / 'inference_results.pt'}")
    return results


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation A1: cascade depth sweep")
    p.add_argument("--config",      type=str,  default="configs/default.yaml")
    p.add_argument("--n_steps",     type=int,  default=None)
    p.add_argument("--skip_1stage", action="store_true",
                   help="Skip 1-stage run (use existing ablation_a1_1stage/)")
    p.add_argument("--skip_2stage", action="store_true",
                   help="Skip 2-stage run (use existing ablation_a1_2stage/)")
    p.add_argument("--skip_3stage", action="store_true",
                   help="Skip 3-stage run (falls back to results/inference_results.pt)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if str(cfg.device) != "auto":
        device_str = str(cfg.device)
    device = torch.device(device_str)

    n_steps = args.n_steps if args.n_steps is not None else int(cfg.inference.test_time_steps)

    print("=" * 60)
    print("  Ablation A1 — Cascade depth sweep")
    print(f"  n_steps={n_steps}  device={device}")
    print("  All variants use the SAME trained FNOs + diffusion model")
    print("=" * 60)

    print("\nLoading models ...")
    fnos      = _load_fnos(cfg, device)
    diffusion = _load_diffusion(cfg, device)

    test_data    = torch.load(
        Path(cfg.data.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )
    n_test       = test_data["u_64"].shape[0]
    traj_indices = list(range(n_test))
    results_root = Path(cfg.paths.results_dir)

    # ── 1-stage ───────────────────────────────────────────────────────────────
    if args.skip_1stage:
        print("\n── Variant: 1-stage  (skipped) ─────────────────────────────")
    else:
        print("\n── Variant: 1-stage  (64→512) ──────────────────────────────")
        pipeline_1 = OneStagePipeline(fnos, diffusion, cfg, device)
        _run_and_save(
            pipeline_1, test_data, traj_indices, n_steps,
            target_resolutions=[512],
            out_dir=results_root / "ablation_a1_1stage",
        )

    # ── 2-stage ───────────────────────────────────────────────────────────────
    if args.skip_2stage:
        print("\n── Variant: 2-stage  (skipped) ─────────────────────────────")
    else:
        print("\n── Variant: 2-stage  (64→256→512) ─────────────────────────")
        pipeline_2 = TwoStagePipeline(fnos, diffusion, cfg, device)
        _run_and_save(
            pipeline_2, test_data, traj_indices, n_steps,
            target_resolutions=[256, 512],
            out_dir=results_root / "ablation_a1_2stage",
        )

    # ── 3-stage (baseline) ────────────────────────────────────────────────────
    if args.skip_3stage:
        print("\n── Variant: 3-stage  (skipped — using existing results/) ───")
    else:
        print("\n── Variant: 3-stage  (64→128→256→512) [baseline] ──────────")
        pipeline_3 = IterativeRefinementPipeline(fnos, diffusion, cfg, device)
        _run_and_save(
            pipeline_3, test_data, traj_indices, n_steps,
            target_resolutions=[128, 256, 512],
            out_dir=results_root / "ablation_a1_3stage",
        )

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── A1 Summary ─────────────────────────────────────────────────")
    print(f"  {'Variant':<22}  {'N=256 RMSE':>12}  {'N=512 RMSE':>12}")
    print(f"  {'-'*50}")

    variants = [
        ("1-stage  (64→512)",    results_root / "ablation_a1_1stage",  [512]),
        ("2-stage  (64→256→512)",results_root / "ablation_a1_2stage",  [256, 512]),
        ("3-stage  (full)",      results_root / "ablation_a1_3stage",  [256, 512]),
    ]
    # 3-stage fallback to main results if skipped
    if args.skip_3stage:
        variants[-1] = ("3-stage  (full)", results_root, [256, 512])

    for label, path_dir, res_list in variants:
        path = path_dir / "inference_results.pt"
        if path.exists():
            r = torch.load(path, map_location="cpu", weights_only=True)
            m = r["metrics"]
            rmse_256 = m.get("rmse_posterior_256", float("nan")) if 256 in res_list else float("nan")
            rmse_512 = m.get("rmse_posterior_512", float("nan"))
            r256_str = f"{rmse_256:12.4f}" if not np.isnan(rmse_256) else f"{'N/A':>12}"
            print(f"  {label:<22}  {r256_str}  {rmse_512:12.4f}")
        else:
            print(f"  {label:<22}  {'not found':>12}  {'not found':>12}")
    print()


if __name__ == "__main__":
    main()
