"""
Ablation A1 (2D): Cascade depth sweep for Kraichnan turbulence.

Tests whether each additional refinement stage meaningfully reduces error.
Three variants — all using the SAME trained FNOs and diffusion model (no
retraining), so only the cascade structure differs:

  1-stage  — direct 32→256 in one step
             Uses FNO_256 (res_idx=2) from the shared diffusion model.
             Coarse input: spectral_upsample_2d(obs_32, 256) directly.
  2-stage  — 32→128→256 (skips the 32→64 intermediate)
             Uses FNO_128 (res_idx=1) and FNO_256 (res_idx=2).
  3-stage  — 32→64→128→256 — current default (baseline)

All three variants reuse the trained weights unchanged; only _STAGES differs.

Output (under results_2d/):
  results_2d/ablation_a1_1stage/inference_results.pt
  results_2d/ablation_a1_2stage/inference_results.pt
  results_2d/ablation_a1_3stage/inference_results.pt

Usage:
    python scripts/ablation_a1_2d.py
        [--config      configs/kraichnan.yaml]
        [--n_steps     N]
        [--skip_1stage]
        [--skip_2stage]
        [--skip_3stage]
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
from src.models.unet_2d import ConditionalUNet2d
from src.models.diffusion import GaussianDiffusion
from src.inference.pipeline_2d import IterativeRefinementPipeline2d
from src.data.dataset_2d import spectral_upsample_2d


# ---------------------------------------------------------------------------
# 1-stage pipeline: 32 → 256 directly
# ---------------------------------------------------------------------------

class OneStagePipeline2d(IterativeRefinementPipeline2d):
    """1-stage: 32→256 using FNO_256 and res_idx=2."""

    _STAGES = [(32, 256, 2)]

    @torch.no_grad()
    def run(self, observations_32: torch.Tensor, n_steps: int | None = None) -> dict:
        if observations_32.dim() == 4:
            observations_32 = observations_32.squeeze(1)
        obs = observations_32.to(self.device)
        T   = min(obs.shape[0], n_steps) if n_steps else obs.shape[0]
        obs = obs[:T]

        posteriors: list[torch.Tensor] = []
        forecasts:  list[torch.Tensor] = []

        w_init = self._upsample(obs[0:1], 256)
        prev_post_256 = w_init
        posteriors.append(w_init.squeeze().cpu())
        forecasts.append(w_init.squeeze().cpu())

        for t in tqdm(range(1, T), desc="1-stage 2D", ncols=80, leave=False):
            w_fc  = self._fno_forward(self.fnos[256], prev_post_256)
            w_co  = self._upsample(obs[t:t+1], 256)
            post  = self._ddim(w_fc, w_co, res_idx=2)
            prev_post_256 = post
            forecasts.append(w_fc.squeeze().cpu())
            posteriors.append(post.squeeze().cpu())

        return {
            "obs_32":        obs.cpu(),
            "posterior_256": torch.stack(posteriors, dim=0),
            "forecast_256":  torch.stack(forecasts,  dim=0),
        }


# ---------------------------------------------------------------------------
# 2-stage pipeline: 32 → 128 → 256
# ---------------------------------------------------------------------------

class TwoStagePipeline2d(IterativeRefinementPipeline2d):
    """2-stage: 32→128 (res_idx=1) then 128→256 (res_idx=2)."""

    _STAGES = [(32, 128, 1), (128, 256, 2)]

    @torch.no_grad()
    def run(self, observations_32: torch.Tensor, n_steps: int | None = None) -> dict:
        if observations_32.dim() == 4:
            observations_32 = observations_32.squeeze(1)
        obs = observations_32.to(self.device)
        T   = min(obs.shape[0], n_steps) if n_steps else obs.shape[0]
        obs = obs[:T]

        target_resolutions = [128, 256]
        posteriors: dict[int, list] = {r: [] for r in target_resolutions}
        forecasts:  dict[int, list] = {r: [] for r in target_resolutions}

        prev_post: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            w_init = self._upsample(obs[0:1], res)
            prev_post[res] = w_init
            posteriors[res].append(w_init.squeeze().cpu())
            forecasts[res].append(w_init.squeeze().cpu())

        for t in tqdm(range(1, T), desc="2-stage 2D", ncols=80, leave=False):
            obs_t = obs[t:t+1]
            stage_posts: dict[int, torch.Tensor] = {}

            for coarse_res, target_res, res_idx in self._STAGES:
                w_fc = self._fno_forward(self.fnos[target_res], prev_post[target_res])
                coarse_field = obs_t if coarse_res == 32 else stage_posts[coarse_res]
                w_co = self._upsample(coarse_field, target_res)
                post = self._ddim(w_fc, w_co, res_idx)
                stage_posts[target_res] = post
                forecasts[target_res].append(w_fc.squeeze().cpu())
                posteriors[target_res].append(post.squeeze().cpu())

            prev_post = {r: stage_posts[r] for r in target_resolutions}

        result: dict = {"obs_32": obs.cpu()}
        for res in target_resolutions:
            result[f"posterior_{res}"] = torch.stack(posteriors[res], dim=0)
            result[f"forecast_{res}"]  = torch.stack(forecasts[res],  dim=0)
        return result


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fnos(cfg, device: torch.device) -> dict[int, FNO2d]:
    fnos: dict[int, FNO2d] = {}
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    for res in [64, 128, 256]:
        model = FNO2d(cfg, res).to(device)
        ckpt  = torch.load(ckpt_dir / f"fno_{res}.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        fnos[res] = model
    print(f"  Loaded FNOs: {list(fnos.keys())}")
    return fnos


def _load_diffusion(cfg, device: torch.device) -> GaussianDiffusion:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt     = torch.load(ckpt_dir / "diffusion_ema.pt", map_location=device, weights_only=True)
    import src.models.diffusion as _dm
    _dm.ConditionalUNet1d = nn.Module
    unet = ConditionalUNet2d(cfg).to(device)
    unet.load_state_dict(ckpt["model"])
    unet.eval()
    diffusion = GaussianDiffusion(unet, cfg).to(device)
    print(f"  Loaded diffusion EMA (step {ckpt['step']})")
    return diffusion


def _rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Run one pipeline variant
# ---------------------------------------------------------------------------

def _run_and_save(
    pipeline,
    test_data: dict,
    traj_indices: list[int],
    n_steps: int,
    target_resolutions: list[int],
    out_dir: Path,
) -> dict:
    collectors: dict[int, list] = {r: [] for r in target_resolutions}
    truth_coll: dict[int, list] = {r: [] for r in target_resolutions}

    for traj_i in tqdm(traj_indices, desc="  Trajectories", ncols=80):
        obs_32 = test_data["w_32"][traj_i, :n_steps].float()
        res    = pipeline.run(obs_32, n_steps=n_steps)
        for r in target_resolutions:
            key = f"posterior_{r}"
            if key in res:
                collectors[r].append(res[key])
                truth_coll[r].append(test_data[f"w_{r}"][traj_i, :n_steps].float().cpu())

    results: dict = {}
    metrics: dict[str, float] = {}
    for r in target_resolutions:
        if collectors[r]:
            post  = torch.stack(collectors[r], dim=0)
            truth = torch.stack(truth_coll[r], dim=0)
            results[f"posterior_{r}"] = post
            results[f"truth_{r}"]     = truth
            rmse_val = _rmse(post, truth)
            metrics[f"rmse_posterior_{r}"] = rmse_val
            print(f"    {r}×{r}  RMSE={rmse_val:.4f}")
    results["metrics"] = metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "inference_results.pt")
    print(f"  Saved {out_dir / 'inference_results.pt'}")
    return results


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation A1 (2D): cascade depth sweep")
    p.add_argument("--config",      type=str,  default="configs/kraichnan.yaml")
    p.add_argument("--results_dir", type=str,  default=None,
                   help="Override cfg.paths.results_dir (e.g. results_2d_v2)")
    p.add_argument("--n_steps",     type=int,  default=None)
    p.add_argument("--skip_1stage", action="store_true")
    p.add_argument("--skip_2stage", action="store_true")
    p.add_argument("--skip_3stage", action="store_true")
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

    print("=" * 65)
    print("  Ablation A1 (2D) — Cascade depth sweep")
    print(f"  n_steps={n_steps}  device={device}")
    print("  All variants use the SAME trained FNOs + diffusion model")
    print("=" * 65)

    print("\nLoading models ...")
    fnos      = _load_fnos(cfg, device)
    diffusion = _load_diffusion(cfg, device)

    test_data    = torch.load(
        Path(cfg.data.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )
    n_test       = test_data["w_32"].shape[0]
    traj_indices = list(range(n_test))
    results_root = Path(args.results_dir) if args.results_dir else Path(cfg.paths.results_dir)

    # ── 1-stage ───────────────────────────────────────────────────────────────
    if args.skip_1stage:
        print("\n── Variant: 1-stage  (skipped)")
    else:
        print("\n── Variant: 1-stage  (32→256) ──────────────────────────────")
        p1 = OneStagePipeline2d(fnos, diffusion, cfg, device)
        _run_and_save(p1, test_data, traj_indices, n_steps,
                      target_resolutions=[256],
                      out_dir=results_root / "ablation_a1_1stage")

    # ── 2-stage ───────────────────────────────────────────────────────────────
    if args.skip_2stage:
        print("\n── Variant: 2-stage  (skipped)")
    else:
        print("\n── Variant: 2-stage  (32→128→256) ─────────────────────────")
        p2 = TwoStagePipeline2d(fnos, diffusion, cfg, device)
        _run_and_save(p2, test_data, traj_indices, n_steps,
                      target_resolutions=[128, 256],
                      out_dir=results_root / "ablation_a1_2stage")

    # ── 3-stage (baseline) ────────────────────────────────────────────────────
    if args.skip_3stage:
        print("\n── Variant: 3-stage  (skipped — using existing results_2d/)")
    else:
        print("\n── Variant: 3-stage  (32→64→128→256) [baseline] ───────────")
        p3 = IterativeRefinementPipeline2d(fnos, diffusion, cfg, device)
        _run_and_save(p3, test_data, traj_indices, n_steps,
                      target_resolutions=[64, 128, 256],
                      out_dir=results_root / "ablation_a1_3stage")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── A1 (2D) Summary ────────────────────────────────────────────")
    print(f"  {'Variant':<28}  {'RMSE@128':>10}  {'RMSE@256':>10}")
    print(f"  {'─'*52}")
    variants = [
        ("1-stage  (32→256)",       results_root / "ablation_a1_1stage",  [256]),
        ("2-stage  (32→128→256)",   results_root / "ablation_a1_2stage",  [128, 256]),
        ("3-stage  (full)",         results_root / "ablation_a1_3stage",  [128, 256]),
    ]
    if args.skip_3stage:
        variants[-1] = ("3-stage  (full)", results_root, [128, 256])

    for label, path_dir, res_list in variants:
        path = path_dir / "inference_results.pt"
        if path.exists():
            r = torch.load(path, map_location="cpu", weights_only=True)
            m = r["metrics"]
            r128 = m.get("rmse_posterior_128", float("nan")) if 128 in res_list else float("nan")
            r256 = m.get("rmse_posterior_256", float("nan"))
            r128s = f"{r128:10.4f}" if not np.isnan(r128) else f"{'N/A':>10}"
            r256s = f"{r256:10.4f}" if not np.isnan(r256) else f"{'N/A':>10}"
            print(f"  {label:<28}  {r128s}  {r256s}")
        else:
            print(f"  {label:<28}  {'not found':>10}")
    print()


if __name__ == "__main__":
    main()
