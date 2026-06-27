"""
Ablation A2 (2D): Propagation signal into each stage for Kraichnan turbulence.

Tests what feeds as the coarse conditioning input to stages r=1 and r=2
(stage r=0 always uses the raw 32×32 observation — that is fixed).

Three variants on the trained 2D 3-stage pipeline (no retraining):

  (i)  posterior  [current/default]
       coarse input to stage r+1 = diffusion posterior from stage r

  (ii) forecast
       coarse input to stage r+1 = FNO forecast at the coarser resolution
       (already at coarse_res from the FNO output)

  (iii) obs_raw
       coarse input to stage r+1 = spectral_downsample_2d of the 32×32 obs
       to coarse_res, then upsample to target_res

Output (under results_2d/):
  results_2d/ablation_a2_posterior/inference_results.pt
  results_2d/ablation_a2_forecast/inference_results.pt
  results_2d/ablation_a2_obs_raw/inference_results.pt

Usage:
    python scripts/ablation_a2_2d.py
        [--config   configs/kraichnan.yaml]
        [--n_steps  N]
        [--variants posterior,forecast,obs_raw]
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
# Variant pipelines
# ---------------------------------------------------------------------------

class _PropagationPipeline2d(IterativeRefinementPipeline2d):
    """Base class — overrides only coarse-field selection."""

    _propagation: str = "posterior"

    @torch.no_grad()
    def run(self, observations_32: torch.Tensor, n_steps: int | None = None) -> dict:
        if observations_32.dim() == 4:
            observations_32 = observations_32.squeeze(1)
        obs = observations_32.to(self.device)
        T   = min(obs.shape[0], n_steps) if n_steps else obs.shape[0]
        obs = obs[:T]

        target_resolutions = [64, 128, 256]
        posteriors: dict[int, list] = {r: [] for r in target_resolutions}
        forecasts:  dict[int, list] = {r: [] for r in target_resolutions}

        prev_post: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            w_init = self._upsample(obs[0:1], res)
            prev_post[res] = w_init
            posteriors[res].append(w_init.squeeze().cpu())
            forecasts[res].append(w_init.squeeze().cpu())

        # Track FNO forecasts at each res for "forecast" variant
        stage_forecasts: dict[int, torch.Tensor] = {}

        for t in tqdm(range(1, T),
                      desc=f"  A2-2D/{self._propagation}", ncols=80, leave=False):
            obs_t = obs[t:t+1]
            stage_posts: dict[int, torch.Tensor] = {}
            stage_forecasts = {}

            for coarse_res, target_res, res_idx in self._STAGES:
                w_fc = self._fno_forward(self.fnos[target_res], prev_post[target_res])
                stage_forecasts[target_res] = w_fc

                if coarse_res == 32:
                    coarse_field = obs_t
                else:
                    if self._propagation == "posterior":
                        coarse_field = stage_posts[coarse_res]

                    elif self._propagation == "forecast":
                        # FNO at coarse_res is already at coarse_res spatial dims
                        coarse_field = stage_forecasts[coarse_res]

                    elif self._propagation == "obs_raw":
                        # 32×32 obs is coarser than coarse_res (64/128) — spectrally
                        # upsample directly to coarse_res (a no-op truncation, since
                        # coarse_res > 32 already contains all of the obs's modes).
                        coarse_field = self._upsample(obs_t, coarse_res)

                    else:
                        raise ValueError(f"Unknown propagation: {self._propagation}")

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


class PosteriorPropagation2d(_PropagationPipeline2d):
    _propagation = "posterior"


class ForecastPropagation2d(_PropagationPipeline2d):
    _propagation = "forecast"


class ObsRawPropagation2d(_PropagationPipeline2d):
    _propagation = "obs_raw"


_VARIANT_CLASSES = {
    "posterior": PosteriorPropagation2d,
    "forecast":  ForecastPropagation2d,
    "obs_raw":   ObsRawPropagation2d,
}


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
# Run one variant
# ---------------------------------------------------------------------------

def _run_variant(
    variant: str,
    fnos: dict,
    diffusion: GaussianDiffusion,
    cfg,
    test_data: dict,
    traj_indices: list[int],
    n_steps: int,
    results_root: Path,
    device: torch.device,
) -> None:
    PipelineClass = _VARIANT_CLASSES[variant]
    pipeline = PipelineClass(fnos, diffusion, cfg, device)

    target_resolutions = [64, 128, 256]
    all_post:  dict[int, list] = {r: [] for r in target_resolutions}
    all_truth: dict[int, list] = {r: [] for r in target_resolutions}

    for traj_i in tqdm(traj_indices, desc=f"Trajectories ({variant})", ncols=80):
        obs_32 = test_data["w_32"][traj_i, :n_steps].float()
        res    = pipeline.run(obs_32, n_steps=n_steps)
        for r in target_resolutions:
            all_post[r].append(res[f"posterior_{r}"])
            all_truth[r].append(test_data[f"w_{r}"][traj_i, :n_steps].float().cpu())

    results: dict = {}
    metrics: dict[str, float] = {}
    for r in target_resolutions:
        post  = torch.stack(all_post[r],  dim=0)
        truth = torch.stack(all_truth[r], dim=0)
        results[f"posterior_{r}"] = post
        results[f"truth_{r}"]     = truth
        rmse_val = _rmse(post, truth)
        metrics[f"rmse_posterior_{r}"] = rmse_val
        print(f"    {r}×{r}  RMSE={rmse_val:.4f}")
    results["metrics"] = metrics

    out_dir = results_root / f"ablation_a2_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "inference_results.pt")
    print(f"  Saved {out_dir / 'inference_results.pt'}")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation A2 (2D): propagation signal sweep")
    p.add_argument("--config",   type=str, default="configs/kraichnan.yaml")
    p.add_argument("--n_steps",  type=int, default=None)
    p.add_argument("--variants", type=str, default="posterior,forecast,obs_raw")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args     = parse_args()
    cfg      = OmegaConf.load(args.config)
    variants = [v.strip() for v in args.variants.split(",")]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if str(cfg.device) != "auto":
        device_str = str(cfg.device)
    device = torch.device(device_str)

    n_steps = args.n_steps if args.n_steps is not None else int(cfg.inference.test_time_steps)

    print("=" * 65)
    print("  Ablation A2 (2D) — Propagation signal sweep")
    print(f"  variants={variants}  n_steps={n_steps}  device={device}")
    print("=" * 65)

    print("\nLoading models ...")
    fnos      = _load_fnos(cfg, device)
    diffusion = _load_diffusion(cfg, device)

    test_data    = torch.load(
        Path(cfg.data.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )
    n_test       = test_data["w_32"].shape[0]
    traj_indices = list(range(n_test))
    results_root = Path(cfg.paths.results_dir)

    for variant in variants:
        if variant not in _VARIANT_CLASSES:
            print(f"  [SKIP] Unknown variant '{variant}'")
            continue
        print(f"\n── Variant: {variant} {'─' * (40 - len(variant))}")
        _run_variant(variant, fnos, diffusion, cfg, test_data,
                     traj_indices, n_steps, results_root, device)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── A2 (2D) Summary ─────────────────────────────────────────────")
    print(f"  {'Variant':<18}  {'RMSE@64':>10}  {'RMSE@128':>10}  {'RMSE@256':>10}")
    print(f"  {'─'*54}")
    for variant in ["posterior", "forecast", "obs_raw"]:
        path = results_root / f"ablation_a2_{variant}" / "inference_results.pt"
        if path.exists():
            r = torch.load(path, map_location="cpu", weights_only=True)
            m = r["metrics"]
            r64  = m.get("rmse_posterior_64",  float("nan"))
            r128 = m.get("rmse_posterior_128", float("nan"))
            r256 = m.get("rmse_posterior_256", float("nan"))
            idx  = ["posterior", "forecast", "obs_raw"].index(variant)
            label = f"({'i'*idx+'ii'[idx>0:][:1] if idx else 'i'}) {variant}"
            label = f"({'i' if idx==0 else 'ii' if idx==1 else 'iii'}) {variant}"
            print(f"  {label:<18}  {r64:10.4f}  {r128:10.4f}  {r256:10.4f}")
        else:
            print(f"  {variant:<18}  {'not run':>10}")
    print()


if __name__ == "__main__":
    main()
