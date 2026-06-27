"""
Ablation A2: Propagation signal into each stage.

Tests what feeds as the coarse conditioning input to stages r=1 and r=2
(stage r=0 always uses the raw 64-pt observation — that is fixed).

Three variants on the trained 1D 3-stage pipeline (no retraining):

  (i)  posterior  [current/default]
       coarse input to stage r+1 = diffusion posterior from stage r
       → tests whether the iterative posterior refinement is beneficial

  (ii) forecast
       coarse input to stage r+1 = FNO forecast at the coarser resolution
       (spectral-downsampled to coarse_res before upsampling to target_res)
       → tests whether the FNO forecast alone suffices as the propagation signal

  (iii) obs_raw
       coarse input to stage r+1 = spectral_downsample of the original
       64-pt observation, upsampled to target_res
       → the "always use the raw obs" baseline; ignores all higher-res info

Hypothesis: variant (i) should win — the diffusion posterior carries
  corrected fine-scale structure that compounds beneficially across stages.
  Variant (iii) is a hard lower bound: it discards all high-res information.

Output directories (under results/):
  results/ablation_a2_posterior/inference_results.pt   (re-run of default)
  results/ablation_a2_forecast/inference_results.pt
  results/ablation_a2_obs_raw/inference_results.pt

Usage:
    python scripts/ablation_a2_propagation_signal.py
        [--config  configs/default.yaml]
        [--n_steps N]
        [--variants posterior,forecast,obs_raw]
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
from src.data.solver import spectral_upsample, spectral_downsample


# ---------------------------------------------------------------------------
# Variant pipelines — each overrides only the coarse-field selection logic
# ---------------------------------------------------------------------------

class _PropagationPipeline(IterativeRefinementPipeline):
    """Base class that exposes the coarse-field selection as a hook."""

    #: set by subclasses — "posterior" | "forecast" | "obs_raw"
    _propagation: str = "posterior"

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

        target_resolutions = [128, 256, 512]
        posteriors: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
        forecasts:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}

        prev_post: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            u_init = spectral_upsample(obs[0:1], res).unsqueeze(1)
            prev_post[res] = u_init
            forecasts[res].append(u_init.squeeze(0).squeeze(0).cpu())
            posteriors[res].append(u_init.squeeze(0).squeeze(0).cpu())

        # Also track the FNO forecasts at intermediate resolutions
        # so the "forecast" variant can use them as propagation signals.
        stage_forecasts: dict[int, torch.Tensor] = {}

        for t in tqdm(range(1, T),
                      desc=f"  A2/{self._propagation}", ncols=80, leave=False):
            obs_t = obs[t:t+1]
            stage_posts: dict[int, torch.Tensor] = {}
            stage_forecasts = {}

            for coarse_res, target_res, res_idx in self._STAGES:
                u_fc = self._fno_forward(self.fnos[target_res], prev_post[target_res])
                stage_forecasts[target_res] = u_fc

                # ── Coarse-field selection: the only thing that differs ──────
                if coarse_res == 64:
                    # Stage 0: always use raw obs — never varied
                    coarse_field = obs_t           # (1, 64)
                else:
                    if self._propagation == "posterior":
                        # (i) current: use diffusion posterior at coarser res
                        coarse_field = stage_posts[coarse_res].squeeze(1)

                    elif self._propagation == "forecast":
                        # (ii) use FNO forecast at coarser res, downsampled to coarse_res
                        # stage_forecasts[coarse_res] is (1,1,coarse_res) already
                        coarse_field = stage_forecasts[coarse_res].squeeze(1)  # (1, coarse_res)

                    elif self._propagation == "obs_raw":
                        # (iii) spectral downsample of original 64-pt obs to coarse_res
                        coarse_field = spectral_downsample(obs_t, coarse_res)  # (1, coarse_res)

                    else:
                        raise ValueError(f"Unknown propagation: {self._propagation}")
                # ──────────────────────────────────────────────────────────────

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


class PosteriorPropagation(_PropagationPipeline):
    """Variant (i): coarse input = diffusion posterior from previous stage [default]."""
    _propagation = "posterior"


class ForecastPropagation(_PropagationPipeline):
    """Variant (ii): coarse input = FNO forecast at previous resolution."""
    _propagation = "forecast"


class ObsRawPropagation(_PropagationPipeline):
    """Variant (iii): coarse input = spectral downsample of original 64-pt obs."""
    _propagation = "obs_raw"


_VARIANT_CLASSES = {
    "posterior": PosteriorPropagation,
    "forecast":  ForecastPropagation,
    "obs_raw":   ObsRawPropagation,
}


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
# Run one variant and save
# ---------------------------------------------------------------------------

def _run_variant(
    variant: str,
    fnos: dict[int, FNO1d],
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

    all_post:  dict[int, list] = {r: [] for r in [128, 256, 512]}
    all_truth: dict[int, list] = {r: [] for r in [128, 256, 512]}
    obs_coll: list[torch.Tensor] = []

    for traj_i in tqdm(traj_indices, desc=f"Trajectories ({variant})", ncols=80):
        obs_64 = test_data["u_64"][traj_i, :n_steps, :]
        res    = pipeline.run(obs_64, n_steps=n_steps)
        obs_coll.append(res["obs_64"])
        for r in [128, 256, 512]:
            all_post[r].append(res[f"posterior_{r}"])
            all_truth[r].append(test_data[f"u_{r}"][traj_i, :n_steps, :].cpu())

    results: dict = {"obs_64": torch.stack(obs_coll, dim=0)}
    metrics: dict[str, float] = {}

    for r in [128, 256, 512]:
        post  = torch.stack(all_post[r],  dim=0)
        truth = torch.stack(all_truth[r], dim=0)
        results[f"posterior_{r}"] = post
        results[f"truth_{r}"]     = truth
        rmse_val = _rmse(post, truth)
        metrics[f"rmse_posterior_{r}"] = rmse_val
        print(f"    N={r:4d}  RMSE={rmse_val:.4f}")
    results["metrics"] = metrics

    out_dir = results_root / f"ablation_a2_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "inference_results.pt")
    print(f"  Saved {out_dir / 'inference_results.pt'}")


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation A2: propagation signal sweep")
    p.add_argument("--config",   type=str, default="configs/default.yaml")
    p.add_argument("--n_steps",  type=int, default=None)
    p.add_argument("--variants", type=str, default="posterior,forecast,obs_raw",
                   help="Comma-separated subset of variants to run "
                        "(default: all three)")
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

    print("=" * 60)
    print("  Ablation A2 — Propagation signal sweep")
    print(f"  variants={variants}  n_steps={n_steps}  device={device}")
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

    for variant in variants:
        if variant not in _VARIANT_CLASSES:
            print(f"  [SKIP] Unknown variant '{variant}' — choose from {list(_VARIANT_CLASSES)}")
            continue
        print(f"\n── Variant: {variant} {'─'*(40 - len(variant))}")
        _run_variant(variant, fnos, diffusion, cfg, test_data,
                     traj_indices, n_steps, results_root, device)

    # ── Print summary table ─────────────────────────────────────────────────
    print("\n── A2 Summary ─────────────────────────────────────────────────")
    print(f"  {'Variant':<18}  {'N=128 RMSE':>12}  {'N=256 RMSE':>12}  {'N=512 RMSE':>12}")
    print(f"  {'-'*60}")
    for variant in ["posterior", "forecast", "obs_raw"]:
        path = results_root / f"ablation_a2_{variant}" / "inference_results.pt"
        if path.exists():
            r = torch.load(path, map_location="cpu", weights_only=True)
            m = r["metrics"]
            r128 = m.get("rmse_posterior_128", float("nan"))
            r256 = m.get("rmse_posterior_256", float("nan"))
            r512 = m.get("rmse_posterior_512", float("nan"))
            label = f"({['i','ii','iii'][['posterior','forecast','obs_raw'].index(variant)]}) {variant}"
            print(f"  {label:<18}  {r128:12.4f}  {r256:12.4f}  {r512:12.4f}")
        else:
            print(f"  {variant:<18}  {'not run':>12}")
    print()


if __name__ == "__main__":
    main()
