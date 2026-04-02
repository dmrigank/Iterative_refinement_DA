"""
2D Iterative Refinement inference pipeline for Kraichnan turbulence.

For each time step t, the pipeline cascades through 3 resolution stages:

  Stage 0  (32→64,   res_idx=0):
    forecast_64   = FNO_64(  prev_posterior_64  )
    coarse_up_64  = spectral_upsample_2d( obs_32_t,  64,  64 )
    posterior_64  = DDIM( cond=[forecast_64,  coarse_up_64],  r=0 )

  Stage 1  (64→128,  res_idx=1):
    forecast_128  = FNO_128( prev_posterior_128 )
    coarse_up_128 = spectral_upsample_2d( posterior_64_t, 128, 128 )
    posterior_128 = DDIM( cond=[forecast_128, coarse_up_128], r=1 )

  Stage 2  (128→256, res_idx=2):
    forecast_256  = FNO_256( prev_posterior_256 )
    coarse_up_256 = spectral_upsample_2d( posterior_128_t, 256, 256 )
    posterior_256 = DDIM( cond=[forecast_256, coarse_up_256], r=2 )

At t=0 there is no previous posterior; each resolution is initialised by
spectrally upsampling the first coarse observation.

Always use EMA weights for the diffusion model (loaded before constructing
the pipeline).
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.fno_2d import FNO2d
from src.models.diffusion import GaussianDiffusion
from src.data.dataset_2d import spectral_upsample_2d


class IterativeRefinementPipeline2d:
    """Resolution-cascaded 2D data assimilation pipeline.

    Args:
        fnos:      Dict {resolution: FNO2d} for resolutions [64, 128, 256].
        diffusion: GaussianDiffusion with EMA weights already loaded into the
                   underlying ConditionalUNet2d.
        cfg:       Full kraichnan.yaml config object.
        device:    Compute device.
    """

    _STAGES: list[tuple[int, int, int]] = [
        # (coarse_res, target_res, res_idx)
        (32,   64, 0),
        (64,  128, 1),
        (128, 256, 2),
    ]

    def __init__(
        self,
        fnos: dict[int, FNO2d],
        diffusion: GaussianDiffusion,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.fnos       = {r: m.to(device).eval() for r, m in fnos.items()}
        self.diffusion  = diffusion.to(device).eval()
        self.cfg        = cfg
        self.device     = device
        self.ddim_steps = int(cfg.inference.ddim_steps)
        self.eta        = float(cfg.inference.eta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fno_forward(self, model: FNO2d, w: torch.Tensor) -> torch.Tensor:
        """Run FNO, ensuring (B, 1, ny, nx) input/output."""
        if w.dim() == 3:       # (B, ny, nx) -> (B, 1, ny, nx)
            w = w.unsqueeze(1)
        return model(w)        # (B, 1, ny, nx)

    def _upsample(self, w: torch.Tensor, target: int) -> torch.Tensor:
        """Spectrally upsample w to (target × target), returning (B, 1, target, target)."""
        if w.dim() == 4:       # (B, 1, ny, nx) -> (B, ny, nx)
            w = w.squeeze(1)
        up = spectral_upsample_2d(w, target_ny=target, target_nx=target)  # (B, target, target)
        return up.unsqueeze(1) # (B, 1, target, target)

    def _ddim(
        self,
        w_forecast: torch.Tensor,
        w_coarse_up: torch.Tensor,
        res_idx: int,
    ) -> torch.Tensor:
        """Run DDIM for one stage. Returns (B, 1, ny, nx)."""
        B = w_forecast.shape[0]
        res_idx_t = torch.full((B,), res_idx, dtype=torch.long, device=self.device)
        return self.diffusion.ddim_sample(
            w_forecast, w_coarse_up, res_idx_t,
            ddim_steps=self.ddim_steps,
            eta=self.eta,
        )  # (B, 1, ny, nx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        observations_32: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run iterative refinement over a trajectory of coarse observations.

        Args:
            observations_32: Coarse vorticity, shape (T, 32, 32) or (T, 1, 32, 32).
            n_steps:         Number of time steps to process (default: all T).

        Returns:
            Dict with keys:
              'posterior_{N}'  : (T, N, N)  diffusion posterior at resolution N
              'forecast_{N}'   : (T, N, N)  FNO one-step forecast at resolution N
              'obs_32'         : (T, 32, 32) input coarse observations
            where N ∈ {64, 128, 256}.
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

        # t=0: initialise each resolution by upsampling the first observation
        prev_post: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            w_init = self._upsample(obs[0:1], res)          # (1, 1, res, res)
            prev_post[res] = w_init
            # No FNO at t=0 — record upsampled obs as both forecast and posterior
            forecasts[res].append(w_init.squeeze().cpu())   # (res, res)
            posteriors[res].append(w_init.squeeze().cpu())  # (res, res)

        # t=1 .. T-1
        for t in tqdm(range(1, T), desc="Inference steps", ncols=90, leave=False):
            obs_t = obs[t:t+1]              # (1, 32, 32)
            stage_posts: dict[int, torch.Tensor] = {}

            for coarse_res, target_res, res_idx in self._STAGES:
                # FNO forecast from previous posterior
                w_fc = self._fno_forward(
                    self.fnos[target_res], prev_post[target_res]
                )  # (1, 1, target_res, target_res)

                # Coarse input: raw 32×32 obs (stage 0) or previous stage posterior
                if coarse_res == 32:
                    coarse_field = obs_t                    # (1, 32, 32)
                else:
                    coarse_field = stage_posts[coarse_res] # (1, 1, coarse_res, coarse_res)

                w_co = self._upsample(coarse_field, target_res)  # (1, 1, target_res, target_res)

                # DDIM correction
                posterior = self._ddim(w_fc, w_co, res_idx)     # (1, 1, target_res, target_res)

                stage_posts[target_res] = posterior
                forecasts[target_res].append(w_fc.squeeze().cpu())
                posteriors[target_res].append(posterior.squeeze().cpu())

            prev_post = {r: stage_posts[r] for r in target_resolutions}

        result: dict[str, torch.Tensor] = {"obs_32": obs.cpu()}
        for res in target_resolutions:
            result[f"posterior_{res}"] = torch.stack(posteriors[res], dim=0)  # (T, res, res)
            result[f"forecast_{res}"]  = torch.stack(forecasts[res],  dim=0)  # (T, res, res)

        return result

    @torch.no_grad()
    def run_fno_only(
        self,
        observations_32: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """FNO-only autoregressive baseline (no diffusion correction).

        Each FNO is initialised from the spectrally upsampled first coarse
        observation and then run autoregressively, ignoring subsequent obs.

        Args:
            observations_32: Coarse vorticity, shape (T, 32, 32) or (T, 1, 32, 32).
            n_steps:         Number of time steps.

        Returns:
            Dict with keys 'fno_only_{N}' for N ∈ {64, 128, 256}, each (T, N, N).
        """
        if observations_32.dim() == 4:
            observations_32 = observations_32.squeeze(1)
        obs = observations_32.to(self.device)
        T   = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        target_resolutions = [64, 128, 256]
        fno_preds: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}

        prev: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            w_init = self._upsample(obs[0:1], res)          # (1, 1, res, res)
            prev[res] = w_init
            fno_preds[res].append(w_init.squeeze().cpu())   # (res, res)

        for t in tqdm(range(1, T), desc="FNO-only inference", ncols=90, leave=False):
            for res in target_resolutions:
                w_fc = self._fno_forward(self.fnos[res], prev[res])  # (1, 1, res, res)
                prev[res] = w_fc
                fno_preds[res].append(w_fc.squeeze().cpu())

        result: dict[str, torch.Tensor] = {}
        for res in target_resolutions:
            result[f"fno_only_{res}"] = torch.stack(fno_preds[res], dim=0)  # (T, res, res)
        return result
