"""
Iterative refinement inference pipeline.

For each time step t, the pipeline cascades through 3 resolution stages:

  Stage 0  (64 -> 128, res_idx=0):
    forecast_128  = FNO_128( prev_posterior_128 )
    coarse_up_128 = spectral_upsample( obs_64_t, 128 )
    posterior_128 = DDIM( x_noisy=noise, cond=[forecast_128, coarse_up_128], r=0 )

  Stage 1  (128 -> 256, res_idx=1):
    forecast_256  = FNO_256( prev_posterior_256 )
    coarse_up_256 = spectral_upsample( posterior_128_t, 256 )   <- stage 0 output
    posterior_256 = DDIM( x_noisy=noise, cond=[forecast_256, coarse_up_256], r=1 )

  Stage 2  (256 -> 512, res_idx=2):
    forecast_512  = FNO_512( prev_posterior_512 )
    coarse_up_512 = spectral_upsample( posterior_256_t, 512 )   <- stage 1 output
    posterior_512 = DDIM( x_noisy=noise, cond=[forecast_512, coarse_up_512], r=2 )

At t=0 there is no previous posterior, so each resolution is initialised by
spectrally upsampling the t=0 coarse observation.

Always use EMA weights for the diffusion model (loaded before calling run()).
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.fno import FNO1d
from src.models.diffusion import GaussianDiffusion
from src.data.solver import spectral_upsample


class IterativeRefinementPipeline:
    """Resolution-cascaded data assimilation pipeline.

    Args:
        fnos:      Dict {resolution: FNO1d} with trained FNO models for
                   resolutions [128, 256, 512].
        diffusion: GaussianDiffusion with EMA weights already loaded into
                   the underlying U-Net (call ema.copy_to(unet) before passing).
        cfg:       Full config object.
        device:    Compute device.
    """

    _STAGES: list[tuple[int, int, int]] = [
        # (coarse_res, target_res, res_idx)
        (64,  128, 0),
        (128, 256, 1),
        (256, 512, 2),
    ]

    def __init__(
        self,
        fnos: dict[int, FNO1d],
        diffusion: GaussianDiffusion,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.fnos      = {r: m.to(device).eval() for r, m in fnos.items()}
        self.diffusion = diffusion.to(device).eval()
        self.cfg       = cfg
        self.device    = device
        self.ddim_steps = int(cfg.inference.ddim_steps)
        self.eta        = float(cfg.inference.eta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fno_forward(self, model: FNO1d, u: torch.Tensor) -> torch.Tensor:
        """Run FNO on u, ensuring (B, 1, N) shape."""
        if u.dim() == 2:          # (B, N) -> (B, 1, N)
            u = u.unsqueeze(1)
        return model(u)            # (B, 1, N)

    def _ddim(
        self,
        u_forecast: torch.Tensor,
        u_coarse_up: torch.Tensor,
        res_idx: int,
    ) -> torch.Tensor:
        """Run DDIM sampling for one stage.

        Args:
            u_forecast:  (B, 1, N)
            u_coarse_up: (B, 1, N)
            res_idx:     integer in {0, 1, 2}

        Returns:
            posterior: (B, 1, N)
        """
        B = u_forecast.shape[0]
        res_idx_t = torch.full((B,), res_idx, dtype=torch.long, device=self.device)
        return self.diffusion.ddim_sample(
            u_forecast, u_coarse_up, res_idx_t,
            ddim_steps=self.ddim_steps,
            eta=self.eta,
        )  # (B, 1, N)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        observations_64: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run iterative refinement over a trajectory of coarse observations.

        Args:
            observations_64: Coarse observations, shape (T, 64) or (T, 1, 64).
            n_steps:         Number of time steps to process (default: all T).

        Returns:
            Dict with keys for every stored quantity:
              'posterior_{N}'  : (T, N)  diffusion posterior at resolution N
              'forecast_{N}'   : (T, N)  FNO forecast at resolution N
              'obs_64'         : (T, 64) the input coarse observations (copy)
            where N ∈ {128, 256, 512}.
        """
        # Normalise shape to (T, 64)
        if observations_64.dim() == 3:
            observations_64 = observations_64.squeeze(1)
        obs = observations_64.to(self.device)            # (T, 64)
        T   = obs.shape[0]
        if n_steps is not None:
            T = min(T, n_steps)
            obs = obs[:T]

        # Storage — collect (T, N) tensors
        target_resolutions = [128, 256, 512]
        posteriors: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}
        forecasts:  dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}

        # t=0: initialise posteriors by upsampling the first observation
        prev_post: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            u_init = spectral_upsample(obs[0:1], res)   # (1, res)  on device
            prev_post[res] = u_init.unsqueeze(1)         # (1, 1, res)
            # Record a "forecast" = upsample (no FNO at t=0) and posterior = upsample
            forecasts[res].append(u_init.squeeze(0).cpu())   # (res,)
            posteriors[res].append(u_init.squeeze(0).cpu())  # (res,)

        # t=1 .. T-1
        for t in tqdm(range(1, T), desc="Inference steps", ncols=80, leave=False):
            obs_t = obs[t:t+1]              # (1, 64)
            # Cascade through stages
            stage_posts: dict[int, torch.Tensor] = {}

            for coarse_res, target_res, res_idx in self._STAGES:
                # FNO forecast from previous posterior at target_res
                u_fc = self._fno_forward(
                    self.fnos[target_res], prev_post[target_res]
                )  # (1, 1, target_res)

                # Coarse-up: upsample the appropriate coarser field
                if coarse_res == 64:
                    coarse_field = obs_t                   # (1, 64)
                else:
                    # Use the just-computed stage posterior at coarser res
                    coarse_field = stage_posts[coarse_res].squeeze(1)  # (1, coarse_res)

                u_co = spectral_upsample(
                    coarse_field, target_res
                ).unsqueeze(1)             # (1, 1, target_res)

                # DDIM correction
                posterior = self._ddim(u_fc, u_co, res_idx)  # (1, 1, target_res)

                stage_posts[target_res] = posterior
                forecasts[target_res].append(u_fc.squeeze().cpu())       # (target_res,)
                posteriors[target_res].append(posterior.squeeze().cpu())  # (target_res,)

            # Update previous posteriors for next step
            prev_post = {r: stage_posts[r] for r in target_resolutions}

        # Stack into tensors (T, N)
        result: dict[str, torch.Tensor] = {
            "obs_64": obs.cpu(),
        }
        for res in target_resolutions:
            result[f"posterior_{res}"] = torch.stack(posteriors[res], dim=0)  # (T, res)
            result[f"forecast_{res}"]  = torch.stack(forecasts[res],  dim=0)  # (T, res)

        return result

    @torch.no_grad()
    def run_fno_only(
        self,
        observations_64: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """FNO-only baseline: no diffusion correction.

        Autoregressively applies each FNO at its resolution, using the
        spectrally upsampled coarse observation as the very first state.

        Args:
            observations_64: Coarse observations, shape (T, 64) or (T, 1, 64).
            n_steps:         Number of time steps to process.

        Returns:
            Dict with keys 'fno_only_{N}' for N ∈ {128, 256, 512}, each (T, N).
        """
        if observations_64.dim() == 3:
            observations_64 = observations_64.squeeze(1)
        obs = observations_64.to(self.device)
        T   = obs.shape[0]
        if n_steps is not None:
            T = min(T, n_steps)
            obs = obs[:T]

        target_resolutions = [128, 256, 512]
        fno_preds: dict[int, list[torch.Tensor]] = {r: [] for r in target_resolutions}

        # Initialise from t=0 upsampled observation
        prev: dict[int, torch.Tensor] = {}
        for res in target_resolutions:
            u_init = spectral_upsample(obs[0:1], res).unsqueeze(1)  # (1,1,res) on device
            prev[res] = u_init
            fno_preds[res].append(u_init.squeeze(0).squeeze(0).cpu())  # (res,)

        for t in tqdm(range(1, T), desc="FNO-only inference", ncols=80, leave=False):
            for res in target_resolutions:
                u_fc = self._fno_forward(self.fnos[res], prev[res])  # (1,1,res)
                prev[res] = u_fc
                fno_preds[res].append(u_fc.squeeze().cpu())

        result: dict[str, torch.Tensor] = {}
        for res in target_resolutions:
            result[f"fno_only_{res}"] = torch.stack(fno_preds[res], dim=0)  # (T, res)
        return result
