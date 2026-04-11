"""
One-shot diffusion SR inference pipeline for Kraichnan turbulence.

For each time step t >= 1:
  w_obs_up_{t}  = spectral_upsample_2d(obs_32_t, 256, 256)
  w_prev        = posterior_{t-1}  (own previous output — autoregressive)
  posterior_t   = DDIM( cond=[w_prev, w_obs_up_t] )

At t=0: no previous posterior available.
  prev_256 = spectral_upsample_2d(obs_32[0], 256, 256)  (consistent with iterative pipeline)

The one-shot model has NO FNO forecaster. The "prior" is entirely the previous
posterior combined with the current upsampled coarse observation.

Also provides a run_bicubic_baseline method: spectrally upsample each 32×32
observation to 256×256 independently (no temporal info, no learning).
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.diffusion import GaussianDiffusion
from src.data.dataset_2d import spectral_upsample_2d


class OneShotPipeline:
    """One-shot 256×256 SR pipeline (autoregressive in time).

    Args:
        diffusion: GaussianDiffusion wrapping OneShotUNet2d, EMA weights loaded.
        cfg:       oneshot_sr.yaml config object.
        device:    Compute device.
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.diffusion  = diffusion.to(device).eval()
        self.cfg        = cfg
        self.device     = device
        self.ddim_steps = int(cfg.inference.ddim_steps)
        self.eta        = float(cfg.inference.eta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsample(self, w: torch.Tensor) -> torch.Tensor:
        """Spectrally upsample (B, ny, nx) or (B, 1, ny, nx) → (B, 1, 256, 256)."""
        if w.dim() == 4:
            w = w.squeeze(1)                                       # (B, ny, nx)
        up = spectral_upsample_2d(w, target_ny=256, target_nx=256)  # (B, 256, 256)
        return up.unsqueeze(1)                                     # (B, 1, 256, 256)

    def _ddim(
        self,
        w_prev: torch.Tensor,    # (B, 1, 256, 256)
        w_obs_up: torch.Tensor,  # (B, 1, 256, 256)
    ) -> torch.Tensor:
        """Run DDIM for one time step. Returns (B, 1, 256, 256)."""
        B = w_prev.shape[0]
        res_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        return self.diffusion.ddim_sample(
            w_prev, w_obs_up, res_idx,
            ddim_steps=self.ddim_steps,
            eta=self.eta,
        )  # (B, 1, 256, 256)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        observations_32: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one-shot SR autoregressively over a trajectory.

        Args:
            observations_32: Coarse vorticity, shape (T, 32, 32) or (T, 1, 32, 32).
            n_steps:         Number of time steps to process (default: all T).

        Returns:
            Dict with keys:
              'posterior_256': (T, 256, 256) — diffusion posterior
              'obs_32':        (T, 32, 32)   — coarse observations
        """
        if observations_32.dim() == 4:
            observations_32 = observations_32.squeeze(1)

        obs = observations_32.to(self.device)   # (T, 32, 32)
        T = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        posteriors: list[torch.Tensor] = []

        # t=0: no previous posterior — init from spectrally upsampled first obs
        w_prev = self._upsample(obs[0:1])       # (1, 1, 256, 256)
        posteriors.append(w_prev.squeeze().cpu())  # (256, 256)

        for t in tqdm(range(1, T), desc="OneShotInference", ncols=90, leave=False):
            w_obs_up  = self._upsample(obs[t:t+1])        # (1, 1, 256, 256)
            posterior = self._ddim(w_prev, w_obs_up)       # (1, 1, 256, 256)
            posteriors.append(posterior.squeeze().cpu())   # (256, 256)
            w_prev = posterior                             # autoregressive

        return {
            "posterior_256": torch.stack(posteriors, dim=0),  # (T, 256, 256)
            "obs_32":        obs.cpu(),                        # (T, 32, 32)
        }

    @torch.no_grad()
    def run_bicubic_baseline(
        self,
        observations_32: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Spectral upsample baseline: each 32×32 obs → 256×256 independently.

        No temporal information, no learned model. Serves as a trivial lower bound.

        Args:
            observations_32: (T, 32, 32) or (T, 1, 32, 32)
            n_steps:         Number of time steps (default: all T)

        Returns:
            Dict with key 'bicubic_256': (T, 256, 256)
        """
        if observations_32.dim() == 4:
            observations_32 = observations_32.squeeze(1)

        obs = observations_32.to(self.device)
        T = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        # Process in one batch call — spectral upsample is parallelisable
        up = spectral_upsample_2d(obs, target_ny=256, target_nx=256)  # (T, 256, 256)
        return {"bicubic_256": up.cpu()}
