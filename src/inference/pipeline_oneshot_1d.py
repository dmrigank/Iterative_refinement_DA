"""
1D one-shot diffusion SR inference pipeline.

Maps 64-pt coarse observations -> 512-pt states autoregressively:
  t=0 : init from spectral upsample of obs_64[0]
  t>=1: posterior = DDIM(cond=[prev_512, obs_up_512[t]])
        prev_512  <- posterior  (own output, not GT)

Also provides a bicubic (spectral upsample) baseline for direct comparison.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.models.diffusion import GaussianDiffusion
from src.data.solver import spectral_upsample


class OneShotPipeline1d:
    """1D one-shot diffusion SR autoregressive inference pipeline.

    Args:
        diffusion: GaussianDiffusion wrapping OneShotUNet1d (EMA weights loaded).
        cfg:       Config object (configs/oneshot_sr_1d.yaml).
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

    def _upsample(self, u64: torch.Tensor) -> torch.Tensor:
        """Spectrally upsample (B, 64) or (B, 1, 64) -> (B, 1, 512)."""
        if u64.dim() == 3:
            u64 = u64.squeeze(1)      # (B, 64)
        up = spectral_upsample(u64, 512)   # (B, 512)
        return up.unsqueeze(1)             # (B, 1, 512)

    def _ddim(
        self,
        u_prev: torch.Tensor,
        u_obs_up: torch.Tensor,
    ) -> torch.Tensor:
        """One DDIM pass.

        Args:
            u_prev:   (B, 1, 512)  — previous posterior (u_forecast slot)
            u_obs_up: (B, 1, 512)  — upsampled coarse obs (u_coarse_up slot)

        Returns:
            posterior: (B, 1, 512)
        """
        B = u_prev.shape[0]
        res_idx = torch.zeros(B, dtype=torch.long, device=self.device)
        return self.diffusion.ddim_sample(
            u_prev, u_obs_up, res_idx,
            ddim_steps=self.ddim_steps,
            eta=self.eta,
        )  # (B, 1, 512)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        observations_64: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive one-shot inference over a trajectory.

        Args:
            observations_64: Coarse observations, shape (T, 64).
            n_steps:         Number of time steps to process (default: all T).

        Returns:
            Dict with keys:
              'posterior_512': (T, 512) — one-shot diffusion posterior
              'obs_64':        (T, 64)  — input coarse observations (copy)
        """
        if observations_64.dim() == 3:
            observations_64 = observations_64.squeeze(1)
        obs = observations_64.to(self.device)   # (T, 64)
        T   = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        posteriors: list[torch.Tensor] = []

        # t=0: initialise from spectral upsample
        prev_512 = self._upsample(obs[0:1])   # (1, 1, 512)
        posteriors.append(prev_512.squeeze().cpu())   # (512,)

        for t in tqdm(range(1, T), desc="OneShot-1D inference", ncols=80, leave=False):
            u_obs_up = self._upsample(obs[t:t+1])     # (1, 1, 512)
            posterior = self._ddim(prev_512, u_obs_up) # (1, 1, 512)
            posteriors.append(posterior.squeeze().cpu())
            prev_512 = posterior   # autoregressive

        return {
            "posterior_512": torch.stack(posteriors, dim=0),   # (T, 512)
            "obs_64":        obs.cpu(),                        # (T, 64)
        }

    @torch.no_grad()
    def run_bicubic(
        self,
        observations_64: torch.Tensor,
        n_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Spectral-upsample (bicubic) baseline: each frame independently.

        Args:
            observations_64: Coarse observations, shape (T, 64).
            n_steps:         Number of time steps to process.

        Returns:
            Dict with key 'bicubic_512': (T, 512).
        """
        if observations_64.dim() == 3:
            observations_64 = observations_64.squeeze(1)
        obs = observations_64.to(self.device)   # (T, 64)
        T   = obs.shape[0]
        if n_steps is not None:
            T   = min(T, n_steps)
            obs = obs[:T]

        # Batch upsample all T frames at once
        up = spectral_upsample(obs, 512)        # (T, 512)
        return {"bicubic_512": up.cpu()}
