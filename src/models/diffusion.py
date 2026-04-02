"""
DDPM/DDIM noise schedule and sampling for the diffusion corrector G.

Noise schedule: cosine (Nichol & Dhariwal 2021)
Training objective: predict added noise (epsilon-prediction)
Inference: DDIM with configurable steps and eta

Variable naming — strictly maintained throughout:
  noise_step (k): diffusion timestep index, range [0, T-1]
  time_idx   (t): physical simulation time index
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.unet import ConditionalUNet1d


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal 2021).

    α̅(t) = f(t) / f(0),   f(t) = cos²( ((t/T + s) / (1+s)) · π/2 )
    β(t) = 1 - α̅(t) / α̅(t-1),  clipped to [0, 0.999]

    Args:
        T: Total diffusion steps
        s: Offset (default 0.008)

    Returns:
        betas: shape (T,)
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T + s) / (1.0 + s)) * (torch.pi / 2.0)) ** 2
    alpha_bars = f / f[0]
    betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
    return betas.clamp(0.0, 0.999).float()


def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear noise schedule.

    Args:
        T: Total diffusion steps
        beta_start: β at step 0
        beta_end:   β at step T-1

    Returns:
        betas: shape (T,)
    """
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


# ---------------------------------------------------------------------------
# GaussianDiffusion
# ---------------------------------------------------------------------------

class GaussianDiffusion(nn.Module):
    """DDPM training + DDIM inference wrapper around ConditionalUNet1d.

    Registers all schedule tensors as buffers so they move with .to(device).

    Args:
        model: The ConditionalUNet1d denoiser.
        cfg:   Full config; reads cfg.diffusion.* and cfg.inference.*
    """

    def __init__(self, model: ConditionalUNet1d, cfg: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.T     = int(cfg.diffusion.T)                         # 1000
        self.cfg   = cfg

        # Build schedule
        schedule = str(cfg.diffusion.schedule)
        if schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif schedule == "linear":
            betas = linear_beta_schedule(
                self.T,
                float(cfg.diffusion.beta_start),
                float(cfg.diffusion.beta_end),
            )
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas      = 1.0 - betas                                 # (T,)
        alpha_bars  = torch.cumprod(alphas, dim=0)                # (T,)  ᾱ_t
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}, ᾱ_0=1

        # Register as buffers — moved to correct device automatically
        self.register_buffer("betas",                  betas)
        self.register_buffer("alphas",                 alphas)
        self.register_buffer("alpha_bars",             alpha_bars)
        self.register_buffer("alpha_bars_prev",        alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars",        alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        # For DDPM posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars),
        )

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        noise_step: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from q(x_k | x_0) = N(√ᾱ_k · x_0, (1-ᾱ_k) · I).

        Works for any spatial shape: (B, 1, N) for 1D or (B, 1, ny, nx) for 2D.

        Args:
            x0:         Clean sample, shape (B, 1, ...)
            noise_step: Timestep indices k, shape (B,), in [0, T-1]
            noise:      Optional pre-sampled noise, same shape as x0; generated if None

        Returns:
            x_noisy: same shape as x0
            noise:   same shape as x0
        """
        if noise is None:
            noise = torch.randn_like(x0)

        B = x0.shape[0]
        assert noise_step.shape == (B,), f"noise_step shape: {noise_step.shape}"

        sqrt_ab   = self.sqrt_alpha_bars[noise_step]             # (B,)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[noise_step]   # (B,)

        # Reshape for broadcasting over all spatial dims (B, 1, ...) -> (B, 1, 1, ..., 1)
        n_extra = x0.dim() - 1  # number of dims after batch
        sqrt_ab   = sqrt_ab.view(B, *([1] * n_extra))
        sqrt_1mab = sqrt_1mab.view(B, *([1] * n_extra))

        x_noisy = sqrt_ab * x0 + sqrt_1mab * noise
        return x_noisy, noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def training_loss(
        self,
        x0: torch.Tensor,
        u_forecast: torch.Tensor,
        u_coarse_up: torch.Tensor,
        resolution_idx: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM training loss: E[‖ε - ε_θ(x_k, k, cond)‖²].

        Samples a uniformly random noise_step k for each element in the batch,
        noises x0, and computes MSE between the true and predicted noise.

        Works for any spatial shape: (B, 1, N) for 1D or (B, 1, ny, nx) for 2D.

        Args:
            x0:            Clean target field, shape (B, 1, ...)
            u_forecast:    FNO prior,            shape (B, 1, ...)
            u_coarse_up:   Upsampled coarse obs, shape (B, 1, ...)
            resolution_idx: Stage index,          shape (B,)

        Returns:
            Scalar MSE loss
        """
        B = x0.shape[0]
        assert resolution_idx.shape == (B,)

        # Sample random noise steps
        noise_step = torch.randint(0, self.T, (B,), device=x0.device)

        x_noisy, noise = self.q_sample(x0, noise_step)

        noise_pred = self.model(
            x_noisy, u_forecast, u_coarse_up, noise_step, resolution_idx
        )

        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        u_forecast: torch.Tensor,
        u_coarse_up: torch.Tensor,
        resolution_idx: torch.Tensor,
        ddim_steps: int | None = None,
        eta: float | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """DDIM reverse sampling.

        Generates evenly-spaced subsequence of noise steps from T-1 down to 0,
        then iteratively denoises.

        Algorithm (Song et al. 2021, eq. 12):
          x̂₀ = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
          σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})
          x_{t-1} = √ᾱ_{t-1} · x̂₀ + √(1-ᾱ_{t-1} - σ_t²) · ε_θ + σ_t · z

        Args:
            u_forecast:     FNO prior,            shape (B, 1, ...)
            u_coarse_up:    Upsampled coarse obs,  shape (B, 1, ...)
            resolution_idx: Stage index,           shape (B,)
            ddim_steps:     Number of DDIM steps (default: cfg.inference.ddim_steps)
            eta:            Stochasticity (0=deterministic, 1=DDPM)
                            (default: cfg.inference.eta)
            return_trajectory: If True, also return list of x̂₀ at each step

        Returns:
            x0_pred: Final denoised sample, same shape as u_forecast
            trajectory (optional): list of tensors same shape as u_forecast, one per DDIM step
        """
        # Works for any spatial shape: (B, 1, N) for 1D or (B, 1, ny, nx) for 2D.
        # Infer the full sample shape from u_forecast (same spatial dims as the target).
        B      = u_forecast.shape[0]
        device = u_forecast.device
        sample_shape = u_forecast.shape  # (B, 1, ...) — same channel/spatial layout as x0

        ddim_steps = ddim_steps if ddim_steps is not None else int(self.cfg.inference.ddim_steps)
        eta        = eta        if eta        is not None else float(self.cfg.inference.eta)

        # Build evenly-spaced subsequence of noise steps (inclusive of T-1, down to 0)
        # e.g. T=1000, ddim_steps=25 -> [999, 959, 919, ..., 39, -1]  (step=-1 means t_prev=None)
        step_indices = torch.linspace(self.T - 1, 0, ddim_steps, dtype=torch.long)
        # Prepend a sentinel -1 for "previous" of the first step (t_prev = before step 0)
        step_prev = torch.cat([step_indices[1:], torch.tensor([-1], dtype=torch.long)])

        # Start from pure noise matching the sample shape
        x = torch.randn(sample_shape, device=device)

        trajectory: list[torch.Tensor] = []

        for k, k_prev in zip(step_indices.tolist(), step_prev.tolist()):
            noise_step    = torch.full((B,), k,    dtype=torch.long, device=device)

            # Predict noise
            eps = self.model(x, u_forecast, u_coarse_up, noise_step, resolution_idx)
            # shape: (B, 1, N)

            ab_t    = self.alpha_bars[k]                         # scalar ᾱ_t
            ab_prev = self.alpha_bars[k_prev] if k_prev >= 0 else torch.ones(1, device=device)

            # Predict x̂₀ from current x_t and predicted noise
            sqrt_ab_t    = ab_t.sqrt()
            sqrt_1mab_t  = (1.0 - ab_t).sqrt()
            x0_pred = (x - sqrt_1mab_t * eps) / sqrt_ab_t       # (B, 1, N)

            # Optionally clip x0 to prevent drift; value is config-driven per problem
            x0_clamp = float(self.cfg.inference.get("x0_clamp", 5.0))
            x0_pred = x0_pred.clamp(-x0_clamp, x0_clamp)

            if return_trajectory:
                trajectory.append(x0_pred.clone())

            if k_prev < 0:
                # Final step — x̂₀ is the answer
                x = x0_pred
                break

            # DDIM update
            sqrt_ab_prev   = ab_prev.sqrt()
            sqrt_1mab_prev = (1.0 - ab_prev).sqrt()

            # σ_t = η · √( (1-ᾱ_{t-1}) / (1-ᾱ_t) · (1 - ᾱ_t/ᾱ_{t-1}) )
            sigma = (
                eta
                * ((1.0 - ab_prev) / (1.0 - ab_t)).sqrt()
                * (1.0 - ab_t / ab_prev).sqrt()
            )

            # Direction pointing to x_t
            dir_xt = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps

            noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
            x = sqrt_ab_prev * x0_pred + dir_xt + sigma * noise  # (B, 1, N)

        if return_trajectory:
            return x, trajectory
        return x
