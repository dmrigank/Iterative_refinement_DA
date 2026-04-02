"""
Embedding modules for diffusion and resolution conditioning.

Two conditioning signals used by the U-Net diffusion corrector:
  1. Diffusion noise_step k  — TimestepEmbedding -> cond_embed_dim
  2. Resolution index r in {0,1,2} — ResolutionEmbedding -> cond_embed_dim

These are summed into a single conditioning vector and injected at every
ResBlock via FiLM (Feature-wise Linear Modulation).

NO physical time embedding — OU forcing is stationary.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal embedding (shared primitive)
# ---------------------------------------------------------------------------

def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar indices.

    Uses log-spaced frequencies identical to the original Transformer and
    DDPM papers:
        emb[i] = sin(x / 10000^(2i/dim))   for i < dim/2
        emb[i] = cos(x / 10000^(2i/dim))   for i >= dim/2

    Args:
        x: Scalar indices, shape (B,), any numeric dtype
        dim: Embedding dimension (must be even)

    Returns:
        Embeddings, shape (B, dim)
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    half = dim // 2
    # log-spaced frequencies: 1/10000^(2i/dim) for i in [0, half)
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=x.device) / half
    )  # (half,)
    # outer product: (B, half)
    args = x.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Timestep embedding (diffusion noise step k)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Embeds diffusion noise_step k into a dense conditioning vector.

    Pipeline: sinusoidal(k) [dim=dim] -> Linear(dim, dim*4) -> SiLU -> Linear(dim*4, cond_dim)

    Args:
        cond_dim: Output conditioning dimension (default 128)
        sinusoidal_dim: Dimension of the sinusoidal embedding (default 128)
    """

    def __init__(self, cond_dim: int = 128, sinusoidal_dim: int = 128) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, sinusoidal_dim * 4),
            nn.SiLU(),
            nn.Linear(sinusoidal_dim * 4, cond_dim),
        )

    def forward(self, noise_step: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_step: Integer noise step indices, shape (B,)

        Returns:
            Embedding, shape (B, cond_dim)
        """
        emb = sinusoidal_embedding(noise_step, self.sinusoidal_dim)  # (B, sinusoidal_dim)
        return self.mlp(emb)                                          # (B, cond_dim)


# ---------------------------------------------------------------------------
# Resolution embedding (stage index r in {0, 1, 2})
# ---------------------------------------------------------------------------

class ResolutionEmbedding(nn.Module):
    """Embeds resolution stage index r in {0, 1, 2} into a conditioning vector.

    Pipeline: sinusoidal(r) [dim=dim] -> Linear(dim, dim*4) -> SiLU -> Linear(dim*4, cond_dim)

    Args:
        cond_dim: Output conditioning dimension (default 128)
        sinusoidal_dim: Dimension of the sinusoidal embedding (default 128)
    """

    def __init__(self, cond_dim: int = 128, sinusoidal_dim: int = 128) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, sinusoidal_dim * 4),
            nn.SiLU(),
            nn.Linear(sinusoidal_dim * 4, cond_dim),
        )

    def forward(self, resolution_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            resolution_idx: Integer resolution stage indices, shape (B,), values in {0, 1, 2}

        Returns:
            Embedding, shape (B, cond_dim)
        """
        emb = sinusoidal_embedding(resolution_idx, self.sinusoidal_dim)  # (B, sinusoidal_dim)
        return self.mlp(emb)                                              # (B, cond_dim)


# ---------------------------------------------------------------------------
# FiLM conditioning layer
# ---------------------------------------------------------------------------

class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation (FiLM) conditioner.

    Projects a conditioning vector to per-channel scale and shift, then applies:
        out = scale * x + shift

    The projection MLP is: Linear(cond_dim, hidden_dim) -> SiLU -> Linear(hidden_dim, 2*channels)
    Output is split evenly into scale and shift, each of shape (B, channels).

    Args:
        cond_dim: Dimension of the input conditioning vector
        channels: Number of feature channels to modulate
        hidden_dim: Hidden dimension of the projection MLP (defaults to cond_dim)
    """

    def __init__(self, cond_dim: int, channels: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * channels),
        )
        # Initialise the final linear to zero so FiLM starts as identity
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map, shape (B, C, L)
            cond: Conditioning vector, shape (B, cond_dim)

        Returns:
            Modulated feature map, shape (B, C, L)
        """
        # (B, 2*C) -> split into scale (B, C) and shift (B, C)
        params = self.mlp(cond)               # (B, 2*C)
        scale, shift = params.chunk(2, dim=-1)  # each (B, C)
        # Broadcast over spatial dimension L
        return scale.unsqueeze(-1) * x + shift.unsqueeze(-1)


# Keep backward-compatible alias used in unet.py skeleton
FiLMLayer = FiLMConditioner
