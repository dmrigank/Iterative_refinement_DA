"""
1D conditional U-Net for the one-shot diffusion SR baseline.

Differences from ConditionalUNet1d (unet.py):
  - FiLM conditioning on diffusion timestep ONLY — no resolution embedding.
  - Standalone class; does not inherit from the existing U-Net.
  - resolution_idx argument accepted in forward() but ignored (for interface
    compatibility with GaussianDiffusion.training_loss and ddim_sample).

Input channels (concatenated along dim=1):
  0: u_noisy    — noised 512-pt target       shape (B, 1, 512)
  1: u_prev     — previous 512-pt state      shape (B, 1, 512)
  2: u_obs_up   — 64-pt obs upsampled to 512 shape (B, 1, 512)
  => concatenated: (B, 3, 512)

Architecture (base_channels=64, channel_mults=[1,2,4]):
  Lift:         Conv1d(3, 64, 1)
  Encoder:
    DownBlock(64  -> 64,  n_res=1) -> (B, 64,  256)
    DownBlock(64  -> 128, n_res=1) -> (B, 128, 128)
    DownBlock(128 -> 256, n_res=1) -> (B, 256, 64)
  Bottleneck:   ResBlock(256 -> 256)
  Decoder:
    UpBlock(256+256 -> 128) -> (B, 128, 128)
    UpBlock(128+128 -> 64)  -> (B, 64,  256)
    UpBlock(64+64   -> 64)  -> (B, 64,  512)
  Head:         GroupNorm -> SiLU -> Conv1d(64, 1, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings import TimestepEmbedding, FiLMConditioner


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.film  = FiLMConditioner(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act   = nn.SiLU()
        self.skip  = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class _DownBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, cond_dim: int,
        n_res: int = 1, n_groups: int = 8,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(n_res):
            blocks.append(_ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels, cond_dim, n_groups,
            ))
        self.blocks     = nn.ModuleList(blocks)
        self.downsample = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            x = block(x, cond)
        return x, self.downsample(x)


class _UpBlock(nn.Module):
    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int,
        cond_dim: int, n_res: int = 1, n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        blocks = []
        for i in range(n_res):
            c_in = (in_channels + skip_channels) if i == 0 else out_channels
            blocks.append(_ResBlock(c_in, out_channels, cond_dim, n_groups))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x, cond)
        return x


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class OneShotUNet1d(nn.Module):
    """1D one-shot diffusion SR U-Net.

    Conditioned on diffusion timestep only (no resolution embedding).
    resolution_idx is accepted but ignored in forward() for interface
    compatibility with GaussianDiffusion.

    Args:
        base_channels:   Base channel width (default 64).
        channel_mults:   Per-level multipliers (default [1, 2, 4]).
        n_res_blocks:    Residual blocks per level (default 1).
        n_groups:        GroupNorm groups (default 8).
        cond_embed_dim:  Conditioning embedding dimension (default 128).
    """

    def __init__(
        self,
        base_channels:  int       = 64,
        channel_mults:  list[int] = None,
        n_res_blocks:   int       = 1,
        n_groups:       int       = 8,
        cond_embed_dim: int       = 128,
    ) -> None:
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 4]

        ch      = [base_channels * m for m in channel_mults]   # [64, 128, 256]
        n_levels = len(ch)
        cond_dim = cond_embed_dim

        # Timestep embedding only
        self.time_emb = TimestepEmbedding(cond_dim=cond_dim)

        # Lift 3 input channels -> base_channels
        self.lift = nn.Conv1d(3, ch[0], kernel_size=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        in_c = ch[0]
        for i in range(n_levels):
            out_c = ch[i]
            self.down_blocks.append(_DownBlock(in_c, out_c, cond_dim, n_res_blocks, n_groups))
            in_c = out_c

        # Bottleneck
        self.mid_block = _ResBlock(ch[-1], ch[-1], cond_dim, n_groups)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_c = ch[i]
            out_c  = ch[i - 1] if i > 0 else ch[0]
            self.up_blocks.append(_UpBlock(in_c, skip_c, out_c, cond_dim, n_res_blocks, n_groups))
            in_c = out_c

        # Output head
        self.head = nn.Sequential(
            nn.GroupNorm(n_groups, ch[0]),
            nn.SiLU(),
            nn.Conv1d(ch[0], 1, kernel_size=1),
        )

    def forward(
        self,
        x_noisy:        torch.Tensor,
        u_prev:         torch.Tensor,
        u_obs_up:       torch.Tensor,
        noise_step:     torch.Tensor,
        resolution_idx: torch.Tensor | None = None,   # accepted but ignored
    ) -> torch.Tensor:
        """
        Args:
            x_noisy:       (B, 1, 512) — noised target
            u_prev:        (B, 1, 512) — previous 512-pt state (fills u_forecast slot)
            u_obs_up:      (B, 1, 512) — upsampled 64-pt observation
            noise_step:    (B,)        — diffusion timestep indices
            resolution_idx: (B,)       — accepted but ignored

        Returns:
            Predicted noise, shape (B, 1, 512)
        """
        B, _, L = x_noisy.shape

        # Conditioning: timestep only
        cond = self.time_emb(noise_step)    # (B, cond_dim)

        # Concatenate inputs
        x = torch.cat([x_noisy, u_prev, u_obs_up], dim=1)   # (B, 3, L)
        x = self.lift(x)                                      # (B, base_ch, L)

        # Encoder
        skips: list[torch.Tensor] = []
        for down in self.down_blocks:
            skip, x = down(x, cond)
            skips.append(skip)

        # Bottleneck
        x = self.mid_block(x, cond)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, cond)

        return self.head(x)   # (B, 1, L)
