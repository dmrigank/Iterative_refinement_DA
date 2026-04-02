"""
1D conditional U-Net for the diffusion corrector G.

Input channels (concatenated along dim=1):
  0: x_noisy      — noised target field          shape (B, 1, L)
  1: u_forecast   — FNO prior at target res       shape (B, 1, L)
  2: u_coarse_up  — coarse obs upsampled to L     shape (B, 1, L)
  => concatenated: (B, 3, L)

Architecture (channel multipliers [1,2,4], base_channels=64):
  Lift:      Conv1d(3, 64, 1)
  Encoder:
    DownBlock(64  -> 64,  n_res=1) -> downsample -> (B, 64,  L/2)
    DownBlock(64  -> 128, n_res=1) -> downsample -> (B, 128, L/4)
    DownBlock(128 -> 256, n_res=1) -> downsample -> (B, 256, L/8)
  Bottleneck:
    ResBlock(256 -> 256, n_res=1)                  (B, 256, L/8)
  Decoder:
    UpBlock(256+256 -> 128, n_res=1) -> upsample   (B, 128, L/4)
    UpBlock(128+128 -> 64,  n_res=1) -> upsample   (B, 64,  L/2)
    UpBlock(64+64   -> 64,  n_res=1) -> upsample   (B, 64,  L)
  Head:      GroupNorm -> SiLU -> Conv1d(64, 1, 1)

Conditioning (summed before injection):
  TimestepEmbedding(noise_step)   -> cond_dim
  ResolutionEmbedding(res_idx)    -> cond_dim
  sum -> cond_dim vector, injected via FiLMConditioner at every ResBlock
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.embeddings import TimestepEmbedding, ResolutionEmbedding, FiLMConditioner


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """1D residual block with GroupNorm, SiLU, and FiLM conditioning.

    Forward pass:
      h = GroupNorm -> SiLU -> Conv(in->out, k=3)
      h = FiLM(h, cond)                    ← conditioning injected here
      h = GroupNorm -> SiLU -> Conv(out->out, k=3)
      skip = x if in==out else Conv1x1(x)
      return h + skip

    Args:
        in_channels:  Input channel count
        out_channels: Output channel count
        cond_dim:     Conditioning vector dimension
        n_groups:     Number of GroupNorm groups
    """

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

        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, L)
            cond: (B, cond_dim)

        Returns:
            (B, C_out, L)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)               # (B, C_out, L)
        h = self.film(h, cond)          # FiLM conditioning
        h = self.act(self.norm2(h))
        h = self.conv2(h)               # (B, C_out, L)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """Encoder block: n_res ResidualBlocks then strided Conv1d downsample (×2).

    Args:
        in_channels:  Input channels
        out_channels: Output channels (after the residual blocks, before downsample)
        cond_dim:     Conditioning dim passed to each ResidualBlock
        n_res:        Number of ResidualBlocks
        n_groups:     GroupNorm groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        n_res: int = 1,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        blocks = []
        for i in range(n_res):
            blocks.append(
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    cond_dim,
                    n_groups,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        # Strided conv for 2× spatial downsampling
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, C_in, L)
            cond: (B, cond_dim)

        Returns:
            (skip, x_down): skip=(B, C_out, L), x_down=(B, C_out, L/2)
        """
        for block in self.blocks:
            x = block(x, cond)
        skip = x
        return skip, self.downsample(x)


class UpBlock(nn.Module):
    """Decoder block: nearest upsample + Conv1d, cat skip, n_res ResidualBlocks.

    Args:
        in_channels:   Channels coming from the lower level (before cat)
        skip_channels: Channels from the skip connection
        out_channels:  Output channels
        cond_dim:      Conditioning dim
        n_res:         Number of ResidualBlocks
        n_groups:      GroupNorm groups
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
        n_res: int = 1,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        # Upsample: nearest-neighbour then Conv1d to smooth
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        blocks = []
        for i in range(n_res):
            c_in = (in_channels + skip_channels) if i == 0 else out_channels
            blocks.append(ResidualBlock(c_in, out_channels, cond_dim, n_groups))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, L)
            skip: (B, C_skip, 2L)   — skip from encoder at the larger resolution
            cond: (B, cond_dim)

        Returns:
            (B, C_out, 2L)
        """
        x = self.upsample(x)                    # (B, C_in, 2L)

        # Handle odd-length mismatches from strided conv downsampling
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")

        x = torch.cat([x, skip], dim=1)         # (B, C_in + C_skip, 2L)
        for block in self.blocks:
            x = block(x, cond)
        return x


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet1d(nn.Module):
    """1D conditional U-Net for the diffusion corrector G.

    Shared across all resolution transitions (r=0,1,2). The resolution
    index is part of the conditioning, not the architecture.

    Args:
        cfg: Full config; reads cfg.unet.* and cfg.diffusion.*
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        base_ch:  int       = int(cfg.unet.base_channels)        # 64
        mults:    list[int] = list(cfg.unet.channel_mults)       # [1, 2, 4]
        n_res:    int       = int(cfg.unet.n_res_blocks)         # 1
        n_groups: int       = int(cfg.unet.group_norm_groups)    # 8
        cond_dim: int       = int(cfg.unet.cond_embed_dim)       # 128
        in_ch:    int       = int(cfg.unet.in_channels)          # 3

        # Channel widths at each level
        ch = [base_ch * m for m in mults]   # e.g. [64, 128, 256]
        n_levels = len(ch)                   # 3

        # --- Conditioning embeddings ---
        self.time_emb = TimestepEmbedding(cond_dim=cond_dim)
        self.res_emb  = ResolutionEmbedding(cond_dim=cond_dim)

        # --- Lifting: 3 -> base_ch ---
        self.lift = nn.Conv1d(in_ch, ch[0], kernel_size=1)

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        in_c = ch[0]
        for i in range(n_levels):
            out_c = ch[i]
            self.down_blocks.append(DownBlock(in_c, out_c, cond_dim, n_res, n_groups))
            in_c = out_c

        # --- Bottleneck ---
        self.mid_blocks = nn.ModuleList(
            [ResidualBlock(ch[-1], ch[-1], cond_dim, n_groups) for _ in range(n_res)]
        )

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(n_levels)):
            skip_c = ch[i]
            out_c  = ch[i - 1] if i > 0 else ch[0]
            self.up_blocks.append(UpBlock(in_c, skip_c, out_c, cond_dim, n_res, n_groups))
            in_c = out_c

        # --- Output head ---
        self.head = nn.Sequential(
            nn.GroupNorm(n_groups, ch[0]),
            nn.SiLU(),
            nn.Conv1d(ch[0], 1, kernel_size=1),
        )

    def forward(
        self,
        x_noisy: torch.Tensor,
        u_forecast: torch.Tensor,
        u_coarse_up: torch.Tensor,
        noise_step: torch.Tensor,
        resolution_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_noisy:       (B, 1, L)  — noised target
            u_forecast:    (B, 1, L)  — FNO prior
            u_coarse_up:   (B, 1, L)  — upsampled coarse observation
            noise_step:    (B,)       — diffusion timestep indices
            resolution_idx:(B,)       — stage index in {0,1,2}

        Returns:
            Predicted noise, shape (B, 1, L)
        """
        B, _, L = x_noisy.shape
        assert u_forecast.shape  == (B, 1, L), f"u_forecast shape mismatch: {u_forecast.shape}"
        assert u_coarse_up.shape == (B, 1, L), f"u_coarse_up shape mismatch: {u_coarse_up.shape}"
        assert noise_step.shape    == (B,), f"noise_step shape: {noise_step.shape}"
        assert resolution_idx.shape == (B,), f"resolution_idx shape: {resolution_idx.shape}"

        # --- Build conditioning vector: (B, cond_dim) ---
        cond = self.time_emb(noise_step) + self.res_emb(resolution_idx)   # (B, cond_dim)

        # --- Concatenate inputs: (B, 3, L) ---
        x = torch.cat([x_noisy, u_forecast, u_coarse_up], dim=1)          # (B, 3, L)

        # --- Lift ---
        x = self.lift(x)                                                   # (B, base_ch, L)

        # --- Encoder ---
        skips: list[torch.Tensor] = []
        for down in self.down_blocks:
            skip, x = down(x, cond)
            skips.append(skip)

        # --- Bottleneck ---
        for mid in self.mid_blocks:
            x = mid(x, cond)

        # --- Decoder ---
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, cond)

        assert x.shape == (B, self.head[-1].in_channels, L), \
            f"Pre-head shape wrong: {x.shape}, expected (B, {self.head[-1].in_channels}, {L})"

        return self.head(x)                                                # (B, 1, L)


# Backward-compatible alias used in diffusion.py skeleton
UNet1d = ConditionalUNet1d
