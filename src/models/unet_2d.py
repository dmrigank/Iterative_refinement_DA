"""
2D conditional U-Net for the diffusion corrector G (Kraichnan testbed).

Architecture summary (base_channels=48, channel_mults=[1,2,4]):
  Input:      (B, 3, ny, nx)   [w_noisy | w_forecast | w_coarse_up]
  InputConv:  CircularConv2d(3, 48, 3, padding=1)
  Encoder:    DownBlock(48->48)  -> DownBlock(48->96)  -> DownBlock(96->192)
  Bottleneck: ResBlock(192)  + SelfAttention2d  + ResBlock(192)
  Decoder:    UpBlock(192+192->96) -> UpBlock(96+96->48) -> UpBlock(48+48->48)
  Output:     GroupNorm -> SiLU -> Conv2d(48, 1, 1)

CRITICAL: All Conv2d layers with spatial kernels use padding_mode='circular'
          because the vorticity domain is doubly periodic.
          1×1 convolutions do not need padding, so they use the default.

Conditioning:
  noise_step (k) -> TimestepEmbedding -> 128-dim
  res_idx    (r) -> ResolutionEmbedding -> 128-dim
  summed -> 128-dim cond vector, injected at every ResBlock via FiLM.

SelfAttention2d is applied only at the bottleneck (smallest spatial dim).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from src.models.embeddings import TimestepEmbedding, ResolutionEmbedding


# ---------------------------------------------------------------------------
# Circular-padded Conv2d helper
# ---------------------------------------------------------------------------

def CircularConv2d(
    in_ch: int,
    out_ch: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Conv2d:
    """Conv2d with circular padding — required for doubly-periodic domain."""
    return nn.Conv2d(
        in_ch, out_ch, kernel_size,
        stride=stride, padding=padding, padding_mode="circular",
    )


# ---------------------------------------------------------------------------
# FiLM for 2D feature maps
# ---------------------------------------------------------------------------

class FiLM2d(nn.Module):
    """Feature-wise Linear Modulation for 2D spatial features (B, C, H, W).

    Projects cond to per-channel scale and shift, then applies:
        out = (1 + scale) * x + shift      (residual parameterisation)
    The final linear is zero-initialised so the block starts as identity.

    Args:
        cond_dim: Conditioning vector dimension
        channels: Number of feature channels to modulate
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, H, W)
            cond: (B, cond_dim)

        Returns:
            (B, C, H, W)
        """
        params = self.proj(cond)                          # (B, 2C)
        scale, shift = params.chunk(2, dim=-1)            # (B, C) each
        scale = scale.unsqueeze(-1).unsqueeze(-1)         # (B, C, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + scale) * x + shift


# ---------------------------------------------------------------------------
# Residual block (2D)
# ---------------------------------------------------------------------------

class ResidualBlock2d(nn.Module):
    """2D residual block with GroupNorm, SiLU, circular Conv2d, and FiLM.

    Forward:
        h = GroupNorm -> SiLU -> CircularConv2d(3×3)
        h = FiLM2d(h, cond)
        h = GroupNorm -> SiLU -> CircularConv2d(3×3)
        return h + skip(x)

    Args:
        in_channels:  Input channel count
        out_channels: Output channel count
        cond_dim:     Conditioning vector dimension
        n_groups:     GroupNorm groups
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
        self.conv1 = CircularConv2d(in_channels, out_channels, 3, padding=1)
        self.film  = FiLM2d(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.conv2 = CircularConv2d(out_channels, out_channels, 3, padding=1)
        self.act   = nn.SiLU()

        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, H, W)
            cond: (B, cond_dim)

        Returns:
            (B, C_out, H, W)
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention (bottleneck only)
# ---------------------------------------------------------------------------

class SelfAttention2d(nn.Module):
    """Single-head self-attention over spatial positions for 2D feature maps.

    Reshapes (B, C, H, W) -> (B, H*W, C), runs scaled dot-product attention,
    reshapes back, then adds a residual projection.

    Args:
        channels: Number of feature channels
        n_groups: GroupNorm groups for the pre-norm
    """

    def __init__(self, channels: int, n_groups: int = 8) -> None:
        super().__init__()
        self.norm  = nn.GroupNorm(n_groups, channels)
        self.qkv   = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj  = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)                        # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)            # (B, C, H, W) each

        # Flatten spatial dims: (B, C, H*W) -> (B, H*W, C)
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, C, -1).permute(0, 2, 1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)

        # Scaled dot-product attention: (B, H*W, H*W)
        attn = torch.softmax(
            torch.bmm(q, k.permute(0, 2, 1)) * self.scale, dim=-1
        )
        out = torch.bmm(attn, v)                 # (B, H*W, C)

        # Reshape back: (B, C, H, W)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class DownBlock2d(nn.Module):
    """Encoder block: n_res ResidualBlock2d(s) [+ optional attention] + stride-2 downsample.

    Args:
        in_channels:          Input channels
        out_channels:         Output channels after residual blocks (before downsample)
        cond_dim:             Conditioning dim
        n_res:                Number of ResidualBlock2d per block
        n_groups:             GroupNorm groups
        use_attention:        Whether to apply SelfAttention2d after the residual blocks
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        n_res: int = 1,
        n_groups: int = 8,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for i in range(n_res):
            blocks.append(
                ResidualBlock2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    cond_dim,
                    n_groups,
                )
            )
        self.blocks     = nn.ModuleList(blocks)
        self.attention  = SelfAttention2d(out_channels, n_groups) if use_attention else None
        # Stride-2 circular conv for 2× spatial downsampling
        self.downsample = CircularConv2d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            skip: (B, out_channels, H, W)      — skip connection to decoder
            x_down: (B, out_channels, H/2, W/2) — downsampled output
        """
        for block in self.blocks:
            x = block(x, cond)
        if self.attention is not None:
            x = self.attention(x)
        skip = x
        return skip, self.downsample(x)


class UpBlock2d(nn.Module):
    """Decoder block: nearest upsample + circular conv, cat skip, n_res ResidualBlock2d(s).

    Args:
        in_channels:   Channels coming from the lower level (before cat with skip)
        skip_channels: Channels of the skip connection from encoder
        out_channels:  Output channels after residual blocks
        cond_dim:      Conditioning dim
        n_res:         Number of ResidualBlock2d per block
        n_groups:      GroupNorm groups
        use_attention: Whether to apply SelfAttention2d after residual blocks
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        cond_dim: int,
        n_res: int = 1,
        n_groups: int = 8,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        # Nearest-neighbour upsample + circular conv to smooth
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            CircularConv2d(in_channels, in_channels, 3, padding=1),
        )
        blocks: list[nn.Module] = []
        for i in range(n_res):
            c_in = (in_channels + skip_channels) if i == 0 else out_channels
            blocks.append(ResidualBlock2d(c_in, out_channels, cond_dim, n_groups))
        self.blocks    = nn.ModuleList(blocks)
        self.attention = SelfAttention2d(out_channels, n_groups) if use_attention else None

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, C_in, H, W)
            skip: (B, C_skip, 2H, 2W) — from matching encoder level
            cond: (B, cond_dim)

        Returns:
            (B, C_out, 2H, 2W)
        """
        x = self.upsample(x)                   # (B, C_in, 2H, 2W)

        # Handle odd-length mismatches from strided conv downsampling
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)        # (B, C_in + C_skip, 2H, 2W)
        for block in self.blocks:
            x = block(x, cond)
        if self.attention is not None:
            x = self.attention(x)
        return x


# ---------------------------------------------------------------------------
# Full 2D conditional U-Net
# ---------------------------------------------------------------------------

class ConditionalUNet2d(nn.Module):
    """2D conditional U-Net for the diffusion corrector G.

    Shared across all resolution transitions (r=0,1,2). The resolution
    index is part of the conditioning, not the architecture, so the same
    weights handle 32→64, 64→128, and 128→256 transitions.

    Args:
        cfg: Full config; reads cfg.unet.* and cfg.diffusion.*
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        base_ch:   int        = int(cfg.unet.base_channels)          # 48
        mults:     list[int]  = list(cfg.unet.channel_mults)         # [1, 2, 4]
        n_res:     int        = int(cfg.unet.n_res_blocks)           # 1
        n_groups:  int        = int(cfg.unet.group_norm_groups)      # 8
        cond_dim:  int        = int(cfg.unet.cond_embed_dim)         # 128
        in_ch:     int        = int(cfg.unet.in_channels)            # 3
        attn_res:  int        = int(cfg.unet.attention_resolution)   # 32

        ch = [base_ch * m for m in mults]    # e.g. [48, 96, 192]
        n_levels = len(ch)                   # 3

        # ── Conditioning embeddings ──────────────────────────────────────────
        self.time_emb = TimestepEmbedding(cond_dim=cond_dim)
        self.res_emb  = ResolutionEmbedding(cond_dim=cond_dim)

        # ── Input conv ───────────────────────────────────────────────────────
        self.input_conv = CircularConv2d(in_ch, ch[0], 3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        # We track the spatial size symbolically so we know when to apply attention.
        # Attention is applied when the spatial dim reaches attn_res — which happens
        # at the bottleneck after all downsampling. DownBlocks themselves never trigger
        # attention here (their output spatial size is still > attn_res for typical inputs).
        self.down_blocks = nn.ModuleList()
        in_c = ch[0]
        for i in range(n_levels):
            out_c = ch[i]
            self.down_blocks.append(
                DownBlock2d(in_c, out_c, cond_dim, n_res, n_groups, use_attention=False)
            )
            in_c = out_c

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid_block1 = ResidualBlock2d(ch[-1], ch[-1], cond_dim, n_groups)
        self.mid_attn   = SelfAttention2d(ch[-1], n_groups)
        self.mid_block2 = ResidualBlock2d(ch[-1], ch[-1], cond_dim, n_groups)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.up_blocks = nn.ModuleList()
        in_c = ch[-1]
        for i in reversed(range(n_levels)):
            skip_c = ch[i]
            out_c  = ch[i - 1] if i > 0 else ch[0]
            self.up_blocks.append(
                UpBlock2d(in_c, skip_c, out_c, cond_dim, n_res, n_groups, use_attention=False)
            )
            in_c = out_c

        # ── Output head ──────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.GroupNorm(n_groups, ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], 1, kernel_size=1),   # 1×1 — no padding needed
        )

    def forward(
        self,
        w_noisy: torch.Tensor,
        w_forecast: torch.Tensor,
        w_coarse_up: torch.Tensor,
        noise_step: torch.Tensor,
        resolution_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            w_noisy:       (B, 1, ny, nx)  — noised target vorticity
            w_forecast:    (B, 1, ny, nx)  — FNO prior at target resolution
            w_coarse_up:   (B, 1, ny, nx)  — upsampled coarse observation
            noise_step:    (B,)            — diffusion timestep indices
            resolution_idx:(B,)            — stage index in {0, 1, 2}

        Returns:
            Predicted noise, shape (B, 1, ny, nx)
        """
        B, _, ny, nx = w_noisy.shape
        assert w_forecast.shape  == (B, 1, ny, nx)
        assert w_coarse_up.shape == (B, 1, ny, nx)
        assert noise_step.shape    == (B,)
        assert resolution_idx.shape == (B,)

        # ── Conditioning vector ───────────────────────────────────────────────
        cond = self.time_emb(noise_step) + self.res_emb(resolution_idx)  # (B, cond_dim)

        # ── Concatenate inputs along channel dim ─────────────────────────────
        x = torch.cat([w_noisy, w_forecast, w_coarse_up], dim=1)  # (B, 3, ny, nx)

        # ── Input conv ────────────────────────────────────────────────────────
        x = self.input_conv(x)                                     # (B, base_ch, ny, nx)

        # ── Encoder ───────────────────────────────────────────────────────────
        skips: list[torch.Tensor] = []
        for down in self.down_blocks:
            skip, x = down(x, cond)
            skips.append(skip)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        # ── Decoder ───────────────────────────────────────────────────────────
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip, cond)

        return self.head(x)    # (B, 1, ny, nx)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "unet": {
            "in_channels": 3,
            "base_channels": 48,
            "channel_mults": [1, 2, 4],
            "n_res_blocks": 1,
            "use_attention": True,
            "attention_resolution": 32,
            "group_norm_groups": 8,
            "cond_embed_dim": 128,
            "padding_mode": "circular",
        },
        "diffusion": {"T": 1000, "schedule": "cosine"},
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ConditionalUNet2d(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    all_passed = True
    for ny, nx, res_idx, noise_step_val in [
        (64,  64,  1, 100),
        (128, 128, 2, 500),
    ]:
        B = 2
        w_noisy     = torch.randn(B, 1, ny, nx, device=device)
        w_forecast  = torch.randn(B, 1, ny, nx, device=device)
        w_coarse_up = torch.randn(B, 1, ny, nx, device=device)
        noise_step  = torch.full((B,), noise_step_val, dtype=torch.long, device=device)
        res_idx_t   = torch.full((B,), res_idx,        dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(w_noisy, w_forecast, w_coarse_up, noise_step, res_idx_t)

        expected = (B, 1, ny, nx)
        ok = (tuple(out.shape) == expected)
        if not ok:
            all_passed = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  [{status}] input=({B}, 3, {ny}, {nx})  "
            f"noise_step={noise_step_val}  res_idx={res_idx}  "
            f"output={tuple(out.shape)}"
        )

    sys.exit(0 if all_passed else 1)
