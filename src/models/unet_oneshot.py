"""
One-shot 2D diffusion U-Net for the SR baseline.

Maps 32×32 → 256×256 directly in a single diffusion pass.

Architecture summary (base_channels=64, channel_mults=[1,2,4]):
  Input:      (B, 3, 256, 256)   [w_noisy | w_prev | w_obs_up]
  InputConv:  CircularConv2d(3, 64, 3, padding=1)
  Encoder:    DownBlock(64->64) -> DownBlock(64->128) -> DownBlock(128->256)
  Bottleneck: ResBlock(256) + SelfAttention2d + ResBlock(256)
  Decoder:    UpBlock(256+256->128) -> UpBlock(128+128->64) -> UpBlock(64+64->64)
  Output:     GroupNorm -> SiLU -> Conv2d(64, 1, 1)

Attention is applied at the bottleneck only (spatial dim = 32 after 3× stride-2
downsampling of the 256×256 input).

Conditioning: diffusion timestep k only (no resolution index — one-shot has no
resolution hierarchy).  TimestepEmbedding -> 128-dim cond vector, injected at
every ResBlock via FiLM.

CRITICAL: All Conv2d layers with spatial kernels use padding_mode='circular'
          because the vorticity domain is doubly periodic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.embeddings import TimestepEmbedding


# ---------------------------------------------------------------------------
# Circular-padded Conv2d helper
# ---------------------------------------------------------------------------

def _circ(in_ch: int, out_ch: int, kernel_size: int,
          stride: int = 1, padding: int = 0) -> nn.Conv2d:
    """Conv2d with circular padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size,
                     stride=stride, padding=padding, padding_mode="circular")


# ---------------------------------------------------------------------------
# FiLM for 2D feature maps
# ---------------------------------------------------------------------------

class _FiLM2d(nn.Module):
    """Feature-wise Linear Modulation: out = (1 + scale) * x + shift."""

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond)                       # (B, 2C)
        scale, shift = params.chunk(2, dim=-1)         # (B, C) each
        scale = scale.unsqueeze(-1).unsqueeze(-1)      # (B, C, 1, 1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return (1.0 + scale) * x + shift


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class _ResBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 n_groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_ch)
        self.conv1 = _circ(in_ch, out_ch, 3, padding=1)
        self.film  = _FiLM2d(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.conv2 = _circ(out_ch, out_ch, 3, padding=1)
        self.act   = nn.SiLU()
        self.skip  = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention (bottleneck only)
# ---------------------------------------------------------------------------

class _SelfAttn2d(nn.Module):
    def __init__(self, channels: int, n_groups: int = 8) -> None:
        super().__init__()
        self.norm  = nn.GroupNorm(n_groups, channels)
        self.qkv   = nn.Conv2d(channels, 3 * channels, 1)
        self.proj  = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)                    # (B, C, H, W) each
        q = q.reshape(B, C, -1).permute(0, 2, 1)                 # (B, HW, C)
        k = k.reshape(B, C, -1).permute(0, 2, 1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)
        attn = torch.softmax(torch.bmm(q, k.permute(0, 2, 1)) * self.scale, dim=-1)
        out  = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(out)


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 n_groups: int = 8, use_attention: bool = False) -> None:
        super().__init__()
        self.res       = _ResBlock2d(in_ch, out_ch, cond_dim, n_groups)
        self.attn      = _SelfAttn2d(out_ch, n_groups) if use_attention else None
        self.downsample = _circ(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        x    = self.res(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        skip = x
        return skip, self.downsample(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int,
                 n_groups: int = 8, use_attention: bool = False) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            _circ(in_ch, in_ch, 3, padding=1),
        )
        self.res  = _ResBlock2d(in_ch + skip_ch, out_ch, cond_dim, n_groups)
        self.attn = _SelfAttn2d(out_ch, n_groups) if use_attention else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, cond)
        if self.attn is not None:
            x = self.attn(x)
        return x


# ---------------------------------------------------------------------------
# One-shot U-Net
# ---------------------------------------------------------------------------

class OneShotUNet2d(nn.Module):
    """One-shot 2D U-Net for diffusion SR baseline (32×32 → 256×256).

    Takes the concatenation of [w_noisy, w_prev, w_obs_up] as input.
    Conditioned on diffusion timestep k only (no resolution index).

    Args:
        base_channels:    Base channel count (default 64)
        channel_mults:    Channel multipliers (default [1, 2, 4])
        cond_embed_dim:   Conditioning embedding dimension (default 128)
        n_groups:         GroupNorm groups (default 8)
    """

    def __init__(
        self,
        base_channels:  int       = 64,
        channel_mults:  list[int] = (1, 2, 4),
        cond_embed_dim: int       = 128,
        n_groups:       int       = 8,
    ) -> None:
        super().__init__()
        ch = [base_channels * m for m in channel_mults]   # e.g. [64, 128, 256]

        # Timestep embedding (no resolution embedding)
        self.time_emb = TimestepEmbedding(cond_dim=cond_embed_dim)

        # Input conv: 3 channels → ch[0]
        self.input_conv = _circ(3, ch[0], 3, padding=1)

        # Encoder
        self.down0 = _DownBlock(ch[0], ch[0], cond_embed_dim, n_groups)
        self.down1 = _DownBlock(ch[0], ch[1], cond_embed_dim, n_groups)
        self.down2 = _DownBlock(ch[1], ch[2], cond_embed_dim, n_groups)

        # Bottleneck (attention at spatial dim 32 = 256 / 2^3)
        self.mid1 = _ResBlock2d(ch[2], ch[2], cond_embed_dim, n_groups)
        self.mid_attn = _SelfAttn2d(ch[2], n_groups)
        self.mid2 = _ResBlock2d(ch[2], ch[2], cond_embed_dim, n_groups)

        # Decoder
        self.up0 = _UpBlock(ch[2], ch[2], ch[1], cond_embed_dim, n_groups)
        self.up1 = _UpBlock(ch[1], ch[1], ch[0], cond_embed_dim, n_groups)
        self.up2 = _UpBlock(ch[0], ch[0], ch[0], cond_embed_dim, n_groups)

        # Output head
        self.head = nn.Sequential(
            nn.GroupNorm(n_groups, ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], 1, kernel_size=1),
        )

    def forward(
        self,
        w_noisy:        torch.Tensor,              # (B, 1, 256, 256)
        w_prev:         torch.Tensor,              # (B, 1, 256, 256)
        w_obs_up:       torch.Tensor,              # (B, 1, 256, 256)
        noise_step:     torch.Tensor,              # (B,)
        resolution_idx: torch.Tensor | None = None,  # ignored — no resolution hierarchy
    ) -> torch.Tensor:
        """
        Returns:
            Predicted noise, shape (B, 1, 256, 256)
        """
        B, _, ny, nx = w_noisy.shape
        assert w_prev.shape    == (B, 1, ny, nx)
        assert w_obs_up.shape  == (B, 1, ny, nx)
        assert noise_step.shape == (B,)

        cond = self.time_emb(noise_step)   # (B, cond_embed_dim)

        x = torch.cat([w_noisy, w_prev, w_obs_up], dim=1)  # (B, 3, ny, nx)
        x = self.input_conv(x)                              # (B, ch[0], ny, nx)

        # Encoder
        skip0, x = self.down0(x, cond)   # skip: (B, ch[0], ny,   nx)
        skip1, x = self.down1(x, cond)   # skip: (B, ch[0], ny/2, nx/2)
        skip2, x = self.down2(x, cond)   # skip: (B, ch[1], ny/4, nx/4)

        # Bottleneck  (spatial: ny/8 = 32 for 256×256 input)
        x = self.mid1(x, cond)
        x = self.mid_attn(x)
        x = self.mid2(x, cond)

        # Decoder
        x = self.up0(x, skip2, cond)
        x = self.up1(x, skip1, cond)
        x = self.up2(x, skip0, cond)

        return self.head(x)              # (B, 1, ny, nx)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = OneShotUNet2d().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    B = 2
    ny, nx = 256, 256

    w_noisy     = torch.randn(B, 1, ny, nx, device=device)
    w_prev      = torch.randn(B, 1, ny, nx, device=device)
    w_obs_up    = torch.randn(B, 1, ny, nx, device=device)
    noise_step  = torch.randint(0, 1000, (B,), device=device)

    with torch.no_grad():
        out = model(w_noisy, w_prev, w_obs_up, noise_step)

    expected = (B, 1, ny, nx)
    ok_shape = tuple(out.shape) == expected
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] input=(B,3,256,256)  output={tuple(out.shape)}")

    # Alignment check: w_prev at t-1 should correlate with w_truth at t
    # (smoke-test: just verify the dataset returns consistent shapes and t-indexing)
    print("\nDataset alignment smoke-test (requires data_2d/ on disk):")
    try:
        from src.data.dataset_oneshot import OneShotDataset
        ds = OneShotDataset("data_2d/train.pt")
        item = ds[0]
        print(f"  [PASS] item 0 shapes: "
              f"w_truth={tuple(item['w_truth'].shape)}, "
              f"w_prev={tuple(item['w_prev'].shape)}, "
              f"w_obs_up={tuple(item['w_obs_up'].shape)}")
        # t=1 for idx=0: w_truth[traj=0, t=1] and w_prev[traj=0, t=0]
        # Verify they are different fields (t vs t-1)
        same = torch.allclose(item["w_truth"], item["w_prev"])
        print(f"  [{'FAIL' if same else 'PASS'}] w_truth != w_prev (t vs t-1)")
    except FileNotFoundError:
        print("  [SKIP] data_2d/ not found — skipping dataset test")

    sys.exit(0 if ok_shape else 1)
