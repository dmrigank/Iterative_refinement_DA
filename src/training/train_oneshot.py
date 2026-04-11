"""
Training script for the one-shot diffusion SR baseline.

Uses GaussianDiffusion (same noise schedule / training loop as iterative
refinement) with OneShotUNet2d as the denoiser.

Interface mapping to GaussianDiffusion.training_loss / ddim_sample:
  u_forecast  <- w_prev     (previous 256×256 state, temporal context)
  u_coarse_up <- w_obs_up   (32×32 obs spectrally upsampled to 256×256)
  resolution_idx <- dummy zeros  (OneShotUNet2d ignores this argument)

Checkpoint format (saved to checkpoints_oneshot/):
  {
    'model':      state_dict,   <- online weights
    'ema_shadow': dict,
    'optimizer':  state_dict,
    'scheduler':  state_dict,
    'step':       int,
    'val_loss':   float,
  }

Final EMA weights saved to checkpoints_oneshot/oneshot_ema.pt.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.dataset_oneshot import OneShotDataset
from src.models.unet_oneshot import OneShotUNet2d
from src.models.diffusion import GaussianDiffusion


# ---------------------------------------------------------------------------
# EMA  (identical pattern to train_diffusion_2d.py)
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters (shadow on CPU float32)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.cpu().clone().float()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data.cpu().float(), alpha=1.0 - self.decay
            )

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name].to(param.device).to(param.dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.cpu().clone().float() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_datasets(cfg: DictConfig) -> tuple[OneShotDataset, OneShotDataset]:
    data_dir = Path(cfg.data.data_dir)   # "data_2d"
    train_ds = OneShotDataset(data_dir / "train.pt")
    val_ds   = OneShotDataset(data_dir / "val.pt")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_oneshot(cfg: DictConfig, device: torch.device) -> GaussianDiffusion:
    """Train the one-shot diffusion SR baseline.

    Args:
        cfg:    oneshot_sr.yaml config.
        device: Compute device.

    Returns:
        GaussianDiffusion with EMA weights loaded (ready for inference).
    """
    print(f"\n{'='*60}")
    print("  Training One-Shot Diffusion SR Baseline")
    print(f"{'='*60}")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)   # "checkpoints_oneshot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dt = cfg.diffusion_training
    total_steps   = int(dt.steps)           # 200_000
    batch_size    = int(dt.batch_size)      # 8
    log_interval  = int(dt.log_interval)    # 500
    val_interval  = int(dt.val_interval)    # 5_000
    save_interval = int(dt.save_interval)   # 25_000
    warmup_steps  = int(dt.warmup_steps)    # 3_000
    ema_decay     = float(dt.ema_decay)     # 0.9999

    # ── Datasets & loaders ───────────────────────────────────────────────────
    print("Building one-shot datasets...")
    train_ds, val_ds = _build_datasets(cfg)
    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    def cycle(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch

    train_iter = cycle(train_loader)

    # ── Model ────────────────────────────────────────────────────────────────
    # GaussianDiffusion type-checks the model against ConditionalUNet1d.
    # Loosen the type so it accepts OneShotUNet2d (structurally compatible).
    import src.models.diffusion as _dm
    _dm.ConditionalUNet1d = nn.Module

    unet      = OneShotUNet2d(
        base_channels  = int(cfg.unet.base_channels),
        channel_mults  = list(cfg.unet.channel_mults),
        cond_embed_dim = int(cfg.unet.cond_embed_dim),
        n_groups       = int(cfg.unet.group_norm_groups),
    ).to(device)

    diffusion = GaussianDiffusion(unet, cfg).to(device)
    n_params  = sum(p.numel() for p in unet.parameters())
    print(f"  OneShotUNet2d parameters: {n_params:,}")

    ema = EMAModel(unet, decay=ema_decay)

    # ── AMP scaler ───────────────────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=float(dt.lr),
        weight_decay=float(dt.weight_decay),
    )
    scheduler = _warmup_cosine(optimizer, warmup_steps, total_steps)

    # ── Resume from latest checkpoint if present ─────────────────────────────
    start_step = 0
    latest_ckpts = sorted(ckpt_dir.glob("oneshot_step_*.pt"))
    if latest_ckpts:
        ckpt_path = latest_ckpts[-1]
        print(f"  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        unet.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema_shadow"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"])
        print(f"  Resumed at step {start_step}")

    # ── Dummy res_idx for GaussianDiffusion interface ─────────────────────────
    # OneShotUNet2d ignores resolution_idx; we pass zeros to satisfy the assert
    # in GaussianDiffusion.training_loss.
    def _make_res_idx(B: int, dev: torch.device) -> torch.Tensor:
        return torch.zeros(B, dtype=torch.long, device=dev)

    # ── Training loop ────────────────────────────────────────────────────────
    unet.train()
    running_loss  = 0.0
    best_val_loss = float("inf")

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        desc="OneShotDiff training",
        ncols=110,
    )

    for step in pbar:
        batch = next(train_iter)

        w_truth  = batch["w_truth" ].to(device, non_blocking=True)   # (B, 1, 256, 256)
        w_prev   = batch["w_prev"  ].to(device, non_blocking=True)   # (B, 1, 256, 256)
        w_obs_up = batch["w_obs_up"].to(device, non_blocking=True)   # (B, 1, 256, 256)
        B = w_truth.shape[0]
        res_idx  = _make_res_idx(B, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Map to GaussianDiffusion.training_loss interface:
            #   u_forecast  <- w_prev    (temporal context)
            #   u_coarse_up <- w_obs_up  (coarse obs upsampled)
            loss = diffusion.training_loss(w_truth, w_prev, w_obs_up, res_idx)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(unet)

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        current_step = step + 1

        # ── Logging ──────────────────────────────────────────────────────────
        if current_step % log_interval == 0:
            avg_loss     = running_loss / log_interval
            running_loss = 0.0
            print(
                f"\n  step={current_step:6d}/{total_steps}  "
                f"train_loss={avg_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # ── Validation ───────────────────────────────────────────────────────
        if current_step % val_interval == 0:
            unet.eval()
            val_loss_sum = 0.0
            val_batches  = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vt  = vbatch["w_truth" ].to(device, non_blocking=True)
                    vp  = vbatch["w_prev"  ].to(device, non_blocking=True)
                    vo  = vbatch["w_obs_up"].to(device, non_blocking=True)
                    vB  = vt.shape[0]
                    vr  = _make_res_idx(vB, device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        val_loss_sum += diffusion.training_loss(vt, vp, vo, vr).item()
                    val_batches += 1

            val_loss = val_loss_sum / max(val_batches, 1)
            print(f"  [VAL] step={current_step}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            unet.train()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if current_step % save_interval == 0:
            ckpt = {
                "model":      unet.state_dict(),
                "ema_shadow": ema.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "step":       current_step,
                "val_loss":   best_val_loss,
            }
            save_path = ckpt_dir / f"oneshot_step_{current_step:06d}.pt"
            torch.save(ckpt, save_path)
            print(f"  Checkpoint saved: {save_path}")

    # ── Final EMA checkpoint ──────────────────────────────────────────────────
    online_state = copy.deepcopy(unet.state_dict())
    ema.copy_to(unet)
    torch.save(
        {
            "model":      unet.state_dict(),   # EMA weights
            "ema_shadow": ema.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "step":       total_steps,
            "val_loss":   best_val_loss,
        },
        ckpt_dir / "oneshot_ema.pt",
    )
    print(f"\n  EMA checkpoint saved: {ckpt_dir / 'oneshot_ema.pt'}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    unet.load_state_dict(online_state)
    ema.copy_to(unet)
    unet.eval()
    return diffusion
