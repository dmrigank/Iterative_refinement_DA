"""
Training script for the 1D one-shot diffusion SR baseline.

One-shot model: maps 64-pt -> 512-pt in a single diffusion pass,
conditioned on the previous 512-pt state.  No FNO, no resolution hierarchy.

Interface with GaussianDiffusion:
  training_loss(x0, u_forecast, u_coarse_up, res_idx)
    x0          <- u_truth    (target 512-pt field)
    u_forecast  <- u_prev     (previous 512-pt state)
    u_coarse_up <- u_obs_up   (64-pt observation upsampled to 512)
    res_idx     <- zeros(B)   (dummy; OneShotUNet1d ignores it)

Checkpoint format: checkpoints_oneshot_1d/oneshot_step_{N:06d}.pt
  {'model', 'ema_shadow', 'optimizer', 'scheduler', 'step', 'val_loss'}

Final EMA checkpoint: checkpoints_oneshot_1d/oneshot_ema.pt
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

from src.data.dataset_oneshot_1d import OneShotDataset1d
from src.models.unet_oneshot_1d import OneShotUNet1d
from src.models.diffusion import GaussianDiffusion


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average — shadow on CPU float32."""

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
        self.shadow = {k: v.clone().float() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _get_warmup_cosine_scheduler(
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

def _build_datasets(cfg: DictConfig) -> tuple[OneShotDataset1d, OneShotDataset1d]:
    data_dir = Path(cfg.data.data_dir)
    train_ds = OneShotDataset1d(data_dir / "train.pt")
    val_ds   = OneShotDataset1d(data_dir / "val.pt")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_oneshot_1d(cfg: DictConfig, device: torch.device) -> GaussianDiffusion:
    """Train the 1D one-shot diffusion SR model.

    Args:
        cfg:    Config object (from configs/oneshot_sr_1d.yaml).
        device: Compute device.

    Returns:
        GaussianDiffusion with EMA weights loaded (ready for inference).
    """
    print(f"\n{'='*60}")
    print("  Training 1D One-Shot Diffusion SR")
    print(f"{'='*60}")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps   = int(cfg.diffusion_training.steps)
    batch_size    = int(cfg.diffusion_training.batch_size)
    log_interval  = int(cfg.diffusion_training.log_interval)
    val_interval  = int(cfg.diffusion_training.val_interval)
    save_interval = int(cfg.diffusion_training.save_interval)
    warmup_steps  = int(cfg.diffusion_training.warmup_steps)
    ema_decay     = float(cfg.diffusion_training.ema_decay)

    # ── Datasets & loaders ──────────────────────────────────────────────────
    print("Building datasets...")
    train_ds, val_ds = _build_datasets(cfg)
    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=True,
    )

    def cycle(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch

    train_iter = cycle(train_loader)

    # ── Model ────────────────────────────────────────────────────────────────
    unet = OneShotUNet1d(
        base_channels  = int(cfg.unet.base_channels),
        channel_mults  = list(cfg.unet.channel_mults),
        n_res_blocks   = int(cfg.unet.n_res_blocks),
        n_groups       = int(cfg.unet.group_norm_groups),
        cond_embed_dim = int(cfg.unet.cond_embed_dim),
    ).to(device)

    diffusion = GaussianDiffusion(unet, cfg).to(device)
    n_params  = sum(p.numel() for p in unet.parameters())
    print(f"  OneShotUNet1d parameters: {n_params:,}")

    ema = EMAModel(unet, decay=ema_decay)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=float(cfg.diffusion_training.lr),
        weight_decay=float(cfg.diffusion_training.weight_decay),
    )
    scheduler = _get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # AMP scaler (CUDA only)
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume from latest checkpoint if present
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

    # ── Training loop ────────────────────────────────────────────────────────
    unet.train()
    running_loss  = 0.0
    best_val_loss = float("inf")

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        desc="OneShot-1D training",
        ncols=100,
    )

    for step in pbar:
        batch = next(train_iter)

        u_truth  = batch["u_truth" ].to(device, non_blocking=True)   # (B, 1, 512)
        u_prev   = batch["u_prev"  ].to(device, non_blocking=True)   # (B, 1, 512)
        u_obs_up = batch["u_obs_up"].to(device, non_blocking=True)   # (B, 1, 512)
        B = u_truth.shape[0]
        res_idx = torch.zeros(B, dtype=torch.long, device=device)     # dummy

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = diffusion.training_loss(u_truth, u_prev, u_obs_up, res_idx)
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
            avg_loss = running_loss / log_interval
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
                    vt = vbatch["u_truth" ].to(device, non_blocking=True)
                    vp = vbatch["u_prev"  ].to(device, non_blocking=True)
                    vo = vbatch["u_obs_up"].to(device, non_blocking=True)
                    vB = vt.shape[0]
                    vr = torch.zeros(vB, dtype=torch.long, device=device)
                    val_loss_sum += diffusion.training_loss(vt, vp, vo, vr).item()
                    val_batches  += 1
            val_loss = val_loss_sum / max(val_batches, 1)
            print(f"  [VAL] step={current_step}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            unet.train()

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

    # ── Final save: EMA weights ──────────────────────────────────────────────
    online_state = copy.deepcopy(unet.state_dict())
    ema.copy_to(unet)
    torch.save(
        {
            "model":      unet.state_dict(),
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
