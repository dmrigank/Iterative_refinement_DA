"""
Training script for the EDSR super-resolution baseline.

Trains EDSR to map 32x32 -> 256x256 vorticity with L1 loss.
No temporal context, no diffusion — each frame is upscaled independently.

Training loop:
  1. Build EDSRDataset for train and val splits.
  2. Standard DataLoader with shuffle=True.
  3. AdamW optimiser, L1 loss, linear warmup + cosine decay.
  4. Log every log_interval steps, validate every val_interval steps.
  5. Save periodic checkpoint and final best checkpoint.

Checkpoint format (checkpoints_edsr/edsr_step_{N:06d}.pt and edsr_best.pt):
  {
    'model':     model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'step':      int,
    'val_loss':  float,
  }
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.dataset_edsr import EDSRDataset
from src.models.edsr import EDSR


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
# Main training function
# ---------------------------------------------------------------------------

def train_edsr(cfg: DictConfig, device: torch.device) -> EDSR:
    """Train the EDSR SR baseline.

    Args:
        cfg:    Full edsr.yaml config object.
        device: Compute device.

    Returns:
        Trained EDSR model in eval mode with best-checkpoint weights loaded.
    """
    print(f"\n{'='*60}")
    print("  Training EDSR SR Baseline (32x32 -> 256x256)")
    print(f"{'='*60}")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tr          = cfg.training
    total_steps   = int(tr.steps)
    batch_size    = int(tr.batch_size)
    log_interval  = int(tr.log_interval)
    val_interval  = int(tr.val_interval)
    save_interval = int(tr.save_interval)
    warmup_steps  = int(tr.warmup_steps)
    grad_clip     = float(tr.grad_clip)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    data_dir = Path(cfg.data.data_dir)
    print("Building datasets...")
    train_ds = EDSRDataset(data_dir / "train.pt")
    val_ds   = EDSRDataset(data_dir / "val.pt")
    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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
    model = EDSR(
        n_resblocks  = int(cfg.model.n_resblocks),
        n_feats      = int(cfg.model.n_feats),
        scale        = int(cfg.model.scale),
        res_scale    = float(cfg.model.res_scale),
        padding_mode = str(cfg.model.padding_mode),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  EDSR parameters: {n_params:,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = nn.L1Loss() if str(tr.loss).lower() == "l1" else nn.MSELoss()
    print(f"  Loss: {tr.loss.upper()}")

    # ── AMP ──────────────────────────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr.lr),
        weight_decay=float(tr.weight_decay),
    )
    scheduler = _get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # ── Resume from latest checkpoint if present ─────────────────────────────
    start_step = 0
    latest_ckpts = sorted(ckpt_dir.glob("edsr_step_*.pt"))
    if latest_ckpts:
        ckpt_path = latest_ckpts[-1]
        print(f"  Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"])
        print(f"  Resumed at step {start_step}")

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    running_loss  = 0.0
    best_val_loss = float("inf")

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        desc="EDSR training",
        ncols=100,
    )

    for step in pbar:
        batch  = next(train_iter)
        w_lr   = batch["w_lr"].to(device, non_blocking=True)   # (B, 1, 32,  32)
        w_hr   = batch["w_hr"].to(device, non_blocking=True)   # (B, 1, 256, 256)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            w_sr   = model(w_lr)          # (B, 1, 256, 256)
            loss   = loss_fn(w_sr, w_hr)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

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
            model.eval()
            val_loss_sum = 0.0
            val_batches  = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vl = vbatch["w_lr"].to(device, non_blocking=True)
                    vh = vbatch["w_hr"].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        val_loss_sum += loss_fn(model(vl), vh).item()
                    val_batches += 1
            val_loss = val_loss_sum / max(val_batches, 1)
            print(f"  [VAL] step={current_step}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best checkpoint
                torch.save(
                    {
                        "model":     model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step":      current_step,
                        "val_loss":  best_val_loss,
                    },
                    ckpt_dir / "edsr_best.pt",
                )
                print(f"  Best checkpoint saved (val_loss={best_val_loss:.4f})")

            model.train()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if current_step % save_interval == 0:
            torch.save(
                {
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step":      current_step,
                    "val_loss":  best_val_loss,
                },
                ckpt_dir / f"edsr_step_{current_step:06d}.pt",
            )
            print(f"  Checkpoint saved: edsr_step_{current_step:06d}.pt")

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step":      total_steps,
            "val_loss":  best_val_loss,
        },
        ckpt_dir / "edsr_final.pt",
    )
    print(f"\n  Final checkpoint saved: {ckpt_dir / 'edsr_final.pt'}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # Load best weights before returning
    best_ckpt = ckpt_dir / "edsr_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded best weights (step {ckpt['step']}, val_loss={ckpt['val_loss']:.4f})")

    model.eval()
    return model
