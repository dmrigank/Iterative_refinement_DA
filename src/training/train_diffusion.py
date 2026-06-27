"""
Training script for the shared diffusion corrector G (Strategy B).

Strategy B pipeline (called after FNO training):
  1. Load FNO forecast data from data/fno_forecasts/.
  2. Build DiffusionDataset: triplets (u_forecast, u_coarse_up, u_truth) across
     all 3 resolution pairs [(64->128, r=0), (128->256, r=1), (256->512, r=2)].
  3. Train G with DDPM epsilon-prediction loss.
  4. Maintain EMA of G weights (decay=0.9999), updated every step.
  5. Log train loss every log_interval steps.
  6. Compute val loss every val_interval steps.
  7. Save checkpoint every save_interval steps and at the end.

Checkpoint format saved to checkpoints/diffusion_{step}.pt and diffusion_ema.pt:
  {
    'model':      model.state_dict(),
    'ema_shadow': ema shadow params,
    'optimizer':  optimizer.state_dict(),
    'scheduler':  scheduler.state_dict(),
    'step':       int,
    'val_loss':   float,
  }
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

from src.data.dataset import DiffusionDataset, ResolutionBatchSampler
from src.models.unet import ConditionalUNet1d
from src.models.diffusion import GaussianDiffusion


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights updated as:
        shadow = decay * shadow + (1 - decay) * param

    Usage:
        ema = EMAModel(model, decay=0.9999)
        # after each optimizer.step():
        ema.update(model)
        # to evaluate with EMA weights:
        ema.copy_to(model)
        # restore online weights afterward if still training:
        # (save online state_dict before copy_to if needed)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Store a deep copy of the initial parameters on CPU to save VRAM
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.cpu().clone().float()
            for name, param in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters from current model parameters."""
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data.cpu().float(), alpha=1.0 - self.decay
            )

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA (shadow) parameters into model in-place."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name].to(param.device).to(param.dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone().float() for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for warmup_steps, then cosine decay to 0.

    Args:
        optimizer:    The optimizer to schedule.
        warmup_steps: Steps over which LR rises linearly from 0 to base_lr.
        total_steps:  Total training steps (LR reaches 0 at the end).

    Returns:
        LambdaLR scheduler (call .step() once per gradient step).
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_diffusion_datasets(
    cfg: DictConfig,
) -> tuple[DiffusionDataset, DiffusionDataset]:
    """Load FNO forecast files and ground-truth data; return train/val datasets.

    forecast[t]  ≈  truth[t+1]  (teacher-forced single-step prediction)
    So we align: truth_data = truth[:, 1:, :]  to pair with forecast[:, 0:T-1, :]

    Resolution pairs: (coarse_res, target_res, res_idx)
      (64, 128, 0), (128, 256, 1), (256, 512, 2)
    """
    data_dir     = Path(cfg.data.data_dir)
    forecast_dir = data_dir / "fno_forecasts"

    resolution_pairs = [
        (64,  128, 0),
        (128, 256, 1),
        (256, 512, 2),
    ]

    datasets = {}
    for split in ("train", "val"):
        raw = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=True)

        # truth_data: dict res -> (n_traj, T-1, N_r), aligned to forecast targets
        truth_data: dict[int, torch.Tensor] = {
            64:  raw["u_64" ][:, 1:, :],   # coarse observations for pair 0
            128: raw["u_128"][:, 1:, :],   # target for pair 0; coarse for pair 1
            256: raw["u_256"][:, 1:, :],   # target for pair 1; coarse for pair 2
            512: raw["u_512"][:, 1:, :],   # target for pair 2
        }

        # forecast_data: dict target_res -> (n_traj, T-1, N_r)
        forecast_data: dict[int, torch.Tensor] = {}
        for _, target_res, _ in resolution_pairs:
            forecast_data[target_res] = torch.load(
                forecast_dir / f"fno_forecast_{target_res}_{split}.pt",
                map_location="cpu",
                weights_only=True,
            )  # (n_traj, T-1, N_r)

        datasets[split] = DiffusionDataset(truth_data, forecast_data, resolution_pairs)

    return datasets["train"], datasets["val"]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_diffusion(cfg: DictConfig, device: torch.device) -> GaussianDiffusion:
    """Train the shared diffusion corrector G (Strategy B).

    Args:
        cfg:    Full config object.
        device: Compute device.

    Returns:
        GaussianDiffusion with EMA weights loaded (ready for inference).
    """
    print(f"\n{'='*60}")
    print("  Training Diffusion Corrector G  (Strategy B)")
    print(f"{'='*60}")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps   = int(cfg.diffusion_training.steps)          # 150 000
    batch_size    = int(cfg.diffusion_training.batch_size)     # 16
    log_interval  = int(cfg.diffusion_training.log_interval)   # 500
    val_interval  = int(cfg.diffusion_training.val_interval)   # 5 000
    save_interval = int(cfg.diffusion_training.save_interval)  # 25 000
    warmup_steps  = int(cfg.diffusion_training.warmup_steps)   # 2 000
    ema_decay     = float(cfg.diffusion_training.ema_decay)    # 0.9999

    # ── Input noise augmentation ─────────────────────────────────────────────
    dt = cfg.diffusion_training
    augment      = bool(dt.get("augment_inputs", False))
    augment_prob = float(dt.get("augment_prob", 1.0))
    raw_scales   = dt.get("augment_noise_scales", [[0.027, 0.0], [0.027, 0.003], [0.027, 0.004]])
    noise_scales = {i: (float(s[0]), float(s[1])) for i, s in enumerate(raw_scales)}
    if augment:
        print(f"  Input augmentation ON  (prob={augment_prob}, scales={noise_scales})")
    else:
        print("  Input augmentation OFF")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    print("Building datasets...")
    train_ds, val_ds = _build_diffusion_datasets(cfg)
    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    # ResolutionBatchSampler ensures every batch has uniform spatial size
    # (items from the same resolution block only).
    train_sampler = ResolutionBatchSampler(train_ds, batch_size=batch_size, drop_last=True)
    val_sampler   = ResolutionBatchSampler(val_ds,   batch_size=batch_size * 2, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    def cycle(loader: DataLoader):
        while True:
            for batch in loader:
                yield batch

    train_iter = cycle(train_loader)

    # ── Model ────────────────────────────────────────────────────────────────
    unet      = ConditionalUNet1d(cfg).to(device)
    diffusion = GaussianDiffusion(unet, cfg).to(device)
    n_params  = sum(p.numel() for p in unet.parameters())
    print(f"  U-Net parameters: {n_params:,}")

    ema = EMAModel(unet, decay=ema_decay)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=float(cfg.diffusion_training.lr),
        weight_decay=float(cfg.diffusion_training.weight_decay),
    )
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # Resume from latest checkpoint if present
    start_step = 0
    latest_ckpts = sorted(ckpt_dir.glob("diffusion_step_*.pt"))
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
        desc="Diffusion training",
        ncols=100,
    )

    for step in pbar:
        batch = next(train_iter)

        u_truth    = batch["u_truth"   ].to(device, non_blocking=True)   # (B,1,N)
        u_forecast = batch["u_forecast"].to(device, non_blocking=True)   # (B,1,N)
        u_coarse   = batch["u_coarse"  ].to(device, non_blocking=True)   # (B,1,N)
        res_idx    = batch["res_idx"   ].to(device, non_blocking=True)   # (B,)

        # ── Input noise augmentation ─────────────────────────────────────────
        # ResolutionBatchSampler guarantees all items in a batch share the same
        # stage, so res_idx[0] gives the stage for the whole batch.
        if augment:
            r = int(res_idx[0].item())
            fc_std, co_std = noise_scales[r]
            if torch.rand(1).item() < augment_prob:
                u_forecast = u_forecast + fc_std * torch.randn_like(u_forecast)
            if co_std > 0.0:
                u_coarse = u_coarse + co_std * torch.randn_like(u_coarse)

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.training_loss(u_truth, u_forecast, u_coarse, res_idx)
        loss.backward()
        nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()
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
                    vt  = vbatch["u_truth"   ].to(device, non_blocking=True)
                    vf  = vbatch["u_forecast"].to(device, non_blocking=True)
                    vc  = vbatch["u_coarse"  ].to(device, non_blocking=True)
                    vr  = vbatch["res_idx"   ].to(device, non_blocking=True)
                    val_loss_sum += diffusion.training_loss(vt, vf, vc, vr).item()
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
            save_path = ckpt_dir / f"diffusion_step_{current_step:06d}.pt"
            torch.save(ckpt, save_path)
            print(f"  Checkpoint saved: {save_path}")

    # ── Final save: EMA weights ───────────────────────────────────────────────
    # Copy EMA into model, save, then restore online weights
    online_state = copy.deepcopy(unet.state_dict())
    ema.copy_to(unet)
    torch.save(
        {
            "model":      unet.state_dict(),   # <- EMA weights
            "ema_shadow": ema.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "step":       total_steps,
            "val_loss":   best_val_loss,
        },
        ckpt_dir / "diffusion_ema.pt",
    )
    print(f"\n  EMA checkpoint saved: {ckpt_dir / 'diffusion_ema.pt'}")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # Restore online weights and return with EMA loaded
    unet.load_state_dict(online_state)
    ema.copy_to(unet)  # final model has EMA weights
    unet.eval()
    return diffusion
