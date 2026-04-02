"""
Training script for the 2D shared diffusion corrector G (Kraichnan testbed).

Strategy B pipeline (called after FNO training + forecast generation):
  1. Load FNO forecast data from data_2d/fno_forecasts/.
  2. Build DiffusionDataset2d: triplets (w_forecast, w_coarse_up, w_truth) across
     all 3 resolution pairs [(32->64, r=0), (64->128, r=1), (128->256, r=2)].
  3. Train G with DDPM epsilon-prediction loss.
  4. Maintain EMA of G weights (decay=0.9999), updated every step.
  5. Log train loss every log_interval steps.
  6. Compute val loss every val_interval steps.
  7. Save checkpoint every save_interval steps and at the end.

Checkpoint format:
  {
    'model':      state_dict,   <- EMA weights in diffusion_ema.pt
    'ema_shadow': dict,
    'optimizer':  state_dict,
    'scheduler':  state_dict,
    'step':       int,
    'val_loss':   float,
  }

Saved to:
  checkpoints_2d/diffusion_step_{N:06d}.pt   (periodic)
  checkpoints_2d/diffusion_ema.pt             (final EMA)
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

from src.data.dataset_2d import DiffusionDataset2d, ResolutionBatchSampler2d
from src.models.unet_2d import ConditionalUNet2d
from src.models.diffusion import GaussianDiffusion


# ---------------------------------------------------------------------------
# EMA  (identical to 1D version — reuse rather than import to keep 2D files self-contained)
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters.

    Shadow copy maintained on CPU float32 to save VRAM.
    """

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

def _build_diffusion_datasets_2d(
    cfg: DictConfig,
) -> tuple[DiffusionDataset2d, DiffusionDataset2d]:
    """Load FNO forecast files and ground-truth vorticity data; return train/val datasets.

    Resolution pairs (2D):
      (32 ->  64, r=0)
      (64 -> 128, r=1)
      (128-> 256, r=2)

    forecast[i] ≈ truth[i+1]  (teacher-forced single-step prediction)
    So we slice: truth_data = truth[:, 1:, :, :]  to align with forecast[:, 0:T-1, :, :]
    """
    data_dir     = Path(cfg.data.data_dir)          # "data_2d"
    forecast_dir = data_dir / "fno_forecasts"

    resolution_pairs = [
        (32,   64, 0),
        (64,  128, 1),
        (128, 256, 2),
    ]

    datasets: dict[str, DiffusionDataset2d] = {}

    for split in ("train", "val"):
        raw = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=True)

        # truth_data: dict res -> (n_traj, T-1, ny, nx)
        # Slice [:, 1:] to align GT targets with teacher-forced forecasts
        truth_data: dict[int, torch.Tensor] = {
            32:  raw["w_32" ][:, 1:, :, :],
            64:  raw["w_64" ][:, 1:, :, :],
            128: raw["w_128"][:, 1:, :, :],
            256: raw["w_256"][:, 1:, :, :],
        }

        # forecast_data: dict target_res -> (n_traj, T-1, ny, nx)
        forecast_data: dict[int, torch.Tensor] = {}
        for _, target_res, _ in resolution_pairs:
            forecast_data[target_res] = torch.load(
                forecast_dir / f"fno_forecast_{target_res}_{split}.pt",
                map_location="cpu",
                weights_only=True,
            )  # (n_traj, T-1, ny, nx)

        datasets[split] = DiffusionDataset2d(truth_data, forecast_data, resolution_pairs)

    return datasets["train"], datasets["val"]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_diffusion_2d(cfg: DictConfig, device: torch.device) -> GaussianDiffusion:
    """Train the shared 2D diffusion corrector G (Strategy B).

    Args:
        cfg:    Full kraichnan.yaml config.
        device: Compute device.

    Returns:
        GaussianDiffusion with EMA weights loaded (ready for inference).
    """
    print(f"\n{'='*60}")
    print("  Training 2D Diffusion Corrector G  (Strategy B)")
    print(f"{'='*60}")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_steps   = int(cfg.diffusion_training.steps)           # 200 000
    batch_size    = int(cfg.diffusion_training.batch_size)      # 8
    log_interval  = int(cfg.diffusion_training.log_interval)    # 500
    val_interval  = int(cfg.diffusion_training.val_interval)    # 5 000
    save_interval = int(cfg.diffusion_training.save_interval)   # 25 000
    warmup_steps  = int(cfg.diffusion_training.warmup_steps)    # 3 000
    ema_decay     = float(cfg.diffusion_training.ema_decay)     # 0.9999

    # ── Input noise augmentation ─────────────────────────────────────────────
    dt = cfg.diffusion_training
    augment        = bool(dt.get("augment_inputs", True))
    augment_prob   = float(dt.get("augment_prob", 0.5))
    # noise_scales[r] = (forecast_std, coarse_std) for stage index r
    raw_scales     = dt.get("augment_noise_scales", [[0.3, 0.05], [0.3, 0.40], [0.3, 0.40]])
    noise_scales   = {i: (float(s[0]), float(s[1])) for i, s in enumerate(raw_scales)}
    if augment:
        print(f"  Input augmentation ON  (prob={augment_prob}, scales={noise_scales})")
    else:
        print("  Input augmentation OFF")

    # ── Datasets & loaders ───────────────────────────────────────────────────
    print("Building 2D diffusion datasets...")
    train_ds, val_ds = _build_diffusion_datasets_2d(cfg)
    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
    for b in range(train_ds.n_blocks):
        print(f"    Block {b}: {train_ds.block_len(b):,} train samples")

    train_sampler = ResolutionBatchSampler2d(train_ds, batch_size=batch_size,     drop_last=True)
    val_sampler   = ResolutionBatchSampler2d(val_ds,   batch_size=batch_size * 2, drop_last=False)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
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
    # Patch GaussianDiffusion's type hint to accept ConditionalUNet2d
    # (the class is structurally compatible — only the annotation is 1D-specific)
    import src.models.diffusion as _dm
    _original_type = getattr(_dm, "ConditionalUNet1d", None)
    _dm.ConditionalUNet1d = nn.Module  # loosen type for isinstance checks if any

    unet      = ConditionalUNet2d(cfg).to(device)
    diffusion = GaussianDiffusion(unet, cfg).to(device)
    n_params  = sum(p.numel() for p in unet.parameters())
    print(f"  ConditionalUNet2d parameters: {n_params:,}")

    ema = EMAModel(unet, decay=ema_decay)

    # ── AMP scaler for mixed precision (float16 forward, float32 loss) ───────
    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=float(cfg.diffusion_training.lr),
        weight_decay=float(cfg.diffusion_training.weight_decay),
    )
    scheduler = _get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # ── Resume from latest checkpoint if present ─────────────────────────────
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
        desc="Diffusion2D training",
        ncols=110,
    )

    for step in pbar:
        batch = next(train_iter)

        w_truth    = batch["w_truth"   ].to(device, non_blocking=True)   # (B, 1, ny, nx)
        w_forecast = batch["w_forecast"].to(device, non_blocking=True)   # (B, 1, ny, nx)
        w_coarse   = batch["w_coarse"  ].to(device, non_blocking=True)   # (B, 1, ny, nx)
        res_idx    = batch["res_idx"   ].to(device, non_blocking=True)   # (B,)

        # ── Input noise augmentation ─────────────────────────────────────────
        # Each sample in the batch has a single res_idx (ResolutionBatchSampler2d
        # guarantees all items in a batch share the same stage), so index [0].
        if augment:
            r = int(res_idx[0].item())
            fc_std, co_std = noise_scales[r]
            if torch.rand(1).item() < augment_prob:
                w_forecast = w_forecast + fc_std * torch.randn_like(w_forecast)
            if torch.rand(1).item() < augment_prob:
                w_coarse = w_coarse + co_std * torch.randn_like(w_coarse)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = diffusion.training_loss(w_truth, w_forecast, w_coarse, res_idx)

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
                    vt = vbatch["w_truth"   ].to(device, non_blocking=True)
                    vf = vbatch["w_forecast"].to(device, non_blocking=True)
                    vc = vbatch["w_coarse"  ].to(device, non_blocking=True)
                    vr = vbatch["res_idx"   ].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        val_loss_sum += diffusion.training_loss(vt, vf, vc, vr).item()
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
            save_path = ckpt_dir / f"diffusion_step_{current_step:06d}.pt"
            torch.save(ckpt, save_path)
            print(f"  Checkpoint saved: {save_path}")

    # ── Final save: EMA weights ───────────────────────────────────────────────
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

    # Restore online weights then swap to EMA for return
    unet.load_state_dict(online_state)
    ema.copy_to(unet)
    unet.eval()
    return diffusion
