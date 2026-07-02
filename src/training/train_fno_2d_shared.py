"""
Training script for the SHARED 2D FNO forecaster — one model, all resolutions.

Unlike train_fno_2d.py (three independent FNO2d instances, one per
resolution), this trains a single SharedFNO2d on batches from all three
resolutions (64, 128, 256) within every epoch. Each batch is spatially
homogeneous (a SharedFNO2d forward pass takes one (B, 1, ny, nx) tensor at
a time — different resolutions cannot be concatenated), so one training
epoch round-robins one full pass over each resolution's DataLoader, taking
one optimizer step per batch exactly as train_fno_2d.py does per-resolution.

Equal weighting across resolutions: each resolution contributes the same
NUMBER of (trajectory, timestep) samples (only spatial size differs), so
batches-per-epoch is similar across resolutions and no extra reweighting
is applied.

After training, generate_fno_forecasts_2d_shared runs the shared model in
teacher-forced mode over all training trajectories and saves outputs to
data_2d/fno_forecasts_shared/ — a SEPARATE directory from the existing
data_2d/fno_forecasts/ (produced by the three separate FNOs), so the
original Strategy-B diffusion dataset is never overwritten.

Checkpoint format (identical to train_fno_2d.py):
  {
    'model':     state_dict,
    'optimizer': state_dict,
    'epoch':     int,
    'val_loss':  float,   # averaged equally across the 3 resolutions
  }
  Saved to: checkpoints_2d/fno_shared.pt  (best val loss)

Forecast file format:
  Tensor of shape (n_traj, T-1, ny, nx), float32
  Saved to: data_2d/fno_forecasts_shared/fno_forecast_{resolution}_{split}.pt
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.dataset_2d import FNODataset2d
from src.models.fno_2d_shared import SharedFNO2d

RESOLUTIONS = [64, 128, 256]


# ---------------------------------------------------------------------------
# Train the shared FNO across all resolutions
# ---------------------------------------------------------------------------

def train_fno_2d_shared(
    cfg: DictConfig,
    device: torch.device,
    k_max_shared: int = 64,
) -> SharedFNO2d:
    """Train one SharedFNO2d on interleaved batches from 64/128/256×256 data.

    Args:
        cfg:          Full 2D config (kraichnan.yaml).
        device:       Compute device.
        k_max_shared: Fixed spectral mode truncation (default 64, matching
                      the existing 256×256 FNO's cutoff).

    Returns:
        SharedFNO2d loaded with best-checkpoint weights.
    """
    print(f"\n{'='*60}")
    print(f"  Training SharedFNO2d  resolutions={RESOLUTIONS}  k_max_shared={k_max_shared}")
    print(f"{'='*60}")

    ckpt_dir  = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "fno_shared.pt"

    # ── Data: one DataLoader per resolution ──────────────────────────────────
    data_dir  = Path(cfg.data.data_dir)
    train_raw = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_raw   = torch.load(data_dir / "val.pt",   map_location="cpu", weights_only=True)

    bs = int(cfg.fno_training.batch_size)

    train_loaders: dict[int, DataLoader] = {}
    val_loaders:   dict[int, DataLoader] = {}
    for res in RESOLUTIONS:
        key = f"w_{res}"
        train_ds = FNODataset2d(train_raw[key], res)
        val_ds   = FNODataset2d(val_raw[key],   res)
        train_loaders[res] = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
        )
        val_loaders[res] = DataLoader(
            val_ds, batch_size=bs * 2, shuffle=False,
            num_workers=2, pin_memory=(device.type == "cuda"),
        )
        print(f"  res={res:3d}×{res}  train={len(train_ds):,}  val={len(val_ds):,}  "
              f"steps/epoch={len(train_loaders[res]):,}")

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model    = SharedFNO2d(cfg, k_max_shared=k_max_shared).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.fno_training.lr),
        weight_decay=float(cfg.fno_training.weight_decay),
    )
    epochs    = int(cfg.fno_training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()

        # Build one combined, shuffled step order for this epoch: a flat list
        # of (resolution, batch) pairs, interleaved so the optimizer alternates
        # across resolutions rather than completing one resolution at a time.
        iters = {res: iter(loader) for res, loader in train_loaders.items()}
        n_steps_per_res = {res: len(loader) for res, loader in train_loaders.items()}
        step_order: list[int] = []
        for res in RESOLUTIONS:
            step_order.extend([res] * n_steps_per_res[res])
        random.shuffle(step_order)

        train_loss_sum: dict[int, float] = {res: 0.0 for res in RESOLUTIONS}
        train_loss_n:   dict[int, int]   = {res: 0   for res in RESOLUTIONS}

        pbar = tqdm(step_order, desc=f"Epoch {epoch:3d}/{epochs} [train]",
                    leave=False, ncols=100)
        for res in pbar:
            w_in, w_tgt = next(iters[res])
            w_in  = w_in.to(device)
            w_tgt = w_tgt.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(w_in)
            loss = criterion(pred, w_tgt)
            loss.backward()
            optimizer.step()

            train_loss_sum[res] += loss.item()
            train_loss_n[res]   += 1
            pbar.set_postfix(res=res, loss=f"{loss.item():.4e}")

        scheduler.step()

        # ── Validate (per resolution, then average) ──────────────────────────
        model.eval()
        val_loss_per_res: dict[int, float] = {}
        with torch.no_grad():
            for res in RESOLUTIONS:
                vsum, vn = 0.0, 0
                for w_in, w_tgt in val_loaders[res]:
                    w_in  = w_in.to(device)
                    w_tgt = w_tgt.to(device)
                    vsum += criterion(model(w_in), w_tgt).item()
                    vn   += 1
                val_loss_per_res[res] = vsum / max(vn, 1)

        val_loss = sum(val_loss_per_res.values()) / len(val_loss_per_res)
        lr_now   = scheduler.get_last_lr()[0]

        train_loss_str = "  ".join(
            f"train@{res}={train_loss_sum[res] / max(train_loss_n[res], 1):.4e}"
            for res in RESOLUTIONS
        )
        val_loss_str = "  ".join(
            f"val@{res}={val_loss_per_res[res]:.4e}" for res in RESOLUTIONS
        )
        print(
            f"  Epoch {epoch:3d}/{epochs}  {train_loss_str}  {val_loss_str}  "
            f"val_avg={val_loss:.4e}  lr={lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch":     epoch,
                    "val_loss":  val_loss,
                },
                ckpt_path,
            )

    print(f"  Best val loss (avg over resolutions): {best_val_loss:.4e}  ->  {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generate FNO forecast dataset (Strategy B) for the shared model
# ---------------------------------------------------------------------------

def generate_fno_forecasts_2d_shared(
    cfg: DictConfig,
    model: SharedFNO2d,
    device: torch.device,
) -> None:
    """Run the trained shared FNO in teacher-forced mode to produce forecasts.

    Mirrors generate_fno_forecasts_2d in train_fno_2d.py, but uses the single
    shared model for every resolution and writes to a SEPARATE output
    directory (data_2d/fno_forecasts_shared/) so the original three-FNO
    forecast dataset in data_2d/fno_forecasts/ is left untouched.

    Output files: data_2d/fno_forecasts_shared/fno_forecast_{resolution}_{split}.pt
    Shape: (n_traj, T-1, ny, nx), float32
    """
    out_dir  = Path(cfg.data.data_dir) / "fno_forecasts_shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data.data_dir)

    model.eval()

    for split in ("train", "val"):
        raw = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=True)

        for res in RESOLUTIONS:
            key = f"w_{res}"
            data = raw[key]               # (n_traj, T, ny, nx)
            n_traj, T, ny, nx = data.shape

            u_in = data[:, :-1, :, :].to(device)   # (n_traj, T-1, ny, nx)
            forecasts = torch.empty(n_traj, T - 1, ny, nx, dtype=torch.float32)

            with torch.no_grad():
                time_chunk = 32
                for traj_i in range(n_traj):
                    for t_start in range(0, T - 1, time_chunk):
                        t_end   = min(t_start + time_chunk, T - 1)
                        u_slice = u_in[traj_i, t_start:t_end]
                        u_flat  = u_slice.unsqueeze(1)
                        pred    = model(u_flat)
                        forecasts[traj_i, t_start:t_end] = pred.squeeze(1).cpu()

            save_path = out_dir / f"fno_forecast_{res}_{split}.pt"
            torch.save(forecasts, save_path)

            truth_next = data[:, 1:, :, :].cpu()
            mse = ((forecasts - truth_next) ** 2).mean().item()
            print(
                f"  Saved {save_path}  "
                f"shape={tuple(forecasts.shape)}  "
                f"MSE vs truth={mse:.4e}"
            )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import numpy as np
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Train the shared 2D FNO forecaster")
    parser.add_argument("--config",       type=str, default="configs/kraichnan.yaml")
    parser.add_argument("--k_max_shared", type=int, default=64)
    parser.add_argument("--skip_forecasts", action="store_true",
                        help="Skip forecast generation after training")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if str(cfg.device) != "auto":
        device_str = str(cfg.device)
    device = torch.device(device_str)
    print(f"Device: {device}")

    model = train_fno_2d_shared(cfg, device, k_max_shared=args.k_max_shared)

    if not args.skip_forecasts:
        print("\nGenerating shared-FNO forecast dataset (Strategy B input)...")
        generate_fno_forecasts_2d_shared(cfg, model, device)
