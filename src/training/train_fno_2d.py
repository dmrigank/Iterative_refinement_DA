"""
Training script for 2D FNO forecasters (one per resolution level).

Trains F_64 (64×64), F_128 (128×128), F_256 (256×256) independently on
ground-truth consecutive snapshot pairs from the 2D training split.

After training all FNOs, generate_fno_forecasts_2d runs each FNO in
teacher-forced mode over all training trajectories and saves the outputs to
data_2d/fno_forecasts/ — these become the inputs for diffusion training
(Strategy B).

Checkpoint format:
  {
    'model':     state_dict,
    'optimizer': state_dict,
    'epoch':     int,
    'val_loss':  float,
  }
  Saved to: checkpoints_2d/fno_{resolution}.pt  (best val loss)

Forecast file format:
  Tensor of shape (n_traj, T-1, ny, nx), float32
  Saved to: data_2d/fno_forecasts/fno_forecast_{resolution}_{split}.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.dataset_2d import FNODataset2d
from src.models.fno_2d import FNO2d


# ---------------------------------------------------------------------------
# Train one FNO
# ---------------------------------------------------------------------------

def train_fno_2d(
    cfg: DictConfig,
    resolution: int,
    device: torch.device,
) -> FNO2d:
    """Train a single 2D FNO at the given resolution.

    Args:
        cfg:        Full 2D config (kraichnan.yaml).
        resolution: Target resolution N (square: N×N).
        device:     Compute device.

    Returns:
        FNO2d loaded with best-checkpoint weights.
    """
    print(f"\n{'='*60}")
    print(f"  Training FNO2d  resolution={resolution}×{resolution}  "
          f"k_max={resolution//4}")
    print(f"{'='*60}")

    ckpt_dir  = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fno_{resolution}.pt"

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir  = Path(cfg.data.data_dir)
    train_raw = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_raw   = torch.load(data_dir / "val.pt",   map_location="cpu", weights_only=True)

    key = f"w_{resolution}"
    train_ds = FNODataset2d(train_raw[key], resolution)
    val_ds   = FNODataset2d(val_raw[key],   resolution)

    bs = int(cfg.fno_training.batch_size)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
    print(f"  Batch size: {bs}  |  Steps/epoch: {len(train_loader):,}")

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model    = FNO2d(cfg, resolution).to(device)
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
        # Train
        model.train()
        train_loss_sum = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{epochs} [train]",
            leave=False, ncols=90,
        )
        for w_in, w_tgt in pbar:
            w_in  = w_in.to(device)
            w_tgt = w_tgt.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(w_in)
            loss = criterion(pred, w_tgt)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4e}")

        scheduler.step()
        train_loss = train_loss_sum / len(train_loader)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for w_in, w_tgt in val_loader:
                w_in  = w_in.to(device)
                w_tgt = w_tgt.to(device)
                val_loss_sum += criterion(model(w_in), w_tgt).item()
        val_loss = val_loss_sum / len(val_loader)
        lr_now   = scheduler.get_last_lr()[0]

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.4e}  val={val_loss:.4e}  lr={lr_now:.2e}"
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

    print(f"  Best val loss: {best_val_loss:.4e}  ->  {ckpt_path}")

    # Load best weights before returning
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Train all resolution levels
# ---------------------------------------------------------------------------

def train_all_fnos_2d(cfg: DictConfig, device: torch.device) -> dict[int, FNO2d]:
    """Train FNO2d for all resolution levels in sequence.

    Returns:
        Dict mapping resolution -> trained FNO2d (best weights loaded).
    """
    # Exclude the coarsest level (32) — no FNO is trained at the observation res
    resolutions = [r for r in cfg.data.resolution_levels if r != 32]  # [64, 128, 256]
    fnos: dict[int, FNO2d] = {}
    for res in resolutions:
        fnos[res] = train_fno_2d(cfg, res, device)
    return fnos


# ---------------------------------------------------------------------------
# Generate FNO forecast dataset (Strategy B)
# ---------------------------------------------------------------------------

def generate_fno_forecasts_2d(
    cfg: DictConfig,
    fnos: dict[int, FNO2d],
    device: torch.device,
) -> None:
    """Run trained FNOs in teacher-forced mode to produce forecast datasets.

    For each resolution r and each split (train, val), runs the FNO one step
    at a time over every ground-truth input w_t and records the predicted
    w_{t+1}.  This is teacher forcing — the FNO always receives the GT field,
    never its own previous output.

    Output files: data_2d/fno_forecasts/fno_forecast_{resolution}_{split}.pt
    Shape: (n_traj, T-1, ny, nx), float32

    Args:
        cfg:   Full 2D config.
        fnos:  Dict mapping resolution -> trained FNO2d.
        device: Compute device.
    """
    out_dir  = Path(cfg.data.data_dir) / "fno_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data.data_dir)

    for split in ("train", "val"):
        raw = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=True)

        for res, model in fnos.items():
            key = f"w_{res}"
            data = raw[key]               # (n_traj, T, ny, nx)
            n_traj, T, ny, nx = data.shape

            model.eval()
            # Teacher-forced: input = GT w_t, target ≈ w_{t+1}
            u_in = data[:, :-1, :, :].to(device)   # (n_traj, T-1, ny, nx)
            forecasts = torch.empty(n_traj, T - 1, ny, nx, dtype=torch.float32)

            with torch.no_grad():
                # Process one trajectory at a time, chunked over time steps
                # to avoid OOM at large resolutions (e.g. 256×256, T-1=199)
                time_chunk = 32  # time steps per GPU batch
                for traj_i in range(n_traj):
                    for t_start in range(0, T - 1, time_chunk):
                        t_end   = min(t_start + time_chunk, T - 1)
                        u_slice = u_in[traj_i, t_start:t_end]   # (tc, ny, nx)
                        u_flat  = u_slice.unsqueeze(1)           # (tc, 1, ny, nx)
                        pred    = model(u_flat)                  # (tc, 1, ny, nx)
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
