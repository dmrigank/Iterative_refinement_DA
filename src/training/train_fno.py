"""
Training script for FNO forecasters (one per resolution level).

Trains F_1 (128), F_2 (256), F_3 (512) independently on ground-truth
consecutive snapshot pairs from the training split.

Checkpoint format: {
    'model':     state_dict,
    'optimizer': state_dict,
    'epoch':     int,
    'val_loss':  float,
}
Saved to: checkpoints/fno_{resolution}.pt  (best val loss)

After training, `generate_fno_forecasts` runs each FNO autoregressively on
all training trajectories and saves the results to data/fno_forecasts/.
These are the inputs for diffusion training (Strategy B).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.dataset import FNODataset
from src.models.fno import FNO1d


# ---------------------------------------------------------------------------
# Training one FNO
# ---------------------------------------------------------------------------

def train_fno(
    cfg: DictConfig,
    resolution: int,
    device: torch.device,
) -> FNO1d:
    """Train a single FNO at the given resolution.

    Loads train/val splits from disk, runs the training loop, saves the
    best-val-loss checkpoint, and returns the trained model.

    Args:
        cfg: Full config object.
        resolution: Target spatial resolution N_r (128, 256, or 512).
        device: Compute device.

    Returns:
        FNO1d loaded with the best checkpoint weights.
    """
    print(f"\n{'='*60}")
    print(f"  Training FNO  resolution={resolution}  k_max={resolution//4}")
    print(f"{'='*60}")

    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"fno_{resolution}.pt"

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = Path(cfg.data.data_dir)
    train_raw = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_raw   = torch.load(data_dir / "val.pt",   map_location="cpu", weights_only=True)

    key = f"u_{resolution}"
    train_ds = FNODataset(train_raw[key], resolution)
    val_ds   = FNODataset(val_raw[key],   resolution)

    bs = int(cfg.fno_training.batch_size)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
    )

    print(f"  Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")
    print(f"  Batch size: {bs}  |  Steps/epoch: {len(train_loader):,}")

    # ── Model + optimiser ─────────────────────────────────────────────────
    model = FNO1d(cfg, resolution).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.fno_training.lr),
        weight_decay=float(cfg.fno_training.weight_decay),
    )
    epochs = int(cfg.fno_training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{epochs} [train]",
            leave=False,
            ncols=90,
        )
        for u_in, u_tgt in pbar:
            u_in  = u_in.to(device)
            u_tgt = u_tgt.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(u_in)
            loss = criterion(pred, u_tgt)
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
            for u_in, u_tgt in val_loader:
                u_in  = u_in.to(device)
                u_tgt = u_tgt.to(device)
                pred = model(u_in)
                val_loss_sum += criterion(pred, u_tgt).item()
        val_loss = val_loss_sum / len(val_loader)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.4e}  val={val_loss:.4e}  "
            f"lr={lr_now:.2e}"
        )

        # Save best checkpoint
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
# Train all resolutions
# ---------------------------------------------------------------------------

def train_all_fnos(cfg: DictConfig, device: torch.device) -> dict[int, FNO1d]:
    """Train FNOs for all resolution levels in sequence.

    Args:
        cfg: Full config object.
        device: Compute device.

    Returns:
        Dict mapping resolution -> trained FNO1d (best weights loaded).
    """
    resolutions = [r for r in cfg.data.resolution_levels if r != 64]  # [128, 256, 512]
    fnos: dict[int, FNO1d] = {}
    for res in resolutions:
        fnos[res] = train_fno(cfg, res, device)
    return fnos


# ---------------------------------------------------------------------------
# Generate FNO forecast dataset (Strategy B)
# ---------------------------------------------------------------------------

def generate_fno_forecasts(
    cfg: DictConfig,
    fnos: dict[int, FNO1d],
    device: torch.device,
) -> None:
    """Run trained FNOs autoregressively to produce forecast datasets.

    For each resolution r and each split (train, val), runs the FNO
    one-step-at-a-time over the full trajectory and saves the outputs.

    The FNO operates in "teacher-forcing" mode: at each step it takes the
    ground-truth u_t as input and predicts u_{t+1}.  This is intentional —
    we want the distribution of single-step forecast errors (u_forecast - u_truth),
    not accumulated autoregressive errors.

    Output files:  data/fno_forecasts/fno_forecast_{resolution}_{split}.pt
    Each file: tensor of shape (n_traj, T-1, N_r), float32
       index [i, t] = FNO(u^r_{i,t})  ≈ u^r_{i,t+1}

    Args:
        cfg: Full config object.
        fnos: Dict mapping resolution -> trained FNO1d.
        device: Compute device.
    """
    out_dir = Path(cfg.data.data_dir) / "fno_forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(cfg.data.data_dir)

    for split in ("train", "val"):
        raw = torch.load(data_dir / f"{split}.pt", map_location="cpu", weights_only=True)

        for res, model in fnos.items():
            key = f"u_{res}"
            data = raw[key]          # (n_traj, T, N_r)
            n_traj, T, N_r = data.shape

            model.eval()
            # Teacher-forced: predict t+1 from ground truth t
            # Run in one big batch over traj dimension for speed
            u_in = data[:, :-1, :].to(device)    # (n_traj, T-1, N_r)
            forecasts = torch.empty_like(u_in, device="cpu")

            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk = 8
                for start in range(0, n_traj, chunk):
                    end = min(start + chunk, n_traj)
                    u_chunk = u_in[start:end]          # (C, T-1, N_r)
                    C, Tm1, Nr = u_chunk.shape
                    # Flatten traj+time -> batch, run FNO, reshape back
                    u_flat = u_chunk.reshape(C * Tm1, 1, Nr)
                    pred_flat = model(u_flat)           # (C*Tm1, 1, Nr)
                    forecasts[start:end] = pred_flat.reshape(C, Tm1, Nr).cpu()

            save_path = out_dir / f"fno_forecast_{res}_{split}.pt"
            torch.save(forecasts, save_path)
            print(
                f"  Saved {save_path}  "
                f"shape={tuple(forecasts.shape)}  "
                f"MSE vs truth={((forecasts - data[:, 1:, :].cpu())**2).mean().item():.4e}"
            )
