"""
Entry point: train the 1D EDSR super-resolution baseline.

Usage:
    python scripts/train_edsr_1d.py [--config configs/edsr_1d.yaml]

Trains EDSR1d to map 64-pt -> 512-pt Burgers fields with L1 loss.
Saves checkpoints to checkpoints_edsr_1d/.

Output:
    checkpoints_edsr_1d/edsr_1d_best.pt    <- best validation loss
    checkpoints_edsr_1d/edsr_1d_final.pt   <- end of training
    checkpoints_edsr_1d/edsr_1d_step_*.pt  <- periodic snapshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.training.train_edsr_1d import train_edsr_1d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 1D EDSR SR baseline")
    p.add_argument("--config", type=str, default="configs/edsr_1d.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if str(cfg.device) != "auto":
        device_str = str(cfg.device)
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    model = train_edsr_1d(cfg, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nEDSR-1D parameter count: {n_params:,}")
    print("Training complete.")


if __name__ == "__main__":
    main()
