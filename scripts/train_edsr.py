"""
Entry point: train the EDSR super-resolution baseline.

Usage:
    python scripts/train_edsr.py [--config configs/edsr.yaml]

Trains EDSR to map 32x32 -> 256x256 vorticity with L1 loss.
Saves checkpoints to checkpoints_edsr/.

Output:
    checkpoints_edsr/edsr_best.pt    <- best validation loss
    checkpoints_edsr/edsr_final.pt   <- end of training
    checkpoints_edsr/edsr_step_*.pt  <- periodic snapshots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.training.train_edsr import train_edsr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EDSR SR baseline")
    parser.add_argument(
        "--config", type=str, default="configs/edsr.yaml",
        help="Path to EDSR config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = str(cfg.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    model = train_edsr(cfg, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nEDSR parameter count: {n_params:,}")
    print("Training complete.")


if __name__ == "__main__":
    main()
