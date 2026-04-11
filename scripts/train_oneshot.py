"""
Entry point: train one-shot diffusion SR baseline.

Usage:
    conda activate diff_da
    python scripts/train_oneshot.py [--config configs/oneshot_sr.yaml]

The one-shot model maps 32×32 → 256×256 directly in a single diffusion pass,
conditioned on the previous 256×256 state and the spectrally upsampled coarse obs.

Checkpoints saved to checkpoints_oneshot/.
Final EMA weights at checkpoints_oneshot/oneshot_ema.pt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.training.train_oneshot import train_oneshot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one-shot diffusion SR baseline")
    p.add_argument("--config", type=str, default="configs/oneshot_sr.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = str(cfg.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device:  {device}")
    print(f"Config:  {args.config}")
    print(f"Steps:   {cfg.diffusion_training.steps:,}")
    print(f"Batch:   {cfg.diffusion_training.batch_size}")

    train_oneshot(cfg, device)


if __name__ == "__main__":
    main()
