"""
Entry point: train the 1D one-shot diffusion SR baseline.

Usage:
    python scripts/train_oneshot_1d.py [--config configs/oneshot_sr_1d.yaml]
                                       [--device cuda|cpu]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.unet_oneshot_1d import OneShotUNet1d
from src.training.train_oneshot_1d import train_oneshot_1d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 1D one-shot diffusion SR baseline")
    p.add_argument("--config", type=str, default="configs/oneshot_sr_1d.yaml")
    p.add_argument("--device", type=str, default=None,
                   help="Device override (default: use cfg.device / auto-detect)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Device selection
    if args.device is not None:
        device_str = args.device
    else:
        device_str = str(cfg.device)
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    # Print parameter count before training
    unet = OneShotUNet1d(
        base_channels  = int(cfg.unet.base_channels),
        channel_mults  = list(cfg.unet.channel_mults),
        n_res_blocks   = int(cfg.unet.n_res_blocks),
        n_groups       = int(cfg.unet.group_norm_groups),
        cond_embed_dim = int(cfg.unet.cond_embed_dim),
    )
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"OneShotUNet1d parameters: {n_params:,}")
    del unet

    train_oneshot_1d(cfg, device)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
