"""
Entry point: train FNO forecasters then the shared diffusion corrector G.

Training order (Strategy B):
  1. Train all FNOs (128, 256, 512 resolutions) to convergence.
  2. Run each trained FNO on training/val data; save forecast outputs.
  3. Train diffusion corrector G on FNO-generated forecasts (not GT).

Usage:
    python scripts/train.py [--config configs/default.yaml] [--stage fno|diffusion|all]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FNO and/or diffusion models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["fno", "diffusion", "all"],
        help="Which training stage to run",
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

    # ── Stage 1: FNO training + forecast generation ──────────────────────────
    if args.stage in ("fno", "all"):
        from src.training.train_fno import train_all_fnos, generate_fno_forecasts

        fnos = train_all_fnos(cfg, device)

        print("\nGenerating FNO forecast datasets (Strategy B)...")
        generate_fno_forecasts(cfg, fnos, device)
        print("FNO forecast generation complete.")

    # ── Stage 2: Diffusion training ──────────────────────────────────────────
    if args.stage in ("diffusion", "all"):
        from src.training.train_diffusion import train_diffusion

        train_diffusion(cfg, device)


if __name__ == "__main__":
    main()
