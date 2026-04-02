"""
Entry point: train 2D FNO forecasters then the shared diffusion corrector G
for the Kraichnan turbulence testbed.

Training order (Strategy B):
  1. Train all 2D FNOs (64, 128, 256 resolutions) to convergence.
  2. Run each trained FNO in teacher-forced mode on train/val data;
     save forecast outputs to data_2d/fno_forecasts/.
  3. Train 2D diffusion corrector G on FNO-generated forecasts (not GT).

Usage:
    python scripts/train_2d.py [--config configs/kraichnan.yaml]
                               [--stage fno|diffusion|all]
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
    parser = argparse.ArgumentParser(description="Train 2D FNO and/or diffusion models")
    parser.add_argument("--config", type=str, default="configs/kraichnan.yaml")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["fno", "diffusion", "all"],
        help="Which training stage to run (default: all)",
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
        from src.training.train_fno_2d import train_all_fnos_2d, generate_fno_forecasts_2d

        fnos = train_all_fnos_2d(cfg, device)

        print("\nGenerating 2D FNO forecast datasets (Strategy B)...")
        generate_fno_forecasts_2d(cfg, fnos, device)
        print("FNO forecast generation complete.")

    # ── Stage 2: Diffusion training ──────────────────────────────────────────
    if args.stage in ("diffusion", "all"):
        from src.training.train_diffusion_2d import train_diffusion_2d

        train_diffusion_2d(cfg, device)
        print("Diffusion training complete.")


if __name__ == "__main__":
    main()
