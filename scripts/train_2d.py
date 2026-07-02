"""
Entry point: train 2D FNO forecasters then the shared diffusion corrector G
for the Kraichnan turbulence testbed.

Training order (Strategy B):
  1. Train all 2D FNOs (64, 128, 256 resolutions) to convergence.
  2. Run each trained FNO in teacher-forced mode on train/val data;
     save forecast outputs to data_2d/fno_forecasts/.
  3. Train 2D diffusion corrector G on FNO-generated forecasts (not GT).

Pass --fno_variant shared to use ONE SharedFNO2d (fixed k_max=64 across all
resolutions, see src/models/fno_2d_shared.py) instead of three independent
FNO2d models. This writes to separate paths so the original (--fno_variant
separate, the default) checkpoints/forecasts/diffusion weights are never
overwritten:
  checkpoints_2d/fno_shared.pt                         (vs fno_{64,128,256}.pt)
  data_2d/fno_forecasts_shared/                         (vs fno_forecasts/)
  checkpoints_2d/diffusion_sharedfno_ema.pt             (vs diffusion_ema.pt)

IMPORTANT: the diffusion corrector is trained on the FNO's own forecast
outputs (Strategy B) — it has learned the specific error/bias pattern of
whichever FNO produced its training data. Switching --fno_variant requires
retraining the diffusion stage too; this script does that automatically
when --stage all or --stage diffusion is used with --fno_variant shared.

Usage:
    python scripts/train_2d.py [--config configs/kraichnan.yaml]
                               [--stage fno|diffusion|all]
                               [--fno_variant separate|shared]
                               [--k_max_shared 64]
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
    parser.add_argument(
        "--fno_variant",
        type=str,
        default="separate",
        choices=["separate", "shared"],
        help="'separate' (default): three independent FNO2d, one per resolution. "
             "'shared': one SharedFNO2d with fixed k_max across all resolutions.",
    )
    parser.add_argument(
        "--k_max_shared",
        type=int,
        default=64,
        help="Fixed spectral mode truncation for --fno_variant shared (default: 64, "
             "matching the existing 256×256 FNO's cutoff)",
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

    shared = (args.fno_variant == "shared")
    forecast_dir_name = "fno_forecasts_shared" if shared else "fno_forecasts"
    ckpt_name          = "diffusion_sharedfno"  if shared else "diffusion"

    # ── Stage 1: FNO training + forecast generation ──────────────────────────
    if args.stage in ("fno", "all"):
        if shared:
            from src.training.train_fno_2d_shared import (
                train_fno_2d_shared, generate_fno_forecasts_2d_shared,
            )

            model = train_fno_2d_shared(cfg, device, k_max_shared=args.k_max_shared)

            print("\nGenerating shared-FNO forecast dataset (Strategy B)...")
            generate_fno_forecasts_2d_shared(cfg, model, device)
            print("Shared FNO forecast generation complete.")
        else:
            from src.training.train_fno_2d import train_all_fnos_2d, generate_fno_forecasts_2d

            fnos = train_all_fnos_2d(cfg, device)

            print("\nGenerating 2D FNO forecast datasets (Strategy B)...")
            generate_fno_forecasts_2d(cfg, fnos, device)
            print("FNO forecast generation complete.")

    # ── Stage 2: Diffusion training ──────────────────────────────────────────
    if args.stage in ("diffusion", "all"):
        from src.training.train_diffusion_2d import train_diffusion_2d

        train_diffusion_2d(cfg, device, forecast_dir_name=forecast_dir_name, ckpt_name=ckpt_name)
        print("Diffusion training complete.")


if __name__ == "__main__":
    main()
