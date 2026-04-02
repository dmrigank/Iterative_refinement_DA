"""
Load existing FNO checkpoints and generate forecast datasets (Strategy B).

Skips FNO training — only runs the forecast generation step.

Usage:
    conda run -n diff_da python scripts/generate_fno_forecasts.py \
        [--config configs/kraichnan.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf

from src.models.fno_2d import FNO2d
from src.training.train_fno_2d import generate_fno_forecasts_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FNO forecast datasets from saved checkpoints")
    parser.add_argument("--config", type=str, default="configs/kraichnan.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    device_str = str(cfg.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Device: {device}")

    ckpt_dir    = Path(cfg.paths.checkpoint_dir)
    resolutions = [r for r in cfg.data.resolution_levels if r != 32]  # [64, 128, 256]

    fnos: dict[int, FNO2d] = {}
    for res in resolutions:
        ckpt_path = ckpt_dir / f"fno_{res}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = FNO2d(cfg, res).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"  Loaded FNO2d  res={res}×{res}  val_loss={ckpt['val_loss']:.4e}  epoch={ckpt['epoch']}")
        fnos[res] = model

    print("\nGenerating FNO forecast datasets (Strategy B)...")
    generate_fno_forecasts_2d(cfg, fnos, device)
    print("Done.")


if __name__ == "__main__":
    main()
