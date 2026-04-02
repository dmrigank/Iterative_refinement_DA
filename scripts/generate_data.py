"""
Entry point: generate training/validation/test datasets.

Usage:
    python scripts/generate_data.py [--config configs/default.yaml]

Outputs (in data/):
    train.pt, val.pt, test.pt
    Each: dict {"u_64": (n_traj, T, 64), "u_128": ..., "u_256": ..., "u_512": ...}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.generate_dataset import generate_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Burgers equation dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to OmegaConf YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = str(cfg.get("device", "cpu"))
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    generate_and_save(cfg, device)


if __name__ == "__main__":
    main()
