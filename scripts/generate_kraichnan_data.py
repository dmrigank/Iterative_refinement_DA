"""
Entry point: generate 2D Kraichnan turbulence dataset.

Runs the pseudospectral solver for each trajectory, builds the 4-level
resolution pyramid [32, 64, 128, 256], and saves train/val/test .pt files
to data_2d/.

NOTE: The solver runs on CPU (NumPy). For 20 trajectories at 256×256 with
burn_in=4000 this takes roughly 4–8 hours depending on hardware.
Start this first — it can run overnight while you implement later stages.

Usage
─────
    # Full run (20 trajectories — several hours)
    python scripts/generate_kraichnan_data.py

    # Quick sanity-check run (1 trajectory — ~10 min)
    python scripts/generate_kraichnan_data.py --n_traj 1 --sanity_check

    # Custom config
    python scripts/generate_kraichnan_data.py --config configs/kraichnan.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.data.generate_kraichnan import generate_and_save_kraichnan


# ---------------------------------------------------------------------------
# Sanity-check plot
# ---------------------------------------------------------------------------

def run_sanity_check(data_dir: Path, plots_dir: Path) -> None:
    """Load the generated data and plot one snapshot at all 4 resolutions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load train split (or whichever was generated)
    pt_path = data_dir / "train.pt"
    if not pt_path.exists():
        pt_path = data_dir / "test.pt"
    if not pt_path.exists():
        print("  No .pt file found for sanity check — skipping plot")
        return

    data = torch.load(pt_path, map_location="cpu", weights_only=True)

    resolutions = [32, 64, 128, 256]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    # Pick trajectory 0, time step 50 (middle of trajectory)
    t_idx = min(50, data[f"w_256"].shape[1] - 1)

    for ax, res in zip(axes, resolutions):
        w = data[f"w_{res}"][0, t_idx].numpy()   # (res, res)
        vmax = float(np.percentile(np.abs(w), 99.5))
        vmax = max(vmax, 1e-6)

        im = ax.imshow(
            w, origin="lower",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            extent=[0, 2 * np.pi, 0, 2 * np.pi],
        )
        ax.set_title(f"{res}×{res}  (rms={np.sqrt(np.mean(w**2)):.2f})", fontsize=13)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Vorticity ω at t-index {t_idx} — resolution pyramid", fontsize=14)
    fig.tight_layout()

    save_path = plots_dir / "data_sanity_check.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Sanity check plot saved to {save_path}")

    # Print value statistics
    print("\n  Vorticity statistics (traj 0, all time steps):")
    print(f"  {'Resolution':>12}  {'mean':>10}  {'rms':>10}  {'max|ω|':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for res in resolutions:
        w_all = data[f"w_{res}"][0].numpy()   # (T, res, res)
        print(
            f"  {res:>10d}×{res:<3d}"
            f"  {w_all.mean():>10.4f}"
            f"  {np.sqrt(np.mean(w_all**2)):>10.4f}"
            f"  {np.max(np.abs(w_all)):>10.4f}"
        )


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2D Kraichnan turbulence dataset")
    parser.add_argument("--config",       type=str, default="configs/kraichnan.yaml")
    parser.add_argument("--n_traj",       type=int, default=None,
                        help="Override number of trajectories (default: cfg.data.n_trajectories)")
    parser.add_argument("--sanity_check", action="store_true",
                        help="After generation, run sanity checks and save a diagnostic plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    n_traj = args.n_traj  # None means use cfg value

    if n_traj is not None:
        print(f"Overriding n_trajectories: {cfg.data.n_trajectories} -> {n_traj}")

    generate_and_save_kraichnan(cfg, n_trajectories_override=n_traj)

    if args.sanity_check:
        print("\nRunning sanity checks...")
        run_sanity_check(
            data_dir=Path(cfg.data.data_dir),
            plots_dir=Path(cfg.paths.plots_dir),
        )


if __name__ == "__main__":
    main()
