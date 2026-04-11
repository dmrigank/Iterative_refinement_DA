"""
Miscellaneous 1D Burgers diagnostics saved into plots_2d_misc/.

Currently produced:
  burgers_resolution_pyramid.{png,pdf}

Usage:
    python scripts/plot_misc_burgers.py [--data data/train.pt]
                                        [--traj 0]
                                        [--snapshot_t auto]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})


PLOTS_DIR = Path("plots_2d_misc")
PLOTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Burgers resolution pyramid")
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.pt",
        help="Path to Burgers dataset split (.pt)",
    )
    parser.add_argument(
        "--traj",
        type=int,
        default=0,
        help="Trajectory index to visualize",
    )
    parser.add_argument(
        "--snapshot_t",
        type=str,
        default="auto",
        help="Snapshot index or 'auto' for the strongest shock in the chosen trajectory",
    )
    return parser.parse_args()


def _select_snapshot(data: dict[str, torch.Tensor], traj: int, snapshot_t: str) -> int:
    if snapshot_t != "auto":
        return int(snapshot_t)

    finest = data["u_512"][traj]
    grad = torch.abs(torch.roll(finest, -1, dims=-1) - torch.roll(finest, 1, dims=-1))
    return int(grad.max(dim=-1).values.argmax())


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.{{png,pdf}}")


def plot_resolution_pyramid(data: dict[str, torch.Tensor], traj: int, snapshot_t: int) -> None:
    resolutions = [64, 128, 256, 512]
    colors = {
        64: "#4C78A8",
        128: "#F58518",
        256: "#54A24B",
        512: "#E45756",
    }
    fields = {res: data[f"u_{res}"][traj, snapshot_t].cpu().numpy() for res in resolutions}

    all_values = np.concatenate([fields[res] for res in resolutions])
    pad = 0.12 * np.max(np.abs(all_values))
    y_min = float(all_values.min() - pad)
    y_max = float(all_values.max() + pad)

    fig = plt.figure(figsize=(8.5, 8.2))
    gs = gridspec.GridSpec(
        7, 1,
        height_ratios=[1.0, 0.18, 1.0, 0.18, 1.0, 0.18, 1.0],
        hspace=0.12,
    )

    x_dense = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    axis_rows = [0, 2, 4, 6]
    arrow_rows = [1, 3, 5]

    for idx, (res, row) in enumerate(zip(resolutions, axis_rows)):
        ax = fig.add_subplot(gs[row, 0])
        x = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)
        y = fields[res]

        ax.plot(x, y, color=colors[res], lw=2.2)
        ax.fill_between(x, y, 0.0, color=colors[res], alpha=0.10)
        ax.set_xlim(0.0, 2.0 * np.pi)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.18)
        ax.set_ylabel("u(x)")
        ax.text(
            0.015,
            0.86,
            f"N={res}",
            transform=ax.transAxes,
            fontsize=10,
            color=colors[res],
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=colors[res], alpha=0.95),
        )

        if idx < len(resolutions) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x")
            ax.set_xticks([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0, 2.0 * np.pi])
            ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

        if res < 512:
            ax.plot(x_dense, np.interp(x_dense, x, y, period=2.0 * np.pi),
                    color=colors[res], lw=0.8, alpha=0.20)

    for row, coarse, fine in zip(arrow_rows, resolutions[:-1], resolutions[1:]):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.axis("off")
        ax.annotate(
            "",
            xy=(0.5, 0.08),
            xytext=(0.5, 0.92),
            arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#444444"),
        )
        ax.text(
            0.5,
            0.55,
            f"{coarse} -> {fine}",
            ha="center",
            va="center",
            fontsize=10,
            color="#444444",
        )

    fig.suptitle(
        f"Burgers Resolution Pyramid: coarse -> fine  (traj={traj}, t={snapshot_t})",
        y=0.995,
    )
    fig.text(0.985, 0.5, "same snapshot across all resolutions", rotation=90,
             va="center", ha="right", fontsize=10, color="#555555")
    fig.tight_layout(rect=[0.0, 0.0, 0.97, 0.98])

    _save(fig, "burgers_resolution_pyramid")


def main() -> None:
    args = parse_args()
    data = torch.load(args.data, map_location="cpu")
    snapshot_t = _select_snapshot(data, args.traj, args.snapshot_t)
    print(f"Using traj={args.traj}, snapshot_t={snapshot_t}")
    plot_resolution_pyramid(data, args.traj, snapshot_t)


if __name__ == "__main__":
    main()
