"""
Plot ablation E1: DDIM step count vs quality and speed.

Reads results from:
  results/ablation_ddim_steps_{N}/inference_results.pt     (1D)
  results_2d/ablation_ddim_steps_{N}/inference_results.pt  (2D)
  results/ablation_ddim_steps_timing.pt                    (1D wall-clock)
  results_2d/ablation_ddim_steps_timing.pt                 (2D wall-clock)

Produces plots_ablations/e1_ddim_steps.{png,pdf}:
  2-row figure:
    Row 0 — 1D Burgers: RMSE @ 512 (left Y) + wall-clock time per step (right Y, log)
    Row 1 — 2D Kraichnan: RMSE @ 256 (left Y) + wall-clock time per step (right Y, log)
  Both rows: X-axis = DDIM steps, markers + error bars from test trajectories.

Usage:
    python scripts/plot_ablation_e1.py
        [--results_1d   results]
        [--results_2d   results_2d]
        [--steps_1d     10,25,50,100]
        [--steps_2d     25,50,100,200]
        [--out_dir      plots_ablations]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3,
})

C_1D   = "#1f77b4"   # blue  — 1D Burgers
C_2D   = "#d62728"   # red   — 2D Kraichnan
C_TIME = "#7f7f7f"   # gray  — wall-clock time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """(n_traj, T, N...) -> (n_traj,) RMSE."""
    n = pred.shape[0]
    return np.array([
        float((pred[i] - truth[i]).pow(2).mean().sqrt())
        for i in range(n)
    ])


def _load_1d(results_dir: Path, steps: list[int]) -> dict:
    """Load per-step RMSE mean/std for 1D @ 512."""
    means, stds, found = [], [], []
    for s in steps:
        p = results_dir / f"ablation_ddim_steps_{s}" / "inference_results.pt"
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        r = torch.load(p, map_location="cpu", weights_only=True)
        traj_rmse = _per_traj_rmse(r["posterior_512"], r["truth_512"])
        means.append(traj_rmse.mean())
        stds.append(traj_rmse.std())
        found.append(s)
    return {"steps": found, "mean": np.array(means), "std": np.array(stds)}


def _load_2d(results_dir: Path, steps: list[int]) -> dict:
    """Load per-step RMSE mean/std for 2D @ 256."""
    means, stds, found = [], [], []
    for s in steps:
        p = results_dir / f"ablation_ddim_steps_{s}" / "inference_results.pt"
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        r = torch.load(p, map_location="cpu", weights_only=True)
        traj_rmse = _per_traj_rmse(r["posterior_256"], r["truth_256"])
        means.append(traj_rmse.mean())
        stds.append(traj_rmse.std())
        found.append(s)
    return {"steps": found, "mean": np.array(means), "std": np.array(stds)}


def _load_timing(timing_path: Path, steps: list[int]) -> np.ndarray | None:
    """Load wall-clock seconds per step count. Returns None if file missing."""
    if not timing_path.exists():
        return None
    t = torch.load(timing_path, map_location="cpu", weights_only=True)
    # keys like "steps_25" -> 45.3
    times = []
    for s in steps:
        key = f"steps_{s}"
        times.append(float(t[key]) if key in t else float("nan"))
    return np.array(times)


def _plot_panel(
    ax_left: plt.Axes,
    data: dict,
    timing: np.ndarray | None,
    color: str,
    ylabel_left: str,
    title: str,
) -> None:
    steps = np.array(data["steps"])
    means = data["mean"]
    stds  = data["std"]

    ax_left.errorbar(
        steps, means, yerr=stds,
        color=color, marker="o", lw=2, ms=7, capsize=5,
        label="RMSE (mean ± std)",
    )
    ax_left.set_xlabel("DDIM steps")
    ax_left.set_ylabel(ylabel_left, color=color)
    ax_left.tick_params(axis="y", labelcolor=color)
    ax_left.set_title(title)
    ax_left.set_xticks(steps)

    if timing is not None and not np.all(np.isnan(timing)):
        ax_right = ax_left.twinx()
        valid = ~np.isnan(timing)
        ax_right.plot(
            steps[valid], timing[valid],
            color=C_TIME, marker="s", lw=1.5, ls="--", ms=6,
            label="Wall-clock time (s)",
        )
        ax_right.set_ylabel("Wall-clock time  (s, log scale)", color=C_TIME)
        ax_right.set_yscale("log")
        ax_right.tick_params(axis="y", labelcolor=C_TIME)

        # Combined legend
        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_left.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    else:
        ax_left.legend(loc="upper right", fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ablation E1: DDIM steps")
    p.add_argument("--results_1d", type=str, default="results")
    p.add_argument("--results_2d", type=str, default="results_2d")
    p.add_argument("--steps_1d",   type=str, default="10,25,50,100")
    p.add_argument("--steps_2d",   type=str, default="25,50,100,200")
    p.add_argument("--out_dir",    type=str, default="plots_ablations")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_1d = [int(x) for x in args.steps_1d.split(",")]
    steps_2d = [int(x) for x in args.steps_2d.split(",")]

    r1d_dir = Path(args.results_1d)
    r2d_dir = Path(args.results_2d)

    print("Loading 1D results ...")
    data_1d   = _load_1d(r1d_dir, steps_1d)
    timing_1d = _load_timing(r1d_dir / "ablation_ddim_steps_timing.pt", steps_1d)

    print("Loading 2D results ...")
    data_2d   = _load_2d(r2d_dir, steps_2d)
    timing_2d = _load_timing(r2d_dir / "ablation_ddim_steps_timing.pt", steps_2d)

    if not data_1d["steps"] and not data_2d["steps"]:
        print("No results found. Run ablation_e1_ddim_steps.py first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    if data_1d["steps"]:
        _plot_panel(
            axes[0], data_1d, timing_1d,
            color=C_1D,
            ylabel_left="RMSE @ 512",
            title="1D Burgers — DDIM steps vs quality / speed",
        )
    else:
        axes[0].set_title("1D Burgers — no data")

    if data_2d["steps"]:
        _plot_panel(
            axes[1], data_2d, timing_2d,
            color=C_2D,
            ylabel_left="RMSE @ 256",
            title="2D Kraichnan — DDIM steps vs quality / speed",
        )
    else:
        axes[1].set_title("2D Kraichnan — no data")

    fig.suptitle(
        "DDIM step count  |  mean ± std across test trajectories",
        fontsize=13,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = out_dir / f"e1_ddim_steps.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/e1_ddim_steps.{{png,pdf}}")


if __name__ == "__main__":
    main()
