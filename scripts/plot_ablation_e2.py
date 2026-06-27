"""
Plot ablation E2: stochasticity η vs quality and temporal consistency.

Reads results from:
  results/ablation_eta_{tag}/inference_results.pt      (1D)
  results_2d/ablation_eta_{tag}/inference_results.pt   (2D)
  where tag = f"eta_{int(eta*100):03d}"  (e.g., eta_000, eta_025, eta_100)

Produces plots_ablations/e2_eta.{png,pdf}:
  2×2 figure:
    Col 0 — 1D Burgers
    Col 1 — 2D Kraichnan
    Row 0 — RMSE @ highest resolution vs η
    Row 1 — Temporal consistency (mean frame-to-frame L2) vs η

Usage:
    python scripts/plot_ablation_e2.py
        [--results_1d  results]
        [--results_2d  results_2d]
        [--eta_values  0.0,0.25,0.5,0.75,1.0]
        [--out_dir     plots_ablations]
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

from src.evaluation.metrics import temporal_consistency
from src.evaluation.metrics_2d import temporal_consistency_2d

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.3,
})

C_1D = "#1f77b4"
C_2D = "#d62728"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eta_tag(eta: float) -> str:
    return f"eta_{int(round(eta * 100)):03d}"


def _per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """(n_traj, T, N...) -> (n_traj,)"""
    n = pred.shape[0]
    return np.array([float((pred[i] - truth[i]).pow(2).mean().sqrt()) for i in range(n)])


def _per_traj_temp_consistency_1d(pred: torch.Tensor) -> np.ndarray:
    """(n_traj, T, 512) -> (n_traj,) mean frame-to-frame L2"""
    n = pred.shape[0]
    return np.array([float(temporal_consistency(pred[i]).mean()) for i in range(n)])


def _per_traj_temp_consistency_2d(pred: torch.Tensor) -> np.ndarray:
    """(n_traj, T, 256, 256) -> (n_traj,) mean frame-to-frame L2"""
    n = pred.shape[0]
    return np.array([float(temporal_consistency_2d(pred[i]).mean()) for i in range(n)])


def _load_1d(results_dir: Path, eta_values: list[float]) -> dict:
    rmse_means, rmse_stds = [], []
    tc_means, tc_stds     = [], []
    found = []
    for eta in eta_values:
        p = results_dir / f"ablation_{_eta_tag(eta)}" / "inference_results.pt"
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        r = torch.load(p, map_location="cpu", weights_only=True)
        post  = r["posterior_512"]
        truth = r["truth_512"]
        traj_rmse = _per_traj_rmse(post, truth)
        traj_tc   = _per_traj_temp_consistency_1d(post)
        rmse_means.append(traj_rmse.mean()); rmse_stds.append(traj_rmse.std())
        tc_means.append(traj_tc.mean());     tc_stds.append(traj_tc.std())
        found.append(eta)
    return {
        "eta": found,
        "rmse_mean": np.array(rmse_means), "rmse_std": np.array(rmse_stds),
        "tc_mean":   np.array(tc_means),   "tc_std":   np.array(tc_stds),
    }


def _load_2d(results_dir: Path, eta_values: list[float]) -> dict:
    rmse_means, rmse_stds = [], []
    tc_means, tc_stds     = [], []
    found = []
    for eta in eta_values:
        p = results_dir / f"ablation_{_eta_tag(eta)}" / "inference_results.pt"
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        r = torch.load(p, map_location="cpu", weights_only=True)
        post  = r["posterior_256"]
        truth = r["truth_256"]
        traj_rmse = _per_traj_rmse(post, truth)
        traj_tc   = _per_traj_temp_consistency_2d(post)
        rmse_means.append(traj_rmse.mean()); rmse_stds.append(traj_rmse.std())
        tc_means.append(traj_tc.mean());     tc_stds.append(traj_tc.std())
        found.append(eta)
    return {
        "eta": found,
        "rmse_mean": np.array(rmse_means), "rmse_std": np.array(rmse_stds),
        "tc_mean":   np.array(tc_means),   "tc_std":   np.array(tc_stds),
    }


def _plot_col(axes_rmse: plt.Axes, axes_tc: plt.Axes, data: dict,
              color: str, rmse_label: str, tc_label: str, title_prefix: str) -> None:
    eta = np.array(data["eta"])

    axes_rmse.errorbar(
        eta, data["rmse_mean"], yerr=data["rmse_std"],
        color=color, marker="o", lw=2, ms=7, capsize=5,
    )
    axes_rmse.set_xlabel("η  (0 = deterministic,  1 = DDPM)")
    axes_rmse.set_ylabel(rmse_label)
    axes_rmse.set_title(f"{title_prefix} — RMSE vs η")
    axes_rmse.set_xticks(eta)

    axes_tc.errorbar(
        eta, data["tc_mean"], yerr=data["tc_std"],
        color=color, marker="s", lw=2, ms=7, capsize=5, ls="--",
    )
    axes_tc.set_xlabel("η")
    axes_tc.set_ylabel(tc_label)
    axes_tc.set_title(f"{title_prefix} — Temporal consistency vs η")
    axes_tc.set_xticks(eta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ablation E2: eta sweep")
    p.add_argument("--results_1d", type=str, default="results")
    p.add_argument("--results_2d", type=str, default="results_2d")
    p.add_argument("--eta_values", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--out_dir",    type=str, default="plots_ablations")
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eta_values = [float(x) for x in args.eta_values.split(",")]

    r1d_dir = Path(args.results_1d)
    r2d_dir = Path(args.results_2d)

    print("Loading 1D results ...")
    data_1d = _load_1d(r1d_dir, eta_values)

    print("Loading 2D results ...")
    data_2d = _load_2d(r2d_dir, eta_values)

    if not data_1d["eta"] and not data_2d["eta"]:
        print("No results found. Run ablation_e2_eta.py first.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if data_1d["eta"]:
        _plot_col(
            axes[0, 0], axes[1, 0], data_1d,
            color=C_1D,
            rmse_label="RMSE @ 512",
            tc_label="Frame-to-frame L2  (↓ smoother)",
            title_prefix="1D Burgers",
        )
    else:
        axes[0, 0].set_title("1D Burgers — no data")
        axes[1, 0].set_title("1D Burgers — no data")

    if data_2d["eta"]:
        _plot_col(
            axes[0, 1], axes[1, 1], data_2d,
            color=C_2D,
            rmse_label="RMSE @ 256",
            tc_label="Frame-to-frame L2  (↓ smoother)",
            title_prefix="2D Kraichnan",
        )
    else:
        axes[0, 1].set_title("2D Kraichnan — no data")
        axes[1, 1].set_title("2D Kraichnan — no data")

    fig.suptitle(
        "Stochasticity η  |  mean ± std across test trajectories\n"
        "Row 0: reconstruction quality  ·  Row 1: temporal smoothness",
        fontsize=12,
    )
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = out_dir / f"e2_eta.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir}/e2_eta.{{png,pdf}}")


if __name__ == "__main__":
    main()
