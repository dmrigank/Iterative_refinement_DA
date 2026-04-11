"""
Comparison figures: iterative refinement vs one-shot diffusion SR baseline.

Figures produced in plots_oneshot/:
  fig1_rmse_comparison.{png,pdf}     — RMSE vs time, iterative vs one-shot vs FNO-only
  fig2_spectrum_comparison.{png,pdf} — Energy spectrum: GT / iterative / one-shot / coarse obs
  fig3_snapshot_comparison.{png,pdf} — Side-by-side snapshots at t=50:
                                        [GT | FNO-only | Iterative | One-shot]
  fig4_per_method_rmse.{png,pdf}     — Bar chart: mean RMSE ± std across test trajectories

Usage:
    python scripts/plot_results_oneshot.py
        [--iterative results_2d/inference_results.pt]
        [--oneshot   results_oneshot/inference_results.pt]
        [--config    configs/kraichnan.yaml]
        [--figures   1,2,3,4]
        [--snapshot_t 50]
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
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf

from src.evaluation.metrics_2d import rmse_over_time_2d, radial_energy_spectrum


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  14,
    "legend.fontsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "axes.grid":       False,
})

PLOTS_DIR = Path("plots_oneshot")
PLOTS_DIR.mkdir(exist_ok=True)

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot"

COLOR_ITER  = "steelblue"
COLOR_ONE   = "darkorange"
COLOR_FNO   = "tomato"
COLOR_TRUTH = "black"


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 1 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_comparison(ri: dict, ro: dict) -> None:
    """RMSE vs time step at 256×256 for all three methods."""

    n_traj    = ri["truth_256"].shape[0]
    truth_all = ri["truth_256"]          # (n_traj, T, 256, 256)
    iter_all  = ri["posterior_256"]
    fno_all   = ri["fno_only_256"]
    one_all   = ro["posterior_256"]

    def _rmse_curves(pred, truth):
        return torch.stack([
            rmse_over_time_2d(pred[i], truth[i]) for i in range(n_traj)
        ])  # (n_traj, T)

    T = truth_all.shape[1]
    T_one = one_all.shape[1]
    t_ax  = np.arange(T)
    t_one = np.arange(T_one)

    rmse_iter = _rmse_curves(iter_all, truth_all)
    rmse_fno  = _rmse_curves(fno_all,  truth_all)
    rmse_one  = _rmse_curves(one_all,  ro["truth_256"])

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n_traj):
        ax.plot(t_ax,  rmse_fno[i].numpy(),  color=COLOR_FNO,  alpha=0.2, lw=0.9)
        ax.plot(t_ax,  rmse_iter[i].numpy(), color=COLOR_ITER, alpha=0.2, lw=0.9)
        ax.plot(t_one, rmse_one[i].numpy(),  color=COLOR_ONE,  alpha=0.2, lw=0.9)

    ax.plot(t_ax,  rmse_fno.mean(0).numpy(),
            color=COLOR_FNO,  lw=2.2, label="FNO-only (autoregressive)")
    ax.plot(t_ax,  rmse_iter.mean(0).numpy(),
            color=COLOR_ITER, lw=2.2, label="Iterative refinement (ours)")
    ax.plot(t_one, rmse_one.mean(0).numpy(),
            color=COLOR_ONE,  lw=2.2, label="One-shot diffusion SR")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Over Time  (256×256)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig1_rmse_comparison")


# ---------------------------------------------------------------------------
# Figure 2 — Energy spectrum comparison
# ---------------------------------------------------------------------------

def plot_spectrum_comparison(ri: dict, ro: dict, k_forcing: float = 4.0) -> None:
    """Log-log E(k): GT / coarse obs / iterative / one-shot."""

    truth_all = ri["truth_256"].reshape(-1, 256, 256)
    iter_all  = ri["posterior_256"].reshape(-1, 256, 256)
    one_all   = ro["posterior_256"].reshape(-1, 256, 256)
    obs_all   = ri["obs_32"].reshape(-1, 32, 32)

    E_truth, k_bins     = radial_energy_spectrum(truth_all)
    E_iter,  _          = radial_energy_spectrum(iter_all)
    E_one,   _          = radial_energy_spectrum(one_all)
    E_obs,   k_bins_obs = radial_energy_spectrum(obs_all)

    k    = k_bins[1:].numpy()
    k_ob = k_bins_obs[1:].numpy()
    Et   = E_truth[1:].numpy()
    Ei   = E_iter[1:].numpy()
    Eo   = E_one[1:].numpy()
    Eobs = E_obs[1:].numpy()

    # k^-3 reference
    idx5  = np.searchsorted(k, 5)
    k_ref = np.array([3.0, k.max()])
    E_ref = Et[idx5] * (5.0 ** 3) * k_ref ** (-3)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k,    Et,   color=COLOR_TRUTH, lw=2.0, label="Ground truth (256×256)")
    ax.loglog(k_ob, Eobs, color="gray",     lw=1.5, ls="--", label="Coarse obs (32×32)")
    ax.loglog(k,    Ei,   color=COLOR_ITER, lw=1.8, label="Iterative refinement")
    ax.loglog(k,    Eo,   color=COLOR_ONE,  lw=1.8, ls="--", label="One-shot diffusion SR")
    ax.loglog(k_ref, E_ref, color="gray",   ls=":",  lw=1.0, label=r"$k^{-3}$")

    k_nyq_32 = 16
    ax.axvline(k_nyq_32, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.text(k_nyq_32 * 1.05, Et.max() * 0.5, "32×32\nNyquist", color="gray", fontsize=9)
    ax.axvline(k_forcing, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.text(k_forcing * 1.1, Et.max() * 0.15, rf"$k_f={k_forcing:.0f}$",
            color="gray", fontsize=10)

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("Radial Energy Spectrum  (all trajectories & time steps)")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=1)
    fig.tight_layout()
    _save(fig, "fig2_spectrum_comparison")


# ---------------------------------------------------------------------------
# Figure 3 — Snapshot comparison (4 columns)
# ---------------------------------------------------------------------------

def plot_snapshot_comparison(ri: dict, ro: dict, t: int = 50, traj: int = 0) -> None:
    """1×4 row: GT | FNO-only | Iterative posterior | One-shot posterior."""

    gt_field   = ri["truth_256"][traj, t].numpy()
    fno_field  = ri["fno_only_256"][traj, t].numpy()
    iter_field = ri["posterior_256"][traj, t].numpy()
    one_field  = ro["posterior_256"][traj, t].numpy()

    fields = [gt_field, fno_field, iter_field, one_field]
    titles = ["Ground Truth", "FNO-only (autoreg.)",
              "Iterative Refinement", "One-Shot Diffusion SR"]

    vmax = float(np.percentile(np.abs(gt_field), 99.5))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    ims = []
    for ax, field, title in zip(axes, fields, titles):
        im = ax.imshow(
            field, cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
            origin="lower", aspect="equal", interpolation="nearest",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)

    fig.subplots_adjust(left=0.03, right=0.91, bottom=0.05, top=0.88, wspace=0.06)
    cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.75])
    fig.colorbar(ims[-1], cax=cbar_ax)
    cbar_ax.tick_params(labelsize=9)
    cbar_ax.set_ylabel("Vorticity  ω", fontsize=10)

    fig.suptitle(f"256×256 Vorticity Comparison  |  t={t},  trajectory {traj}",
                 fontsize=13, y=1.00)
    _save(fig, "fig3_snapshot_comparison")


# ---------------------------------------------------------------------------
# Figure 4 — Bar chart: mean RMSE per method
# ---------------------------------------------------------------------------

def plot_per_method_rmse(ri: dict, ro: dict) -> None:
    """Grouped bar chart: FNO-only / Iterative / One-shot, at 256×256."""

    n_traj    = ri["truth_256"].shape[0]
    truth_all = ri["truth_256"]

    def _per_traj_mean_rmse(pred, truth):
        n = pred.shape[0]
        vals = [rmse_over_time_2d(pred[i], truth[i]).mean().item() for i in range(n)]
        return torch.tensor(vals)

    pt_iter = _per_traj_mean_rmse(ri["posterior_256"], truth_all)
    pt_fno  = _per_traj_mean_rmse(ri["fno_only_256"],  truth_all)
    pt_one  = _per_traj_mean_rmse(ro["posterior_256"], ro["truth_256"])

    means = [pt_fno.mean().item(), pt_iter.mean().item(), pt_one.mean().item()]
    stds  = [
        pt_fno.std().item()  if n_traj > 1 else 0.0,
        pt_iter.std().item() if n_traj > 1 else 0.0,
        pt_one.std().item()  if ro["posterior_256"].shape[0] > 1 else 0.0,
    ]
    labels = ["FNO-only\n(autoreg.)", "Iterative\nRefinement", "One-Shot\nDiffusion SR"]
    colors = [COLOR_FNO, COLOR_ITER, COLOR_ONE]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.5)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean RMSE  (256×256)")
    ax.set_title("Method Comparison  |  Mean ± Std across test trajectories")
    fig.tight_layout()
    _save(fig, "fig4_per_method_rmse")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate one-shot vs iterative comparison figures")
    p.add_argument("--iterative",   type=str, default="results_2d/inference_results.pt")
    p.add_argument("--oneshot",     type=str, default="results_oneshot/inference_results.pt")
    p.add_argument("--config",      type=str, default="configs/kraichnan.yaml")
    p.add_argument("--figures",     type=str, default="1,2,3,4",
                   help="Comma-separated figures to generate (default: all)")
    p.add_argument("--snapshot_t",  type=int, default=50,
                   help="Time step for Fig 3 snapshot (default: 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    figs = {int(x) for x in args.figures.split(",")}

    print(f"Loading iterative results from {args.iterative} ...")
    ri = torch.load(args.iterative, map_location="cpu", weights_only=True)
    print(f"  Trajectories: {ri['truth_256'].shape[0]},  "
          f"time steps: {ri['truth_256'].shape[1]}")

    print(f"Loading one-shot results from {args.oneshot} ...")
    ro = torch.load(args.oneshot, map_location="cpu", weights_only=True)
    print(f"  Trajectories: {ro['posterior_256'].shape[0]},  "
          f"time steps: {ro['posterior_256'].shape[1]}")

    if 1 in figs:
        print("\nFigure 1: RMSE over time comparison ...")
        plot_rmse_comparison(ri, ro)

    if 2 in figs:
        print("\nFigure 2: Energy spectrum comparison ...")
        plot_spectrum_comparison(ri, ro,
                                 k_forcing=float(cfg.pde.forcing_band_center))

    if 3 in figs:
        print("\nFigure 3: Snapshot comparison ...")
        plot_snapshot_comparison(ri, ro, t=args.snapshot_t)

    if 4 in figs:
        print("\nFigure 4: Per-method RMSE bar chart ...")
        plot_per_method_rmse(ri, ro)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
