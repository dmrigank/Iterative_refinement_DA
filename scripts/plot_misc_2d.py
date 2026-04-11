"""
Miscellaneous 2D diagnostic figures for the Kraichnan turbulence testbed.

Figures produced in plots_2d_misc/:
  fig1_coarse_vs_fine.{png,pdf}     — 32×32 obs vs 256×256 ground truth
  fig2_spectrum_annotated.{png,pdf} — energy spectrum with Nyquist / cascade shading
  fig3_resolution_pyramid.{png,pdf} — vorticity thumbnails at 32/64/128/256 with arrows
  fig4_evolution.gif                — animation: GT vs FNO-only vs |error|
  fig5_sequential_inverse_problem.{png,pdf}
                                    — fine trajectory above coarse sequence with inference arrow

Usage:
    python scripts/plot_misc_2d.py [--config configs/kraichnan.yaml]
                                   [--results results_2d/inference_results.pt]
                                   [--traj 0]
                                   [--figures 1,2,3,4]
                                   [--snapshot_t 50]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import OmegaConf

from src.evaluation.metrics_2d import radial_energy_spectrum


# ---------------------------------------------------------------------------
# Global style & output dir
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "legend.fontsize": 10,
    "figure.dpi":      200,
    "savefig.dpi":     200,
    "axes.grid":       False,
})

PLOTS_DIR = Path("plots_2d_misc")
PLOTS_DIR.mkdir(exist_ok=True)

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot"


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 1 — 32×32 coarse obs vs 256×256 ground truth
# ---------------------------------------------------------------------------

def plot_coarse_vs_fine(r: dict, t: int = 50, traj: int = 0) -> None:
    """Side-by-side: pixelated 32×32 coarse obs and full 256×256 truth."""

    obs   = r["obs_32"  ][traj, t].numpy()    # (32, 32)
    truth = r["truth_256"][traj, t].numpy()   # (256, 256)

    # Shared symmetric colorscale from the truth field
    vmax = float(np.percentile(np.abs(truth), 99.5))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    im0 = axes[0].imshow(
        obs, cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
        origin="lower", aspect="equal", interpolation="nearest",
    )
    axes[0].set_title("32×32 coarse observation", fontsize=11)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im1 = axes[1].imshow(
        truth, cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
        origin="lower", aspect="equal", interpolation="bilinear",
    )
    axes[1].set_title("256×256 ground truth", fontsize=11)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Shared colorbar at the bottom
    fig.subplots_adjust(bottom=0.18, wspace=0.06)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.70, 0.04])
    fig.colorbar(im1, cax=cbar_ax, orientation="horizontal")
    cbar_ax.tick_params(labelsize=9)
    cbar_ax.set_xlabel("Vorticity  ω", fontsize=10, labelpad=2)

    _save(fig, "fig1_coarse_vs_fine")


# ---------------------------------------------------------------------------
# Figure 2 — Annotated energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum_annotated(r: dict) -> None:
    """Log-log E(k) with Nyquist lines, cascade shading, k^-3 reference."""

    truth_all = r["truth_256"].reshape(-1, 256, 256)
    E_truth, k_bins = radial_energy_spectrum(truth_all)

    k  = k_bins[1:].numpy()
    Et = E_truth[1:].numpy()

    # k^-3 reference anchored to truth at k=5
    k_ref = np.array([3.0, k.max()])
    idx5  = np.searchsorted(k, 5)
    E_ref = Et[idx5] * (5.0 ** 3) * k_ref ** (-3)

    fig, ax = plt.subplots(figsize=(3.5, 3))

    k_nyq_32  = 16    # 32/2
    k_nyq_256 = 128   # 256/2

    # Draw spectrum first so shading sits on top but curves are always visible
    ax.loglog(k, Et, "k-", lw=2.0, zorder=4, label="Ground truth  $E(k)$")
    ax.loglog(k_ref, E_ref, color="gray", ls="--", lw=1.2, zorder=4, label=r"$k^{-3}$")

    # Shading: green first (reconstructed band), then red (fully unobserved) on top
    # so the green strip is visible as a lighter region inside the red zone
    ax.axvspan(k_nyq_32, k_nyq_256, alpha=0.15, color="red",  zorder=1,
               label="Unobserved modes  (k > 16)")
    ax.axvspan(k_nyq_32, 30,        alpha=0.25, color="green", zorder=2,
               label="Reconstructed by cascade  (16 < k < 30)")

    # Nyquist lines
    ax.axvline(k_nyq_32,  color="darkred",  ls="--", lw=1.2, alpha=0.9, zorder=3)
    ax.axvline(k_nyq_256, color="navy",     ls="--", lw=1.2, alpha=0.7, zorder=3)

    # Nyquist labels inside the plot, positioned along the y-axis
    ylo, yhi = ax.get_ylim()
    # Place at a fixed fraction of the log y-range
    y_label = 10 ** (0.15 * math.log10(max(Et)) + 0.85 * math.log10(max(Et) * 1e-16))
    ax.text(k_nyq_32  * 1.05, Et.max() * 2e-1, "32×32 Nyquist",
            color="darkred", fontsize=7, va="top", rotation=90,
            rotation_mode="anchor", zorder=5)
    ax.text(k_nyq_256 * 1.05, Et.max() * 2e-1, "256×256 Nyquist",
            color="navy",    fontsize=7, va="top", rotation=90,
            rotation_mode="anchor", zorder=5)

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_xlim(left=1)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig2_spectrum_annotated")


# ---------------------------------------------------------------------------
# Figure 3 — Resolution pyramid with FNO/diffusion arrows
# ---------------------------------------------------------------------------

def plot_resolution_pyramid(r: dict, t: int = 50, traj: int = 0) -> None:
    """Horizontal resolution pyramid from coarse to fine."""

    resolutions = [32, 64, 128, 256]
    fields = {
        32:  r["obs_32"   ][traj, t].numpy(),
        64:  r["posterior_64" ][traj, t].numpy(),
        128: r["posterior_128"][traj, t].numpy(),
        256: r["posterior_256"][traj, t].numpy(),
    }

    vmax = float(np.percentile(np.abs(fields[256]), 99.5))
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(9.5, 3.2))
    gs = gridspec.GridSpec(
        1, 7, figure=fig,
        width_ratios=[0.9, 0.18, 1.15, 0.22, 1.45, 0.26, 1.8],
        wspace=0.0,
        left=0.04, right=0.92, top=0.82, bottom=0.18,
    )

    img_cols = [0, 2, 4, 6]
    gap_cols = [1, 3, 5]

    for res, gs_col in zip(resolutions, img_cols):
        ax = fig.add_subplot(gs[0, gs_col])
        ax.imshow(
            fields[res], cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
            origin="lower", aspect="equal", interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.04, 0.04, f"{res}×{res}", ha="left", va="bottom",
                transform=ax.transAxes, fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))

    for gap_col in gap_cols:
        ax = fig.add_subplot(gs[0, gap_col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.annotate(
            "", xy=(0.92, 0.5), xytext=(0.08, 0.5),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        )

    fig.suptitle("Resolution pyramid", fontsize=13, y=0.93)

    _save(fig, "fig3_resolution_pyramid")


# ---------------------------------------------------------------------------
# Figure 4 — Animation: GT vs FNO-only vs |error|
# ---------------------------------------------------------------------------

def plot_evolution_gif(r: dict, traj: int = 0, fps: int = 5) -> None:
    """GIF of GT | FNO-only | |GT - FNO-only| over all time steps at 256×256."""

    truth_all = r["truth_256"  ][traj]   # (T, 256, 256)
    fno_all   = r["fno_only_256"][traj]  # (T, 256, 256)
    T = truth_all.shape[0]

    # Colorscales fixed across all frames for consistent animation
    vmax = float(np.percentile(np.abs(truth_all.numpy()), 99.5))
    vmax = max(vmax, 1e-6)

    err_all = (fno_all - truth_all).abs()
    emax = float(np.percentile(err_all.numpy(), 99.5))
    emax = max(emax, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.12, top=0.88, wspace=0.06)

    titles = ["Ground Truth", "FNO-only (autoreg.)", "|Error|"]
    cmaps  = [CMAP_FIELD, CMAP_FIELD, CMAP_ERR]
    vmins  = [-vmax, -vmax, 0.0]
    vmaxs  = [ vmax,  vmax, emax]

    ims = []
    for ax, title, cmap, vmin, vmx in zip(axes, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(
            np.zeros((256, 256)), cmap=cmap, vmin=vmin, vmax=vmx,
            origin="lower", aspect="equal", interpolation="nearest",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")
        ims.append(im)

    title_text = fig.suptitle("t=0", fontsize=12, y=0.97)

    def update(frame: int):
        ims[0].set_data(truth_all[frame].numpy())
        ims[1].set_data(fno_all[frame].numpy())
        ims[2].set_data((fno_all[frame] - truth_all[frame]).abs().numpy())
        title_text.set_text(f"t = {frame}")
        return ims + [title_text]

    ani = FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=True)

    gif_path = PLOTS_DIR / "fig4_evolution.gif"
    ani.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  Saved fig4_evolution.gif  ({T} frames, {fps} fps)")


# ---------------------------------------------------------------------------
# Figure 5 — Sequential inverse problem schematic
# ---------------------------------------------------------------------------

def plot_sequential_inverse_problem(
    r: dict,
    traj: int = 0,
    n_show: int = 5,
) -> None:
    """Visual definition of the sequential inverse problem in 2D.

    Top row:    fine 256x256 trajectory snapshots.
    Bottom row: corresponding coarse 32x32 observations.
    Middle:     annotation indicating the goal of reconstructing the fine
                trajectory from the full coarse sequence.
    """

    T = r["truth_256"].shape[1]
    n_show = max(3, min(n_show, T))
    t_indices = np.linspace(0, T - 1, n_show, dtype=int)

    truth_seq = r["truth_256"][traj, t_indices].numpy()
    obs_seq   = r["obs_32"][traj, t_indices].numpy()

    vmax = float(np.percentile(np.abs(truth_seq), 99.5))
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(2.35 * n_show + 1.8, 6.1))
    gs = gridspec.GridSpec(
        2, n_show, figure=fig,
        hspace=0.22, wspace=0.08,
        left=0.10, right=0.90, top=0.86, bottom=0.16,
    )

    axes_top: list[plt.Axes] = []
    axes_bot: list[plt.Axes] = []

    for col, t in enumerate(t_indices):
        ax_top = fig.add_subplot(gs[0, col])
        ax_bot = fig.add_subplot(gs[1, col])
        axes_top.append(ax_top)
        axes_bot.append(ax_bot)

        im_top = ax_top.imshow(
            truth_seq[col],
            cmap=CMAP_FIELD,
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            aspect="equal",
            interpolation="bilinear",
        )
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(f"$t={t}$", fontsize=12, pad=5)

        ax_bot.imshow(
            obs_seq[col],
            cmap=CMAP_FIELD,
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
        )
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])

    # Row labels
    fig.text(
        0.035, 0.67,
        "Fine trajectory\n$\\{\\omega_t\\}_{t=0}^{T}$\n(256$\\times$256)",
        va="center", ha="center", rotation=90, fontsize=14,
    )
    fig.text(
        0.035, 0.29,
        "Coarse observations\n$\\{y_t\\}_{t=0}^{T}$\n(32$\\times$32)",
        va="center", ha="center", rotation=90, fontsize=14, color="dimgray",
    )

    # Time-direction arrows across the two rows
    for ax_row in (axes_top, axes_bot):
        for ax_l, ax_r in zip(ax_row[:-1], ax_row[1:]):
            bb_l = ax_l.get_position()
            bb_r = ax_r.get_position()
            y = 0.5 * (bb_l.y0 + bb_l.y1)
            arrow = FancyArrowPatch(
                (bb_l.x1 + 0.005, y),
                (bb_r.x0 - 0.005, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.2,
                color="black" if ax_row is axes_top else "dimgray",
                alpha=0.9,
            )
            fig.add_artist(arrow)

    # Inference arrow from coarse sequence to fine trajectory on the right
    bb_top = axes_top[-1].get_position()
    bb_bot = axes_bot[-1].get_position()
    x_arrow = bb_top.x1 + 0.035
    infer_arrow = FancyArrowPatch(
        (x_arrow, bb_bot.y1 - 0.01),
        (x_arrow, bb_top.y0 + 0.01),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        lw=2.0,
        color="#2f2f2f",
    )
    fig.add_artist(infer_arrow)
    fig.text(
        x_arrow + 0.025,
        0.5 * (bb_top.y0 + bb_bot.y1),
        "Sequential inverse problem:\nrecover the fine trajectory\nfrom the coarse sequence",
        va="center",
        ha="left",
        fontsize=12,
        color="#2f2f2f",
    )

    # Shared colorbar
    cax = fig.add_axes([0.18, 0.05, 0.52, 0.03])
    cbar = fig.colorbar(im_top, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Vorticity  $\\omega$", fontsize=11, labelpad=3)

    fig.suptitle(
        "Sequential Inverse Problem",
        fontsize=15,
        y=0.95,
    )
    fig.text(
        0.46, 0.115,
        r"Given the full coarse sequence $\{y_t\}_{t=0}^{T}$, infer the latent fine trajectory $\{\omega_t\}_{t=0}^{T}$",
        ha="center",
        va="center",
        fontsize=12,
    )

    _save(fig, "fig5_sequential_inverse_problem")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 2D miscellaneous figures")
    p.add_argument("--config",     type=str, default="configs/kraichnan.yaml")
    p.add_argument("--results",    type=str, default="results_2d/inference_results.pt")
    p.add_argument("--traj",       type=int, default=0,
                   help="Trajectory index to use for field plots (default: 0)")
    p.add_argument("--snapshot_t", type=int, default=50,
                   help="Time step for snapshot figures (default: 50)")
    p.add_argument("--figures",    type=str, default="1,2,3,4,5",
                   help="Comma-separated figures to generate (default: all)")
    p.add_argument("--fps",        type=int, default=5,
                   help="Frames per second for GIF animation (default: 5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    figs = {int(x) for x in args.figures.split(",")}

    print(f"Loading inference results from {args.results} ...")
    r = torch.load(args.results, map_location="cpu", weights_only=True)
    T = r["truth_256"].shape[1]
    print(f"  Trajectories: {r['truth_256'].shape[0]},  time steps: {T}")

    t = min(args.snapshot_t, T - 1)

    if 1 in figs:
        print("\nFigure 1: Coarse vs fine ...")
        plot_coarse_vs_fine(r, t=t, traj=args.traj)

    if 2 in figs:
        print("\nFigure 2: Annotated energy spectrum ...")
        plot_spectrum_annotated(r)

    if 3 in figs:
        print("\nFigure 3: Resolution pyramid ...")
        plot_resolution_pyramid(r, t=t, traj=args.traj)

    if 4 in figs:
        print("\nFigure 4: Evolution GIF ...")
        plot_evolution_gif(r, traj=args.traj, fps=args.fps)

    if 5 in figs:
        print("\nFigure 5: Sequential inverse problem schematic ...")
        plot_sequential_inverse_problem(r, traj=args.traj)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
