"""
Publication-quality energy spectrum figures for the paper.

Produces two figures:

spectrum_schematic.{png,pdf}
    Single-panel schematic showing E(k) averaged over all time/trajectories.
    Highlights that IR uniquely recovers fine-scale energy beyond the obs. Nyquist.
    OI is excluded — it is not a fair method-level comparison (see run_inference_oi.py).
    Methods shown: Ground truth | Bicubic | EDSR | One-shot SR | IR (ours).

fig3a_spectrum_panels.{png,pdf}
    2×2 panel version of the above, one panel per time slice (t=T/4, T/2, 3T/4, T-1),
    following the same style as plots_edsr/fig3_spectrum.png.
    Shows that the spectral advantage of IR is consistent across the whole trajectory.

Usage:
    python scripts/plot_spectrum_schematic.py
        [--iterative  results_2d/inference_results.pt]
        [--oneshot    results_oneshot/inference_results.pt]
        [--edsr       results_edsr/inference_results.pt]
        [--out_dir    plots_2d_misc]
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
from matplotlib.lines import Line2D

from src.evaluation.metrics_2d import radial_energy_spectrum


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       13,
    "axes.labelsize":  14,
    "axes.titlesize":  14,
    "legend.fontsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "axes.grid":       True,
    "grid.alpha":      0.22,
    "grid.linestyle":  ":",
})

C_TRUTH = "#111111"
C_BIC   = "#aaaaaa"
C_EDSR  = "#ff7f0e"
C_ONE   = "#2ca02c"
C_IR    = "#1f77b4"

K_NYQ   = 16
K_FORCE = 4


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _E_avg(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Radial E(k) averaged over all (traj, time) frames."""
    E, k = radial_energy_spectrum(x.reshape(-1, *x.shape[-2:]))
    return E.numpy(), k.numpy()


def _E_at_t(x: torch.Tensor, t: int) -> np.ndarray:
    """Radial E(k) averaged over trajectories at a single time step t."""
    E, _ = radial_energy_spectrum(x[:, t])   # (n_traj, 256, 256)
    return E.numpy()


def _k_ref(k: np.ndarray, E_truth: np.ndarray, k_anchor: float = 5.0,
           k_max: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """k^{-3} reference line anchored at k_anchor."""
    idx    = np.argmin(np.abs(k - k_anchor))
    k_line = np.array([3.0, k_max])
    E_line = E_truth[idx] * (k_anchor ** 3) * k_line ** (-3)
    return k_line, E_line


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_dir}/{stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 1 — Schematic (single panel, annotated, no OI)
# ---------------------------------------------------------------------------

def plot_schematic(
    ri: dict, ro: dict, re: dict,
    out_dir: Path,
    k_max_plot: int = 90,
) -> None:
    truth = ri["truth_256"].float()

    E_truth, k = _E_avg(truth)
    E_bic,   _ = _E_avg(ro["bicubic_256"].float())
    E_edsr,  _ = _E_avg(re["sr_256"].float())
    E_one,   _ = _E_avg(ro["posterior_256"].float())
    E_ir,    _ = _E_avg(ri["posterior_256"].float())

    mask     = (k >= 1) & (k <= k_max_plot)
    mask_bic = (k >= 1) & (k <= K_NYQ + 1)
    k_p      = k[mask]

    k_line, E_line = _k_ref(k[mask], E_truth[mask])

    fig, ax = plt.subplots(figsize=(9.8, 6.2), constrained_layout=True)

    # ── Regime shading ────────────────────────────────────────────────────────
    ax.axvspan(1, K_NYQ,       color="#1f77b4", alpha=0.07, zorder=0)
    ax.axvspan(K_NYQ, k_max_plot, color="#d62728", alpha=0.05, zorder=0)

    # Region labels use an axis-fraction y coordinate so they stay in the
    # available headroom regardless of the spectral range.
    region_bbox = dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88, lw=1.0)
    ax.text(1.15, 0.965,
            "Observed modes\n$k \\leq 16$",
            transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9.5, color="#1a5fa8",
            fontweight="bold",
            bbox={**region_bbox, "ec": "#1f77b4"})
    ax.text(np.sqrt(K_NYQ * k_max_plot), 0.965,
            "Fine-scale recovery\n$k > 16$",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9.5, color="#8b0000",
            fontweight="bold",
            bbox={**region_bbox, "ec": "#d62728"})

    # ── Curves ────────────────────────────────────────────────────────────────
    ax.loglog(k_p, E_truth[mask],
              color=C_TRUTH, lw=3.0, zorder=10, label="Ground truth")

    # Bicubic: solid up to Nyquist then vertical drop to show hard cutoff
    ax.loglog(k[mask_bic], E_bic[mask_bic],
              color=C_BIC, lw=2.2, ls=(0, (4, 2)), zorder=5,
              label="Bicubic  (zero for k > 16)")
    ax.loglog([K_NYQ, K_NYQ + 0.01],
              [E_bic[mask_bic][-1], E_truth[mask].min() * 5e-2],
              color=C_BIC, lw=2.2, ls=(0, (4, 2)), zorder=5)

    ax.loglog(k_p, E_edsr[mask],
              color=C_EDSR, lw=2.0, ls=(0, (3, 1.5)), zorder=6,
              label="EDSR  (deterministic SR)")
    ax.loglog(k_p, E_one[mask],
              color=C_ONE,  lw=2.0, ls="--",          zorder=7,
              label="One-shot diffusion SR")
    ax.loglog(k_p, E_ir[mask],
              color=C_IR,   lw=3.2,                   zorder=9,
              label="Iterative Refinement (ours)")
    ax.loglog(k_line, E_line,
              color="gray", lw=1.2, ls=":",            zorder=4,
              label=r"$k^{-3}$  enstrophy cascade")

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axvline(K_NYQ,   color="#444444", ls="--", lw=1.3, zorder=8, alpha=0.65)
    ax.axvline(K_FORCE, color="#888888", ls=":",  lw=1.0, zorder=3, alpha=0.55)

    # Obs Nyquist label, placed beside the reference line without touching the
    # low-energy annotations.
    ax.text(K_NYQ * 0.94, 0.36,
            r"$k^\mathrm{Nyq}_\mathrm{obs}=16$",
            transform=ax.get_xaxis_transform(),
            ha="right", va="center", fontsize=9.5, color="#444444", rotation=90,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72))
    ax.text(K_FORCE * 1.08, 0.12,
            r"$k_f = 4$",
            transform=ax.get_xaxis_transform(),
            ha="left", va="bottom", fontsize=9.5, color="#666666")

    # ── Quantitative annotations at k=48 and k=32 ────────────────────────────
    # Placed in a gap well away from curves using fixed data coordinates.
    # All arrows point FROM text TO the curve.

    # k=48: IR gap
    k48   = 48
    i48   = np.argmin(np.abs(k_p - k48))
    k48_p = k_p[i48]
    Et48  = E_truth[mask][i48]
    Ei48  = E_ir[mask][i48]
    Eo48  = E_one[mask][i48]
    frac_ir  = Ei48 / Et48
    frac_one = Eo48 / Et48

    note_bbox = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.86)

    # IR: arrow from label (above the curve) pointing down to the IR curve
    ax.annotate(
        f"IR: {frac_ir*100:.0f}% of GT",
        xy=(k48_p, Ei48),
        xytext=(25.0, 7.0e-3),
        fontsize=10.5, color=C_IR, fontweight="bold", ha="left",
        bbox=note_bbox,
        arrowprops=dict(arrowstyle="->", color=C_IR, lw=1.6,
                        shrinkA=4, shrinkB=4, mutation_scale=12,
                        connectionstyle="arc3,rad=0.15"),
    )

    # One-shot: keep text inside the plotting area with a short, clean pointer.
    ax.annotate(
        f"One-shot:\n{frac_one*100:.0f}% of GT",
        xy=(k48_p, Eo48),
        xytext=(61.0, 7.8e-4),
        fontsize=10.5, color=C_ONE, fontweight="bold", ha="left",
        bbox=note_bbox,
        arrowprops=dict(arrowstyle="->", color=C_ONE, lw=1.6,
                        shrinkA=4, shrinkB=4, mutation_scale=12,
                        connectionstyle="arc3,rad=-0.18"),
    )

    # EDSR: annotate at k=32 (still above noise floor)
    k32   = 32
    i32   = np.argmin(np.abs(k_p - k32))
    k32_p = k_p[i32]
    Ee32  = E_edsr[mask][i32]
    Et32  = E_truth[mask][i32]
    frac_e = Ee32 / Et32
    ax.annotate(
        f"EDSR:\n{frac_e*100:.0f}% at k=32",
        xy=(k32_p, Ee32),
        xytext=(20.2, 1.45e-5),
        fontsize=10.5, color=C_EDSR, fontweight="bold", ha="left",
        bbox=note_bbox,
        arrowprops=dict(arrowstyle="->", color=C_EDSR, lw=1.5,
                        shrinkA=4, shrinkB=4, mutation_scale=12,
                        connectionstyle="arc3,rad=-0.28"),
    )

    # Explicit target markers keep annotation endpoints legible where several
    # spectra are close together.
    ax.scatter([k48_p, k48_p, k32_p], [Ei48, Eo48, Ee32],
               s=[34, 34, 38],
               facecolors=["white", "white", "white"],
               edgecolors=[C_IR, C_ONE, C_EDSR],
               linewidths=1.8, zorder=18)

    # Bicubic zero label
    ax.annotate(
        "Bicubic cutoff\nzero for $k > 16$",
        xy=(K_NYQ + 0.7, E_truth[mask].min() * 28),
        xytext=(18.6, 1.05e-4),
        fontsize=9.6, color="#7b7b7b", fontweight="bold", ha="left",
        bbox=note_bbox,
        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3,
                        connectionstyle="arc3,rad=0.24"),
    )

    # ── Legend (lower-left, away from annotations) ────────────────────────────
    handles = [
        Line2D([0],[0], color=C_TRUTH, lw=3.0,               label="Ground truth"),
        Line2D([0],[0], color=C_BIC,   lw=2.2, ls=(0,(4,2)), label="Bicubic  (zero for k > 16)"),
        Line2D([0],[0], color=C_EDSR,  lw=2.0, ls=(0,(3,1.5)), label="EDSR  (deterministic SR)"),
        Line2D([0],[0], color=C_ONE,   lw=2.0, ls="--",       label="One-shot diffusion SR"),
        Line2D([0],[0], color=C_IR,    lw=3.2,                label="Iterative Refinement (ours)"),
        Line2D([0],[0], color="gray",  lw=1.2, ls=":",        label=r"$k^{-3}$  enstrophy cascade"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=9.8,
                    framealpha=0.94, edgecolor="#cccccc", ncol=1,
                    borderpad=0.45, handlelength=2.4, labelspacing=0.42)
    leg.set_zorder(20)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlabel("Wavenumber  $k$", fontsize=14)
    ax.set_ylabel(r"$E(k) = \langle|\hat{\omega}_k|^2\rangle$", fontsize=14)
    ax.set_xlim(1, k_max_plot)
    E_min = E_truth[mask].min()
    E_max = E_truth[mask].max()
    ax.set_ylim(E_min * 0.13, E_max * 5.0)
    ax.set_title(
        "Iterative diffusion refinement uniquely recovers\n"
        "fine-scale vorticity structure beyond the observation resolution",
        fontsize=15, pad=12,
    )
    _save(fig, out_dir, "spectrum_schematic")


# ---------------------------------------------------------------------------
# Figure 3a — 2×2 panel E(k) at four time slices
# ---------------------------------------------------------------------------

def plot_spectrum_panels(
    ri: dict, ro: dict, re: dict,
    out_dir: Path,
    k_max_full: int = 100,
    k_max_bic:  int = 22,
) -> None:
    """2×2 grid of E(k) plots at t = T/4, T/2, 3T/4, T-1.

    Mirrors the style of plots_edsr/fig3_spectrum.png but shows how the
    spectral advantage of IR is maintained consistently across the trajectory.
    Averaged over all test trajectories at each time slice.
    """
    # Align to the shortest T across all result files
    T = min(
        ri["posterior_256"].shape[1],
        ro["posterior_256"].shape[1],
        re["sr_256"].shape[1],
        ri["truth_256"].shape[1],
    )
    n_traj = min(
        ri["truth_256"].shape[0],
        ro["posterior_256"].shape[0],
        re["sr_256"].shape[0],
    )
    truth = ri["truth_256"][:n_traj, :T].float()

    # Four representative time indices
    t_slices = [T // 4, T // 2, 3 * T // 4, T - 1]
    t_labels = ["t = T/4", "t = T/2", "t = 3T/4", "t = T−1"]

    # Compute E(k) at each time slice for each method (avg over trajectories)
    def _E_slice(x: torch.Tensor, t: int) -> np.ndarray:
        E, _ = radial_energy_spectrum(x[:n_traj, t])
        return E.numpy()

    # Pre-fetch all slice spectra — trim each tensor to common (n_traj, T) first
    def _all_slices(x: torch.Tensor) -> list[np.ndarray]:
        return [_E_slice(x[:n_traj, :T].float(), t) for t in t_slices]

    Et_slices    = _all_slices(ri["truth_256"])
    Ebic_slices  = _all_slices(ro["bicubic_256"])
    Eedsr_slices = _all_slices(re["sr_256"])
    Eone_slices  = _all_slices(ro["posterior_256"])
    Eir_slices   = _all_slices(ri["posterior_256"])

    # Wavenumber axis (shared)
    _, k_bins = radial_energy_spectrum(truth[:, 0])
    k = k_bins[1:].numpy()
    mask_full = k <= k_max_full
    mask_bic  = k <= k_max_bic
    k_full    = k[mask_full]
    k_bic     = k[mask_bic]

    # Global y limits for shared axes
    y_max = max(Et[1:][mask_full].max() for Et in Et_slices) * 3.5
    y_min = min(Et[1:][mask_full].min() for Et in Et_slices) * 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5),
                             sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for panel_idx, (ax, t, label) in enumerate(
            zip(axes_flat, t_slices, t_labels)):

        Et   = Et_slices[panel_idx][1:]
        Ebic = Ebic_slices[panel_idx][1:]
        Ee   = Eedsr_slices[panel_idx][1:]
        Eo   = Eone_slices[panel_idx][1:]
        Ei   = Eir_slices[panel_idx][1:]

        # k^{-3} reference anchored at k=5
        k_line, E_line = _k_ref(k_full, Et[mask_full], k_anchor=5.0,
                                 k_max=float(k_max_full))

        # Curves
        ax.loglog(k_full, Et[mask_full],
                  color=C_TRUTH, lw=2.3, ls="-",       label="Ground truth")
        ax.loglog(k_bic,  Ebic[mask_bic],
                  color=C_BIC,   lw=1.8, ls="-.",      label="Bicubic")
        ax.loglog(k_full, Ee[mask_full],
                  color=C_EDSR,  lw=1.8, ls=(0,(3,1.5)), label="EDSR")
        ax.loglog(k_full, Eo[mask_full],
                  color=C_ONE,   lw=1.8, ls="--",      label="One-shot SR")
        ax.loglog(k_full, Ei[mask_full],
                  color=C_IR,    lw=2.3,                label="Iterative Refinement")
        ax.loglog(k_line, E_line,
                  color="gray",  lw=1.0, ls=":",       label=r"$k^{-3}$")

        # Obs Nyquist marker
        ax.axvline(K_NYQ,   color="gray", ls=":", lw=0.9, alpha=0.65)
        ax.axvline(K_FORCE, color="gray", ls=":", lw=0.8, alpha=0.50)

        ax.text(K_NYQ * 1.05, y_max * 0.45, "Nyquist\n(k=16)",
                color="gray", fontsize=8.5, va="top")

        # Time-step label in upper-right corner
        ax.text(0.97, 0.97, label,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#aaaaaa", alpha=0.88))

        # Axis labels only on outer panels
        if panel_idx in (2, 3):   # bottom row
            ax.set_xlabel("Wavenumber  $k$", fontsize=13)
        if panel_idx in (0, 2):   # left column
            ax.set_ylabel(r"$E(k) = \langle|\hat{\omega}_k|^2\rangle$",
                          fontsize=13)

        ax.set_xlim(1, k_max_full * 1.05)
        ax.set_ylim(y_min, y_max)

    # Shared legend above the panels
    handles = [
        Line2D([0],[0], color=C_TRUTH, lw=2.3,               label="Ground truth"),
        Line2D([0],[0], color=C_BIC,   lw=1.8, ls="-.",      label="Bicubic"),
        Line2D([0],[0], color=C_EDSR,  lw=1.8, ls=(0,(3,1.5)), label="EDSR"),
        Line2D([0],[0], color=C_ONE,   lw=1.8, ls="--",      label="One-shot SR"),
        Line2D([0],[0], color=C_IR,    lw=2.3,                label="Iterative Refinement (ours)"),
        Line2D([0],[0], color="gray",  lw=1.0, ls=":",       label=r"$k^{-3}$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               fontsize=11, framealpha=0.93, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, 0.995))

    fig.suptitle(
        "Radial energy spectrum at four trajectory time points  |  256×256\n"
        "Averaged over test trajectories",
        fontsize=14, y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir, "fig3a_spectrum_panels")


# ---------------------------------------------------------------------------
# Arg parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spectrum schematic + panel figures")
    p.add_argument("--iterative", type=str, default="results_2d/inference_results.pt")
    p.add_argument("--oneshot",   type=str, default="results_oneshot/inference_results.pt")
    p.add_argument("--edsr",      type=str, default="results_edsr/inference_results.pt")
    p.add_argument("--out_dir",   type=str, default="plots_2d_misc")
    p.add_argument("--figures",   type=str, default="schematic,panels",
                   help="Comma-separated: schematic, panels (default: both)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs    = {f.strip() for f in args.figures.split(",")}

    print("Loading results ...")
    ri = torch.load(args.iterative, map_location="cpu", weights_only=True)
    ro = torch.load(args.oneshot,   map_location="cpu", weights_only=True)
    re = torch.load(args.edsr,      map_location="cpu", weights_only=True)

    if "schematic" in figs:
        print("\nFigure: spectrum schematic ...")
        plot_schematic(ri, ro, re, out_dir)

    if "panels" in figs:
        print("\nFigure: fig3a spectrum panels ...")
        plot_spectrum_panels(ri, ro, re, out_dir)


if __name__ == "__main__":
    main()
