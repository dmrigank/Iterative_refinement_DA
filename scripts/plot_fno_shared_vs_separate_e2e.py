"""
End-to-end comparison: separate-FNO iterative refinement vs. shared-FNO
iterative refinement, against ground truth.

Loads:
  results_2d/inference_results.pt           — separate FNOs + diffusion (original)
  results_2d_sharedfno/inference_results.pt — shared FNO + retrained diffusion

Figures saved to plots_2d_sharedfno_comparison/ (style matches plots_edsr/):
  fig1_snapshot.{png,pdf}   — 2-row × 3-col snapshot (field + |error| rows) at one t
  fig3_spectrum.{png,pdf}   — Log-log E(k), all trajectories & time steps, 256×256
  fig6_rollout.{png,pdf}    — 5-row × N-col temporal rollout across one trajectory

Usage:
    python scripts/plot_fno_shared_vs_separate_e2e.py
        [--separate   results_2d/inference_results.pt]
        [--shared     results_2d_sharedfno/inference_results.pt]
        [--snapshot_t 50]
        [--traj       0]
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

from src.evaluation.metrics_2d import radial_energy_spectrum

# ---------------------------------------------------------------------------
# Style — matches plot_results_edsr.py
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  14,
    "legend.fontsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,
})

PLOTS_DIR = Path("plots_2d_sharedfno_comparison")
PLOTS_DIR.mkdir(exist_ok=True)

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "inferno"

C_TRUTH  = "black"
C_SEP    = "#1f77b4"   # blue — separate FNOs (original)
C_SHARED = "#d62728"   # red  — shared FNO (new)


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 1 — 2-row × 3-col snapshot
# ---------------------------------------------------------------------------

def plot_snapshot(rs: dict, rh: dict, truth_all: torch.Tensor,
                  t: int = 50, traj: int = 0) -> None:
    """2 rows × 3 cols at 256×256.

    Row 0: GT | Separate-FNO IR | Shared-FNO IR
    Row 1: (blank) | |Error| Separate | |Error| Shared
    """
    T = truth_all.shape[1]
    t = min(t, T - 1)

    gt_f  = truth_all[traj, t].numpy()
    sep_f = rs["posterior_256"][traj, t].numpy()
    sh_f  = rh["posterior_256"][traj, t].numpy()

    err_sep = np.abs(sep_f - gt_f)
    err_sh  = np.abs(sh_f  - gt_f)

    vmax = max(float(np.percentile(np.abs(gt_f), 99.5)), 1e-6)
    emax = max(float(np.percentile(np.maximum(err_sep, err_sh), 99.5)), 1e-6)

    col_titles  = ["Ground Truth", "Separate FNOs\n(original)", "Shared FNO\n(new)"]
    row0_fields = [gt_f,  sep_f,    sh_f   ]
    row1_errors = [None,  err_sep,  err_sh ]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.10, wspace=0.06,
        left=0.05, right=0.88, top=0.90, bottom=0.04,
    )

    im_field = im_err = None

    for col in range(3):
        ax0 = fig.add_subplot(gs[0, col])
        im  = ax0.imshow(row0_fields[col], cmap=CMAP_FIELD,
                         vmin=-vmax, vmax=vmax,
                         origin="lower", aspect="equal", interpolation="nearest")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(col_titles[col], fontsize=11, pad=5)
        if col == 0:
            ax0.set_ylabel("Field  ω", fontsize=10, labelpad=4)
        im_field = im

        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_xticks([]); ax1.set_yticks([])
        if row1_errors[col] is None:
            ax1.axis("off")
        else:
            im_e = ax1.imshow(row1_errors[col], cmap=CMAP_ERR,
                              vmin=0.0, vmax=emax,
                              origin="lower", aspect="equal", interpolation="nearest")
            im_err = im_e
        if col == 0:
            ax1.set_ylabel("|Error|", fontsize=10, labelpad=4)

    cbar1 = fig.add_axes([0.90, 0.52, 0.018, 0.38])
    fig.colorbar(im_field, cax=cbar1)
    cbar1.tick_params(labelsize=9)
    cbar1.set_ylabel("Vorticity  ω", fontsize=10)

    cbar2 = fig.add_axes([0.90, 0.06, 0.018, 0.38])
    fig.colorbar(im_err, cax=cbar2)
    cbar2.tick_params(labelsize=9)
    cbar2.set_ylabel("|Error|", fontsize=10)

    fig.suptitle(
        f"256×256 Vorticity Comparison — Separate vs. Shared FNO  |  "
        f"t={t}, trajectory {traj}",
        fontsize=13, y=0.98,
    )
    _save(fig, "fig1_snapshot")


# ---------------------------------------------------------------------------
# Figure 2 — Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(rs: dict, rh: dict, truth_all: torch.Tensor,
                  k_forcing: float = 4.0, k_max_full: int = 100) -> None:
    """Log-log E(k) for both methods, averaged over time and trajectories."""
    truth_flat = truth_all.reshape(-1, 256, 256)
    sep_flat   = rs["posterior_256"].reshape(-1, 256, 256)
    sh_flat    = rh["posterior_256"].reshape(-1, 256, 256)

    E_truth, k_bins = radial_energy_spectrum(truth_flat)
    E_sep,   _      = radial_energy_spectrum(sep_flat)
    E_sh,    _      = radial_energy_spectrum(sh_flat)

    k  = k_bins[1:].numpy()
    Et = E_truth[1:].numpy()
    Es = E_sep[1:].numpy()
    Eh = E_sh[1:].numpy()

    mask_full = k <= k_max_full

    # k^-3 reference anchored at k=5
    idx5  = np.searchsorted(k, 5)
    k_ref = np.array([3.0, float(k_max_full)])
    E_ref = Et[idx5] * (5.0 ** 3) * k_ref ** (-3)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k[mask_full], Et[mask_full], color=C_TRUTH,  lw=2.4, label="Ground truth",
               zorder=2)
    ax.loglog(k[mask_full], Es[mask_full], color=C_SEP,    lw=2.6, ls="-",
               label="Separate FNOs (original)", zorder=3, alpha=0.85)
    ax.loglog(k[mask_full], Eh[mask_full], color=C_SHARED, lw=1.6, ls="--",
               label="Shared FNO (new)", zorder=4)
    ax.loglog(k_ref, E_ref, color="gray",  lw=1.0, ls=":", label=r"$k^{-3}$", zorder=1)

    k_nyq = 16
    ax.axvline(k_nyq,     color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(k_nyq * 1.05, Et[mask_full].max() * 0.4, "32×32\nNyquist",
            color="gray", fontsize=9, va="top")
    ax.axvline(k_forcing, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(k_forcing * 1.1, Et[mask_full].max() * 0.15,
            rf"$k_f={k_forcing:.0f}$", color="gray", fontsize=10, va="top")

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("Radial Energy Spectrum  (256×256, all trajectories & time steps)")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=1, right=k_max_full * 1.1)
    fig.tight_layout()
    _save(fig, "fig3_spectrum")


# ---------------------------------------------------------------------------
# Figure 3 — Temporal rollout
# ---------------------------------------------------------------------------

def plot_rollout(rs: dict, rh: dict, truth_all: torch.Tensor,
                 traj: int = 0, n_cols: int = 5) -> None:
    """5 rows × n_cols cols at 256×256.

    Rows:
      0 — Ground Truth
      1 — Separate-FNO IR
      2 — |Error| Separate
      3 — Shared-FNO IR
      4 — |Error| Shared

    Columns: n_cols evenly-spaced time steps from t=1 to t=T-1.
    """
    T = truth_all.shape[1]
    t_steps = np.linspace(1, T - 1, n_cols, dtype=int)

    truth = truth_all[traj]              # (T, 256, 256)
    sep   = rs["posterior_256"][traj]    # (T, 256, 256)
    sh    = rh["posterior_256"][traj]    # (T, 256, 256)

    vmax = max(float(np.percentile(np.abs(truth.numpy()), 99.5)), 1e-6)

    err_stack = torch.stack([
        (sep[t_steps] - truth[t_steps]).abs(),
        (sh[t_steps]  - truth[t_steps]).abs(),
    ], dim=0)
    emax = max(float(err_stack.max().item()), 1e-6)

    row_labels = [
        "Ground Truth",
        "Separate FNOs",
        "|Error| Separate",
        "Shared FNO",
        "|Error| Shared",
    ]
    n_rows = len(row_labels)

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.06, wspace=0.04,
        left=0.10, right=0.92, top=0.955, bottom=0.02,
    )

    last_im_field = last_im_err = None

    for col_idx, t in enumerate(t_steps):
        row_data = [
            (truth[t].numpy(),                CMAP_FIELD, -vmax, vmax),
            (sep[t].numpy(),                  CMAP_FIELD, -vmax, vmax),
            ((sep[t] - truth[t]).abs().numpy(), CMAP_ERR, 0.0,  emax),
            (sh[t].numpy(),                   CMAP_FIELD, -vmax, vmax),
            ((sh[t] - truth[t]).abs().numpy(),  CMAP_ERR, 0.0,  emax),
        ]

        for row_idx, (field, cmap, vmin, vmx) in enumerate(row_data):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmx,
                           origin="lower", aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"t = {t}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9, labelpad=4)
            if cmap == CMAP_FIELD:
                last_im_field = im
            else:
                last_im_err = im

    cbar1 = fig.add_axes([0.933, 0.50, 0.012, 0.45])
    fig.colorbar(last_im_field, cax=cbar1)
    cbar1.tick_params(labelsize=9)
    cbar1.set_ylabel("Vorticity  ω", fontsize=10)

    cbar2 = fig.add_axes([0.933, 0.03, 0.012, 0.44])
    fig.colorbar(last_im_err, cax=cbar2)
    cbar2.tick_params(labelsize=9)
    cbar2.set_ylabel("|Error|", fontsize=10)

    fig.suptitle(
        f"Temporal Rollout — Separate vs. Shared FNO  |  256×256,  trajectory {traj}",
        fontsize=13, y=0.992,
    )
    _save(fig, "fig6_rollout")


# ---------------------------------------------------------------------------
# Arg parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare separate-FNO vs shared-FNO iterative refinement")
    p.add_argument("--separate",   type=str, default="results_2d/inference_results.pt")
    p.add_argument("--shared",     type=str, default="results_2d_sharedfno/inference_results.pt")
    p.add_argument("--snapshot_t", type=int, default=50)
    p.add_argument("--traj",       type=int, default=0)
    p.add_argument("--figures",    type=str, default="1,2,3")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    figs = {int(x) for x in args.figures.split(",")}

    print("Loading results...")
    rs = torch.load(args.separate, map_location="cpu", weights_only=True)
    rh = torch.load(args.shared,   map_location="cpu", weights_only=True)

    n_traj = min(rs["truth_256"].shape[0], rh["truth_256"].shape[0])
    T      = min(rs["truth_256"].shape[1], rh["truth_256"].shape[1])

    truth_all = rs["truth_256"][:n_traj, :T].float()

    def _trim(d: dict) -> dict:
        return {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                for k, v in d.items()}
    rs = _trim(rs)
    rh = _trim(rh)

    print(f"Using n_traj={n_traj}, T={T}")
    print(f"  Separate FNO  RMSE@256 = {rs['metrics']['rmse_posterior_256']:.4f}")
    print(f"  Shared FNO    RMSE@256 = {rh['metrics']['rmse_posterior_256']:.4f}")

    if 1 in figs:
        print("\nFigure 1: snapshot comparison...")
        plot_snapshot(rs, rh, truth_all, t=args.snapshot_t, traj=args.traj)

    if 2 in figs:
        print("\nFigure 2: energy spectrum...")
        plot_spectrum(rs, rh, truth_all)

    if 3 in figs:
        print("\nFigure 3: temporal rollout...")
        plot_rollout(rs, rh, truth_all, traj=args.traj)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
