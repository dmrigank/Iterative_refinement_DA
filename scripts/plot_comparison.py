"""
Publication-quality comparison figures across all methods.

Loads:
  results_2d/inference_results.pt        — iterative refinement
  results_oneshot/inference_results.pt   — one-shot SR + bicubic baseline
  data_2d/test.pt                        — ground truth

Figures saved to plots_oneshot/:
  fig1_method_comparison.{png,pdf}     — 2-row × 4-col snapshot (field + error rows)
  fig2_rmse_comparison.{png,pdf}       — RMSE over time, all methods
  fig3_spectrum_comparison.{png,pdf}   — Log-log energy spectrum (truncated)
  fig4_temporal_consistency.{png,pdf}  — Frame-to-frame L2 displacement
  fig5_summary_bars.{png,pdf}          — Grouped bar chart of all metrics
  fig6_rollout_comparison.{png,pdf}    — 7-row × 5-col temporal rollout

Usage:
    python scripts/plot_comparison.py
        [--iterative  results_2d/inference_results.pt]
        [--oneshot    results_oneshot/inference_results.pt]
        [--data_dir   data_2d]
        [--config     configs/kraichnan.yaml]
        [--figures    1,2,3,4,5,6]
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
from omegaconf import OmegaConf

from src.evaluation.metrics_2d import (
    radial_energy_spectrum,
    rmse_over_time_2d,
    temporal_consistency_2d,
    structural_similarity_2d,
    rmse_2d,
)


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
})

PLOTS_DIR = Path("plots_oneshot")
PLOTS_DIR.mkdir(exist_ok=True)

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "inferno"

C_TRUTH = "black"
C_BIC   = "gray"
C_ONE   = "#2ca02c"   # green
C_ITER  = "#1f77b4"   # blue
C_FNO   = "#d62728"   # red


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Spectral RMSE helper (scalar: RMS difference of E(k) curves in log space)
# ---------------------------------------------------------------------------

def _spectral_rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """RMS error in log10(E(k)) between two (n_time, ny, nx) fields."""
    E_pred,  k = radial_energy_spectrum(pred.reshape(-1, *pred.shape[-2:]))
    E_truth, _ = radial_energy_spectrum(truth.reshape(-1, *truth.shape[-2:]))
    # compare over k >= 1 in log space
    mask = k >= 1
    log_pred  = E_pred[mask].clamp(min=1e-30).log10()
    log_truth = E_truth[mask].clamp(min=1e-30).log10()
    return float((log_pred - log_truth).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Figure 1 — 2-row × 4-col snapshot (field row + error row)
# ---------------------------------------------------------------------------

def plot_method_comparison(ri: dict, ro: dict, truth_all: torch.Tensor,
                           t: int = 50, traj: int = 0) -> None:
    """2 rows × 4 cols at 256×256.
    Row 0: GT | FNO-only | One-Shot SR | Iterative Refinement
    Row 1: (blank) | |Error| FNO-only | |Error| One-Shot | |Error| Iterative
    Both error rows share the same colorscale (derived from the max FNO-only error).
    """
    T = truth_all.shape[1]
    t = min(t, T - 1)

    gt_field   = truth_all[traj, t].numpy()
    fno_field  = ri["fno_only_256"][traj, t].numpy()
    one_field  = ro["posterior_256"][traj, t].numpy()
    iter_field = ri["posterior_256"][traj, t].numpy()

    err_fno  = np.abs(fno_field  - gt_field)
    err_one  = np.abs(one_field  - gt_field)
    err_iter = np.abs(iter_field - gt_field)

    vmax = float(np.percentile(np.abs(gt_field), 99.5))
    vmax = max(vmax, 1e-6)
    # Shared error scale — derived from FNO-only error (largest, sets the ceiling)
    emax = float(np.percentile(err_fno, 99.5))
    emax = max(emax, 1e-6)

    col_titles  = ["Ground Truth", "FNO-only (autoreg.)",
                   "One-Shot SR", "Iterative Refinement"]
    row0_fields = [gt_field,  fno_field,  one_field,  iter_field]
    row1_fields = [None,      err_fno,    err_one,    err_iter]

    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.06, wspace=0.05,
        left=0.04, right=0.92, top=0.93, bottom=0.04,
    )

    im_field = None   # for shared field colorbar
    im_err   = None   # for shared error colorbar

    for col in range(4):
        # Row 0: field
        ax0 = fig.add_subplot(gs[0, col])
        im  = ax0.imshow(row0_fields[col], cmap=CMAP_FIELD,
                         vmin=-vmax, vmax=vmax,
                         origin="lower", aspect="equal", interpolation="nearest")
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(col_titles[col], fontsize=12, pad=5)
        if col == 0:
            ax0.set_ylabel("Field  ω", fontsize=10, labelpad=4)
        im_field = im

        # Row 1: error (blank for GT column)
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_xticks([])
        ax1.set_yticks([])
        if row1_fields[col] is None:
            ax1.axis("off")
        else:
            im_e = ax1.imshow(row1_fields[col], cmap=CMAP_ERR,
                              vmin=0.0, vmax=emax,
                              origin="lower", aspect="equal", interpolation="nearest")
            im_err = im_e
        if col == 0:
            ax1.set_ylabel("|Error|", fontsize=10, labelpad=4)

    # Colorbars on the right
    # Field colorbar spans row 0
    cbar_ax1 = fig.add_axes([0.933, 0.52, 0.012, 0.40])
    fig.colorbar(im_field, cax=cbar_ax1)
    cbar_ax1.tick_params(labelsize=9)
    cbar_ax1.set_ylabel("Vorticity  ω", fontsize=10)

    # Error colorbar spans row 1
    cbar_ax2 = fig.add_axes([0.933, 0.06, 0.012, 0.40])
    fig.colorbar(im_err, cax=cbar_ax2)
    cbar_ax2.tick_params(labelsize=9)
    cbar_ax2.set_ylabel("|Error|", fontsize=10)

    fig.suptitle(f"256×256 Vorticity Comparison  |  t={t},  trajectory {traj}",
                 fontsize=13, y=0.98)
    _save(fig, "fig1_method_comparison")


# ---------------------------------------------------------------------------
# Figure 2 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_comparison(ri: dict, ro: dict, truth_all: torch.Tensor) -> None:
    """All methods on one RMSE-vs-time plot at 256×256."""

    n_traj = truth_all.shape[0]
    T      = truth_all.shape[1]

    iter_post = ri["posterior_256"]   # (n_traj, T, 256, 256)
    fno_only  = ri["fno_only_256"]
    one_post  = ro["posterior_256"]
    bic       = ro["bicubic_256"]
    truth     = truth_all

    def _curves(pred, tru):
        return torch.stack([rmse_over_time_2d(pred[i], tru[i])
                            for i in range(n_traj)])   # (n_traj, T)

    c_iter = _curves(iter_post, truth)
    c_fno  = _curves(fno_only,  truth)
    c_one  = _curves(one_post,  truth)
    c_bic  = _curves(bic,       truth)

    t_ax = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n_traj):
        ax.plot(t_ax, c_bic[i].numpy(),  color=C_BIC,  alpha=0.25, lw=0.8)
        ax.plot(t_ax, c_fno[i].numpy(),  color=C_FNO,  alpha=0.25, lw=0.8)
        ax.plot(t_ax, c_one[i].numpy(),  color=C_ONE,  alpha=0.25, lw=0.8)
        ax.plot(t_ax, c_iter[i].numpy(), color=C_ITER, alpha=0.25, lw=0.8)

    ax.plot(t_ax, c_bic.mean(0).numpy(),
            color=C_BIC,  lw=2.0, ls="--",   label="Bicubic (spectral upsample)")
    ax.plot(t_ax, c_fno.mean(0).numpy(),
            color=C_FNO,  lw=2.0, ls=":",    label="FNO-only (autoregressive)")
    ax.plot(t_ax, c_one.mean(0).numpy(),
            color=C_ONE,  lw=2.2, ls="-",    label="One-Shot SR")
    ax.plot(t_ax, c_iter.mean(0).numpy(),
            color=C_ITER, lw=2.2, ls="-",    label="Iterative Refinement")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Over Time  (256×256)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "fig2_rmse_comparison")


# ---------------------------------------------------------------------------
# Figure 3 — Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum_comparison(ri: dict, ro: dict, truth_all: torch.Tensor,
                             k_forcing: float = 4.0,
                             k_max_full: int = 100,
                             k_max_bic: int = 22) -> None:
    """Log-log E(k) for all methods, averaged over time and trajectories.

    Truncation:
      - GT, One-Shot SR, Iterative Refinement: show up to k_max_full (default 100)
        — beyond this the diffusion noise dominates and the curves diverge.
      - Bicubic: show up to k_max_bic (default 22, just above 32×32 Nyquist=16)
        — the bicubic spectrum falls off sharply at k>16; showing further is
        misleading since those modes are pure spectral interpolation artifacts.
    """
    truth_flat = truth_all.reshape(-1, 256, 256)
    iter_flat  = ri["posterior_256"].reshape(-1, 256, 256)
    one_flat   = ro["posterior_256"].reshape(-1, 256, 256)
    bic_flat   = ro["bicubic_256"].reshape(-1, 256, 256)

    E_truth, k_bins = radial_energy_spectrum(truth_flat)
    E_iter,  _      = radial_energy_spectrum(iter_flat)
    E_one,   _      = radial_energy_spectrum(one_flat)
    E_bic,   _      = radial_energy_spectrum(bic_flat)

    k  = k_bins[1:].numpy()
    Et = E_truth[1:].numpy()
    Ei = E_iter[1:].numpy()
    Eo = E_one[1:].numpy()
    Eb = E_bic[1:].numpy()

    # Masks for truncation
    mask_full = k <= k_max_full
    mask_bic  = k <= k_max_bic

    # k^-3 reference anchored at k=5, up to k_max_full
    idx5  = np.searchsorted(k, 5)
    k_ref = np.array([3.0, float(k_max_full)])
    E_ref = Et[idx5] * (5.0 ** 3) * k_ref ** (-3)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k[mask_full], Et[mask_full], color=C_TRUTH, lw=2.0, label="Ground truth")
    ax.loglog(k[mask_bic],  Eb[mask_bic],  color=C_BIC,  lw=1.5, ls="--", label="Bicubic")
    ax.loglog(k[mask_full], Eo[mask_full], color=C_ONE,  lw=1.8, label="One-Shot SR")
    ax.loglog(k[mask_full], Ei[mask_full], color=C_ITER, lw=1.8, label="Iterative Refinement")
    ax.loglog(k_ref, E_ref, color="gray",  lw=1.0, ls=":", label=r"$k^{-3}$")

    k_nyq_32 = 16
    ax.axvline(k_nyq_32,  color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(k_nyq_32 * 1.05, Et[mask_full].max() * 0.4, "32×32\nNyquist",
            color="gray", fontsize=9, va="top")
    ax.axvline(k_forcing, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(k_forcing * 1.1, Et[mask_full].max() * 0.15, rf"$k_f={k_forcing:.0f}$",
            color="gray", fontsize=10, va="top")

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("Radial Energy Spectrum  (256×256, all trajectories & time steps)")
    ax.legend(loc="upper right")
    ax.set_xlim(left=1, right=k_max_full * 1.1)
    fig.tight_layout()
    _save(fig, "fig3_spectrum_comparison")


# ---------------------------------------------------------------------------
# Figure 4 — Temporal consistency
# ---------------------------------------------------------------------------

def plot_temporal_consistency(ri: dict, ro: dict, truth_all: torch.Tensor) -> None:
    """Frame-to-frame L2 displacement over time, all methods."""

    n_traj = truth_all.shape[0]

    def _tc_curves(seq):
        # seq: (n_traj, T, ny, nx)
        return torch.stack([temporal_consistency_2d(seq[i]) for i in range(n_traj)])
        # -> (n_traj, T-1)

    tc_truth = _tc_curves(truth_all)
    tc_iter  = _tc_curves(ri["posterior_256"])
    tc_one   = _tc_curves(ro["posterior_256"])
    tc_bic   = _tc_curves(ro["bicubic_256"])

    T  = tc_truth.shape[1]
    t_ax = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n_traj):
        ax.plot(t_ax, tc_bic[i].numpy(),   color=C_BIC,   alpha=0.25, lw=0.8)
        ax.plot(t_ax, tc_one[i].numpy(),   color=C_ONE,   alpha=0.25, lw=0.8)
        ax.plot(t_ax, tc_iter[i].numpy(),  color=C_ITER,  alpha=0.25, lw=0.8)
        ax.plot(t_ax, tc_truth[i].numpy(), color=C_TRUTH, alpha=0.20, lw=0.8)

    ax.plot(t_ax, tc_bic.mean(0).numpy(),
            color=C_BIC,   lw=2.0, ls="--",  label="Bicubic")
    ax.plot(t_ax, tc_one.mean(0).numpy(),
            color=C_ONE,   lw=2.2, ls="-",   label="One-Shot SR")
    ax.plot(t_ax, tc_iter.mean(0).numpy(),
            color=C_ITER,  lw=2.2, ls="-",   label="Iterative Refinement")
    ax.plot(t_ax, tc_truth.mean(0).numpy(),
            color=C_TRUTH, lw=2.0, ls="-.",  label="Ground Truth")

    ax.set_xlabel("Time step  $t$")
    ax.set_ylabel(r"$\|w_t - w_{t-1}\|_2$")
    ax.set_title("Temporal Consistency  (frame-to-frame L2 displacement, 256×256)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "fig4_temporal_consistency")


# ---------------------------------------------------------------------------
# Figure 5 — Summary grouped bar chart
# ---------------------------------------------------------------------------

def plot_summary_bars(ri: dict, ro: dict, truth_all: torch.Tensor) -> None:
    """Grouped bar chart: RMSE / Spectral RMSE / Temporal Consistency / SSIM."""

    n_traj = truth_all.shape[0]

    methods   = ["Bicubic", "One-Shot SR", "Iterative\nRefinement"]
    colors    = [C_BIC, C_ONE, C_ITER]
    preds     = [ro["bicubic_256"], ro["posterior_256"], ri["posterior_256"]]
    truths    = [truth_all, truth_all, truth_all]

    metric_names  = ["RMSE", "Spectral RMSE", "Temp. Consistency", "SSIM"]
    n_metrics = len(metric_names)
    n_methods = len(methods)

    # Compute per-trajectory metrics for error bars
    all_means = np.zeros((n_methods, n_metrics))
    all_stds  = np.zeros((n_methods, n_metrics))

    for m_idx, (pred, truth) in enumerate(zip(preds, truths)):
        rmse_vals = np.array([
            float(rmse_2d(pred[i], truth[i]))
            for i in range(n_traj)
        ])
        spec_vals = np.array([
            _spectral_rmse(pred[i:i+1], truth[i:i+1])
            for i in range(n_traj)
        ])
        tc_vals = np.array([
            float(temporal_consistency_2d(pred[i]).mean())
            for i in range(n_traj)
        ])
        ssim_vals = np.array([
            float(structural_similarity_2d(pred[i], truth[i]))
            for i in range(n_traj)
        ])

        vals = [rmse_vals, spec_vals, tc_vals, ssim_vals]
        for k_idx, v in enumerate(vals):
            all_means[m_idx, k_idx] = v.mean()
            all_stds[m_idx, k_idx]  = v.std() if n_traj > 1 else 0.0

    fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5))
    x = np.arange(n_methods)
    width = 0.55

    for k_idx, (ax, mname) in enumerate(zip(axes, metric_names)):
        bars = ax.bar(
            x,
            all_means[:, k_idx],
            width,
            yerr=all_stds[:, k_idx],
            capsize=5,
            color=colors,
            alpha=0.85,
        )
        for bar, mean in zip(bars, all_means[:, k_idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + all_stds[:, k_idx].max() * 0.05 + 1e-9,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_title(mname)
        ax.set_ylabel("Value" if k_idx == 0 else "")

    fig.suptitle("Method Comparison  |  256×256  (mean ± std, test trajectories)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "fig5_summary_bars")

    # Return metrics dict for table printing
    return {
        "methods":     methods,
        "metric_names": metric_names,
        "means":       all_means,
        "stds":        all_stds,
    }


# ---------------------------------------------------------------------------
# Figure 6 — Temporal rollout (7 rows × 5 cols)
# ---------------------------------------------------------------------------

def plot_rollout_comparison(ri: dict, ro: dict, truth_all: torch.Tensor,
                            traj: int = 0, n_cols: int = 5) -> None:
    """7 rows × 5 cols at 256×256.

    Rows:
      0 — Ground Truth
      1 — FNO-only (autoregressive)
      2 — |Error| FNO-only
      3 — One-Shot SR
      4 — |Error| One-Shot SR
      5 — Iterative Refinement
      6 — |Error| Iterative Refinement

    Columns: 5 evenly-spaced time steps from t=1 to t=T-1.
    Error colorscale shared across all 3 error rows (derived from FNO-only error).
    """
    T = truth_all.shape[1]
    t_steps = np.linspace(1, T - 1, n_cols, dtype=int)

    truth = truth_all[traj]              # (T, 256, 256)
    fno   = ri["fno_only_256"][traj]     # (T, 256, 256)
    one   = ro["posterior_256"][traj]    # (T, 256, 256)
    itr   = ri["posterior_256"][traj]    # (T, 256, 256)

    # Field colorscale from truth
    vmax = float(np.percentile(np.abs(truth.numpy()), 99.5))
    vmax = max(vmax, 1e-6)

    # Shared error colorscale — derived from FNO-only error (sets the ceiling)
    err_fno_all = (fno - truth).abs()
    emax = float(np.percentile(err_fno_all.numpy(), 99.5))
    emax = max(emax, 1e-6)

    row_labels = [
        "Ground Truth",
        "FNO-only (autoreg.)",
        "|Error| FNO-only",
        "One-Shot SR",
        "|Error| One-Shot SR",
        "Iterative Refinement",
        "|Error| Iterative",
    ]
    n_rows = len(row_labels)

    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.06, wspace=0.04,
        left=0.10, right=0.92, top=0.95, bottom=0.02,
    )

    # Store one image handle per row for colorbars
    last_im_field = None
    last_im_err   = None

    for col_idx, t in enumerate(t_steps):
        err_fno_t  = (fno[t]  - truth[t]).abs().numpy()
        err_one_t  = (one[t]  - truth[t]).abs().numpy()
        err_iter_t = (itr[t]  - truth[t]).abs().numpy()

        row_data = [
            (truth[t].numpy(), CMAP_FIELD, -vmax, vmax),
            (fno[t].numpy(),   CMAP_FIELD, -vmax, vmax),
            (err_fno_t,        CMAP_ERR,    0.0,  emax),
            (one[t].numpy(),   CMAP_FIELD, -vmax, vmax),
            (err_one_t,        CMAP_ERR,    0.0,  emax),
            (itr[t].numpy(),   CMAP_FIELD, -vmax, vmax),
            (err_iter_t,       CMAP_ERR,    0.0,  emax),
        ]

        for row_idx, (field, cmap, vmin, vmx) in enumerate(row_data):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmx,
                           origin="lower", aspect="equal", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"t = {t}", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9, labelpad=4)

            # Track last image for colorbars
            if cmap == CMAP_FIELD:
                last_im_field = im
            else:
                last_im_err = im

    # Colorbars on the right
    cbar_ax1 = fig.add_axes([0.933, 0.50, 0.012, 0.44])
    fig.colorbar(last_im_field, cax=cbar_ax1)
    cbar_ax1.tick_params(labelsize=9)
    cbar_ax1.set_ylabel("Vorticity  ω", fontsize=10)

    cbar_ax2 = fig.add_axes([0.933, 0.03, 0.012, 0.44])
    fig.colorbar(last_im_err, cax=cbar_ax2)
    cbar_ax2.tick_params(labelsize=9)
    cbar_ax2.set_ylabel("|Error|", fontsize=10)

    fig.suptitle(
        f"Temporal Rollout Comparison  |  256×256,  trajectory {traj}",
        fontsize=13, y=0.975,
    )
    _save(fig, "fig6_rollout_comparison")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_table(table: dict) -> None:
    methods      = table["methods"]
    metric_names = table["metric_names"]
    means        = table["means"]

    col_w = 16
    header = f"{'Method':<28}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, method in enumerate(methods):
        row = f"{method.replace(chr(10), ' '):<28}"
        row += "".join(f"{means[i, k]:>{col_w}.4f}" for k in range(len(metric_names)))
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate comparison figures (all methods)")
    p.add_argument("--iterative",  type=str, default="results_2d/inference_results.pt")
    p.add_argument("--oneshot",    type=str, default="results_oneshot/inference_results.pt")
    p.add_argument("--data_dir",   type=str, default="data_2d")
    p.add_argument("--config",     type=str, default="configs/kraichnan.yaml")
    p.add_argument("--figures",    type=str, default="1,2,3,4,5,6",
                   help="Comma-separated figures to generate (default: all)")
    p.add_argument("--snapshot_t", type=int, default=50,
                   help="Time step for Fig 1 snapshot (default: 50)")
    p.add_argument("--traj",       type=int, default=0,
                   help="Trajectory index for Fig 1 (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    figs = {int(x) for x in args.figures.split(",")}

    print(f"Loading iterative results from {args.iterative} ...")
    ri = torch.load(args.iterative, map_location="cpu", weights_only=True)

    print(f"Loading one-shot results from {args.oneshot} ...")
    ro = torch.load(args.oneshot, map_location="cpu", weights_only=True)

    # Load ground truth from test split (source of truth independent of both result files)
    print(f"Loading test ground truth from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )
    T_os = ro["posterior_256"].shape[1]
    T_it = ri["posterior_256"].shape[1]
    T = min(T_os, T_it)
    n_traj = min(ri["truth_256"].shape[0], ro["posterior_256"].shape[0],
                 test_data["w_256"].shape[0])
    truth_all = test_data["w_256"][:n_traj, :T].float()   # (n_traj, T, 256, 256)

    # Trim all result tensors to the common T
    ri = {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
          for k, v in ri.items()}
    ro = {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
          for k, v in ro.items()}

    k_f = float(cfg.pde.forcing_band_center)

    table = None

    if 1 in figs:
        print("\nFigure 1: Method comparison snapshot ...")
        plot_method_comparison(ri, ro, truth_all,
                               t=args.snapshot_t, traj=args.traj)

    if 2 in figs:
        print("\nFigure 2: RMSE over time ...")
        plot_rmse_comparison(ri, ro, truth_all)

    if 3 in figs:
        print("\nFigure 3: Energy spectrum ...")
        plot_spectrum_comparison(ri, ro, truth_all, k_forcing=k_f)

    if 4 in figs:
        print("\nFigure 4: Temporal consistency ...")
        plot_temporal_consistency(ri, ro, truth_all)

    if 5 in figs:
        print("\nFigure 5: Summary bar chart ...")
        table = plot_summary_bars(ri, ro, truth_all)

    if 6 in figs:
        print("\nFigure 6: Temporal rollout comparison ...")
        plot_rollout_comparison(ri, ro, truth_all, traj=args.traj)

    if table is not None:
        print_table(table)
    elif any(f in figs for f in [1, 2, 3, 4]):
        # Compute metrics for table even if fig5 not requested
        print("\nComputing metrics for summary table ...")
        # Build minimal table inline
        methods   = ["Bicubic", "One-Shot SR", "Iterative Refinement"]
        colors    = [C_BIC, C_ONE, C_ITER]
        preds     = [ro["bicubic_256"], ro["posterior_256"], ri["posterior_256"]]
        n_traj_t  = truth_all.shape[0]
        mnames    = ["RMSE", "Spectral RMSE", "Temp. Consist.", "SSIM"]
        means     = np.zeros((3, 4))
        for m_idx, pred in enumerate(preds):
            means[m_idx, 0] = float(rmse_2d(pred, truth_all))
            means[m_idx, 1] = _spectral_rmse(pred, truth_all)
            means[m_idx, 2] = float(
                torch.stack([temporal_consistency_2d(pred[i]).mean()
                             for i in range(n_traj_t)]).mean()
            )
            means[m_idx, 3] = float(structural_similarity_2d(
                pred.reshape(-1, 256, 256), truth_all.reshape(-1, 256, 256)
            ))
        print_table({"methods": methods, "metric_names": mnames, "means": means})

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
