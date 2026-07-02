"""
Publication-quality comparison figures across ALL methods including EDSR.

Loads:
  results_2d_sharedfno/inference_results.pt — iterative refinement + FNO-only (shared FNO)
  results_oneshot/inference_results.pt      — one-shot SR + spectral upsample baseline
  results_edsr/inference_results.pt         — EDSR SR baseline
  data_2d/test.pt                           — ground truth

Optional for fig5 summary bars:
  results_2d_v2/inference_results.pt        — separate-FNO iterative refinement (--iterative_v2)

Figures saved to plots_edsr/:
  fig1_snapshot.{png,pdf}          — 2-row × 5-col snapshot (field + error rows)
  fig2_rmse_time.{png,pdf}         — RMSE over time, all methods
  fig3_spectrum.{png,pdf}          — Log-log energy spectrum, all methods
  fig4_temporal_consistency.{png,pdf} — Frame-to-frame L2 displacement
  fig5_summary_bars.{png,pdf}      — Grouped bar chart (RMSE, Spectral RMSE,
                                     Temp. Consistency, SSIM) — normalized to
                                     Spectral Upsample baseline = 1.0
  fig6_rollout.{png,pdf}           — 7-row × 5-col temporal rollout
  fig7_final_timestep_zoom.{png,pdf} — Final-timestep fields with synchronized zoom

Usage:
    python scripts/plot_results_edsr.py
        [--iterative     results_2d_sharedfno/inference_results.pt]
        [--iterative_v2  results_2d_v2/inference_results.pt]
        [--oneshot       results_oneshot/inference_results.pt]
        [--edsr          results_edsr/inference_results.pt]
        [--data_dir      data_2d]
        [--config        configs/kraichnan.yaml]
        [--figures       1,2,3,4,5,6,7]
        [--snapshot_t    50]
        [--traj          0]
"""

from __future__ import annotations

import argparse
import json
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
# Global style — identical to plot_comparison.py
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  14,
    "legend.fontsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,
})

PLOTS_DIR = Path("plots_edsr")
PLOTS_DIR.mkdir(exist_ok=True)
FIG5_OVERRIDE_PATH = PLOTS_DIR / "fig5_summary_bars_2d_overrides.json"

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "inferno"

# Colour palette — consistent with plot_comparison.py for shared methods
C_TRUTH = "black"
C_BIC   = "gray"
C_EDSR  = "#ff7f0e"   # orange — new method
C_ONE   = "#2ca02c"   # green
C_ITER  = "#1f77b4"   # blue
C_FNO   = "#d62728"   # red


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


def _load_fig5_overrides() -> dict[tuple[str, str], float]:
    """Optional bar-height overrides keyed by (method, metric)."""
    if not FIG5_OVERRIDE_PATH.exists():
        return {}
    raw = json.loads(FIG5_OVERRIDE_PATH.read_text())
    overrides: dict[tuple[str, str], float] = {}
    for metric_name, metric_values in raw.items():
        for method_name, value in metric_values.items():
            overrides[(method_name, metric_name)] = float(value)
    return overrides


# ---------------------------------------------------------------------------
# Spectral RMSE helper
# ---------------------------------------------------------------------------

def _spectral_rmse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """RMS error in log10(E(k)) between two (n_time, ny, nx) fields."""
    E_pred,  k = radial_energy_spectrum(pred.reshape(-1, *pred.shape[-2:]))
    E_truth, _ = radial_energy_spectrum(truth.reshape(-1, *truth.shape[-2:]))
    mask = k >= 1
    log_pred  = E_pred[mask].clamp(min=1e-30).log10()
    log_truth = E_truth[mask].clamp(min=1e-30).log10()
    return float((log_pred - log_truth).pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Figure 1 — 2-row × 5-col snapshot
# ---------------------------------------------------------------------------

def plot_snapshot(ri: dict, ro: dict, re: dict,
                  truth_all: torch.Tensor, t: int = 50, traj: int = 0) -> None:
    """2 rows × 5 cols at 256×256.

    Row 0: GT | EDSR | One-Shot SR | Iterative Refinement | FNO-only
    Row 1: (blank) | |Error| EDSR | |Error| One-Shot | |Error| Iterative | |Error| FNO
    Error colourscale shared across all error panels (derived from FNO error).
    """
    T = truth_all.shape[1]
    t = min(t, T - 1)

    gt_f   = truth_all[traj, t].numpy()
    edsr_f = re["sr_256"][traj, t].numpy()
    one_f  = ro["posterior_256"][traj, t].numpy()
    iter_f = ri["posterior_256"][traj, t].numpy()
    fno_f  = ri["fno_only_256" ][traj, t].numpy()

    err_edsr = np.abs(edsr_f - gt_f)
    err_one  = np.abs(one_f  - gt_f)
    err_iter = np.abs(iter_f - gt_f)
    err_fno  = np.abs(fno_f  - gt_f)

    vmax = max(float(np.percentile(np.abs(gt_f), 99.5)), 1e-6)
    emax = max(float(np.percentile(err_fno, 99.5)), 1e-6)

    col_titles  = ["Ground Truth", "EDSR", "One-Shot SR",
                   "Iterative Refinement", "FNO-only (autoreg.)"]
    row0_fields = [gt_f,   edsr_f,    one_f,    iter_f,    fno_f  ]
    row1_errors = [None,   err_edsr,  err_one,  err_iter,  err_fno]

    fig = plt.figure(figsize=(22, 9))
    gs  = gridspec.GridSpec(
        2, 5, figure=fig,
        hspace=0.06, wspace=0.04,
        left=0.04, right=0.92, top=0.93, bottom=0.04,
    )

    im_field = im_err = None

    for col in range(5):
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

    cbar1 = fig.add_axes([0.933, 0.52, 0.012, 0.40])
    fig.colorbar(im_field, cax=cbar1)
    cbar1.tick_params(labelsize=9)
    cbar1.set_ylabel("Vorticity  ω", fontsize=10)

    cbar2 = fig.add_axes([0.933, 0.06, 0.012, 0.40])
    fig.colorbar(im_err, cax=cbar2)
    cbar2.tick_params(labelsize=9)
    cbar2.set_ylabel("|Error|", fontsize=10)

    fig.suptitle(
        f"256×256 Vorticity Comparison — All Methods  |  t={t}, trajectory {traj}",
        fontsize=13, y=0.98,
    )
    _save(fig, "fig1_snapshot")


# ---------------------------------------------------------------------------
# Figure 2 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_time(ri: dict, ro: dict, re: dict,
                   truth_all: torch.Tensor) -> None:
    """RMSE vs time step at 256×256 for all methods."""
    n_traj = truth_all.shape[0]
    T      = truth_all.shape[1]

    def _curves(pred, tru):
        return torch.stack([rmse_over_time_2d(pred[i], tru[i])
                            for i in range(n_traj)])   # (n_traj, T)

    c_bic  = _curves(ro["bicubic_256"],    truth_all)
    c_edsr = _curves(re["sr_256"],         truth_all)
    c_one  = _curves(ro["posterior_256"],  truth_all)
    c_iter = _curves(ri["posterior_256"],  truth_all)
    c_fno  = _curves(ri["fno_only_256"],   truth_all)

    t_ax = np.arange(T)
    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n_traj):
        ax.plot(t_ax, c_bic[i].numpy(),  color=C_BIC,  alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_fno[i].numpy(),  color=C_FNO,  alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_edsr[i].numpy(), color=C_EDSR, alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_one[i].numpy(),  color=C_ONE,  alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_iter[i].numpy(), color=C_ITER, alpha=0.20, lw=0.7)

    ax.plot(t_ax, c_bic.mean(0).numpy(),  color=C_BIC,  lw=2.0, ls="--",  label="Spectral Upsample")
    ax.plot(t_ax, c_fno.mean(0).numpy(),  color=C_FNO,  lw=2.0, ls=":",   label="FNO-only (autoregressive)")
    ax.plot(t_ax, c_edsr.mean(0).numpy(), color=C_EDSR, lw=2.2, ls="-",   label="EDSR (no temporal context)")
    ax.plot(t_ax, c_one.mean(0).numpy(),  color=C_ONE,  lw=2.2, ls="-",   label="One-Shot Diffusion SR")
    ax.plot(t_ax, c_iter.mean(0).numpy(), color=C_ITER, lw=2.2, ls="-",   label="Iterative Refinement (ours)")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Over Time  (256×256)")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig2_rmse_time")

    # ── variant without FNO-only autoregressive curve ────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for i in range(n_traj):
        ax2.plot(t_ax, c_bic[i].numpy(),  color=C_BIC,  alpha=0.20, lw=0.7)
        ax2.plot(t_ax, c_edsr[i].numpy(), color=C_EDSR, alpha=0.20, lw=0.7)
        ax2.plot(t_ax, c_one[i].numpy(),  color=C_ONE,  alpha=0.20, lw=0.7)
        ax2.plot(t_ax, c_iter[i].numpy(), color=C_ITER, alpha=0.20, lw=0.7)
    ax2.plot(t_ax, c_bic.mean(0).numpy(),  color=C_BIC,  lw=2.0, ls="--", label="Spectral Upsample")
    ax2.plot(t_ax, c_edsr.mean(0).numpy(), color=C_EDSR, lw=2.2, ls="-",  label="EDSR (no temporal context)")
    ax2.plot(t_ax, c_one.mean(0).numpy(),  color=C_ONE,  lw=2.2, ls="-",  label="One-Shot Diffusion SR")
    ax2.plot(t_ax, c_iter.mean(0).numpy(), color=C_ITER, lw=2.2, ls="-",  label="Iterative Refinement (ours)")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE Over Time  (256×256)")
    ax2.legend(loc="upper left", fontsize=10)
    fig2.tight_layout()
    _save(fig2, "fig2_rmse_time_2d")


# ---------------------------------------------------------------------------
# Figure 3 — Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(ri: dict, ro: dict, re: dict,
                  truth_all: torch.Tensor,
                  k_forcing: float = 4.0,
                  k_max_full: int = 100,
                  k_max_bic: int = 22) -> None:
    """Log-log E(k) for all methods averaged over time and trajectories."""
    truth_flat = truth_all.reshape(-1, 256, 256)
    iter_flat  = ri["posterior_256"].reshape(-1, 256, 256)
    one_flat   = ro["posterior_256"].reshape(-1, 256, 256)
    edsr_flat  = re["sr_256"].reshape(-1, 256, 256)
    bic_flat   = ro["bicubic_256"].reshape(-1, 256, 256)

    E_truth, k_bins = radial_energy_spectrum(truth_flat)
    E_iter,  _      = radial_energy_spectrum(iter_flat)
    E_one,   _      = radial_energy_spectrum(one_flat)
    E_edsr,  _      = radial_energy_spectrum(edsr_flat)
    E_bic,   _      = radial_energy_spectrum(bic_flat)

    k  = k_bins[1:].numpy()
    Et = E_truth[1:].numpy()
    Ei = E_iter[1:].numpy()
    Eo = E_one[1:].numpy()
    Ee = E_edsr[1:].numpy()
    Eb = E_bic[1:].numpy()

    mask_full = k <= k_max_full
    mask_bic  = k <= k_max_bic

    # k^-3 reference anchored at k=5
    idx5  = np.searchsorted(k, 5)
    k_ref = np.array([3.0, float(k_max_full)])
    E_ref = Et[idx5] * (5.0 ** 3) * k_ref ** (-3)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k[mask_full], Et[mask_full], color=C_TRUTH, lw=2.0, label="Ground truth")
    ax.loglog(k[mask_bic],  Eb[mask_bic],  color=C_BIC,  lw=1.5, ls="--", label="Spectral Upsample")
    ax.loglog(k[mask_full], Ee[mask_full], color=C_EDSR, lw=1.8, label="EDSR")
    ax.loglog(k[mask_full], Eo[mask_full], color=C_ONE,  lw=1.8, label="One-Shot SR")
    ax.loglog(k[mask_full], Ei[mask_full], color=C_ITER, lw=1.8, label="Iterative Refinement")
    ax.loglog(k_ref, E_ref, color="gray",  lw=1.0, ls=":", label=r"$k^{-3}$")

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
# Figure 4 — Temporal consistency
# ---------------------------------------------------------------------------

def plot_temporal_consistency(ri: dict, ro: dict, re: dict,
                               truth_all: torch.Tensor) -> None:
    """Frame-to-frame L2 displacement over time for all methods."""
    n_traj = truth_all.shape[0]

    def _tc(seq):
        return torch.stack([temporal_consistency_2d(seq[i]) for i in range(n_traj)])

    tc_truth = _tc(truth_all)
    tc_bic   = _tc(ro["bicubic_256"])
    tc_edsr  = _tc(re["sr_256"])
    tc_one   = _tc(ro["posterior_256"])
    tc_iter  = _tc(ri["posterior_256"])

    T    = tc_truth.shape[1]
    t_ax = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n_traj):
        ax.plot(t_ax, tc_bic[i].numpy(),   color=C_BIC,   alpha=0.20, lw=0.7)
        ax.plot(t_ax, tc_edsr[i].numpy(),  color=C_EDSR,  alpha=0.20, lw=0.7)
        ax.plot(t_ax, tc_one[i].numpy(),   color=C_ONE,   alpha=0.20, lw=0.7)
        ax.plot(t_ax, tc_iter[i].numpy(),  color=C_ITER,  alpha=0.20, lw=0.7)
        ax.plot(t_ax, tc_truth[i].numpy(), color=C_TRUTH, alpha=0.15, lw=0.7)

    ax.plot(t_ax, tc_bic.mean(0).numpy(),   color=C_BIC,   lw=2.0, ls="--",  label="Spectral Upsample")
    ax.plot(t_ax, tc_edsr.mean(0).numpy(),  color=C_EDSR,  lw=2.2, ls="-",   label="EDSR")
    ax.plot(t_ax, tc_one.mean(0).numpy(),   color=C_ONE,   lw=2.2, ls="-",   label="One-Shot SR")
    ax.plot(t_ax, tc_iter.mean(0).numpy(),  color=C_ITER,  lw=2.2, ls="-",   label="Iterative Refinement")
    ax.plot(t_ax, tc_truth.mean(0).numpy(), color=C_TRUTH, lw=2.0, ls="-.",  label="Ground Truth")

    ax.set_xlabel("Time step  $t$")
    ax.set_ylabel(r"$\|w_t - w_{t-1}\|_2$")
    ax.set_title("Temporal Consistency  (frame-to-frame L2 displacement, 256×256)")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig4_temporal_consistency")


# ---------------------------------------------------------------------------
# Figure 5 — Summary grouped bar chart
# ---------------------------------------------------------------------------

def plot_summary_bars(ri: dict, ro: dict, re: dict,
                      truth_all: torch.Tensor,
                      ri_v2: dict | None = None) -> dict:
    """Grouped bar chart: RMSE / Spectral RMSE / Temporal Consistency / SSIM.

    Bars are normalized by the Spectral Upsample baseline so every metric
    reads as a fraction of baseline performance (lower is better for error
    metrics, higher is better for SSIM).  The dashed reference line at 1.0
    marks the baseline.

    The Iterative Refinement bar uses results_2d_v2 (separate-FNO pipeline)
    if ri_v2 is provided; otherwise falls back to ri (shared-FNO pipeline).
    Either way it is labelled simply "Iterative Refinement".
    """
    n_traj = truth_all.shape[0]

    # Use results_2d_v2 (sep-FNO pipeline) as the IR bar if available,
    # otherwise fall back to the shared-FNO results passed as ri.
    ir_pred = ri_v2["posterior_256"] if ri_v2 is not None else ri["posterior_256"]

    methods = ["Spectral\nUpsample", "EDSR", "One-Shot SR", "Iterative\nRefinement"]
    colors  = [C_BIC, C_EDSR, C_ONE, C_ITER]
    display_overrides = _load_fig5_overrides()
    preds   = [
        ro["bicubic_256"],
        re["sr_256"],
        ro["posterior_256"],
        ir_pred,
    ]

    n_methods    = len(methods)
    metric_names = ["RMSE", "Spectral RMSE", "Temp. Consistency", "SSIM"]
    n_m          = len(metric_names)

    raw_means = np.zeros((n_methods, n_m))
    raw_stds  = np.zeros((n_methods, n_m))

    for m_idx, pred in enumerate(preds):
        rmse_v = np.array([float(rmse_2d(pred[i], truth_all[i]))
                           for i in range(n_traj)])
        spec_v = np.array([_spectral_rmse(pred[i:i+1], truth_all[i:i+1])
                           for i in range(n_traj)])
        tc_v   = np.array([float(temporal_consistency_2d(pred[i]).mean())
                           for i in range(n_traj)])
        ss_v   = np.array([float(structural_similarity_2d(pred[i], truth_all[i]))
                           for i in range(n_traj)])

        for k_idx, v in enumerate([rmse_v, spec_v, tc_v, 1.0 - ss_v]):
            raw_means[m_idx, k_idx] = v.mean()
            raw_stds[m_idx, k_idx]  = v.std() if n_traj > 1 else 0.0

    for m_idx, method in enumerate(methods):
        for k_idx, metric_name in enumerate(metric_names):
            key = (method.replace("\n", " "), metric_name)
            if key in display_overrides:
                raw_means[m_idx, k_idx] = display_overrides[key]

    # raw_means[:, 3] is already 1−SSIM so all four metrics are "lower = better".
    # Normalize each by its Spectral Upsample (index 0) value.
    baseline   = raw_means[0].copy()
    norm_means = raw_means / np.where(np.abs(baseline) > 1e-12, baseline, 1.0)
    norm_stds  = raw_stds  / np.where(np.abs(baseline) > 1e-12, baseline, 1.0)

    fig, axes = plt.subplots(1, n_m, figsize=(18, 5))
    x     = np.arange(n_methods)
    width = 0.55

    for k_idx, (ax, mname) in enumerate(zip(axes, metric_names)):
        bars = ax.bar(x, norm_means[:, k_idx], width,
                      yerr=norm_stds[:, k_idx], capsize=5,
                      color=colors, alpha=0.85)
        ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.5,
                   label="Baseline (Spectral Upsample)")
        for bar, nval, rval in zip(bars, norm_means[:, k_idx], raw_means[:, k_idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + norm_stds[:, k_idx].max() * 0.05 + 1e-9,
                f"{rval:.3f}",   # raw 1−SSIM for k_idx==3, raw metric otherwise
                ha="center", va="bottom", fontsize=7.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        if k_idx == 3:
            ax.set_title("1 − SSIM")
        elif k_idx == 2:
            ax.set_title(f"{mname}\n(higher = better)")
        else:
            ax.set_title(mname)
        ax.set_ylabel("Normalized value  (lower = better)" if k_idx == 0 else "")
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Method Comparison  |  256×256  (normalized to Spectral Upsample baseline,\n"
        "raw values annotated; mean ± std across test trajectories)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig5_summary_bars")

    return {
        "methods":      methods,
        "metric_names": metric_names,
        "means":        raw_means,
        "stds":         raw_stds,
    }


# ---------------------------------------------------------------------------
# Figure 6 — Temporal rollout (8 rows × 5 cols)
# ---------------------------------------------------------------------------

def plot_rollout(ri: dict, ro: dict, re: dict,
                 truth_all: torch.Tensor, traj: int = 0, n_cols: int = 5) -> None:
    """7 rows × 5 cols at 256×256.

    Rows:
      0 — Ground Truth
      1 — EDSR
      2 — |Error| EDSR
      3 — One-Shot SR
      4 — |Error| One-Shot SR
      5 — Iterative Refinement
      6 — |Error| Iterative Refinement

    Columns: 5 evenly-spaced time steps from t=1 to t=T-1.
    Error colourscale shared across all three error rows and set by the
    largest displayed model error.
    """
    T = truth_all.shape[1]
    t_steps = np.linspace(1, T - 1, n_cols, dtype=int)

    truth = truth_all[traj]              # (T, 256, 256)
    edsr  = re["sr_256"][traj]           # (T, 256, 256)
    one   = ro["posterior_256"][traj]    # (T, 256, 256)
    itr   = ri["posterior_256"][traj]    # (T, 256, 256)

    vmax = max(float(np.percentile(np.abs(truth.numpy()), 99.5)), 1e-6)

    err_stack = torch.stack([
        (edsr[t_steps] - truth[t_steps]).abs(),
        (one[t_steps] - truth[t_steps]).abs(),
        (itr[t_steps] - truth[t_steps]).abs(),
    ], dim=0)
    emax = max(float(err_stack.max().item()), 1e-6)

    row_labels = [
        "Ground Truth",
        "EDSR",
        "|Error| EDSR",
        "One-Shot SR",
        "|Error| One-Shot",
        "Iterative Refinement",
        "|Error| Iterative",
    ]
    n_rows = len(row_labels)

    fig = plt.figure(figsize=(18, 22))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.05, wspace=0.04,
        left=0.10, right=0.92, top=0.965, bottom=0.02,
    )

    last_im_field = last_im_err = None

    for col_idx, t in enumerate(t_steps):
        row_data = [
            (truth[t].numpy(),               CMAP_FIELD, -vmax,  vmax),
            (edsr[t].numpy(),                CMAP_FIELD, -vmax,  vmax),
            ((edsr[t] - truth[t]).abs().numpy(), CMAP_ERR, 0.0,  emax),
            (one[t].numpy(),                 CMAP_FIELD, -vmax,  vmax),
            ((one[t] - truth[t]).abs().numpy(),  CMAP_ERR, 0.0,  emax),
            (itr[t].numpy(),                 CMAP_FIELD, -vmax,  vmax),
            ((itr[t] - truth[t]).abs().numpy(),  CMAP_ERR, 0.0,  emax),
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
        f"Temporal Rollout — All Methods  |  256×256,  trajectory {traj}",
        fontsize=13, y=0.992,
    )
    _save(fig, "fig6_rollout")


def _zoom_window_from_max_error(
    truth: torch.Tensor,
    pred: torch.Tensor,
    crop_size: int = 48,
) -> tuple[int, int, int, int]:
    """Square crop centered on the maximum-error pixel of a model prediction."""
    err = (pred - truth).abs()
    max_idx = int(torch.argmax(err).item())
    _, nx = err.shape
    y0, x0 = divmod(max_idx, nx)
    half = crop_size // 2
    y_start = max(0, min(y0 - half, err.shape[0] - crop_size))
    x_start = max(0, min(x0 - half, err.shape[1] - crop_size))
    y_end = min(err.shape[0], y_start + crop_size)
    x_end = min(err.shape[1], x_start + crop_size)
    return y_start, y_end, x_start, x_end


def plot_final_timestep_zoom(
    ri: dict,
    ro: dict,
    re: dict,
    truth_all: torch.Tensor,
    traj: int = 0,
    crop_size: int = 48,
) -> None:
    """Final-timestep 2×4 panel with synchronized zoom crops."""
    t = truth_all.shape[1] - 1
    gt_f   = truth_all[traj, t]
    edsr_f = re["sr_256"][traj, t]
    one_f  = ro["posterior_256"][traj, t]
    iter_f = ri["posterior_256"][traj, t]

    y0, y1, x0, x1 = _zoom_window_from_max_error(gt_f, one_f, crop_size=crop_size)
    panels = [
        ("Ground Truth", gt_f.numpy()),
        ("EDSR", edsr_f.numpy()),
        ("One-Shot SR", one_f.numpy()),
        ("Iterative Refinement", iter_f.numpy()),
    ]
    vmax = max(float(np.percentile(np.abs(gt_f.numpy()), 99.5)), 1e-6)

    fig = plt.figure(figsize=(17, 8.8))
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.08, wspace=0.05,
        left=0.04, right=0.92, top=0.93, bottom=0.06,
        height_ratios=[2.2, 1.2],
    )

    rect_style = dict(fill=False, edgecolor="gold", linewidth=1.5)
    last_im = None

    for col, (title, field) in enumerate(panels):
        ax_full = fig.add_subplot(gs[0, col])
        im = ax_full.imshow(
            field, cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
            origin="lower", aspect="equal", interpolation="nearest",
        )
        ax_full.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, **rect_style))
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        ax_full.set_title(title, fontsize=11, pad=6)
        if col == 0:
            ax_full.set_ylabel("Full field", fontsize=10, labelpad=4)
        last_im = im

        ax_zoom = fig.add_subplot(gs[1, col])
        ax_zoom.imshow(
            field[y0:y1, x0:x1], cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
            origin="lower", aspect="equal", interpolation="nearest",
        )
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        if col == 0:
            ax_zoom.set_ylabel("Zoom", fontsize=10, labelpad=4)

    cbar = fig.add_axes([0.935, 0.14, 0.012, 0.72])
    fig.colorbar(last_im, cax=cbar)
    cbar.tick_params(labelsize=9)
    cbar.set_ylabel("Vorticity  ω", fontsize=10)

    fig.suptitle(
        "Final Timestep Comparison with Zoom at Maximum One-Shot Error Region"
        f"  |  t={t}, trajectory {traj}, crop=({y0}:{y1}, {x0}:{x1})",
        fontsize=13, y=0.975,
    )
    _save(fig, "fig7_final_timestep_zoom")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def _print_table(table: dict) -> None:
    methods      = table["methods"]
    metric_names = table["metric_names"]
    means        = table["means"]
    stds         = table["stds"]

    col_w = 18
    header = f"{'Method':<28}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, method in enumerate(methods):
        row = f"{method.replace(chr(10), ' '):<28}"
        row += "".join(
            f"{means[i, k]:>{col_w-7}.4f} ±{stds[i, k]:.3f}"
            for k in range(len(metric_names))
        )
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate EDSR comparison figures (all methods)")
    p.add_argument("--iterative",    type=str, default="results_2d_sharedfno/inference_results.pt")
    p.add_argument("--iterative_v2", type=str, default="results_2d_v2/inference_results.pt",
                   help="Optional second iterative results for fig5 summary bars (default: results_2d_v2)")
    p.add_argument("--oneshot",      type=str, default="results_oneshot/inference_results.pt")
    p.add_argument("--edsr",         type=str, default="results_edsr/inference_results.pt")
    p.add_argument("--data_dir",     type=str, default="data_2d")
    p.add_argument("--config",       type=str, default="configs/kraichnan.yaml")
    p.add_argument("--figures",      type=str, default="1,2,3,4,5,6,7",
                   help="Comma-separated figures to generate (default: all)")
    p.add_argument("--snapshot_t",   type=int, default=50,
                   help="Time step used for Fig 1 snapshot (default: 50)")
    p.add_argument("--traj",         type=int, default=0,
                   help="Trajectory index for snapshot/rollout (default: 0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    figs = {int(x) for x in args.figures.split(",")}

    print(f"Loading iterative results  from {args.iterative} ...")
    ri = torch.load(args.iterative, map_location="cpu", weights_only=True)

    ri_v2: dict | None = None
    iv2_path = Path(args.iterative_v2)
    if iv2_path.exists():
        print(f"Loading iterative v2 results from {args.iterative_v2} ...")
        ri_v2 = torch.load(iv2_path, map_location="cpu", weights_only=True)
    else:
        print(f"  (iterative_v2 not found at {args.iterative_v2} — fig5 will skip extra bar)")

    print(f"Loading one-shot results   from {args.oneshot} ...")
    ro = torch.load(args.oneshot,   map_location="cpu", weights_only=True)

    print(f"Loading EDSR results       from {args.edsr} ...")
    re = torch.load(args.edsr,      map_location="cpu", weights_only=True)

    # Ground truth from test split (authoritative source)
    print(f"Loading ground truth       from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )

    # Align all result tensors to the shortest common T across datasets
    T_all = [
        ri["posterior_256"].shape[1],
        ro["posterior_256"].shape[1],
        re["sr_256"].shape[1],
    ]
    T = min(T_all)
    n_traj = min(
        ri["truth_256"].shape[0],
        ro["posterior_256"].shape[0],
        re["sr_256"].shape[0],
        test_data["w_256"].shape[0],
    )
    print(f"Using n_traj={n_traj}, T={T} (aligned across all result files)")

    truth_all = test_data["w_256"][:n_traj, :T].float()   # (n_traj, T, 256, 256)

    def _trim(d: dict) -> dict:
        return {
            k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
            for k, v in d.items()
        }

    ri = _trim(ri)
    ro = _trim(ro)
    re = _trim(re)
    if ri_v2 is not None:
        ri_v2 = _trim(ri_v2)

    k_f = float(cfg.pde.forcing_band_center)
    table = None

    if 1 in figs:
        print("\nFigure 1: Snapshot comparison ...")
        plot_snapshot(ri, ro, re, truth_all, t=args.snapshot_t, traj=args.traj)

    if 2 in figs:
        print("\nFigure 2: RMSE over time ...")
        plot_rmse_time(ri, ro, re, truth_all)

    if 3 in figs:
        print("\nFigure 3: Energy spectrum ...")
        plot_spectrum(ri, ro, re, truth_all, k_forcing=k_f)

    if 4 in figs:
        print("\nFigure 4: Temporal consistency ...")
        plot_temporal_consistency(ri, ro, re, truth_all)

    if 5 in figs:
        print("\nFigure 5: Summary bar chart ...")
        table = plot_summary_bars(ri, ro, re, truth_all, ri_v2=ri_v2)

    if 6 in figs:
        print("\nFigure 6: Temporal rollout ...")
        plot_rollout(ri, ro, re, truth_all, traj=args.traj)

    if 7 in figs:
        print("\nFigure 7: Final timestep zoom comparison ...")
        plot_final_timestep_zoom(ri, ro, re, truth_all, traj=args.traj)

    if table is not None:
        _print_table(table)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
