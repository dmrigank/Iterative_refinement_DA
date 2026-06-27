"""
Publication-quality 1D comparison figures across all methods.

Loads:
  results/inference_results.pt            — iterative refinement (fno_only_512, posterior_512)
  results_oneshot_1d/inference_results.pt — one-shot SR + bicubic baseline
  data/test.pt                            — ground truth

All figures saved to plots_oneshot_1d/:
  fig1_snapshot_comparison.{png,pdf}  — 2-row × 4-col grid (curves + error rows)
  fig2_rmse_comparison.{png,pdf}      — RMSE over time, all methods
  fig3_spectrum_comparison.{png,pdf}  — Log-log energy spectrum
  fig4_hovmoller_comparison.{png,pdf} — Hovmöller (x vs t), 4 panels
  fig5_summary_bars.{png,pdf}         — Grouped bar chart (4 metric groups)
  fig6_spectral_rmse.{png,pdf}        — Per-mode spectral RMSE (one-shot vs iterative)

FNO-only blowup check:
  If FNO RMSE > 5× bicubic RMSE, or any NaN/Inf detected, FNO is excluded from
  all plots with a printed warning.

Usage:
    python scripts/plot_comparison_1d.py
        [--iterative  results/inference_results.pt]
        [--oneshot    results_oneshot_1d/inference_results.pt]
        [--data_dir   data]
        [--config     configs/oneshot_sr_1d.yaml]
        [--figures    1,2,3,4,5,6]
        [--n_steps    300]
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

from src.evaluation.metrics import (
    energy_spectrum,
    rmse_over_time,
    temporal_consistency,
    spectral_rmse,
)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Method colors
C_BIC  = "#888888"   # gray   — bicubic / spectral upsample
C_FNO  = "#d62728"   # red    — FNO-only autoregressive
C_ONE  = "#2ca02c"   # green  — one-shot SR
C_ITER = "#1f77b4"   # blue   — iterative refinement
C_GT   = "black"     # black  — ground truth

PLOTS_DIR = Path("plots_oneshot_1d")


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOTS_DIR}/{name}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# FNO blowup detection
# ---------------------------------------------------------------------------

def _fno_blown_up_with_bic(ri: dict, ro: dict, truth_all: torch.Tensor) -> bool:
    """Full blowup check including comparison against bicubic RMSE."""
    if "fno_only_512" not in ri:
        return False
    fno = ri["fno_only_512"]
    truth = truth_all[:fno.shape[0], :fno.shape[1]]
    if not torch.isfinite(fno).all():
        print("  [FNO blowup] NaN/Inf detected in fno_only_512 — skipping FNO from plots.")
        return True
    fno_rmse = float((fno - truth).pow(2).mean().sqrt())
    bic = ro["bicubic_512"][:fno.shape[0], :fno.shape[1]]
    bic_rmse = float((bic - truth).pow(2).mean().sqrt())
    if fno_rmse > 5.0 * bic_rmse:
        print(
            f"  [FNO blowup] FNO RMSE={fno_rmse:.4f} > 5× bicubic RMSE={bic_rmse:.4f} "
            "— skipping FNO from plots."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Figure 1 — Multi-timestep snapshot comparison (2-row × 4-col)
# ---------------------------------------------------------------------------

def plot_snapshot_comparison(
    ri: dict,
    ro: dict,
    truth_all: torch.Tensor,
    traj: int = 0,
    fno_ok: bool = True,
) -> None:
    """2-row × 4-col grid.

    Columns: t = 0, T//4, T//2, T-1
    Row 0:   overlaid curves (GT, Bicubic, One-Shot SR, Iterative, [FNO])
    Row 1:   pointwise L2 |error| for each method vs GT (separate sub-panel per method)
    """
    T = truth_all.shape[1]
    N = truth_all.shape[-1]    # 512
    t_indices = [0, T // 4, T // 2, T - 1]
    x512 = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Each method gets 2 rows: field curve (with GT overlay) + |error|.
    # Row pairs: (Bicubic, One-Shot SR, Iterative Refinement, [FNO-only if ok])
    methods = [
        ("Bicubic",              ro["bicubic_512"][traj],    C_BIC,  "--"),
        ("One-Shot SR",          ro["posterior_512"][traj],  C_ONE,  "-"),
        ("Iterative Refinement", ri["posterior_512"][traj],  C_ITER, "-"),
    ]
    if fno_ok and "fno_only_512" in ri:
        methods.append(("FNO-only (autoreg.)", ri["fno_only_512"][traj], C_FNO, ":"))

    n_methods = len(methods)
    n_rows    = n_methods * 2   # field row + error row per method
    n_cols    = 4

    # Height ratios: field rows slightly taller than error rows
    height_ratios = []
    for _ in range(n_methods):
        height_ratios += [1.6, 1.0]

    fig = plt.figure(figsize=(n_cols * 4.2, n_rows * 1.8))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        height_ratios=height_ratios,
        hspace=0.15, wspace=0.10,
        left=0.07, right=0.97, top=0.94, bottom=0.04,
    )

    for col_idx, t in enumerate(t_indices):
        gt = truth_all[traj, t].numpy()

        for m_idx, (label, pred_traj, color, ls) in enumerate(methods):
            u   = pred_traj[t].numpy()
            err = np.abs(u - gt)
            row_field = m_idx * 2
            row_err   = m_idx * 2 + 1

            # --- Field row: method curve + GT overlay ---
            ax_f = fig.add_subplot(gs[row_field, col_idx])
            ax_f.plot(x512, gt, color=C_GT, lw=1.5, alpha=0.6, zorder=4)
            ax_f.plot(x512, u,  color=color, lw=1.6, ls=ls, zorder=5)
            ax_f.set_xlim(0, 2 * np.pi)
            ax_f.set_xticks([])
            if col_idx == 0:
                ax_f.set_ylabel(label, fontsize=9, color=color, labelpad=3)
            else:
                ax_f.set_yticklabels([])
            # Column title only on the very top row
            if m_idx == 0:
                ax_f.set_title(f"t = {t}", fontsize=11)

            # --- Error row ---
            ax_e = fig.add_subplot(gs[row_err, col_idx])
            ax_e.plot(x512, err, color=color, lw=1.3)
            ax_e.fill_between(x512, 0, err, color=color, alpha=0.20)
            ax_e.set_xlim(0, 2 * np.pi)
            ax_e.set_xticks([0, np.pi, 2 * np.pi])
            # Only label x-axis on the very bottom row
            if m_idx == n_methods - 1:
                ax_e.set_xticklabels(["0", "π", "2π"], fontsize=8)
                ax_e.set_xlabel("x", fontsize=9)
            else:
                ax_e.set_xticklabels([])
            if col_idx == 0:
                ax_e.set_ylabel("|err|", fontsize=8, color=color, labelpad=3)
            else:
                ax_e.set_yticklabels([])

    # Thin legend strip: gray line = GT
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], color=C_GT, lw=1.5, alpha=0.6, label="GT (overlay)")]
    for label, _, color, ls in methods:
        legend_handles.append(Line2D([0], [0], color=color, lw=1.6, ls=ls, label=label))
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=len(legend_handles), fontsize=9,
               bbox_to_anchor=(0.5, 0.975), framealpha=0.85)

    fig.suptitle(
        f"Snapshot comparison  |  512-pt  |  traj {traj}  "
        f"[columns: t = 0, T/4, T/2, T−1]",
        fontsize=12, y=0.998,
    )
    _save(fig, "fig1_snapshot_comparison")


# ---------------------------------------------------------------------------
# Figure 2 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_comparison(
    ri: dict,
    ro: dict,
    truth_all: torch.Tensor,
    fno_ok: bool = True,
) -> None:
    """RMSE vs time step at 512-pt, mean ± 1 std across trajectories."""
    n_traj = truth_all.shape[0]

    def _per_traj_rmse_time(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
        # pred, truth: (n_traj, T, 512)
        return torch.stack([
            rmse_over_time(pred[i], truth[i]) for i in range(n_traj)
        ]).numpy()   # (n_traj, T)

    bic_rt  = _per_traj_rmse_time(ro["bicubic_512"],   truth_all)
    one_rt  = _per_traj_rmse_time(ro["posterior_512"], truth_all)
    iter_rt = _per_traj_rmse_time(ri["posterior_512"], truth_all)
    T       = bic_rt.shape[1]
    t_axis  = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot_curve(data, color, label, ls="-"):
        mu  = data.mean(axis=0)
        sig = data.std(axis=0)
        ax.plot(t_axis, mu, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(t_axis, mu - sig, mu + sig, color=color, alpha=0.15)

    _plot_curve(bic_rt,  C_BIC,  "Bicubic",               ls="--")

    if fno_ok and "fno_only_512" in ri:
        fno_rt = _per_traj_rmse_time(ri["fno_only_512"], truth_all)
        _plot_curve(fno_rt, C_FNO, "FNO-only (autoreg.)", ls=":")

    _plot_curve(one_rt,  C_ONE,  "One-Shot SR")
    _plot_curve(iter_rt, C_ITER, "Iterative Refinement")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE  (512-pt)")
    ax.set_title("RMSE over time  |  mean ± 1 std across test trajectories")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig2_rmse_comparison")


# ---------------------------------------------------------------------------
# Figure 3 — Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum_comparison(
    ri: dict,
    ro: dict,
    truth_all: torch.Tensor,
    traj: int = 0,
    k_max_plot: int = 192,
    k_max_bicubic: int = 32,
) -> None:
    """Per-trajectory energy spectra at the same four timesteps used in Fig. 1.

    Each panel shows the plain spectrum E(k) on log-log axes, following the
    same visual language as the main paper spectrum figure. Bicubic is
    intentionally truncated to the trustworthy low-k range because its
    high-k tail is interpolation artifact rather than recovered physics.
    """
    T = truth_all.shape[1]
    t_indices = [0, T // 4, T // 2, T - 1]
    k_full = np.arange(1, truth_all.shape[-1] // 2 + 1)   # 1..256
    k_obs = np.arange(1, ro["obs_64"].shape[-1] // 2 + 1)  # 1..32
    full_mask = k_full <= k_max_plot
    bic_mask = k_full <= min(k_max_plot, k_max_bicubic)

    spectra: dict[tuple[str, int], np.ndarray] = {}
    y_candidates = []
    for t in t_indices:
        E_gt = energy_spectrum(truth_all[traj, t]).numpy()[1:]
        E_obs = energy_spectrum(ro["obs_64"][traj, t]).numpy()[1:]
        E_bic = energy_spectrum(ro["bicubic_512"][traj, t]).numpy()[1:]
        E_one = energy_spectrum(ro["posterior_512"][traj, t]).numpy()[1:]
        E_iter = energy_spectrum(ri["posterior_512"][traj, t]).numpy()[1:]

        spectra[("gt", t)] = E_gt
        spectra[("obs", t)] = E_obs
        spectra[("bic", t)] = E_bic
        spectra[("one", t)] = E_one
        spectra[("iter", t)] = E_iter

        y_candidates.extend([
            E_gt[full_mask],
            E_obs,
            E_bic[bic_mask],
            E_one[full_mask],
            E_iter[full_mask],
        ])

    y_all = np.concatenate(y_candidates)
    y_positive = y_all[y_all > 0]
    y_min = float(y_positive.min()) * 0.7
    y_max = float(y_all.max()) * 1.2

    ref_k = np.array([3.0, float(k_full[full_mask].max())])
    ref_anchor = 5

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, t in zip(axes, t_indices):
        E_gt = spectra[("gt", t)]
        E_obs = spectra[("obs", t)]
        E_bic = spectra[("bic", t)]
        E_one = spectra[("one", t)]
        E_iter = spectra[("iter", t)]

        ref_amp = E_gt[ref_anchor - 1] * (float(ref_anchor) ** 2)
        ref_y = ref_amp * ref_k ** (-2.0)

        ax.loglog(k_full[full_mask], E_gt[full_mask], color=C_GT, lw=2.3, ls="-", label="Ground truth")
        ax.loglog(k_obs, E_obs, color="dimgray", lw=1.8, ls=":", label="LR obs (64)")
        ax.loglog(k_full[bic_mask], E_bic[bic_mask], color=C_BIC, lw=2.0, ls="-.", label="Bicubic")
        ax.loglog(k_full[full_mask], E_one[full_mask], color=C_ONE, lw=2.0, ls="--", label="One-Shot SR")
        ax.loglog(
            k_full[full_mask],
            E_iter[full_mask],
            color=C_ITER,
            lw=2.0,
            ls=(0, (7, 2.2)),
            label="Iterative Refinement",
        )
        ax.loglog(ref_k, ref_y, color="gray", lw=1.0, ls=(0, (1, 2.2)), label=r"$k^{-2}$")

        ax.axvline(32, color="gray", ls=":", lw=0.9, alpha=0.65)
        ax.text(
            0.97,
            0.95,
            f"t = {t}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82),
        )
        ax.set_xlim(1, k_max_plot)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel(r"$E(k) = |\hat{u}_k|^2$")
    axes[2].set_ylabel(r"$E(k) = |\hat{u}_k|^2$")
    axes[2].set_xlabel("Wavenumber $k$")
    axes[3].set_xlabel("Wavenumber $k$")

    axes[1].text(
        0.05,
        0.11,
        "Bicubic shown only through k = 32\n(to suppress interpolation-driven high-k tail)",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="dimgray",
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none", alpha=0.84),
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=C_GT, lw=2.3, ls="-", label="Ground truth"),
        Line2D([0], [0], color="dimgray", lw=1.8, ls=":", label="LR obs (64)"),
        Line2D([0], [0], color=C_BIC, lw=2.0, ls="-.", label="Bicubic"),
        Line2D([0], [0], color=C_ONE, lw=2.0, ls="--", label="One-Shot SR"),
        Line2D([0], [0], color=C_ITER, lw=2.0, ls=(0, (7, 2.2)), label="Iterative Refinement"),
        Line2D([0], [0], color="gray", lw=1.0, ls=(0, (1, 2.2)), label=r"$k^{-2}$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.965),
        framealpha=0.9,
    )
    fig.suptitle(
        f"Energy spectrum comparison  |  traj {traj}  [t = 0, T/4, T/2, T-1]",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0.04, 0.05, 0.99, 0.90))
    _save(fig, "fig3_spectrum_comparison")


# ---------------------------------------------------------------------------
# Figure 4 — Hovmöller comparison
# ---------------------------------------------------------------------------

def plot_hovmoller_comparison(
    ri: dict,
    ro: dict,
    truth_all: torch.Tensor,
    traj: int = 0,
    n_show: int = 100,
) -> None:
    """4 Hovmöller panels: GT, One-Shot SR, Iterative, Bicubic."""
    T    = min(n_show, truth_all.shape[1])
    N    = truth_all.shape[-1]   # 512
    x    = np.linspace(0, 2 * np.pi, N, endpoint=False)

    gt   = truth_all[traj, :T].numpy()
    one  = ro["posterior_512"][traj, :T].numpy()
    itr  = ri["posterior_512"][traj, :T].numpy()
    bic  = ro["bicubic_512"  ][traj, :T].numpy()

    vmax = float(np.percentile(np.abs(gt), 99.5))
    vmax = max(vmax, 1e-6)

    fields = [gt, one, itr, bic]
    titles = ["Ground Truth", "One-Shot SR", "Iterative Refinement", "Bicubic"]
    colors_border = [C_GT, C_ONE, C_ITER, C_BIC]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    ims = []
    for ax, field, title, bc in zip(axes, fields, titles, colors_border):
        im = ax.pcolormesh(
            x, np.arange(T), field,
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
        )
        ims.append(im)
        ax.set_title(title, color=bc if bc != C_GT else "black")
        ax.set_xlabel("x")
        for spine in ax.spines.values():
            spine.set_edgecolor(bc)
            spine.set_linewidth(2)

    axes[0].set_ylabel("Time step")
    fig.subplots_adjust(right=0.88, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.015, 0.75])
    fig.colorbar(ims[0], cax=cbar_ax, label="u")
    fig.suptitle(f"Hovmöller comparison  |  512-pt  |  traj {traj}", fontsize=14)
    _save(fig, "fig4_hovmoller_comparison")


# ---------------------------------------------------------------------------
# Figure 5 — Summary bar chart
# ---------------------------------------------------------------------------

def _spectral_rmse_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean spectral RMSE (log10 amplitude difference), averaged over modes k>=1."""
    n, T, N = pred.shape
    sp_pred  = energy_spectrum(pred.reshape(n * T, N))
    sp_truth = energy_spectrum(truth.reshape(n * T, N))
    # log10 ratio, k >= 1
    ratio = (sp_pred[1:] / sp_truth[1:].clamp(min=1e-30)).log10().abs().mean()
    return float(ratio)


def _temporal_consistency_scalar(pred: torch.Tensor) -> float:
    """Mean frame-to-frame L2 displacement over all trajectories."""
    n_traj = pred.shape[0]
    return float(
        torch.stack([temporal_consistency(pred[i]).mean() for i in range(n_traj)]).mean()
    )


def _ssim_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Structural similarity (mean over all time steps and trajectories).

    Uses a simplified version: 1 - normalised MSE as a proxy,
    since torchmetrics may not be available.
    """
    sigma_pred  = pred.std()
    sigma_truth = truth.std()
    cov         = ((pred - pred.mean()) * (truth - truth.mean())).mean()
    C1, C2      = 0.01 ** 2, 0.03 ** 2
    mu_p, mu_t  = pred.mean(), truth.mean()
    ssim = (
        (2 * mu_p * mu_t + C1) * (2 * cov + C2) /
        ((mu_p**2 + mu_t**2 + C1) * (sigma_pred**2 + sigma_truth**2 + C2))
    )
    return float(ssim)


def plot_summary_bars(
    ri: dict,
    ro: dict,
    truth_all: torch.Tensor,
    fno_ok: bool = True,
) -> dict:
    """4-metric grouped bar chart + table."""
    T = min(ri["posterior_512"].shape[1], truth_all.shape[1], ro["posterior_512"].shape[1])
    truth = truth_all[:, :T].float()

    methods = ["Bicubic", "One-Shot SR", "Iterative\nRefinement"]
    colors  = [C_BIC, C_ONE, C_ITER]
    preds   = [
        ro["bicubic_512"  ][:, :T].float(),
        ro["posterior_512"][:, :T].float(),
        ri["posterior_512"][:, :T].float(),
    ]
    if fno_ok and "fno_only_512" in ri:
        methods.insert(2, "FNO-only\n(autoreg.)")
        colors.insert( 2, C_FNO)
        preds.insert(  2, ri["fno_only_512"][:, :T].float())

    n_methods = len(methods)
    n_traj    = truth.shape[0]

    mnames = ["RMSE", "Spectral RMSE", "Temp. Consist.", "SSIM"]
    means  = np.zeros((n_methods, 4))
    stds   = np.zeros((n_methods, 4))

    for m_idx, pred in enumerate(preds):
        # Per-trajectory RMSE
        rmse_traj = np.array([
            float(pred[i].sub(truth[i]).pow(2).mean().sqrt())
            for i in range(n_traj)
        ])
        means[m_idx, 0] = rmse_traj.mean()
        stds[ m_idx, 0] = rmse_traj.std()
        means[m_idx, 1] = _spectral_rmse_scalar(pred, truth)
        stds[ m_idx, 1] = 0.0   # single scalar
        means[m_idx, 2] = _temporal_consistency_scalar(pred)
        stds[ m_idx, 2] = 0.0
        means[m_idx, 3] = _ssim_scalar(pred, truth)
        stds[ m_idx, 3] = 0.0

    x = np.arange(4)
    width = 0.8 / n_methods
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for m_idx, (method, color, offset) in enumerate(zip(methods, colors, offsets)):
        ax.bar(x + offset, means[m_idx], width,
               yerr=stds[m_idx], label=method.replace("\n", " "),
               color=color, alpha=0.85, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(mnames)
    ax.set_ylabel("Metric value")
    ax.set_title("Summary metrics  |  512-pt  |  mean ± 1 std over test trajectories")
    ax.legend(loc="upper right", framealpha=0.8)
    fig.tight_layout()
    _save(fig, "fig5_summary_bars")

    return {
        "methods":      [m.replace("\n", " ") for m in methods],
        "metric_names": mnames,
        "means":        means,
    }


# ---------------------------------------------------------------------------
# Figure 6 — Per-mode spectral RMSE (one-shot vs iterative)
# ---------------------------------------------------------------------------

def plot_spectral_rmse_comparison(
    ri: dict,
    ro: dict,
    truth_all: torch.Tensor,
    k_max_plot: int = 248,
) -> None:
    """Per-wavenumber spectral RMSE highlighting one-shot vs iterative.

    Plots |log10(E_pred(k) / E_gt(k))| for each method as a function of k.
    A value of 0 means perfect spectral fidelity; larger = worse.

    The gap between one-shot and iterative is shaded to make the difference
    visually prominent.  Summary values are annotated on the plot.
    """
    def _E(x: torch.Tensor) -> torch.Tensor:
        n, T, N = x.shape
        return energy_spectrum(x.reshape(n * T, N))   # (N//2+1,)

    E_gt   = _E(truth_all)
    E_one  = _E(ro["posterior_512"])
    E_iter = _E(ri["posterior_512"])

    eps = 1e-30

    def _srmse_per_k(E_pred: torch.Tensor) -> np.ndarray:
        return (E_pred[1:] / E_gt[1:].clamp(min=eps)).log10().abs().numpy()

    sr_one  = _srmse_per_k(E_one)
    sr_iter = _srmse_per_k(E_iter)

    N  = truth_all.shape[-1]
    k  = np.arange(1, N // 2 + 1)   # k = 1 .. 256
    mask = k <= k_max_plot
    k_plot = k[mask]
    sr_one_plot = sr_one[mask]
    sr_iter_plot = sr_iter[mask]

    # Scalar summaries over the plotted range
    scalar_one  = float(sr_one_plot.mean())
    scalar_iter = float(sr_iter_plot.mean())

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded gap between one-shot and iterative
    ax.fill_between(k_plot, sr_iter_plot, sr_one_plot, where=(sr_one_plot >= sr_iter_plot),
                    color=C_ONE, alpha=0.18, label="Gap (one-shot − iterative)")

    ax.plot(k_plot, sr_one_plot,  color=C_ONE,  lw=2,   label=f"One-Shot SR  (mean={scalar_one:.4f})")
    ax.plot(k_plot, sr_iter_plot, color=C_ITER, lw=2,   label=f"Iterative Refinement  (mean={scalar_iter:.4f})")

    # Annotate scalar values with horizontal arrows
    k_anno = int(0.65 * len(k_plot))   # annotation x position (~65% along plotted range)
    ax.annotate(
        f"{scalar_one:.4f}",
        xy=(k_plot[k_anno], sr_one_plot[k_anno]),
        xytext=(k_plot[k_anno] + 15, sr_one_plot[k_anno] + 0.04),
        color=C_ONE, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ONE, lw=1.2),
    )
    ax.annotate(
        f"{scalar_iter:.4f}",
        xy=(k_plot[k_anno], sr_iter_plot[k_anno]),
        xytext=(k_plot[k_anno] + 15, sr_iter_plot[k_anno] - 0.04),
        color=C_ITER, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ITER, lw=1.2),
    )

    # Reference line at 0
    ax.axhline(0, color="black", lw=0.8, ls=":")

    ax.set_xlabel("Wavenumber  k")
    ax.set_ylabel(r"$|\log_{10}(E_\mathrm{pred}(k) / E_\mathrm{GT}(k))|$")
    ax.set_title(
        "Per-mode spectral RMSE  |  0 = perfect,  larger = worse\n"
        "Shaded region shows improvement of iterative over one-shot"
    )
    ax.set_xlim(1, k_max_plot)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig6_spectral_rmse")


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def print_table(table: dict) -> None:
    methods      = table["methods"]
    metric_names = table["metric_names"]
    means        = table["means"]
    col_w = 16
    header = f"{'Method':<30}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for i, method in enumerate(methods):
        row = f"{method:<30}" + "".join(f"{means[i, k]:>{col_w}.4f}"
                                        for k in range(len(metric_names)))
        print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1D comparison figures")
    p.add_argument("--iterative",  type=str, default="results/inference_results.pt")
    p.add_argument("--oneshot",    type=str, default="results_oneshot_1d/inference_results.pt")
    p.add_argument("--data_dir",   type=str, default="data")
    p.add_argument("--config",     type=str, default="configs/oneshot_sr_1d.yaml")
    p.add_argument("--figures",    type=str, default="1,2,3,4,5,6",
                   help="Comma-separated figures to generate (default: all)")
    p.add_argument("--n_steps",    type=int, default=None,
                   help="Optional number of time steps to plot/use from the loaded evaluations")
    p.add_argument("--traj",       type=int, default=0,
                   help="Trajectory index for Figs 1, 3, 4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    figs = {int(x) for x in args.figures.split(",")}

    print(f"Loading iterative results from {args.iterative} ...")
    ri = torch.load(args.iterative, map_location="cpu", weights_only=True)

    print(f"Loading one-shot results from {args.oneshot} ...")
    ro = torch.load(args.oneshot, map_location="cpu", weights_only=True)

    print(f"Loading test ground truth from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )

    # Align trajectory and time dimensions
    T_os   = ro["posterior_512"].shape[1]
    T_it   = ri["posterior_512"].shape[1]
    T_data = test_data["u_512"].shape[1]
    T      = min(T_os, T_it, T_data)
    if args.n_steps is not None:
        if args.n_steps <= 0:
            raise ValueError(f"--n_steps must be positive, got {args.n_steps}")
        T = min(T, args.n_steps)
    n_traj = min(ri["posterior_512"].shape[0],
                 ro["posterior_512"].shape[0],
                 test_data["u_512"].shape[0])

    print(f"Using {T} time steps for plotting/comparison.")

    truth_all = test_data["u_512"][:n_traj, :T].float()  # (n_traj, T, 512)

    ri = {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
          for k, v in ri.items()}
    ro = {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
          for k, v in ro.items()}

    # FNO blowup check — single call, result shared across all figures
    fno_ok = not _fno_blown_up_with_bic(ri, ro, truth_all)

    table = None

    if 1 in figs:
        print("\nFigure 1: Snapshot comparison (2-row × 4-col) ...")
        plot_snapshot_comparison(ri, ro, truth_all, traj=args.traj, fno_ok=fno_ok)

    if 2 in figs:
        print("\nFigure 2: RMSE over time ...")
        plot_rmse_comparison(ri, ro, truth_all, fno_ok=fno_ok)

    if 3 in figs:
        print("\nFigure 3: Energy spectrum ...")
        plot_spectrum_comparison(ri, ro, truth_all, traj=args.traj)

    if 4 in figs:
        print("\nFigure 4: Hovmöller comparison ...")
        plot_hovmoller_comparison(ri, ro, truth_all, traj=args.traj)

    if 5 in figs:
        print("\nFigure 5: Summary bar chart ...")
        table = plot_summary_bars(ri, ro, truth_all, fno_ok=fno_ok)

    if 6 in figs:
        print("\nFigure 6: Per-mode spectral RMSE ...")
        plot_spectral_rmse_comparison(ri, ro, truth_all)

    if table is not None:
        print_table(table)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
