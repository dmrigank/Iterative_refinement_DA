"""
Publication-quality 1D comparison figures across ALL methods including EDSR-1D.

Loads:
  results/inference_results.pt         — iterative refinement (posterior_512, fno_only_512)
  results_oneshot_1d/inference_results.pt — one-shot SR + bicubic baseline
  results_edsr_1d/inference_results.pt — EDSR-1D SR baseline
  data/test.pt                          — ground truth

All figures saved to plots_edsr_1d/:
  fig1_snapshot.{png,pdf}         — multi-timestep field curves + |error| rows
  fig2_rmse_time.{png,pdf}        — RMSE over time, all methods
  fig3_spectrum.{png,pdf}         — Log-log energy spectrum, all methods
  fig4_hovmoller.{png,pdf}        — Hovmöller (x vs t), 5 panels
  fig5_summary_bars.{png,pdf}     — Grouped bar chart (RMSE, Spectral RMSE,
                                    Temp. Consistency, SSIM)
  fig6_spectral_rmse.{png,pdf}    — Per-mode spectral RMSE, all methods

FNO-only blowup check:
  If FNO RMSE > 5× bicubic RMSE, or any NaN/Inf detected, FNO is excluded
  from all plots with a printed warning.

Usage:
    python scripts/plot_results_edsr_1d.py
        [--iterative  results/inference_results.pt]
        [--oneshot    results_oneshot_1d/inference_results.pt]
        [--edsr       results_edsr_1d/inference_results.pt]
        [--data_dir   data]
        [--figures    1,2,3,4,5,6]
        [--n_steps    N]
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
from matplotlib.lines import Line2D

from src.evaluation.metrics import (
    energy_spectrum,
    rmse_over_time,
    temporal_consistency,
    spectral_rmse,
)

# ---------------------------------------------------------------------------
# Global style — identical to plot_comparison_1d.py
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

# Colours — consistent with plot_comparison_1d.py for shared methods
C_GT   = "black"
C_BIC  = "#888888"   # gray
C_EDSR = "#ff7f0e"   # orange — new
C_ONE  = "#2ca02c"   # green
C_ITER = "#1f77b4"   # blue
C_FNO  = "#d62728"   # red

PLOTS_DIR = Path("plots_edsr_1d")


def _save(fig: plt.Figure, name: str, aliases: tuple[str, ...] = ()) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    stems = (name, *aliases)
    for stem in stems:
        for ext in ("png", "pdf"):
            fig.savefig(PLOTS_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    saved = ", ".join(f"{PLOTS_DIR}/{stem}.{{png,pdf}}" for stem in stems)
    print(f"  Saved {saved}")


# ---------------------------------------------------------------------------
# FNO blowup detection (unchanged from plot_comparison_1d.py)
# ---------------------------------------------------------------------------

def _fno_blown_up(ri: dict, ro: dict, truth_all: torch.Tensor) -> bool:
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
            f"  [FNO blowup] FNO RMSE={fno_rmse:.4f} > 5× bicubic={bic_rmse:.4f} "
            "— skipping FNO."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Shared helper: build ordered method list
# ---------------------------------------------------------------------------

def _method_list(
    ri: dict, ro: dict, re: dict, traj: int, fno_ok: bool
) -> list[tuple[str, torch.Tensor, str, str]]:
    """Return (label, pred_traj, color, linestyle) ordered for display."""
    methods = [
        ("Bicubic",              ro["bicubic_512"  ][traj], C_BIC,  "--"),
        ("EDSR-1D",              re["sr_512"        ][traj], C_EDSR, "-"),
        ("One-Shot SR",          ro["posterior_512" ][traj], C_ONE,  "-"),
        ("Iterative Refinement", ri["posterior_512" ][traj], C_ITER, "-"),
    ]
    if fno_ok and "fno_only_512" in ri:
        methods.append(("FNO-only (autoreg.)", ri["fno_only_512"][traj], C_FNO, ":"))
    return methods


# ---------------------------------------------------------------------------
# Figure 1 — Multi-timestep snapshot comparison
# ---------------------------------------------------------------------------

def plot_snapshot(
    ri: dict, ro: dict, re: dict,
    truth_all: torch.Tensor,
    traj: int = 0,
    fno_ok: bool = True,
) -> None:
    """Field curve + |error| rows for each method, 4 time columns."""
    T = truth_all.shape[1]
    N = truth_all.shape[-1]
    t_indices = [0, T // 4, T // 2, T - 1]
    x512 = np.linspace(0, 2 * np.pi, N, endpoint=False)

    methods = _method_list(ri, ro, re, traj, fno_ok)
    n_methods = len(methods)
    n_rows    = n_methods * 2
    error_limits = {}
    for t in t_indices:
        gt = truth_all[traj, t]
        max_error = max(
            float((pred_traj[t] - gt).abs().max())
            for _, pred_traj, _, _ in methods
        )
        error_limits[t] = max(1.05 * max_error, 1e-6)

    height_ratios = []
    for _ in range(n_methods):
        height_ratios += [1.6, 1.0]

    fig = plt.figure(figsize=(4 * 4.2, n_rows * 1.8))
    gs  = gridspec.GridSpec(
        n_rows, 4, figure=fig,
        height_ratios=height_ratios,
        hspace=0.15, wspace=0.10,
        left=0.07, right=0.97, top=0.94, bottom=0.04,
    )

    for col_idx, t in enumerate(t_indices):
        gt = truth_all[traj, t].numpy()

        for m_idx, (label, pred_traj, color, ls) in enumerate(methods):
            u   = pred_traj[t].numpy()
            err = np.abs(u - gt)
            row_f = m_idx * 2
            row_e = m_idx * 2 + 1

            ax_f = fig.add_subplot(gs[row_f, col_idx])
            ax_f.plot(x512, gt, color=C_GT,  lw=1.5, alpha=0.6, zorder=4)
            ax_f.plot(x512, u,  color=color, lw=1.6, ls=ls,     zorder=5)
            ax_f.set_xlim(0, 2 * np.pi)
            ax_f.set_xticks([])
            if col_idx == 0:
                ax_f.set_ylabel(label, fontsize=9, color=color, labelpad=3)
            else:
                ax_f.set_yticklabels([])
            if m_idx == 0:
                ax_f.set_title(f"t = {t}", fontsize=11)

            ax_e = fig.add_subplot(gs[row_e, col_idx])
            ax_e.plot(x512, err, color=color, lw=1.3)
            ax_e.fill_between(x512, 0, err, color=color, alpha=0.20)
            ax_e.set_xlim(0, 2 * np.pi)
            ax_e.set_ylim(0, error_limits[t])
            ax_e.set_xticks([0, np.pi, 2 * np.pi])
            if m_idx == n_methods - 1:
                ax_e.set_xticklabels(["0", "π", "2π"], fontsize=8)
                ax_e.set_xlabel("x", fontsize=9)
            else:
                ax_e.set_xticklabels([])
            if col_idx == 0:
                ax_e.set_ylabel("|err|", fontsize=8, color=color, labelpad=3)
            else:
                ax_e.set_yticklabels([])

    legend_handles = [Line2D([0], [0], color=C_GT, lw=1.5, alpha=0.6, label="GT (overlay)")]
    for label, _, color, ls in methods:
        legend_handles.append(Line2D([0], [0], color=color, lw=1.6, ls=ls, label=label))
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=len(legend_handles), fontsize=9,
               bbox_to_anchor=(0.5, 0.975), framealpha=0.85)
    fig.suptitle(
        f"Snapshot comparison  |  512-pt  |  traj {traj}  "
        "[columns: t = 0, T/4, T/2, T−1]",
        fontsize=12, y=0.998,
    )
    _save(fig, "fig1_snapshot", aliases=("fig1_snapshot_1d",))


# ---------------------------------------------------------------------------
# Figure 2 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_time(
    ri: dict, ro: dict, re: dict,
    truth_all: torch.Tensor,
    fno_ok: bool = True,
) -> None:
    """RMSE vs time step, mean ± 1 std across trajectories, all methods."""
    n_traj = truth_all.shape[0]

    def _curves(pred: torch.Tensor) -> np.ndarray:
        return torch.stack([
            rmse_over_time(pred[i], truth_all[i]) for i in range(n_traj)
        ]).numpy()   # (n_traj, T)

    c_bic  = _curves(ro["bicubic_512"])
    c_edsr = _curves(re["sr_512"])
    c_one  = _curves(ro["posterior_512"])
    c_iter = _curves(ri["posterior_512"])
    T      = c_bic.shape[1]
    t_ax   = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot(data: np.ndarray, color: str, label: str, ls: str = "-") -> None:
        mu  = data.mean(axis=0)
        sig = data.std(axis=0)
        ax.plot(t_ax, mu, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(t_ax, mu - sig, mu + sig, color=color, alpha=0.15)

    _plot(c_bic,  C_BIC,  "Bicubic",               ls="--")
    if fno_ok and "fno_only_512" in ri:
        _plot(_curves(ri["fno_only_512"]), C_FNO, "FNO-only (autoreg.)", ls=":")
    _plot(c_edsr, C_EDSR, "EDSR-1D (no temporal context)")
    _plot(c_one,  C_ONE,  "One-Shot SR")
    _plot(c_iter, C_ITER, "Iterative Refinement")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE  (512-pt)")
    ax.set_title("RMSE over time  |  mean ± 1 std across test trajectories")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig2_rmse_time")


# ---------------------------------------------------------------------------
# Figure 3 — Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    ri: dict, ro: dict, re: dict,
    truth_all: torch.Tensor,
    traj: int = 0,
    k_max_plot: int = 96,
    k_max_bicubic: int = 32,
    k_max_edsr: int = 63,
) -> None:
    """2×2 panel energy spectrum averaged over trajectories at representative times."""
    T = truth_all.shape[1]
    t_indices = [T // 4, T // 2, (3 * T) // 4, T - 1]
    k_full = np.arange(1, truth_all.shape[-1] // 2 + 1)
    k_obs  = np.arange(1, ro["obs_64"].shape[-1] // 2 + 1)
    full_mask = k_full <= k_max_plot
    bic_mask  = k_full <= min(k_max_plot, k_max_bicubic)
    edsr_mask = k_full <= min(k_max_plot, k_max_edsr)

    spectra: dict = {}
    y_cands = []
    for t in t_indices:
        E_gt   = energy_spectrum(truth_all[:, t]).numpy()[1:]
        E_obs  = energy_spectrum(ro["obs_64"][:, t]).numpy()[1:]
        E_bic  = energy_spectrum(ro["bicubic_512"][:, t]).numpy()[1:]
        E_edsr = energy_spectrum(re["sr_512"][:, t]).numpy()[1:]
        E_one  = energy_spectrum(ro["posterior_512"][:, t]).numpy()[1:]
        E_iter = energy_spectrum(ri["posterior_512"][:, t]).numpy()[1:]
        spectra[t] = (E_gt, E_obs, E_bic, E_edsr, E_one, E_iter)
        y_cands.extend([E_gt[full_mask], E_obs, E_bic[bic_mask],
                        E_edsr[edsr_mask], E_one[full_mask], E_iter[full_mask]])

    y_all      = np.concatenate(y_cands)
    y_positive = y_all[y_all > 0]
    y_min = float(y_positive.min()) * 0.7
    y_max = float(y_all.max()) * 1.2
    ref_k = np.array([3.0, float(k_full[full_mask].max())])

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, t in zip(axes, t_indices):
        E_gt, E_obs, E_bic, E_edsr, E_one, E_iter = spectra[t]
        ref_amp = E_gt[4] * (5.0 ** 2)   # anchor k^-2 at k=5
        ref_y   = ref_amp * ref_k ** (-2.0)

        ax.loglog(k_full[full_mask], E_gt[full_mask],   color=C_GT,   lw=2.3, ls="-",         label="Ground truth")
        ax.loglog(k_obs,             E_obs,             color="dimgray", lw=1.8, ls=":",       label="LR obs (64)")
        ax.loglog(k_full[bic_mask],  E_bic[bic_mask],   color=C_BIC,  lw=2.0, ls="-.",        label="Bicubic")
        ax.loglog(k_full[edsr_mask], E_edsr[edsr_mask], color=C_EDSR, lw=2.0, ls=(0,(4,1.5)), label="EDSR-1D")
        ax.loglog(k_full[full_mask], E_one[full_mask],  color=C_ONE,  lw=2.0, ls="--",        label="One-Shot SR")
        ax.loglog(k_full[full_mask], E_iter[full_mask], color=C_ITER, lw=2.0, ls=(0,(7,2.2)), label="Iterative Refinement")
        ax.loglog(ref_k, ref_y, color="gray", lw=1.0, ls=(0,(1,2.2)), label=r"$k^{-2}$")

        ax.axvline(32, color="gray", ls=":", lw=0.9, alpha=0.65)
        ax.text(0.97, 0.95, f"t = {t}", transform=ax.transAxes,
                ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82))
        ax.set_xlim(1, k_max_plot)
        ax.set_ylim(y_min, y_max)

    axes[0].set_ylabel(r"$E(k) = |\hat{u}_k|^2$")
    axes[2].set_ylabel(r"$E(k) = |\hat{u}_k|^2$")
    axes[2].set_xlabel("Wavenumber $k$")
    axes[3].set_xlabel("Wavenumber $k$")
    axes[1].text(0.05, 0.11,
                 f"Spectra averaged over {truth_all.shape[0]} trajectories\n"
                 f"High-k tail truncated at k = {k_max_plot} for readability\n"
                 f"EDSR shown only through k = {k_max_edsr}; bicubic through k = {k_max_bicubic}",
                 transform=axes[1].transAxes, ha="left", va="bottom",
                 fontsize=9, color="dimgray",
                 bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none", alpha=0.84))

    legend_handles = [
        Line2D([0],[0], color=C_GT,    lw=2.3, ls="-",         label="Ground truth"),
        Line2D([0],[0], color="dimgray", lw=1.8, ls=":",        label="LR obs (64)"),
        Line2D([0],[0], color=C_BIC,   lw=2.0, ls="-.",         label="Bicubic"),
        Line2D([0],[0], color=C_EDSR,  lw=2.0, ls=(0,(4,1.5)), label="EDSR-1D"),
        Line2D([0],[0], color=C_ONE,   lw=2.0, ls="--",         label="One-Shot SR"),
        Line2D([0],[0], color=C_ITER,  lw=2.0, ls=(0,(7,2.2)), label="Iterative Refinement"),
        Line2D([0],[0], color="gray",  lw=1.0, ls=(0,(1,2.2)), label=r"$k^{-2}$"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.965), framealpha=0.9)
    fig.suptitle("Energy spectrum comparison  |  trajectory-averaged  [t = T/4, T/2, 3T/4, T]",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.04, 0.05, 0.99, 0.90))
    _save(fig, "fig3_spectrum")


# ---------------------------------------------------------------------------
# Figure 4 — Hovmöller comparison (5 panels)
# ---------------------------------------------------------------------------

def plot_hovmoller(
    ri: dict, ro: dict, re: dict,
    truth_all: torch.Tensor,
    traj: int = 0,
    n_show: int = 100,
) -> None:
    """5 Hovmöller panels: GT | EDSR | One-Shot SR | Iterative | Bicubic."""
    T   = min(n_show, truth_all.shape[1])
    N   = truth_all.shape[-1]
    x   = np.linspace(0, 2 * np.pi, N, endpoint=False)

    gt   = truth_all[traj, :T].numpy()
    edsr = re["sr_512"       ][traj, :T].numpy()
    one  = ro["posterior_512"][traj, :T].numpy()
    itr  = ri["posterior_512"][traj, :T].numpy()
    bic  = ro["bicubic_512"  ][traj, :T].numpy()

    vmax = max(float(np.percentile(np.abs(gt), 99.5)), 1e-6)

    panels = [
        ("Ground Truth",          gt,   C_GT),
        ("EDSR-1D",               edsr, C_EDSR),
        ("One-Shot SR",           one,  C_ONE),
        ("Iterative Refinement",  itr,  C_ITER),
        ("Bicubic",               bic,  C_BIC),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    ims = []
    for ax, (title, field, bc) in zip(axes, panels):
        im = ax.pcolormesh(x, np.arange(T), field,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ims.append(im)
        ax.set_title(title, color=bc if bc != C_GT else "black", fontsize=11)
        ax.set_xlabel("x")
        for spine in ax.spines.values():
            spine.set_edgecolor(bc)
            spine.set_linewidth(2)

    axes[0].set_ylabel("Time step")
    fig.subplots_adjust(right=0.90, wspace=0.06)
    cbar_ax = fig.add_axes([0.915, 0.12, 0.012, 0.75])
    fig.colorbar(ims[0], cax=cbar_ax, label="u")
    fig.suptitle(f"Hovmöller comparison  |  512-pt  |  traj {traj}", fontsize=14)
    _save(fig, "fig4_hovmoller")


# ---------------------------------------------------------------------------
# Figure 5 — Summary bar chart
# ---------------------------------------------------------------------------

def _spectral_rmse_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    n, T, N = pred.shape
    E_pred  = energy_spectrum(pred.reshape(n * T, N))
    E_truth = energy_spectrum(truth.reshape(n * T, N))
    return float((E_pred[1:] / E_truth[1:].clamp(min=1e-30)).log10().abs().mean())


def _tc_scalar(pred: torch.Tensor) -> float:
    n_traj = pred.shape[0]
    return float(
        torch.stack([temporal_consistency(pred[i]).mean()
                     for i in range(n_traj)]).mean()
    )


def _ssim_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    sigma_p = pred.std()
    sigma_t = truth.std()
    cov     = ((pred - pred.mean()) * (truth - truth.mean())).mean()
    C1, C2  = 0.01 ** 2, 0.03 ** 2
    mu_p, mu_t = pred.mean(), truth.mean()
    return float(
        (2 * mu_p * mu_t + C1) * (2 * cov + C2) /
        ((mu_p**2 + mu_t**2 + C1) * (sigma_p**2 + sigma_t**2 + C2))
    )


def plot_summary_bars(
    ri: dict, ro: dict, re: dict,
    truth_all: torch.Tensor,
    fno_ok: bool = True,
) -> dict:
    """4-metric grouped bar chart (RMSE, Spectral RMSE, Temp. Consistency, SSIM)."""
    T = min(ri["posterior_512"].shape[1],
            ro["posterior_512"].shape[1],
            re["sr_512"].shape[1],
            truth_all.shape[1])
    truth = truth_all[:, :T].float()
    n_traj = truth.shape[0]

    methods = ["Bicubic", "EDSR-1D", "One-Shot SR", "Iterative\nRefinement"]
    colors  = [C_BIC, C_EDSR, C_ONE, C_ITER]
    preds   = [
        ro["bicubic_512"  ][:, :T].float(),
        re["sr_512"       ][:, :T].float(),
        ro["posterior_512"][:, :T].float(),
        ri["posterior_512"][:, :T].float(),
    ]
    if fno_ok and "fno_only_512" in ri:
        methods.insert(3, "FNO-only\n(autoreg.)")
        colors.insert( 3, C_FNO)
        preds.insert(  3, ri["fno_only_512"][:, :T].float())

    n_methods = len(methods)
    mnames    = ["RMSE", "Spectral RMSE", "Temp. Consistency", "SSIM"]
    means     = np.zeros((n_methods, 4))
    stds      = np.zeros((n_methods, 4))

    for m_idx, pred in enumerate(preds):
        rmse_traj = np.array([
            float(pred[i].sub(truth[i]).pow(2).mean().sqrt())
            for i in range(n_traj)
        ])
        means[m_idx, 0] = rmse_traj.mean()
        stds[ m_idx, 0] = rmse_traj.std()
        means[m_idx, 1] = _spectral_rmse_scalar(pred, truth)
        means[m_idx, 2] = _tc_scalar(pred)
        means[m_idx, 3] = _ssim_scalar(pred, truth)
        # stds for scalar metrics are 0 (computed over all traj+time at once)

    fig, axes = plt.subplots(1, len(mnames), figsize=(18, 5))
    x = np.arange(n_methods)
    width = 0.55

    for k_idx, (ax, metric_name) in enumerate(zip(axes, mnames)):
        bars = ax.bar(
            x,
            means[:, k_idx],
            width,
            yerr=stds[:, k_idx],
            capsize=5,
            color=colors,
            alpha=0.85,
        )
        label_pad = max(stds[:, k_idx].max() * 0.05, means[:, k_idx].max() * 0.01, 1e-9)
        for bar, mean in zip(bars, means[:, k_idx]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + label_pad,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(metric_name)
        ax.set_ylabel("Value" if k_idx == 0 else "")

    fig.suptitle(
        "Method Comparison  |  512-pt  (mean ± std across test trajectories)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig5_summary_bars", aliases=("fig5_summary_bars_1d",))

    return {
        "methods":      [m.replace("\n", " ") for m in methods],
        "metric_names": mnames,
        "means":        means,
        "stds":         stds,
    }


# ---------------------------------------------------------------------------
# Figure 6 — Per-mode spectral RMSE, all methods
# ---------------------------------------------------------------------------

def plot_spectral_rmse(
    ri: dict, ro: dict, re: dict,
    truth_all: torch.Tensor,
    k_max_plot: int = 248,
) -> None:
    """Per-wavenumber |log10(E_pred / E_gt)| for all methods."""
    def _E(x: torch.Tensor) -> torch.Tensor:
        n, T, N = x.shape
        return energy_spectrum(x.reshape(n * T, N))

    E_gt   = _E(truth_all)
    E_bic  = _E(ro["bicubic_512"])
    E_edsr = _E(re["sr_512"])
    E_one  = _E(ro["posterior_512"])
    E_iter = _E(ri["posterior_512"])

    eps = 1e-30

    def _srmse(E_pred: torch.Tensor) -> np.ndarray:
        return (E_pred[1:] / E_gt[1:].clamp(min=eps)).log10().abs().numpy()

    sr_bic  = _srmse(E_bic)
    sr_edsr = _srmse(E_edsr)
    sr_one  = _srmse(E_one)
    sr_iter = _srmse(E_iter)

    N    = truth_all.shape[-1]
    k    = np.arange(1, N // 2 + 1)
    mask = k <= k_max_plot
    k_p  = k[mask]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shade gap between iterative and next-best competitor
    ax.fill_between(k_p, sr_iter[mask], sr_one[mask],
                    where=(sr_one[mask] >= sr_iter[mask]),
                    color=C_ITER, alpha=0.12, label="Gap: Iterative vs One-Shot")

    ax.plot(k_p, sr_bic[mask],  color=C_BIC,  lw=1.8, ls="--",
            label=f"Bicubic       (mean={sr_bic[mask].mean():.4f})")
    ax.plot(k_p, sr_edsr[mask], color=C_EDSR, lw=2.0,
            label=f"EDSR-1D       (mean={sr_edsr[mask].mean():.4f})")
    ax.plot(k_p, sr_one[mask],  color=C_ONE,  lw=2.0,
            label=f"One-Shot SR   (mean={sr_one[mask].mean():.4f})")
    ax.plot(k_p, sr_iter[mask], color=C_ITER, lw=2.0,
            label=f"Iterative Ref.(mean={sr_iter[mask].mean():.4f})")

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Wavenumber  k")
    ax.set_ylabel(r"$|\log_{10}(E_\mathrm{pred}(k) / E_\mathrm{GT}(k))|$")
    ax.set_title(
        "Per-mode spectral RMSE  |  0 = perfect, larger = worse\n"
        "Shaded region: improvement of Iterative Refinement over One-Shot SR"
    )
    ax.set_xlim(1, k_max_plot)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.85, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig6_spectral_rmse")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def _print_table(table: dict) -> None:
    methods      = table["methods"]
    metric_names = table["metric_names"]
    means        = table["means"]
    col_w = 18
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
    p = argparse.ArgumentParser(description="1D EDSR comparison figures (all methods)")
    p.add_argument("--iterative", type=str, default="results/inference_results.pt")
    p.add_argument("--oneshot",   type=str, default="results_oneshot_1d/inference_results.pt")
    p.add_argument("--edsr",      type=str, default="results_edsr_1d/inference_results.pt")
    p.add_argument("--data_dir",  type=str, default="data")
    p.add_argument("--figures",   type=str, default="1,2,3,4,5,6",
                   help="Comma-separated figures to generate (default: all)")
    p.add_argument("--n_steps",   type=int, default=None,
                   help="Trim all results to this many time steps")
    p.add_argument("--traj",      type=int, default=0,
                   help="Trajectory index for snapshot/spectrum/Hovmöller (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    figs = {int(x) for x in args.figures.split(",")}

    print(f"Loading iterative results  from {args.iterative} ...")
    ri = torch.load(args.iterative, map_location="cpu", weights_only=True)

    print(f"Loading one-shot results   from {args.oneshot} ...")
    ro = torch.load(args.oneshot,   map_location="cpu", weights_only=True)

    print(f"Loading EDSR-1D results    from {args.edsr} ...")
    re = torch.load(args.edsr,      map_location="cpu", weights_only=True)

    print(f"Loading ground truth       from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )

    # Align all result dicts to the shortest common (n_traj, T)
    T = min(
        ri["posterior_512"].shape[1],
        ro["posterior_512"].shape[1],
        re["sr_512"].shape[1],
        test_data["u_512"].shape[1],
    )
    if args.n_steps is not None:
        T = min(T, args.n_steps)

    n_traj = min(
        ri["posterior_512"].shape[0],
        ro["posterior_512"].shape[0],
        re["sr_512"].shape[0],
        test_data["u_512"].shape[0],
    )
    print(f"Using n_traj={n_traj}, T={T} (aligned across all result files)")

    truth_all = test_data["u_512"][:n_traj, :T].float()   # (n_traj, T, 512)

    def _trim(d: dict) -> dict:
        return {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                for k, v in d.items()}

    ri = _trim(ri)
    ro = _trim(ro)
    re = _trim(re)

    fno_ok = not _fno_blown_up(ri, ro, truth_all)
    table  = None

    if 1 in figs:
        print("\nFigure 1: Snapshot comparison ...")
        plot_snapshot(ri, ro, re, truth_all, traj=args.traj, fno_ok=fno_ok)

    if 2 in figs:
        print("\nFigure 2: RMSE over time ...")
        plot_rmse_time(ri, ro, re, truth_all, fno_ok=fno_ok)

    if 3 in figs:
        print("\nFigure 3: Energy spectrum ...")
        plot_spectrum(ri, ro, re, truth_all, traj=args.traj)

    if 4 in figs:
        print("\nFigure 4: Hovmöller comparison ...")
        plot_hovmoller(ri, ro, re, truth_all, traj=args.traj)

    if 5 in figs:
        print("\nFigure 5: Summary bar chart ...")
        table = plot_summary_bars(ri, ro, re, truth_all, fno_ok=fno_ok)

    if 6 in figs:
        print("\nFigure 6: Per-mode spectral RMSE ...")
        plot_spectral_rmse(ri, ro, re, truth_all)

    if table is not None:
        _print_table(table)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
