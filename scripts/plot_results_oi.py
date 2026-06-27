"""
Comparison plots for the Spectral OI DA baseline vs all other methods.

The central finding: OI beats IR on aggregate RMSE because RMSE is dominated
by the low-k modes that the 32x32 observation constrains directly.  OI applies
the observation exactly at those modes (Kalman gain ≈ 1 for k≤16); IR adds
diffusion noise at those same modes.  At high-k (k>16), OI and IR are
equivalent — both fall back to the FNO forecast.

Figures saved to plots_oi/:
  fig1_snapshot.{png,pdf}          — 2×5 field + error grid, one timestep
  fig2_rmse_time.{png,pdf}         — RMSE over time, all methods
  fig3_spectrum_fullband.{png,pdf}  — Log-log E(k), all methods
  fig4_per_band_rmse.{png,pdf}      — RMSE decomposed by spectral band
                                      [k≤16 obs / 16<k≤32 transition / k>32 sub-Nyquist]
  fig5_gains.{png,pdf}              — Kalman gain K(k) vs wavenumber
  fig6_summary_bars.{png,pdf}       — Full RMSE + spectral RMSE bar chart

Usage:
    python scripts/plot_results_oi.py
        [--oi          results_oi/inference_results.pt]
        [--iterative   results_2d/inference_results.pt]
        [--oneshot     results_oneshot/inference_results.pt]
        [--data_dir    data_2d]
        [--config      configs/kraichnan.yaml]
        [--snapshot_t  50]
        [--traj        0]
        [--figures     1,2,3,4,5,6]
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
    structural_similarity_2d,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  14,
    "legend.fontsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,
})

PLOTS_DIR  = Path("plots_oi")
CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "inferno"

C_TRUTH = "black"
C_BIC   = "gray"
C_OI    = "#9467bd"   # purple — OI (classical linear DA)
C_ONE   = "#2ca02c"   # green
C_ITER  = "#1f77b4"   # blue
C_FNO   = "#d62728"   # red


def _save(fig: plt.Figure, stem: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Per-band RMSE helper (core to the OI analysis)
# ---------------------------------------------------------------------------

def band_rmse(pred: torch.Tensor, truth: torch.Tensor,
              k_lo: float, k_hi: float) -> float:
    """RMSE restricted to Fourier modes in the radial band [k_lo, k_hi).

    Zeroes out all modes outside the band in rfft2 space, then irfft2 back
    and computes RMSE of the resulting band-filtered error.

    Args:
        pred, truth: (..., ny, nx) — any leading dims
        k_lo, k_hi:  radial wavenumber range (integer mode counts)

    Returns:
        Scalar RMSE within the band.
    """
    ny, nx = pred.shape[-2], pred.shape[-1]
    kx = torch.fft.fftfreq(ny,  d=1.0 / ny)
    ky = torch.fft.rfftfreq(nx, d=1.0 / nx)
    kx_g = kx.view(-1, 1).expand(ny, nx // 2 + 1)
    ky_g = ky.view(1, -1).expand(ny, nx // 2 + 1)
    k_mag = (kx_g.pow(2) + ky_g.pow(2)).sqrt()
    mask  = (k_mag >= k_lo) & (k_mag < k_hi)

    P = torch.fft.rfft2(pred.float())
    T = torch.fft.rfft2(truth.float())
    err = P - T
    # mask is (ny, nx//2+1); zero out modes outside band via multiplication
    err = err * mask.to(err.device)
    err_field = torch.fft.irfft2(err, s=(ny, nx))
    return float(err_field.pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Figure 1 — Snapshot: 2 rows × 5 cols
# ---------------------------------------------------------------------------

def plot_snapshot(
    r_oi: dict, r_ir: dict, r_os: dict,
    truth_all: torch.Tensor,
    t: int = 50, traj: int = 0,
) -> None:
    """Row 0: GT | FNO-fc | OI | One-Shot | Iterative
       Row 1: (blank) | |err| FNO | |err| OI | |err| One-Shot | |err| Iterative
    Error scale anchored to the FNO forecast error (largest)."""
    T = truth_all.shape[1]
    t = min(t, T - 1)

    gt_f   = truth_all[traj, t].numpy()
    fc_f   = r_ir["forecast_256"][traj, t].numpy()
    oi_f   = r_oi["oi_256"      ][traj, t].numpy()
    one_f  = r_os["posterior_256"][traj, t].numpy()
    iter_f = r_ir["posterior_256"][traj, t].numpy()

    err_fc   = np.abs(fc_f   - gt_f)
    err_oi   = np.abs(oi_f   - gt_f)
    err_one  = np.abs(one_f  - gt_f)
    err_iter = np.abs(iter_f - gt_f)

    vmax = max(float(np.percentile(np.abs(gt_f), 99.5)), 1e-6)
    emax = max(float(np.percentile(err_fc, 99.5)), 1e-6)

    col_titles = ["Ground Truth", "FNO forecast\n(background)",
                  "Spectral OI\n(linear DA)", "One-Shot SR", "Iterative\nRefinement"]
    row0 = [gt_f,   fc_f,    oi_f,    one_f,    iter_f   ]
    row1 = [None,   err_fc,  err_oi,  err_one,  err_iter ]

    fig = plt.figure(figsize=(22, 9))
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                            hspace=0.06, wspace=0.04,
                            left=0.04, right=0.92, top=0.91, bottom=0.04)

    im_field = im_err = None
    for col in range(5):
        ax0 = fig.add_subplot(gs[0, col])
        im  = ax0.imshow(row0[col], cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                         origin="lower", aspect="equal", interpolation="nearest")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(col_titles[col], fontsize=10, pad=4)
        if col == 0:
            ax0.set_ylabel("Field  ω", fontsize=10, labelpad=3)
        im_field = im

        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_xticks([]); ax1.set_yticks([])
        if row1[col] is None:
            ax1.axis("off")
        else:
            im_e = ax1.imshow(row1[col], cmap=CMAP_ERR, vmin=0, vmax=emax,
                              origin="lower", aspect="equal", interpolation="nearest")
            im_err = im_e
        if col == 0:
            ax1.set_ylabel("|Error|", fontsize=10, labelpad=3)

    fig.add_axes([0.933, 0.52, 0.012, 0.38]).set_visible(False)
    cb1 = fig.colorbar(im_field, ax=fig.axes, fraction=0.012,
                       pad=0.01, shrink=0.45, anchor=(0, 1.0))
    cb1.set_label("Vorticity  ω", fontsize=10)

    cb2 = fig.colorbar(im_err, ax=fig.axes, fraction=0.012,
                       pad=0.01, shrink=0.45, anchor=(0, 0.0))
    cb2.set_label("|Error|", fontsize=10)

    fig.suptitle(
        f"256×256 Vorticity — OI baseline comparison  |  t={t}, traj={traj}",
        fontsize=13, y=0.97,
    )
    _save(fig, "fig1_snapshot")


# ---------------------------------------------------------------------------
# Figure 2 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_time(
    r_oi: dict, r_ir: dict, r_os: dict,
    truth_all: torch.Tensor,
) -> None:
    n_traj = truth_all.shape[0]
    T      = truth_all.shape[1]

    def _curves(pred, tru):
        return torch.stack([rmse_over_time_2d(pred[i], tru[i])
                            for i in range(n_traj)]).numpy()

    c_bic  = _curves(r_os["bicubic_256"],    truth_all)
    c_fc   = _curves(r_ir["forecast_256"],   truth_all)
    c_oi   = _curves(r_oi["oi_256"],         truth_all)
    c_one  = _curves(r_os["posterior_256"],  truth_all)
    c_iter = _curves(r_ir["posterior_256"],  truth_all)
    t_ax   = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i in range(n_traj):
        ax.plot(t_ax, c_bic[i],  color=C_BIC,   alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_fc[i],   color=C_FNO,   alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_oi[i],   color=C_OI,    alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_one[i],  color=C_ONE,   alpha=0.20, lw=0.7)
        ax.plot(t_ax, c_iter[i], color=C_ITER,  alpha=0.20, lw=0.7)

    ax.plot(t_ax, c_bic.mean(0),  color=C_BIC,  lw=2.0, ls="--",  label="Bicubic (spectral)")
    ax.plot(t_ax, c_fc.mean(0),   color=C_FNO,  lw=2.0, ls=":",   label="FNO forecast (background)")
    ax.plot(t_ax, c_oi.mean(0),   color=C_OI,   lw=2.4, ls="-",   label="Spectral OI + FNO  (linear DA)")
    ax.plot(t_ax, c_one.mean(0),  color=C_ONE,  lw=2.0, ls="-",   label="One-Shot SR")
    ax.plot(t_ax, c_iter.mean(0), color=C_ITER, lw=2.0, ls="-",   label="Iterative Refinement (ours)")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE @ 256×256")
    ax.set_title("RMSE over time — OI vs nonlinear methods")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig2_rmse_time")


# ---------------------------------------------------------------------------
# Figure 3 — Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(
    r_oi: dict, r_ir: dict, r_os: dict,
    truth_all: torch.Tensor,
    k_forcing: float = 4.0,
    traj: int = 0,
) -> None:
    """Multi-panel E(k): one column per time snapshot (t=0, T/4, T/2, 3T/4, T-1).
    Methods shown: Ground Truth, Bicubic, Spectral OI, Iterative Refinement.
    FNO forecast and One-Shot SR removed to keep the comparison focused.
    """
    T = truth_all.shape[1]
    t_indices = [T // 4, T // 2, 3 * T // 4, T - 1]
    t_labels  = ["$t=T/4$", "$t=T/2$", "$t=3T/4$", "$t=T$"]

    def _E_at_t(tensor, t_idx):
        # single frame from trajectory `traj`: shape (1, ny, nx)
        frame = tensor[traj, t_idx].unsqueeze(0)
        E, k  = radial_energy_spectrum(frame)
        return E.numpy(), k.numpy()

    k_ref_arr = np.array([3.0, 120.0])

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)

    for col, (t_idx, t_label) in enumerate(zip(t_indices, t_labels)):
        ax = axes[col]

        E_truth, k_bins = _E_at_t(truth_all,              t_idx)
        E_bic,   _      = _E_at_t(r_os["bicubic_256"],    t_idx)
        E_oi,    _      = _E_at_t(r_oi["oi_256"],         t_idx)
        E_iter,  _      = _E_at_t(r_ir["posterior_256"],  t_idx)

        k = k_bins[1:]
        mask_full = k <= 120
        mask_bic  = k <= 20   # bicubic is zero-padded above obs Nyquist

        # k^{-3} reference anchored to truth at k=5
        idx_anchor = 4
        E_ref = E_truth[1:][idx_anchor] * (5.0 ** 3) * k_ref_arr ** (-3)

        ax.loglog(k[mask_full], E_truth[1:][mask_full],
                  color=C_TRUTH, lw=2.0,        label="Ground truth")
        ax.loglog(k[mask_bic],  E_bic[1:][mask_bic],
                  color=C_BIC,   lw=1.4, ls="--", label="Bicubic")
        ax.loglog(k[mask_full], E_oi[1:][mask_full],
                  color=C_OI,    lw=2.0, ls="-.", label="Spectral OI")
        ax.loglog(k[mask_full], E_iter[1:][mask_full],
                  color=C_ITER,  lw=2.0,          label="Iter. Refinement")
        ax.loglog(k_ref_arr, E_ref,
                  color="gray",  lw=0.9, ls=":",  label=r"$k^{-3}$")

        # Observation Nyquist
        ax.axvline(16, color="purple", ls="--", lw=0.9, alpha=0.6)

        ax.set_xlabel("Wavenumber $k$", fontsize=10)
        ax.set_title(t_label, fontsize=12)
        ax.set_xlim(left=1, right=130)
        ax.grid(True, which="both", ls="--", alpha=0.2)

        if col == 0:
            ax.set_ylabel("$E(k)$", fontsize=11)
            ax.legend(loc="lower left", fontsize=8.5, framealpha=0.85)
        else:
            # mark Nyquist on all panels but only label on first
            pass

        # annotate Nyquist line on top panel only
        if col == 0:
            ymin, ymax = ax.get_ylim()
            ax.text(17, ymax * 0.5, "$k_{obs}$=16", color="purple",
                    fontsize=8, va="top")

    fig.suptitle(
        "Radial energy spectrum at equally-spaced timesteps  "
        "(Ground Truth / Bicubic / Spectral OI / Iterative Refinement)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig3_spectrum_fullband")


# ---------------------------------------------------------------------------
# Figure 4 — Per-band RMSE breakdown (key figure for this analysis)
# ---------------------------------------------------------------------------

def plot_per_band_rmse(
    r_oi: dict, r_ir: dict, r_os: dict,
    truth_all: torch.Tensor,
) -> None:
    """Bar chart of RMSE decomposed into three spectral bands.

    This is the diagnostic figure that explains WHY OI beats IR on aggregate
    RMSE: OI uses the observation to zero out error at k≤16, while IR cannot
    fully recover those modes.  At k>16 both methods are equivalent.
    """
    methods = ["Bicubic", "FNO\nforecast", "Spectral\nOI + FNO",
               "One-Shot\nSR", "Iterative\nRefinement"]
    colors  = [C_BIC, C_FNO, C_OI, C_ONE, C_ITER]
    preds   = [
        r_os["bicubic_256"],
        r_ir["forecast_256"],
        r_oi["oi_256"],
        r_os["posterior_256"],
        r_ir["posterior_256"],
    ]
    bands = [
        ("k ≤ 16\n(obs-constrained)", 1,  17),
        ("16 < k ≤ 32\n(transition)",  16, 33),
        ("k > 32\n(unobserved)",       32, 129),
    ]
    band_labels = [b[0] for b in bands]
    n_methods   = len(methods)
    n_bands     = len(bands)

    means = np.zeros((n_methods, n_bands))
    for m_idx, pred in enumerate(preds):
        n = min(pred.shape[0], truth_all.shape[0])
        T = min(pred.shape[1], truth_all.shape[1])
        p = pred[:n, :T].float()
        tr = truth_all[:n, :T]
        for b_idx, (_, k_lo, k_hi) in enumerate(bands):
            means[m_idx, b_idx] = band_rmse(p, tr, k_lo, k_hi)

    x       = np.arange(n_bands)
    width   = 0.14
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for m_idx, (method, color, offset) in enumerate(zip(methods, colors, offsets)):
        bars = ax.bar(x + offset, means[m_idx], width,
                      label=method.replace("\n", " "),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, means[m_idx]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + means.max() * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=11)
    ax.set_ylabel("RMSE (band-filtered)")
    ax.set_title(
        "Per-band RMSE decomposition  |  256×256\n"
        "OI advantage is entirely at k≤16 (observed modes); "
        "IR and OI are equivalent at k>16"
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig4_per_band_rmse")

    # Print table
    print("\n── Per-band RMSE table ──────────────────────────────────────────")
    header = f"  {'Method':<24}" + "".join(f"  {b[0].replace(chr(10),' '):>20}" for b in bands)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for m_idx, method in enumerate(methods):
        row = f"  {method.replace(chr(10),' '):<24}"
        row += "".join(f"  {means[m_idx, b_idx]:>20.4f}" for b_idx in range(n_bands))
        print(row)


# ---------------------------------------------------------------------------
# Figure 5 — Kalman gain K(k) vs wavenumber
# ---------------------------------------------------------------------------

def plot_gains(r_oi: dict) -> None:
    """Show the per-shell Kalman gain K(k) used by the OI analysis."""
    if "gains" not in r_oi:
        print("  [SKIP] gains not in results file")
        return

    gains  = r_oi["gains"].numpy()
    k      = np.arange(len(gains))
    k_obs  = 16

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k[1:], gains[1:], color=C_OI, lw=2.5)
    ax.fill_between(k[1:], 0, gains[1:], color=C_OI, alpha=0.15)

    ax.axvline(k_obs, color="purple", ls="--", lw=1.2, alpha=0.7,
               label=f"32×32 Nyquist  k={k_obs}")
    ax.axhline(1.0,   color="gray",   ls=":",  lw=1.0, alpha=0.6)
    ax.axhline(0.0,   color="gray",   ls=":",  lw=1.0, alpha=0.6)

    ax.set_xlabel("Wavenumber  $k$")
    ax.set_ylabel("Kalman gain  $K(k)$")
    ax.set_title(
        "OI Kalman gain per radial shell\n"
        r"$K(k) = \sigma_b^2(k) / (\sigma_b^2(k) + \sigma_r^2(k))$"
        "  —  estimated from training data"
    )
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10)

    # Annotate the two regimes
    ax.text(k_obs * 0.5, 0.85, "Obs-constrained\n(K≈1, use obs)",
            ha="center", va="top", fontsize=10, color=C_OI)
    ax.text(k_obs * 1.7, 0.15, "Unobserved\n(K=0, keep forecast)",
            ha="left", va="bottom", fontsize=10, color=C_FNO)
    fig.tight_layout()
    _save(fig, "fig5_gains")


# ---------------------------------------------------------------------------
# Figure 6 — Summary bars: full RMSE + spectral RMSE
# ---------------------------------------------------------------------------

def _spectral_rmse_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    E_pred,  k = radial_energy_spectrum(pred.reshape(-1, *pred.shape[-2:]))
    E_truth, _ = radial_energy_spectrum(truth.reshape(-1, *truth.shape[-2:]))
    mask = k >= 1
    return float((E_pred[mask].clamp(1e-30).log10()
                  - E_truth[mask].clamp(1e-30).log10()).pow(2).mean().sqrt())


def plot_summary_bars(
    r_oi: dict, r_ir: dict, r_os: dict,
    truth_all: torch.Tensor,
) -> None:
    n_traj = truth_all.shape[0]
    T      = truth_all.shape[1]

    methods = ["Bicubic", "FNO forecast", "Spectral OI\n+ FNO", "One-Shot SR",
               "Iterative\nRefinement"]
    colors  = [C_BIC, C_FNO, C_OI, C_ONE, C_ITER]
    preds   = [
        r_os["bicubic_256"],
        r_ir["forecast_256"],
        r_oi["oi_256"],
        r_os["posterior_256"],
        r_ir["posterior_256"],
    ]

    metric_names = ["RMSE", "Spectral RMSE\n(log E(k))"]
    n_m = len(metric_names)
    means = np.zeros((len(methods), n_m))
    stds  = np.zeros((len(methods), n_m))

    for m_idx, pred in enumerate(preds):
        n = min(pred.shape[0], n_traj)
        Tc = min(pred.shape[1], T)
        p  = pred[:n, :Tc].float()
        tr = truth_all[:n, :Tc]
        rmse_traj = np.array([
            float((p[i] - tr[i]).pow(2).mean().sqrt()) for i in range(n)
        ])
        means[m_idx, 0] = rmse_traj.mean()
        stds[ m_idx, 0] = rmse_traj.std()
        means[m_idx, 1] = _spectral_rmse_scalar(p, tr)

    x      = np.arange(n_m)
    width  = 0.14
    offsets = np.linspace(-(len(methods)-1)/2, (len(methods)-1)/2, len(methods)) * width

    fig, axes = plt.subplots(1, n_m, figsize=(11, 5))
    for k_idx, (ax, mname) in enumerate(zip(axes, metric_names)):
        for m_idx, (method, color, offset) in enumerate(zip(methods, colors, offsets)):
            bar = ax.bar(offset, means[m_idx, k_idx], width,
                         yerr=stds[m_idx, k_idx] if k_idx == 0 else 0,
                         capsize=4, color=color, alpha=0.85,
                         label=method.replace("\n", " "))
            ax.text(offset, means[m_idx, k_idx] + stds[m_idx, k_idx] * 0.05 + 1e-4,
                    f"{means[m_idx,k_idx]:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=80)
        ax.set_xticks([])
        ax.set_title(mname)
        ax.set_ylabel("Value" if k_idx == 0 else "")
        if k_idx == n_m - 1:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    fig.suptitle(
        "Summary metrics  |  256×256  (mean ± std across test trajectories)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_summary_bars")


# ---------------------------------------------------------------------------
# Arg parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OI baseline comparison plots")
    p.add_argument("--oi",         type=str, default="results_oi/inference_results.pt")
    p.add_argument("--iterative",  type=str, default="results_2d/inference_results.pt")
    p.add_argument("--oneshot",    type=str, default="results_oneshot/inference_results.pt")
    p.add_argument("--data_dir",   type=str, default="data_2d")
    p.add_argument("--config",     type=str, default="configs/kraichnan.yaml")
    p.add_argument("--snapshot_t", type=int, default=50)
    p.add_argument("--traj",       type=int, default=0)
    p.add_argument("--figures",    type=str, default="1,2,3,4,5,6")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    figs = {int(x) for x in args.figures.split(",")}
    cfg  = OmegaConf.load(args.config)

    print(f"Loading OI results       from {args.oi} ...")
    r_oi = torch.load(args.oi,        map_location="cpu", weights_only=True)

    print(f"Loading iterative results from {args.iterative} ...")
    r_ir = torch.load(args.iterative,  map_location="cpu", weights_only=True)

    print(f"Loading one-shot results  from {args.oneshot} ...")
    r_os = torch.load(args.oneshot,    map_location="cpu", weights_only=True)

    print(f"Loading ground truth      from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )

    # Align to common (n_traj, T)
    T = min(r_oi["oi_256"].shape[1],
            r_ir["posterior_256"].shape[1],
            r_os["posterior_256"].shape[1],
            test_data["w_256"].shape[1])
    n_traj = min(r_oi["oi_256"].shape[0],
                 r_ir["posterior_256"].shape[0],
                 r_os["posterior_256"].shape[0],
                 test_data["w_256"].shape[0])

    truth_all = test_data["w_256"][:n_traj, :T].float()
    print(f"Using n_traj={n_traj}, T={T}")

    def _trim(d):
        return {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                for k, v in d.items()}

    r_oi = _trim(r_oi)
    r_ir = _trim(r_ir)
    r_os = _trim(r_os)

    k_f = float(cfg.pde.forcing_band_center)

    if 1 in figs:
        print("\nFigure 1: Snapshot comparison ...")
        plot_snapshot(r_oi, r_ir, r_os, truth_all,
                      t=args.snapshot_t, traj=args.traj)

    if 2 in figs:
        print("\nFigure 2: RMSE over time ...")
        plot_rmse_time(r_oi, r_ir, r_os, truth_all)

    if 3 in figs:
        print("\nFigure 3: Energy spectrum ...")
        plot_spectrum(r_oi, r_ir, r_os, truth_all, k_forcing=k_f, traj=args.traj)

    if 4 in figs:
        print("\nFigure 4: Per-band RMSE decomposition ...")
        plot_per_band_rmse(r_oi, r_ir, r_os, truth_all)

    if 5 in figs:
        print("\nFigure 5: Kalman gains K(k) ...")
        plot_gains(r_oi)

    if 6 in figs:
        print("\nFigure 6: Summary bars ...")
        plot_summary_bars(r_oi, r_ir, r_os, truth_all)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
