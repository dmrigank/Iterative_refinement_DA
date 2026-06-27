"""
plot_oi_vs_ir.py — Visualise where Iterative Refinement beats Spectral OI.

Three figures saved to plots_oi/:

  fig_band_rmse.png/pdf   — per-wavenumber-band RMSE bar chart
                             Shows OI is exact at k<16 but blind above;
                             IR adds value at unobserved scales.

  fig_spectrum_compare.png/pdf — mean E(k) for GT / Bicubic / OI / IR
                             Shows OI has a hard spectral cutoff at k≈16;
                             IR follows the k^{-3} cascade past it.

  fig_error_fields.png/pdf — spatial error maps (pred − truth) at one
                             representative time step, side-by-side
                             for OI and IR, plus a high-pass version
                             (k > 16 modes only) to isolate fine-scale
                             residuals.

Usage
-----
    python scripts/plot_oi_vs_ir.py \
        [--oi_results  results_oi/inference_results.pt] \
        [--ir_results  results_2d/inference_results.pt] \
        [--oneshot_results results_oneshot/inference_results.pt] \
        [--out_dir     plots_oi]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset_2d import spectral_upsample_2d, spectral_downsample_2d
from src.evaluation.metrics_2d import radial_energy_spectrum

# ── matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "gt":      "#222222",
    "bicubic": "#888888",
    "oi":      "#2196F3",   # blue
    "ir":      "#E53935",   # red
    "fno":     "#FF9800",   # orange
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(((a - b) ** 2).mean()))


def _spectral_bandpass(w: torch.Tensor, k_lo: int, k_hi: int) -> torch.Tensor:
    """Keep only radial wavenumbers in [k_lo, k_hi) (inclusive low, exclusive high)."""
    ny, nx = w.shape[-2], w.shape[-1]
    W = torch.fft.rfft2(w)
    ky = torch.fft.fftfreq(ny, d=1.0 / ny).to(w.device).float()   # (ny,)
    kx = torch.arange(nx // 2 + 1, device=w.device).float()        # (nx//2+1,) — rfft
    KX, KY = torch.meshgrid(kx, ky, indexing="xy")                  # (ny, nx//2+1) each
    K = torch.sqrt(KX ** 2 + KY ** 2)
    mask = (K >= k_lo) & (K < k_hi)                         # (ny, nx//2+1)
    W_masked = W * mask.unsqueeze(0).unsqueeze(0) if w.dim() == 4 else W * mask
    return torch.fft.irfft2(W_masked, s=(ny, nx))


def _band_rmse(pred: torch.Tensor, truth: torch.Tensor,
               k_lo: int, k_hi: int) -> float:
    """RMSE computed on the bandpassed versions of pred and truth."""
    p_bp = _spectral_bandpass(pred, k_lo, k_hi)
    t_bp = _spectral_bandpass(truth, k_lo, k_hi)
    return _rmse(p_bp, t_bp)


def _bicubic_up(obs32: torch.Tensor) -> torch.Tensor:
    """Bicubic upsample from 32×32 → 256×256."""
    return torch.nn.functional.interpolate(
        obs32.float(), size=(256, 256), mode="bicubic", align_corners=False
    )


def _mean_spectrum(fields: torch.Tensor, max_k: int = 130) -> tuple[np.ndarray, np.ndarray]:
    """
    fields: (N, ny, nx) — average radial energy spectrum over N samples.
    Returns (k_bins, E_mean).
    """
    # Compute on the whole batch at once for consistency
    E, k = radial_energy_spectrum(fields)   # (N, ny, nx)
    E_arr = np.array(E)
    k_arr = np.array(k)
    if E_arr.ndim == 1:
        E_mean = E_arr
    else:
        E_mean = E_arr.mean(axis=0)
    mask = k_arr < max_k
    return k_arr[mask], E_mean[mask]


# ── per-band RMSE figure ──────────────────────────────────────────────────────

def plot_band_rmse(oi: torch.Tensor, ir: torch.Tensor,
                   fno: torch.Tensor, bicubic: torch.Tensor,
                   truth: torch.Tensor, out_dir: str) -> None:
    """Bar chart of RMSE in three wavenumber bands."""
    # bands: observed | transition | unobserved (at 256×256, Nyquist of 32×32 = 16)
    bands = [
        ("Observed\n$k < 16$",      1,  16),
        ("Transition\n$16 \\leq k < 64$", 16, 64),
        ("Unobserved\n$k \\geq 64$",      64, 129),
    ]

    methods = ["Bicubic", "FNO-only", "Spectral OI", "Iter. Refinement (ours)"]
    preds   = [bicubic,   fno,        oi,             ir]
    colors  = [COLORS["bicubic"], COLORS["fno"], COLORS["oi"], COLORS["ir"]]

    # flatten (traj, T, ny, nx) → (N, ny, nx) for RMSE
    def flat(x): return x.reshape(-1, x.shape[-2], x.shape[-1])
    truth_f = flat(truth)

    results = {}  # method → [band0, band1, band2]
    for name, pred in zip(methods, preds):
        pred_f = flat(pred)
        results[name] = [
            _band_rmse(pred_f, truth_f, lo, hi)
            for _, lo, hi in bands
        ]

    n_bands   = len(bands)
    n_methods = len(methods)
    x         = np.arange(n_bands)
    width     = 0.18
    offsets   = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, color) in enumerate(zip(methods, colors)):
        vals = results[name]
        bars = ax.bar(x + offsets[i], vals, width, label=name, color=color,
                      edgecolor="white", linewidth=0.5, zorder=3)
        # value labels on top of each bar
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bands])
    ax.set_ylabel("RMSE (vorticity)")
    ax.set_title("Per-Band RMSE: OI vs Iterative Refinement")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # annotate the key insight
    ax.annotate(
        "OI wins here\n(obs exact by construction)",
        xy=(0, results["Spectral OI"][0]),
        xytext=(0.22, results["Spectral OI"][0] + 0.04),
        fontsize=8.5, color=COLORS["oi"],
        arrowprops=dict(arrowstyle="->", color=COLORS["oi"], lw=1.2),
    )
    ax.annotate(
        "IR wins here\n(synthesises unobserved scales)",
        xy=(2, results["Iter. Refinement (ours)"][2]),
        xytext=(1.62, results["Iter. Refinement (ours)"][2] + 0.04),
        fontsize=8.5, color=COLORS["ir"],
        arrowprops=dict(arrowstyle="->", color=COLORS["ir"], lw=1.2),
    )

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_band_rmse.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_band_rmse")


# ── energy spectrum comparison ────────────────────────────────────────────────

def plot_spectrum_compare(oi: torch.Tensor, ir: torch.Tensor,
                          fno: torch.Tensor, bicubic: torch.Tensor,
                          truth: torch.Tensor, out_dir: str) -> None:
    """Log-log E(k) for GT / Bicubic / OI / IR showing spectral cutoff."""
    # Use a subset for speed: first trajectory, all time steps
    def flat_traj0(x): return x[0].reshape(-1, x.shape[-2], x.shape[-1])

    print("  Computing spectra (this may take ~30 s)...")
    k_t,  E_t   = _mean_spectrum(flat_traj0(truth))
    k_b,  E_b   = _mean_spectrum(flat_traj0(bicubic))
    k_oi, E_oi  = _mean_spectrum(flat_traj0(oi))
    k_ir, E_ir  = _mean_spectrum(flat_traj0(ir))
    k_fn, E_fn  = _mean_spectrum(flat_traj0(fno))

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(k_t,  E_t,  color=COLORS["gt"],      lw=2.0, label="Ground truth", zorder=5)
    ax.loglog(k_b,  E_b,  color=COLORS["bicubic"], lw=1.5, ls="--", label="Bicubic")
    ax.loglog(k_fn, E_fn, color=COLORS["fno"],     lw=1.5, ls=":",  label="FNO-only")
    ax.loglog(k_oi, E_oi, color=COLORS["oi"],      lw=2.0, ls="-.", label="Spectral OI")
    ax.loglog(k_ir, E_ir, color=COLORS["ir"],      lw=2.0, label="Iter. Refinement (ours)")

    # k^{-3} reference
    k_ref = np.array([10.0, 100.0])
    E_ref = 5e-3 * k_ref ** (-3)
    ax.loglog(k_ref, E_ref, "k--", lw=1.0, alpha=0.4, label=r"$k^{-3}$")

    # Nyquist of 32×32 obs at 256×256 grid
    ax.axvline(16, color="gray", lw=1.2, ls=":", alpha=0.7)
    ax.text(17, ax.get_ylim()[0] * 3, "$k_{obs}=16$", color="gray",
            fontsize=9, va="bottom")

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel(r"Energy $E(k)$")
    ax.set_title("Radial Energy Spectrum: OI vs Iterative Refinement")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(left=1)
    ax.grid(True, which="both", linestyle="--", alpha=0.25)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_spectrum_compare.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_spectrum_compare")


# ── spatial error maps ────────────────────────────────────────────────────────

def plot_error_fields(oi: torch.Tensor, ir: torch.Tensor,
                      truth: torch.Tensor, out_dir: str,
                      traj_idx: int = 0, t_idx: int = 50) -> None:
    """
    2×3 grid:
      row 0: full error  (OI−truth, IR−truth, truth field for reference)
      row 1: high-pass error (k≥16) — the part OI cannot correct
    """
    oi_t  = oi [traj_idx, t_idx]   # (256, 256)
    ir_t  = ir [traj_idx, t_idx]
    gt_t  = truth[traj_idx, t_idx]

    err_oi = oi_t - gt_t
    err_ir = ir_t - gt_t

    # high-pass: k >= 16
    def hp(x): return _spectral_bandpass(x.unsqueeze(0).unsqueeze(0), 16, 200).squeeze()

    hp_oi  = hp(err_oi)
    hp_ir  = hp(err_ir)
    hp_gt  = hp(gt_t)    # high-pass ground truth for reference

    vmax_err  = float(err_oi.abs().quantile(0.995))
    vmax_full = float(gt_t.abs().quantile(0.995))
    vmax_hp   = float(hp_gt.abs().quantile(0.995))

    titles_row0 = ["OI error  (full band)", "IR error  (full band)", "Ground truth  ω"]
    titles_row1 = ["OI error  ($k \\geq 16$ only)", "IR error  ($k \\geq 16$ only)",
                   "GT  ($k \\geq 16$ only)"]
    fields_row0 = [err_oi, err_ir, gt_t]
    fields_row1 = [hp_oi,  hp_ir,  hp_gt]
    vmaxes_row0 = [vmax_err, vmax_err, vmax_full]
    vmaxes_row1 = [vmax_hp,  vmax_hp,  vmax_hp]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    for col, (f, title, vm) in enumerate(zip(fields_row0, titles_row0, vmaxes_row0)):
        im = axes[0, col].imshow(f.numpy(), cmap="RdBu_r",
                                 vmin=-vm, vmax=vm, origin="lower")
        axes[0, col].set_title(title, fontsize=11)
        axes[0, col].axis("off")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

    for col, (f, title, vm) in enumerate(zip(fields_row1, titles_row1, vmaxes_row1)):
        im = axes[1, col].imshow(f.numpy(), cmap="RdBu_r",
                                 vmin=-vm, vmax=vm, origin="lower")
        axes[1, col].set_title(title, fontsize=11)
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Spatial error maps — trajectory {traj_idx}, t={t_idx}\n"
        "Bottom row isolates $k \\geq 16$ (unobserved scales): "
        "IR recovers fine-scale structure that OI leaves uncorrected.",
        fontsize=11
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_error_fields.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_error_fields")


# ── cumulative RMSE curves ────────────────────────────────────────────────────

def plot_rmse_curves(oi: torch.Tensor, ir: torch.Tensor,
                     fno: torch.Tensor, bicubic: torch.Tensor,
                     truth: torch.Tensor, out_dir: str) -> None:
    """RMSE vs time-step mean ± 1σ for all four methods at 256×256."""
    def traj_rmse(pred, tru):
        # pred, tru: (n_traj, T, ny, nx)
        err = (pred - tru) ** 2
        rmse = err.mean(dim=(-2, -1)).sqrt()   # (n_traj, T)
        return rmse.mean(0).numpy(), rmse.std(0).numpy()

    mean_oi,  std_oi  = traj_rmse(oi,      truth)
    mean_ir,  std_ir  = traj_rmse(ir,      truth)
    mean_fno, std_fno = traj_rmse(fno,     truth)
    mean_bic, std_bic = traj_rmse(bicubic, truth)

    T   = mean_oi.shape[0]
    ts  = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    def _plot(mean, std, color, label, ls="-"):
        ax.plot(ts, mean, color=color, lw=2.0, ls=ls, label=label)
        ax.fill_between(ts, mean - std, mean + std, color=color, alpha=0.12)

    _plot(mean_bic, std_bic, COLORS["bicubic"], "Bicubic",                  "--")
    _plot(mean_fno, std_fno, COLORS["fno"],     "FNO-only",                  ":")
    _plot(mean_oi,  std_oi,  COLORS["oi"],      "Spectral OI",               "-.")
    _plot(mean_ir,  std_ir,  COLORS["ir"],      "Iter. Refinement (ours)",   "-")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE (vorticity @ 256×256)")
    ax.set_title("RMSE over time — OI vs Iterative Refinement")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(linestyle="--", alpha=0.35)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig_rmse_curves.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_rmse_curves")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oi_results",     default="results_oi/inference_results.pt")
    parser.add_argument("--ir_results",     default="results_2d/inference_results.pt")
    parser.add_argument("--out_dir",        default="plots_oi")
    parser.add_argument("--traj_idx",       type=int, default=0)
    parser.add_argument("--t_idx",          type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading results...")
    ro = torch.load(args.oi_results, map_location="cpu")
    ri = torch.load(args.ir_results, map_location="cpu")

    T_common = min(ro["oi_256"].shape[1], ri["posterior_256"].shape[1])
    n_traj   = min(ro["oi_256"].shape[0], ri["posterior_256"].shape[0])

    oi_256   = ro["oi_256"     ][:n_traj, :T_common].float()
    fc_256   = ro["forecast_256"][:n_traj, :T_common].float()
    ir_256   = ri["posterior_256"][:n_traj, :T_common].float()
    truth    = ri["truth_256"  ][:n_traj, :T_common].float()
    obs_32   = ro["obs_32"     ][:n_traj, :T_common].float()

    # bicubic: upsample obs_32 → 256×256
    print("Computing bicubic baseline...")
    flat_obs = obs_32.reshape(-1, 1, 32, 32)
    bic_flat = _bicubic_up(flat_obs)                           # (N, 1, 256, 256)
    bicubic  = bic_flat.squeeze(1).reshape(n_traj, T_common, 256, 256)

    print(f"\nSummary RMSE (all trajs, all T):")
    print(f"  Bicubic:   {_rmse(bicubic, truth):.4f}")
    print(f"  FNO-only:  {_rmse(fc_256,  truth):.4f}")
    print(f"  Spectral OI: {_rmse(oi_256, truth):.4f}")
    print(f"  IR (ours): {_rmse(ir_256,  truth):.4f}")

    print("\nPlotting band RMSE...")
    plot_band_rmse(oi_256, ir_256, fc_256, bicubic, truth, args.out_dir)

    print("Plotting energy spectrum...")
    plot_spectrum_compare(oi_256, ir_256, fc_256, bicubic, truth, args.out_dir)

    print("Plotting error fields...")
    plot_error_fields(oi_256, ir_256, truth, args.out_dir,
                      traj_idx=args.traj_idx, t_idx=args.t_idx)

    print("Plotting RMSE curves...")
    plot_rmse_curves(oi_256, ir_256, fc_256, bicubic, truth, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
