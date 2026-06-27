"""
Plots for Ablation A1 (cascade depth) and A2 (propagation signal).

Loads results written by ablation_a1_cascade_depth.py and
ablation_a2_propagation_signal.py.  All A1 variants are read from tagged
subdirectories under results/; the 3-stage baseline falls back to results/
if ablation_a1_3stage/ does not exist.

Figures saved to plots_ablation_a1_a2/:
  fig1_a1_rmse_bars.{png,pdf}      — RMSE at 256 and 512 for 1/2/3-stage
  fig2_a1_rmse_time.{png,pdf}      — RMSE-over-time at 512, three depths
  fig3_a1_spectrum.{png,pdf}       — Energy spectrum at 512, three depths
  fig4_a2_rmse_bars.{png,pdf}      — RMSE at 128/256/512 for three A2 variants
  fig5_a2_rmse_time.{png,pdf}      — RMSE-over-time at 512, three A2 variants
  fig6_a2_spectrum.{png,pdf}       — Energy spectrum at 512, three A2 variants

Usage:
    python scripts/plot_ablation_a1_a2.py
        [--results_dir  results]

        [--data_dir     data]
        [--n_steps      N]
        [--traj         0]
        [--figures      1,2,3,4,5,6]
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

from src.evaluation.metrics import energy_spectrum, rmse_over_time, spectral_rmse

# ---------------------------------------------------------------------------
# Style — same as plot_comparison_1d.py
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

PLOTS_DIR = Path("plots_ablation_a1_a2")

# A1 colours
C_1STAGE = "#e377c2"   # pink — 1-stage (one-shot)
C_2STAGE = "#ff7f0e"   # orange — 2-stage
C_3STAGE = "#1f77b4"   # blue — 3-stage (full method)
C_GT     = "black"

# A2 colours
C_POST    = "#1f77b4"   # blue — posterior (current/best)
C_FORE    = "#ff7f0e"   # orange — FNO forecast
C_OBSRAW  = "#888888"   # gray — raw obs downsample


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOTS_DIR}/{name}.{{png,pdf}}")


def _rmse_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


def _per_traj_rmse_time(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """(n_traj, T, N) -> (n_traj, T) per-step RMSE."""
    n_traj = pred.shape[0]
    return torch.stack([
        rmse_over_time(pred[i], truth[i]) for i in range(n_traj)
    ]).numpy()


def _energy_spectrum_avg(x: torch.Tensor) -> np.ndarray:
    """(n_traj, T, N) -> (N//2+1,) average spectrum."""
    n, T, N = x.shape
    return energy_spectrum(x.reshape(n * T, N)).numpy()


# ---------------------------------------------------------------------------
# Figure 1 — A1: RMSE bar chart at 256 and 512
# ---------------------------------------------------------------------------

def plot_a1_rmse_bars(
    r_1stage: dict | None,
    r_2stage: dict,
    r_3stage: dict,
    truth_dict: dict[int, torch.Tensor],
) -> None:
    """Grouped bars: RMSE at N=256 and N=512 for each cascade depth."""
    resolutions = [256, 512]
    methods     = ["1-stage\n(one-shot)", "2-stage", "3-stage\n(full)"]
    colors      = [C_1STAGE, C_2STAGE, C_3STAGE]

    means = np.zeros((3, len(resolutions)))
    stds  = np.zeros((3, len(resolutions)))

    for res_idx, res in enumerate(resolutions):
        truth = truth_dict[res]
        n_traj = truth.shape[0]

        # 1-stage: only available at 512 (no 256 in one-shot)
        if r_1stage is not None and f"posterior_{res}" in r_1stage:
            post = r_1stage[f"posterior_{res}"][:n_traj, :truth.shape[1]]
            traj_rmse = np.array([_rmse_scalar(post[i], truth[i]) for i in range(n_traj)])
        else:
            traj_rmse = np.array([float("nan")])
        means[0, res_idx] = np.nanmean(traj_rmse)
        stds[ 0, res_idx] = np.nanstd(traj_rmse)

        # 2-stage: no 256 if we skipped 128→256; but the 2-stage goes 64→256 directly
        if f"posterior_{res}" in r_2stage:
            post = r_2stage[f"posterior_{res}"][:n_traj, :truth.shape[1]]
            traj_rmse = np.array([_rmse_scalar(post[i], truth[i]) for i in range(n_traj)])
        else:
            traj_rmse = np.array([float("nan")])
        means[1, res_idx] = np.nanmean(traj_rmse)
        stds[ 1, res_idx] = np.nanstd(traj_rmse)

        # 3-stage
        if f"posterior_{res}" in r_3stage:
            post = r_3stage[f"posterior_{res}"][:n_traj, :truth.shape[1]]
            traj_rmse = np.array([_rmse_scalar(post[i], truth[i]) for i in range(n_traj)])
        else:
            traj_rmse = np.array([float("nan")])
        means[2, res_idx] = np.nanmean(traj_rmse)
        stds[ 2, res_idx] = np.nanstd(traj_rmse)

    x       = np.arange(len(resolutions))
    width   = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m_idx, (method, color, offset) in enumerate(zip(methods, colors, offsets)):
        bars = ax.bar(x + offset, means[m_idx], width,
                      yerr=stds[m_idx], capsize=5,
                      label=method.replace("\n", " "),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, means[m_idx]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + stds[m_idx].max() * 0.03 + 1e-9,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={r}" for r in resolutions])
    ax.set_ylabel("RMSE")
    ax.set_title("A1 — Cascade depth: RMSE at 256 and 512")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig1_a1_rmse_bars")


# ---------------------------------------------------------------------------
# Figure 2 — A1: RMSE over time at 512
# ---------------------------------------------------------------------------

def plot_a1_rmse_time(
    r_1stage: dict | None,
    r_2stage: dict,
    r_3stage: dict,
    truth_512: torch.Tensor,
) -> None:
    T = truth_512.shape[1]
    t_ax = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot(r: dict | None, key: str, color: str, label: str, ls: str = "-") -> None:
        if r is None or key not in r:
            return
        data = _per_traj_rmse_time(r[key][:, :T], truth_512)
        mu   = data.mean(axis=0)
        sig  = data.std(axis=0)
        ax.plot(t_ax, mu, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(t_ax, mu - sig, mu + sig, color=color, alpha=0.15)

    _plot(r_1stage, "posterior_512", C_1STAGE, "1-stage (64→512)", ls="--")
    _plot(r_2stage, "posterior_512", C_2STAGE, "2-stage")
    _plot(r_3stage, "posterior_512", C_3STAGE, "3-stage (full)")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE  (512-pt)")
    ax.set_title("A1 — Cascade depth: RMSE over time at 512")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig2_a1_rmse_time")


# ---------------------------------------------------------------------------
# Figure 3 — A1: Per-mode spectral error at 512
# ---------------------------------------------------------------------------

def plot_a1_spectrum(
    r_1stage: dict | None,
    r_2stage: dict,
    r_3stage: dict,
    truth_512: torch.Tensor,
    k_max: int = 200,
) -> None:
    """Per-wavenumber spectral RMSE sqrt(mean(|FFT(pred-truth)|²)) at 512-pt.

    Uses spectral_rmse(pred, truth) which computes the RMS error of the
    *difference field* in Fourier space — capturing both phase and amplitude
    errors at each wavenumber. This is the correct metric here: the diffusion
    model always reproduces the marginal E(k) (|log E_pred/E_gt| ≈ 0 for all
    variants), but phase errors from coarser cascades appear clearly in the
    spectrum of the residual (pred - truth).
    """
    n_traj, T, N = truth_512.shape
    truth_flat = truth_512.reshape(n_traj * T, N)   # (n_traj*T, 512)
    k   = np.arange(N // 2 + 1)
    mask = (k >= 1) & (k <= k_max)
    k_p  = k[mask]

    def _sr(r: dict | None, key: str) -> np.ndarray | None:
        if r is None or key not in r:
            return None
        pred = r[key][:n_traj, :T].reshape(n_traj * T, N)
        return spectral_rmse(pred, truth_flat)[mask].numpy()

    sr_1 = _sr(r_1stage, "posterior_512")
    sr_2 = _sr(r_2stage, "posterior_512")
    sr_3 = _sr(r_3stage, "posterior_512")

    fig, ax = plt.subplots(figsize=(9, 5))

    if sr_1 is not None and sr_3 is not None:
        ax.fill_between(k_p, sr_3, sr_1,
                        where=(sr_1 >= sr_3),
                        color=C_3STAGE, alpha=0.10,
                        label="Gap: 3-stage improvement over 1-stage")

    if sr_1 is not None:
        ax.plot(k_p, sr_1, color=C_1STAGE, lw=2.0, ls="--",
                label=f"1-stage (64→512)   (mean={sr_1.mean():.4f})")
    if sr_2 is not None:
        ax.plot(k_p, sr_2, color=C_2STAGE, lw=2.0,
                label=f"2-stage             (mean={sr_2.mean():.4f})")
    if sr_3 is not None:
        ax.plot(k_p, sr_3, color=C_3STAGE, lw=2.0,
                label=f"3-stage (full)      (mean={sr_3.mean():.4f})")

    ax.axvline(32, color="gray", ls=":", lw=0.9, alpha=0.65)
    ax.text(33, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.01,
            "k=32", color="gray", fontsize=8, va="top")
    ax.set_xlabel("Wavenumber  k")
    ax.set_ylabel(r"Spectral RMSE  $\sqrt{\langle|\hat{\epsilon}_k|^2\rangle}$")
    ax.set_title(
        "A1 — Cascade depth: per-mode spectral RMSE at 512-pt\n"
        r"$\mathrm{SRMSE}(k) = \sqrt{\langle|\mathcal{F}(\hat{u}-u)_k|^2\rangle}$"
        "  —  lower is better"
    )
    ax.set_xlim(1, k_max)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.85, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig3_a1_spectrum")


# ---------------------------------------------------------------------------
# Figure 4 — A2: RMSE bar chart at 128, 256, 512
# ---------------------------------------------------------------------------

def plot_a2_rmse_bars(
    r_post: dict,
    r_fore: dict,
    r_obs:  dict,
    truth_dict: dict[int, torch.Tensor],
) -> None:
    resolutions = [128, 256, 512]
    methods     = ["(i) Posterior\n[current]", "(ii) FNO forecast", "(iii) Raw obs"]
    colors      = [C_POST, C_FORE, C_OBSRAW]
    results_all = [r_post, r_fore, r_obs]

    means = np.zeros((3, 3))
    stds  = np.zeros((3, 3))

    for res_idx, res in enumerate(resolutions):
        truth  = truth_dict[res]
        n_traj = truth.shape[0]
        for m_idx, r in enumerate(results_all):
            key = f"posterior_{res}"
            if key in r:
                post = r[key][:n_traj, :truth.shape[1]]
                vals = np.array([_rmse_scalar(post[i], truth[i]) for i in range(n_traj)])
            else:
                vals = np.array([float("nan")])
            means[m_idx, res_idx] = np.nanmean(vals)
            stds[ m_idx, res_idx] = np.nanstd(vals)

    x       = np.arange(3)
    width   = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m_idx, (method, color, offset) in enumerate(zip(methods, colors, offsets)):
        bars = ax.bar(x + offset, means[m_idx], width,
                      yerr=stds[m_idx], capsize=5,
                      label=method.replace("\n", " "),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, means[m_idx]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + stds[m_idx].max() * 0.03 + 1e-9,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={r}" for r in resolutions])
    ax.set_ylabel("RMSE")
    ax.set_title("A2 — Propagation signal: RMSE at each resolution")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig4_a2_rmse_bars")


# ---------------------------------------------------------------------------
# Figure 5 — A2: RMSE over time at 512
# ---------------------------------------------------------------------------

def plot_a2_rmse_time(
    r_post: dict,
    r_fore: dict,
    r_obs:  dict,
    truth_512: torch.Tensor,
) -> None:
    T    = truth_512.shape[1]
    t_ax = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot(r: dict, color: str, label: str, ls: str = "-") -> None:
        if "posterior_512" not in r:
            return
        data = _per_traj_rmse_time(r["posterior_512"][:, :T], truth_512)
        mu   = data.mean(axis=0)
        sig  = data.std(axis=0)
        ax.plot(t_ax, mu, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(t_ax, mu - sig, mu + sig, color=color, alpha=0.15)

    _plot(r_post, C_POST,   "(i) Posterior [current]")
    _plot(r_fore, C_FORE,   "(ii) FNO forecast",     ls="--")
    _plot(r_obs,  C_OBSRAW, "(iii) Raw obs downsample", ls=":")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE  (512-pt)")
    ax.set_title("A2 — Propagation signal: RMSE over time at 512")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig5_a2_rmse_time")


# ---------------------------------------------------------------------------
# Figure 6 — A2: Per-mode spectral RMSE at 512
# ---------------------------------------------------------------------------

def plot_a2_spectrum(
    r_post: dict,
    r_fore: dict,
    r_obs:  dict,
    truth_512: torch.Tensor,
    k_max: int = 200,
) -> None:
    """Per-wavenumber spectral RMSE sqrt(mean(|FFT(pred-truth)|²)) at 512-pt.

    Uses spectral_rmse(pred, truth) which operates on the *error field*
    (pred - truth) in Fourier space.  This captures phase errors that are
    invisible to |log10(E_pred/E_gt)| — the marginal spectrum E(k) is
    matched by the diffusion model for all three variants, but the spectrum
    of the residual reveals that the forecast and obs_raw variants have
    substantially worse phase alignment at high wavenumbers.
    """
    n_traj, T, N = truth_512.shape
    truth_flat = truth_512.reshape(n_traj * T, N)
    k    = np.arange(N // 2 + 1)
    mask = (k >= 1) & (k <= k_max)
    k_p  = k[mask]

    def _sr(r: dict) -> np.ndarray | None:
        if "posterior_512" not in r:
            return None
        pred = r["posterior_512"][:n_traj, :T].reshape(n_traj * T, N)
        return spectral_rmse(pred, truth_flat)[mask].numpy()

    sr_post = _sr(r_post)
    sr_fore = _sr(r_fore)
    sr_obs  = _sr(r_obs)

    fig, ax = plt.subplots(figsize=(9, 5))

    if sr_post is not None and sr_fore is not None:
        ax.fill_between(k_p, sr_post, sr_fore,
                        where=(sr_fore >= sr_post),
                        color=C_FORE, alpha=0.12,
                        label="Gap: forecast worse than posterior")

    if sr_post is not None:
        ax.plot(k_p, sr_post, color=C_POST,   lw=2.0,
                label=f"(i)  Posterior [current]  (mean={sr_post.mean():.4f})")
    if sr_fore is not None:
        ax.plot(k_p, sr_fore, color=C_FORE,   lw=2.0, ls="--",
                label=f"(ii) FNO forecast          (mean={sr_fore.mean():.4f})")
    if sr_obs is not None:
        ax.plot(k_p, sr_obs,  color=C_OBSRAW, lw=2.0, ls=":",
                label=f"(iii) Raw obs downsample   (mean={sr_obs.mean():.4f})")

    ax.axvline(32, color="gray", ls=":", lw=0.9, alpha=0.65)
    ax.text(33, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.01,
            "k=32", color="gray", fontsize=8, va="top")
    ax.set_xlabel("Wavenumber  k")
    ax.set_ylabel(r"Spectral RMSE  $\sqrt{\langle|\hat{\epsilon}_k|^2\rangle}$")
    ax.set_title(
        "A2 — Propagation signal: per-mode spectral RMSE at 512-pt\n"
        r"$\mathrm{SRMSE}(k) = \sqrt{\langle|\mathcal{F}(\hat{u}-u)_k|^2\rangle}$"
        "  —  lower is better"
    )
    ax.set_xlim(1, k_max)
    ax.set_ylim(bottom=0)
    ax.legend(framealpha=0.85, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig6_a2_spectrum")


# ---------------------------------------------------------------------------
# Console summary tables
# ---------------------------------------------------------------------------

def _print_a1_table(r_1: dict | None, r_2: dict, r_3: dict,
                    truth_dict: dict[int, torch.Tensor]) -> None:
    print("\n── A1 table ────────────────────────────────────────────────")
    print(f"  {'Variant':<22}  {'RMSE@256':>10}  {'RMSE@512':>10}")
    print(f"  {'─'*46}")
    for label, r, res_list in [
        ("1-stage (64→512)",  r_1, [512]),
        ("2-stage",           r_2, [256, 512]),
        ("3-stage (full)",    r_3, [256, 512]),
    ]:
        row = f"  {label:<22}"
        for res in [256, 512]:
            key = f"posterior_{res}"
            if r is not None and key in r and res in res_list:
                truth = truth_dict[res]
                post  = r[key][:truth.shape[0], :truth.shape[1]]
                row  += f"  {_rmse_scalar(post, truth):10.4f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)


def _print_a2_table(r_post: dict, r_fore: dict, r_obs: dict,
                    truth_dict: dict[int, torch.Tensor]) -> None:
    print("\n── A2 table ────────────────────────────────────────────────")
    print(f"  {'Variant':<28}  {'RMSE@128':>10}  {'RMSE@256':>10}  {'RMSE@512':>10}")
    print(f"  {'─'*64}")
    for label, r in [
        ("(i)  posterior [current]", r_post),
        ("(ii) forecast",            r_fore),
        ("(iii) obs_raw",            r_obs),
    ]:
        row = f"  {label:<28}"
        for res in [128, 256, 512]:
            key = f"posterior_{res}"
            if key in r:
                truth = truth_dict[res]
                post  = r[key][:truth.shape[0], :truth.shape[1]]
                row  += f"  {_rmse_scalar(post, truth):10.4f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)


# ---------------------------------------------------------------------------
# Arg parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ablations A1 and A2")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--data_dir",    type=str, default="data")
    p.add_argument("--n_steps",     type=int, default=None)
    p.add_argument("--traj",        type=int, default=0)
    p.add_argument("--figures",     type=str, default="1,2,3,4,5,6")
    return p.parse_args()


def _load(path: Path) -> dict | None:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    print(f"  [WARN] Not found: {path}")
    return None


def main() -> None:
    args = parse_args()
    figs = {int(x) for x in args.figures.split(",")}
    rdir = Path(args.results_dir)

    # ── Load all result files ─────────────────────────────────────────────────
    print("Loading results ...")
    r_1stage  = _load(rdir / "ablation_a1_1stage" / "inference_results.pt") or {}
    r_2stage  = _load(rdir / "ablation_a1_2stage" / "inference_results.pt") or {}
    r_3stage  = (_load(rdir / "ablation_a1_3stage" / "inference_results.pt")
                 or _load(rdir / "inference_results.pt") or {})
    r_post    = _load(rdir / "ablation_a2_posterior" / "inference_results.pt") or {}
    r_fore    = _load(rdir / "ablation_a2_forecast"  / "inference_results.pt") or {}
    r_obs     = _load(rdir / "ablation_a2_obs_raw"   / "inference_results.pt") or {}

    # ── Ground truth ─────────────────────────────────────────────────────────
    print(f"Loading ground truth from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )

    # Align T across all loaded results
    T_candidates = [test_data["u_512"].shape[1]]
    for r in [r_1stage, r_2stage, r_3stage, r_post, r_fore, r_obs]:
        if r and "posterior_512" in r:
            T_candidates.append(r["posterior_512"].shape[1])
    T = min(T_candidates)
    if args.n_steps is not None:
        T = min(T, args.n_steps)

    n_traj = test_data["u_512"].shape[0]
    truth_dict: dict[int, torch.Tensor] = {
        res: test_data[f"u_{res}"][:n_traj, :T].float()
        for res in [128, 256, 512]
    }
    truth_512 = truth_dict[512]

    # Trim all result dicts to (n_traj, T)
    def _trim(d: dict) -> dict:
        return {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                for k, v in d.items()}

    r_1stage = _trim(r_1stage) if r_1stage else None
    r_2stage = _trim(r_2stage)
    r_3stage = _trim(r_3stage)
    r_post   = _trim(r_post)
    r_fore   = _trim(r_fore)
    r_obs    = _trim(r_obs)

    print(f"Using n_traj={n_traj}, T={T}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if 1 in figs:
        print("\nFigure 1: A1 RMSE bars ...")
        plot_a1_rmse_bars(r_1stage, r_2stage, r_3stage, truth_dict)

    if 2 in figs:
        print("\nFigure 2: A1 RMSE over time ...")
        plot_a1_rmse_time(r_1stage, r_2stage, r_3stage, truth_512)

    if 3 in figs:
        print("\nFigure 3: A1 energy spectrum ...")
        plot_a1_spectrum(r_1stage, r_2stage, r_3stage, truth_512)

    if 4 in figs:
        print("\nFigure 4: A2 RMSE bars ...")
        plot_a2_rmse_bars(r_post, r_fore, r_obs, truth_dict)

    if 5 in figs:
        print("\nFigure 5: A2 RMSE over time ...")
        plot_a2_rmse_time(r_post, r_fore, r_obs, truth_512)

    if 6 in figs:
        print("\nFigure 6: A2 energy spectrum ...")
        plot_a2_spectrum(r_post, r_fore, r_obs, truth_512)

    _print_a1_table(r_1stage, r_2stage, r_3stage, truth_dict)
    _print_a2_table(r_post, r_fore, r_obs, truth_dict)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
