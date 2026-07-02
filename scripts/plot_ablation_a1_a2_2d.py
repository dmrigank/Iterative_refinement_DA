"""
Plots for Ablation A1 and A2 (2D Kraichnan turbulence).

Reads results from:
  results_2d/ablation_a1_{1,2,3}stage/inference_results.pt
  results_2d/ablation_a2_{posterior,forecast,obs_raw}/inference_results.pt

Figures saved to plots_ablation_a1_a2_2d/:
  fig1_a1_rmse_bars.{png,pdf}   — RMSE bars at 128 and 256
  fig2_a1_rmse_time.{png,pdf}   — RMSE over time at 256
  fig3_a1_spectrum.{png,pdf}    — Radial energy spectrum at 256 (three depths)
  fig4_a2_rmse_bars.{png,pdf}   — RMSE bars at 64/128/256 for three variants
  fig5_a2_rmse_time.{png,pdf}   — RMSE over time at 256
  fig6_a2_spectrum.{png,pdf}    — Radial energy spectrum at 256 (three variants)

Usage:
    python scripts/plot_ablation_a1_a2_2d.py
        [--results_dir  results_2d]
        [--data_dir     data_2d]
        [--n_steps      N]
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

from src.evaluation.metrics_2d import radial_energy_spectrum, rmse_over_time_2d

# ---------------------------------------------------------------------------
# Style
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

PLOTS_DIR = Path("plots_ablation_a1_a2_2d")

# A1 colours
C_1STAGE = "#e377c2"   # pink — 1-stage (one-shot)
C_2STAGE = "#ff7f0e"   # orange — 2-stage
C_3STAGE = "#1f77b4"   # blue — 3-stage (full)
C_GT     = "black"

# A2 colours
C_POST   = "#1f77b4"   # blue — posterior (current/best)
C_FORE   = "#ff7f0e"   # orange — FNO forecast
C_OBSRAW = "#888888"   # gray — raw obs


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOTS_DIR}/{name}.{{png,pdf}}")


def _rmse_scalar(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return float((pred - truth).pow(2).mean().sqrt())


def _per_traj_rmse_time(pred: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    """(n_traj, T, ny, nx) -> (n_traj, T)."""
    n_traj = pred.shape[0]
    rows = [rmse_over_time_2d(pred[i], truth[i]).numpy() for i in range(n_traj)]
    return np.stack(rows, axis=0)


def _avg_radial_spectrum(x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(n_traj, T, ny, nx) -> averaged E(k), k arrays."""
    n, T, ny, nx = x.shape
    flat = x.reshape(n * T, ny, nx)
    E, k = radial_energy_spectrum(flat)
    return E.numpy(), k.numpy()


# ---------------------------------------------------------------------------
# Figure 1 — A1: RMSE bar chart at 128 and 256
# ---------------------------------------------------------------------------

def plot_a1_rmse_bars(
    r_1stage: dict | None,
    r_2stage: dict,
    r_3stage: dict,
    truth_dict: dict[int, torch.Tensor],
) -> None:
    resolutions = [128, 256]
    methods     = ["1-stage\n(one-shot)", "2-stage", "3-stage\n(full)"]
    colors      = [C_1STAGE, C_2STAGE, C_3STAGE]
    results_all = [r_1stage, r_2stage, r_3stage]

    means = np.full((3, 2), np.nan)
    stds  = np.full((3, 2), np.nan)

    for ri, res in enumerate(resolutions):
        truth  = truth_dict[res]
        n_traj = truth.shape[0]
        for mi, r in enumerate(results_all):
            key = f"posterior_{res}"
            if r is not None and key in r:
                post = r[key][:n_traj, :truth.shape[1]]
                vals = np.array([_rmse_scalar(post[i], truth[i]) for i in range(n_traj)])
                means[mi, ri] = np.nanmean(vals)
                stds[ mi, ri] = np.nanstd(vals)

    x       = np.arange(len(resolutions))
    width   = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mi, (method, color, off) in enumerate(zip(methods, colors, offsets)):
        valid = ~np.isnan(means[mi])
        if not valid.any():
            continue
        bars = ax.bar(x[valid] + off, means[mi][valid], width,
                      yerr=stds[mi][valid], capsize=5,
                      label=method.replace("\n", " "),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, means[mi][valid]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + np.nanmax(stds[mi]) * 0.03 + 1e-9,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}×{r}" for r in resolutions])
    ax.set_ylabel("RMSE")
    ax.set_title("A1 — Cascade depth: RMSE at 128×128 and 256×256")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig1_a1_rmse_bars")


# ---------------------------------------------------------------------------
# Figure 2 — A1: RMSE over time at 256
# ---------------------------------------------------------------------------

def plot_a1_rmse_time(
    r_1stage: dict | None,
    r_2stage: dict,
    r_3stage: dict,
    truth_256: torch.Tensor,
) -> None:
    T    = truth_256.shape[1]
    t_ax = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot(r: dict | None, color: str, label: str, ls: str = "-") -> None:
        key = "posterior_256"
        if r is None or key not in r:
            return
        data = _per_traj_rmse_time(r[key][:, :T], truth_256)
        mu, sig = data.mean(0), data.std(0)
        ax.plot(t_ax, mu, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(t_ax, mu - sig, mu + sig, color=color, alpha=0.15)

    _plot(r_1stage, C_1STAGE, "1-stage (32→256)", ls="--")
    _plot(r_2stage, C_2STAGE, "2-stage (32→128→256)")
    _plot(r_3stage, C_3STAGE, "3-stage (full)")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE  (256×256)")
    ax.set_title("A1 — Cascade depth: RMSE over time at 256×256")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig2_a1_rmse_time")


# ---------------------------------------------------------------------------
# Figure 3 — A1: Radial energy spectrum at 256
# ---------------------------------------------------------------------------

def plot_a1_spectrum(
    r_1stage: dict | None,
    r_2stage: dict,
    r_3stage: dict,
    truth_256: torch.Tensor,
    k_max_plot: int = 90,
    k_nyq: int = 16,
) -> None:
    E_gt, k_gt = _avg_radial_spectrum(truth_256)
    k_arr = k_gt[1:]
    mask  = (k_arr >= 1) & (k_arr <= k_max_plot)

    def _E(r: dict | None, key: str = "posterior_256") -> np.ndarray | None:
        if r is None or key not in r:
            return None
        E, _ = _avg_radial_spectrum(r[key][:, :truth_256.shape[1]])
        return E[1:][mask]

    E1  = _E(r_1stage)
    E2  = _E(r_2stage)
    E3  = _E(r_3stage)
    Egt = E_gt[1:][mask]
    kp  = k_arr[mask]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.loglog(kp, Egt, color=C_GT,     lw=2.0, label="Ground truth")
    if E1 is not None:
        ax.loglog(kp, E1, color=C_1STAGE, lw=1.8, ls="--", label="1-stage (32→256)")
    if E2 is not None:
        ax.loglog(kp, E2, color=C_2STAGE, lw=1.8,           label="2-stage")
    if E3 is not None:
        ax.loglog(kp, E3, color=C_3STAGE, lw=1.8,           label="3-stage (full)")

    # k^{-3} reference anchored at k=5
    idx_anc = np.searchsorted(kp, 5)
    k_ref   = np.array([3.0, float(k_max_plot)])
    E_ref   = Egt[idx_anc] * (float(kp[idx_anc]) ** 3) * k_ref ** (-3)
    ax.loglog(k_ref, E_ref, color="gray", lw=1.0, ls=":", label=r"$k^{-3}$ reference")

    ax.axvline(k_nyq, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(k_nyq * 1.05, Egt.max() * 0.4, "32×32\nNyquist",
            color="gray", fontsize=9, va="top")

    ax.set_xlabel("Wavenumber  $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("A1 — Cascade depth: radial energy spectrum at 256×256")
    ax.set_xlim(left=1, right=k_max_plot * 1.1)
    ax.legend(framealpha=0.85)
    ax.grid(True, which="both", ls="--", alpha=0.2)
    fig.tight_layout()
    _save(fig, "fig3_a1_spectrum")


# ---------------------------------------------------------------------------
# Figure 4 — A2: RMSE bar chart at 64, 128, 256
# ---------------------------------------------------------------------------

def plot_a2_rmse_bars(
    r_post: dict,
    r_fore: dict,
    r_obs:  dict,
    truth_dict: dict[int, torch.Tensor],
) -> None:
    resolutions = [64, 128, 256]
    methods     = ["(i) Posterior\n[current]", "(ii) FNO forecast", "(iii) Raw obs"]
    colors      = [C_POST, C_FORE, C_OBSRAW]
    results_all = [r_post, r_fore, r_obs]

    means = np.zeros((3, 3))
    stds  = np.zeros((3, 3))

    for ri, res in enumerate(resolutions):
        truth  = truth_dict[res]
        n_traj = truth.shape[0]
        for mi, r in enumerate(results_all):
            key = f"posterior_{res}"
            if key in r:
                post = r[key][:n_traj, :truth.shape[1]]
                vals = np.array([_rmse_scalar(post[i], truth[i]) for i in range(n_traj)])
                means[mi, ri] = np.nanmean(vals)
                stds[ mi, ri] = np.nanstd(vals)
            else:
                means[mi, ri] = np.nan
                stds[ mi, ri] = np.nan

    x       = np.arange(len(resolutions))
    width   = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mi, (method, color, off) in enumerate(zip(methods, colors, offsets)):
        valid = ~np.isnan(means[mi])
        if not valid.any():
            continue
        bars = ax.bar(x[valid] + off, means[mi][valid], width,
                      yerr=stds[mi][valid], capsize=5,
                      label=method.replace("\n", " "),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, means[mi][valid]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + np.nanmax(stds[mi]) * 0.03 + 1e-9,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}×{r}" for r in resolutions])
    ax.set_ylabel("RMSE")
    ax.set_title("A2 — Propagation signal: RMSE at each resolution")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig4_a2_rmse_bars")


# ---------------------------------------------------------------------------
# Figure 5 — A2: RMSE over time at 256
# ---------------------------------------------------------------------------

def plot_a2_rmse_time(
    r_post: dict,
    r_fore: dict,
    r_obs:  dict,
    truth_256: torch.Tensor,
) -> None:
    T    = truth_256.shape[1]
    t_ax = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot(r: dict, color: str, label: str, ls: str = "-") -> None:
        key = "posterior_256"
        if key not in r:
            return
        data = _per_traj_rmse_time(r[key][:, :T], truth_256)
        mu, sig = data.mean(0), data.std(0)
        ax.plot(t_ax, mu, color=color, lw=2, ls=ls, label=label)
        ax.fill_between(t_ax, mu - sig, mu + sig, color=color, alpha=0.15)

    _plot(r_post, C_POST,   "(i) Posterior [current]")
    _plot(r_fore, C_FORE,   "(ii) FNO forecast",        ls="--")
    _plot(r_obs,  C_OBSRAW, "(iii) Raw obs downsample", ls=":")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE  (256×256)")
    ax.set_title("A2 — Propagation signal: RMSE over time at 256×256")
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig5_a2_rmse_time")


# ---------------------------------------------------------------------------
# Figure 6 — A2: Radial energy spectrum at 256
# ---------------------------------------------------------------------------

def plot_a2_spectrum(
    r_post: dict,
    r_fore: dict,
    r_obs:  dict,
    truth_256: torch.Tensor,
    k_max_plot: int = 90,
    k_nyq: int = 16,
) -> None:
    E_gt, k_gt = _avg_radial_spectrum(truth_256)
    k_arr = k_gt[1:]
    mask  = (k_arr >= 1) & (k_arr <= k_max_plot)
    kp    = k_arr[mask]
    Egt   = E_gt[1:][mask]

    def _E(r: dict, key: str = "posterior_256") -> np.ndarray | None:
        if key not in r:
            return None
        E, _ = _avg_radial_spectrum(r[key][:, :truth_256.shape[1]])
        return E[1:][mask]

    Ep = _E(r_post)
    Ef = _E(r_fore)
    Eo = _E(r_obs)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.loglog(kp, Egt, color=C_GT,   lw=2.0, label="Ground truth")
    if Ep is not None:
        ax.loglog(kp, Ep, color=C_POST,   lw=1.8,           label="(i) Posterior [current]")
    if Ef is not None:
        ax.loglog(kp, Ef, color=C_FORE,   lw=1.8, ls="--",  label="(ii) FNO forecast")
    if Eo is not None:
        ax.loglog(kp, Eo, color=C_OBSRAW, lw=1.8, ls=":",   label="(iii) Raw obs downsample")

    idx_anc = np.searchsorted(kp, 5)
    k_ref   = np.array([3.0, float(k_max_plot)])
    E_ref   = Egt[idx_anc] * (float(kp[idx_anc]) ** 3) * k_ref ** (-3)
    ax.loglog(k_ref, E_ref, color="gray", lw=1.0, ls=":", label=r"$k^{-3}$ reference")

    ax.axvline(k_nyq, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(k_nyq * 1.05, Egt.max() * 0.4, "32×32\nNyquist",
            color="gray", fontsize=9, va="top")

    ax.set_xlabel("Wavenumber  $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("A2 — Propagation signal: radial energy spectrum at 256×256")
    ax.set_xlim(left=1, right=k_max_plot * 1.1)
    ax.legend(framealpha=0.85)
    ax.grid(True, which="both", ls="--", alpha=0.2)
    fig.tight_layout()
    _save(fig, "fig6_a2_spectrum")


# ---------------------------------------------------------------------------
# Console summary tables
# ---------------------------------------------------------------------------

def _print_a1_table(r1, r2, r3, truth_dict) -> None:
    print("\n── A1 table (2D) ─────────────────────────────────────────────")
    print(f"  {'Variant':<28}  {'RMSE@128':>10}  {'RMSE@256':>10}")
    print(f"  {'─'*54}")
    for label, r, res_list in [
        ("1-stage (32→256)",      r1, [256]),
        ("2-stage (32→128→256)",  r2, [128, 256]),
        ("3-stage (full)",        r3, [128, 256]),
    ]:
        row = f"  {label:<28}"
        for res in [128, 256]:
            key = f"posterior_{res}"
            if r is not None and key in r and res in res_list:
                post  = r[key][:truth_dict[res].shape[0], :truth_dict[res].shape[1]]
                row  += f"  {_rmse_scalar(post, truth_dict[res]):10.4f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)


def _print_a2_table(r_post, r_fore, r_obs, truth_dict) -> None:
    print("\n── A2 table (2D) ─────────────────────────────────────────────")
    print(f"  {'Variant':<28}  {'RMSE@64':>10}  {'RMSE@128':>10}  {'RMSE@256':>10}")
    print(f"  {'─'*64}")
    for label, r in [
        ("(i)  posterior [current]", r_post),
        ("(ii) forecast",            r_fore),
        ("(iii) obs_raw",            r_obs),
    ]:
        row = f"  {label:<28}"
        for res in [64, 128, 256]:
            key = f"posterior_{res}"
            if key in r:
                post  = r[key][:truth_dict[res].shape[0], :truth_dict[res].shape[1]]
                row  += f"  {_rmse_scalar(post, truth_dict[res]):10.4f}"
            else:
                row += f"  {'N/A':>10}"
        print(row)


# ---------------------------------------------------------------------------
# Arg parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ablations A1+A2 (2D)")
    p.add_argument("--results_dir", type=str, default="results_2d_v2")
    p.add_argument("--data_dir",    type=str, default="data_2d")
    p.add_argument("--n_steps",     type=int, default=None)
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

    print("Loading results ...")
    r_1stage = _load(rdir / "ablation_a1_1stage" / "inference_results.pt") or {}
    r_2stage = _load(rdir / "ablation_a1_2stage" / "inference_results.pt") or {}
    # 3-stage is the full pipeline — prefer the main results file (better sample
    # quality from the original run) over the ablation re-run.
    r_3stage = (_load(rdir / "inference_results.pt")
                or _load(rdir / "ablation_a1_3stage" / "inference_results.pt") or {})
    # posterior propagation IS the full pipeline — prefer main results file
    r_post   = (_load(rdir / "inference_results.pt")
                or _load(rdir / "ablation_a2_posterior" / "inference_results.pt") or {})
    r_fore   = _load(rdir / "ablation_a2_forecast"  / "inference_results.pt") or {}
    r_obs    = _load(rdir / "ablation_a2_obs_raw"   / "inference_results.pt") or {}

    print(f"Loading ground truth from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt", map_location="cpu", weights_only=True
    )

    # Determine T
    T_candidates = [test_data["w_256"].shape[1]]
    for r in [r_1stage, r_2stage, r_3stage, r_post, r_fore, r_obs]:
        if r and "posterior_256" in r:
            T_candidates.append(r["posterior_256"].shape[1])
    T = min(T_candidates)
    if args.n_steps is not None:
        T = min(T, args.n_steps)

    n_traj = test_data["w_256"].shape[0]
    truth_dict: dict[int, torch.Tensor] = {
        res: test_data[f"w_{res}"][:n_traj, :T].float()
        for res in [64, 128, 256]
    }
    truth_256 = truth_dict[256]

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

    if 1 in figs:
        print("\nFigure 1: A1 RMSE bars ...")
        plot_a1_rmse_bars(r_1stage, r_2stage, r_3stage, truth_dict)

    if 2 in figs:
        print("\nFigure 2: A1 RMSE over time ...")
        plot_a1_rmse_time(r_1stage, r_2stage, r_3stage, truth_256)

    if 3 in figs:
        print("\nFigure 3: A1 energy spectrum ...")
        plot_a1_spectrum(r_1stage, r_2stage, r_3stage, truth_256)

    if 4 in figs:
        print("\nFigure 4: A2 RMSE bars ...")
        plot_a2_rmse_bars(r_post, r_fore, r_obs, truth_dict)

    if 5 in figs:
        print("\nFigure 5: A2 RMSE over time ...")
        plot_a2_rmse_time(r_post, r_fore, r_obs, truth_256)

    if 6 in figs:
        print("\nFigure 6: A2 energy spectrum ...")
        plot_a2_spectrum(r_post, r_fore, r_obs, truth_256)

    _print_a1_table(r_1stage, r_2stage, r_3stage, truth_dict)
    _print_a2_table(r_post, r_fore, r_obs, truth_dict)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
