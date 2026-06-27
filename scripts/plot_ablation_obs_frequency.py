"""
Plots for the observation frequency ablation.

Loads results from results_2d/ablation_obs_freq/obs_freq_K{k}/ for each K
and produces four figures saved to plots_ablation_obs_freq/:

  fig1_rmse_vs_k.{png,pdf}
    — Bar chart: mean RMSE at 256×256 vs observation interval K.
      Shows the full-sequence RMSE, plus split bars for assimilation
      timesteps vs free-forecast timesteps.

  fig2_rmse_time_all_k.{png,pdf}
    — RMSE-over-time curves for all K values on one plot.
      Vertical dashed lines mark assimilation timesteps for the largest K
      shown, so the "sawtooth" pattern of forecast drift + correction is
      visible.

  fig3_rmse_time_panel.{png,pdf}
    — One subplot per K value showing RMSE over time with assimilation
      timesteps shaded.  Makes the drift-and-correct cycle explicit.

  fig4_snapshot_comparison.{png,pdf}
    — Side-by-side 256×256 vorticity fields at a free-forecast timestep
      (midway between two corrections) for K=1, K=3, K=5, K=10.
      Ground truth shown for reference.

Usage:
    python scripts/plot_ablation_obs_frequency.py
        [--results_dir  results_2d/ablation_obs_freq]
        [--data_dir     data_2d]
        [--k_values     1,2,3,5,10]
        [--n_steps      N]
        [--traj         0]
        [--figures      1,2,3,4]
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
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.size":       12,
    "axes.labelsize":  13,
    "axes.titlesize":  14,
    "legend.fontsize": 11,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})

PLOTS_DIR = Path("plots_ablation_obs_freq")

# Colour ramp from blue (K=1, dense) to red (K=10, sparse)
_K_COLORS = {
    1:  "#1f77b4",   # blue
    2:  "#4e9fd4",   # light blue
    3:  "#2ca02c",   # green
    5:  "#ff7f0e",   # orange
    10: "#d62728",   # red
}
CMAP_FIELD = "RdBu_r"


def _color(k: int) -> str:
    return _K_COLORS.get(k, "#888888")


def _save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: Path) -> dict | None:
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def _rmse_curve(r: dict, truth_all: torch.Tensor) -> np.ndarray:
    """(T,) RMSE at each step, averaged over trajectories."""
    if "rmse_curve" in r and isinstance(r["rmse_curve"], torch.Tensor):
        return r["rmse_curve"].numpy()
    # Recompute from tensors if not stored
    post  = r["posterior_256"]
    n     = min(post.shape[0], truth_all.shape[0])
    T     = min(post.shape[1], truth_all.shape[1])
    return (post[:n, :T] - truth_all[:n, :T]).pow(2).mean(dim=(0, 2, 3)).sqrt().numpy()


def _mean_rmse(r: dict, truth_all: torch.Tensor) -> float:
    if "metrics" in r and "rmse_full" in r["metrics"]:
        return float(r["metrics"]["rmse_full"])
    return float(_rmse_curve(r, truth_all).mean())


def _obs_mask(r: dict, T: int) -> np.ndarray:
    """Boolean mask of length T: True at assimilation timesteps."""
    if "obs_mask" in r:
        m = r["obs_mask"]
        return m[:T].numpy().astype(bool)
    # Reconstruct from metrics if not stored
    k = r["metrics"].get("k", 1) if "metrics" in r else 1
    return np.array([(t % k == 0) for t in range(T)], dtype=bool)


# ---------------------------------------------------------------------------
# Figure 1 — RMSE vs K bar chart
# ---------------------------------------------------------------------------

def plot_rmse_vs_k(
    data: dict[int, dict],
    truth_all: torch.Tensor,
    dt_per_step: float,
) -> None:
    """Bar chart of mean RMSE at 256×256 as a function of K.

    Three bar groups per K:
      - Full sequence RMSE
      - RMSE at assimilation timesteps only
      - RMSE at free-forecast timesteps only (blank for K=1)
    """
    k_vals    = sorted(data.keys())
    x         = np.arange(len(k_vals))
    width     = 0.25
    offsets   = [-width, 0, width]

    rmse_full  = []
    rmse_obs   = []
    rmse_free  = []

    for k in k_vals:
        r = data[k]
        m = r.get("metrics", {})
        rmse_full.append(m.get("rmse_full",  _mean_rmse(r, truth_all)))
        rmse_obs.append( m.get("rmse_at_obs",  float("nan")))
        rmse_free.append(m.get("rmse_at_free", float("nan")))

    fig, ax = plt.subplots(figsize=(9, 5))

    bars_full = ax.bar(x + offsets[0], rmse_full, width,
                       color=[_color(k) for k in k_vals], alpha=0.9,
                       label="Full sequence")
    bars_obs  = ax.bar(x + offsets[1], rmse_obs,  width,
                       color=[_color(k) for k in k_vals], alpha=0.55,
                       hatch="//", label="At assimilation steps")
    bars_free = ax.bar(x + offsets[2], rmse_free, width,
                       color=[_color(k) for k in k_vals], alpha=0.35,
                       hatch="xx", label="At free-forecast steps")

    # Annotate bar heights
    for bar, val in zip(bars_full, rmse_full):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}\n({k*dt_per_step:.2f} t.u.)" for k in k_vals])
    ax.set_xlabel("Observation interval K  (time units per correction)")
    ax.set_ylabel("RMSE @ 256×256")
    ax.set_title(
        "Observation frequency ablation: RMSE vs K\n"
        "K=1 = standard (every step); larger K = sparser observations"
    )
    ax.legend(loc="upper left", framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig1_rmse_vs_k")


# ---------------------------------------------------------------------------
# Figure 2 — RMSE over time, all K on one plot
# ---------------------------------------------------------------------------

def plot_rmse_time_all_k(
    data: dict[int, dict],
    truth_all: torch.Tensor,
    dt_per_step: float,
) -> None:
    """All K values on one RMSE-over-time plot.

    Vertical dashed lines mark assimilation timesteps for the largest K,
    making the drift-and-correct cycle visible.
    """
    k_vals = sorted(data.keys())
    T_min  = min(
        min(data[k]["posterior_256"].shape[1] for k in k_vals),
        truth_all.shape[1],
    )
    t_ax = np.arange(T_min) * dt_per_step   # physical time axis

    fig, ax = plt.subplots(figsize=(11, 5))

    # Shade free-forecast windows for largest K
    k_max = max(k_vals)
    if k_max > 1:
        mask = _obs_mask(data[k_max], T_min)
        in_free = False
        free_start = 0
        for t in range(T_min):
            if not mask[t] and not in_free:
                free_start = t
                in_free = True
            elif mask[t] and in_free:
                ax.axvspan(t_ax[free_start], t_ax[t],
                           color=_color(k_max), alpha=0.06, linewidth=0)
                in_free = False
        if in_free:
            ax.axvspan(t_ax[free_start], t_ax[T_min - 1],
                       color=_color(k_max), alpha=0.06, linewidth=0)

    for k in k_vals:
        curve = _rmse_curve(data[k], truth_all)[:T_min]
        ax.plot(t_ax, curve, color=_color(k), lw=2.0,
                ls="--" if k == 1 else "-",
                label=f"K={k}  ({k*dt_per_step:.2f} t.u.)")

    ax.set_xlabel("Physical time  (t.u.)")
    ax.set_ylabel("RMSE @ 256×256")
    ax.set_title(
        "Observation frequency: RMSE over time\n"
        f"Shaded regions = free-forecast intervals (K={k_max})"
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig2_rmse_time_all_k")


# ---------------------------------------------------------------------------
# Figure 3 — Per-K panel: RMSE over time with assimilation shading
# ---------------------------------------------------------------------------

def plot_rmse_time_panel(
    data: dict[int, dict],
    truth_all: torch.Tensor,
    dt_per_step: float,
) -> None:
    """One subplot per K, showing RMSE over time with free-forecast windows shaded."""
    k_vals = sorted(data.keys())
    n_k    = len(k_vals)
    T_min  = min(
        min(data[k]["posterior_256"].shape[1] for k in k_vals),
        truth_all.shape[1],
    )
    t_ax = np.arange(T_min) * dt_per_step

    fig, axes = plt.subplots(n_k, 1, figsize=(11, 2.5 * n_k), sharex=True, sharey=False)
    if n_k == 1:
        axes = [axes]

    # Global y-max for shared visual context
    y_max = max(
        float(_rmse_curve(data[k], truth_all)[:T_min].max())
        for k in k_vals
    ) * 1.15

    for ax, k in zip(axes, k_vals):
        curve = _rmse_curve(data[k], truth_all)[:T_min]
        mask  = _obs_mask(data[k], T_min)
        color = _color(k)

        # Shade free-forecast windows
        in_free = False
        free_start = 0
        for t in range(T_min):
            if not mask[t] and not in_free:
                free_start = t
                in_free = True
            elif mask[t] and in_free:
                ax.axvspan(t_ax[free_start], t_ax[t],
                           color=color, alpha=0.12, linewidth=0)
                in_free = False
        if in_free:
            ax.axvspan(t_ax[free_start], t_ax[-1],
                       color=color, alpha=0.12, linewidth=0)

        # Mark assimilation timesteps with small ticks
        obs_times = t_ax[mask]
        ax.vlines(obs_times, 0, y_max * 0.05,
                  color=color, alpha=0.5, lw=0.8)

        ax.plot(t_ax, curve, color=color, lw=1.8)
        mean_rmse = float(curve.mean())
        ax.set_ylabel(f"RMSE\nK={k}", fontsize=10)
        ax.set_ylim(0, y_max)
        ax.text(0.99, 0.92, f"mean={mean_rmse:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

        # Annotate shading legend once
        if k == k_vals[0]:
            ax.set_title(
                "Observation frequency: RMSE over time per K\n"
                "Shaded = free-forecast windows;  ticks = assimilation steps"
            )

    axes[-1].set_xlabel("Physical time  (t.u.)")
    fig.tight_layout(h_pad=0.4)
    _save(fig, "fig3_rmse_time_panel")


# ---------------------------------------------------------------------------
# Figure 4 — Vorticity snapshot at a free-forecast midpoint
# ---------------------------------------------------------------------------

def plot_snapshot_comparison(
    data: dict[int, dict],
    truth_all: torch.Tensor,
    traj: int = 0,
) -> None:
    """Show 256×256 vorticity at a free-forecast midpoint for each K.

    For K>1, picks a timestep midway between two consecutive assimilation
    steps so the effect of forecast drift is maximally visible.
    For K=1, picks an arbitrary representative timestep.
    """
    k_vals      = sorted(data.keys())
    show_k_vals = [k for k in k_vals]   # show all
    n_panels    = 1 + len(show_k_vals)  # GT + one per K

    T_min = min(
        min(data[k]["posterior_256"].shape[1] for k in k_vals),
        truth_all.shape[1],
    )

    def _pick_t(k: int) -> int:
        """Pick a representative timestep: midpoint of first free-forecast window."""
        if k == 1:
            return T_min // 2
        # Find first free-forecast window midpoint
        mask = _obs_mask(data[k], T_min)
        for t in range(1, T_min):
            if not mask[t]:
                # t is free; find next obs
                t_next_obs = t
                while t_next_obs < T_min and not mask[t_next_obs]:
                    t_next_obs += 1
                mid = (t + t_next_obs) // 2
                return min(mid, T_min - 1)
        return T_min // 2

    fig = plt.figure(figsize=(4.5 * n_panels, 5))
    gs  = gridspec.GridSpec(1, n_panels, figure=fig,
                            wspace=0.05, left=0.03, right=0.91,
                            top=0.88, bottom=0.05)

    t_repr = _pick_t(max(k_vals))   # use the same timestep for all panels

    gt_field = truth_all[traj, t_repr].numpy()
    vmax = max(float(np.percentile(np.abs(gt_field), 99.5)), 1e-6)

    # Ground truth
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(gt_field, cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
               origin="lower", aspect="equal", interpolation="nearest")
    ax0.set_title("Ground Truth", fontsize=11)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_ylabel(f"t = {t_repr}", fontsize=9, labelpad=3)

    im_ref = None
    for col, k in enumerate(show_k_vals, start=1):
        r = data[k]
        post = r["posterior_256"]
        if post.shape[1] <= t_repr or post.shape[0] <= traj:
            continue
        field = post[traj, t_repr].numpy()
        mask  = _obs_mask(r, T_min)
        is_free = not bool(mask[t_repr])

        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(field, cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                       origin="lower", aspect="equal", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        step_label = "free-forecast" if is_free else "post-assimilation"
        ax.set_title(f"K={k}  ({step_label})", fontsize=10,
                     color=_color(k))
        # Border colour indicates free vs obs
        for spine in ax.spines.values():
            spine.set_edgecolor(_color(k))
            spine.set_linewidth(2)
        im_ref = im

    # Shared colorbar
    cbar_ax = fig.add_axes([0.925, 0.10, 0.012, 0.75])
    if im_ref is not None:
        fig.colorbar(im_ref, cax=cbar_ax, label="Vorticity  ω")

    fig.suptitle(
        f"256×256 Vorticity at t={t_repr} (free-forecast midpoint for largest K)\n"
        "Coloured borders = K colour; denser obs → closer to truth",
        fontsize=12, y=0.97,
    )
    _save(fig, "fig4_snapshot_comparison")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def _print_summary(data: dict[int, dict], truth_all: torch.Tensor,
                   dt_per_step: float) -> None:
    print("\n── Observation frequency summary ───────────────────────────────")
    print(f"  {'K':>4}  {'Δt (t.u.)':>10}  {'RMSE (full)':>12}  "
          f"{'RMSE @ obs':>12}  {'RMSE @ free':>12}")
    print(f"  {'─'*56}")
    for k in sorted(data.keys()):
        r = data[k]
        m = r.get("metrics", {})
        rf  = m.get("rmse_full",     _mean_rmse(r, truth_all))
        ro  = m.get("rmse_at_obs",   float("nan"))
        rfr = m.get("rmse_at_free",  float("nan"))
        dt  = k * dt_per_step
        print(f"  {k:>4}  {dt:>10.3f}  {rf:>12.4f}  {ro:>12.4f}  {rfr:>12.4f}")


# ---------------------------------------------------------------------------
# Arg parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot observation-frequency ablation"
    )
    p.add_argument("--results_dir", type=str,
                   default="results_2d/ablation_obs_freq")
    p.add_argument("--data_dir",    type=str, default="data_2d")
    p.add_argument("--k_values",    type=str, default="1,2,3,5,10")
    p.add_argument("--n_steps",     type=int, default=None)
    p.add_argument("--traj",        type=int, default=0)
    p.add_argument("--figures",     type=str, default="1,2,3,4")
    p.add_argument("--dt_per_step", type=float, default=0.05,
                   help="Physical time per saved step (dt * save_every, default 0.05)")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    figs     = {int(x) for x in args.figures.split(",")}
    k_values = [int(x) for x in args.k_values.split(",")]
    rdir     = Path(args.results_dir)

    # ── Load results ──────────────────────────────────────────────────────────
    print("Loading results ...")
    data: dict[int, dict] = {}
    for k in k_values:
        path = rdir / f"obs_freq_K{k}" / "inference_results.pt"
        r    = _load(path)
        if r is not None:
            data[k] = r
            print(f"  K={k}: loaded {path}")
        else:
            print(f"  K={k}: NOT FOUND at {path} — skipping")

    if not data:
        print("No results found. Run ablation_obs_frequency.py first.")
        return

    # ── Load ground truth ────────────────────────────────────────────────────
    print(f"Loading ground truth from {args.data_dir}/test.pt ...")
    test_data = torch.load(
        Path(args.data_dir) / "test.pt",
        map_location="cpu", weights_only=True,
    )

    # Align to shortest T across all loaded results
    T_candidates = [test_data["w_256"].shape[1]]
    for r in data.values():
        if "posterior_256" in r:
            T_candidates.append(r["posterior_256"].shape[1])
    T = min(T_candidates)
    if args.n_steps is not None:
        T = min(T, args.n_steps)

    n_traj    = min(test_data["w_256"].shape[0],
                    min(r["posterior_256"].shape[0] for r in data.values()))
    truth_all = test_data["w_256"][:n_traj, :T].float()

    def _trim(d: dict) -> dict:
        return {k: (v[:n_traj, :T] if isinstance(v, torch.Tensor) and v.dim() >= 2 else v)
                for k, v in d.items()}

    data = {k: _trim(r) for k, r in data.items()}
    print(f"Using n_traj={n_traj}, T={T}")

    dt = args.dt_per_step

    # ── Figures ───────────────────────────────────────────────────────────────
    if 1 in figs:
        print("\nFigure 1: RMSE vs K bar chart ...")
        plot_rmse_vs_k(data, truth_all, dt)

    if 2 in figs:
        print("\nFigure 2: RMSE over time, all K ...")
        plot_rmse_time_all_k(data, truth_all, dt)

    if 3 in figs:
        print("\nFigure 3: RMSE over time panel ...")
        plot_rmse_time_panel(data, truth_all, dt)

    if 4 in figs:
        print("\nFigure 4: Snapshot comparison ...")
        plot_snapshot_comparison(data, truth_all, traj=args.traj)

    _print_summary(data, truth_all, dt)
    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
