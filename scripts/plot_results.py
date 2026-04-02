"""
Entry point: generate all result figures.

Reads inference outputs from results/inference_results.pt and produces
publication-quality figures in plots/.  Saves each figure as PNG (300 dpi)
and PDF.

Figures generated
─────────────────
1. Hovmöller diagram (x vs t): GT, FNO forecast, diffusion posterior, abs error
   – 512-resolution, ~100 time steps, median-RMSE trajectory
2. Snapshot comparison: GT (black), FNO (red dashed), posterior (blue),
   coarse obs (gray dots) at 64/128/256/512, single time step
3. RMSE over time: FNO-only vs iterative refinement at 512, ±1 std envelope
4. Energy spectrum: E(k) log-log at 512; GT, FNO, posterior; k^{-2} reference
5. Per-stage RMSE bar chart at 128/256/512: posterior vs FNO-only
6. Diffusion denoising trajectory: x̂₀ at DDIM steps [0,5,10,15,20,25]
7. Comprehensive Hovmöller: coarse (64), GT (512), autoreg FNO (512, with blowup
   visualisation), posterior (512), and per-method error panels below each

Usage
─────
    python scripts/plot_results.py [--config configs/default.yaml]
                                   [--traj_idx I]   (default: median-RMSE)
                                   [--time_idx T]   (default: 100)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")  # older matplotlib fallback

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "text.usetex": False,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str, save_dir: Path) -> None:
    """Save figure as PNG and PDF."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = save_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {save_dir / name}.{{png,pdf}}")


def _rmse_traj(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """RMSE over time for a single trajectory.

    Args:
        pred:  (T, N)
        truth: (T, N)

    Returns:
        (T,) per-timestep RMSE
    """
    return (pred - truth).pow(2).mean(dim=-1).sqrt()


def _select_median_traj(
    post_all: torch.Tensor,
    truth_all: torch.Tensor,
) -> int:
    """Return the trajectory index whose mean RMSE is closest to the median."""
    # post_all: (n_traj, T, N)
    mean_rmse = (post_all - truth_all).pow(2).mean(dim=(1, 2)).sqrt()  # (n_traj,)
    median_val = mean_rmse.median()
    return int((mean_rmse - median_val).abs().argmin().item())


def _energy_spectrum(u: torch.Tensor) -> torch.Tensor:
    """E(k) = |û_k|² averaged over all leading dims. Returns (N//2+1,)."""
    u_hat = torch.fft.rfft(u, norm="forward")
    E = u_hat.abs().pow(2)
    for _ in range(E.dim() - 1):
        E = E.mean(dim=0)
    return E  # (N//2+1,)


# ---------------------------------------------------------------------------
# Figure 1 — Hovmöller diagram
# ---------------------------------------------------------------------------

def plot_hovmoller(
    results: dict,
    save_dir: Path,
    traj_idx: int | None = None,
    n_time: int = 100,
) -> None:
    """Hovmöller (x–t) diagram at 512 resolution.

    Panels: Ground Truth | FNO forecast | Diffusion posterior | |Error|
    """
    res = 512
    post_all  = results[f"posterior_{res}"]   # (n_traj, T, N)
    fc_all    = results[f"forecast_{res}"]    # (n_traj, T, N)
    truth_all = results[f"truth_{res}"]       # (n_traj, T, N)

    if traj_idx is None:
        traj_idx = _select_median_traj(post_all, truth_all)

    T_avail = post_all.shape[1]
    n_time  = min(n_time, T_avail)

    post  = post_all [traj_idx, :n_time].numpy()   # (T, N)
    fc    = fc_all   [traj_idx, :n_time].numpy()
    truth = truth_all[traj_idx, :n_time].numpy()
    err   = np.abs(post - truth)

    vmax = np.percentile(np.abs(truth), 99)
    vmin = -vmax
    emax = np.percentile(err, 99)

    x = np.linspace(0, 2 * np.pi, res, endpoint=False)
    t_axis = np.arange(n_time)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    titles = ["Ground Truth", "FNO Forecast", "Diffusion Posterior", "|Error|"]
    fields = [truth, fc, post, err]
    cmaps  = [plt.cm.RdBu_r, plt.cm.RdBu_r, plt.cm.RdBu_r, plt.cm.viridis]
    vmins  = [vmin, vmin, vmin, 0.0]
    vmaxs  = [vmax, vmax, vmax, emax]

    for ax, field, title, cmap, vlo, vhi in zip(
        axes, fields, titles, cmaps, vmins, vmaxs
    ):
        im = ax.pcolormesh(
            x, t_axis, field,
            cmap=cmap, vmin=vlo, vmax=vhi,
            shading="auto", rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("$x$")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Time step $t$")
    fig.suptitle(
        f"Hovmöller diagram — resolution {res} — trajectory {traj_idx}",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig1_hovmoller", save_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Snapshot comparison
# ---------------------------------------------------------------------------

def plot_snapshot_comparison(
    results: dict,
    save_dir: Path,
    traj_idx: int | None = None,
    time_idx: int = 100,
) -> None:
    """Snapshot at a single time step across all 4 resolution levels.

    Shows GT (black), FNO (red dashed), posterior (blue), coarse obs (gray dots).
    """
    resolutions = [128, 256, 512]

    # Use 512-resolution posterior to pick median trajectory
    post_512  = results["posterior_512"]
    truth_512 = results["truth_512"]

    if traj_idx is None:
        traj_idx = _select_median_traj(post_512, truth_512)

    time_idx = min(time_idx, post_512.shape[1] - 1)

    obs_64  = results["obs_64"][traj_idx, time_idx].numpy()   # (64,)
    x_64    = np.linspace(0, 2 * np.pi, 64,  endpoint=False)

    fig, axes = plt.subplots(len(resolutions), 1, figsize=(10, 10), sharex=False)

    for ax, res in zip(axes, resolutions):
        post  = results[f"posterior_{res}"][traj_idx, time_idx].numpy()
        fc    = results[f"forecast_{res}" ][traj_idx, time_idx].numpy()
        truth = results[f"truth_{res}"    ][traj_idx, time_idx].numpy()

        x_r = np.linspace(0, 2 * np.pi, res, endpoint=False)

        ax.plot(x_r, truth, color="black",   lw=1.5,  label="Ground truth")
        ax.plot(x_r, fc,    color="red",     lw=1.2,  ls="--", label="FNO forecast")
        ax.plot(x_r, post,  color="steelblue", lw=1.5, ls="--", label="Diffusion posterior")
        # Coarse obs plotted at their native 64-pt grid (as dots)
        ax.scatter(x_64, obs_64, s=15, color="gray", alpha=0.7, zorder=5,
                   label="Coarse obs (64pt)")

        ax.set_ylabel("$u$")
        ax.set_title(f"Resolution {res}")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim(0, 2 * np.pi)

    axes[-1].set_xlabel("$x$")
    fig.suptitle(
        f"Snapshot comparison — $t$={time_idx}, trajectory {traj_idx}",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, "fig2_snapshot", save_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_over_time(results: dict, save_dir: Path) -> None:
    """RMSE over time at 512 resolution with ±1 std envelope."""
    res = 512
    post_all  = results[f"posterior_{res}"]    # (n_traj, T, N)
    fno_all   = results[f"fno_only_{res}"]     # (n_traj, T, N)
    truth_all = results[f"truth_{res}"]        # (n_traj, T, N)
    n_traj, T, N = post_all.shape

    # Per-trajectory RMSE over time
    rmse_post = torch.stack(
        [_rmse_traj(post_all[i], truth_all[i]) for i in range(n_traj)]
    )  # (n_traj, T)
    rmse_fno  = torch.stack(
        [_rmse_traj(fno_all[i], truth_all[i]) for i in range(n_traj)]
    )  # (n_traj, T)

    rmse_post_mean = rmse_post.mean(dim=0).numpy()
    rmse_post_std  = rmse_post.std(dim=0).numpy()
    rmse_fno_mean  = rmse_fno.mean(dim=0).numpy()
    rmse_fno_std   = rmse_fno.std(dim=0).numpy()

    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))

    color_post = "steelblue"
    color_fno  = "tomato"

    ax.plot(t, rmse_post_mean, color=color_post, lw=2.0, label="Iterative refinement")
    ax.fill_between(
        t,
        rmse_post_mean - rmse_post_std,
        rmse_post_mean + rmse_post_std,
        color=color_post, alpha=0.2,
    )

    ax.plot(t, rmse_fno_mean, color=color_fno, lw=2.0, ls="--", label="FNO-only")
    ax.fill_between(
        t,
        rmse_fno_mean - rmse_fno_std,
        rmse_fno_mean + rmse_fno_std,
        color=color_fno, alpha=0.2,
    )

    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("RMSE")
    ax.set_title(f"RMSE over time — resolution {res} (±1 std, {n_traj} trajectories)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig3_rmse_over_time", save_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Energy spectrum
# ---------------------------------------------------------------------------

def plot_energy_spectrum(results: dict, save_dir: Path) -> None:
    """Log-log energy spectrum E(k) at 512 resolution with k^{-2} reference.
    Shows ground truth, coarse obs (64-pt), and diffusion posterior.
    Coarse obs spectrum is computed at native 64-pt resolution and plotted
    over k=1..32 (its Nyquist), then truth and posterior over full k=1..256.
    All curves share the same y-axis scale.
    """
    res = 512
    post_all  = results[f"posterior_{res}"]   # (n_traj, T, 512)
    truth_all = results[f"truth_{res}"]       # (n_traj, T, 512)
    obs_all   = results["obs_64"]             # (n_traj, T, 64)

    E_truth = _energy_spectrum(truth_all).numpy()   # (257,)
    E_post  = _energy_spectrum(post_all).numpy()    # (257,)
    E_obs   = _energy_spectrum(obs_all).numpy()     # (33,)  native 64-pt

    k_full = np.arange(1, len(E_truth))             # 1..256
    k_obs  = np.arange(1, len(E_obs))               # 1..32

    # k^{-2} reference anchored to truth at k=5
    k_ref   = np.array([3.0, float(k_full.max())])
    ref_amp = E_truth[5] * (5.0 ** 2)
    y_ref   = ref_amp * k_ref ** (-2.0)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(k_full, E_truth[1:], color="black",     lw=2.0, label="Ground truth (512)")
    ax.loglog(k_obs,  E_obs[1:],   color="gray",      lw=1.5, ls="--", label="Coarse obs (64)")
    ax.loglog(k_full, E_post[1:],  color="steelblue", lw=1.5, label="Diffusion posterior (512)")
    ax.loglog(k_ref,  y_ref,       color="gray",      lw=1.0, ls=":",  label=r"$k^{-2}$")

    # Mark 64-pt Nyquist
    ax.axvline(32, color="gray", ls=":", lw=1.0, alpha=0.7)
    ax.text(32 * 1.05, E_truth[1] * 0.3, "64-pt\nNyquist", color="gray", fontsize=9, va="top")

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("$E(k) = |\\hat{u}_k|^2$")
    ax.set_title(f"Energy spectrum — resolution {res} (time & traj averaged)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig4_energy_spectrum", save_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Per-stage RMSE bar chart
# ---------------------------------------------------------------------------

def plot_per_stage_rmse(results: dict, save_dir: Path) -> None:
    """Bar chart: RMSE at each resolution for posterior vs FNO-only."""
    resolutions = [128, 256, 512]

    rmse_post_vals = []
    rmse_post_stds = []
    rmse_fno_vals  = []
    rmse_fno_stds  = []

    for res in resolutions:
        post_all  = results[f"posterior_{res}"]   # (n_traj, T, N)
        fno_all   = results[f"fno_only_{res}"]
        truth_all = results[f"truth_{res}"]
        n_traj    = post_all.shape[0]

        # Mean RMSE per trajectory
        per_traj_post = torch.stack(
            [(post_all[i] - truth_all[i]).pow(2).mean().sqrt() for i in range(n_traj)]
        )
        per_traj_fno = torch.stack(
            [(fno_all[i] - truth_all[i]).pow(2).mean().sqrt() for i in range(n_traj)]
        )

        rmse_post_vals.append(per_traj_post.mean().item())
        rmse_post_stds.append(per_traj_post.std().item())
        rmse_fno_vals.append(per_traj_fno.mean().item())
        rmse_fno_stds.append(per_traj_fno.std().item())

    x = np.arange(len(resolutions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(
        x - width / 2, rmse_post_vals, width,
        yerr=rmse_post_stds, capsize=4,
        color="steelblue", alpha=0.85, label="Diffusion posterior",
    )
    bars2 = ax.bar(
        x + width / 2, rmse_fno_vals, width,
        yerr=rmse_fno_stds, capsize=4,
        color="tomato", alpha=0.85, label="FNO-only",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in resolutions])
    ax.set_xlabel("Resolution $N$")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-stage RMSE (mean ± std across trajectories)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig5_per_stage_rmse", save_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Diffusion denoising trajectory
# ---------------------------------------------------------------------------

def plot_diffusion_sampling(
    results: dict,
    cfg,
    save_dir: Path,
    traj_idx: int | None = None,
    time_idx: int = 100,
) -> None:
    """Re-run DDIM with return_trajectory=True and show x̂₀ at steps [0,5,10,15,20,25]."""
    from pathlib import Path as _Path
    import torch

    # ── resolve device ────────────────────────────────────────────────────────
    device_str = str(cfg.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # ── select trajectory and time step ──────────────────────────────────────
    res = 512
    post_all  = results[f"posterior_{res}"]
    truth_all = results[f"truth_{res}"]

    if traj_idx is None:
        traj_idx = _select_median_traj(post_all, truth_all)

    time_idx = min(time_idx, post_all.shape[1] - 1)

    # ── load models ───────────────────────────────────────────────────────────
    from src.models.fno import FNO1d
    from src.models.unet import ConditionalUNet1d
    from src.models.diffusion import GaussianDiffusion
    from src.data.solver import spectral_upsample

    ckpt_dir = _Path(cfg.paths.checkpoint_dir)

    print("  Loading models for Figure 6...")

    # Load FNO at 512
    fno_512 = FNO1d(cfg, 512).to(device)
    ckpt_fno = torch.load(
        ckpt_dir / "fno_512.pt", map_location=device, weights_only=True
    )
    fno_512.load_state_dict(ckpt_fno["model"])
    fno_512.eval()

    # Load diffusion EMA
    unet = ConditionalUNet1d(cfg).to(device)
    ckpt_diff = torch.load(
        ckpt_dir / "diffusion_ema.pt", map_location=device, weights_only=True
    )
    unet.load_state_dict(ckpt_diff["model"])
    unet.eval()
    diffusion = GaussianDiffusion(unet, cfg).to(device)

    # ── build conditioning for chosen (traj, t) ───────────────────────────────
    # posterior_512 at (traj_idx, time_idx) came from:
    #   u_fc = FNO(prev_post_512) and u_co = upsample(prev_post_256)
    # We re-create this by looking at what was stored:
    #   forecast_512[traj_idx, time_idx] is the stored FNO forecast
    #   We need u_coarse_up_512 = upsample(posterior_256[traj_idx, time_idx])

    # Reconstruct u_forecast
    u_fc_np  = results[f"forecast_{res}"][traj_idx, time_idx].numpy()   # (512,)
    u_fc     = torch.tensor(u_fc_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,512)

    # For coarse input to stage 2: posterior_256[traj, time]
    if time_idx == 0:
        # At t=0 there's no previous posterior; use upsampled obs_64
        obs_np  = results["obs_64"][traj_idx, 0].numpy()  # (64,)
        obs_t   = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1,64)
        u_co_256 = spectral_upsample(obs_t, 256)  # (1,256)
    else:
        u_co_256_np = results["posterior_256"][traj_idx, time_idx].numpy()  # (256,)
        u_co_256    = torch.tensor(u_co_256_np, dtype=torch.float32).unsqueeze(0).to(device)

    u_co = spectral_upsample(u_co_256, 512).unsqueeze(1)  # (1,1,512)

    res_idx_t = torch.full((1,), 2, dtype=torch.long, device=device)  # stage 2 -> res_idx=2

    # ── run DDIM with trajectory ──────────────────────────────────────────────
    print("  Running DDIM sampling (return_trajectory=True)...")
    with torch.no_grad():
        x_final, traj_list = diffusion.ddim_sample(
            u_fc, u_co, res_idx_t,
            ddim_steps=int(cfg.inference.ddim_steps),
            eta=float(cfg.inference.eta),
            return_trajectory=True,
        )
    # traj_list: list of 25 tensors, each (1, 1, 512)

    ddim_steps = len(traj_list)   # 25
    show_at    = [0, 5, 10, 15, 20, ddim_steps - 1]
    show_at    = [s for s in show_at if s < ddim_steps]

    truth_np = truth_all[traj_idx, time_idx].numpy()    # (512,)
    x_r      = np.linspace(0, 2 * np.pi, 512, endpoint=False)

    n_panels = len(show_at)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    axes = axes.flatten()

    vmax = max(np.abs(truth_np).max(), 1.0)

    for panel_i, step_i in enumerate(show_at):
        ax     = axes[panel_i]
        x0hat  = traj_list[step_i].squeeze().cpu().numpy()   # (512,)

        ax.plot(x_r, truth_np, color="black", lw=1.2, alpha=0.6, label="GT")
        ax.plot(x_r, x0hat,    color="steelblue", lw=1.2,        label=f"$\\hat{{x}}_0$")
        ax.set_ylim(-vmax * 1.2, vmax * 1.2)
        ax.set_title(f"DDIM step {step_i + 1}/{ddim_steps}")
        ax.set_xlabel("$x$")
        if panel_i % 3 == 0:
            ax.set_ylabel("$u$")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Denoising trajectory — res {res}, traj {traj_idx}, $t$={time_idx}",
        fontsize=14,
    )
    fig.tight_layout()
    _save(fig, "fig6_denoising_trajectory", save_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — Comprehensive Hovmöller: coarse / GT / autoreg / posterior + errors
# ---------------------------------------------------------------------------

def plot_hovmoller_comprehensive(
    results: dict,
    save_dir: Path,
    traj_idx: int | None = None,
    n_time: int = 100,
) -> None:
    """6-panel Hovmöller layout at 512 resolution.

    Top row (equal width, shared y-axis):
      Col 0: Coarse obs (64-pt, upsampled for display)
      Col 1: Ground truth (512-pt)
      Col 2: Autoreg FNO-only (512-pt) — colormap saturates on blowup
      Col 3: Diffusion posterior (512-pt)

    Bottom row (cols 1-3 only, sharing the same x-axis as top):
      Col 1: empty (spacer — no error for GT)
      Col 2: |Autoreg − GT|  (log-scaled if blowup detected)
      Col 3: |Posterior − GT|

    The FNO-only blowup is made visible by:
      - Hard-clipping the colormap to ±vmax (GT 99th-pct)
      - Setting cmap.set_over / set_under to bright yellow/orange so
        blown-up pixels are unmistakably saturated rather than just dark red.
      - Annotating the first time step where |u_fno|_∞ > 5·vmax with a
        horizontal dashed line labelled "blowup".
    """
    res = 512
    post_all  = results[f"posterior_{res}"]   # (n_traj, T, 512)
    fno_all   = results[f"fno_only_{res}"]    # (n_traj, T, 512)
    truth_all = results[f"truth_{res}"]       # (n_traj, T, 512)
    obs_all   = results["obs_64"]             # (n_traj, T, 64)

    if traj_idx is None:
        traj_idx = _select_median_traj(post_all, truth_all)

    T_avail = post_all.shape[1]
    n_time  = min(n_time, T_avail)

    truth = truth_all[traj_idx, :n_time].numpy()   # (T, 512)
    fno   = fno_all  [traj_idx, :n_time].numpy()
    post  = post_all [traj_idx, :n_time].numpy()
    obs64 = obs_all  [traj_idx, :n_time].numpy()   # (T, 64)

    # Upsample coarse obs to 512 for display (spectral, but numpy-friendly via torch)
    obs64_t  = torch.tensor(obs64, dtype=torch.float32)
    from src.data.solver import spectral_upsample as _sup
    coarse_display = _sup(obs64_t, 512).numpy()    # (T, 512)

    # ── colour limits ──────────────────────────────────────────────────────────
    vmax      = float(np.percentile(np.abs(truth), 99))
    vmin      = -vmax
    blowup_thr = 5.0 * vmax   # threshold for "blowup" annotation

    # Error limits: use posterior error as reference (FNO error may be huge)
    err_post = np.abs(post  - truth)
    err_fno  = np.abs(fno   - truth)
    emax_post = float(np.percentile(err_post, 99))
    # For FNO error, cap display at 3× the posterior error max so the colour
    # scale is still informative even when the FNO has blown up.
    emax_fno  = max(emax_post * 3.0, float(np.percentile(err_fno[np.isfinite(err_fno)], 99))
                    if np.any(np.isfinite(err_fno)) else emax_post * 3.0)

    # ── detect blowup time step ────────────────────────────────────────────────
    fno_inf_norm = np.max(np.abs(np.where(np.isfinite(fno), fno, 0.0)), axis=-1)  # (T,)
    blowup_steps = np.where(fno_inf_norm > blowup_thr)[0]
    blowup_t     = int(blowup_steps[0]) if len(blowup_steps) else None

    x512 = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    t_ax = np.arange(n_time)

    # ── build colormap for FNO that saturates on blowup ───────────────────────
    import copy
    fno_cmap = copy.copy(plt.cm.RdBu_r)
    fno_cmap.set_over("gold")
    fno_cmap.set_under("limegreen")

    # Safe-clip fno for display (replace inf/nan with ±5*vmax)
    fno_display = np.clip(np.where(np.isfinite(fno), fno, np.sign(fno + 1e-9) * 5 * vmax),
                          -5 * vmax, 5 * vmax)

    # ── figure layout via GridSpec ─────────────────────────────────────────────
    # 6 columns: [coarse | gt | fno | post] on top row
    #            [spacer | spacer | fno_err | post_err] on bottom row
    # Bottom row is 55% the height of the top row.
    fig = plt.figure(figsize=(20, 9))
    gs  = gridspec.GridSpec(
        2, 4,
        height_ratios=[1.0, 0.55],
        hspace=0.08,   # tight vertical gap so error panels feel attached
        wspace=0.35,
    )

    ax_coarse   = fig.add_subplot(gs[0, 0])
    ax_gt       = fig.add_subplot(gs[0, 1], sharey=ax_coarse)
    ax_fno      = fig.add_subplot(gs[0, 2], sharey=ax_coarse)
    ax_post     = fig.add_subplot(gs[0, 3], sharey=ax_coarse)
    ax_fno_err  = fig.add_subplot(gs[1, 2], sharex=ax_fno,  sharey=None)
    ax_post_err = fig.add_subplot(gs[1, 3], sharex=ax_post, sharey=ax_fno_err)

    # Hide the two unused bottom cells
    for col in (0, 1):
        fig.add_subplot(gs[1, col]).set_visible(False)

    kw_field = dict(shading="auto", rasterized=True)
    kw_err   = dict(shading="auto", rasterized=True, cmap=plt.cm.YlOrRd)

    # ── top row: field panels ──────────────────────────────────────────────────
    im0 = ax_coarse.pcolormesh(x512, t_ax, coarse_display,
                                cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax, **kw_field)
    ax_coarse.set_title("Coarse obs (64-pt)", fontsize=13)

    im1 = ax_gt.pcolormesh(x512, t_ax, truth,
                            cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax, **kw_field)
    ax_gt.set_title("Ground truth (512-pt)", fontsize=13)

    im2 = ax_fno.pcolormesh(x512, t_ax, fno_display,
                             cmap=fno_cmap, vmin=vmin, vmax=vmax, **kw_field)
    ax_fno.set_title("Autoreg FNO-only (512-pt)", fontsize=13)

    im3 = ax_post.pcolormesh(x512, t_ax, post,
                              cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax, **kw_field)
    ax_post.set_title("Diffusion posterior (512-pt)", fontsize=13)

    # ── blowup annotation ─────────────────────────────────────────────────────
    if blowup_t is not None:
        ax_fno.axhline(blowup_t, color="gold", lw=1.5, ls="--")
        ax_fno.text(
            0.02, blowup_t / n_time + 0.02,
            f"blowup ($t$={blowup_t})",
            transform=ax_fno.transAxes,
            color="gold", fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"),
        )

    # ── bottom row: error panels ───────────────────────────────────────────────
    im_fe = ax_fno_err.pcolormesh(x512, t_ax, err_fno.clip(0, emax_fno),
                                   vmin=0, vmax=emax_fno, **kw_err)
    ax_fno_err.set_title("|FNO − GT|", fontsize=12)

    im_pe = ax_post_err.pcolormesh(x512, t_ax, err_post,
                                    vmin=0, vmax=emax_post, **kw_err)
    ax_post_err.set_title("|Posterior − GT|", fontsize=12)

    # ── labels & ticks ────────────────────────────────────────────────────────
    ax_coarse.set_ylabel("Time step $t$", fontsize=12)
    ax_fno_err.set_ylabel("Time step $t$", fontsize=11)

    for ax in (ax_gt, ax_fno, ax_post):
        plt.setp(ax.get_yticklabels(), visible=False)

    for ax in (ax_coarse, ax_gt, ax_fno, ax_post):
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in (ax_fno_err, ax_post_err):
        ax.set_xlabel("$x$", fontsize=12)
    plt.setp(ax_post_err.get_yticklabels(), visible=False)

    # ── colorbars ─────────────────────────────────────────────────────────────
    # Shared field colorbar (coarse / GT / posterior use same scale)
    cb_field = fig.colorbar(im1, ax=[ax_coarse, ax_gt, ax_post],
                             orientation="vertical", fraction=0.025, pad=0.02)
    cb_field.set_label("$u$", fontsize=11)

    # Separate colorbar for FNO (shows saturation)
    cb_fno = fig.colorbar(im2, ax=ax_fno,
                           orientation="vertical", fraction=0.046, pad=0.04,
                           extend="both")
    cb_fno.set_label("$u$ (clipped)", fontsize=10)

    # Error colorbars
    cb_fe = fig.colorbar(im_fe, ax=ax_fno_err,
                          orientation="vertical", fraction=0.046, pad=0.04,
                          extend="max")
    cb_fe.set_label("|err|", fontsize=10)

    cb_pe = fig.colorbar(im_pe, ax=ax_post_err,
                          orientation="vertical", fraction=0.046, pad=0.04)
    cb_pe.set_label("|err|", fontsize=10)

    # ── suptitle ──────────────────────────────────────────────────────────────
    blowup_note = f" · FNO blowup at $t$={blowup_t}" if blowup_t is not None else " · FNO stable"
    fig.suptitle(
        f"Hovmöller comparison — resolution {res} — trajectory {traj_idx}{blowup_note}",
        fontsize=14, y=1.01,
    )

    _save(fig, "fig7_hovmoller_comprehensive", save_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate result figures")
    parser.add_argument("--config",   type=str, default="configs/default.yaml")
    parser.add_argument("--traj_idx", type=int, default=None,
                        help="Trajectory index (default: median-RMSE trajectory)")
    parser.add_argument("--time_idx", type=int, default=100,
                        help="Time index for snapshot/denoising plots (default: 100)")
    parser.add_argument(
        "--figures", type=str, default="1,2,3,4,5,6,7",
        help="Comma-separated list of figure numbers to generate (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    results_path = Path(cfg.paths.results_dir) / "inference_results.pt"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run scripts/run_inference.py first.")
        sys.exit(1)

    print(f"Loading inference results from {results_path} ...")
    results = torch.load(results_path, map_location="cpu", weights_only=False)

    save_dir = Path(cfg.paths.plots_dir)

    figures_to_run = {int(x) for x in args.figures.split(",")}
    traj_idx = args.traj_idx
    time_idx = args.time_idx

    # Pre-select median trajectory once so all figures use the same one
    if traj_idx is None:
        post_512  = results["posterior_512"]
        truth_512 = results["truth_512"]
        traj_idx  = _select_median_traj(post_512, truth_512)
        print(f"Selected median-RMSE trajectory: {traj_idx}")

    if 1 in figures_to_run:
        print("\nFigure 1: Hovmöller diagram ...")
        plot_hovmoller(results, save_dir, traj_idx=traj_idx)

    if 2 in figures_to_run:
        print("\nFigure 2: Snapshot comparison ...")
        plot_snapshot_comparison(results, save_dir, traj_idx=traj_idx, time_idx=time_idx)

    if 3 in figures_to_run:
        print("\nFigure 3: RMSE over time ...")
        plot_rmse_over_time(results, save_dir)

    if 4 in figures_to_run:
        print("\nFigure 4: Energy spectrum ...")
        plot_energy_spectrum(results, save_dir)

    if 5 in figures_to_run:
        print("\nFigure 5: Per-stage RMSE ...")
        plot_per_stage_rmse(results, save_dir)

    if 6 in figures_to_run:
        print("\nFigure 6: Denoising trajectory ...")
        plot_diffusion_sampling(results, cfg, save_dir, traj_idx=traj_idx, time_idx=time_idx)

    if 7 in figures_to_run:
        print("\nFigure 7: Comprehensive Hovmöller ...")
        plot_hovmoller_comprehensive(results, save_dir, traj_idx=traj_idx)

    print(f"\nAll figures saved to: {save_dir}/")


if __name__ == "__main__":
    main()
