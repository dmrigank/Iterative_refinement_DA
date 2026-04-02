"""
Generate all 2D result figures for the Kraichnan turbulence testbed.

Figures produced:
  fig1_vorticity_snapshots.{png,pdf}  — 4×4 grid of field comparisons
  fig2_energy_spectrum.{png,pdf}      — radial energy spectrum log-log
  fig3_rmse_time.{png,pdf}            — RMSE vs time step at 256×256
  fig4_per_stage.{png,pdf}            — per-resolution RMSE bar chart
  fig5_temporal_evolution.{png,pdf}   — 6 time steps of GT vs posterior
  fig6_denoising.{png,pdf}            — DDIM intermediate denoising steps

Usage:
    python scripts/plot_results_2d.py [--config configs/kraichnan.yaml]
                                      [--figures 1,2,3,4,5,6]
                                      [--results results_2d/inference_results.pt]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf

from src.evaluation.metrics_2d import rmse_over_time_2d, radial_energy_spectrum
from src.data.dataset_2d import spectral_upsample_2d


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "legend.fontsize":   11,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.grid":         False,
})

PLOTS_DIR = Path("plots_2d")
PLOTS_DIR.mkdir(exist_ok=True)

CMAP_FIELD = "RdBu_r"
CMAP_ERR   = "hot"


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 1 — Vorticity snapshots (4 rows × 4 cols)
# ---------------------------------------------------------------------------

def plot_vorticity_snapshots(r: dict, t: int = 150, traj: int = 0) -> None:
    """4×5 grid: rows=resolutions [32,64,128,256],
    cols=[GT, FNO-only, |Error FNO-only|, Posterior, |Error Posterior|].
    Both error columns share the same colorscale (derived from FNO-only error).
    """

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(
        4, 5, figure=fig,
        hspace=0.12, wspace=0.08,
        left=0.06, right=0.96, top=0.93, bottom=0.04,
    )

    col_titles = ["Ground Truth", "FNO-only (autoreg.)", "|Error| FNO-only",
                  "Diffusion Posterior", "|Error| Posterior"]
    resolutions = [32, 64, 128, 256]
    row_labels  = ["32×32 (obs)", "64×64", "128×128", "256×256"]

    for row, res in enumerate(resolutions):
        if res == 32:
            gt_field   = r["obs_32"][traj, t].numpy()
            fno_field  = gt_field
            post_field = gt_field
        else:
            gt_field   = r[f"truth_{res}"][traj, t].numpy()
            fno_field  = r[f"fno_only_{res}"][traj, t].numpy()
            post_field = r[f"posterior_{res}"][traj, t].numpy()

        err_fno  = np.abs(fno_field  - gt_field)
        err_post = np.abs(post_field - gt_field)

        vmax = float(np.percentile(np.abs(gt_field), 99.5))
        vmax = max(vmax, 1e-6)

        # Both error columns share the scale of the FNO-only error
        emax = float(np.percentile(err_fno, 99.5))
        emax = max(emax, 1e-6)

        fields = [gt_field, fno_field, err_fno, post_field, err_post]
        cmaps  = [CMAP_FIELD, CMAP_FIELD, CMAP_ERR, CMAP_FIELD, CMAP_ERR]
        vmins  = [-vmax, -vmax, 0.0, -vmax, 0.0]
        vmaxs  = [ vmax,  vmax, emax,  vmax, emax]

        for col, (field, cmap, vmin, vmx) in enumerate(zip(fields, cmaps, vmins, vmaxs)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(
                field, cmap=cmap, vmin=vmin, vmax=vmx,
                origin="lower", aspect="equal", interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, pad=6)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11, labelpad=4)
            if res == 32 and col in (1, 2, 3, 4):
                ax.text(
                    0.5, 0.5, "obs only\n(no model)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray",
                )

            # Colourbar on the two rightmost error columns (cols 2 and 4)
            if col in (2, 4):
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Vorticity Field Comparison  |  t={t},  trajectory {traj}",
        fontsize=14, y=0.97,
    )
    _save(fig, "fig1_vorticity_snapshots")


# ---------------------------------------------------------------------------
# Figure 2 — Radial energy spectrum
# ---------------------------------------------------------------------------

def plot_energy_spectrum(r: dict, k_forcing: float = 4.0) -> None:
    """Log-log radial energy spectrum at 256×256 (and 32×32 for coarse obs),
    averaged over all time steps and trajectories.
    Shows ground truth, coarse 32×32 obs, and diffusion posterior.
    Coarse obs spectrum computed at native 32×32 resolution (k up to 22),
    truth and posterior at full 256×256 (k up to ~181). Shared y-axis scale.
    """

    truth_all = r["truth_256"].reshape(-1, 256, 256)
    post_all  = r["posterior_256"].reshape(-1, 256, 256)
    obs_all   = r["obs_32"].reshape(-1, 32, 32)

    E_truth, k_bins     = radial_energy_spectrum(truth_all)
    E_post,  _          = radial_energy_spectrum(post_all)
    E_obs,   k_bins_obs = radial_energy_spectrum(obs_all)

    # Drop k=0
    k    = k_bins[1:].numpy()
    k_ob = k_bins_obs[1:].numpy()
    Et   = E_truth[1:].numpy()
    Ep   = E_post[1:].numpy()
    Eobs = E_obs[1:].numpy()

    # k^-3 reference anchored to truth at k=5
    idx5  = np.searchsorted(k, 5)
    k_ref = np.array([3.0, k.max()])
    E_ref = Et[idx5] * (5.0 ** 3) * k_ref ** (-3)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(k,    Et,   "k-",  lw=2.0, label="Ground truth (256×256)")
    ax.loglog(k_ob, Eobs, color="gray", lw=1.5, ls="--", label="Coarse obs (32×32)")
    ax.loglog(k,    Ep,   "b-",  lw=1.5, label="Diffusion posterior (256×256)")
    ax.loglog(k_ref, E_ref, color="gray", ls=":", lw=1.2, label=r"$k^{-3}$")

    # Mark 32×32 Nyquist and forcing wavenumber
    k_nyq_32 = 16
    ax.axvline(k_nyq_32, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.text(k_nyq_32 * 1.05, Et.max() * 0.5, "32×32\nNyquist",
            color="gray", fontsize=9, va="top")
    ax.axvline(k_forcing, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax.text(k_forcing * 1.1, Et.max() * 0.15, rf"$k_f={k_forcing:.0f}$",
            color="gray", fontsize=10, va="top")

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel("$E(k)$")
    ax.set_title("Radial Energy Spectrum  (256×256, all time steps & trajectories)")
    ax.legend(loc="upper right")
    ax.set_xlim(left=1)
    fig.tight_layout()
    _save(fig, "fig2_energy_spectrum")


# ---------------------------------------------------------------------------
# Figure 3 — RMSE over time
# ---------------------------------------------------------------------------

def plot_rmse_over_time(r: dict) -> None:
    """RMSE vs time step at 256×256, one line per trajectory + mean."""

    n_traj = r["truth_256"].shape[0]
    post_all  = r["posterior_256"]   # (n_traj, T, 256, 256)
    fno_all   = r["fno_only_256"]
    truth_all = r["truth_256"]

    rmse_post = torch.stack([
        rmse_over_time_2d(post_all[i], truth_all[i]) for i in range(n_traj)
    ])  # (n_traj, T)
    rmse_fno = torch.stack([
        rmse_over_time_2d(fno_all[i],  truth_all[i]) for i in range(n_traj)
    ])  # (n_traj, T)

    T   = rmse_post.shape[1]
    t_ax = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i in range(n_traj):
        ax.plot(t_ax, rmse_fno[i].numpy(),  color="red",  alpha=0.35, lw=1.0)
        ax.plot(t_ax, rmse_post[i].numpy(), color="blue", alpha=0.35, lw=1.0)

    ax.plot(t_ax, rmse_fno.mean(0).numpy(),
            color="red",  lw=2.2, label="FNO-only (autoregressive)")
    ax.plot(t_ax, rmse_post.mean(0).numpy(),
            color="blue", lw=2.2, label="Iterative refinement (diffusion)")

    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Over Time  (256×256)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig3_rmse_time")


# ---------------------------------------------------------------------------
# Figure 4 — Per-stage RMSE bar chart
# ---------------------------------------------------------------------------

def plot_per_stage_rmse(r: dict) -> None:
    """Grouped bar chart of mean ± std RMSE at each resolution."""

    resolutions = [64, 128, 256]
    n_traj = r["truth_64"].shape[0]

    means_post, stds_post = [], []
    means_fno,  stds_fno  = [], []

    for res in resolutions:
        post_all  = r[f"posterior_{res}"]   # (n_traj, T, res, res)
        fno_all   = r[f"fno_only_{res}"]
        truth_all = r[f"truth_{res}"]

        per_traj_post = torch.stack([
            rmse_over_time_2d(post_all[i], truth_all[i]).mean()
            for i in range(n_traj)
        ])
        per_traj_fno = torch.stack([
            rmse_over_time_2d(fno_all[i], truth_all[i]).mean()
            for i in range(n_traj)
        ])

        means_post.append(per_traj_post.mean().item())
        stds_post.append( per_traj_post.std().item() if n_traj > 1 else 0.0)
        means_fno.append( per_traj_fno.mean().item())
        stds_fno.append(  per_traj_fno.std().item()  if n_traj > 1 else 0.0)

    x      = np.arange(len(resolutions))
    width  = 0.35
    labels = [f"{r}×{r}" for r in resolutions]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, means_post, width,
                   yerr=stds_post, capsize=4,
                   color="steelblue", alpha=0.85, label="Diffusion posterior")
    bars2 = ax.bar(x + width / 2, means_fno,  width,
                   yerr=stds_fno,  capsize=4,
                   color="tomato",   alpha=0.85, label="FNO-only")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mean RMSE")
    ax.set_title("Per-Resolution RMSE Comparison")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig4_per_stage")


# ---------------------------------------------------------------------------
# Figure 5 — Temporal evolution (GT vs posterior)
# ---------------------------------------------------------------------------

def plot_temporal_evolution(r: dict, traj: int = 0, n_cols: int = 6) -> None:
    """5×6 grid: rows=[GT, FNO-only, |Error FNO-only|, Posterior, |Error Posterior|]
    at evenly-spaced time steps. Both error rows share the FNO-only error colorscale.
    """

    T = r["truth_256"].shape[1]
    t_steps = np.linspace(0, T - 1, n_cols, dtype=int)

    truth_all = r["truth_256"][traj]       # (T, 256, 256)
    fno_all   = r["fno_only_256"][traj]    # (T, 256, 256)
    post_all  = r["posterior_256"][traj]   # (T, 256, 256)

    # Field colorscale from truth
    vmax = float(np.percentile(np.abs(truth_all.numpy()), 99.5))
    vmax = max(vmax, 1e-6)

    # Error colorscale from FNO-only errors (shared across both error rows)
    err_fno_all  = (fno_all  - truth_all).abs()
    emax = float(np.percentile(err_fno_all.numpy(), 99.5))
    emax = max(emax, 1e-6)

    n_rows = 5
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.08, wspace=0.05,
        left=0.08, right=0.94, top=0.93, bottom=0.03,
    )

    row_labels = ["Ground Truth", "FNO-only (autoreg.)",
                  "|Error| FNO-only", "Diffusion Posterior", "|Error| Posterior"]

    for col, t in enumerate(t_steps):
        err_fno  = (fno_all[t]  - truth_all[t]).abs().numpy()
        err_post = (post_all[t] - truth_all[t]).abs().numpy()

        row_fields = [
            truth_all[t].numpy(),
            fno_all[t].numpy(),
            err_fno,
            post_all[t].numpy(),
            err_post,
        ]
        row_cmaps = [CMAP_FIELD, CMAP_FIELD, CMAP_ERR, CMAP_FIELD, CMAP_ERR]
        row_vmins = [-vmax, -vmax, 0.0, -vmax, 0.0]
        row_vmaxs = [ vmax,  vmax, emax,  vmax, emax]

        for row, (field, cmap, vmin, vmx) in enumerate(
            zip(row_fields, row_cmaps, row_vmins, row_vmaxs)
        ):
            ax = fig.add_subplot(gs[row, col])
            im_row = ax.imshow(
                field, cmap=cmap, vmin=vmin, vmax=vmx,
                origin="lower", aspect="equal", interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={t}", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10, labelpad=4)

            # Store last imshow per row for colorbars
            if col == n_cols - 1:
                # attach colorbar to the last column of each row
                cb = fig.colorbar(im_row, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Temporal Evolution  |  256×256,  trajectory {traj}",
        fontsize=14, y=0.97,
    )
    _save(fig, "fig5_temporal_evolution")


# ---------------------------------------------------------------------------
# Figure 6 — DDIM denoising trajectory
# ---------------------------------------------------------------------------

def plot_denoising_trajectory(cfg, device: torch.device) -> None:
    """2×3 grid of intermediate DDIM denoising steps (128→256 transition)."""
    from src.models.fno_2d import FNO2d
    from src.models.unet_2d import ConditionalUNet2d
    from src.models.diffusion import GaussianDiffusion
    from src.inference.pipeline_2d import IterativeRefinementPipeline2d
    import src.models.diffusion as _dm
    _dm.ConditionalUNet1d = nn.Module

    ckpt_dir = Path(cfg.paths.checkpoint_dir)

    # Load only FNO128, FNO256 (we need 128→256 stage, res_idx=2)
    fnos: dict[int, FNO2d] = {}
    for res in [64, 128, 256]:
        m = FNO2d(cfg, res).to(device)
        ck = torch.load(ckpt_dir / f"fno_{res}.pt", map_location=device, weights_only=True)
        m.load_state_dict(ck["model"])
        m.eval()
        fnos[res] = m

    ck = torch.load(ckpt_dir / "diffusion_ema.pt", map_location=device, weights_only=True)
    unet = ConditionalUNet2d(cfg).to(device)
    unet.load_state_dict(ck["model"])
    unet.eval()
    diffusion = GaussianDiffusion(unet, cfg).to(device)

    # Load test data, pick trajectory 0, timestep 50
    test_data = torch.load(
        Path(cfg.data.data_dir) / "test.pt", map_location=device, weights_only=True
    )
    t_pick = 50

    # Get the 128×128 posterior at t=50 (acts as the coarse input for 128→256)
    # We run the pipeline up to stage 1 to get it honestly
    pipeline = IterativeRefinementPipeline2d(fnos, diffusion, cfg, device)
    obs_32   = test_data["w_32"][0, :t_pick + 2]   # need up to t+1
    res_full = pipeline.run(obs_32, n_steps=t_pick + 2)

    post_128_t = res_full["posterior_128"][t_pick].to(device)  # (128, 128)
    prev_256_t = res_full["posterior_256"][t_pick - 1].to(device)  # (128, 128)

    # Stage 2: 128→256, res_idx=2
    # forecast_256 from prev posterior at 256
    w_prev_256 = prev_256_t.unsqueeze(0).unsqueeze(0)   # (1, 1, 256, 256)
    w_fc_256   = fnos[256](w_prev_256)                   # (1, 1, 256, 256)

    # Coarse up from 128 posterior
    from src.data.dataset_2d import spectral_upsample_2d
    w_co_256 = spectral_upsample_2d(
        post_128_t.unsqueeze(0), target_ny=256, target_nx=256
    ).unsqueeze(1)  # (1, 1, 256, 256)

    res_idx_t = torch.tensor([2], dtype=torch.long, device=device)

    # Run DDIM with full trajectory returned
    with torch.no_grad():
        _, trajectory = diffusion.ddim_sample(
            w_fc_256, w_co_256, res_idx_t,
            ddim_steps=100, eta=0.0,
            return_trajectory=True,
        )
    # trajectory is a list of 100 (1,1,256,256) tensors: x̂₀ estimate at each step

    # Pick steps to show: indices 0,4,9,14,19,24 → DDIM steps 1,5,10,15,20,25
    show_indices = [0, 19, 39, 59, 89, 99] #[0, 4, 9, 14, 19, 24]
    ddim_labels  = [1, 20, 40, 60, 80, 100] #[1, 5, 10, 15, 20, 25]
    frames = [trajectory[i].squeeze().cpu().numpy() for i in show_indices]

    # Use the final (converged) frame to set the colorscale — early noisy frames
    # have much larger amplitudes and would wash out the converged structure.
    vmax = float(np.percentile(np.abs(frames[-1]), 99.5))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
    for ax, frame, label in zip(axes, frames, ddim_labels):
        im = ax.imshow(
            frame, cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
            origin="lower", aspect="equal", interpolation="nearest",
        )
        ax.set_title(f"DDIM step {label}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared colourbar
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.04, top=0.88, wspace=0.06)
    cbar_ax = fig.add_axes([0.93, 0.10, 0.015, 0.75])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=9)

    fig.suptitle(
        "DDIM Denoising Trajectory  |  128→256 transition,  t=50,  trajectory 0",
        fontsize=13, y=1.01,
    )
    _save(fig, "fig6_denoising")


# ---------------------------------------------------------------------------
# Figure 7 — GT vs Prior vs Posterior temporal comparison (5 snapshots)
# ---------------------------------------------------------------------------

def plot_temporal_comparison(r: dict, traj: int = 0, n_cols: int = 5) -> None:
    """4×5 grid: rows=[GT, FNO Prior, Posterior, |Prior - Posterior|]
    at 5 evenly-spaced time steps up to the final timestep.
    """

    T = r["truth_256"].shape[1]
    t_steps = np.linspace(0, T - 1, n_cols, dtype=int)

    truth_all = r["truth_256"][traj]      # (T, 256, 256)
    prior_all = r["forecast_256"][traj]   # (T, 256, 256)  — 1-step FNO forecast
    post_all  = r["posterior_256"][traj]  # (T, 256, 256)

    # Field colorscale from truth
    vmax = float(np.percentile(np.abs(truth_all.numpy()), 99.5))
    vmax = max(vmax, 1e-6)

    # Error colorscale from |prior - posterior| across all selected timesteps
    diff_frames = [(prior_all[t] - post_all[t]).abs().numpy() for t in t_steps]
    emax = float(np.percentile(np.concatenate([e.ravel() for e in diff_frames]), 99.5))
    emax = max(emax, 1e-6)

    n_rows = 4
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.08, wspace=0.05,
        left=0.09, right=0.93, top=0.93, bottom=0.03,
    )

    row_labels = ["Ground Truth", "FNO Prior (1-step)",
                  "Diffusion Posterior", "|Prior − Posterior|"]

    for col, t in enumerate(t_steps):
        diff = (prior_all[t] - post_all[t]).abs().numpy()

        row_fields = [
            truth_all[t].numpy(),
            prior_all[t].numpy(),
            post_all[t].numpy(),
            diff,
        ]
        row_cmaps = [CMAP_FIELD, CMAP_FIELD, CMAP_FIELD, CMAP_ERR]
        row_vmins = [-vmax, -vmax, -vmax, 0.0]
        row_vmaxs = [ vmax,  vmax,  vmax, emax]

        for row, (field, cmap, vmin, vmx) in enumerate(
            zip(row_fields, row_cmaps, row_vmins, row_vmaxs)
        ):
            ax = fig.add_subplot(gs[row, col])
            im_row = ax.imshow(
                field, cmap=cmap, vmin=vmin, vmax=vmx,
                origin="lower", aspect="equal", interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={t}", fontsize=11)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10, labelpad=4)

            if col == n_cols - 1:
                cb = fig.colorbar(im_row, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"GT vs Prior vs Posterior  |  256×256,  trajectory {traj}",
        fontsize=14, y=0.97,
    )
    _save(fig, "fig7_temporal_comparison")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 2D result figures")
    p.add_argument("--config",  type=str, default="configs/kraichnan.yaml")
    p.add_argument("--results", type=str, default="results_2d/inference_results.pt")
    p.add_argument("--figures", type=str, default="1,2,3,4,5,6,7",
                   help="Comma-separated list of figures to generate (default: all)")
    p.add_argument("--snapshot_t", type=int, default=50,
                   help="Time step for Fig 1 vorticity snapshot (default: 50)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    cfg     = OmegaConf.load(args.config)
    figs    = {int(x) for x in args.figures.split(",")}

    device_str = str(cfg.device)
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    print(f"Loading inference results from {args.results} ...")
    r = torch.load(args.results, map_location="cpu", weights_only=True)
    print(f"  Trajectories: {r['truth_256'].shape[0]},  "
          f"time steps: {r['truth_256'].shape[1]}")

    if 1 in figs:
        print("\nFigure 1: Vorticity snapshots ...")
        plot_vorticity_snapshots(r, t=args.snapshot_t)

    if 2 in figs:
        print("\nFigure 2: Radial energy spectrum ...")
        plot_energy_spectrum(r, k_forcing=float(cfg.pde.forcing_band_center))

    if 3 in figs:
        print("\nFigure 3: RMSE over time ...")
        plot_rmse_over_time(r)

    if 4 in figs:
        print("\nFigure 4: Per-stage RMSE ...")
        plot_per_stage_rmse(r)

    if 5 in figs:
        print("\nFigure 5: Temporal evolution ...")
        plot_temporal_evolution(r)

    if 6 in figs:
        print("\nFigure 6: Denoising trajectory (requires loading models) ...")
        plot_denoising_trajectory(cfg, device)

    if 7 in figs:
        print("\nFigure 7: GT vs Prior vs Posterior temporal comparison ...")
        plot_temporal_comparison(r)

    print(f"\nAll figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
