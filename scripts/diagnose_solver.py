"""
Solver diagnostic: generate 3 trajectories at 2048-pt resolution and
produce a set of diagnostic plots.

Plots saved to plots/solver_diagnostics/:
  fig_trajectories.png   — Space-time Hovmöller + 4 snapshots per trajectory
  fig_spectra.png        — Time-averaged energy spectrum vs k^{-2} reference
  fig_energy.png         — Total energy (∫u²dx / 2π) over time
  fig_gibbs.png          — Zoom-in near the sharpest shock to show Gibbs ringing

Run from project root:
    python scripts/diagnose_solver.py [--device cpu]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.solver import StochasticBurgersSolver

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 200,
    "axes.grid": True, "grid.alpha": 0.25,
})

OUT_DIR = Path("plots/solver_diagnostics")
N_TRAJ  = 3          # more trajectories to show ensemble mean vs individual variance
N_STEPS = 50_000      # 50k steps × dt=1e-4 = 5.0 time units of data
SAVE_EVERY = 250      # keep 200 snapshots (δt_output = 0.025)
SPINUP_STEPS = 100_000 # 10.0 time units spinup — matches actual dataset (200 snapshots × δ=0.05)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _energy_spectrum(u: torch.Tensor) -> np.ndarray:
    """
    Compute 1D energy spectrum E(k) = |û(k)|² / N  averaged over a batch of
    snapshots.  u: (..., N).  Returns (N//2+1,) numpy array.
    """
    N  = u.shape[-1]
    uh = torch.fft.rfft(u.double())
    ek = (uh.real**2 + uh.imag**2) / N
    # Average over all leading dims
    return ek.reshape(-1, N // 2 + 1).mean(0).cpu().numpy()


def main() -> None:
    args   = parse_args()
    device = _resolve_device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load("configs/default.yaml")
    torch.manual_seed(0)
    np.random.seed(0)

    # ------------------------------------------------------------------
    # 1.  Run solver
    # ------------------------------------------------------------------
    print(f"Device: {device}")
    print(f"Running {N_TRAJ} trajectories  |  spinup {SPINUP_STEPS} steps  "
          f"|  data {N_STEPS} steps  |  save_every {SAVE_EVERY}")

    solver = StochasticBurgersSolver(cfg, device, batch_size=N_TRAJ, seed=7)
    solver.initialize()

    print("  Spinup ...")
    solver.run_steps(SPINUP_STEPS)

    print("  Collecting snapshots ...")
    snaps = solver.solve(N_STEPS, SAVE_EVERY, show_progress=True)
    # snaps: (N_TRAJ, T, 2048) float32

    T_out = snaps.shape[1]
    N     = snaps.shape[2]
    x     = np.linspace(0, 2 * np.pi, N, endpoint=False)
    t_arr = np.arange(T_out) * SAVE_EVERY * float(cfg.pde.solver_dt)

    print(f"  Snapshots shape: {tuple(snaps.shape)}")
    print(f"  Range: [{float(snaps.min()):.3f}, {float(snaps.max()):.3f}]")
    print(f"  RMS: {float(snaps.pow(2).mean().sqrt()):.4f}")

    snaps_np = snaps.cpu().numpy()  # (N_TRAJ, T, 2048)

    N_PLOT = min(N_TRAJ, 3)  # only plot first 3 trajectories for Hovmöller/Gibbs

    # ------------------------------------------------------------------
    # 2.  Hovmöller + snapshots per trajectory
    # ------------------------------------------------------------------
    print("Plotting trajectories ...")
    fig, axes = plt.subplots(N_PLOT, 5, figsize=(18, 4 * N_PLOT),
                             gridspec_kw={"width_ratios": [2.5, 1, 1, 1, 1]})

    snap_t_indices = [0, T_out // 4, T_out // 2, T_out - 1]
    snap_t_labels  = ["t=0", "t=T/4", "t=T/2", "t=T−1"]

    for b in range(N_PLOT):
        field = snaps_np[b]          # (T, 2048)
        vmax  = np.percentile(np.abs(field), 99.5)

        # Hovmöller
        ax_h = axes[b, 0]
        im   = ax_h.pcolormesh(x, t_arr, field, cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax_h.set_xlabel("x")
        ax_h.set_ylabel("t")
        ax_h.set_title(f"Traj {b}  Hovmöller  |  range ±{vmax:.2f}")
        plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)

        # 4 snapshots
        for j, (ti, tl) in enumerate(zip(snap_t_indices, snap_t_labels)):
            ax = axes[b, j + 1]
            ax.plot(x, field[ti], lw=0.9, color="steelblue")
            ax.axhline(0, color="k", lw=0.5, ls="--")
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_title(f"Traj {b}  {tl}  (t={t_arr[ti]:.2f})")

    fig.tight_layout()
    path = OUT_DIR / "fig_trajectories.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # 3.  Energy spectrum
    # ------------------------------------------------------------------
    print("Plotting energy spectra ...")
    k_arr  = np.arange(N // 2 + 1)
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for b in range(N_TRAJ):
        ek = _energy_spectrum(snaps[b])
        label = f"Traj {b}" if b < N_PLOT else None
        color = colors[b] if b < N_PLOT else "gray"
        alpha = 1.0 if b < N_PLOT else 0.25
        ax.loglog(k_arr[1:], ek[1:], color=color, lw=1.5, alpha=alpha, label=label)

    # Reference k^{-2} (Burgers enstrophy cascade)
    k_ref = np.array([5, N // 4])
    ax.loglog(k_ref, 0.01 * k_ref ** -2.0, "k--", lw=1.2, label=r"$k^{-2}$")

    ax.set_xlabel("Wavenumber  k")
    ax.set_ylabel(r"$E(k) = |\hat{u}_k|^2 / N$")
    ax.set_title("Time-averaged energy spectrum  (all snapshots)")
    ax.legend()
    # Mark filter cutoff and rolloff region
    ax.axvline(N // 3, color="red", lw=0.8, ls=":", label=f"filter cutoff k={N//3}")
    ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "fig_spectra.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # 4.  Total energy over time
    # ------------------------------------------------------------------
    print("Plotting energy evolution ...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    all_energies = []
    for b in range(N_TRAJ):
        energy = 0.5 * (snaps_np[b] ** 2).mean(axis=-1)  # (T,)
        all_energies.append(energy)
        ax.plot(t_arr, energy, color="steelblue", lw=0.8, alpha=0.35)
    mean_energy = np.stack(all_energies).mean(axis=0)
    ax.plot(t_arr, mean_energy, color="black", lw=2.0, label=f"Ensemble mean (N={N_TRAJ})")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\langle u^2 \rangle / 2$")
    ax.set_title("Total kinetic energy  —  individual trajectories (faded) + ensemble mean (black)")
    ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "fig_energy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # 5.  Zoom near sharpest shock — Gibbs diagnostic
    # ------------------------------------------------------------------
    print("Plotting Gibbs zoom ...")
    fig, axes = plt.subplots(N_PLOT, 1, figsize=(9, 3.5 * N_PLOT), sharex=False)
    if N_PLOT == 1:
        axes = [axes]

    for b in range(N_PLOT):
        field = snaps_np[b]   # (T, 2048)

        # Find snapshot with largest gradient (most shock-like)
        grad_max = np.abs(np.diff(field, axis=1)).max(axis=1)  # (T,)
        t_shock  = int(np.argmax(grad_max))
        snap     = field[t_shock]

        # Find position of maximum |du/dx|
        grad_abs = np.abs(np.diff(snap))
        xi_shock = int(np.argmax(grad_abs))

        # Zoom window: ±100 points around shock
        half_w = 100
        lo = max(0, xi_shock - half_w)
        hi = min(N - 1, xi_shock + half_w + 1)

        ax = axes[b]
        ax.plot(x[lo:hi], snap[lo:hi], lw=1.2, color="steelblue",
                label=f"t={t_arr[t_shock]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"Traj {b}  —  Zoom near sharpest shock  "
                     f"(|du/dx|_max = {grad_abs[xi_shock]:.3f})")
        ax.legend()

        # Print some stats
        ringing = np.abs(snap[lo:hi] - np.median(snap[lo:hi]))
        print(f"  Traj {b}  shock snapshot t={t_arr[t_shock]:.3f}  "
              f"max|du/dx|={grad_abs[xi_shock]:.3f}  "
              f"ringing amplitude={ringing.max():.4f}")

    fig.tight_layout()
    path = OUT_DIR / "fig_gibbs.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # 6.  Summary statistics
    # ------------------------------------------------------------------
    print("\n── Summary ─────────────────────────────────────────────────")
    for b in range(N_PLOT):
        field  = snaps_np[b]
        energy = 0.5 * (field ** 2).mean(axis=-1)
        grad   = np.abs(np.diff(field, axis=1)).max(axis=1)
        print(f"  Traj {b}:  u ∈ [{field.min():.3f}, {field.max():.3f}]  "
              f"E_mean={energy.mean():.4f} ± {energy.std():.4f}  "
              f"max_grad={grad.max():.3f}")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
