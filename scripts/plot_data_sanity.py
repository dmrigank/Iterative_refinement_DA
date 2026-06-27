"""
Data sanity check plots for the 1D Burgers dataset.

Reproduces:
  plots/data_sanity_check.png        — 4-row stacked plot, one row per resolution
  plots/data_sanity_check_overlay.png — all resolutions overlaid on one axes

Also adds two new diagnostics:
  plots/data_sanity_hovmoller.png    — Hovmöller for 3 trajectories at N=512
  plots/data_sanity_spectrum.png     — Time-averaged energy spectrum at all resolutions

Usage:
    python scripts/plot_data_sanity.py [--split train] [--traj 0] [--t 100]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-paper")
except OSError:
    plt.style.use("seaborn-paper")

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25,
})

RESOLUTIONS = [64, 128, 256, 512]
COLORS      = {64: "#1f77b4", 128: "#ff7f0e", 256: "#2ca02c", 512: "#d62728"}
PLOTS_DIR   = Path("plots")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split",   type=str, default="train",
                   help="Which split to load: train / val / test")
    p.add_argument("--traj",    type=int, default=0,
                   help="Trajectory index for snapshot plots")
    p.add_argument("--t",       type=int, default=100,
                   help="Time-step index for snapshot plots")
    p.add_argument("--data_dir", type=str, default="data")
    return p.parse_args()


def _load(data_dir: Path, split: str) -> dict[int, torch.Tensor]:
    path = data_dir / f"{split}.pt"
    raw  = torch.load(path, map_location="cpu", weights_only=True)
    return {res: raw[f"u_{res}"].float() for res in RESOLUTIONS}


def _x_axis(n: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


def _energy_spectrum(u: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """u: (T, N) → k array, E(k) time-averaged."""
    N  = u.shape[-1]
    uh = torch.fft.rfft(u.double())
    ek = ((uh.real ** 2 + uh.imag ** 2) / N).mean(0).numpy()
    k  = np.arange(N // 2 + 1)
    return k, ek


# ---------------------------------------------------------------------------
# Fig 1: stacked subplots, one per resolution
# ---------------------------------------------------------------------------
def plot_stacked(data: dict[int, torch.Tensor], traj: int, t: int,
                 out_path: Path) -> None:
    fig, axes = plt.subplots(len(RESOLUTIONS), 1, figsize=(9, 10), sharex=False)
    fig.suptitle(
        f"Data Sanity Check — Resolution Pyramid\n"
        f"(trajectory {traj}, time step {t})",
        fontsize=12,
    )
    for ax, res in zip(axes, RESOLUTIONS):
        field = data[res][traj, t].numpy()   # (N,)
        x     = _x_axis(res)
        ax.plot(x, field, lw=1.0, color=COLORS[res], label=f"N={res}")
        ax.set_ylabel("u(x)")
        ax.set_xlim(0, 2 * np.pi)
        ylim = max(abs(field.min()), abs(field.max())) * 1.15
        ax.set_ylim(-ylim, ylim)
        ax.legend(loc="upper right")
        if ax is axes[-1]:
            ax.set_xlabel("x")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 2: overlay
# ---------------------------------------------------------------------------
def plot_overlay(data: dict[int, torch.Tensor], traj: int, t: int,
                 out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title("Overlay: all resolutions should agree on large-scale structure")

    # Plot highest-res first (thicker), lower-res on top (thinner) so coarse
    # features are visible even where lines overlap
    for res in reversed(RESOLUTIONS):
        field = data[res][traj, t].numpy()
        x     = _x_axis(res)
        lw    = {64: 2.5, 128: 1.8, 256: 1.2, 512: 0.8}[res]
        ax.plot(x, field, lw=lw, color=COLORS[res], label=f"N={res}", alpha=0.9)

    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 3: Hovmöller at N=512 for 3 trajectories
# ---------------------------------------------------------------------------
def plot_hovmoller(data: dict[int, torch.Tensor], out_path: Path) -> None:
    field_all = data[512]          # (n_traj, T, 512)
    n_show    = min(3, field_all.shape[0])
    T         = field_all.shape[1]
    x         = _x_axis(512)
    t_arr     = np.arange(T) * 0.05   # δt = 0.05 per snapshot

    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        field = field_all[i].numpy()   # (T, 512)
        vmax  = float(np.percentile(np.abs(field), 99.5))
        im    = ax.pcolormesh(x, t_arr, field, cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.set_title(f"Traj {i}  (±{vmax:.2f})")
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("t")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Hovmöller — N=512  (new dataset)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 4: energy spectrum at all resolutions
# ---------------------------------------------------------------------------
def plot_spectrum(data: dict[int, torch.Tensor], traj: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Time-averaged energy spectrum  (traj {traj})")

    for res in RESOLUTIONS:
        field    = data[res][traj]          # (T, res)
        k, ek    = _energy_spectrum(field)
        ax.loglog(k[1:], ek[1:], color=COLORS[res], lw=1.5, label=f"N={res}")

    # k^{-2} reference
    k_ref = np.array([3, 200])
    # scale reference to N=512 spectrum at k=10
    field_512 = data[512][traj]
    _, ek_512 = _energy_spectrum(field_512)
    ref_scale = ek_512[10] * 10 ** 2
    ax.loglog(k_ref, ref_scale * k_ref ** -2.0, "k--", lw=1.2, label=r"$k^{-2}$")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel(r"$E(k) = |\hat{u}_k|^2 / N$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args     = parse_args()
    data_dir = Path(args.data_dir)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.split}.pt from {data_dir} ...")
    data = _load(data_dir, args.split)

    n_traj, T, _ = data[512].shape
    traj = min(args.traj, n_traj - 1)
    t    = min(args.t,    T - 1)
    print(f"  {n_traj} trajectories × {T} timesteps  |  using traj={traj}, t={t}")
    for res in RESOLUTIONS:
        d = data[res]
        print(f"  u_{res}: min={d.min():.3f}  max={d.max():.3f}  "
              f"rms={d.pow(2).mean().sqrt():.4f}")

    plot_stacked(data, traj, t,
                 PLOTS_DIR / "data_sanity_check.png")

    plot_overlay(data, traj, t,
                 PLOTS_DIR / "data_sanity_check_overlay.png")

    plot_hovmoller(data,
                   PLOTS_DIR / "data_sanity_hovmoller.png")

    plot_spectrum(data, traj,
                  PLOTS_DIR / "data_sanity_spectrum.png")

    print("Done.")


if __name__ == "__main__":
    main()
