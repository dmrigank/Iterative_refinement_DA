"""
plot_multiscale_schematic.py — Visualises the multiscale nature of the problem.

Single combined figure saved to plots_2d_misc/fig_multiscale.{png,pdf}:

  Row 1 (native resolution):   vorticity at 32x32, 64x64, 128x128, 256x256
  Row 2 (upsampled to 256x256): each coarse field spectrally upsampled —
                                shows what information is missing at each scale
  Row 3 (E(k) panel spanning full width):
      Radial energy spectrum for all four resolutions on a single log-log axes.
      The hard cutoff at each resolution's Nyquist wavenumber makes the
      missing fine-scale information explicit.

Usage:
    python scripts/plot_multiscale_schematic.py
        [--traj  0]
        [--t     80]
        [--out   plots_2d_misc]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset_2d import spectral_upsample_2d
from src.evaluation.metrics_2d import radial_energy_spectrum

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi":     150,
    "savefig.dpi":    300,
})

RESOLUTIONS = [32, 64, 128, 256]
RES_COLORS  = {
    32:  "#d62728",   # red
    64:  "#ff7f0e",   # orange
    128: "#2ca02c",   # green
    256: "#1f77b4",   # blue
}
RES_LABELS = {
    32:  "32×32  (observation)",
    64:  "64×64",
    128: "128×128",
    256: "256×256  (target)",
}


def _nyquist(res: int) -> int:
    return res // 2

def _kmax(res: int) -> int:
    """Physical wavenumber cutoff: 2/3 dealiasing rule."""
    return res // 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=int, default=0)
    parser.add_argument("--t",    type=int, default=80)
    parser.add_argument("--out",  type=str, default="plots_2d_misc")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data = torch.load("data_2d/test.pt", map_location="cpu", weights_only=True)
    traj, t = args.traj, args.t
    T = data["w_32"].shape[1]
    t = min(t, T - 1)

    # Extract fields at each resolution
    fields = {res: data[f"w_{res}"][traj, t].float() for res in RESOLUTIONS}

    # Spectrally upsample each coarse field to 256x256
    fields_up = {}
    for res in RESOLUTIONS:
        f = fields[res].unsqueeze(0)   # (1, res, res)
        fields_up[res] = spectral_upsample_2d(f, 256, 256).squeeze(0)

    # Compute radial energy spectra (on the native-res fields, averaged over
    # a small temporal window for stability)
    window = 5
    t_lo = max(0, t - window // 2)
    t_hi = min(T, t_lo + window)
    spectra = {}
    for res in RESOLUTIONS:
        frames = data[f"w_{res}"][traj, t_lo:t_hi].float()  # (W, res, res)
        E, k = radial_energy_spectrum(frames)
        spectra[res] = (E.numpy(), k.numpy())

    # Shared color scale: 99.5th percentile of 256x256 field
    vmax = float(fields[256].abs().quantile(0.995))

    # ── Layout ────────────────────────────────────────────────────────────────
    # 3 rows: [native fields] [upsampled fields] [spectrum]
    # Row 1 and 2 each have 4 columns; row 3 spans all columns
    fig = plt.figure(figsize=(16, 13))
    gs  = gridspec.GridSpec(
        3, 4,
        figure=fig,
        height_ratios=[1, 1, 1.1],
        hspace=0.35, wspace=0.08,
        left=0.06, right=0.97, top=0.93, bottom=0.06,
    )

    # ── Row 0: native resolution fields ──────────────────────────────────────
    for col, res in enumerate(RESOLUTIONS):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(fields[res].numpy(), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, origin="lower",
                       interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{res}×{res}", fontsize=12, fontweight="bold",
                     color=RES_COLORS[res])
        # Mark pixel size relative to 256
        scale = 256 // res
        if scale > 1:
            ax.set_xlabel(f"1 pixel = {scale}×{scale} target pixels",
                          fontsize=8, color="gray")
        if col == 0:
            ax.set_ylabel("Native resolution", fontsize=10, labelpad=6)

    # colorbar added after both rows are drawn (see below)

    # ── Row 1: spectrally upsampled to 256×256 ────────────────────────────────
    for col, res in enumerate(RESOLUTIONS):
        ax = fig.add_subplot(gs[1, col])
        f_up = fields_up[res].numpy()
        im2 = ax.imshow(f_up, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, origin="lower",
                        interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])

        if res == 256:
            ax.set_title("256×256\n(ground truth)", fontsize=10,
                         color=RES_COLORS[res])
        else:
            # RMSE of upsampled vs ground truth
            err = float((fields_up[res] - fields_up[256]).pow(2).mean().sqrt())
            ax.set_title(f"↑ to 256×256\nRMSE = {err:.3f}", fontsize=10,
                         color=RES_COLORS[res])
        if col == 0:
            ax.set_ylabel("Upsampled to 256×256", fontsize=10, labelpad=6)

    # Single shared colorbar for both image rows (same vmax)
    fig.colorbar(im, ax=[fig.axes[i] for i in range(8)],
                 fraction=0.012, pad=0.01, shrink=0.45).set_label("ω", fontsize=11)

    # ── Row 2: energy spectrum (spans all 4 columns) ──────────────────────────
    ax_spec = fig.add_subplot(gs[2, :])

    k_ref = np.array([3.0, float(_kmax(256))])
    # anchor k^{-3} to 256x256 spectrum at k=5
    E_256, k_256 = spectra[256]
    k_arr_256 = k_256[1:]
    idx_anchor = np.searchsorted(k_arr_256, 5)
    E_ref = E_256[1:][idx_anchor] * (float(k_arr_256[idx_anchor]) ** 3) * k_ref ** (-3)

    ax_spec.loglog(k_ref, E_ref, color="gray", lw=1.0, ls=":", zorder=1,
                   label=r"$k^{-3}$ reference")

    # Plot each resolution's spectrum — they coincide at low k then cut off
    # at their respective Nyquist. Plot from highest to lowest so low-res
    # curves sit on top.
    for res in reversed(RESOLUTIONS):
        E, k = spectra[res]
        k_arr = k[1:]
        k_cut = _kmax(res)
        mask  = k_arr <= k_cut
        lw    = 3.0 if res == 32 else 2.5 if res == 64 else 2.0 if res == 128 else 1.5
        ax_spec.loglog(k_arr[mask], E[1:][mask],
                       color=RES_COLORS[res], lw=lw,
                       label=RES_LABELS[res], zorder=3 + RESOLUTIONS.index(res))

        # filled dot marking the cutoff
        ax_spec.plot(k_cut, E[1:][mask][-1],
                     "o", color=RES_COLORS[res], ms=8, zorder=6)

        # vertical dashed line at cutoff
        ax_spec.axvline(k_cut, color=RES_COLORS[res], lw=0.8, ls="--", alpha=0.4,
                        zorder=2)

    # shade the "missing" region between 32x32 cutoff and 256x256 cutoff
    ax_spec.axvspan(_kmax(32), _kmax(256),
                    color="lightgray", alpha=0.3, zorder=0,
                    label=f"Unobserved scales  ($k > {_kmax(32)}$)")

    ax_spec.set_xlabel("Wavenumber  $k$", fontsize=12)
    ax_spec.set_ylabel("$E(k)$", fontsize=12)
    ax_spec.set_title("Radial Energy Spectrum — information missing at coarser resolutions",
                      fontsize=12)
    ax_spec.legend(loc="lower left", fontsize=9, framealpha=0.9, ncol=2)
    ax_spec.set_xlim(left=1, right=_kmax(256) * 1.3)
    ax_spec.grid(True, which="both", ls="--", alpha=0.2)

    # Add cutoff labels now that ylim is determined by the data
    y_lo = ax_spec.get_ylim()[0]
    for res in RESOLUTIONS:
        k_cut = _kmax(res)
        ax_spec.text(k_cut * 1.07, y_lo * 3,
                     f"$k_{{\\rm max}}={k_cut}$",
                     color=RES_COLORS[res], fontsize=8,
                     va="bottom", rotation=90, zorder=4)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Multiscale structure of 2D Kraichnan turbulence  "
        f"(traj={traj}, t={t})",
        fontsize=13, y=0.97,
    )

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_multiscale.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fig_multiscale.{{png,pdf}} to {args.out}/")


if __name__ == "__main__":
    main()
