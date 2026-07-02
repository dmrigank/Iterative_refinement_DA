"""
Compare one-step FNO predictions: three separate FNOs vs the single shared FNO.

For a chosen trajectory and timestep t, loads w_t at each resolution
(64, 128, 256), runs both FNO variants to predict w_{t+1}, and produces:

  Row layout (one row per resolution, 4 columns):
    [Ground truth w_{t+1}]  [Separate FNO pred]  [Shared FNO pred]  [E(k) of all three]

Saved to plots_2d_misc/fig_fno_shared_vs_separate.{png,pdf}.

A second figure shows |error| fields (separate vs shared) per resolution:
  plots_2d_misc/fig_fno_shared_vs_separate_error.{png,pdf}

Usage:
    python scripts/plot_fno_shared_vs_separate.py
        [--config       configs/kraichnan.yaml]
        [--traj         0]
        [--t            80]
        [--k_max_shared 64]
        [--out          plots_2d_misc]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from src.models.fno_2d import FNO2d
from src.models.fno_2d_shared import SharedFNO2d
from src.evaluation.metrics_2d import radial_energy_spectrum, rmse_2d

RESOLUTIONS = [64, 128, 256]

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi":     150,
    "savefig.dpi":    300,
})

C_GT       = "black"
C_SEP      = "#1f77b4"   # blue — separate (3 independent FNOs)
C_SHARED   = "#d62728"   # red  — shared (1 FNO, fixed k_max)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_separate_fnos(cfg, device: torch.device) -> dict[int, FNO2d]:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    fnos: dict[int, FNO2d] = {}
    for res in RESOLUTIONS:
        model = FNO2d(cfg, res).to(device)
        ckpt  = torch.load(ckpt_dir / f"fno_{res}.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()
        fnos[res] = model
        print(f"  Loaded FNO2d {res}×{res}  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4e})")
    return fnos


def _load_shared_fno(cfg, device: torch.device, k_max_shared: int) -> SharedFNO2d:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    model    = SharedFNO2d(cfg, k_max_shared=k_max_shared).to(device)
    ckpt     = torch.load(ckpt_dir / "fno_shared.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded SharedFNO2d  k_max_shared={k_max_shared}  "
          f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4e})")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=str, default="configs/kraichnan.yaml")
    parser.add_argument("--traj",         type=int, default=0)
    parser.add_argument("--t",            type=int, default=80)
    parser.add_argument("--k_max_shared", type=int, default=64)
    parser.add_argument("--out",          type=str, default="plots_2d_misc")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading models...")
    fnos_sep    = _load_separate_fnos(cfg, device)
    fno_shared  = _load_shared_fno(cfg, device, k_max_shared=args.k_max_shared)

    data = torch.load(Path(cfg.data.data_dir) / "test.pt", map_location="cpu", weights_only=True)
    traj, t = args.traj, args.t
    T = data["w_64"].shape[1]
    t = min(t, T - 2)   # need t and t+1 both in range

    # ── Run both variants at each resolution ─────────────────────────────────
    w_t1_truth: dict[int, torch.Tensor] = {}
    w_t1_sep:   dict[int, torch.Tensor] = {}
    w_t1_shared: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for res in RESOLUTIONS:
            w_t  = data[f"w_{res}"][traj, t].float().unsqueeze(0).unsqueeze(0).to(device)   # (1,1,res,res)
            w_t1 = data[f"w_{res}"][traj, t + 1].float()                                      # (res,res)

            pred_sep    = fnos_sep[res](w_t).squeeze(0).squeeze(0).cpu()
            pred_shared = fno_shared(w_t).squeeze(0).squeeze(0).cpu()

            w_t1_truth[res]   = w_t1
            w_t1_sep[res]     = pred_sep
            w_t1_shared[res]  = pred_shared

    # ── Per-resolution RMSE printout ──────────────────────────────────────────
    print(f"\n── One-step prediction RMSE  (traj={traj}, t={t}->{t+1}) ──────────────")
    print(f"  {'Resolution':>12}  {'Separate':>10}  {'Shared':>10}")
    rmse_sep_dict: dict[int, float] = {}
    rmse_shared_dict: dict[int, float] = {}
    for res in RESOLUTIONS:
        r_sep    = float(rmse_2d(w_t1_sep[res],    w_t1_truth[res]))
        r_shared = float(rmse_2d(w_t1_shared[res], w_t1_truth[res]))
        rmse_sep_dict[res]    = r_sep
        rmse_shared_dict[res] = r_shared
        print(f"  {res:>10}×{res:<1}  {r_sep:10.4f}  {r_shared:10.4f}")

    # ── Shared color scale per resolution (99.5th percentile of truth) ───────
    vmax = {res: float(w_t1_truth[res].abs().quantile(0.995)) for res in RESOLUTIONS}

    # =========================================================================
    # Figure 1: vorticity fields (GT / separate / shared) + E(k) per resolution
    # =========================================================================
    fig, axes = plt.subplots(len(RESOLUTIONS), 4, figsize=(17, 4.2 * len(RESOLUTIONS)))

    im0_per_row: dict[int, "plt.cm.ScalarMappable"] = {}

    for row, res in enumerate(RESOLUTIONS):
        vm = vmax[res]

        ax_gt  = axes[row, 0]
        ax_sep = axes[row, 1]
        ax_sh  = axes[row, 2]
        ax_ek  = axes[row, 3]

        im0 = ax_gt.imshow(w_t1_truth[res].numpy(), cmap="RdBu_r", vmin=-vm, vmax=vm,
                            origin="lower", interpolation="nearest", aspect="equal")
        ax_sep.imshow(w_t1_sep[res].numpy(), cmap="RdBu_r", vmin=-vm, vmax=vm,
                      origin="lower", interpolation="nearest", aspect="equal")
        ax_sh.imshow(w_t1_shared[res].numpy(), cmap="RdBu_r", vmin=-vm, vmax=vm,
                     origin="lower", interpolation="nearest", aspect="equal")
        im0_per_row[row] = im0

        for ax in (ax_gt, ax_sep, ax_sh):
            ax.set_xticks([]); ax.set_yticks([])

        ax_gt.set_ylabel(f"{res}×{res}", fontsize=13, fontweight="bold")
        if row == 0:
            ax_gt.set_title("Ground truth  $w_{t+1}$")
            ax_sep.set_title("Separate FNOs")
            ax_sh.set_title("Shared FNO")
            ax_ek.set_title("Radial energy spectrum")

        ax_sep.set_xlabel(f"RMSE = {rmse_sep_dict[res]:.4f}", fontsize=9, color=C_SEP)
        ax_sh.set_xlabel(f"RMSE = {rmse_shared_dict[res]:.4f}", fontsize=9, color=C_SHARED)

        # ── E(k) panel ─────────────────────────────────────────────────────────
        E_gt, k_gt = radial_energy_spectrum(w_t1_truth[res])
        E_sep, _   = radial_energy_spectrum(w_t1_sep[res])
        E_sh,  _   = radial_energy_spectrum(w_t1_shared[res])

        k_arr  = k_gt.numpy()[1:]
        k_max_phys = res // 3   # 2/3 dealiasing cutoff
        mask   = (k_arr >= 1) & (k_arr <= k_max_phys)

        ax_ek.loglog(k_arr[mask], E_gt.numpy()[1:][mask],  color=C_GT,     lw=2.0, label="Ground truth")
        ax_ek.loglog(k_arr[mask], E_sep.numpy()[1:][mask], color=C_SEP,    lw=1.8, ls="--", label="Separate FNOs")
        ax_ek.loglog(k_arr[mask], E_sh.numpy()[1:][mask],  color=C_SHARED, lw=1.8, ls=":",  label="Shared FNO")

        ax_ek.set_xlabel("Wavenumber  $k$")
        ax_ek.set_ylabel("$E(k)$")
        ax_ek.grid(True, which="both", ls="--", alpha=0.25)
        ax_ek.legend(fontsize=8, framealpha=0.85, loc="lower left")
        ax_ek.set_xlim(left=1, right=k_max_phys * 1.1)

    fig.suptitle(
        f"FNO one-step prediction: separate vs. shared design  "
        f"(traj={traj}, t={t}→{t+1})",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.97])

    # Colorbars added AFTER tight_layout, once axis positions are finalized,
    # as thin insets along the right edge of column 2 (ax_sh) — avoids the
    # auto-placed colorbar overlapping the 4th (spectrum) column.
    for row in range(len(RESOLUTIONS)):
        pos = axes[row, 2].get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.010, pos.height])
        fig.colorbar(im0_per_row[row], cax=cax).set_label("ω", fontsize=9)

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_fno_shared_vs_separate.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved fig_fno_shared_vs_separate.{{png,pdf}} to {args.out}/")

    # =========================================================================
    # Figure 2: |error| fields — separate vs shared, per resolution
    # =========================================================================
    fig2, axes2 = plt.subplots(len(RESOLUTIONS), 2, figsize=(8, 4.2 * len(RESOLUTIONS)))

    im_e2_per_row: dict[int, "plt.cm.ScalarMappable"] = {}

    for row, res in enumerate(RESOLUTIONS):
        err_sep    = (w_t1_sep[res]    - w_t1_truth[res]).abs()
        err_shared = (w_t1_shared[res] - w_t1_truth[res]).abs()
        vmax_err   = float(max(err_sep.max(), err_shared.max()))

        ax_e1 = axes2[row, 0]
        ax_e2 = axes2[row, 1]

        ax_e1.imshow(err_sep.numpy(), cmap="inferno", vmin=0, vmax=vmax_err,
                     origin="lower", interpolation="nearest", aspect="equal")
        im_e2 = ax_e2.imshow(err_shared.numpy(), cmap="inferno", vmin=0, vmax=vmax_err,
                              origin="lower", interpolation="nearest", aspect="equal")
        im_e2_per_row[row] = im_e2

        for ax in (ax_e1, ax_e2):
            ax.set_xticks([]); ax.set_yticks([])

        ax_e1.set_ylabel(f"{res}×{res}", fontsize=13, fontweight="bold")
        if row == 0:
            ax_e1.set_title("|Separate FNO error|")
            ax_e2.set_title("|Shared FNO error|")

        ax_e1.set_xlabel(f"RMSE = {rmse_sep_dict[res]:.4f}", fontsize=9, color=C_SEP)
        ax_e2.set_xlabel(f"RMSE = {rmse_shared_dict[res]:.4f}", fontsize=9, color=C_SHARED)

    fig2.suptitle(
        f"FNO one-step prediction error: separate vs. shared design  "
        f"(traj={traj}, t={t}→{t+1})",
        fontsize=14, y=0.995,
    )
    fig2.tight_layout(rect=[0, 0, 0.90, 0.97])

    # Colorbars added AFTER tight_layout (each row has its own vmax_err, so
    # each gets its own colorbar) as thin insets right of column 2.
    for row in range(len(RESOLUTIONS)):
        pos = axes2[row, 1].get_position()
        cax = fig2.add_axes([pos.x1 + 0.02, pos.y0, 0.018, pos.height])
        fig2.colorbar(im_e2_per_row[row], cax=cax).set_label("|error|", fontsize=9)

    for ext in ("png", "pdf"):
        fig2.savefig(os.path.join(args.out, f"fig_fno_shared_vs_separate_error.{ext}"), bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved fig_fno_shared_vs_separate_error.{{png,pdf}} to {args.out}/")


if __name__ == "__main__":
    main()
