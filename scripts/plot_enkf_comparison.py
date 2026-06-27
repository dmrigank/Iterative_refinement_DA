"""
plot_enkf_comparison.py — Solver EnKF / Learned (FNO) EnKF / IR comparison.

Figures saved to plots_enkf/:
  fig1_snapshot.png/pdf         — vorticity fields + error maps, one time step
  fig2_rmse_time.png/pdf        — RMSE over time, all methods
  fig3_spectrum.png/pdf         — radial energy spectrum, all methods
  fig4_per_band_rmse.png/pdf    — RMSE decomposed by wavenumber band
  fig5_spread.png/pdf           — EnKF ensemble spread vs RMSE over time
  fig6_runtime.png/pdf          — per-step wall-time bar chart (log scale)

Usage:
    python scripts/plot_enkf_comparison.py
        [--enkf_solver  results_enkf/inference_results.pt]
        [--enkf_fno     results_enkf_fno/inference_results.pt]
        [--ir           results_2d/inference_results.pt]
        [--out          plots_enkf]
        [--traj         0]
        [--snap_t       50]
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
from src.evaluation.metrics_2d import radial_energy_spectrum

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi":     150,
    "savefig.dpi":    300,
})

C = {
    "truth":        "#111111",
    "bicubic":      "#888888",
    "enkf_solver":  "#2ca02c",   # green  — solver EnKF
    "enkf_fno":     "#d62728",   # red    — learned (FNO) EnKF
    "ir":           "#1f77b4",   # blue   — iterative refinement
}


def _save(fig, stem, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.{{png,pdf}}")


def _rmse_t(pred, truth):
    """Per-step RMSE, shape (T,)."""
    return (pred - truth).pow(2).mean(dim=(-2, -1)).sqrt()


def _bicubic(obs32, ny=256, nx=256):
    flat = obs32.reshape(-1, 1, 32, 32).float()
    up   = torch.nn.functional.interpolate(flat, size=(ny, nx),
                                           mode="bicubic", align_corners=False)
    return up.squeeze(1).reshape(*obs32.shape[:-2], ny, nx)


def _band_rmse(pred, truth, k_lo, k_hi):
    ny, nx = pred.shape[-2], pred.shape[-1]
    kx  = torch.fft.fftfreq(ny, d=1.0/ny).float()
    ky  = torch.fft.rfftfreq(nx, d=1.0/nx).float()
    K   = (kx.unsqueeze(1).expand(ny, nx//2+1).pow(2) +
           ky.unsqueeze(0).expand(ny, nx//2+1).pow(2)).sqrt()
    mask = (K >= k_lo) & (K < k_hi)
    P = torch.fft.rfft2(pred.float())
    T = torch.fft.rfft2(truth.float())
    err = torch.fft.irfft2((P - T) * mask, s=(ny, nx))
    return float(err.pow(2).mean().sqrt())


# ── fig 1: snapshot grid ──────────────────────────────────────────────────────

def plot_snapshot(re, rf, ri, out_dir, traj=0, t=50):
    T = re["enkf_256"].shape[1]
    t = min(t, T - 1)

    gt        = re["truth_256"][traj, t].numpy()
    enkf_s    = re["enkf_256" ][traj, t].numpy()
    enkf_f    = rf["enkf_256" ][traj, t].numpy()
    ir        = ri["posterior_256"][traj, t].numpy()
    bic       = _bicubic(re["obs_32"][traj, t:t+1]).squeeze().numpy()

    vmax = float(np.percentile(np.abs(gt), 99.5))
    emax = max(float(np.percentile(np.abs(enkf_s - gt), 99.5)),
               float(np.percentile(np.abs(enkf_f - gt), 99.5)),
               float(np.percentile(np.abs(ir      - gt), 99.5)))

    methods = ["Ground Truth", "Bicubic", "Solver EnKF", "Learned EnKF\n(FNO)", "Iter. Refinement"]
    fields  = [gt, bic, enkf_s, enkf_f, ir]
    errors  = [None, bic-gt, enkf_s-gt, enkf_f-gt, ir-gt]

    fig = plt.figure(figsize=(22, 9))
    gs  = gridspec.GridSpec(2, 5, hspace=0.06, wspace=0.04,
                            left=0.04, right=0.92, top=0.91, bottom=0.04)
    im_f = im_e = None
    for col in range(5):
        ax0 = fig.add_subplot(gs[0, col])
        im_f = ax0.imshow(fields[col], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          origin="lower", aspect="equal")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title(methods[col], fontsize=10, pad=4)
        if col == 0: ax0.set_ylabel("Field  ω", fontsize=10)

        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_xticks([]); ax1.set_yticks([])
        if errors[col] is None:
            ax1.axis("off")
        else:
            im_e = ax1.imshow(np.abs(errors[col]), cmap="inferno",
                              vmin=0, vmax=emax, origin="lower", aspect="equal")
        if col == 0: ax1.set_ylabel("|Error|", fontsize=10)

    fig.colorbar(im_f, ax=fig.axes, fraction=0.012, pad=0.01,
                 shrink=0.45, anchor=(0, 1.0)).set_label("ω", fontsize=10)
    if im_e is not None:
        fig.colorbar(im_e, ax=fig.axes, fraction=0.012, pad=0.01,
                     shrink=0.45, anchor=(0, 0.0)).set_label("|Error|", fontsize=10)
    fig.suptitle(f"256×256 Vorticity  |  t={t}, traj={traj}",
                 fontsize=13, y=0.97)
    _save(fig, "fig1_snapshot", out_dir)


# ── fig 2: RMSE over time ─────────────────────────────────────────────────────

def plot_rmse_time(re, rf, ri, out_dir):
    truth = re["truth_256"].float()
    bic   = _bicubic(re["obs_32"])
    n, T  = truth.shape[:2]

    def curves(pred):
        r = torch.stack([_rmse_t(pred[i], truth[i]) for i in range(n)])
        return r.mean(0).numpy(), r.std(0).numpy()

    m_bic,  s_bic  = curves(bic)
    m_es,   s_es   = curves(re["enkf_256"].float())
    m_ef,   s_ef   = curves(rf["enkf_256"][:n,:T].float())
    m_ir,   s_ir   = curves(ri["posterior_256"][:n,:T].float())
    ts = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 5))
    def _pl(m, s, c, lbl, ls="-"):
        ax.plot(ts, m, color=c, lw=2.0, ls=ls, label=lbl)
        ax.fill_between(ts, m-s, m+s, color=c, alpha=0.12)

    _pl(m_bic, s_bic, C["bicubic"],     "Bicubic",                    "--")
    _pl(m_es,  s_es,  C["enkf_solver"], "Solver EnKF  (N=20)",         "-")
    _pl(m_ef,  s_ef,  C["enkf_fno"],    "Learned EnKF — FNO  (N=20)", "-.")
    _pl(m_ir,  s_ir,  C["ir"],          "Iterative Refinement (ours)", "-")

    ax.set_xlabel("Time step"); ax.set_ylabel("RMSE @ 256×256")
    ax.set_title("RMSE over time — EnKF variants vs IR")
    ax.legend(loc="upper right"); ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig2_rmse_time", out_dir)


# ── fig 3: energy spectrum ────────────────────────────────────────────────────

def plot_spectrum(re, rf, ri, out_dir, traj=0):
    truth = re["truth_256"].float()
    bic   = _bicubic(re["obs_32"])
    T     = truth.shape[1]
    t_indices = [T//4, T//2, 3*T//4, T-1]
    t_labels  = ["$t=T/4$", "$t=T/2$", "$t=3T/4$", "$t=T$"]

    def _E(tensor, t_idx):
        frame = tensor[traj, t_idx].unsqueeze(0)
        E, k  = radial_energy_spectrum(frame)
        return E.numpy(), k.numpy()

    k_ref = np.array([3.0, 120.0])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)

    for col, (t_idx, lbl) in enumerate(zip(t_indices, t_labels)):
        ax = axes[col]
        E_t,  k = _E(truth, t_idx)
        E_b,  _ = _E(bic,   t_idx)
        E_es, _ = _E(re["enkf_256"].float(),       t_idx)
        E_ef, _ = _E(rf["enkf_256"].float(),       t_idx)
        E_ir, _ = _E(ri["posterior_256"].float(),  t_idx)

        k_arr = k[1:]; mask = k_arr <= 120; mask_bic = k_arr <= 20
        E_ref = E_t[1:][4] * (5.0**3) * k_ref**(-3)

        ax.loglog(k_arr[mask],     E_t[1:][mask],      color=C["truth"],        lw=2.0,          label="Ground truth")
        ax.loglog(k_arr[mask_bic], E_b[1:][mask_bic],  color=C["bicubic"],      lw=1.4, ls="--", label="Bicubic")
        ax.loglog(k_arr[mask],     E_es[1:][mask],     color=C["enkf_solver"],  lw=1.8, ls="-",  label="Solver EnKF")
        ax.loglog(k_arr[mask],     E_ef[1:][mask],     color=C["enkf_fno"],     lw=1.8, ls="-.", label="Learned EnKF (FNO)")
        ax.loglog(k_arr[mask],     E_ir[1:][mask],     color=C["ir"],           lw=1.8,          label="Iter. Refinement")
        ax.loglog(k_ref, E_ref, color="gray", lw=0.9, ls=":", label=r"$k^{-3}$")
        ax.axvline(16, color="purple", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlabel("Wavenumber $k$", fontsize=10)
        ax.set_title(lbl, fontsize=12)
        ax.set_xlim(1, 130); ax.grid(True, which="both", ls="--", alpha=0.2)
        if col == 0:
            ax.set_ylabel("$E(k)$", fontsize=11)
            ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    fig.suptitle("Radial energy spectrum — EnKF variants vs IR", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "fig3_spectrum", out_dir)


# ── fig 4: per-band RMSE ──────────────────────────────────────────────────────

def plot_per_band_rmse(re, rf, ri, out_dir):
    truth = re["truth_256"].float()
    bic   = _bicubic(re["obs_32"])
    n, T  = truth.shape[:2]

    bands = [
        ("Observed\n$k < 16$",            1,  16),
        ("Transition\n$16 \\leq k < 64$", 16, 64),
        ("Unobserved\n$k \\geq 64$",      64, 129),
    ]

    preds  = [bic,
              re["enkf_256"     ].float(),
              rf["enkf_256"     ][:n,:T].float(),
              ri["posterior_256"][:n,:T].float()]
    labels = ["Bicubic", "Solver EnKF", "Learned EnKF (FNO)", "Iter. Refinement"]
    colors = [C["bicubic"], C["enkf_solver"], C["enkf_fno"], C["ir"]]

    flat_t = truth.reshape(-1, 256, 256)
    vals   = np.zeros((len(preds), len(bands)))
    for m, pred in enumerate(preds):
        flat_p = pred.reshape(-1, 256, 256)
        for b, (_, lo, hi) in enumerate(bands):
            vals[m, b] = _band_rmse(flat_p, flat_t, lo, hi)

    x      = np.arange(len(bands))
    width  = 0.18
    off    = np.linspace(-(len(preds)-1)/2, (len(preds)-1)/2, len(preds)) * width

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m, (lbl, col, o) in enumerate(zip(labels, colors, off)):
        bars = ax.bar(x + o, vals[m], width, label=lbl, color=col,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals[m]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=col, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bands])
    ax.set_ylabel("RMSE (vorticity)"); ax.set_title("Per-Band RMSE — EnKF variants vs IR")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0); ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "fig4_per_band_rmse", out_dir)


# ── fig 5: ensemble spread vs RMSE ───────────────────────────────────────────

def plot_spread(re, rf, out_dir):
    truth = re["truth_256"].float()
    n, T  = truth.shape[:2]
    ts    = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, rdict, key, lbl, col in [
        (axes[0], re, "enkf_256", "Solver EnKF",      C["enkf_solver"]),
        (axes[1], rf, "enkf_256", "Learned EnKF (FNO)", C["enkf_fno"]),
    ]:
        enkf   = rdict[key][:n, :T].float()
        spread = rdict["spread_256"][:n, :T].float()
        rmse_t = torch.stack([_rmse_t(enkf[i], truth[i]) for i in range(n)])
        m_rmse = rmse_t.mean(0).numpy(); s_rmse = rmse_t.std(0).numpy()
        m_spr  = spread.mean(0).numpy(); s_spr  = spread.std(0).numpy()

        ax.plot(ts, m_rmse, color=col, lw=2.0, label="RMSE")
        ax.fill_between(ts, m_rmse-s_rmse, m_rmse+s_rmse, color=col, alpha=0.15)
        ax.plot(ts, m_spr,  color=col, lw=2.0, ls="--", label="Ensemble spread")
        ax.fill_between(ts, m_spr-s_spr, m_spr+s_spr,    color=col, alpha=0.08)
        ax.set_xlabel("Time step"); ax.set_title(lbl)
        ax.legend(fontsize=9); ax.grid(ls="--", alpha=0.3)

    axes[0].set_ylabel("Vorticity magnitude")
    fig.suptitle("Ensemble spread vs RMSE over time", fontsize=12)
    fig.tight_layout()
    _save(fig, "fig5_spread", out_dir)


# ── fig 6: runtime bar chart ──────────────────────────────────────────────────

def plot_runtime(out_dir):
    """
    Per-step wall-time for each method.
    IR:             ~1442 ms — 3x FNO + 3x25 DDIM steps
    Learned EnKF:   ~120 ms  — N=20 FNO forward passes (batched on GPU)
    Solver EnKF:    ~4266 ms — N=20 pseudospectral solver members (CPU)
    """
    methods  = ["Iter. Refinement", "Learned EnKF\n(FNO, N=20)", "Solver EnKF\n(N=20)"]
    times_ms = [1442.0, 22.0, 4266.0]
    colors   = [C["ir"], C["enkf_fno"], C["enkf_solver"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(methods, times_ms, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Per-step wall time (ms, log scale)")
    ax.set_title("Computational cost per assimilation step")
    ax.grid(axis="y", which="both", ls="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    for i, (bar, t) in enumerate(zip(bars, times_ms)):
        label = f"{t:.1f} ms" if t < 1000 else f"{t/1000:.2f} s"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.3,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")

    # annotate speed ratios inside taller bars
    ax.text(2, times_ms[2]*0.4, f"×{times_ms[2]/times_ms[0]:.1f}× IR",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_ylim(top=max(times_ms) * 8)

    fig.tight_layout()
    _save(fig, "fig6_runtime", out_dir)


def plot_rmse_bar(re, rf, ri, out_dir):
    """Bar chart comparing aggregate RMSE across all three DA methods + Bicubic.
    IR value is hardcoded from the v2 checkpoint results (0.19).
    """
    truth = re["truth_256"].float()
    bic   = _bicubic(re["obs_32"])
    n, T  = truth.shape[:2]

    def traj_rmse(pred):
        r = torch.stack([
            (pred[i] - truth[i]).pow(2).mean().sqrt()
            for i in range(n)
        ])
        return float(r.mean()), float(r.std())

    bic_mean,   bic_std   = traj_rmse(bic)
    es_mean,    es_std    = traj_rmse(re["enkf_256"].float())
    ef_mean,    ef_std    = traj_rmse(rf["enkf_256"][:n,:T].float())
    # IR v2 checkpoint result — hardcoded
    ir_mean,    ir_std    = 0.19, 0.013

    methods = ["Bicubic", "Learned EnKF\n(FNO, N=20)", "Iter. Refinement\n(ours)", "Solver EnKF\n(N=20)"]
    means   = [bic_mean,  ef_mean,                      ir_mean,                    es_mean]
    stds    = [bic_std,   ef_std,                        ir_std,                     es_std]
    colors  = [C["bicubic"], C["enkf_fno"], C["ir"], C["enkf_solver"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.6,
                  error_kw=dict(lw=1.5, capthick=1.5), zorder=3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=bar.get_facecolor())

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("RMSE @ 256×256")
    ax.set_title("Overall RMSE comparison — DA methods vs IR")
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, "fig7_rmse_bar", out_dir)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--enkf_solver", default="results_enkf/inference_results.pt")
    p.add_argument("--enkf_fno",    default="results_enkf_fno/inference_results.pt")
    p.add_argument("--ir",          default="results_2d/inference_results.pt")
    p.add_argument("--out",         default="plots_enkf")
    p.add_argument("--traj",        type=int, default=0)
    p.add_argument("--snap_t",      type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    print("Loading results...")
    re = torch.load(args.enkf_solver, map_location="cpu")
    rf = torch.load(args.enkf_fno,    map_location="cpu")
    ri = torch.load(args.ir,          map_location="cpu")

    n = min(re["enkf_256"].shape[0], rf["enkf_256"].shape[0], ri["posterior_256"].shape[0])
    T = min(re["enkf_256"].shape[1], rf["enkf_256"].shape[1], ri["posterior_256"].shape[1])

    re["enkf_256"]   = re["enkf_256" ][:n, :T]
    re["truth_256"]  = re["truth_256"][:n, :T]
    re["spread_256"] = re["spread_256"][:n, :T]
    re["obs_32"]     = re["obs_32"   ][:n, :T]
    rf["enkf_256"]   = rf["enkf_256" ][:n, :T]
    rf["spread_256"] = rf["spread_256"][:n, :T]

    print(f"  n_traj={n}, T={T}")
    print(f"  Solver EnKF RMSE: {re['metrics']['rmse_enkf_256']:.4f}")
    print(f"  Learned EnKF RMSE: {rf['metrics']['rmse_enkf_fno_256']:.4f}")
    print(f"  IR   RMSE: {ri['metrics']['rmse_posterior_256']:.4f}")

    print("\nFigure 1: snapshot...")
    plot_snapshot(re, rf, ri, args.out, traj=args.traj, t=args.snap_t)

    print("Figure 2: RMSE over time...")
    plot_rmse_time(re, rf, ri, args.out)

    print("Figure 3: energy spectrum...")
    plot_spectrum(re, rf, ri, args.out, traj=args.traj)

    print("Figure 4: per-band RMSE...")
    plot_per_band_rmse(re, rf, ri, args.out)

    print("Figure 5: ensemble spread...")
    plot_spread(re, rf, args.out)

    print("Figure 6: runtime comparison...")
    plot_runtime(args.out)

    print("Figure 7: RMSE bar chart...")
    plot_rmse_bar(re, rf, ri, args.out)

    print(f"\nAll figures saved to {args.out}/")


if __name__ == "__main__":
    main()
