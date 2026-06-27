"""
Run 1D EDSR inference on the Burgers test set and compare against all methods.

Loads the best EDSR-1D checkpoint, runs frame-by-frame SR on each test
trajectory, then prints a comparison table alongside the iterative refinement
and one-shot results.

Usage:
    python scripts/run_inference_edsr_1d.py
        [--config   configs/edsr_1d.yaml]
        [--checkpoint checkpoints_edsr_1d/edsr_1d_best.pt]
        [--n_steps  N]
        [--iterative results/inference_results.pt]
        [--oneshot   results_oneshot_1d/inference_results.pt]

Output: results_edsr_1d/inference_results.pt
  Dict keys:
    'sr_512':      (n_traj, T, 512) — EDSR super-resolved field
    'truth_512':   (n_traj, T, 512) — ground truth
    'obs_64':      (n_traj, T, 64)  — coarse observations
    'trmse_curve': (T,)             — RMSE at each time step (for plotting)
    'metrics':     dict of metric_name -> value
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.edsr_1d import EDSR1d
from src.evaluation.metrics import rmse_over_time, temporal_consistency, spectral_rmse


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_edsr_1d(cfg, ckpt_path: Path, device: torch.device) -> EDSR1d:
    model = EDSR1d(
        n_resblocks = int(cfg.model.n_resblocks),
        n_feats     = int(cfg.model.n_feats),
        scale       = int(cfg.model.scale),
        res_scale   = float(cfg.model.res_scale),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded EDSR-1D from {ckpt_path}")
    print(f"  Parameters: {n_params:,}  |  step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    return model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(n_traj, T, N) -> (n_traj,) mean-over-time spatial RMSE."""
    return (pred - truth).pow(2).mean(dim=2).sqrt().mean(dim=1)


def _temporal_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(n_traj, T, N) -> (T,) RMSE at each step averaged over trajectories."""
    return (pred - truth).pow(2).mean(dim=2).sqrt().mean(dim=0)


def _mean_temp_consistency(pred: torch.Tensor) -> tuple[float, float]:
    """Per-trajectory mean frame-to-frame L2 -> (mean, std)."""
    n_traj = pred.shape[0]
    vals = torch.tensor([
        float(temporal_consistency(pred[i]).mean())
        for i in range(n_traj)
    ])
    return float(vals.mean()), float(vals.std()) if n_traj > 1 else 0.0


def _mean_spectral_rmse(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """Per-trajectory spectral RMSE (mean over modes) -> (mean, std)."""
    n_traj = pred.shape[0]
    vals = torch.tensor([
        float(spectral_rmse(pred[i], truth[i]).mean())
        for i in range(n_traj)
    ])
    return float(vals.mean()), float(vals.std()) if n_traj > 1 else 0.0


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 1D EDSR SR inference")
    p.add_argument("--config",     type=str, default="configs/edsr_1d.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override checkpoint (default: best in checkpoint_dir)")
    p.add_argument("--n_steps",    type=int, default=None,
                   help="Time steps per trajectory (default: cfg.inference.test_time_steps)")
    p.add_argument("--iterative",  type=str, default="results/inference_results.pt",
                   help="Path to 1D iterative refinement results")
    p.add_argument("--oneshot",    type=str, default="results_oneshot_1d/inference_results.pt",
                   help="Path to 1D one-shot SR results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if str(cfg.device) != "auto":
        device_str = str(cfg.device)
    device = torch.device(device_str)
    print(f"Device: {device}")

    n_steps = (args.n_steps if args.n_steps is not None
               else int(cfg.inference.test_time_steps))

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        best_path  = ckpt_dir / "edsr_1d_best.pt"
        final_path = ckpt_dir / "edsr_1d_final.pt"
        if best_path.exists():
            ckpt_path = best_path
        elif final_path.exists():
            ckpt_path = final_path
        else:
            step_ckpts = sorted(ckpt_dir.glob("edsr_1d_step_*.pt"))
            if not step_ckpts:
                print(f"ERROR: No EDSR-1D checkpoint found in {ckpt_dir}.")
                print("  Run: python scripts/train_edsr_1d.py")
                sys.exit(1)
            ckpt_path = step_ckpts[-1]

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading EDSR-1D model...")
    model = _load_edsr_1d(cfg, ckpt_path, device)

    # ── Load test data ────────────────────────────────────────────────────────
    data_dir  = Path(cfg.data.data_dir)
    test_data = torch.load(data_dir / "test.pt", map_location="cpu", weights_only=True)
    n_test    = test_data["u_64"].shape[0]
    print(f"\nTest trajectories: {n_test}  ({n_steps} steps each)")

    # ── Run inference: frame-by-frame, no temporal context ────────────────────
    all_sr:    list[torch.Tensor] = []
    all_truth: list[torch.Tensor] = []
    all_obs64: list[torch.Tensor] = []

    infer_batch = 256   # 1D sequences are cheap; process many at once

    for traj_i in tqdm(range(n_test), desc="Trajectories", ncols=80):
        obs_64  = test_data["u_64" ][traj_i, :n_steps].float()   # (T, 64)
        truth   = test_data["u_512"][traj_i, :n_steps].float()   # (T, 512)

        T = obs_64.shape[0]
        sr_frames: list[torch.Tensor] = []

        for start in range(0, T, infer_batch):
            end   = min(start + infer_batch, T)
            batch = obs_64[start:end].unsqueeze(1).to(device)   # (B, 1, 64)
            with torch.no_grad():
                out = model(batch)                               # (B, 1, 512)
            sr_frames.append(out.squeeze(1).cpu())              # (B, 512)

        all_sr.append(torch.cat(sr_frames, dim=0))   # (T, 512)
        all_truth.append(truth)
        all_obs64.append(obs_64)

    sr_all    = torch.stack(all_sr,    dim=0)   # (n_traj, T, 512)
    truth_all = torch.stack(all_truth, dim=0)
    obs64_all = torch.stack(all_obs64, dim=0)

    # ── EDSR metrics ──────────────────────────────────────────────────────────
    pt_edsr       = _per_traj_rmse(sr_all, truth_all)
    rmse_edsr     = float(pt_edsr.mean())
    rmse_edsr_std = float(pt_edsr.std()) if n_test > 1 else 0.0

    tc_mean,   tc_std   = _mean_temp_consistency(sr_all)
    spec_mean, spec_std = _mean_spectral_rmse(sr_all, truth_all)
    trmse_edsr = _temporal_rmse(sr_all, truth_all)   # (T,)

    metrics: dict = {
        "rmse_edsr_512":          rmse_edsr,
        "rmse_edsr_std_512":      rmse_edsr_std,
        "temp_consistency_edsr":  tc_mean,
        "temp_consistency_edsr_std": tc_std,
        "spectral_rmse_edsr":     spec_mean,
        "spectral_rmse_edsr_std": spec_std,
    }

    # ── Load other results for comparison ─────────────────────────────────────
    ri, ro = None, None
    T_common = n_steps

    iter_path = Path(args.iterative)
    if iter_path.exists():
        ri = torch.load(iter_path, map_location="cpu", weights_only=True)
        T_common = min(T_common, ri["posterior_512"].shape[1])

    oneshot_path = Path(args.oneshot)
    if oneshot_path.exists():
        ro = torch.load(oneshot_path, map_location="cpu", weights_only=True)
        T_common = min(T_common, ro["posterior_512"].shape[1])

    sr_c    = sr_all[:,    :T_common]
    truth_c = truth_all[:, :T_common]

    def _gather(pred_c, truth_c):
        pt       = _per_traj_rmse(pred_c, truth_c)
        tc_m, tc_s   = _mean_temp_consistency(pred_c)
        sp_m, sp_s   = _mean_spectral_rmse(pred_c, truth_c)
        return (float(pt.mean()), float(pt.std()) if pred_c.shape[0] > 1 else 0.0,
                tc_m, tc_s, sp_m, sp_s)

    rows: list[tuple] = []

    # Bicubic from one-shot results
    if ro is not None and "bicubic_512" in ro:
        bic_c  = ro["bicubic_512"][:, :T_common]
        tr_c   = ro["truth_512"  ][:, :T_common]
        rows.append(("Bicubic (spectral upsample)",) + _gather(bic_c, tr_c))

    rows.append(("EDSR-1D (no temporal context)",) + _gather(sr_c, truth_c))

    if ro is not None:
        post_c = ro["posterior_512"][:, :T_common]
        tr_o   = ro["truth_512"    ][:, :T_common]
        rows.append(("One-Shot SR",) + _gather(post_c, tr_o))

    if ri is not None:
        # Handle both key names present in existing results files
        fno_key  = "fno_only_512" if "fno_only_512" in ri else "forecast_512"
        post_key = "posterior_512"
        tr_i = ri.get("truth_512", truth_c)[:n_test, :T_common]
        if fno_key in ri:
            fno_c = ri[fno_key][:n_test, :T_common]
            # Skip FNO if it has NaN/Inf (known divergence in 1D results)
            if torch.isfinite(fno_c).all():
                rows.append(("FNO-only (autoregressive)",) + _gather(fno_c, tr_i))
            else:
                print("  [INFO] FNO-only results contain non-finite values — skipped")
        if post_key in ri:
            iter_c = ri[post_key][:n_test, :T_common]
            rows.append(("Iterative Refinement (ours)",) + _gather(iter_c, tr_i))

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*88}")
    print(f"  RMSE Comparison @ 512-pt  ({n_test} trajectories × {T_common} steps)")
    print(f"{'='*88}")
    hdr = (f"  {'Method':<36}  {'RMSE':>8}  {'±':>6}  "
           f"{'Temp.Cons.':>10}  {'±':>6}  {'Spec.RMSE':>9}  {'±':>6}")
    print(hdr)
    print(f"  {'-'*84}")
    for label, rm, rm_s, tc, tc_s, sp, sp_s in rows:
        print(f"  {label:<36}  {rm:8.4f}  {rm_s:6.4f}  "
              f"{tc:10.4f}  {tc_s:6.4f}  {sp:9.4f}  {sp_s:6.4f}")
    print(f"{'='*88}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"

    torch.save(
        {
            "sr_512":      sr_all,
            "truth_512":   truth_all,
            "obs_64":      obs64_all,
            "trmse_curve": trmse_edsr,
            "metrics":     metrics,
        },
        save_path,
    )
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
