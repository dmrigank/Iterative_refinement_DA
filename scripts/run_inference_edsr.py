"""
Run EDSR inference on the 2D Kraichnan test set and compare against all methods.

Loads the best EDSR checkpoint, runs frame-by-frame SR on each test trajectory,
and prints a comparison table alongside existing results.

Usage:
    python scripts/run_inference_edsr.py [--config configs/edsr.yaml]
                                         [--checkpoint checkpoints_edsr/edsr_best.pt]
                                         [--n_steps N]
                                         [--iterative results_2d/inference_results.pt]
                                         [--oneshot   results_oneshot/inference_results.pt]

Output: results_edsr/inference_results.pt
  Dict keys:
    'sr_256':    (n_traj, T, 256, 256) — EDSR super-resolved vorticity
    'truth_256': (n_traj, T, 256, 256) — ground truth
    'obs_32':    (n_traj, T, 32,  32)  — coarse observations
    'metrics':   dict of metric_name -> value
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

from src.models.edsr import EDSR
from src.evaluation.metrics_2d import (
    rmse_over_time_2d,
    temporal_consistency_2d,
    structural_similarity_2d,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_edsr(cfg, ckpt_path: Path, device: torch.device) -> EDSR:
    model = EDSR(
        n_resblocks  = int(cfg.model.n_resblocks),
        n_feats      = int(cfg.model.n_feats),
        scale        = int(cfg.model.scale),
        res_scale    = float(cfg.model.res_scale),
        padding_mode = str(cfg.model.padding_mode),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded EDSR from {ckpt_path}")
    print(f"  Parameters: {n_params:,}  |  step={ckpt['step']}  val_loss={ckpt['val_loss']:.4f}")
    return model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(n_traj, T, H, W) -> (n_traj,) mean-over-time spatial RMSE."""
    return (pred - truth).pow(2).mean(dim=(2, 3)).sqrt().mean(dim=1)


def _temporal_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """(n_traj, T, H, W) -> (T,) RMSE at each time step, averaged over trajectories."""
    # per-step RMSE: (n_traj, T)
    per_step = (pred - truth).pow(2).mean(dim=(2, 3)).sqrt()
    return per_step.mean(dim=0)   # (T,)


def _mean_ssim(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """Per-trajectory SSIM -> (mean, std)."""
    n_traj = pred.shape[0]
    vals = torch.tensor([
        float(structural_similarity_2d(pred[i], truth[i]))
        for i in range(n_traj)
    ])
    return float(vals.mean()), float(vals.std()) if n_traj > 1 else 0.0


def _mean_temp_consistency(pred: torch.Tensor) -> tuple[float, float]:
    """Per-trajectory mean frame-to-frame L2 -> (mean, std)."""
    n_traj = pred.shape[0]
    vals = torch.tensor([
        float(temporal_consistency_2d(pred[i]).mean())
        for i in range(n_traj)
    ])
    return float(vals.mean()), float(vals.std()) if n_traj > 1 else 0.0


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EDSR SR inference")
    p.add_argument("--config",     type=str, default="configs/edsr.yaml")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override checkpoint path (default: best in checkpoint_dir)")
    p.add_argument("--n_steps",    type=int, default=None,
                   help="Time steps per trajectory (default: cfg.inference.test_time_steps)")
    p.add_argument("--iterative",  type=str,
                   default="results_2d/inference_results.pt",
                   help="Path to iterative refinement results for comparison")
    p.add_argument("--oneshot",    type=str,
                   default="results_oneshot/inference_results.pt",
                   help="Path to one-shot SR results for comparison")
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

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        best_path  = ckpt_dir / "edsr_best.pt"
        final_path = ckpt_dir / "edsr_final.pt"
        if best_path.exists():
            ckpt_path = best_path
        elif final_path.exists():
            ckpt_path = final_path
        else:
            step_ckpts = sorted(ckpt_dir.glob("edsr_step_*.pt"))
            if not step_ckpts:
                print(f"ERROR: No EDSR checkpoint found in {ckpt_dir}.")
                print("  Run: python scripts/train_edsr.py")
                sys.exit(1)
            ckpt_path = step_ckpts[-1]

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading EDSR model...")
    model = _load_edsr(cfg, ckpt_path, device)

    # ── Load test data ────────────────────────────────────────────────────────
    data_dir  = Path(cfg.data.data_dir)
    test_data = torch.load(data_dir / "test.pt", map_location="cpu", weights_only=True)
    n_test    = test_data["w_32"].shape[0]
    print(f"\nTest trajectories: {n_test}  ({n_steps} steps each)")

    # ── Run inference: frame-by-frame, no temporal context ────────────────────
    all_sr:    list[torch.Tensor] = []
    all_truth: list[torch.Tensor] = []
    all_obs32: list[torch.Tensor] = []

    infer_batch = 16   # process this many frames at once to avoid OOM at 256x256

    for traj_i in tqdm(range(n_test), desc="Trajectories", ncols=90):
        obs_32    = test_data["w_32" ][traj_i, :n_steps].float()   # (T, 32,  32)
        truth_256 = test_data["w_256"][traj_i, :n_steps].float()   # (T, 256, 256)

        T = obs_32.shape[0]
        sr_frames: list[torch.Tensor] = []

        for start in range(0, T, infer_batch):
            end   = min(start + infer_batch, T)
            batch = obs_32[start:end].unsqueeze(1).to(device)   # (B, 1, 32, 32)
            with torch.no_grad():
                out = model(batch)                               # (B, 1, 256, 256)
            sr_frames.append(out.squeeze(1).cpu())              # (B, 256, 256)

        all_sr.append(torch.cat(sr_frames, dim=0))   # (T, 256, 256)
        all_truth.append(truth_256)
        all_obs32.append(obs_32)

    # ── Assemble ──────────────────────────────────────────────────────────────
    sr_all    = torch.stack(all_sr,    dim=0)   # (n_traj, T, 256, 256)
    truth_all = torch.stack(all_truth, dim=0)
    obs32_all = torch.stack(all_obs32, dim=0)

    # ── EDSR metrics ──────────────────────────────────────────────────────────
    pt_edsr   = _per_traj_rmse(sr_all, truth_all)              # (n_traj,)
    rmse_edsr = float(pt_edsr.mean())
    rmse_edsr_std = float(pt_edsr.std()) if n_test > 1 else 0.0

    tc_edsr_mean, tc_edsr_std = _mean_temp_consistency(sr_all)
    ssim_edsr_mean, ssim_edsr_std = _mean_ssim(sr_all, truth_all)

    # Temporal RMSE curve (T,) — used for the table and saved for plotting
    trmse_edsr = _temporal_rmse(sr_all, truth_all)             # (T,)

    metrics: dict = {
        "rmse_edsr_256":          rmse_edsr,
        "rmse_edsr_std_256":      rmse_edsr_std,
        "temp_consistency_edsr":  tc_edsr_mean,
        "temp_consistency_edsr_std": tc_edsr_std,
        "ssim_edsr":              ssim_edsr_mean,
        "ssim_edsr_std":          ssim_edsr_std,
    }

    # ── Load other results for comparison ─────────────────────────────────────
    ri, ro = None, None
    T_common = n_steps

    iter_path = Path(args.iterative)
    if iter_path.exists():
        ri = torch.load(iter_path, map_location="cpu", weights_only=True)
        T_common = min(T_common, ri["posterior_256"].shape[1])

    oneshot_path = Path(args.oneshot)
    if oneshot_path.exists():
        ro = torch.load(oneshot_path, map_location="cpu", weights_only=True)
        T_common = min(T_common, ro["posterior_256"].shape[1])

    # Trim EDSR tensors to T_common for fair comparison
    sr_c    = sr_all[:,    :T_common]
    truth_c = truth_all[:, :T_common]

    def _gather(pred_c, truth_c):
        """Return (rmse, rmse_std, tc_mean, tc_std, ssim_mean, ssim_std)."""
        pt      = _per_traj_rmse(pred_c, truth_c)
        tc_m, tc_s   = _mean_temp_consistency(pred_c)
        ss_m, ss_s   = _mean_ssim(pred_c, truth_c)
        return (float(pt.mean()), float(pt.std()) if pred_c.shape[0] > 1 else 0.0,
                tc_m, tc_s, ss_m, ss_s)

    rows: list[tuple] = []   # (label, rmse, rmse_std, tc, tc_std, ssim, ssim_std)

    # Bicubic — sourced from oneshot results which computes it
    if ro is not None and "bicubic_256" in ro:
        bic_c = ro["bicubic_256"][:, :T_common]
        tr_c  = ro["truth_256"  ][:, :T_common]
        rows.append(("Spectral upsample (bicubic)",) + _gather(bic_c, tr_c))

    # EDSR (aligned to T_common)
    rows.append(("EDSR (no temporal context)",) + _gather(sr_c, truth_c))

    # One-shot diffusion SR
    if ro is not None:
        post_c  = ro["posterior_256"][:, :T_common]
        truth_o = ro["truth_256"    ][:, :T_common]
        rows.append(("One-Shot Diffusion SR",) + _gather(post_c, truth_o))

    # FNO-only + Iterative refinement
    if ri is not None:
        fno_c  = ri["fno_only_256" ][:, :T_common]
        post_c = ri["posterior_256"][:, :T_common]
        tr_i   = ri["truth_256"    ][:, :T_common]
        rows.append(("FNO-only (autoregressive)",)    + _gather(fno_c,  tr_i))
        rows.append(("Iterative Refinement (ours)",)  + _gather(post_c, tr_i))

    # ── Print full comparison table ───────────────────────────────────────────
    n_traj_s = sr_all.shape[0]
    print(f"\n{'='*90}")
    print(f"  Full Method Comparison @ 256×256  "
          f"({n_traj_s} trajectories × {T_common} steps)")
    print(f"{'='*90}")
    hdr = (f"  {'Method':<38}  {'RMSE':>8}  {'±':>6}  "
           f"{'Temp.Cons.':>10}  {'±':>6}  {'SSIM':>8}  {'±':>6}")
    print(hdr)
    print(f"  {'-'*86}")
    for label, rm, rm_s, tc, tc_s, ss, ss_s in rows:
        print(f"  {label:<38}  {rm:8.4f}  {rm_s:6.4f}  "
              f"{tc:10.4f}  {tc_s:6.4f}  {ss:8.4f}  {ss_s:6.4f}")
    print(f"{'='*90}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / "inference_results.pt"

    torch.save(
        {
            "sr_256":        sr_all,
            "truth_256":     truth_all,
            "obs_32":        obs32_all,
            "trmse_curve":   trmse_edsr,   # (T,) for plotting
            "metrics":       metrics,
        },
        save_path,
    )
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
