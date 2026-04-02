"""
Evaluate iterative refinement inference results against ground truth.

Loads results/inference_results.pt and computes all metrics from
src/evaluation/metrics.py, then prints a formatted table and saves a
JSON summary to results/metrics.json.

Metrics computed (per resolution where applicable)
───────────────────────────────────────────────────
  rmse                  — overall RMSE across all trajectories and time steps
  rmse_over_time        — mean ± std of per-timestep RMSE across trajectories
  spectral_rmse         — per-mode RMSE (saved to metrics.json, not printed)
  shock_position_error  — mean shock position error in grid-index units
  temporal_consistency  — mean frame-to-frame L2 displacement
  crps                  — if n_samples > 1 in results (ensemble CRPS)

Each metric is computed for:
  - diffusion posterior
  - FNO-only autoregressive baseline
  - FNO forecast (the prior before diffusion correction)

Usage
─────
    python scripts/evaluate.py [--config configs/default.yaml]
                               [--results results/inference_results.pt]
                               [--resolutions 128,256,512]
                               [--no_shock]   skip shock_position_error (slow)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf

from src.evaluation.metrics import (
    rmse,
    rmse_over_time,
    spectral_rmse,
    shock_position_error,
    temporal_consistency,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat(x: torch.Tensor) -> torch.Tensor:
    """Flatten (n_traj, T, N) -> (n_traj*T, N) for scalar metrics."""
    n, t, n_pts = x.shape
    return x.reshape(n * t, n_pts)


def _per_traj_rmse(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """Mean and std of per-trajectory RMSE. pred/truth: (n_traj, T, N)."""
    n_traj = pred.shape[0]
    vals = torch.stack(
        [rmse(pred[i], truth[i]) for i in range(n_traj)]
    )  # (n_traj,)
    return float(vals.mean()), float(vals.std())


def _per_traj_shock(pred: torch.Tensor, truth: torch.Tensor) -> tuple[float, float]:
    """Mean and std of per-trajectory shock_position_error. pred/truth: (n_traj, T, N)."""
    n_traj = pred.shape[0]
    vals = torch.stack(
        [shock_position_error(pred[i], truth[i]) for i in range(n_traj)]
    )
    return float(vals.mean()), float(vals.std())


def _per_traj_tc(pred: torch.Tensor) -> tuple[float, float]:
    """Mean and std of per-trajectory mean temporal consistency. pred: (n_traj, T, N)."""
    n_traj = pred.shape[0]
    vals = torch.stack(
        [temporal_consistency(pred[i]).mean() for i in range(n_traj)]
    )
    return float(vals.mean()), float(vals.std())


def _format_row(label: str, mean: float, std: float) -> str:
    return f"  {label:<35s}  {mean:>12.6f}  ±  {std:>10.6f}"


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    results: dict,
    resolutions: list[int],
    compute_shock: bool,
) -> dict:
    """Run all metrics and return a nested dict ready for JSON serialisation."""
    output: dict = {}

    for res in resolutions:
        print(f"\n{'─'*65}")
        print(f"  Resolution {res}")
        print(f"{'─'*65}")
        print(f"  {'Metric':<35s}  {'Mean':>12s}     {'Std':>10s}")
        print(f"  {'-'*35}  {'-'*12}     {'-'*10}")

        post_all  = results[f"posterior_{res}"]   # (n_traj, T, N)
        fno_all   = results[f"fno_only_{res}"]
        fc_all    = results[f"forecast_{res}"]
        truth_all = results[f"truth_{res}"]

        res_metrics: dict = {}

        for tag, pred in [
            ("posterior", post_all),
            ("fno_only",  fno_all),
            ("forecast",  fc_all),
        ]:
            tag_metrics: dict = {}

            # ── RMSE ──────────────────────────────────────────────────────
            mean_r, std_r = _per_traj_rmse(pred, truth_all)
            tag_metrics["rmse_mean"] = mean_r
            tag_metrics["rmse_std"]  = std_r
            print(_format_row(f"{tag} RMSE", mean_r, std_r))

            # ── RMSE over time (store mean profile, report scalar) ─────────
            # Average per-timestep RMSE over trajectories -> (T,)
            n_traj = pred.shape[0]
            rmse_t = torch.stack(
                [rmse_over_time(pred[i], truth_all[i]) for i in range(n_traj)]
            ).mean(dim=0)  # (T,)
            tag_metrics["rmse_over_time"] = rmse_t.tolist()

            # ── Spectral RMSE ──────────────────────────────────────────────
            spec_err = spectral_rmse(pred, truth_all)   # (N//2+1,)
            tag_metrics["spectral_rmse"] = spec_err.tolist()

            # ── Temporal consistency ───────────────────────────────────────
            mean_tc, std_tc = _per_traj_tc(pred)
            tag_metrics["temporal_consistency_mean"] = mean_tc
            tag_metrics["temporal_consistency_std"]  = std_tc
            print(_format_row(f"{tag} temporal consistency", mean_tc, std_tc))

            # ── Shock position error ───────────────────────────────────────
            if compute_shock:
                mean_s, std_s = _per_traj_shock(pred, truth_all)
                tag_metrics["shock_position_error_mean"] = mean_s
                tag_metrics["shock_position_error_std"]  = std_s
                print(_format_row(f"{tag} shock pos. error", mean_s, std_s))

            res_metrics[tag] = tag_metrics

        # ── Improvement: posterior vs FNO-only ────────────────────────────
        improvement = (
            res_metrics["fno_only"]["rmse_mean"]
            - res_metrics["posterior"]["rmse_mean"]
        )
        rel_improvement = (
            improvement / res_metrics["fno_only"]["rmse_mean"] * 100.0
            if res_metrics["fno_only"]["rmse_mean"] > 0 else 0.0
        )
        res_metrics["rmse_improvement_absolute"] = improvement
        res_metrics["rmse_improvement_relative_pct"] = rel_improvement
        print(
            f"\n  {'RMSE improvement (post vs FNO-only)':<35s}"
            f"  {improvement:>+12.6f}   ({rel_improvement:+.2f}%)"
        )

        output[str(res)] = res_metrics

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference results")
    parser.add_argument("--config",      type=str, default="configs/default.yaml")
    parser.add_argument("--results",     type=str, default=None,
                        help="Path to inference_results.pt (default: cfg.paths.results_dir/inference_results.pt)")
    parser.add_argument("--resolutions", type=str, default="128,256,512",
                        help="Comma-separated resolutions to evaluate (default: 128,256,512)")
    parser.add_argument("--no_shock",    action="store_true",
                        help="Skip shock_position_error (can be slow for large T)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Path to save metrics JSON (default: results/metrics.json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    results_path = Path(
        args.results if args.results is not None
        else Path(cfg.paths.results_dir) / "inference_results.pt"
    )
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run scripts/run_inference.py first.")
        sys.exit(1)

    output_path = Path(
        args.output if args.output is not None
        else Path(cfg.paths.results_dir) / "metrics.json"
    )

    resolutions = [int(r) for r in args.resolutions.split(",")]

    print(f"Loading results from {results_path} ...")
    results = torch.load(results_path, map_location="cpu", weights_only=False)

    n_traj = results["posterior_512"].shape[0]
    T      = results["posterior_512"].shape[1]
    print(f"Trajectories: {n_traj}  |  Time steps: {T}")
    print(f"Resolutions:  {resolutions}")
    if args.no_shock:
        print("(shock_position_error skipped)")

    metrics = evaluate(results, resolutions, compute_shock=not args.no_shock)

    # ── Print top-level summary ───────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  Summary — posterior RMSE per resolution")
    print(f"{'═'*65}")
    print(f"  {'Resolution':>12}  {'Posterior RMSE':>16}  {'FNO-only RMSE':>15}  {'Improvement':>12}")
    print(f"  {'-'*12}  {'-'*16}  {'-'*15}  {'-'*12}")
    for res in resolutions:
        m = metrics[str(res)]
        print(
            f"  {res:>12d}  "
            f"{m['posterior']['rmse_mean']:>16.6f}  "
            f"{m['fno_only']['rmse_mean']:>15.6f}  "
            f"{m['rmse_improvement_relative_pct']:>+11.2f}%"
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
