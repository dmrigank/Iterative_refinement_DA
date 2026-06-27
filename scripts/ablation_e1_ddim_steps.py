"""
Ablation E1: DDIM step count sweep.

Runs inference at multiple DDIM step counts for both the 1D Burgers and 2D
Kraichnan pipelines, saving results to tagged subdirectories so that the
plotting script can compare them without overwriting the default results.

1D steps to sweep:  [10, 25, 50, 100]
2D steps to sweep:  [25, 50, 100, 200]

Output directories:
  results/ablation_ddim_steps_{N}/inference_results.pt     (1D)
  results_2d/ablation_ddim_steps_{N}/inference_results.pt  (2D)

Usage:
    python scripts/ablation_e1_ddim_steps.py [--skip_1d] [--skip_2d]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


STEPS_1D = [10, 25, 50, 100]
STEPS_2D = [25, 50, 100, 200]

PYTHON  = sys.executable
ROOT    = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> float:
    """Run a command and return wall-clock seconds."""
    print(f"\n  $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [WARNING] Command exited with code {result.returncode}")
    return elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation E1: DDIM step count sweep")
    p.add_argument("--skip_1d", action="store_true", help="Skip 1D Burgers sweeps")
    p.add_argument("--skip_2d", action="store_true", help="Skip 2D Kraichnan sweeps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Ablation E1 — DDIM step count sweep")
    print("=" * 60)

    timing: dict[str, float] = {}

    # ── 1D Burgers ────────────────────────────────────────────────────────────
    if not args.skip_1d:
        print("\n── 1D Burgers ──────────────────────────────────────────────")
        for steps in STEPS_1D:
            out_dir = f"results/ablation_ddim_steps_{steps}"
            print(f"\n  ddim_steps={steps}  ->  {out_dir}/")
            elapsed = run([
                PYTHON, "scripts/run_inference.py",
                "--ddim_steps",  str(steps),
                "--results_dir", out_dir,
            ])
            timing[f"1d_steps_{steps}"] = elapsed
            print(f"  Elapsed: {elapsed:.1f}s")
    else:
        print("\n  [skip_1d] Skipping 1D sweeps.")

    # ── 2D Kraichnan ──────────────────────────────────────────────────────────
    if not args.skip_2d:
        print("\n── 2D Kraichnan ────────────────────────────────────────────")
        for steps in STEPS_2D:
            out_dir = f"results_2d/ablation_ddim_steps_{steps}"
            print(f"\n  ddim_steps={steps}  ->  {out_dir}/")
            elapsed = run([
                PYTHON, "scripts/run_inference_2d.py",
                "--ddim_steps",  str(steps),
                "--results_dir", out_dir,
            ])
            timing[f"2d_steps_{steps}"] = elapsed
            print(f"  Elapsed: {elapsed:.1f}s")
    else:
        print("\n  [skip_2d] Skipping 2D sweeps.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Timing summary ──────────────────────────────────────────")
    for key, secs in timing.items():
        print(f"  {key:<30}  {secs:7.1f}s")

    # Save timing to a small file so the plotting script can read wall-clock data
    timing_path_1d = ROOT / "results" / "ablation_ddim_steps_timing.pt"
    timing_path_2d = ROOT / "results_2d" / "ablation_ddim_steps_timing.pt"

    import torch
    if not args.skip_1d:
        timing_path_1d.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k.removeprefix("1d_"): v for k, v in timing.items() if k.startswith("1d_")},
                   timing_path_1d)
        print(f"\n  Timing saved: {timing_path_1d}")
    if not args.skip_2d:
        timing_path_2d.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k.removeprefix("2d_"): v for k, v in timing.items() if k.startswith("2d_")},
                   timing_path_2d)
        print(f"  Timing saved: {timing_path_2d}")

    print("\nDone.")


if __name__ == "__main__":
    main()
