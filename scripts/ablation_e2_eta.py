"""
Ablation E2: Stochasticity η sweep.

Runs inference at multiple η values for both the 1D Burgers and 2D Kraichnan
pipelines.  η=0 is fully deterministic DDIM; η=1 recovers DDPM sampling.

η values: [0.0, 0.25, 0.5, 0.75, 1.0]

Output directories:
  results/ablation_eta_{v}/inference_results.pt       (1D, v = eta×100 as int, e.g. eta_000)
  results_2d/ablation_eta_{v}/inference_results.pt    (2D)

Usage:
    python scripts/ablation_e2_eta.py [--skip_1d] [--skip_2d]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ETA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]

PYTHON = sys.executable
ROOT   = Path(__file__).resolve().parent.parent


def _eta_tag(eta: float) -> str:
    """Convert 0.25 -> 'eta_025', 1.0 -> 'eta_100'."""
    return f"eta_{int(round(eta * 100)):03d}"


def run(cmd: list[str]) -> float:
    print(f"\n  $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"  [WARNING] Command exited with code {result.returncode}")
    return elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation E2: stochasticity η sweep")
    p.add_argument("--skip_1d", action="store_true", help="Skip 1D Burgers sweeps")
    p.add_argument("--skip_2d", action="store_true", help="Skip 2D Kraichnan sweeps")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Ablation E2 — Stochasticity η sweep")
    print("=" * 60)

    timing: dict[str, float] = {}

    # ── 1D Burgers ────────────────────────────────────────────────────────────
    if not args.skip_1d:
        print("\n── 1D Burgers ──────────────────────────────────────────────")
        for eta in ETA_VALUES:
            tag     = _eta_tag(eta)
            out_dir = f"results/ablation_{tag}"
            print(f"\n  eta={eta}  ->  {out_dir}/")
            elapsed = run([
                PYTHON, "scripts/run_inference.py",
                "--eta",         str(eta),
                "--results_dir", out_dir,
            ])
            timing[f"1d_{tag}"] = elapsed
            print(f"  Elapsed: {elapsed:.1f}s")
    else:
        print("\n  [skip_1d] Skipping 1D sweeps.")

    # ── 2D Kraichnan ──────────────────────────────────────────────────────────
    if not args.skip_2d:
        print("\n── 2D Kraichnan ────────────────────────────────────────────")
        for eta in ETA_VALUES:
            tag     = _eta_tag(eta)
            out_dir = f"results_2d/ablation_{tag}"
            print(f"\n  eta={eta}  ->  {out_dir}/")
            elapsed = run([
                PYTHON, "scripts/run_inference_2d.py",
                "--eta",         str(eta),
                "--results_dir", out_dir,
            ])
            timing[f"2d_{tag}"] = elapsed
            print(f"  Elapsed: {elapsed:.1f}s")
    else:
        print("\n  [skip_2d] Skipping 2D sweeps.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Timing summary ──────────────────────────────────────────")
    for key, secs in timing.items():
        print(f"  {key:<30}  {secs:7.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
