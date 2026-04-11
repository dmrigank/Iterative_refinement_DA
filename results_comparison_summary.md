# Results Comparison Summary

Aligned RMSE comparison across four methods using the common overlapping subset in each case.

## 2D Case

Sources:
- `results_oneshot/inference_results.pt` for `Bicubic` and `One-Shot SR`
- `results_2d/inference_results.pt` for `Iterative Refinement` and `FNO-only forecaster`

Aligned evaluation subset:
- `3 trajectories x 100 steps @ 256x256`

| Case | Method | RMSE | Per-traj std |
|---|---:|---:|---:|
| 2D (256x256) | Bicubic | 0.335469 | 0.008368 |
| 2D (256x256) | One-Shot SR | 0.353179 | 0.119480 |
| 2D (256x256) | Iterative Refinement | 0.190911 | 0.003848 |
| 2D (256x256) | FNO-only forecaster | 2.568060 | 0.198535 |

## 1D Case

Sources:
- `results_oneshot_1d/inference_results.pt` for `Bicubic` and `One-Shot SR`
- `results/inference_results.pt` for `Iterative Refinement` and `FNO-only forecaster`

Aligned evaluation subset:
- `5 trajectories x 700 steps @ 512`

| Case | Method | RMSE | Per-traj std |
|---|---:|---:|---:|
| 1D (512) | Bicubic | 0.084944 | 0.017270 |
| 1D (512) | One-Shot SR | 0.006873 | 0.002098 |
| 1D (512) | Iterative Refinement | 0.005241 | 0.000402 |
| 1D (512) | FNO-only forecaster | NaN | NaN |

## Notes

- The `2D` one-shot results file stores `150` steps, but the iterative `2D` results file stores `100` steps, so the four-way comparison is aligned on the first `100` steps.
- The `1D` `FNO-only forecaster` contains non-finite values in `results/inference_results.pt`, so its RMSE and per-trajectory standard deviation evaluate to `NaN`.
