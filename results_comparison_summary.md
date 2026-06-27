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

---

## Model Parameter Counts

Parameter counts for every learned component used in each testbed.

### 1D Stochastic Burgers  (64 → 128 → 256 → 512)

| Component | Role | Params |
|---|---|---:|
| FNO-128 | Forecaster, stage 0 (128-pt) | 200,929 |
| FNO-256 | Forecaster, stage 1 (256-pt) | 397,537 |
| FNO-512 | Forecaster, stage 2 (512-pt) | 790,753 |
| **FNO total (3 models)** | | **1,389,219** |
| Shared diffusion U-Net | Corrector G, all 3 stages | 2,412,353 |
| **Iterative Refinement total** | FNOs + diffusion | **3,801,572** |
| | | |
| One-shot U-Net | Direct 64→512 diffusion SR | 2,280,641 |
| EDSR-1D | Deterministic SR, 64→512 | 445,121 |

Notes:
- The shared diffusion U-Net is used at all three resolution transitions (Strategy B); its parameter count is **not** multiplied by 3.
- The 1D FNO weights scale roughly as `width² × k_max` per Fourier layer; FNO-512 is larger because `k_max = 512/4 = 128`.
- EDSR-1D uses nearest-neighbour + Conv1d upsampling (×2 × 3 stages); no pixel-shuffle.

### 2D Kraichnan Turbulence  (32 → 64 → 128 → 256)

| Component | Role | Params |
|---|---|---:|
| FNO-64 | Forecaster, stage 0 (64×64) | 4,199,137 |
| FNO-128 | Forecaster, stage 1 (128×128) | 16,782,049 |
| FNO-256 | Forecaster, stage 2 (256×256) | 67,113,697 |
| **FNO total (3 models)** | | **88,094,883** |
| Shared diffusion U-Net | Corrector G, all 3 stages | 4,171,537 |
| **Iterative Refinement total** | FNOs + diffusion | **92,266,420** |
| | | |
| One-shot U-Net | Direct 32→256 diffusion SR | 6,970,689 |
| EDSR-2D | Deterministic SR, 32→256 via PixelShuffle | 1,662,977 |

Notes:
- The 2D FNO parameter counts are large because `k_max = N/4` in each dimension, giving `SpectralConv2d` weight tensors of shape `(width, width, k_max, k_max)`. At 256×256 this is `32×32×64×64 = 4.2M` parameters **per layer** (× 4 layers × 4 real tensors = ~67M for FNO-256). This is a known property of the FNO-2D architecture at high resolution.
- The shared diffusion U-Net (4.2M) is much smaller than any individual FNO at 128×128 or 256×256, and is shared across all three resolution transitions.
- EDSR-2D uses sub-pixel convolution (PixelShuffle) for upsampling; no temporal context.
