# Resolution-Agnostic Iterative Generative Data Assimilation

A PyTorch implementation of iterative refinement data assimilation using conditional diffusion models, demonstrated on two testbeds: 1D stochastic Burgers equation and 2D Kraichnan turbulence.

## Method

Classical data assimilation recovers high-resolution states from low-resolution observations via a forecast-then-correct cycle. We learn this cycle using neural networks:

1. **Forecaster F** (FNO): Advances the state one time step at a given resolution
2. **Corrector G** (Conditional Diffusion U-Net): Refines the forecast using the coarser-resolution observation

A single shared diffusion model G operates across all resolution levels, conditioned on a resolution embedding. This enables **iterative refinement** through a resolution hierarchy, where each stage's posterior becomes the conditioning signal for the next.

## Testbeds

### 1D Stochastic Burgers Equation (Proof of Concept)

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} + f(x,t)$$

- Domain: [0, 2π], periodic BCs, ν = 5×10⁻⁴
- Resolution hierarchy: 64 → 128 → 256 → 512
- Stochastic OU forcing (16 modes) maintains statistical stationarity with persistent shocks

### 2D Kraichnan Turbulence (Primary Testbed)

$$\frac{\partial \omega}{\partial t} + J(\psi, \omega) = \nu \nabla^2 \omega - \mu \omega + f(x, y, t)$$

- Domain: [0, 2π]², periodic BCs, ν = 10⁻³, Ekman drag μ = 0.05
- Resolution hierarchy: 32×32 → 64×64 → 128×128 → 256×256
- Band-limited stochastic forcing (k_f = 4) with OU temporal correlation
- Direct enstrophy cascade (E(k) ~ k⁻³) creates filamentary structures that are genuinely unresolved at coarse scales

The 2D testbed is the main scientific contribution. The dual cascade of 2D turbulence creates energetically significant structure at all scales, making the super-resolution task genuinely ill-posed and the resolution hierarchy meaningful at every stage.

## Resolution Hierarchies

**1D Burgers:**

| Level | Grid Points | Role |
|-------|-------------|------|
| r=0 | 64 | Observations (input) |
| r=1 | 128 | Intermediate reconstruction |
| r=2 | 256 | Intermediate reconstruction |
| r=3 | 512 | Final high-resolution output |

**2D Kraichnan:**

| Level | Grid | Role |
|-------|------|------|
| r=0 | 32×32 | Observations — forcing scale barely resolved |
| r=1 | 64×64 | Intermediate reconstruction |
| r=2 | 128×128 | Intermediate reconstruction |
| r=3 | 256×256 | Final high-resolution output |

## Quick Start

### 1D Pipeline

```bash
pip install -e ".[dev]"
python scripts/generate_data.py
python scripts/train.py
python scripts/run_inference.py
python scripts/plot_results.py
```

### 2D Pipeline

```bash
# Generate 2D Kraichnan turbulence data (slow — ~6 hours on CPU, start first)
python scripts/generate_kraichnan_data.py

# Train 2D models
python scripts/train_2d.py                  # Full pipeline: FNOs → forecasts → diffusion
python scripts/train_2d.py --stage fno      # FNOs only
python scripts/train_2d.py --stage diffusion # Diffusion only (requires FNO checkpoints)

# Inference and evaluation
python scripts/run_inference_2d.py
python scripts/plot_results_2d.py
```

## Project Structure

```
├── configs/
│   ├── default.yaml              # 1D hyperparameters
│   └── kraichnan.yaml            # 2D hyperparameters
├── src/
│   ├── data/
│   │   ├── solver.py             # 1D Burgers solver (batched PyTorch GPU)
│   │   ├── kraichnan_generator.py# 2D Kraichnan solver (NumPy CPU)
│   │   ├── generate_dataset.py   # 1D data generation + resolution pyramid
│   │   ├── generate_kraichnan.py # 2D data generation + resolution pyramid
│   │   ├── dataset.py            # 1D PyTorch Datasets
│   │   └── dataset_2d.py         # 2D PyTorch Datasets
│   ├── models/
│   │   ├── fno.py                # 1D FNO (SpectralConv1d)
│   │   ├── fno_2d.py             # 2D FNO (SpectralConv2d)
│   │   ├── unet.py               # 1D conditional U-Net
│   │   ├── unet_2d.py            # 2D conditional U-Net (circular padding)
│   │   ├── diffusion.py          # GaussianDiffusion (shared, dimension-agnostic)
│   │   └── embeddings.py         # Embeddings + FiLM (shared)
│   ├── training/
│   │   ├── train_fno.py          # 1D FNO training
│   │   ├── train_fno_2d.py       # 2D FNO training
│   │   ├── train_diffusion.py    # 1D diffusion training
│   │   └── train_diffusion_2d.py # 2D diffusion training
│   ├── inference/
│   │   ├── pipeline.py           # 1D iterative refinement
│   │   └── pipeline_2d.py        # 2D iterative refinement
│   └── evaluation/
│       ├── metrics.py            # 1D metrics
│       └── metrics_2d.py         # 2D metrics (radial spectrum, SSIM, etc.)
├── scripts/
│   ├── generate_data.py          # 1D entry point
│   ├── generate_kraichnan_data.py# 2D entry point
│   ├── train.py                  # 1D training
│   ├── train_2d.py               # 2D training
│   ├── run_inference.py          # 1D inference
│   ├── run_inference_2d.py       # 2D inference
│   ├── plot_results.py           # 1D figures
│   └── plot_results_2d.py        # 2D figures
├── data/                         # 1D data
├── data_2d/                      # 2D data
│   └── fno_forecasts/
├── checkpoints/                  # 1D weights
├── checkpoints_2d/               # 2D weights
├── results/                      # 1D outputs
├── results_2d/                   # 2D outputs
├── plots/                        # 1D figures
└── plots_2d/                     # 2D figures
```

## Key Design Choices

### Circular padding (2D only)
All Conv2d layers in the 2D U-Net use `padding_mode='circular'` since the domain is doubly periodic. Vortex filaments wrap around boundaries — zero padding creates artificial discontinuities.

### No innovation channel
The corrector receives the forecast and upsampled observation as separate channels — it can learn to compute the mismatch implicitly. 3 input channels: [noisy target, forecast, coarse-up].

### No temporal embedding
Both testbeds use stationary stochastic forcing, so physical time carries no useful signal. Only diffusion noise level and resolution index are used for FiLM conditioning.

### Strategy B training
The diffusion model trains on actual FNO forecast outputs (teacher-forced), not ground truth, eliminating train-test distribution mismatch.

### Spectral up/downsampling
All resolution transfers use Fourier-space truncation/zero-padding. 1D uses `rfft/irfft`, 2D uses `rfft2/irfft2`. Amplitude scaling: `(target/source)` for 1D, `(target/source)²` for 2D.

### Self-attention (2D only)
Applied at the U-Net bottleneck (32×32) to capture long-range vortex interactions. Not used at higher resolutions due to quadratic memory cost.

## Hardware

Designed to train end-to-end on a single RTX 4090 (24GB). The 1D pipeline completes in under a day. The 2D pipeline is roughly: ~6 hours data generation (CPU), ~2 hours FNO training, ~12–18 hours diffusion training.

## References

- Muthukumar & Willett, "Resolution-Agnostic Iterative Generative Data Assimilation" (2025)
- Huang et al., "DiffDA: a Diffusion Model for Weather-scale Data Assimilation" (2024)
- Li et al., "Fourier Neural Operator for Parametric PDEs" (2021)
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Song et al., "Denoising Diffusion Implicit Models" (2021)
- Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)
