# Claude Code Prompt Sequence — One-Shot SR Baseline (1D Burgers)
# Implements a one-shot diffusion SR model that maps 64-pt → 512-pt directly,
# conditioned on the previous 512-pt state (temporal memory).
# Mirrors the 2D one-shot baseline but for the 1D Burgers testbed.
#
# RULES:
# - Do NOT modify any existing files
# - Create NEW files for the 1D one-shot baseline
# - Reuse existing shared modules: diffusion.py, embeddings.py
# - All 1D one-shot outputs go to checkpoints_oneshot_1d/, results_oneshot_1d/, plots_oneshot_1d/
# - Data comes from data/ (same 1D Burgers dataset, already generated)

# ===========================================================================
# PROMPT 1: Config, Dataset, U-Net, Training, Inference, and Plots
# ===========================================================================

"""
Read CLAUDE.md for context on the 1D Burgers testbed (Part 1). We are implementing
a ONE-SHOT diffusion SR baseline for 1D Burgers, consistent with the 2D one-shot
baseline we already built for Kraichnan.

The one-shot model maps 64-pt → 512-pt directly in a single diffusion pass,
conditioned on the previous 512-pt state. There is NO resolution hierarchy,
NO FNO forecaster, and NO resolution embedding.

Create these directories:
- checkpoints_oneshot_1d/
- results_oneshot_1d/
- plots_oneshot_1d/

Create configs/oneshot_sr_1d.yaml with these settings:

```yaml
data:
  data_dir: "data"
  train_trajectories: 40
  val_trajectories: 5
  test_trajectories: 5
  coarse_resolution: 64
  target_resolution: 512

unet:
  in_channels: 3  # [u_noisy, u_prev_512, u_obs_64_upsampled_to_512]
  base_channels: 64  # match 2D one-shot for fair capacity (iterative 1D uses 64 too)
  channel_mults: [1, 2, 4]
  n_res_blocks: 1
  use_attention: false  # 1D, no attention needed
  group_norm_groups: 8
  cond_embed_dim: 128

diffusion:
  T: 1000
  schedule: "cosine"

diffusion_training:
  steps: 150000  # same as 1D iterative refinement
  batch_size: 16
  lr: 2.0e-4
  weight_decay: 1.0e-5
  warmup_steps: 2000
  ema_decay: 0.9999
  log_interval: 500
  save_interval: 25000
  val_interval: 5000

inference:
  ddim_steps: 100  # MUST match iterative refinement settings
  eta: 1.0         # MUST match iterative refinement settings
  test_time_steps: 200

paths:
  checkpoint_dir: "checkpoints_oneshot_1d"
  results_dir: "results_oneshot_1d"
  plots_dir: "plots_oneshot_1d"

seed: 42
device: "auto"
```

Now create all the implementation files. Since 1D is simpler than 2D and the patterns
are established from the 2D one-shot baseline, implement everything in one pass.

=== File 1: src/data/dataset_oneshot_1d.py ===

Class OneShotDataset1d(Dataset):
- Loads data from data/ (the existing 1D Burgers dataset)
- Load train.pt or val.pt or test.pt which contain dict with keys "u_64", "u_128",
  "u_256", "u_512" each of shape (n_traj, T, N)
- Creates training triplets aligned as:
  - u_truth = u_512[:, 1:]     (target at time t, shape (n_traj, T-1, 512))
  - u_prev  = u_512[:, :-1]    (previous 512-pt state at time t-1)
  - u_obs   = u_64[:, 1:]      (coarse observation at time t)
- The coarse observation u_obs is spectrally upsampled to 512 points at construction time
  using the existing spectral_upsample function from src.data.dataset (import it)
- Flatten over trajectories: n_samples = n_traj * (T - 1)
- __getitem__ returns dict: "u_truth" (1, 512), "u_prev" (1, 512), "u_obs_up" (1, 512)
- Standard DataLoader with shuffle=True (single resolution, no custom sampler needed)

=== File 2: src/models/unet_oneshot_1d.py ===

Class OneShotUNet1d:
- Same architecture as the existing 1D ConditionalUNet1d from unet.py, but:
  - FiLM conditioning: ONLY diffusion timestep k (no resolution index)
  - Conditioning MLP: SinusoidalEmbedding(k) → MLP → cond_embed_dim
  - Import SinusoidalEmbedding from src.models.embeddings
  - Input: (batch, 3, 512) — channels [u_noisy, u_prev, u_obs_up]
  - Output: (batch, 1, 512) — predicted noise
  - Base channels 64, multipliers [1, 2, 4]
  - 1 residual block per level, no attention
  - GroupNorm with 8 groups
  - Standard padding (NOT circular — check what the existing 1D unet uses and match it;
    the existing 1D iterative model likely uses default zero padding since 1D Burgers
    results were already good)
- Write it as a standalone class, don't inherit from or modify the existing U-Net

=== File 3: src/training/train_oneshot_1d.py ===

Function train_oneshot_1d(cfg, device):
- Builds OneShotDataset1d for train and val
- Standard DataLoader with shuffle=True, batch_size=16
- Wraps OneShotUNet1d in GaussianDiffusion (reuse from src.models.diffusion)

For interfacing with GaussianDiffusion:
- Check how GaussianDiffusion.training_loss is called. It likely takes
  (x_0, w_forecast, w_coarse, res_idx). Pass:
  - x_0 = u_truth (the target)
  - w_forecast = u_prev (previous state as "forecast" slot)
  - w_coarse = u_obs_up (upsampled observation as "coarse" slot)
  - res_idx = torch.zeros(batch, dtype=torch.long) (dummy, model ignores it)
- The OneShotUNet1d forward signature should accept (x_concat, noise_step, res_idx)
  where res_idx is accepted but ignored, to maintain interface compatibility.

Training loop — same pattern as train_diffusion_2d.py:
- DDPM epsilon-prediction loss
- AdamW, lr=2e-4, weight_decay=1e-5
- Linear warmup (2k steps) + cosine decay
- EMA with decay 0.9999 (copy EMAModel class from train_diffusion_2d.py)
- Gradient clipping max_norm=1.0
- AMP mixed precision on CUDA
- Log every 500, validate every 5000, checkpoint every 25000
- Total: 150k steps
- Save EMA checkpoint to checkpoints_oneshot_1d/oneshot_ema.pt

=== File 4: src/inference/pipeline_oneshot_1d.py ===

Class OneShotPipeline1d:
- Constructor: trained diffusion model (OneShotUNet1d + GaussianDiffusion with EMA), config, device

Method run(observations_64, n_steps):
- observations_64: (T, 64) coarse observations
- Initialize: prev_512 = spectral_upsample(obs_64[0], 512) at t=0
- For each t >= 1:
  1. u_obs_up = spectral_upsample(obs_64[t], 512)
  2. posterior = ddim_sample(cond=[prev_512, u_obs_up], res_idx=0)
  3. prev_512 = posterior
  4. Store posterior
- Return dict: "posterior_512" (T, 512), "obs_64" (T, 64)

Method run_bicubic(observations_64, n_steps):
- Simply spectral upsample each 64-pt observation to 512-pt independently
- Return dict: "bicubic_512" (T, 512)

For ddim_sample, use the same interface trick as training: pass u_prev in the
w_forecast slot and u_obs_up in the w_coarse slot, dummy res_idx=0.

=== File 5: scripts/train_oneshot_1d.py ===

Entry point:
- Load config from configs/oneshot_sr_1d.yaml
- Call train_oneshot_1d(cfg, device)
- Print parameter count

=== File 6: scripts/run_inference_oneshot_1d.py ===

Entry point:
- Load config, load trained one-shot model (EMA)
- Load test data from data/
- Run one-shot pipeline on each test trajectory (5 trajectories × 200 steps)
- Run bicubic baseline
- Load iterative refinement results from results/ for comparison
- Save results to results_oneshot_1d/
- Print comparison RMSE table

=== File 7: scripts/plot_comparison_1d.py ===

Load results from results/ (iterative), results_oneshot_1d/ (one-shot + bicubic),
and data/ (ground truth). Save figures to plots_oneshot_1d/.

FIGURE 1 — Snapshot Comparison (plots_oneshot_1d/fig1_snapshot_comparison.png + .pdf):
- Single time step (e.g., t=100), 512-pt resolution
- One panel, overlaid curves:
  - Ground truth (black, solid, lw=2)
  - Bicubic / spectral upsample (gray, dashed)
  - One-Shot SR (green, solid)
  - Iterative Refinement (blue, solid)
  - Coarse observation (gray dots, 64 points)
- x-axis: spatial coordinate x ∈ [0, 2π], y-axis: u
- Legend, clean styling

FIGURE 2 — RMSE Over Time (plots_oneshot_1d/fig2_rmse_comparison.png + .pdf):
- At 512-pt resolution: RMSE vs time step
- Curves: Bicubic (gray dashed), FNO-only (red dotted), One-Shot SR (green),
  Iterative Refinement (blue)
- Mean as thick line, ± 1 std shaded envelope across test trajectories
- FNO-only may blow up — let it go off scale

FIGURE 3 — Energy Spectrum (plots_oneshot_1d/fig3_spectrum_comparison.png + .pdf):
- Log-log E(k) = |û_k|² vs wavenumber k at 512-pt
- Curves: Ground truth (black), Bicubic (gray dashed), One-Shot SR (green),
  Iterative Refinement (blue)
- k⁻² reference line (Burgers spectrum, NOT k⁻³)
- Average over time steps and test trajectories

FIGURE 4 — Hovmöller Comparison (plots_oneshot_1d/fig4_hovmoller_comparison.png + .pdf):
- 4 panels side by side at 512-pt: [Ground Truth, One-Shot SR, Iterative Refinement, Bicubic]
- x-axis: spatial x, y-axis: time step (show 100 steps)
- RdBu_r colormap, shared vmin/vmax

FIGURE 5 — Summary Bar Chart (plots_oneshot_1d/fig5_summary_bars.png + .pdf):
- Same format as the 2D version: 4 metric groups [RMSE, Spectral RMSE, Temp. Consistency, SSIM]
- 3 bars per group: Bicubic, One-Shot SR, Iterative Refinement
- Error bars from test trajectories

Print formatted comparison table to stdout.

Use these matplotlib settings:
```python
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'legend.fontsize': 11, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight', 'axes.grid': True, 'grid.alpha': 0.3,
})
```

Save all figures as PNG (300 dpi) and PDF.

=== After Implementation ===

Run the full pipeline:
1. python scripts/train_oneshot_1d.py          (~6-8 hours on 4090)
2. python scripts/run_inference_oneshot_1d.py   (~10 min)
3. python scripts/plot_comparison_1d.py         (~1 min)

CRITICAL CHECKS:
- Verify DDIM steps = 100 and eta = 1.0 match the iterative refinement inference settings.
  Check what scripts/run_inference.py uses for the iterative model and make sure they match
  EXACTLY. If the iterative model used different settings, either update oneshot to match
  or re-run iterative inference with matching settings. Inconsistent sampling settings
  invalidate the comparison (we learned this the hard way with 2D).
- Verify the one-shot model parameter count is comparable to or larger than the iterative
  model's total parameters (sum of FNO params + diffusion params). Print both counts.
- If one-shot RMSE is worse than bicubic, check: (a) sampling settings match, (b) EMA weights
  loaded correctly, (c) temporal conditioning is wired correctly (u_prev should be the
  model's own previous output at inference, not ground truth).
"""
