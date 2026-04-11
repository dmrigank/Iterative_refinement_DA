# Claude Code Prompt Sequence — One-Shot SR Baseline
# Implements a one-shot diffusion SR model that maps 32×32 → 256×256 directly,
# conditioned on the previous 256×256 state (temporal memory).
# This is the primary baseline for comparison against iterative refinement.
#
# RULES:
# - Do NOT modify any existing files (1D, 2D iterative refinement, or shared modules)
# - Create NEW files for the one-shot baseline
# - Reuse existing shared modules: diffusion.py, embeddings.py, dataset_2d.py (spectral ops)
# - All one-shot outputs go to checkpoints_oneshot/, results_oneshot/, plots_oneshot/
# - Data comes from data_2d/ (same Kraichnan dataset, already generated)

# ===========================================================================
# PROMPT 1: Scaffolding, Dataset, and U-Net
# ===========================================================================

"""
Read CLAUDE.md (especially the Part 2 section on 2D Kraichnan) for context. We are
implementing a ONE-SHOT diffusion SR baseline to compare against the iterative
refinement approach.

The one-shot model maps 32×32 → 256×256 directly in a single diffusion pass,
conditioned on the previous 256×256 state. There is NO resolution hierarchy,
NO FNO forecaster, and NO resolution embedding. This is a simpler architecture
that tests whether the multi-stage cascade adds value.

First, create directories:
- checkpoints_oneshot/
- results_oneshot/
- plots_oneshot/

Then create these new files (do NOT modify any existing files):

1. configs/oneshot_sr.yaml — check if it already exists, use it if so.

2. src/data/dataset_oneshot.py — OneShotDataset:
   - Loads data from data_2d/ (same Kraichnan data as iterative refinement)
   - Creates training triplets: (w_truth_256_t, w_prev_256_{t-1}, w_obs_32_t)
   - w_truth_256_t: ground truth 256×256 vorticity at time t (the target)
   - w_prev_256_{t-1}: ground truth 256×256 at time t-1 (temporal conditioning — 
     represents "previous reconstruction" — at training time we use GT, at inference 
     we'll use the model's own previous output)
   - w_obs_32_t: 32×32 coarse observation at time t, spectrally upsampled to 256×256
   - Each __getitem__ returns dict with keys: "w_truth" (1, 256, 256), 
     "w_prev" (1, 256, 256), "w_obs_up" (1, 256, 256)
   - Samples start at t=1 (need t-1 for previous state), so n_samples = n_traj * (T-1)
   - Import spectral_upsample_2d from src.data.dataset_2d
   - All spatial dims are 256×256 (single resolution), so no custom batch sampler needed —
     use a standard DataLoader with shuffle=True.

3. src/models/unet_oneshot.py — OneShotUNet2d:
   - Very similar to ConditionalUNet2d from unet_2d.py, but with key differences:
   - Input: (batch, 3, 256, 256) where channels are [w_noisy, w_prev, w_obs_up]
   - FiLM conditioning: ONLY diffusion timestep k. NO resolution index (there's only
     one resolution transition). The conditioning MLP is: SinusoidalEmbedding(k) → 
     MLP → cond_embed_dim vector → FiLM at each residual block.
   - All Conv2d layers use padding_mode='circular' (same as iterative U-Net)
   - Base channels: 64 (larger than iterative's 48 to give fair capacity comparison)
   - Channel multipliers: [1, 2, 4] → 64, 128, 256
   - 1 residual block per level
   - Self-attention at bottleneck (when spatial dim = 32)
   - GroupNorm with 8 groups
   - Output: (batch, 1, 256, 256) predicted noise
   - Import SinusoidalEmbedding from src.models.embeddings (reuse, don't reimplement)
   - DO NOT import or depend on ConditionalUNet2d — write a standalone class to avoid
     coupling, even though the structure is similar.

After creating these files, write a quick test:
- Instantiate OneShotUNet2d, run forward pass with random (2, 3, 256, 256) input
  and noise_step=100. Check output is (2, 1, 256, 256).
- Instantiate OneShotDataset with test split, verify shapes and that t-1/t alignment
  is correct by checking that w_prev at index i corresponds to the time step before
  w_truth at index i.
"""

# ===========================================================================
# PROMPT 2: Training Script
# ===========================================================================

"""
Create src/training/train_oneshot.py and scripts/train_oneshot.py.

For train_oneshot.py — function train_oneshot(cfg, device):
- Builds OneShotDataset for train and val splits
- Standard DataLoader with shuffle=True, batch_size from config (8)
- Wraps OneShotUNet2d in GaussianDiffusion (reuse from src.models.diffusion)

CRITICAL — check how GaussianDiffusion.training_loss works. It currently takes
(w_truth, w_forecast, w_coarse, res_idx). For the one-shot model:
  - w_truth = ground truth 256×256 (same role)
  - The two conditioning fields are w_prev and w_obs_up instead of w_forecast and w_coarse
  - res_idx is not used

Two options:
(a) If GaussianDiffusion.training_loss is flexible enough that w_forecast/w_coarse are
    just conditioning tensors concatenated to the noisy input, pass w_prev as w_forecast
    and w_obs_up as w_coarse, and pass a dummy res_idx=0.
(b) If GaussianDiffusion has hardcoded resolution conditioning that would break, write a
    thin wrapper OneShotDiffusion that handles the training loss and DDIM sampling for
    the one-shot case, reusing the noise schedule from GaussianDiffusion.

Check the existing diffusion.py source code to determine which option works. The key
question is: does the model inside GaussianDiffusion receive res_idx directly, or does
GaussianDiffusion handle it? If GaussianDiffusion passes res_idx to the model's forward
method, then option (a) works IF OneShotUNet2d.forward simply ignores res_idx. If the
model's forward signature requires res_idx, add it as an optional parameter that defaults
to None and is ignored.

PREFER option (a) — minimal new code. If the interface mismatch is too deep, use option (b).

Training loop (same structure as train_diffusion_2d.py):
- DDPM epsilon-prediction loss
- AdamW optimizer, lr=2e-4, weight_decay=1e-5
- Linear warmup (3k steps) + cosine decay
- EMA with decay 0.9999 (reuse EMAModel pattern from train_diffusion_2d.py, copy the class)
- AMP mixed precision on CUDA
- Gradient clipping max_norm=1.0
- Log every 500 steps, validate every 5000, checkpoint every 25000
- Save final EMA checkpoint to checkpoints_oneshot/oneshot_ema.pt
- Total: 200k steps (same as iterative refinement for fair comparison)

For scripts/train_oneshot.py:
- Load config from configs/oneshot_sr.yaml
- Call train_oneshot(cfg, device)
- Print parameter count for comparison with iterative refinement

Run training. Should take ~12-18 hours on 4090 (similar to iterative diffusion since
the U-Net is slightly larger but there's no resolution batching complexity).
"""

# ===========================================================================
# PROMPT 3: Inference Pipeline
# ===========================================================================

"""
Create src/inference/pipeline_oneshot.py and scripts/run_inference_oneshot.py.

For pipeline_oneshot.py — class OneShotPipeline:

Constructor takes: trained diffusion model (OneShotUNet2d wrapped in GaussianDiffusion
with EMA weights loaded), config, device.

Method run(observations_32, ground_truth_256, n_steps):
- observations_32: (T, 32, 32) coarse observations
- ground_truth_256: (T, 256, 256) only needed to initialize t=0 — OR we can initialize
  from spectrally upsampled obs like the iterative method does. Use spectral upsampling
  for consistency: prev_256 = spectral_upsample_2d(obs_32[0], 256, 256)

- For each t >= 1:
  1. Upsample current coarse obs: w_obs_up = spectral_upsample_2d(obs_32_t, 256, 256)
  2. Use previous output as temporal conditioning: w_prev = posterior_{t-1}
  3. Run DDIM: posterior_t = ddim_sample(cond=[w_prev, w_obs_up])
  4. Store posterior_t, it becomes w_prev for next step

- Return dict: "posterior_256" (T, 256, 256), "obs_32" (T, 32, 32)

IMPORTANT: The one-shot model has NO FNO forecaster. The "prior" is purely the previous
posterior + current observation. There is no explicit dynamical forecast step.

Also implement a run_bicubic_baseline(observations_32, n_steps) method that simply
spectrally upsamples each 32×32 observation to 256×256 independently (no temporal info,
no learning). This is Baseline 3 (trivial lower bound).

For scripts/run_inference_oneshot.py:
- Load config from configs/oneshot_sr.yaml
- Load trained one-shot model (EMA weights from checkpoints_oneshot/oneshot_ema.pt)
- Load test data from data_2d/ (same test set)
- Run one-shot pipeline on each test trajectory
- Run bicubic baseline
- Also load iterative refinement results from results_2d/ for comparison
- Save one-shot results to results_oneshot/
- Print comparison RMSE table: Bicubic vs One-Shot SR vs Iterative Refinement at 256×256

Run inference. Should be faster than iterative (no FNO, single-stage DDIM) — ~10-15 min.
"""

# ===========================================================================
# PROMPT 4: Comparison Plots
# ===========================================================================

"""
Create scripts/plot_comparison.py. This generates publication-quality comparison figures
across ALL methods. Save to plots_oneshot/.

Load results from:
- results_2d/ (iterative refinement: posterior_256, forecast_256, fno_only_256)
- results_oneshot/ (one-shot SR: posterior_256, bicubic_256)
- data_2d/ (ground truth for test trajectories)

Use these matplotlib settings:
```python
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
})
```

FIGURE 1 — Vorticity Comparison (plots_oneshot/fig1_method_comparison.png + .pdf):
- 5 columns × 1 row at 256×256: [Ground Truth, Bicubic, One-Shot SR, Iterative Refinement, |Iter Error|]
- Single representative time step (e.g., t=50)
- RdBu_r colormap, same vmin/vmax across first 4 panels
- Error panel uses 'inferno' colormap
- This is THE key visual comparison figure for the paper

FIGURE 2 — RMSE Over Time (plots_oneshot/fig2_rmse_comparison.png + .pdf):
- All methods on one plot at 256×256:
  - Bicubic (gray, dashed)
  - One-Shot SR (green, solid)
  - Iterative Refinement (blue, solid)
  - FNO-only autoregressive (red, dotted) — may go off-scale if it blows up
- x-axis: time step, y-axis: RMSE
- If 2 test trajectories, show mean as thick line, individual as thin
- Legend with method names

FIGURE 3 — Energy Spectrum (plots_oneshot/fig3_spectrum_comparison.png + .pdf):
- Log-log E(k) vs k at 256×256, averaged over time and trajectories:
  - Ground truth (black, solid, lw=2)
  - Bicubic (gray, dashed)
  - One-Shot SR (green)
  - Iterative Refinement (blue)
- k⁻³ reference line
- Forcing wavenumber k_f=4 vertical line
- This figure should reveal whether the one-shot model has the same spectral noise
  problem as iterative refinement, or different spectral characteristics

FIGURE 4 — Temporal Consistency (plots_oneshot/fig4_temporal_consistency.png + .pdf):
- Frame-to-frame L2 displacement ||w_t - w_{t-1}||_2 over time
- Curves for: Ground truth (black), One-Shot SR (green), Iterative Refinement (blue)
- The iterative method should be smoother due to the FNO dynamical prior
- This demonstrates the advantage of having a forecaster in the loop

FIGURE 5 — Summary Bar Chart (plots_oneshot/fig5_summary_bars.png + .pdf):
- Grouped bar chart with 4 groups of metrics: [RMSE, Spectral RMSE, Temporal Consistency, SSIM]
- 3 bars per group: Bicubic, One-Shot SR, Iterative Refinement
- Error bars from test trajectories
- This is the summary figure for a paper table

Import metrics from src.evaluation.metrics_2d (radial_energy_spectrum, rmse_2d,
temporal_consistency_2d, structural_similarity_2d). If any metric is missing, implement
it inline in the plotting script.

Print a formatted comparison table to stdout:
```
Method                  RMSE    Spectral RMSE   Temp. Consist.   SSIM
─────────────────────────────────────────────────────────────────────
Bicubic                 X.XXX   X.XXX           X.XXX            X.XXX
One-Shot SR             X.XXX   X.XXX           X.XXX            X.XXX
Iterative Refinement    X.XXX   X.XXX           X.XXX            X.XXX
```

Save all figures as PNG (300 dpi) and PDF.
"""

# ===========================================================================
# PROMPT 5: Validate and Debug
# ===========================================================================

"""
Run the full one-shot baseline pipeline:
1. python scripts/train_oneshot.py
2. python scripts/run_inference_oneshot.py
3. python scripts/plot_comparison.py

If training fails:
- Check that GaussianDiffusion interface is compatible with OneShotUNet2d
  (the most likely issue is the res_idx handling)
- Check memory — base_channels=64 is larger than iterative's 48, may need
  batch_size=4 if OOM

If inference fails:
- Check that EMA weights are loaded correctly
- Check that DDIM sampling handles the 3-channel concatenation properly

After plotting, examine the results and verify:

1. Fig 1: Does the iterative refinement posterior look sharper/cleaner than
   the one-shot posterior? Are the spectral noise characteristics different?

2. Fig 2: Does the iterative method have lower RMSE than one-shot across time?
   If not, that's actually an important finding too — it would mean the resolution
   hierarchy isn't helping for this problem configuration.

3. Fig 3: Check the spectral behavior of each method:
   - Bicubic: should roll off sharply at k ≈ 16 (Nyquist of 32×32)
   - One-shot: likely has broadband spectral noise (same issue as iterative?)
   - Iterative: compare directly to one-shot at high k

4. Fig 4: The iterative method should show smoother temporal evolution due to
   the FNO dynamical prior. One-shot generates each frame conditioned only on
   the previous output, with no explicit dynamical model.

5. Print the comparison table and check if the relative rankings make sense.

If the one-shot model BEATS iterative refinement, the most likely explanations are:
- The iterative method's cascading error amplification (noise from stage 0 → 1 → 2)
  outweighs the benefits of hierarchical decomposition
- The one-shot model's larger capacity (base 64 vs 48) is compensating
- The noise augmentation fix from earlier wasn't applied to the iterative model

If the one-shot model has similar spectral noise issues, that confirms the problem
is fundamental to diffusion-based SR on turbulence (not specific to the cascade),
and the spectral post-filter or spectral loss fix is needed for both methods.
"""
