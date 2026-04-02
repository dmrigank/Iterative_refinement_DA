# Claude Code Prompt Sequence
# Copy-paste these prompts in order into Claude Code.
# Each prompt builds on the previous one. Wait for completion before moving to the next.

# ===========================================================================
# PROMPT 1: Project Scaffolding
# ===========================================================================

"""
Read CLAUDE.md thoroughly. Then create the full directory structure and all __init__.py files as specified. Also create the configs/default.yaml if it doesn't exist (check first). Create empty placeholder files for every module listed in the architecture section of CLAUDE.md. Make sure every Python package has an __init__.py. Don't implement any logic yet — just the skeleton.
"""

# ===========================================================================
# PROMPT 2: Pseudospectral Burgers Solver + Data Generation
# ===========================================================================

"""
Implement src/data/solver.py and src/data/generate_dataset.py and scripts/generate_data.py. Read CLAUDE.md for all specs.

For solver.py, implement class StochasticBurgersSolver:
- Pseudospectral method on [0, 2π] periodic domain at N=2048 grid points
- IMEX time stepping: Crank-Nicolson for diffusion (implicit in Fourier space), RK4 for the nonlinear term + forcing (explicit)
- Nonlinear term computed as 0.5 * d(u^2)/dx in Fourier space with 2/3 dealiasing rule
- OU forcing: maintain K_f=16 independent OU processes. Update rule per solver step: η_k += (-η_k/τ) * dt + σ_k * sqrt(2*dt/τ) * randn. The forcing in physical space is f(x,t) = Σ σ_k * η_k(t) * sin(kx), with σ_k = ou_sigma_base * k^{-1}
- Method `solve(n_steps, save_every)` that returns snapshots at the specified cadence
- Method `spectral_downsample(u, target_n)` and `spectral_upsample(u, target_n)` as standalone functions with the CORRECT amplitude scaling factor (target_n / source_n). This is critical — see CLAUDE.md.

For generate_dataset.py, implement:
- Generate n_trajectories trajectories, each with spinup_snapshots + snapshots_per_trajectory snapshots
- Discard spinup, keep the rest
- For each snapshot, create the full resolution pyramid [64, 128, 256, 512] via spectral downsampling from the 2048-point truth
- Save as a dict: {"u_64": tensor, "u_128": tensor, "u_256": tensor, "u_512": tensor} for each trajectory
- Split into train/val/test and save separately as .pt files
- Use config from configs/default.yaml via OmegaConf

For scripts/generate_data.py: simple entry point that loads config and calls generate_dataset.

After implementing, run `python scripts/generate_data.py` to verify it works. Check that the data files are created and the shapes look right. Quick sanity check: plot one snapshot at all 4 resolutions to verify the spectral downsampling preserves the large-scale structure while removing small scales. Save this sanity check plot to plots/data_sanity_check.png.
"""

# ===========================================================================
# PROMPT 3: FNO Forecaster
# ===========================================================================

"""
Implement src/models/fno.py and src/models/embeddings.py.

For fno.py, implement class FNO1d:
- Architecture: Lifting layer (1 -> width) via pointwise Conv1d, then n_fourier_layers SpectralConv1d layers each followed by a bypass Conv1d (standard FNO architecture), then projection (width -> 1) via two pointwise Conv1d layers with GELU between
- SpectralConv1d: multiply in Fourier space up to k_max modes. Use complex-valued weight matrix of shape (width, width, k_max). Forward: rfft -> multiply by weights for first k_max modes -> irfft, then add bypass
- k_max is passed as a constructor argument (will be N_r/4 for each resolution)
- GELU activation between all layers
- Input shape: (batch, 1, N_r), Output shape: (batch, 1, N_r)

For embeddings.py, implement:
- SinusoidalEmbedding(dim): standard sinusoidal positional embedding that takes a scalar (int or float) and returns a vector of size dim. Use log-spaced frequencies.
- FiLMConditioner: takes an embedding vector, passes through MLP (embed_dim -> hidden -> 2*channels), splits into scale and shift, applies as: out = scale * x + shift. The MLP should be: Linear -> SiLU -> Linear.
- ResolutionEmbedding: wraps SinusoidalEmbedding — takes resolution index r (0, 1, or 2), embeds it via sinusoidal embedding, then MLP to cond_embed_dim.

All dimensions should be read from config. Write clean, well-typed code.
"""

# ===========================================================================
# PROMPT 4: Train FNO Forecasters
# ===========================================================================

"""
Implement src/data/dataset.py, src/training/train_fno.py, and scripts/train.py (FNO part only).

For dataset.py:
- Class FNODataset(Dataset): loads a .pt trajectory file, creates single-step pairs (u^r_{t}, u^r_{t+1}) at a specified resolution. __getitem__ returns (input, target) both as (1, N_r) tensors.
- Class DiffusionDataset(Dataset): loads trajectory data + FNO forecast data. __getitem__ returns a dict with keys: "u_truth" (ground truth at resolution r+1), "u_forecast" (FNO forecast at resolution r+1), "u_coarse" (observation at resolution r, spectrally upsampled to resolution r+1), "res_idx" (int, 0/1/2). Samples uniformly across resolution pairs.

For train_fno.py:
- Function train_fno(resolution, config) that trains one FNO for a given resolution level
- Standard PyTorch training loop: MSE loss, AdamW optimizer, CosineAnnealingLR scheduler
- Validation every epoch, save best model by val loss
- tqdm progress bar with loss display
- Save checkpoint to checkpoints/fno_{resolution}.pt

For scripts/train.py (just the FNO section for now):
- Load config, train FNO for resolutions [128, 256, 512] sequentially
- After training all FNOs, run each FNO autoregressively on ALL training trajectories and save the forecast outputs. This is critical for Strategy B — the diffusion model will train on these outputs, not ground truth.
- Save forecast data as .pt files in data/fno_forecasts/

After implementing, run the FNO training. It should complete in ~1 hour total for all three resolutions. Check training curves make sense — loss should decrease smoothly.
"""

# ===========================================================================
# PROMPT 5: Diffusion Model (U-Net + Noise Schedule)
# ===========================================================================

"""
Implement src/models/unet.py and src/models/diffusion.py.

For unet.py, implement class ConditionalUNet1d:
- Input: (batch, in_channels=3, length) where channels are [x_noisy, u_forecast, u_coarse_up]
- Encoder: sequence of DownBlock modules. Each DownBlock has: n_res_blocks ResidualBlock(s), then a downsample (Conv1d stride 2)
- Bottleneck: n_res_blocks ResidualBlock(s) at the coarsest level
- Decoder: sequence of UpBlock modules. Each UpBlock has: upsample (nearest + Conv1d), concatenate skip connection, then n_res_blocks ResidualBlock(s)
- Output: Conv1d(base_channels, 1, 1) to predict noise

ResidualBlock:
- GroupNorm -> SiLU -> Conv1d(in_ch, out_ch, 3, padding=1) -> GroupNorm -> SiLU -> Conv1d(out_ch, out_ch, 3, padding=1)
- FiLM conditioning applied BETWEEN the two conv layers: after first GroupNorm+SiLU+Conv, apply FiLM (scale * h + shift) from the conditioning embedding, then second GroupNorm+SiLU+Conv
- Skip connection with 1x1 conv if in_ch != out_ch

The conditioning embedding is: sinusoidal_embed(noise_step) + sinusoidal_embed(res_idx), both mapped through separate MLPs to cond_embed_dim, then summed. This summed vector is projected per-block to (scale, shift) via the FiLMConditioner.

For diffusion.py, implement class GaussianDiffusion:
- Cosine noise schedule (Nichol & Dhariwal): α_bar(t) = f(t)/f(0) where f(t) = cos((t/T + s)/(1+s) * π/2)^2, s=0.008
- Pre-compute and store: betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars
- Method q_sample(x_0, t, noise=None): forward diffusion, returns noisy x_t
- Method p_losses(model, x_0, t, cond): compute training loss (MSE between predicted and true noise)
- Method ddim_sample(model, shape, cond, n_steps=25, eta=0.0): DDIM sampling loop. Takes the conditioning dict, generates evenly-spaced subsequence of noise steps, iteratively denoises. Returns the final sample AND optionally the full trajectory of intermediate states (for visualization).

Be very careful with tensor shapes. The model predicts noise of shape (batch, 1, length). All conditioning tensors should be (batch, 1, length) before concatenation along dim=1 to form the (batch, 3, length) input.

Implement thoroughly and add shape assertions at key points to catch bugs early.
"""

# ===========================================================================
# PROMPT 6: Train Diffusion Corrector
# ===========================================================================

"""
Implement src/training/train_diffusion.py and update scripts/train.py to include diffusion training after FNO training.

For train_diffusion.py:
- Function train_diffusion(config) that trains the shared diffusion corrector G
- Uses DiffusionDataset which loads FNO forecast data (Strategy B)
- Training loop:
  1. Sample a batch from DiffusionDataset (uniformly across resolution pairs)
  2. Each sample provides: u_truth (target), u_forecast (FNO output), u_coarse (upsampled coarse obs), res_idx
  3. Sample random noise timestep k ~ Uniform(0, T-1)
  4. Add noise to u_truth: x_noisy = sqrt(ᾱ_k) * u_truth + sqrt(1-ᾱ_k) * ε
  5. Concatenate [x_noisy, u_forecast, u_coarse] along channel dim -> (batch, 3, length)
  6. Predict noise: ε_pred = model(x_concat, k, res_idx)
  7. Loss = MSE(ε_pred, ε)
- EMA: maintain an exponential moving average of model weights, updated every step
- Logging: print loss every log_interval steps
- Validation: every val_interval steps, compute val loss on held-out data
- Save checkpoint every save_interval steps, always save the EMA weights
- Save final checkpoint as checkpoints/diffusion_ema.pt

For scripts/train.py update:
- After FNO training and forecast generation, call train_diffusion(config)
- The full script flow: train FNOs -> generate forecasts -> train diffusion

After implementing, start the training. The diffusion training should take ~6-8 hours. Monitor the loss — it should decrease from ~1.0 to ~0.02-0.05 range.
"""

# ===========================================================================
# PROMPT 7: Inference Pipeline
# ===========================================================================

"""
Implement src/inference/pipeline.py and scripts/run_inference.py.

For pipeline.py, implement class IterativeRefinementPipeline:
- Constructor takes: trained FNO models (dict: resolution -> model), trained diffusion model (EMA weights), diffusion helper, config
- Method `run(observations_64, n_steps)`:
  - observations_64: tensor of shape (n_steps, 64) — the coarse observations over time
  - Initialize: at t=0, spectrally upsample u_64_0 to all resolutions [128, 256, 512]
  - For each t >= 1:
    - Stage 1 (64 -> 128):
      - forecast_128 = fno_128(prev_posterior_128)
      - coarse_up = spectral_upsample(obs_64_t, 128)
      - posterior_128 = ddim_sample(cond={forecast_128, coarse_up, res_idx=0})
    - Stage 2 (128 -> 256):
      - forecast_256 = fno_256(prev_posterior_256)
      - coarse_up = spectral_upsample(posterior_128_t, 256)  # use stage 1 output!
      - posterior_256 = ddim_sample(cond={forecast_256, coarse_up, res_idx=1})
    - Stage 3 (256 -> 512):
      - forecast_512 = fno_512(prev_posterior_512)
      - coarse_up = spectral_upsample(posterior_256_t, 512)  # use stage 2 output!
      - posterior_512 = ddim_sample(cond={forecast_512, coarse_up, res_idx=2})
    - Store all posteriors and forecasts for evaluation
  - Return dict with all results at all resolution levels

- Method `run_fno_only(observations_64, n_steps)`: same but skip diffusion correction, just run FNO forecasts autoregressively at each resolution. This is baseline 4 (forecast only).

For scripts/run_inference.py:
- Load config, load all trained models (FNOs + diffusion EMA)
- Load test trajectories
- Run full iterative refinement pipeline on each test trajectory
- Run FNO-only baseline on each test trajectory
- Save all results (posteriors, forecasts, ground truth) as .pt files in results/
- Print summary RMSE at each resolution level

After implementing, run inference on the test set. This should take ~10-20 minutes for 5 trajectories × 200 steps.
"""

# ===========================================================================
# PROMPT 8: Metrics and Evaluation
# ===========================================================================

"""
Implement src/evaluation/metrics.py.

Implement the following functions, all operating on PyTorch tensors:

1. rmse(pred, truth): Root mean squared error, averaged over spatial dimension
   - Input: (n_time, n_grid), Output: (n_time,) or scalar if averaged

2. rmse_over_time(pred, truth): RMSE at each time step
   - Input: (n_time, n_grid), Output: (n_time,)

3. energy_spectrum(u):
   - Compute E(k) = |û_k|^2 for a single field u of shape (n_grid,)
   - Or average over time: input (n_time, n_grid), output (n_modes,)

4. shock_positions(u, threshold_quantile=0.95):
   - Detect shocks as locations where |du/dx| exceeds the threshold_quantile of |du/dx|
   - du/dx computed spectrally
   - Returns list of shock positions (as grid indices)

5. shock_position_error(pred, truth):
   - Find shocks in both, compute minimum-distance matching error
   - Average over matched pairs

6. temporal_consistency(u_sequence):
   - Compute ||u_t - u_{t-1}||_2 at each time step
   - Input: (n_time, n_grid), Output: (n_time-1,)

Keep these clean and efficient. All should work on GPU tensors.
"""

# ===========================================================================
# PROMPT 9: Publication-Quality Plots
# ===========================================================================

"""
Implement scripts/plot_results.py. This is the most important visual output. Read the plotting requirements in CLAUDE.md carefully.

Load all results from results/ directory. Generate the following 6 figures:

FIGURE 1 — Hovmöller Diagram (plots/fig1_hovmoller.png + .pdf):
- 4 panels side by side: (a) Ground truth at 512, (b) FNO forecast at 512, (c) Diffusion posterior at 512, (d) Absolute error |posterior - truth|
- x-axis: spatial coordinate x ∈ [0, 2π], y-axis: time step (show 100 steps)
- Use diverging colormap (RdBu_r) for the field, sequential (viridis) for the error
- Shared colorbar per row. Title each panel.
- Pick the test trajectory with median RMSE for display.

FIGURE 2 — Snapshot Comparison (plots/fig2_snapshot.png + .pdf):
- Pick a single representative time step (e.g., t=50) from the median-RMSE trajectory
- 4 vertically stacked subplots, one per resolution: 64, 128, 256, 512
- Each subplot overlays: ground truth (solid black, linewidth 1.5), FNO forecast (dashed red), diffusion posterior (solid blue, linewidth 1.2), and at the coarsest applicable level show the observation as gray scatter dots
- Legend in the top subplot only
- Shared x-axis [0, 2π], y-label for each subplot showing resolution

FIGURE 3 — RMSE Over Time (plots/fig3_rmse_time.png + .pdf):
- At 512 resolution: RMSE vs time step for FNO-only (red) and iterative refinement (blue)
- Compute mean ± 1 std across test trajectories
- Shaded envelope for the std
- Log-scale y-axis if the values span a large range
- x-axis: time step, y-axis: RMSE

FIGURE 4 — Energy Spectrum (plots/fig4_spectrum.png + .pdf):
- Log-log plot of E(k) vs wavenumber k at 512 resolution
- Three curves: ground truth (black), FNO forecast (red dashed), diffusion posterior (blue)
- Average over all time steps and test trajectories
- Add a reference line showing k^{-2} scaling (gray dashed, labeled)
- x-axis: wavenumber k, y-axis: E(k)

FIGURE 5 — Per-Stage RMSE (plots/fig5_per_stage.png + .pdf):
- Bar chart or grouped bars showing RMSE at each resolution level (128, 256, 512)
- Two groups: FNO-only vs iterative refinement
- Error bars showing ± 1 std across test trajectories
- This demonstrates that each stage of refinement reduces error

FIGURE 6 — Diffusion Denoising Trajectory (plots/fig6_denoising.png + .pdf):
- Pick one time step, one resolution (256 -> 512 transition)
- Show the state at DDIM steps [0, 5, 10, 15, 20, 25] (or whatever the actual step indices are)
- 6 subplots arranged in 2 rows × 3 columns
- Each subplot shows: the current denoised estimate (blue), ground truth (black dashed), FNO forecast prior (red dashed, light)
- Title each subplot with the DDIM step number
- This requires modifying the DDIM sampler to return intermediate states (should already be implemented from prompt 5)

Use these matplotlib settings at the top:
```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})
```

Save every figure as both PNG (300 dpi) and PDF. Print a summary message listing all generated figures.
"""

# ===========================================================================
# PROMPT 10: Debug and Validate End-to-End
# ===========================================================================

"""
Run the full pipeline end to end:
1. python scripts/generate_data.py
2. python scripts/train.py
3. python scripts/run_inference.py
4. python scripts/plot_results.py

If any step fails, debug and fix it. Common issues to watch for:
- Shape mismatches in the U-Net (especially after skip connections)
- NaN losses in diffusion training (check noise schedule, learning rate)
- Spectral scaling errors (fields at different resolutions should have similar magnitudes)
- DDIM sampling producing garbage (check that EMA weights are loaded, not training weights)
- FNO forecast generating NaN after many autoregressive steps (may need gradient clipping during training)

After everything runs, check the plots. The key things to verify:
- Fig 1: The diffusion posterior should look sharper than the FNO forecast, especially near shocks
- Fig 2: The posterior (blue) should track the truth (black) better than the forecast (red) at all resolutions
- Fig 3: Iterative refinement RMSE should be lower than FNO-only, though the gap may grow over time
- Fig 4: The posterior's spectrum should match truth more closely than the FNO forecast (which will roll off too early)
- Fig 5: RMSE should decrease from 128 to 256 to 512 for the iterative method
- Fig 6: The denoising trajectory should show progressive refinement from noise to a clean field

If the results look wrong (e.g., diffusion posterior is WORSE than FNO forecast), the most likely culprit is the training data pipeline for Strategy B — make sure the diffusion model is actually training on FNO forecast outputs and not ground truth.
"""
