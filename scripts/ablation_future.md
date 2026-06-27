# Future Ablation Studies

Groups A–D and F require retraining. Prioritised by scientific importance to reviewers.

---

## Group A — Cascade Structure (core contribution)

### A1. Cascade depth sweep
- **Variants**: 1-stage (one-shot, already built), 2-stage (skip 64→128, go 64→256→512), 3-stage (current)
- **Metric**: RMSE and spectral RMSE at each resolution
- **Hypothesis**: each additional stage reduces error at high k; one-shot misses fine-scale structure
- **Implementation**: new pipeline config skipping the 128-resolution stage; reuse the trained diffusion model unchanged — only modify `IterativeRefinementPipeline._STAGES` in `src/inference/pipeline.py`
- **Cost**: no retraining needed (inference-only change) — highest ROI ablation after E1/E2

### A2. Propagation signal into each stage
- **Variants**: (i) current — use diffusion posterior of stage r as coarse input to stage r+1; (ii) use FNO forecast instead; (iii) use raw spectral downsample of original 64-pt observation
- **Metric**: RMSE and spectral RMSE at 256 and 512
- **Implementation**: one-line change in `src/inference/pipeline.py` lines 164–172 (swap `stage_posts[coarse_res]`); no retraining
- **Cost**: inference-only

---

## Group B — Resolution Conditioning (resolution-agnostic claim)

### B1. No resolution embedding
- **Variants**: (i) current — sinusoidal res_idx summed with timestep in FiLM; (ii) zero out resolution embedding (`self.res_emb` output zeroed before summation); (iii) three separate U-Nets, one per stage
- **Metric**: RMSE at each stage (128, 256, 512)
- **Hypothesis**: removing embedding hurts disproportionately at higher resolutions; 3 separate models match current quality at 3× parameter cost
- **Implementation**: B1-ii — set `cond = self.time_emb(noise_step)` only in `src/models/unet.py:297`; retrain
- **Key file**: `src/models/unet.py` line 297

### B2. Conditioning fusion: sum vs. concatenation
- **Variants**: (i) current — `cond = time_emb + res_emb` (128-dim); (ii) `cond = cat([time_emb, res_emb])` (256-dim, adjust FiLMConditioner input)
- **Hypothesis**: negligible quality difference; validates lean summed design
- **Implementation**: `src/models/unet.py:297` + `cond_embed_dim` config; retrain

---

## Group C — Input Channel Design

### C1. Innovation channel
- **Variants**: (i) current — 3 channels [x_noisy, u_forecast, u_coarse_up]; (ii) 4 channels (add `u_forecast − u_coarse_up` as channel 3); (iii) 2 channels (drop u_forecast, keep only noisy + coarse_up)
- **Metric**: training convergence curve + final RMSE
- **Hypothesis**: 4-channel adds no benefit (model already learns the difference); 2-channel drops temporal information and significantly hurts quality
- **Implementation**: change `in_channels` in `configs/default.yaml` and `forward()` in `src/models/unet.py:300`; retrain
- **Note**: directly validates the CLAUDE.md design decision "NO separate innovation channel"

---

## Group D — Training Strategy (highest scientific impact)

### D1. Strategy A (ground-truth forecast) vs. Strategy B (FNO-generated, current)
- **Variants**: (i) Strategy B — diffusion trained on FNO outputs (current); (ii) Strategy A — diffusion trained with ground-truth u_t as the "forecast" slot
- **Metric**: inference RMSE as a function of time step (Strategy A diverges as FNO errors compound)
- **Hypothesis**: Strategy A fails at inference because it never sees FNO distribution-shift errors; Strategy B is consistent
- **Implementation**: modify `src/training/train_diffusion.py:_build_diffusion_datasets()` to substitute `truth_data[coarse_res]` for FNO forecast files; retrain diffusion model only (FNOs unchanged)
- **Cost**: one diffusion training run (~150k steps, ~3h on 4090)
- **Note**: most important ablation for reviewers — directly defends the training pipeline design

---

## Group F — Architecture Capacity

### F1. Residual blocks per level
- **Variants**: 1 (current), 2
- **Metric**: RMSE at 512, model parameter count, training time
- **Hypothesis**: 2 blocks increases parameters by ~70% for modest quality gain; validates lean design
- **Implementation**: `n_res_blocks: 2` in `configs/default.yaml`; retrain

---

## Recommended Execution Order

| Priority | Ablation | Retraining? | Estimated cost |
|---|---|---|---|
| 1 | A1 — cascade depth (2-stage) | No | ~1h inference |
| 2 | A2 — propagation signal | No | ~1h inference |
| 3 | D1 — Strategy A vs B | Yes (diffusion only) | ~3h on 4090 |
| 4 | B1 — no resolution embedding | Yes (diffusion) | ~3h |
| 5 | C1 — innovation channel | Yes (diffusion) | ~3h |
| 6 | B2 — sum vs concat | Yes (diffusion) | ~3h |
| 7 | F1 — 2 res blocks | Yes (diffusion) | ~4h |

A1 and A2 are free (inference-only modifications to pipeline.py) and should be run
immediately alongside the E1/E2 sweeps.
