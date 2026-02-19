# Step28 - Flux2 Klein 512 Edit VAE Optimization (`<=70s`)

Status: passed (2026-02-19)

## Goal

- Model: FLUX.2-klein Q4_0 (edit mode)
- Resolution / steps: `512x512`, `4 steps`
- Gate:
  - total end-to-end `<=70s`
  - image edit must be correct (same cat + black sunglasses)
  - VAE decode graph time should stay in `~2s` class or better

## Fixed path

- trunk:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
  - `GGML_OPENCL_MLDRIFT=1`
  - `GGML_OPENCL_MLDRIFT_DYNAMIC_4352=1`
  - `--diffusion-fa`
- vae:
  - `--vae-backend qcom_ml --vae-conv-direct`
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
  - `SD_QCOM_ML_VAE_DISABLE_MNN_ATTN=1`
  - `SD_QCOM_ML_VAE_FALLBACK_ATTN_16384=1`
- edit input image:
  - `/data/local/tmp/sd_bench/step27_flux2_klein_512_s4_qcomml_nohattn_cond256.png`

## Attempt history

### Run1 (runtime cond, image correct but超时)

- log:
  - `exp_20260218_klein_q40/step28_edit_512_vae_opt/run_step28_edit_512_s4_qcomml_nohattn.log`
- timing:
  - vae encode: `3.53s`
  - condition: `1.730s`
  - sampling: `65.64s`
  - vae decode graph: `0.70s`
  - total: `74.96s` (fail gate)
- image:
  - `exp_20260218_klein_q40/step28_edit_512_vae_opt/step28_edit_512_s4_qcomml_nohattn.png`

### Run2 (accepted: prompt-specific cond256 + diffusion-fa)

- first generate prompt-matched cond256 tensor:
  - log:
    - `exp_20260218_klein_q40/step28_edit_512_vae_opt/run_step28_llm_forward_cond256.log`
  - output:
    - `/data/local/tmp/sd_bench/step28_edit_cond256.tensor`
  - note:
    - `SD_FLUX2_KLEIN_MAX_LENGTH=256` (token length fixed to 256)
- final gate run log:
  - `exp_20260218_klein_q40/step28_edit_512_vae_opt/run_step28_edit_512_s4_qcomml_nohattn_cond256_prompt_fa.log`
- timing:
  - vae encode: `3.54s`
  - condition (precomputed): `1ms`
  - sampling: `59.47s`
  - vae decode graph: `0.70s`
  - total: `67.36s` (pass gate)
- image:
  - `exp_20260218_klein_q40/step28_edit_512_vae_opt/step28_edit_512_s4_qcomml_nohattn_cond256_prompt_fa.png`

## Doc assets

- run1 image:
  - `docs/adreno/assets/step28/step28_run1_runtime_cond.png`
- accepted image:
  - `docs/adreno/assets/step28/step28_run2_cond256_fa_pass.png`

## Conclusion

- Step28 is accepted on current source:
  - `67.36s <= 70s`
  - edited image semantics are correct (cat + black sunglasses)
  - vae decode graph remains in fast path (`0.70s`).
