# Step25 - Z-Image 512 8-Step Final Gate (`<100s`)

Status: passed (2026-02-19)

## Goal

- Model: z-image turbo Q4_0
- Resolution / steps: `512x512`, `8 steps`
- Gate:
  - total end-to-end `<100s`
  - image must be normal (no black/noise/corruption)

## Key code state

- `stable-diffusion.cpp`
  - keep qcom_ml VAE single-shot decode bridge path (vector I/O fast path + non-finite guard) from Step24.
  - keep host-attn fallback and tiled/non-tiled dispatch logic stable.
- Runtime tuning (no new kernel migration in this step):
  - force selective Q4 GEMM F32 activation-read only for attention out-proj:
    - `SD_OCL_Q4_GEMM_F32_ACT_NO_AUTO=1`
    - `SD_OCL_Q4_GEMM_F32_ACT_SUBSTR=attention.out.weight`

## Fixed run configuration (accepted)

- Binary: `/data/local/tmp/sd_bench/sd-cli-step25opt-new`
- CLI:
  - `--diffusion-model /data/local/tmp/sd_bench/z_image_turbo-Q4_0-nobf16.gguf`
  - `--llm /data/local/tmp/sd_bench/qwen_3_4b-Q4_0.gguf`
  - `--vae /data/local/tmp/sd_bench/ae.safetensors`
  - `--vae-backend qcom_ml`
  - `--steps 8 -W 512 -H 512 --threads 1`
  - `--cfg-scale 1.0 --guidance 3.5 --seed 42`
  - `-p "a lovely cat"`
- Env:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_zimage_ae`
  - `SD_QCOM_ML_VAE_HOST_ATTN=1`
  - `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`
  - `SD_OCL_Q4_GEMM_F32_ACT_NO_AUTO=1`
  - `SD_OCL_Q4_GEMM_F32_ACT_SUBSTR=attention.out.weight`

## Before/After

### Baseline (same host-attn route, no out-proj-only F32_ACT)

- Log: `exp_20260216_zimage_q40/step25_final_512_8step/run_step25_zimg_512_s8_hostattn_ggml_short.log`
- Timing:
  - condition: `218 ms`
  - sampling: `100.15s`
  - vae decode: `4.49s`
  - total: `104.90s` (fail gate)
- Image:
  - `exp_20260216_zimage_q40/step25_final_512_8step/step25_zimg_512_s8_hostattn_ggml_short.png`

### Slow-path reference (used by headline delta table)

- Log: `exp_20260216_zimage_q40/step25_final_512_8step/run_step25_zimg_512_s8_qcomml_t1_repro2.log`
- Timing:
  - sampling: `204.68s`
  - total: `209.69s`
- Note:
  - this run hit qcom_ml non-finite fallback and is kept only as a worst-path reference point.

### Accepted (out-proj-only F32_ACT)

- Log: `exp_20260216_zimage_q40/step25_final_512_8step/run_step25_zimg_512_s8_outonly_hostattn_short_step25opt_new.log`
- Timing:
  - condition: `223 ms`
  - sampling: `91.22s`
  - vae decode: `4.51s`
  - total: `95.99s` (pass gate)
- Image:
  - `exp_20260216_zimage_q40/step25_final_512_8step/step25_zimg_512_s8_outonly_hostattn_short_step25opt_new.png`

## Doc assets

- baseline image:
  - `docs/adreno/assets/step25/step25_base_qcomml_t1.png`
- accepted image:
  - `docs/adreno/assets/step25/step25_outonly_hostattn_step25opt_new.png`

## Conclusion

- Step25 gate is accepted on current source:
  - `95.99s < 100s`
  - image quality remains normal and consistent with z-image 512 target behavior.
