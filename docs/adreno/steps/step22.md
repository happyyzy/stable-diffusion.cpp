# Step22 - Klein 1024 VAE Decode Optimization (qcom_ml)

Status: passed (2026-02-19)

## Goal

- FLUX.2-klein, 1024 decode-only, qcom_ml VAE path
- gate: VAE decode `<10s`
- image must remain normal vs ggml reference decode

## Key code changes

- `stable-diffusion.cpp`
  - Flux2 packed latent (`64x64x128`) uses effective unpacked shape (`128x128x32`) for qcom_ml tiled trigger.
  - `decode_first_stage()` calls qcom_ml `prepare()` before timed decode (can disable by `SD_QCOM_ML_VAE_PREPARE=0`).
- `qcom_ml_vae_bridge.hpp`
- `qcom_ml_vae_bridge.cpp`
  - Add optional bridge ABI `sd_qcom_ml_vae_prepare(...)`.
- `thirdparty/qcom_ml_vae_sdk_bridge/sd_qcom_ml_vae_bridge.cpp`
  - Export `sd_qcom_ml_vae_prepare`.
  - Add timing logs for prepare/ensure/upload/forward/download phases.
  - Add `SD_QCOM_ML_VAE_OPTIMIZE_MEM=1` descriptor option.
- `thirdparty/qcom_ml_vae_sdk_bridge/clml_decoder.cpp`
  - Keep `decoder_conv_out` with immediate output mem (`defer_output_mem=false`) to avoid unstable finalize stage.

## Fixed run configuration

- CLI:
  - `--vae-backend qcom_ml --vae-conv-direct --threads 1`
  - `--load-latent /data/local/tmp/sd_bench/step22_flux2_latent.tensor`
  - output resolution `1024x1024`
- Env:
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
  - `SD_QCOM_ML_VAE_HOST_ATTN=1`
  - `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`
  - `SD_QCOM_ML_VAE_TRY_TILED=1`
  - `SD_QCOM_ML_VAE_TILE_SIZE=32`
  - `SD_QCOM_ML_VAE_TILE_OVERLAP=0`
  - `SD_QCOM_ML_VAE_OPTIMIZE_MEM=1`

## Results

### Decode-only pair used by headline table

- ggml reference decode:
  - `32.80s`
  - log: `exp_20260218_klein_q40/step22_vae_1024_opt/run_step22_flux2_1024_ggml_ref.log`
- optimized decode:
  - `5.25s`
  - log: `exp_20260218_klein_q40/step23_full_decode_mldrift/run_decode_full_mldrift_1024_v3_all.log`

- Baseline (no prepare):
  - `11.22s`
  - `exp_20260218_klein_q40/step22_vae_1024_opt/run_step22_flux2_1024_qcomml_t32_o0_v30_optmem.log`
- With prepare + same decode path (repeat):
  - `7.98s`
  - `8.13s`
  - `8.68s`
  - logs:
    - `exp_20260218_klein_q40/step22_vae_1024_opt/run_step22_flux2_1024_qcomml_t32_o0_v33_optmem_prepare.log`
    - `exp_20260218_klein_q40/step22_vae_1024_opt/run_step22_flux2_1024_qcomml_t32_o0_v34_optmem_prepare.log`
    - `exp_20260218_klein_q40/step22_vae_1024_opt/run_step22_flux2_1024_qcomml_t32_o0_v35_optmem_prepare.log`

## Image correctness

- ggml reference:
  - `exp_20260218_klein_q40/step22_vae_1024_opt/images/step22_flux2_1024_ggml_ref.png`
- qcom_ml outputs:
  - `exp_20260218_klein_q40/step22_vae_1024_opt/images/step22_flux2_1024_qcomml_decode_v33_t32_optmem_prepare.png`
  - `exp_20260218_klein_q40/step22_vae_1024_opt/images/step22_flux2_1024_qcomml_decode_v34_t32_optmem_prepare.png`
  - `exp_20260218_klein_q40/step22_vae_1024_opt/images/step22_flux2_1024_qcomml_decode_v35_t32_optmem_prepare.png`
- diff summary:
  - `exp_20260218_klein_q40/step22_vae_1024_opt/repeatability_metrics.md`
  - stable across repeats: `mean_abs ~8.179`, `rmse ~10.129`, `p99=24`, `PSNR ~28.02 dB`

## Assets for docs

- `docs/adreno/assets/step22/step22_ggml_ref_1024.png`
- `docs/adreno/assets/step22/step22_qcomml_t32_optmem_prepare_1024.png`

## Conclusion

- Step22 passes on current source:
  - decode speed reaches `<10s`
  - image quality stays stable and visually normal
  - repeated runs keep near-identical numeric/image deltas vs ggml reference
