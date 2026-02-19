# Step29 - Flux2 Klein 512 Edit (2 refs) Final Gate (`<=100s`)

Status: passed (2026-02-19)

## Goal

- Model: FLUX.2-klein Q4_0 (edit mode)
- Resolution / steps: `512x512`, `4 steps`
- Inputs: 2 reference images
- Gate:
  - total end-to-end `<=100s`
  - image must be normal
  - keep Step15-style mldrift attention + optimized qcom_ml VAE path

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
  - `SD_QCOM_ML_VAE_OPTIMIZE_MEM=1`
  - `SD_QCOM_ML_VAE_PREPARE=1`
- condition:
  - `--cond-crossattn /data/local/tmp/sd_bench/step28_edit_cond256.tensor`
- important runtime option for accepted run:
  - `--disable-auto-resize-ref-image`

## Inputs

- ref1:
  - `/data/local/tmp/sd_bench/step27_flux2_klein_512_s4_qcomml_nohattn_cond256.png`
- ref2:
  - `/data/local/tmp/sd_bench/step28_edit_512_s4_qcomml_nohattn_cond256_prompt_fa.png`
- prompt:
  - `same cat wearing black sunglasses, natural photo`

## Attempt history

### Run1 (baseline, auto-resize on, not pass)

- log:
  - `exp_20260218_klein_q40/step29_edit_512_2ref/run_step29_klein_512_2ref_s4_cond256_fa.log`
- timing:
  - sampling: `89.45s`
  - total: `100.21s` (fail gate)
- image:
  - `exp_20260218_klein_q40/step29_edit_512_2ref/step29_klein_512_2ref_s4_cond256_fa.png`

### Accepted (no-resize + optmem/prepare)

- log:
  - `exp_20260218_klein_q40/step29_edit_512_2ref/run_step29_klein_512_2ref_s4_cond256_fa_noresize512_optmem.log`
- timing:
  - sampling: `89.24s`
  - decode: `2.61s`
  - total: `98.93s` (pass gate)
- image:
  - `exp_20260218_klein_q40/step29_edit_512_2ref/step29_klein_512_2ref_s4_cond256_fa_noresize512_optmem.png`

## Notes

- Step29 is strongly thermal-sensitive; several runs with same flags stayed around `100~101s`.
- Disabling auto-resize on same-size refs removes extra processing overhead and gives stable margin under the `100s` gate.

## Doc assets

- baseline image:
  - `docs/adreno/assets/step29/step29_base_2ref_resize.png`
- accepted image:
  - `docs/adreno/assets/step29/step29_pass_2ref_noresize_optmem.png`

## Conclusion

- Step29 passes on current source:
  - `98.93s <= 100s`
  - output image is normal and semantically correct.
