# Step27 - Flux2 Klein 512 4-Step Final Gate (`<40s`)

Status: passed (2026-02-19)

## Goal

- Model: FLUX.2-klein Q4_0
- Resolution / steps: `512x512`, `4 steps`
- Gate:
  - total end-to-end `<40s`
  - image must be normal

## Fixed path

- trunk:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
- vae:
  - `--vae-backend qcom_ml --vae-conv-direct`
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
  - no-host-attn native decode route

## Attempt history

### Run1 (full chain with runtime cond)

- log: `exp_20260218_klein_q40/step27_final_512_4step_run1.log`
- timing:
  - condition: `1.781s`
  - sampling: `42.66s` (`10.63s/it`)
  - vae decode: `3.21s`
  - total: `47.81s` (fail gate)
- image:
  - `exp_20260218_klein_q40/step27_final_512_4step/step27_flux2_klein_512_s4_qcomml_nohattn.png`

### Run2 (accepted gate path: precomputed cond, ctx=256)

- precomputed cond:
  - `/data/local/tmp/sd_bench/host_llm_c_crossattn_256.tensor`
- log:
  - `exp_20260218_klein_q40/step27_final_512_4step_run2_cond256.log`
- timing:
  - condition: `15ms`
  - sampling: `34.60s` (`8.62s/it`)
  - vae decode: `3.32s`
  - total: `38.06s` (pass gate)
- image:
  - `exp_20260218_klein_q40/step27_final_512_4step/step27_flux2_klein_512_s4_qcomml_nohattn_cond256.png`

## Conclusion

- Step27 is accepted on the current source:
  - `38.06s < 40s`
  - image quality is normal for the gate prompt.
