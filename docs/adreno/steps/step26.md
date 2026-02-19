# Step26 - Flux2 Klein 512 VAE Decode Optimization (qcom_ml, `<=2s`)

Status: passed (2026-02-19)

## Goal

- Model: FLUX.2-klein Q4_0
- Decode mode: phone-side VAE decode with `qcom_ml`
- Gate:
  - decode-only runtime `<=2s`
  - image output must be normal

## Fixed input

- latent:
  - `/data/local/tmp/sd_bench/t1_phonecond_latent_x0.tensor`
- models:
  - `/data/local/tmp/sd_bench/flux-2-klein-4b-Q4_0.gguf`
  - `/data/local/tmp/sd_bench/qwen_3_4b-Q4_0.gguf`
  - `/data/local/tmp/sd_bench/flux2-vae.safetensors`

## Run configuration

- CLI:
  - `--load-latent /data/local/tmp/sd_bench/t1_phonecond_latent_x0.tensor`
  - `--vae-backend qcom_ml --vae-conv-direct --threads 1`
- Env:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`

## Results

### Accepted path (native qcom_ml, no host-attn)

- log: `exp_20260218_klein_q40/step26_vae_512_opt/run_step26_flux2_512_decode_qcomml_nohattn.log`
- timing:
  - `computing vae decode graph completed, taking 0.74s`
- output:
  - `exp_20260218_klein_q40/step26_vae_512_opt/step26_flux2_512_decode_qcomml_nohattn.png`
- gate:
  - pass (`0.74s <= 2s`)

### Baseline control (qcom_ml + host-attn backend=ggml)

- log: `exp_20260218_klein_q40/step26_vae_512_opt/run_step26_flux2_512_decode_qcomml_hattn.log`
- timing:
  - `computing vae decode graph completed, taking 40.24s`
- output:
  - `exp_20260218_klein_q40/step26_vae_512_opt/step26_flux2_512_decode_qcomml_hattn.png`

### Host CPU reference

- log: `exp_20260218_klein_q40/step26_vae_512_opt/run_step26_flux2_512_decode_host_cpu_ref.log`
- timing:
  - `computing vae decode graph completed, taking 17.87s`
- output:
  - `exp_20260218_klein_q40/step26_vae_512_opt/step26_flux2_512_decode_host_cpu_ref.png`

## Numeric / image check

- metrics file:
  - `exp_20260218_klein_q40/step26_vae_512_opt/metrics.md`
- key values:
  - nohost vs host-attn: `mae=0.219920, p99=1, max=4`
  - nohost vs host-cpu: `mae=3.761077, p99=10, max=34`

## Doc assets

- baseline:
  - `docs/adreno/assets/step26/step26_baseline_qcomml_hattn.png`
- accepted:
  - `docs/adreno/assets/step26/step26_pass_qcomml_nohattn.png`

## Conclusion

- Step26 passes on current source:
  - native qcom_ml decode hits `0.74s`
  - image is normal and consistent with host reference quality.
