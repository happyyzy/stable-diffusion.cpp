# Step24 - Z-Image 512 VAE Decode (`qcom_ml`) Optimization

## Goal

- GOAL.md Step24:
  - `qcom_ml` route for z-image 512 VAE decode
  - target decode time `<= 2s`
  - image must stay correct
  - and verify whether native `qcom_ml` attention can run without fallback

## Code Changes (this round)

- `stable-diffusion.cpp/stable-diffusion.cpp`
  - Optimize tensor copy/export/import in `sd_tensor_to_f32_vector()`, `sd_f32_vector_to_tensor()`, `sd_copy_tensor_values()`
  - Add contiguous fast-paths using bulk transfer (`ggml_backend_tensor_get/set/copy` or `memcpy`) to reduce per-element host overhead in `qcom_ml` VAE bridge path
  - Add no-host diagnostics for qcom_ml decode non-finite output:
    - report `nonfinite_count/total`, first bad index/value in error message
    - optional debug-only env `SD_QCOM_ML_VAE_SANITIZE_NONFINITE=1` to clamp bad values to zero for visualization

## Result Summary

### 1) Decode-only (`qcom_ml` + host-attn backend=ggml)

- Command path:
  - binary: `/data/local/tmp/sd_bench/sd-cli-step24opt`
  - latent: `/data/local/tmp/sd_bench/zimg_lat_step17auto_s4.tensor`
- Log:
  - `exp_20260216_zimage_q40/step24_vae_512_opt/run_step24opt_zimg_decodeonly_qcomml_hattn_final.log`
- Key metric:
  - `computing vae decode graph completed, taking 1.79s`
- Output:
  - `exp_20260216_zimage_q40/step24_vae_512_opt/step24opt_zimg_decodeonly_qcomml_hattn_final.png`
- lightweight doc assets:
  - `docs/adreno/assets/step24/step24_ref_hostdecode.png`
  - `docs/adreno/assets/step24/step24_qcomml_hattn_decodeonly.png`
- Numeric/image check vs host decode reference:
  - ref: `exp_20260216_zimage_q40/zimage_step17auto_s4_hostdecode.png`
  - MAE `0.8134`, RMSE `1.6086`, p99 `6`, max `72`
  - metrics file: `exp_20260216_zimage_q40/step24_vae_512_opt/metrics_vs_hostdecode.md`

### 2) Native `qcom_ml` attention (host-attn disabled) validation

- Base no-host run:
  - `exp_20260216_zimage_q40/step24_vae_512_opt/run_step24opt_zimg_decodeonly_qcomml_nohattn.log`
  - still hits `non-finite output from qcom_ml decode` and falls back
- Recheck with count-enabled binary:
  - `exp_20260216_zimage_q40/step24_vae_512_opt/run_step24opt_zimg_decodeonly_qcomml_nohattn_count.log`
  - explicit failure: `non-finite output from qcom_ml decode (786432/786432, first_idx=0, first_value=nan)`
- Diagnostic sanitize run (not for acceptance):
  - `exp_20260216_zimage_q40/step24_vae_512_opt/run_step24opt_zimg_decodeonly_qcomml_nohattn_sanitize2.log`
  - image: `exp_20260216_zimage_q40/step24_vae_512_opt/step24opt_zimg_decodeonly_qcomml_nohattn_sanitize2.png`
  - result: all outputs sanitized from non-finite; image unusable (near-black)
- MHA knob scans:
  - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_scan_nohattn/summary.md`
  - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_scan_nohattn_ext/summary.md`
  - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_scan_nohattn_wt0/summary.md`
- Full grid scan (`arith x softmax x bias`, 48 configs):
  - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_scan_nohattn_fullgrid/summary.md`
- Observed patterns:
  - Most configs: non-finite output + fallback
  - `MHA_WT=0` family: CLML MHA creation assert/abort (`clCreateMLOpMultiHeadAttentionForwardQCOM` failure, code -1102)
  - Full-grid statistics: `0/48` usable (`36/48` create-fail, `12/48` runtime non-finite)
  - FP32 model attempt: fails earlier in CLML op creation (GroupNorm path), not a valid workaround
  - No tested no-host config reached stable finite output in this round

## Status

- `<=2s` decode target: **met** (with host-attn backend path)
- image correctness gate: **met**
- strict “all ops fully native `qcom_ml` (no host-attn/fallback)” gate: **not met yet** (current blocker in SDK/native MHA stability)
