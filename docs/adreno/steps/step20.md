# Step20 - Z-Image 1024 VAE Decode Optimization (WIP)

Status: in progress (not passed)

## Goal

- z-image turbo, 1024 resolution, Adreno OpenCL VAE decode
- target `<10s`
- keep numeric/image correctness vs host decode baseline

## Current baseline

- decode-only, same latent, `--vae-conv-direct`, `threads=1`
- decode time: `33.29s`
- log: `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_vae_decode_ocl_convdirect_prof.log`

## Optrace / bottleneck

- op timing run: `34.36s`
- top op: `CONV_2D 29754.848 ms` (dominant bottleneck)
- log: `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_vae_decode_ocl_convdirect_optime.log`

Per-op detail (`GGML_OPENCL_OP_TIMING_DETAIL=1`) shows the heaviest conv output shapes:

- `ne=[512,512,256,1]` (7 calls, total `6588.674 ms`)
- `ne=[1024,1024,128,1]` (7 calls, total `6571.885 ms`)
- `ne=[256,256,512,1]` (7 calls, total `6439.028 ms`)
- plus heavy singles:
  - `ne=[512,512,512,1]` (`3672.878 ms`)
- `ne=[1024,1024,256,1]` (`3640.317 ms`)

With v2 detail logging (includes src shapes), top conv specs are:

- `w=[3,3,512,512], x=[256,256,512,1] -> y=[256,256,512,1]`
- `w=[3,3,256,256], x=[512,512,256,1] -> y=[512,512,256,1]`
- `w=[3,3,128,128], x=[1024,1024,128,1] -> y=[1024,1024,128,1]`
- `w=[3,3,512,512], x=[512,512,512,1] -> y=[512,512,512,1]`
- `w=[3,3,256,256], x=[1024,1024,256,1] -> y=[1024,1024,256,1]`

Artifacts:

- `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_vae_decode_optime_detail.log`
- `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_vae_decode_optime_detail_v2.log`
- `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_conv_shape_summary.csv`
- `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_conv_node_top20.csv`
- `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_conv_spec_summary_v2.csv`
- `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_next_conv_targets.md`

## Control experiments

- no conv-direct path: OOM (requested ~8.5GB compute buffer)
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_vae_decode_noconvdirect.log`
- `--force-sdxl-vae-conv-scale`: `33.23s` (no material gain)
- `threads=8`: `33.24s` (no material gain)
- f16 VAE weights only: `34.45s` (no material gain)

## Numeric / image check

- decode images remain normal and close to host decode baseline
- diff table:
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_image_diff_vs_host.md`

## Code touch (instrumentation)

- `ggml/src/ggml-opencl/ggml-opencl.cpp`
  - add optional env gate `GGML_OPENCL_OP_TIMING_DETAIL=1`
  - when `GGML_OPENCL_OP_TIMING=1` is enabled, print per-op timing detail:
    - op type
    - node name
    - output shape (`ne`)
    - input/weight shapes (`src0_ne`, `src1_ne`)

This instrumentation is default-off and does not change runtime path unless env is enabled.

## 2026-02-18: tuned conv probing (still not passed)

- Added experimental conv knobs (default off):
  - `GGML_OPENCL_CONV2D_TUNED=1`: enable VAE 3x3 tuned tile variants (`BS_CRS=32`, `BS_NPQ=128`)
  - `GGML_OPENCL_CONV2D_QCOM_ACCEL16=1`: append `-qcom-accelerate-16-bit` for conv programs (Adreno only)
- Added dtype in op timing detail line (`dst_type/src0_type/src1_type`) to close type ambiguity.

### Results

- Default path (both knobs off): `33.17s`
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20opt7_vae_decode_ocl_convdirect_default.log`
- Tuned path on (`GGML_OPENCL_CONV2D_TUNED=1`): no material gain (~33s)
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20opt4_vae_decode_ocl_convdirect_tunedlog.log`
- QCOM accel16 on (`GGML_OPENCL_CONV2D_QCOM_ACCEL16=1`): regression to `41.24s`
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20opt6_vae_decode_ocl_convdirect_qcom16.log`

### New bottleneck certainty

- `CONV_2D` heavy calls are all `src0=f16, src1=f32` (39 calls total):
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_conv_dtype_spec_summary.md`
- Top specs remain:
  - `w=[3,3,512,512], x=[256,256,512,1] -> y=[256,256,512,1]`
  - `w=[3,3,256,256], x=[512,512,256,1] -> y=[512,512,256,1]`
  - `w=[3,3,128,128], x=[1024,1024,128,1] -> y=[1024,1024,128,1]`

See full run log:
- `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20opt5_vae_decode_optime_detail_types.log`
