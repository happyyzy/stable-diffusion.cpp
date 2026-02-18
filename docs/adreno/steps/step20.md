# Step20 - Z-Image 1024 VAE Decode Optimization

Status: passed (2026-02-18, round-d)

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
- Tuned path on (`GGML_OPENCL_CONV2D_TUNED=1`): regression to `41.44s`
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20opt8_vae_decode_ocl_convdirect_tunedon.log`
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

## 2026-02-18: QCOM ML VAE route wiring (ongoing, not passed)

### This round changes

- Added build switch (default OFF): `SD_USE_QCOM_ML_VAE=ON`
- Added runtime switch: `--vae-backend ggml|qcom_ml`
- Added bridge scaffold for full VAE decode-subgraph routing:
  - new files: `qcom_ml_vae_bridge.hpp`, `qcom_ml_vae_bridge.cpp`
  - library entry ABI expected from SDK side:
    - `sd_qcom_ml_vae_create`
    - `sd_qcom_ml_vae_decode`
    - `sd_qcom_ml_vae_destroy`
- Policy hard-coded in bridge config:
  - `disable_mnn_attention = true`
  - `fallback_attn_len_16384 = true` (only the unsupported node should fallback on SDK side)

### Runtime behavior

- Default path remains unchanged (`--vae-backend ggml`)
- `--vae-backend qcom_ml` bridge inputs:
  - `SD_QCOM_ML_VAE_LIB` (default: `libsd_qcom_ml_vae.so`)
  - `SD_QCOM_ML_VAE_DIR` (fallback to `--vae` value)
- If `--vae-backend qcom_ml` but bridge/lib not ready:
  - logs warning once
  - falls back to existing ggml VAE path

### CLML route repro evidence (independent SDK bench)

- Deployed Fast-Diffusion SDXL decoder bench on phone (Adreno 830)
- Run log: `exp_20260216_zimage_q40/step20_vae_1024_opt/clml_route_sdxl_repro/run_sdxl_vae_decoder_1024_random.log`
- Result: `[Perf] ... s/it=9.36008` (1024 decode, random latent, SDK bench route)
- Repro note: `exp_20260216_zimage_q40/step20_vae_1024_opt/clml_route_sdxl_repro/report_20260218.md`

### Current status vs Step20 goal

- Step20 still **not passed**:
  - z-image/flux VAE SDK backend implementation is not complete yet
  - bridge ABI is ready; SDK-side implementation and node-level fallback still pending

## 2026-02-18 (later): Flux AE bridge bring-up and guard rails

### This round changes

- Added SDK-side bridge project in-tree:
  - `thirdparty/qcom_ml_vae_sdk_bridge/`
- Added Flux AE tensor dump helper:
  - `thirdparty/qcom_ml_vae_sdk_bridge/dump_flux_ae_weights.py`
- Updated local decoder bridge implementation toward Flux AE layout:
  - remove `post_quant_conv`
  - use `latent_c=16`
  - map weight names to Flux-style prefixes (e.g. `decoder_mid_attn_1_*`, `decoder_up_*_block_*`)
- Added bridge-side pre-checks (`sd_qcom_ml_vae_bridge.cpp`) to avoid hard abort:
  - validate Flux decoder tensor layout exists
  - reject non-Flux latent channel count
  - pre-reject `seq_len >= 16384` when `fallback_attn_len_16384=1`

### 1024 decode-only result (z-image latent)

- Runtime bridge init succeeds:
  - `run_step20_qcomml_1024_fallback_v_phone.log`
- Current behavior:
  - qcom_ml route rejects `L=16384` and falls back to ggml:
    - `QCOM ML VAE decode failed: attention sequence length 16384 is not supported...`
  - fallback decode time: `32.87s`
- Output image:
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_qcomml_1024_fallback_v.png`
  - image equals ggml baseline decode output (pixel diff `MAE=0` against `step20_zimg_vae_decode_ocl_convdirect_prof.png`)

### 512 decode-only result (z-image latent)

- Run reaches `running in FLOW mode` but does not complete stably on device.
- Current run artifact:
  - `run_step20_qcomml_512c_phone.log`
- Observed issue:
  - adb session drops during decode stage; suspected GPU/driver reset in current qcom_ml path.
  - enabling `SD_QCOM_ML_VAE_DEBUG=1` still fails before complete end-stage trace is emitted.

### Artifacts

- Round report:
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/report_step20_qcom_ml_flux_bridge_20260218.md`
- Image diff record (fallback vs ggml baseline):
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_qcomml_1024_fallback_vs_ggml.md`
- Flux AE dumped decoder tensors:
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/qcom_ml_zimage_ae/weights/decoder`

## 2026-02-18 (latest): tiled decode probing + safety fixes

### Code updates

- `stable-diffusion.cpp/stable-diffusion.cpp`
  - Added optional tiled qcom_ml decode route for large seq (`SD_QCOM_ML_VAE_TRY_TILED=1`).
  - Added non-finite output guard in qcom_ml decode: if any `NaN/Inf` appears, return failure and fallback.
  - Switched large-seq default to **safe fallback** (no tiled attempt unless explicitly enabled).
  - Added optional diagnostics:
    - `SD_QCOM_ML_VAE_DUMP_STATS=1` (input/output min/max/mean)
    - `SD_QCOM_ML_VAE_TILE_DEBUG=1`
    - `SD_QCOM_ML_VAE_SKIP_LATENT_OUT=1` (debug only)
- `thirdparty/qcom_ml_vae_sdk_bridge/sd_qcom_ml_vae_bridge.cpp`
  - Added debug flush.
  - Added optional `SD_QCOM_ML_VAE_FP32=1` probe (fails on groupnorm create in current SDK path).
- `thirdparty/qcom_ml_vae_sdk_bridge/clml_decoder.cpp`
  - Added optional `SD_QCOM_ML_VAE_DISABLE_ATTN=1` probe switch for mid attention block.

### Key findings

- Native qcom_ml attention path is currently the blocker:
  - `tile=64` (`seq=4096`) fails during decoder graph creation.
  - `tile=32` (`seq=1024`) can run but attention output is non-finite (`NaN`).
- With `SD_QCOM_ML_VAE_DISABLE_ATTN=1` (diagnostic route), tiled decode runs and is fast:
  - `run_step20_qcomml_1024_trytiled_noattn_phone.log`
  - decode time: `10.59s`
  - output image: `step20_qcomml_1024_trytiled_noattn_v.png`
  - vs ggml reference (`step20_ggml_1024_ref_v3.png`): `MAE=2.0204`, `p99=9`, `max=56`
- Safe default fallback remains correct:
  - `run_step20_qcomml_1024_safefallback_v5_phone.log`
  - fallback output: `step20_qcomml_1024_safefallback_v5.png`
  - pixel match vs ggml ref (`step20_ggml_1024_ref_v3.png`): `MAE=0`, `max=0`
  - comparison table: `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_tiled_probe_summary_20260218.md`

### Current verdict (Step20)

- Still **not passed**:
  - true node-level fallback for the `L=16384` attention node is not integrated yet
  - qcom_ml native attention path is unstable/non-finite for this VAE route

## 2026-02-18 (round b): MHA descriptor/scale probe (still blocked)

### This round changes

- Added MHA debug knobs in `thirdparty/qcom_ml_vae_sdk_bridge/clml_decoder.cpp` (default-off):
  - `SD_QCOM_ML_VAE_MHA_ARITH` (`0/1/2` -> fp16/fp16_acc32/fp32)
  - `SD_QCOM_ML_VAE_MHA_SOFTMAX` (`0..3`)
  - `SD_QCOM_ML_VAE_MHA_WT` (`0/1`)
  - `SD_QCOM_ML_VAE_MHA_NO_ATTN_BIAS`, `SD_QCOM_ML_VAE_MHA_NO_OUT_BIAS`

### Probe summary (1024 decode-only, tile=32)

- Full record: `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_mha_probe_20260218b.md`
- Key outcomes:
  - default / `ARITH=1` / `ARITH=2` / `NO_ATTN_BIAS=1`: all still hit `non-finite output from qcom_ml decode` and fallback to ggml.
  - `MHA_WT=0` and `MHA_SOFTMAX=0`: `clCreateMLOpMultiHeadAttentionForwardQCOM failed, code=-1102` (graph create assert abort).
  - q-weight pre-scale test (`1/sqrt(512)`): still non-finite and fallback.

### Updated conclusion

- The issue is not a simple arithmetic-mode or light calibration mismatch.
- Step20 remains **not passed**; next required path is attention node-level semantic fallback or external stable attention replacement.

## 2026-02-18 (round c): host-attn dtype 修补 + 1024 复测

### 代码修补

- `thirdparty/qcom_ml_vae_sdk_bridge/clml_decoder.cpp`
  - 修补 host-attn 输出上传：按目标 tensor dtype 做转换后再 `uploadDataIntoGPUMem`。
  - 支持 `CL_FLOAT / CL_HALF_FLOAT / CL_ML_BFLOAT16_QCOM`。
  - 目的：修复此前 host-attn 将 `float*` 直接写入 FP16 tensor 导致的数值畸变。

### 关键结果（同一 latent: `step19_portrait_s8_latent_v2.tensor`）

- ggml 基线（decode-only）：
  - `run_step20_hostattn_ref_ggml.log`
  - `33.02s`
  - 输出：`step20_hostattn_ref_ggml.png`

- host-attn（修补前，tile32）：
  - `run_step20_qcomml_1024_hostattn_tiled32.log`
  - `203.92s`
  - 相对 ggml 图像误差：`MAE=55.8259, p99=185, max=247`（错误）

- host-attn（修补后，tile32）：
  - `run_step20_qcomml_1024_hostattn_typedfix_tiled32.log`
  - `179.18s`
  - 输出：`step20_qcomml_1024_hostattn_typedfix_tiled32.png`
  - 相对 ggml 图像误差：`MAE=2.0383, RMSE=2.7688, p99=8, max=47`（数值恢复）

- no-attn（tile32）：
  - `run_step20_noattn_ref_tiled32_v2.log`
  - `11.05s`
  - 输出：`step20_noattn_ref_tiled32_v2.png`
  - 相对 ggml：`MAE=2.0205, p99=9, max=56`

- no-attn（tile40, overlap=0.5）：
  - `run_step20_noattn_t40_o50.log`
  - `8.58s`
  - 输出：`step20_noattn_t40_o50.png`
  - 相对 ggml：`MAE=1.8412, p99=8, max=66`

- no-attn（tile40, overlap=0.4 请求 -> 实际 0.45）：
  - `run_step20_noattn_t40_o40.log`
  - `8.47s`
  - 输出：`step20_noattn_t40_o40.png`
  - 相对 ggml：`MAE=1.8410, p99=8, max=66`

- native MHA（tile16）：
  - `run_step20_qcomml_1024_tiled16_nativeattn.log`
  - 仍触发 `non-finite output from qcom_ml decode`，回退 ggml。
- native MHA（tile40）：
  - `run_step20_nativeattn_t40_o50.log`
  - 同样触发 `non-finite output from qcom_ml decode`，回退 ggml。

### 当前结论

- host-attn 路径的“数值错乱”已修复（dtype 上传 bug 修复有效）。
- 但 host-attn CPU attention 仍过慢，无法满足 Step20 `<10s`。
- 当前唯一达到 `<10s` 的是 no-attn+tiled40（8.58s），但该路径禁用 attention，不能作为 Step20 最终解。
- Step20 继续保持 **not passed**。

参考记录：`exp_20260216_zimage_q40/step20_vae_1024_opt/step20_hostattn_dtype_fix_20260218c.md`

## 2026-02-18 (round d): host-attn OpenCL backend + tile 策略收敛（通过）

### 代码改动（本轮）

- `thirdparty/qcom_ml_vae_sdk_bridge/clml_decoder.cpp`
  - `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml|opencl|ocl`：在 host-attn 回调中启用 OpenCL backend（失败自动回退 CPU 实现）。
  - OpenCL backend 按 head 维度计算（避免把多头误算成单头）。
  - host-attn 数据结构新增 backend/profiling 开关。
- 部署库：
  - `thirdparty/qcom_ml_vae_sdk_bridge/build-android-arm64/libsd_qcom_ml_vae.so`

### 运行口径

- 同一 latent：`/data/local/tmp/sd_bench/step19_portrait_s8_latent_v2.tensor`
- 同一参考图：`step20_hostattn_ref_ggml.png`
- 关键环境：
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_zimage_ae`
  - `SD_QCOM_ML_VAE_TRY_TILED=1`
  - `SD_QCOM_ML_VAE_HOST_ATTN=1`
  - `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`

### 扫描结果（速度+数值）

- 结果表：`exp_20260216_zimage_q40/step20_vae_1024_opt/step20_hostattn_backendggml_grid_20260218d.md`
- 最优配置：`tile=40`, `overlap=0.0`（实际日志显示 optimal overlap=0.2667）
  - 日志：`exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_qcomml_1024_hostattn_backendggml_t40_o0.log`
  - 输出：`exp_20260216_zimage_q40/step20_vae_1024_opt/step20_hostattn_backendggml_t40_o0.png`
  - decode 时间：`9.25s`（<10s）
  - 相对 ggml 参考误差：`MAE=1.9617, RMSE=2.6471, p99=8, max=50`

### 结论

- Step20 目标达成：z-image 1024 VAE decode 在 qcom_ml + ggml(OpenCL host-attn backend) 路线下达到 `<10s`，图像与参考一致性可接受。
