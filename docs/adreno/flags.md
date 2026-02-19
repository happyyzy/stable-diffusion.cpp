# Adreno Runtime Flags (Fork + PR Mapping)

This document is the single entry for all Adreno-specific knobs introduced in this fork.

- `work/main`: keeps historical knobs for experiment replay.
- `pr/main`: uses neutral names for upstream-facing patches.
- If both old/new names exist, **new name wins**.

## 1) Build-Time

| Flag | Default | Purpose |
|---|---|---|
| `SD_USE_QCOM_ML_VAE` | `OFF` | Build qcom-ml VAE bridge and enable `--vae-backend qcom_ml` |

## 2) Runtime - Attention Fast Path

### Preferred names (`pr/main`)

| Flag | Default | Purpose |
|---|---|---|
| `GGML_OPENCL_REPLAY_FA` | `0` | Enable Adreno replay flash-attention fast path |
| `GGML_OPENCL_REPLAY_DYNAMIC_4352` | `0` | Enable dynamic M support (`M < 4352`, aligned shapes) |
| `GGML_OPENCL_REPLAY_Q_SCALE_MUL` | `1.0` | Extra multiplier on Q scaling (numeric tuning) |
| `GGML_OPENCL_REPLAY_NO_SCALE` | `0` | Disable Q scaling in replay path |
| `GGML_OPENCL_REPLAY_NO_REORDER_OUT` | `0` | Skip output reorder and copy raw output buffer |
| `GGML_OPENCL_REPLAY_INPUT_MAP` | unset | Debug override for q/k/v input buffer mapping |
| `GGML_OPENCL_REPLAY_OUTPUT_BUF` | unset | Debug override for output source buffer |
| `GGML_OPENCL_REPLAY_H30_IO_FIRST` | `0` | IO-first schedule for h30 kernels |
| `GGML_OPENCL_REPLAY_H30_OP_SCHEDULE` | `0` | Replay-op schedule for h30 kernels |
| `GGML_OPENCL_REPLAY_H30_SIMPLE_ORDER` | `0` | Force simple kernel order for h30 |
| `GGML_OPENCL_REPLAY_KV_KEEP_HEAD` | unset | Keep head tokens in KV crop mode |
| `GGML_OPENCL_REPLAY_KV_KEEP_TAIL` | unset | Keep tail tokens in KV crop mode |
| `GGML_OPENCL_REPLAY_KV_CROP_TAIL` | `0` | Crop KV from tail instead of head |
| `GGML_OPENCL_REPLAY_FORCE_FINISH` | `0` | Force `clFinish` after replay attention call |

### Legacy compatibility (`work/main`)

- `GGML_OPENCL_MLDRIFT` -> `GGML_OPENCL_REPLAY_FA`
- `GGML_OPENCL_MLDRIFT_DYNAMIC_4352` -> `GGML_OPENCL_REPLAY_DYNAMIC_4352`
- `GGML_OPENCL_MLDRIFT_Q_SCALE_MUL` -> `GGML_OPENCL_REPLAY_Q_SCALE_MUL`
- `GGML_OPENCL_MLDRIFT_NO_SCALE` -> `GGML_OPENCL_REPLAY_NO_SCALE`
- `GGML_OPENCL_MLDRIFT_NO_REORDER_OUT` -> `GGML_OPENCL_REPLAY_NO_REORDER_OUT`
- `GGML_OPENCL_MLDRIFT_INPUT_MAP` -> `GGML_OPENCL_REPLAY_INPUT_MAP`
- `GGML_OPENCL_MLDRIFT_OUTPUT_BUF` -> `GGML_OPENCL_REPLAY_OUTPUT_BUF`
- `GGML_OPENCL_MLDRIFT_H30_IO_FIRST` -> `GGML_OPENCL_REPLAY_H30_IO_FIRST`
- `GGML_OPENCL_MLDRIFT_H30_OP_SCHEDULE` -> `GGML_OPENCL_REPLAY_H30_OP_SCHEDULE`
- `GGML_OPENCL_MLDRIFT_H30_SIMPLE_ORDER` -> `GGML_OPENCL_REPLAY_H30_SIMPLE_ORDER`
- `GGML_OPENCL_MLDRIFT_KV_KEEP_HEAD` -> `GGML_OPENCL_REPLAY_KV_KEEP_HEAD`
- `GGML_OPENCL_MLDRIFT_KV_KEEP_TAIL` -> `GGML_OPENCL_REPLAY_KV_KEEP_TAIL`
- `GGML_OPENCL_MLDRIFT_KV_CROP_TAIL` -> `GGML_OPENCL_REPLAY_KV_CROP_TAIL`
- `GGML_OPENCL_MLDRIFT_FORCE_FINISH` -> `GGML_OPENCL_REPLAY_FORCE_FINISH`

## 3) Runtime - Q4 Stability Controls

| Flag | Default | Purpose |
|---|---|---|
| `SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR` | unset | Enable selective fp16 chunk-acc path by tensor-name substring |
| `SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS` | unset | Chunk-acc loop count for the above path |
| `SD_OCL_Q4_GEMM_F32_ACT_NO_AUTO` | unset | Disable auto heuristic, use explicit tensor list only |
| `SD_OCL_Q4_GEMM_F32_ACT_SUBSTR` | unset | Enable selective f32 activation-read by tensor-name substring |

## 4) Runtime - QCOM-ML VAE

| Flag | Default | Purpose |
|---|---|---|
| `SD_QCOM_ML_VAE_DIR` | unset | qcom-ml model directory |
| `SD_QCOM_ML_VAE_HOST_ATTN` | `0` | Enable host-attention callback path |
| `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND` | `cpu` | `ggml/opencl/cpu/replay` |
| `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND_PROFILE` | `0` | Print host-attention backend timing |
| `SD_QCOM_ML_VAE_TRY_TILED` | `0` | Enable tiled decode |
| `SD_QCOM_ML_VAE_TILE_SIZE` | `0`(auto) | Tile size override |
| `SD_QCOM_ML_VAE_TILE_OVERLAP` | `0.5`(auto) | Tile overlap override |
| `SD_QCOM_ML_VAE_OPTIMIZE_MEM` | `0` | Enable optimize-device-memory descriptor |
| `SD_QCOM_ML_VAE_PREPARE` | `1` | Pre-build graph before timed decode |
| `SD_QCOM_ML_VAE_DISABLE_MNN_ATTN` | `0` | Disable MNN attention path in bridge |
| `SD_QCOM_ML_VAE_FALLBACK_ATTN_16384` | `0` | Allow fallback route for long-seq attention |
| `SD_QCOM_ML_VAE_REPLAY_LIB_DIR` | `/data/local/tmp/litert_bench` | Replay dispatch shared library directory |
| `SD_QCOM_ML_VAE_REPLAY_Q_SCALE_MUL` | `1.0` | q-scale multiplier for replay host-attn backend |

### Legacy compatibility (`work/main`)

- `SD_QCOM_ML_VAE_MLDRIFT_LIB_DIR` -> `SD_QCOM_ML_VAE_REPLAY_LIB_DIR`
- `SD_QCOM_ML_VAE_MLDRIFT_Q_SCALE_MUL` -> `SD_QCOM_ML_VAE_REPLAY_Q_SCALE_MUL`
- `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=mldrift` -> `replay`

## 5) CLI Switches (Adreno paths)

| Switch | Value | Purpose |
|---|---|---|
| `--vae-backend` | `qcom_ml` | Route VAE decode to qcom-ml |
| `--vae-conv-direct` | enabled | Use direct conv path in decode graph |
| `--diffusion-fa` | enabled | Enable diffusion flash-attention route |
| `--disable-auto-resize-ref-image` | enabled (edit 2-ref gate) | Keep reference input shape fixed |

## 6) Accepted Presets

### Step27 (Flux2 Klein 512 4-step `<40s`)

- trunk:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
- qcom-ml:
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
- cond:
  - `--cond-crossattn /data/local/tmp/sd_bench/host_llm_c_crossattn_256.tensor`

### Step28 (Flux2 Klein 512 edit `<70s`)

- trunk:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
  - `GGML_OPENCL_REPLAY_FA=1`
  - `GGML_OPENCL_REPLAY_DYNAMIC_4352=1`
  - `--diffusion-fa`
- qcom-ml:
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
  - `SD_QCOM_ML_VAE_DISABLE_MNN_ATTN=1`
  - `SD_QCOM_ML_VAE_FALLBACK_ATTN_16384=1`
- cond:
  - `--cond-crossattn /data/local/tmp/sd_bench/step28_edit_cond256.tensor`

### Step29 (Flux2 Klein 512 edit 2-ref `<100s`)

- keep Step28 trunk and qcom-ml settings, plus:
  - `SD_QCOM_ML_VAE_OPTIMIZE_MEM=1`
  - `SD_QCOM_ML_VAE_PREPARE=1`
  - `--disable-auto-resize-ref-image`

## 7) Migration Rule for Scripts

When moving scripts from `work/main` to `pr/main`:

1. replace all `*MLDRIFT*` env names with `*REPLAY*`;
2. replace `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=mldrift` with `replay`;
3. keep old names only for historical replay logs, not for new benchmark scripts.
