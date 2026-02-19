# Adreno Runtime Flags (Step-tracked)

This file tracks non-default knobs used in accepted Adreno steps.

Rule:
- each accepted step records the exact flags it used;
- this file keeps the consolidated map and the latest verified step.

## Build-time

| Flag | Default | Purpose | Verified in |
|---|---|---|---|
| `SD_USE_QCOM_ML_VAE` | `OFF` | Enable qcom_ml VAE bridge build and `--vae-backend qcom_ml` route | Step20/21 |

## Runtime (trunk / attention / q4)

| Flag | Default | Purpose | Verified in |
|---|---|---|---|
| `GGML_OPENCL_MLDRIFT` | `0` | Enable mldrift attention route in OpenCL backend | Step19/21 |
| `GGML_OPENCL_MLDRIFT_H30_IO_FIRST` | `0` | IO-first scheduling for h30 kernels (z-image 1024 path) | Step19/21 |
| `GGML_OPENCL_MLDRIFT_KV_KEEP_HEAD` | unset | Keep head slice of KV when cropping (`4352 -> 4224`) | Step18/21 |
| `SD_FLUX2_KLEIN_MAX_LENGTH` | unset | Override Flux2-Klein conditioner max token length (used to build prompt-matched cond256) | Step28 |
| `SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR` | unset | Selective fp16 chunk-acc scope for q4 GEMM stability (by tensor name substring) | Step19/21 |
| `SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS` | unset | Chunk size for the selective q4 GEMM stabilization path | Step19/21 |
| `SD_OCL_Q4_GEMM_F32_ACT_NO_AUTO` | unset | Disable auto heuristic and rely only on explicit F32 activation-read match list | Step25 |
| `SD_OCL_Q4_GEMM_F32_ACT_SUBSTR` | unset | Enable selective Q4 GEMM F32 activation-read by tensor-name substring | Step25 |

## Runtime (qcom_ml VAE)

| Flag | Default | Purpose | Verified in |
|---|---|---|---|
| `SD_QCOM_ML_VAE_DIR` | unset | Model directory for qcom_ml VAE assets | Step20/21 |
| `SD_QCOM_ML_VAE_HOST_ATTN` | `0` | Enable host-attn fallback callback in qcom_ml route | Step20 |
| `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND` | `cpu` | Host-attn backend: `ggml|opencl|ocl|cpu` | Step20/21 |
| `SD_QCOM_ML_VAE_TRY_TILED` | `0` | Enable tiled decode for large sequence VAE decode | Step20/21 |
| `SD_QCOM_ML_VAE_TILE_SIZE` | `0` (auto) | Force tiled decode tile size | Step20/21/22 |
| `SD_QCOM_ML_VAE_TILE_OVERLAP` | `0.5` (auto) | Force tiled decode overlap | Step20/21/22 |
| `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND_PROFILE` | `0` | Print backend attention timing breakdown for profiling | Step20 |
| `SD_QCOM_ML_VAE_OPTIMIZE_MEM` | `0` | Enable optimize-device-memory model descriptor for qcom_ml bridge | Step22 |
| `SD_QCOM_ML_VAE_PREPARE` | `1` | Pre-build qcom_ml graph before timed decode (set `0` to disable) | Step22 |
| `SD_QCOM_ML_VAE_SANITIZE_NONFINITE` | `0` | Debug-only: clamp qcom_ml decode non-finite outputs to 0 instead of hard fail | Step24 (diagnostic) |
| `SD_QCOM_ML_VAE_MHA_ARITH` | unset | Override native qcom_ml MHA arithmetic mode for no-host scans (`0/1/2`) | Step24 (diagnostic) |
| `SD_QCOM_ML_VAE_MHA_SOFTMAX` | unset | Override native qcom_ml MHA softmax mode for no-host scans | Step24 (diagnostic) |
| `SD_QCOM_ML_VAE_MHA_WT` | unset | Override native qcom_ml MHA weight transform (`0:none,1:transpose`) | Step24 (diagnostic) |
| `SD_QCOM_ML_VAE_MHA_NO_ATTN_BIAS` | `0` | Disable q/k/v projection bias in native MHA path for scan | Step24 (diagnostic) |
| `SD_QCOM_ML_VAE_MHA_NO_OUT_BIAS` | `0` | Disable out projection bias in native MHA path for scan | Step24 (diagnostic) |
| `SD_QCOM_ML_VAE_MHA_FORCE_HEADS` | unset | Debug-only: force native MHA head count when divisible by channel size | Step24 (diagnostic) |

## CLI switches (used in Step20/21)

| Switch | Value | Purpose |
|---|---|---|
| `--vae-backend` | `qcom_ml` | Route VAE decode to qcom_ml bridge |
| `--vae-conv-direct` | enabled | Use direct conv path in decode graph |

## Step21 frozen preset

Use this exact set for Step21 gate replay:

- `GGML_OPENCL_MLDRIFT=1`
- `GGML_OPENCL_MLDRIFT_H30_IO_FIRST=1`
- `GGML_OPENCL_MLDRIFT_KV_KEEP_HEAD=4096`
- `SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR=context_refiner.0.attention.out.weight,noise_refiner.0.attention.out.weight`
- `SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS=64`
- `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_zimage_ae`
- `SD_QCOM_ML_VAE_TRY_TILED=1`
- `SD_QCOM_ML_VAE_HOST_ATTN=1`
- `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`

## Step22 frozen preset (Klein 1024 VAE decode)

- `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
- `SD_QCOM_ML_VAE_TRY_TILED=1`
- `SD_QCOM_ML_VAE_TILE_SIZE=32`
- `SD_QCOM_ML_VAE_TILE_OVERLAP=0`
- `SD_QCOM_ML_VAE_HOST_ATTN=1`
- `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`
- `SD_QCOM_ML_VAE_OPTIMIZE_MEM=1`

## Step25 frozen preset (Z-Image 512 8-step final gate)

- `GGML_OPENCL_USE_ADRENO_KERNELS=1`
- `GGML_OPENCL_SOA_Q=1`
- `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_zimage_ae`
- `SD_QCOM_ML_VAE_HOST_ATTN=1`
- `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`
- `SD_OCL_Q4_GEMM_F32_ACT_NO_AUTO=1`
- `SD_OCL_Q4_GEMM_F32_ACT_SUBSTR=attention.out.weight`

## Step26 frozen preset (Flux2 Klein 512 VAE decode-only)

- `GGML_OPENCL_USE_ADRENO_KERNELS=1`
- `GGML_OPENCL_SOA_Q=1`
- `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
- no-host-attn native route:
  - do not set `SD_QCOM_ML_VAE_HOST_ATTN`
  - do not set `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND`

## Step27 frozen preset (Flux2 Klein 512 4-step final gate)

- trunk:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
- qcom_ml vae:
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
  - no-host-attn native route (unset `SD_QCOM_ML_VAE_HOST_ATTN*`)
- gate run uses precomputed condition tensor:
  - `--cond-crossattn /data/local/tmp/sd_bench/host_llm_c_crossattn_256.tensor`

## Step28 frozen preset (Flux2 Klein 512 edit <=70s)

- trunk:
  - `GGML_OPENCL_USE_ADRENO_KERNELS=1`
  - `GGML_OPENCL_SOA_Q=1`
  - `GGML_OPENCL_MLDRIFT=1`
  - `GGML_OPENCL_MLDRIFT_DYNAMIC_4352=1`
  - `--diffusion-fa`
- qcom_ml vae:
  - `SD_QCOM_ML_VAE_DIR=/data/local/tmp/sd_bench/qcom_ml_flux2_vae`
  - `SD_QCOM_ML_VAE_DISABLE_MNN_ATTN=1`
  - `SD_QCOM_ML_VAE_FALLBACK_ATTN_16384=1`
- prompt-specific cond256 build:
  - `SD_FLUX2_KLEIN_MAX_LENGTH=256` + `--llm-forward-only --llm-forward-dump /data/local/tmp/sd_bench/step28_edit_cond256.tensor`
- gate run condition input:
  - `--cond-crossattn /data/local/tmp/sd_bench/step28_edit_cond256.tensor`
