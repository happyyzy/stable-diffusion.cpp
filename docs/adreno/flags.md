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
| `SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR` | unset | Selective fp16 chunk-acc scope for q4 GEMM stability (by tensor name substring) | Step19/21 |
| `SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS` | unset | Chunk size for the selective q4 GEMM stabilization path | Step19/21 |

## Runtime (qcom_ml VAE)

| Flag | Default | Purpose | Verified in |
|---|---|---|---|
| `SD_QCOM_ML_VAE_DIR` | unset | Model directory for qcom_ml VAE assets | Step20/21 |
| `SD_QCOM_ML_VAE_HOST_ATTN` | `0` | Enable host-attn fallback callback in qcom_ml route | Step20 |
| `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND` | `cpu` | Host-attn backend: `ggml|opencl|ocl|cpu` | Step20/21 |
| `SD_QCOM_ML_VAE_TRY_TILED` | `0` | Enable tiled decode for large sequence VAE decode | Step20/21 |
| `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND_PROFILE` | `0` | Print backend attention timing breakdown for profiling | Step20 |

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
