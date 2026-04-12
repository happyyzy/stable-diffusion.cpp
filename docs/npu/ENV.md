# NPU Runtime Environment Variables

## High-level sd.cpp policy vars (`SD_NPU_*`)

These are consumed by sd.cpp and mapped to lower-level GGML/HTP knobs.

- `SD_NPU_BACKEND`
  - Sets `SD_ACCEL_BACKEND` (for example `MyHTP`, `HTP`, `Hexagon`).
- `SD_NPU_OP_PROFILE=1`
  - Enables `GGML_OP_PROFILE=1`.
- `SD_NPU_OP_PROFILE_CSV=<path>`
  - Sets `GGML_OP_PROFILE_CSV`.
- `SD_NPU_OP_PROFILE_SHAPE_CSV=<path>`
  - Sets `GGML_OP_PROFILE_SHAPE_CSV`.
- `SD_NPU_HTP_STATS=1`
  - Enables `GGML_HTP_STATS=1`.
- `SD_NPU_HTP_FALLBACK=1`
  - Enables `GGML_HTP_FALLBACK_STATS=1`.
- `SD_NPU_HTP_FALLBACK_CSV=<path>`
  - Sets `GGML_HTP_FALLBACK_CSV`.
- `SD_NPU_CONTRACT_STRICT=1`
  - Sets `GGML_HTP_CONTRACT_STRICT=1`.
- `SD_NPU_EXPECT_CONTRACT_ID=<id>`
  - Sets `GGML_HTP_CONTRACT_EXPECT=<id>`.

## Low-level HTP vars (direct)

- `GGML_HTP_STATS=1`
- `GGML_HTP_FALLBACK_STATS=1`
- `GGML_HTP_FALLBACK_CSV=<path>`
- `GGML_HTP_CONTRACT_EXPECT=<id>`
- `GGML_HTP_CONTRACT_STRICT=1`

Existing HTP kernel gates remain valid (`GGML_HTP_ENABLE_*`, `GGML_HTP_MATMUL_*`, etc.).

## Example

```bash
export SD_NPU_BACKEND=MyHTP
export SD_NPU_OP_PROFILE=1
export SD_NPU_OP_PROFILE_CSV=/tmp/zimg_op_profile.csv
export SD_NPU_OP_PROFILE_SHAPE_CSV=/tmp/zimg_op_profile_shape.csv
export SD_NPU_HTP_STATS=1
export SD_NPU_HTP_FALLBACK=1
export SD_NPU_HTP_FALLBACK_CSV=/tmp/zimg_htp_fallback.csv
export SD_NPU_EXPECT_CONTRACT_ID=sdcpp.hmx.v1
export SD_NPU_CONTRACT_STRICT=1
```
