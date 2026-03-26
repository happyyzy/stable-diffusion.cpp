# flux2-klein-qknorm-rope-qonly-20260326

Date: 2026-03-26

Goal:

- Keep the current non-`COMMON` Klein `q8only` route on top of
  `reqfix1-q80-outstat-flashprep-directinput-cold1460-20260323`
- Fuse `q/k norm + rope + layout` through the active HTP `MAP_CUSTOM3` path
- Preserve small latent drift while recovering a large part of the host-side
  `REPEAT/MUL/CONT` chain

Base source closure before this round:

- `sdcpp`: `57e8236`
- `ggml`: `94da3aac`
- `htp_ops`: `bfbb417`

Working runtime built from this round:

- `sd-cli`: `fd4b999f8d2f7430b30b0dce89e9799551365c21`
- `libggml-htp-v79.so`: `993213ee1150bbfff60f7f2f84fbcd63e1f3fb83`
- `libhtp_ops_skel.so`: `3808c6af1ee93a60ebf6e15f7567deea5ca5b550`

Phone baseline restored after each run:

- `sd-cli`: `41697b12d1e80a7b6988faa71c0d117d4d245f5c`
- `libhtp_ops_skel.so`: `aa08b462c3daa9cf465a56f82a8995fac4a90892`

Klein assets used:

- phone model dir:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324`
- model:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf`
- vae:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors`
- cond:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor`

Enabled gates for the accepted `qonly` path:

- `GGML_HTP_ZIMG_ROPE=1`
- `GGML_HTP_FLUX_QKNORM_ROPE=1`
- `GGML_HTP_FLUX_QKNORM_ROPE_DIRECT_INPUT=1`
- `GGML_HTP_FLASH_QKNORM_DIRECT=1`
- `GGML_HTP_ENABLE_F16_MATMUL=0`
- `GGML_HTP_FLASH_PREP_IN_KERNEL=0`
- `GGML_HTP_DEFER_UNMAP=1`

Explicitly not enabled in the accepted `qonly` result:

- `GGML_HTP_FLASH_QKNORM_DIRECT_K`

## Step-1 A/B

Run directory:

- `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/klein_flux_qknorm_qonly_ab_20260326`

Result:

- `off`: `sampling completed = 21.34s`
- `qonly`: `sampling completed = 14.00s`
- `off -> qonly` speedup: about `1.52x`

Numeric compare (`off` vs `qonly`):

- `MAE = 0.0023623439483344555`
- `max_abs = 0.06975579261779785`
- `RMSE = 0.003631638828665018`
- `cos = 0.9999907612800598`

Per-op delta:

- `REPEAT`: `2965.316 ms -> 0.177 ms`
- `MUL`: `2558.717 ms -> 476.351 ms`
- `CONT`: `1574.452 ms -> 649.008 ms`
- `MAP_CUSTOM3`: `0 -> 555.898 ms`
- `FLASH_ATTN_EXT`: `2792.589 ms -> 3068.573 ms`
- `CONCAT`: `1517.517 ms -> 1583.142 ms`

Interpretation:

- The fused path is active and removes most of the previous host-side
  `REPEAT/MUL/CONT` cost.
- `FLASH_ATTN_EXT` still gets slower after the fused path is enabled.
- Disabling `K direct` does not materially change the latent drift relative to
  the earlier fused run.

## Step-4 Sample + Host HIP Decode

Run directory:

- `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/qonly_qonly_1024_s4_hostdecode_20260326`

Phone-side note:

- The current `qonly` 4-step path aborts in `rpcmem_mapper` with
  `fastrpc_mmap failed with return code: 0x1` unless the per-allocation cap is
  reduced.
- The successful run used one extra mapper knob:
  - `GGML_HTP_RPCMEM_MAX_MIB=128`

Phone sampling result:

- `sampling completed = 50.92s`
- latent:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/qonly_qonly_1024_s4_hostdecode_20260326/qonly_qonly_1024_s4.latent.tensor`
- latent sha1:
  `3bd9fe7a0edcef1ce2dd00de3a35b3f693cdf45b`

Host HIP decode result:

- output image:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/qonly_qonly_1024_s4_hostdecode_20260326/qonly_qonly_1024_s4_host_decode.png`
- image sha1:
  `6e4097b97c5946e7903e0533019841fe5a447fcd`
- host decode log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/qonly_qonly_1024_s4_hostdecode_20260326/host_decode.log`
- host VAE decode time:
  `92.02s`

## Z-Image Regression Check

Run directory:

- `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/zimg_flash_directinput_recheck_20260326`

Reference tag under comparison:

- `reqfix1-q80-outstat-flashprep-directinput-cold1460-20260323`

Reference z-image latent:

- `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/scratch/flash_direct_input_cold_on_1774275958/latent.tensor`
- sha1:
  `3d7e244f8090c63a98703751be6be7bec83270aa`

Current recheck result:

- `sampling completed`: `15.76s`
- `GGML_OP_PROFILE total`: `15.233s`
- `offload`: `11.178s`
- `cpu_fallback`: `4.055s`
- current latent sha1:
  `3d7e244f8090c63a98703751be6be7bec83270aa`
- direct tensor compare vs reference:
  - `MAE = 0.0`
  - `max_abs = 0.0`
  - `RMSE = 0.0`
  - `cos = 1.0`

Current z-image drift summary:

- Numerics are exactly preserved.
- Performance regresses by about `1.16s` vs the recorded `14.60s` cold result.
- Main moved buckets:
  - `MUL_MAT`: `7194.230 ms -> 7738.908 ms`
  - `FLASH_ATTN_EXT`: `2435.487 ms -> 2584.435 ms`
  - `MAP_CUSTOM3`: `875.217 ms -> 942.370 ms`
  - `CONT`: `291.346 ms -> 318.956 ms`

## Current Closure Meaning

- For Klein step-1, the current `qonly` fused route is numerically stable enough
  for small-drift acceptance and gives a clear single-step throughput win.
- For Klein multi-step sampling, the current route is functionally usable with
  `GGML_HTP_RPCMEM_MAX_MIB=128`, and the decoded 4-step output has been verified
  through host HIP VAE decode.
- The same closure is not yet a clean replacement for the recorded z-image
  `cold1460` tag because z-image performance regresses even though numerics stay
  exact.
