# reqfix1-q80-outstat-flashprep-directinput-cold1460-20260323

Date: 2026-03-23

Base tag:

- `reqfix1-q80-outstat-flashprep-cold1741-20260323`

Scope:

- Keep the accepted Q80 out-stationary + flashprep mainline.
- Preserve numerics first.
- Remove more host-side flash-adjacent layout work by letting fused `MAP_CUSTOM3`
  qknorm+rope consume strided `q/k` views directly.
- Keep `FLASH_QKNORM_DIRECT` on.

Source closure used for this version:

- `sdcpp/ggml`: `94da3aac` `ggml-htp: support strided zimg qknorm rope input`
- `htp_ops`: `bfbb417` `htp_ops: land flash q-direct and strided qknorm rope input`

Runtime closure used for the recorded cold run:

- `sd-cli`: `e112a2d1f567692abaf2ac97de042a9f3e6755b5`
- `libggml-htp-v79.so`: `993213ee1150bbfff60f7f2f84fbcd63e1f3fb83`
- `libhtp_ops.so`: `8413c540d6fa2394044abf193a06deb0caac0365`
- `libhtp_ops_skel.so`: `1efcc7c0e3d6320855bdee9218a02495f27ae3b1`
- model: `fix1_w2q8_exportfix_20260319.gguf` `d0a0c67e9d57cf3120c41abd897979e96113bc65`
- vae: `ae.safetensors` `95e36a32f0e2368fc24023b11c776d22c158855e`
- cond: `zimg_cond.tensor` `9c4bf53969623073b012c1481e00edb0799aa51f`

Recorded cold run:

- run dir:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/scratch/flash_direct_input_cold_on_1774275958`
- log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/scratch/flash_direct_input_cold_on_1774275958/run.log`
- shape csv:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/scratch/flash_direct_input_cold_on_1774275958/shape.csv`
- latent:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/scratch/flash_direct_input_cold_on_1774275958/latent.tensor`

Cold run result:

- `sampling completed`: `14.60s`
- `generate_image completed`: `14.61s`
- `GGML_OP_PROFILE total`: `14.078s`
- `offload`: `10.435s`
- `cpu_fallback`: `3.643s`
- latent sha1: `3d7e244f8090c63a98703751be6be7bec83270aa`

Key per-op counters from the recorded cold run:

- `FLASH_ATTN_EXT`: `2435.487 ms`
- `MAP_CUSTOM3`: `875.217 ms`
- `CONT`: `291.346 ms`

Behavior summary:

- Numerics stay aligned with the accepted latent baseline.
- The new direct-input gate is:
  `GGML_HTP_ZIMG_QKNORM_ROPE_DIRECT_INPUT=1`
- The gate is not a no-op:
  it reduces `MAP_CUSTOM3` time and removes a large part of the `CONT` chain in
  front of flash.
- The remaining main issue is flash-kernel-side time drift; host-side prep is no
  longer the dominant blocker in this segment.

Supporting A/B/A/B note:

- quick retest directory:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/scratch/flash_direct_input_abab_1774274993`
- observed `on` samples: `15.39s`, `15.67s`
- observed `off` samples: `16.02s`, `16.58s`

