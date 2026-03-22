# Reqfix1 Q80 Out-Stationary Stage Freeze 2026-03-22

This document freezes the current stage-success state of the reqfix1 task-2
mainline.

It records a real milestone:

- current `Q80 W2 out_stationary ON` can generate a normal-looking `8-step`
  image latent on device
- that latent can be decoded successfully on host HIP

It does **not** claim task-2 acceptance under `AGENTS.md`.

## Meaning

Frozen here:

- current valid-best `1-step ON` device closure
- current `8-step ON` device latent and host HIP decode output
- exact runtime hashes for the phone closure

Not claimed here:

- exact `ON == OFF == 22.3s baseline` numerics
- cold `< 20s` single-step acceptance
- full local reproduction of the exact phone `sd-cli` binary hash

## Source Layers

- `sdcpp` source baseline before this doc-only freeze commit:
  - `13dc5576dd14e3126cca8b1fc7e214e3ef8c3d0a`
- `ggml` source baseline before freeze commit:
  - `93554bf4ff43203282d8952204d86ac3be9ba826`
- `htp_ops` source baseline before freeze commit:
  - `d1d3c78b8bb837bf267853e1f1ef8c83459da46f`

Freeze commits:

- `ggml` freeze commit:
  - `f6401b14`
  - message: `ggml-htp: freeze q80 out-stationary host plumbing state`
- `htp_ops` freeze commit:
  - `a4a0e1f`
  - message: `htp_ops: freeze current q80 out-stationary stage state`

## Device Runtime Closure

Phone directory:

- `/data/local/tmp/s07_step31`

Exact runtime hashes:

- `sd-cli`: `46d9d199f5e98e293a815e272c73b74ddefc808c`
- `libggml-htp-v79.so`: `5c1f3e42d34200a1b289763fbee88081b8de4dce`
- `libhtp_ops.so`: `8413c540d6fa2394044abf193a06deb0caac0365`
- `libhtp_ops_skel.so`: `6144d66f3856e41f64bd5e0ff30996949e32b9c1`

Local build artifacts matching the phone closure:

- local `libggml-htp-v79.so` matches `5c1f3e42...`
- local `libhtp_ops.so` matches `8413c540...`
- local `htp_ops/hexagon_ReleaseG_toolv19_v75_nosymg/ship/libhtp_ops_skel.so`
  matches `6144d66f...`

Important caveat:

- at freeze time, the exact phone `sd-cli` hash `46d9d199...` was not found in
  the visible local `*bin/sd-cli` outputs
- the visible local `sdcpp_build_fuse1_pure/bin/sd-cli` hash is
  `ee908bc2a4db34b4189fbfe682f4c3b6f179dd29`
- therefore the phone `sd-cli` is frozen here as a runtime artifact hash

## Weight / Input Closure

Device `ON` runs:

- diffusion model:
  - `/data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf`
  - hash `d0a0c67e9d57cf3120c41abd897979e96113bc65`
- VAE:
  - `/data/local/tmp/s07_step31/ae.safetensors`
  - hash `95e36a32f0e2368fc24023b11c776d22c158855e`
- cond:
  - `/data/local/tmp/s07_step31/zimg_cond.tensor`
  - hash `9c4bf53969623073b012c1481e00edb0799aa51f`

Host HIP decode workaround:

- model:
  - `/media/happyyzy/Elements SE/ggufhtp_offload_20260317_more/reqfix1_hvx_rope_main_results/task1_fresh_export_prepack_20260319/fix1_w2q8_from_bf16_overlaymerge_prepack_fresh.gguf`
  - hash `48153e96d42858a74bb5a8d9173df30d78ed6f23`
- same VAE hash `95e36a32...`
- same cond hash `9c4bf539...`

## Current Valid-Best 1-Step ON

Artifacts:

- device log:
  - `/data/local/tmp/s07_step31/task2_partitioned_fixed_on_20260322.log`
  - hash `65d1fbacfaba3b1ab22c95c27d24c25bb41cc653`
- device latent:
  - `/data/local/tmp/s07_step31/task2_partitioned_fixed_on_20260322.tensor`
  - hash `1f3a078d442e4dda3a1127caff5aed543350b83e`

Result:

- `sampling completed, taking 21.69s`
- `generate_image completed in 21.70s`
- `offload_time=14.525s`
- `MUL_MAT,HMX total=7919.527 ms`

## Current 8-Step ON Stage-Success Run

Artifacts:

- pulled log:
  - `scratch/task2_q80_outstat_20260320/current_on_s8_hipdecode_20260322/task2_current_on_s8_20260322.log`
  - hash `b94435a2bee6291ad5d075fa51bdace65445602b`
- pulled latent:
  - `scratch/task2_q80_outstat_20260320/current_on_s8_hipdecode_20260322/task2_current_on_s8_20260322.tensor`
  - hash `c8b5dc9018006d814d157f4fdb807220a8c28e23`
- decoded PNG:
  - `scratch/task2_q80_outstat_20260320/current_on_s8_hipdecode_20260322/task2_current_on_s8_20260322.hip.decoded.png`
  - hash `34a93cc877fbe003fe7ee4276d0ef9fc93a151a7`

Result:

- `sampling completed, taking 178.32s`
- latent dump succeeded
- host HIP decode succeeded
- resulting image is normal-looking and is the stage-success artifact for this
  freeze

## Conclusion

This freeze captures a meaningful task-2 stage-success checkpoint:

- `ON` no longer collapses into an obviously broken output
- `8-step` visual output is normal-looking
- the current valid-best `1-step ON` closure remains traceable

This freeze should be used as:

- a recovery point for future work
- a traceable tag anchor for `sdcpp`, `ggml`, and `htp_ops`

This freeze should **not** be used as:

- proof that task 2 is accepted
- proof that the current out-stationary path is numerically exact against the
  `22.3s` gold baseline

## Tag Name

Use the same annotated tag name in all three repos:

- `reqfix1-q80-outstat-stagefreeze-20260322`
