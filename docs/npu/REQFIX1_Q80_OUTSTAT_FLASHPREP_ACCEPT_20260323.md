# Reqfix1 Q80 Out-Stationary FlashPrep Accept 2026-03-23

This document freezes the accepted `FLASH_ATTN_EXT` optimization state reached
on top of:

- `reqfix1-q80-outstat-kkpipe-cold1993-20260322`

It records a real accepted milestone:

- active `f16` flash path supports `null-mask + tail-mask + no-host-kv-pad`
- `ON` and `OFF` are numerically aligned on the 1-step latent
- cold single-step time is below `19s`

## Acceptance Meaning

Claimed here:

- exact `OFF/ON` latent equality for the accepted run
- accepted `ON` single-step device closure
- traceable source/runtime hashes for the accepted run

Not claimed here:

- final `kv_scale`-in-kernel acceptance
- final `15s` target

## Source Layers

Accepted source base before the doc/tag commit in `sdcpp`:

- `sdcpp` source base:
  - `dfea6fd8b10b0f2e8ac40030565819c2af409fd1`
- `ggml` accept commit:
  - `99de940d2b25b9600c3a0dad96f825403a340a99`
  - message: `ggml-htp: accept flashprep null-mask no-host-kv-pad path`
- `htp_ops` accept commit:
  - `612f3f02e10de86e189de14ec5c3ee19e1808bd1`
  - message: `htp_ops: accept active f16 flash prep path`

## Accepted Runtime Closure

Accepted run used this temporary phone closure in:

- `/data/local/tmp/s07_step31`

Accepted runtime hashes:

- `sd-cli`: `7f80279e8c31a94aada6e3214eda3d956948f526`
- `libggml-htp-v79.so`: `5c1f3e42d34200a1b289763fbee88081b8de4dce`
- `libhtp_ops.so`: `8413c540d6fa2394044abf193a06deb0caac0365`
- `libhtp_ops_skel.so`: `26fe20b65af5ce0aeae8047d38e08e011cb8a2c3`

Phone was restored after validation to the baseline closure:

- `sd-cli`: `46d9d199f5e98e293a815e272c73b74ddefc808c`
- `libggml-htp-v79.so`: `5c1f3e42d34200a1b289763fbee88081b8de4dce`
- `libhtp_ops.so`: `8413c540d6fa2394044abf193a06deb0caac0365`
- `libhtp_ops_skel.so`: `6144d66f3856e41f64bd5e0ff30996949e32b9c1`

## Weight / Input Closure

Accepted device runs used:

- diffusion model:
  - `/data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf`
  - hash `d0a0c67e9d57cf3120c41abd897979e96113bc65`
- VAE:
  - `/data/local/tmp/s07_step31/ae.safetensors`
  - hash `95e36a32f0e2368fc24023b11c776d22c158855e`
- cond:
  - `/data/local/tmp/s07_step31/zimg_cond.tensor`
  - hash `9c4bf53969623073b012c1481e00edb0799aa51f`

## Accepted OFF/ON Result

Validation method:

- one-step `1024x1024`
- `steps=1`
- precomputed `cond-crossattn`
- `skip decode`
- same source/runtime/model/weight closure
- only variable:
  - `GGML_HTP_FLASH_PREP_IN_KERNEL=0/1`

Accepted result:

- `OFF`
  - `sampling completed, taking 21.54s`
  - latent sha1 `1f3a078d442e4dda3a1127caff5aed543350b83e`
- `ON`
  - `sampling completed, taking 17.41s`
  - latent sha1 `1f3a078d442e4dda3a1127caff5aed543350b83e`

This is the accepted equality:

- `OFF latent == ON latent`
- accepted `ON` is `< 19s`

## What Changed

Accepted active flash path behavior:

- active `f16` flash core accepts `mask == null`
- tail columns are masked inside the kernel
- host-side `kv_pad` allocation is removed on the optimized null-mask path
- synthetic full mask creation is skipped on that path

Accepted host-side constraint:

- `kv_scale` is still applied on the host path
- moving `kv_scale` into the current HVX kernel path produced incorrect
  numerics and is not part of this accepted freeze

## Observed Gain

Compared with accepted `OFF`:

- `FLASH_ATTN_EXT`: `5244.685 ms -> 3401.928 ms`
- `PAD`: `941.891 ms -> 0`
- `CONCAT`: `610.908 ms -> 8.909 ms`
- `REPEAT`: `892.251 ms -> 0.155 ms`

## Artifact Paths

Accepted pulled artifacts are stored at:

- `scratch/flashprep_accept_20260323/flashprep2_mod_off_20260323.log`
- `scratch/flashprep_accept_20260323/flashprep2_mod_off_20260323.shape.csv`
- `scratch/flashprep_accept_20260323/flashprep2_mod_off_20260323.tensor`
- `scratch/flashprep_accept_20260323/flashprep2_mod_on_20260323.log`
- `scratch/flashprep_accept_20260323/flashprep2_mod_on_20260323.shape.csv`
- `scratch/flashprep_accept_20260323/flashprep2_mod_on_20260323.tensor`

## Tag Name

Use the same annotated tag name in all three repos:

- `reqfix1-q80-outstat-flashprep-cold1741-20260323`
