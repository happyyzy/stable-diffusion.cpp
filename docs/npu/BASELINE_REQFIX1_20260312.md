# ReqFix1 Z-Image Baseline (2026-03-12)

This note records the first "looks-correct" end-to-end images for the
`full-offload + flash on + prepacked (split_m=0)` baseline, and the matching
1024 UNet step profile used for performance work.

## Images (8-step)

External-drive folder:

- `/media/happyyzy/Elements SE/from_150gb_ubuntu_disk/exp_s07_hmx_port_20260226/reqfix1_images_8step_ab_20260312/`

Key outputs:

- `zimg_512_s8_baseline_prepacked_1773298934.png`
- `zimg_512_s8_ablate_coreq4_and_out_cpu_1773299073.png` (diagnostic; very slow)
- `zimg_1024_s8_baseline_prepacked_1773301585.png` (VAE decode with `--vae-tiling`)

The corresponding `*.log` files are in the same directory.

## 1024 UNet 1-step op-profile (skip decode)

Folder:

- `/media/happyyzy/Elements SE/from_150gb_ubuntu_disk/exp_s07_hmx_port_20260226/reqfix1_opprof_1024_fix1_20260312/`

Top-level timing split (per-step):

- Total: ~34.769s
- CPU: 17.964s (51.67%)
- HMX: 11.817s (33.99%)
- HVX: 4.988s (14.35%)

Main hotspots (per-step):

- CPU RoPE decomposition (`REPEAT+MUL+ADD 2x64x4128x30`): ~8.700s
- HVX `FLASH_ATTN_EXT`: ~4.676s
- HMX GEMMs:
  - FFN `w2`: ~4.611s
  - FFN `w1/w3`: ~3.568s
  - Attn `qkv`: ~1.985s
  - Attn `out_proj`: ~0.771s

