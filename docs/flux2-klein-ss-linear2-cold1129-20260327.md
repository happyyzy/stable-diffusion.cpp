# reqfix1-klein-ss-linear2-cold1129-20260327

Base:

- previous tag: `reqfix1-klein-qknorm-rope-qonly-20260326`
- change: add fused single-stream `linear2(attn, mlp, q8 weight)` HTP/HMX path
- source closure:
  - `ggml`: `b2bdf38fcc96d0ff69ca14f5c3e7c88833d3a816`
  - `htp_ops`: `73844baba43d6b88d7348d6518d389c787c1fe9b`

Acceptance record:

- phone: cooled before run
- mode: `steps=1`, `1024x1024`, `skip decode`
- model:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf`
- vae:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors`
- cond:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor`
- gate:
  `GGML_HTP_FLUX_SS_LINEAR2_FUSED=1`

Recorded rerun:

- `sampling completed = 11.29s`
- `GGML_OP_PROFILE total = 10.257s`
- `offload = 8.103s`
- `cpu_fallback = 2.153s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80 offload_time=8.102s unsupported_mul_mat=0 flash_seen=100 flash_mask_null=100 flash_type_mismatch=0`

Artifacts:

- summary:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/klein_flux_sslinear2_rerun_20260327/summary.json`
- off/on validation:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/klein_flux_sslinear2_ab_20260326/summary_skipdecode.json`

Runtime used for the rerun:

- `sd-cli`: `340672906441af8e498b907bd0cd3255e1dd8e03`
- `libhtp_ops_skel.so`: `b786c93c6a2e237db1e5c9fe1e4ea0974d3b7747`

Phone restore after run:

- `sd-cli`: `41697b12d1e80a7b6988faa71c0d117d4d245f5c`
- `libhtp_ops_skel.so`: `aa08b462c3daa9cf465a56f82a8995fac4a90892`
