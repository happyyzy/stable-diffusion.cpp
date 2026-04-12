# flux2-klein-ss-linear2-fused-offon-20260326

Date: 2026-03-26

Base tag:

- `reqfix1-klein-qknorm-rope-qonly-20260326`

Goal:

- Add a real HTP/HMX fused single-stream `linear2` op on top of the accepted
  Klein `qonly` closure
- Remove the host-side `concat(attn, mlp)` cost before `linear2`
- Keep numerics unchanged under the same phone/model/runtime-assets envelope

Current local runtime built from this round:

- `sd-cli`: `340672906441af8e498b907bd0cd3255e1dd8e03`
- `libhtp_ops_skel.so`: `b786c93c6a2e237db1e5c9fe1e4ea0974d3b7747`

Phone baseline restored after the run:

- `sd-cli`: `41697b12d1e80a7b6988faa71c0d117d4d245f5c`
- `libhtp_ops_skel.so`: `aa08b462c3daa9cf465a56f82a8995fac4a90892`

Run directory:

- `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/klein_flux_sslinear2_ab_20260326`

Phone assets:

- model:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf`
- vae:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors`
- cond:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor`

Step-1 off/on acceptance setup:

- exact same qonly gates as the accepted `reqfix1-klein-qknorm-rope-qonly-20260326`
- extra run knob for this round:
  - `GGML_HTP_FLUX_SS_LINEAR2_FUSED=0/1`
- exact skip-decode acceptance knob:
  - `SD_SKIP_DECODE=1`

Result:

- `off`: `sampling completed = 13.05s`
- `on`: `sampling completed = 11.98s`
- wall-clock A/B including adb shell return:
  - `off`: `15.32s`
  - `on`: `14.29s`

Numerics:

- `off` vs `on`
  - `MAE = 0.0`
  - `max_abs = 0.0`
  - `RMSE = 0.0`
  - `cos = 1.0000001192092896`
  - latent sha1:
    - `off = 96aa121c4670ac971cbaabfa2e8f3924d01ec651`
    - `on = 96aa121c4670ac971cbaabfa2e8f3924d01ec651`

Per-op movement:

- `CONCAT`: `1454.162 ms -> 51.117 ms`
- `MUL_MAT`: `5284.968 ms -> 3782.038 ms`
- `MAP_CUSTOM3`: `502.300 ms -> 2289.046 ms`
- `cpu_fallback`: `3.726s -> 2.369s`
- `offload`: `8.473s -> 8.820s`

Interpretation:

- The fused single-stream `linear2` path is active.
- It removes about `1.40s` of host `CONCAT`.
- It converts `20` former `MUL_MAT` calls into `20` additional `MAP_CUSTOM3`
  offloads:
  - `mul_mat`: `83 -> 63`
  - `other`: `60 -> 80`
- Under this exact step-1 skip-decode acceptance path, the new fused op is
  numerically exact against `off` and improves single-step sampling by about
  `1.07s`.
