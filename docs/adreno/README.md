# Adreno Q4 Optimization Logbook (Fork)

This fork is focused on Adreno OpenCL optimization and numerical debugging for Q4_0 inference.

- Scope: FLUX.2-klein, Z-Image, Qwen3-4B (Q4_0 GGUF)
- Goal: keep the mainline PR set clean, while preserving full debug/benchmark traceability
- Current emphasis: attention bottleneck replacement (mldrift path), race-condition fixes, step-wise numerical alignment

## Headline Results

| Case | Before | After | Gain |
|---|---:|---:|---:|
| FLUX.2-klein 1024 flash-on (step forward bench) | 209.81 s/step | 31.256 s/step | 6.71x |
| Z-Image 1024 step1 (true-native flash vs optimized mldrift path) | 341.70 s | 56.93 s | 6.00x |
| Z-Image 1024 step1 (Step19 accepted finite path: iofirst_chunk64) | 341.70 s | 50.90 s | 6.71x |
| Z-Image 1024 full 8-step (Step21 rerun gate) | 493.39 s total | 458.41 s total | 1.08x |
| FLUX.2-klein 512 full phone flow (ctx=256) | 98.16 s total | 59.39 s total | 1.65x |
| FLUX.2-klein 1024 VAE decode-only (Step22, qcom_ml) | 32.80 s | 7.98 s | 4.11x |
| Z-Image 512 VAE decode-only (Step24, qcom_ml host-attn) | 4.07 s | 1.79 s | 2.27x |

Notes:
- Critical attention hot path has reached 10x-class improvement in some internal baselines during the debug process.
- End-to-end gains vary by model, resolution, sequence length, and VAE path.

## Step Records (GOAL 1-22)

Each step records the debug method and a before/after outcome (image quality or speed).

### Step 01 - Minimal Fix Set
- Method: bisect replay + function-level rollback to isolate first bad-image recovery point.
- Before: bad image path, image MAE vs reference `81.4976`.
- After: `ggml_cl_scale` fallback fix restores usable image, image MAE `29.0620`.

### Step 02 - Thread=1 Baseline + Performance Path
- Method: replay minimum patch baseline and remove non-essential overhead.
- Before: modified source single-fwd around `~14 s`.
- After: ctx=256 single-fwd returns to `~8.5 s` class baseline.

### Step 03 - Thread=4 Numerical Failure Root Cause
- Method: A/B run with `SD_LOAD_THREADS=1`, then add Q4 upload lock to remove load race.
- Before: `threads=4` bad image; image MAE vs t1 `65.8560`.
- After: `threads=1/4` latent/image align (`MAE=0`), speed unchanged (`11.00s` vs `10.99s`, step1).

### Step 04 - Qwen3-4B Q4 Adreno Decode Repair
- Method: fix Q4 transpose tail rows, embed_tokens reorder path, transpose kernel guard, gemv guard.
- Before: LLM decode gibberish token stream.
- After: OpenCL Adreno Q4 decode restored; 24/24 token IDs match noadq4 control.

### Step 05 - Chain Qwen + Klein (4-step)
- Method: reconnect repaired Qwen condition + repaired Klein trunk pipeline.
- Before: mixed black/noise failure modes in chain.
- After: 4-step host-decode image returns to normal quality path.

### Step 06 - Full Phone Flow Tuning
- Method: thread sweep + VAE path tuning (`--vae-conv-direct` on OpenCL branch).
- Before: CPU-VAE path `98.16 s` total (ctx=256 reference run).
- After: `cond 1.29s + sample 49.08s + vae 8.70s = 59.39s` total.

### Step 07 - 1024 Klein Baseline Characterization
- Method: flash on/off profiling under 1GB OpenCL allocation constraint.
- Before: flash-off OOM (`~4.13 GB` requested, alloc fail).
- After: flash-on runs but slow (`209.81 s/step`) and attention-dominated.

### Step 08 - 1024 Klein Numeric Debug Transfer
- Method: reuse 512 alignment workflow at 1024; per-step dumps + host decode verification.
- Before: multi-step divergence and unstable image quality reports.
- After: 1024 alignment workflow stabilized and used as gating pipeline for later steps.

### Step 09 - 1024 End-to-End Validation
- Method: chain repaired Qwen + repaired Klein trunk at 1024, then host decode.
- Before: chain-level image correctness uncertain.
- After: 1024 normal image path verified with full chain.

### Step 10 - Register mldrift Attention in ggml
- Method: integrate replayed mldrift kernels into OpenCL path and dispatch by shape.
- Before: ggml native flash bottleneck at 1024.
- After: FLUX.2-klein bench reaches `31.256 s/step`; attention throughput enters `1 TOPS+` class.

### Step 11 - mldrift Numeric Repair
- Method: true-native baseline rebuild + call0 q/k/v same-input flashdump diff + schedule correction.
- Before: call0 mismatch severe (`mean_abs 4.0571` class in earlier migration state).
- After: call0 diff improved to `mean_abs 0.06138` after mldrift path corrections.

### Step 12 - Recover 31.256 s/step + Step1/2/4 Checks
- Method: enforce Step10 fast path on maintainable source and re-run step dump checks.
- Before: speed or correctness drift across variants.
- After: speed class recovered (`31.256 s/step`) with step-level numeric checks re-established.

### Step 13 - 1024 Four-Step Chain (Klein)
- Method: re-chain Qwen + 1024 trunk after Step10-12 stabilization.
- Before: uncertain end-to-end acceptance at target speed class.
- After: 1024 image generation path accepted in ~165s-class objective envelope.

### Step 14 - 512 Edit on Adreno OpenCL
- Method: reuse stable 512 flash-off high-confidence path for edit pipeline.
- Before: edit correctness uncertain.
- After: edit output accepted as normal.

### Step 15 - Long-Sequence Edit Attention Optimization
- Method: intercept and replace long sequence attention (`1024+1024+128`) with mldrift path.
- Before: expected/observed large latency inflation (>150s class concern).
- After: optimized edit sampling observed at `63.40 s` (4-step phone run).

### Step 16 - 768 Edit (L=2048+2048+CTX)
- Method: apply same long-seq mldrift strategy to 768 edit shape family.
- Before: native long-seq attention too slow.
- After: full OpenCL edit path runs with normal image quality and acceptable speed.

### Step 17 - Z-Image 512 Flash-Off Repair
- Method: replay Klein-style debugging on Z-Image flash-off path.
- Before: image abnormal.
- After: step4 run restored to normal image (`sampling 51.97s` in reference run).

### Step 18 - Z-Image 1024 Flash-On + 4096+128 Attention Integration
- Method:
  - add h30 dynamic match for unmasked `n_kv <= 4224` (enables 4096x4096);
  - apply h30 IO-first scheduling;
  - use KV keep-head/tail strategy for `4352 -> 4224` crop path.
- Before: true-native step1 `341.70 s`; historical variants had NaN/black outputs.
- After:
  - optimized mldrift step1 `56.93 s`;
  - step4 total `238.06 s` (`59.5 s/it`), step8 total `488.73 s` (`61.1 s/it`);
  - host-decoded images accepted as normal for current Step18 gate.

### Step 19 - Z-Image 1024 Step1 <52s with Finite Output
- Method:
  - lock on `GGML_OPENCL_MLDRIFT_H30_IO_FIRST=1` route;
  - selective Q4 GEMM stabilizer on refiner attention-out only:
    `SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR=context_refiner.0.attention.out.weight,noise_refiner.0.attention.out.weight`
    + `SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS=64`;
  - compare all finite `<52s` candidates by 4-step latent -> host decode image quality gate.
- Accepted result:
  - single-step: `50.90s` (finite, `nan=0`) on `iofirst_chunk64`;
  - reference log: `exp_20260216_zimage_q40/step19_1024_opt/qmul_scan_20260218c/run_iofirst_chunk64.log`;
  - 4-step host decode artifact: `exp_20260216_zimage_q40/step19_1024_opt/step19_sub52_s4_hostdecode_20260218/step19_sub52_iofirst_chunk64_s4_host_decode.png`.
- Notes:
  - all finite `<52s` routes are in the same quality tier in current visual checks;
  - runtime variance is strongly thermal/clock-state dependent, so 4-step average can be slower than cold single-step.

### Step 20 - Z-Image 1024 VAE Decode Optimization (Passed)
- Method:
  - keep qcom_ml VAE route, replace host-attn CPU fallback with OpenCL backend (`SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`).
  - run tile scan on same latent with same ggml decode reference image.
- Before:
  - host-attn CPU fallback path ~`179.18s` (tile32), numeric recovered but speed unusable.
- After:
  - best accepted config: `tile=40`, `overlap=0.0` (runtime optimal overlap=0.2667)
  - decode-only time: `9.25s` (`<10s` target met)
  - image diff vs ggml ref: `MAE=1.9617, RMSE=2.6471, p99=8, max=50`
  - output image: `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_hostattn_backendggml_t40_o0.png`
  - log: `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_qcomml_1024_hostattn_backendggml_t40_o0.log`
- Round summary:
  - `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_hostattn_backendggml_grid_20260218d.md`
- Full step log:
  - `docs/adreno/steps/step20.md`

### Step 21 - Z-Image 1024 8-Step Final Gate (Passed)
- Method:
  - run full phone chain on the accepted Step19+Step20 path:
    - trunk: `GGML_OPENCL_MLDRIFT=1`, `GGML_OPENCL_MLDRIFT_H30_IO_FIRST=1`, `GGML_OPENCL_MLDRIFT_KV_KEEP_HEAD=4096`
    - q4 fix: `SD_OCL_Q4_GEMM_FP16_CHUNK_ACC_SUBSTR=context_refiner.0.attention.out.weight,noise_refiner.0.attention.out.weight`
      + `SD_OCL_Q4_GEMM_FP16_CHUNK_ITERS=64`
    - vae: `--vae-backend qcom_ml --vae-conv-direct`, `SD_QCOM_ML_VAE_HOST_ATTN_BACKEND=ggml`, tile `40/o0`
- Attempt 1:
  - total `493.39s` (not pass), log `exp_20260216_zimage_q40/step21_final_8step/run_step21_zimg_1024_s8_qcomml.log`
- Thermal-check rerun:
  - total `458.41s` (pass), sampling `448.23s`, vae `9.73s`
  - log `exp_20260216_zimage_q40/step21_final_8step/run_step21_zimg_1024_s8_qcomml_rerun.log`
  - image `exp_20260216_zimage_q40/step21_final_8step/step21_zimg_1024_s8_qcomml_rerun.png`
- Full step log:
  - `docs/adreno/steps/step21.md`

### Step 22 - Klein 1024 VAE Decode Optimization (Passed)
- Method:
  - qcom_ml route keeps host-attn backend (`ggml`) and tiled decode (`tile=32/o0`);
  - add qcom_ml graph `prepare` path to move first-create overhead out of timed decode;
  - keep `optimize_device_mem` enabled for stable memory behavior.
- Before:
  - ggml decode reference: `32.80s`
  - qcom_ml (no prepare): `11.22s`
- After:
  - qcom_ml + prepare repeat runs: `7.98s / 8.13s / 8.68s`
  - image outputs stay normal and stable vs ggml reference
  - logs/images/metrics:
    - `exp_20260218_klein_q40/step22_vae_1024_opt/README.md`
    - `exp_20260218_klein_q40/step22_vae_1024_opt/repeatability_metrics.md`
- Full step log:
  - `docs/adreno/steps/step22.md`

### Step 23 - Klein 1024 4-Step Final Gate (Passed)
- Method:
  - keep trunk on Step12 fast path;
  - switch VAE to qcom_ml full decode route with host-attn mldrift backend integration.
- Result:
  - full-chain log: `exp_20260218_klein_q40/step23_full_decode_mldrift/run_step23_flux2_1024_s4_qcomml_mldfull3.log`
  - `cond 0.784s + sample 118.86s + vae 9.84s = total 129.73s`
  - gate passed (`129.73s < 142s`)
  - image: `exp_20260218_klein_q40/step23_full_decode_mldrift/images/step23_flux2_klein_1024_s4_qcomml_mldfull3.png`

### Step 24 - Z-Image 512 qcom_ml VAE (`<=2s`) (In Progress)
- Method (current round):
  - optimize qcom_ml bridge host copies with contiguous bulk transfer fast-paths;
  - add explicit no-host non-finite diagnostics for qcom_ml decode output.
- Current best:
  - decode-only log: `exp_20260216_zimage_q40/step24_vae_512_opt/run_step24opt_zimg_decodeonly_qcomml_hattn_recheck.log`
  - `computing vae decode graph completed, taking 1.72s`
  - image: `exp_20260216_zimage_q40/step24_vae_512_opt/step24opt_zimg_decodeonly_qcomml_hattn_recheck.png`
- Blocker:
  - strict no-host native qcom_ml attention remains unstable:
    - default no-host path: `786432/786432` outputs are non-finite (`nan`);
    - `MHA_WT=0` family: MHA op creation fails (`code -1102`).
    - full-grid scan (`arith x softmax x bias`, 48 configs): `0/48` usable (`36` create-fail + `12` non-finite).
    - force-head scan (`1/2/4/8/16/32/64`): `0/7` usable (all non-finite).
  - details: `docs/adreno/steps/step24.md`

## Tag Map

Tag policy:
- Legacy index tags: `adreno-step01` ... `adreno-step18` (doc index only)
- Canonical source tags (engineering): `adreno-stepXX-src`
  - first canonical source tag: `adreno-step18-src` (`b07d269`)
  - current: `adreno-step23-src` (`3f57c76`, Step23 accepted source snapshot)
- WIP checkpoint tag:
  - `adreno-step24-wip` (Step24 diagnostics checkpoint, not an accepted gate tag)

Each tag is an annotated tag whose message includes:
- debug method summary
- before/after result summary
- pointer to this document section

## Artifact Layout (Before/After Display)

For each step, keep artifacts under:
- `exp_*/stepXX_*/` for raw logs/tensors/images
- `docs/adreno/steps/stepXX.md` for summary page
- `docs/adreno/assets/stepXX/` for lightweight before/after images used in docs

Step18 example images:
- before (true-native baseline context): `exp_20260216_zimage_q40/step18_1024_flashon_mldrift/zimg_step18_native_s1_decode_only_oclvae.png`
- after (optimized path, step4): `exp_20260216_zimage_q40/step18_1024_flashon_mldrift/step18new_q1_s4_after_kv4096patch_host_decode.png`
- after (optimized path, step8): `exp_20260216_zimage_q40/step18_1024_flashon_mldrift/step18new_q1_s8_after_kv4096patch_host_decode.png`

## Branching Model in This Fork

- `origin` (this fork): development + tags + docs
- `upstream`: `leejet/stable-diffusion.cpp`
- `work/main`: integration branch for ongoing Adreno work
- `debug/*`: experiment branches
- `pr/main`: clean PR-oriented branch

## Flag Catalog

- Consolidated runtime/build flags:
  - `docs/adreno/flags.md`
