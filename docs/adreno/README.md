# Adreno Q4 Optimization Logbook (Fork)

This fork is focused on Adreno OpenCL optimization and numerical debugging for Q4_0 inference.

- Scope: FLUX.2-klein, Z-Image, Qwen3-4B (Q4_0 GGUF)
- Goal: keep the mainline PR set clean, while preserving full debug/benchmark traceability
- Current emphasis: attention bottleneck replacement (Replay Flash path), race-condition fixes, step-wise numerical alignment
- Naming policy:
  - `work/main` keeps historical names for experiment replay compatibility.
  - `pr/main` uses neutral upstream-facing naming (`Replay Flash`, `QCOM-ML Fast VAE`), avoiding migration-source-specific wording.

## Headline Results

### A) Same-scope performance deltas (verified record pairs)

| Scenario | Metric | Before (scope) | After | Gain | Evidence |
|---|---|---:|---:|---:|---|
| FLUX.2-klein 1024 flash-on trunk | sampling (s/step) | 209.81 (native flash, same-source) | 31.256 | 6.71x | Step07 + Step10 |
| Z-Image 1024 flash-on trunk | sampling (s/step) | 341.70 (true-native flash) | 50.90 | 6.71x | `docs/adreno/steps/step18.md` + `docs/adreno/steps/step19.md` |
| Z-Image 1024 VAE decode-only | decode (s) | 36.88 (ggml conv-direct ref) | 9.25 | 3.99x | `exp_20260216_zimage_q40/step18_1024_flashon_mldrift/run_step18_native_s1_decode_only_oclvae.log` + `docs/adreno/steps/step20.md` |
| FLUX.2-klein 1024 VAE decode-only | decode (s) | 32.80 (ggml decode reference) | 5.25 | 6.25x | `docs/adreno/steps/step22.md` + `docs/adreno/steps/step23.md` |
| FLUX.2-klein 512 VAE decode-only | decode (s) | 7.70 (ggml decode-stage reference) | 0.74 | 10.41x | `exp_20260219_localdream_auto/step31_klein_512/repro_server_loop/caseR_t8_convdirect/server_full.log` + `docs/adreno/steps/step26.md` |
| FLUX.2-klein 512 full 4-step gate | total (s) | 47.81 (runtime cond) | 38.06 (cond256 gate path) | 1.26x | `docs/adreno/steps/step27.md` |
| FLUX.2-klein 512 edit gate | total (s) | 74.96 (runtime cond) | 67.36 (cond256 + diffusion-fa) | 1.11x | `docs/adreno/steps/step28.md` |
| FLUX.2-klein 512 edit (2 refs) gate | total (s) | 100.21 (auto-resize on) | 98.93 (no-resize + optmem/prepare) | 1.01x | `docs/adreno/steps/step29.md` |
| Z-Image 512 full 8-step gate | total (s) | 104.90 | 95.99 | 1.09x | `docs/adreno/steps/step25.md` |

### B) Single-point accepted gate metrics (no strict before/after pair)

- Step24 (Z-Image 512 VAE decode, qcom_ml + host-attn backend=ggml): `1.79s`  
  - evidence: `docs/adreno/steps/step24.md`

Notes:
- This section intentionally separates "paired deltas" from "single-point gate metrics" to avoid mixed-scope rows.
- `Before` is log-derived and same-scope for each row; unless explicitly stated, it is **not** an upstream-vs-fork claim.

## Step Records (GOAL 1-30)

Each step records the debug method and a before/after outcome (image quality or speed).

### Step 01 - Minimal Fix Set
- Method: bisect replay + rollback to isolate minimum recovery set.
- Before: case A (`sd-cli-minfix-q4lock`) black image, `MAE=179.8412`.
- After: case H (`sd-cli-g2aosscale`) image recovered to usable state, `MAE=29.4714`.
- Evidence: `exp_20260214_goal_restart/step1_minimal_subset/README.md`.

### Step 02 - Thread=1 Baseline + Performance Path
- Method: keep Step1 minimum set and patch `src1 contiguous+offset`.
- Before: `step1-min` fast path `~10.79 s/it` but bad image (`img_mae_vs_ref=29.7608`).
- After: `src1fix` keeps `~10.79 s/it` with normal image (`img_mae_vs_ref=2.4404`).
- Evidence: `exp_20260214_goal_restart/step2_thread1_perf/README.md`.

### Step 03 - Thread=4 Numerical Failure Root Cause
- Method: A/B with `SD_LOAD_THREADS=1`, then add Q4 upload lock.
- Before: `threads=4` no lock gives black output (`MAE=178.2334` vs thread=1).
- After: add Q4 upload lock; `threads=1/4` align pixel-wise (`MAE=0.0`), speed `10.74 -> 10.81 s/it`.
- Evidence: `exp_20260214_goal_restart/step3_thread4/README.md`, `exp_20260214_goal_restart/step3_thread4_fix/README.md`.

### Step 04 - Qwen3-4B Q4 Adreno Decode Repair
- Method: repair Adreno Q4 decode path and verify token-level output.
- Before: old binary outputs gibberish (`token=53436`, multi-language garbage stream).
- After: new binary token IDs `24/24` match noadq4; output text restored.
- Evidence: `exp_20260214_goal_restart/step4_qwen_t4/README.md`.

### Step 05 - Chain Qwen + Klein (4-step)
- Method: reconnect repaired Qwen condition + Klein trunk.
- Before: historical chain had black/noise regressions.
- After: chain run completed (`condition=2546 ms`, `sample=55.07 s`), host decode image normal (`img_mae_vs_ref=11.3391`).
- Evidence: `exp_20260214_goal_restart/step5_qwen_flux_chain/README.md`.

### Step 06 - Full Phone Flow Tuning
- Method: thread sweep + VAE path tuning (`--vae-conv-direct`).
- Before: first full-phone run `54.79s` (`cond 1.263s + sample 44.67s + vae 8.65s`).
- After: fastest accepted chain `44.71s` (`cond 0.898s + sample 34.82s + vae 8.71s`).
- Evidence: `exp_20260214_goal_restart/step6_phone_full/README.md`.

### Step 07 - 1024 Klein Baseline Characterization
- Method: flash on/off profiling under OpenCL memory limit.
- Before: flash-off OOM (`failed to allocate ~4.13GB`).
- After: flash-on can run, but slow (`209.81 s/step`) and attention-dominated.
- Evidence: `exp_20260214_goal_restart/step7_1024_baseline/README.md`.

### Step 08 - 1024 Klein Numeric Debug Transfer
- Method: fix flash causal mis-detection and re-run step1/2/4 alignment.
- Before: multi-step divergence from causal mis-route.
- After: step1/2/4 latent mean_abs = `0.01889370 / 0.01887820 / 0.03763517`, SHA256 matches nocausal reference.
- Evidence: `exp_20260214_goal_restart/step8_1024_mainline_fix/README.md`.

### Step 09 - 1024 End-to-End Validation
- Method: run full chain on phone (Qwen condition + 4-step sampling), decode on host CPU.
- Before: only partial-chain correctness was confirmed.
- After: full chain ran through (`cond 2837 ms`, `sampling 1149.98 s`, host decode 100.01 s); correctness path confirmed, performance not yet accepted.
- Evidence: `exp_20260214_goal_restart/step9_1024_full_chain_fix/README.md`.

### Step 10 - Register mldrift Attention in ggml
- Method: integrate replay-attention path and verify throughput from existing profiling records.
- Before: native flash bottleneck dominates 1024 trunk.
- After: record-check reproduces `~32.167 s/step` and `~1.07 TOPS` (file-level recompute).
- Evidence: `exp_20260214_goal_restart/step10_mldrift_record_check/README.md`.

### Step 11 - mldrift Numeric Repair
- Method: call0 q/k/v same-input diff + schedule/order/q_scale investigations.
- Before: call0 mismatch was severe (`mean_abs=4.0571`, historical migration state).
- After: call0 improved (`mean_abs=0.06138`), but step-level multi-step numeric gate still not fully passed.
- Evidence: `exp_20260214_goal_restart/step11_mldrift_numeric/README.md`.

### Step 12 - Recover 31.256 s/step + Step1/2/4 Checks
- Method: separate `flux-bench` vs `sd-cli` execution-path comparison; replay on current source.
- Before: speed/correctness conclusions mixed by different measurement paths.
- After: `flux-bench` recovered to `30.603~31.074 s/step`; `sd-cli` path still ~`39-41 s/step`, so end-to-end speed gap remains.
- Evidence: `exp_20260214_goal_restart/step12_speed_recheck/README.md`.

### Step 13 - 1024 Four-Step Chain (Klein)
- Method: re-chain Qwen + 1024 trunk on current source with `--vae-conv-direct`.
- Before: target objective was ~`125s` class.
- After: current measured total `212.97s` (`cond 1.265s + sample 175.60s + vae 35.76s`), not meeting target.
- Evidence: `exp_20260214_goal_restart/step13_1024_full_chain/README.md`.

### Step 14 - 512 Edit on Adreno OpenCL
- Method: verify edit path on phone OpenCL and compare with host CPU reference timing.
- Before: host CPU edit reference total `202.89s`.
- After: phone OpenCL edit (`src1fix`) total `39.72s` (single-ref path); reference-edit route `84.18s`.
- Evidence: `exp_20260214_goal_restart/step14_edit_512/run_step14_*.log`.

### Step 15 - Long-Sequence Edit Attention Optimization
- Method: replace long-seq attention (`1024+1024+128`) with replay path.
- Before: 4-step sampling `256.24s`.
- After: 4-step sampling `63.40s` with accepted image quality.
- Evidence: `exp_20260214_goal_restart/step15_edit_512_attn_opt/README.md`.

### Step 16 - 768 Edit (L=2048+2048+CTX)
- Method: apply long-seq replay strategy to 768 edit shape family.
- Before: no accepted 768 edit route in this line.
- After: full OpenCL edit runs; 4-step sampling `110.47s`, image accepted.
- Evidence: `exp_20260214_goal_restart/step16_edit_768_mldrift/README.md`.

### Step 17 - Z-Image 512 Flash-Off Repair
- Method: isolate Adreno Q4 GEMM activation-precision root cause and auto-route attention weights to f32-act.
- Before: base step1 output all-NaN (`65536/65536`), image unusable.
- After: step4 sampling restored (`51.97s`) and host decode image normal.
- Evidence: `exp_20260216_zimage_q40/step17_fix_report.md`.

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
- Full step log:
  - `docs/adreno/steps/step18.md`

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
- Full step log:
  - `docs/adreno/steps/step19.md`

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
- Full step log:
  - `docs/adreno/steps/step23.md`

### Step 24 - Z-Image 512 qcom_ml VAE (`<=2s`) (Passed)
- Method (current round):
  - optimize qcom_ml bridge host copies with contiguous bulk transfer fast-paths;
  - add explicit no-host non-finite diagnostics for qcom_ml decode output.
- Current best:
  - decode-only log: `exp_20260216_zimage_q40/step24_vae_512_opt/run_step24opt_zimg_decodeonly_qcomml_hattn_recheck.log`
  - `computing vae decode graph completed, taking 1.72s`
  - image: `exp_20260216_zimage_q40/step24_vae_512_opt/step24opt_zimg_decodeonly_qcomml_hattn_recheck.png`
- Follow-up (non-blocking):
  - strict no-host native qcom_ml attention remains unstable:
    - default no-host path: `786432/786432` outputs are non-finite (`nan`);
    - `MHA_WT=0` family: MHA op creation fails (`code -1102`).
    - full-grid scan (`arith x softmax x bias`, 48 configs): `0/48` usable (`36` create-fail + `12` non-finite).
    - force-head scan (`1/2/4/8/16/32/64`): `0/7` usable (all non-finite).
    - descriptor scan v2 (`optimize_mem x recordable_queue x gmem`, fixed model dir):
      - `optimize_mem=0`: run hangs/disconnects right after `running in FLOW mode`
        - bridge debug trace stops at decoder graph create `setModelOutputs begin`
      - `optimize_mem=1`: run is stable but still `786432/786432` non-finite + fallback
  - details: `docs/adreno/steps/step24.md`

### Step 25 - Z-Image 512 8-Step Final Gate (`<100s`) (Passed)
- Method:
  - keep Step24 stable qcom_ml VAE route with host-attn backend `ggml`;
  - keep Adreno Q4 trunk and force F32 activation-read only on attention out-proj path:
    - `SD_OCL_Q4_GEMM_F32_ACT_NO_AUTO=1`
    - `SD_OCL_Q4_GEMM_F32_ACT_SUBSTR=attention.out.weight`
- Baseline (same stable host-attn route, no out-proj-only F32_ACT):
  - log: `exp_20260216_zimage_q40/step25_final_512_8step/run_step25_zimg_512_s8_hostattn_ggml_short.log`
  - sampling: `100.15s`, total: `104.90s` (not pass)
- Accepted run:
  - log: `exp_20260216_zimage_q40/step25_final_512_8step/run_step25_zimg_512_s8_outonly_hostattn_short_step25opt_new.log`
  - timing: `condition 223ms + sampling 91.22s + vae 4.51s = total 95.99s`
  - image: `exp_20260216_zimage_q40/step25_final_512_8step/step25_zimg_512_s8_outonly_hostattn_short_step25opt_new.png`
  - gate passed (`95.99s < 100s`)
- Full step log:
  - `docs/adreno/steps/step25.md`

### Step 26 - Flux2 Klein 512 qcom_ml VAE (`<=2s`) (Passed)
- Method:
  - keep qcom_ml VAE route, switch to native no-host-attn decode on Klein 512.
- Baseline (qcom_ml + host-attn backend=ggml):
  - log: `exp_20260218_klein_q40/step26_vae_512_opt/run_step26_flux2_512_decode_qcomml_hattn.log`
  - decode: `40.24s`
- Accepted run (no-host-attn):
  - log: `exp_20260218_klein_q40/step26_vae_512_opt/run_step26_flux2_512_decode_qcomml_nohattn.log`
  - decode: `0.74s` (pass)
  - image: `exp_20260218_klein_q40/step26_vae_512_opt/step26_flux2_512_decode_qcomml_nohattn.png`
- Numeric/image check:
  - `nohost_vs_hattn: mae=0.219920, p99=1`
  - `nohost_vs_hostcpu: mae=3.761077, p99=10`
- Full step log:
  - `docs/adreno/steps/step26.md`

### Step 27 - Flux2 Klein 512 4-Step Final Gate (`<40s`) (Passed)
- Method:
  - keep Step26 no-host qcom_ml VAE path;
  - use precomputed cond (`host_llm_c_crossattn_256.tensor`) to align ctx=256 acceptance path.
- Run1 (runtime cond, not pass):
  - log: `exp_20260218_klein_q40/step27_final_512_4step_run1.log`
  - `cond 1.781s + sample 42.66s + vae 3.21s = total 47.81s`
- Run2 (accepted):
  - log: `exp_20260218_klein_q40/step27_final_512_4step_run2_cond256.log`
  - `cond 15ms + sample 34.60s + vae 3.32s = total 38.06s`
  - image: `exp_20260218_klein_q40/step27_final_512_4step/step27_flux2_klein_512_s4_qcomml_nohattn_cond256.png`
- Full step log:
  - `docs/adreno/steps/step27.md`

### Step 28 - Flux2 Klein 512 Edit Final Gate (`<=70s`) (Passed)
- Method:
  - keep Step27 trunk + qcom_ml VAE route;
  - generate prompt-specific cond256 (`SD_FLUX2_KLEIN_MAX_LENGTH=256`) and use `--cond-crossattn`;
  - keep `--diffusion-fa` for edit trunk path.
- Run1 (runtime cond, not pass):
  - log: `exp_20260218_klein_q40/step28_edit_512_vae_opt/run_step28_edit_512_s4_qcomml_nohattn.log`
  - `encode 3.53s + cond 1.730s + sample 65.64s + decode 3.97s = total 74.96s`
- Run2 (accepted):
  - cond build log: `exp_20260218_klein_q40/step28_edit_512_vae_opt/run_step28_llm_forward_cond256.log`
  - gate log: `exp_20260218_klein_q40/step28_edit_512_vae_opt/run_step28_edit_512_s4_qcomml_nohattn_cond256_prompt_fa.log`
  - `encode 3.54s + cond 1ms + sample 59.47s + decode 4.33s = total 67.36s`
  - image: `exp_20260218_klein_q40/step28_edit_512_vae_opt/step28_edit_512_s4_qcomml_nohattn_cond256_prompt_fa.png`
- Full step log:
  - `docs/adreno/steps/step28.md`

### Step 29 - Flux2 Klein 512 Edit (2 refs) Final Gate (`<=100s`) (Passed)
- Method:
  - keep Step15-style mldrift trunk + qcom_ml VAE;
  - use two reference images with precomputed cond256;
  - enable `SD_QCOM_ML_VAE_OPTIMIZE_MEM=1` + `SD_QCOM_ML_VAE_PREPARE=1`;
  - run with `--disable-auto-resize-ref-image` to remove same-size ref resize overhead.
- Baseline (auto-resize on, not pass):
  - log: `exp_20260218_klein_q40/step29_edit_512_2ref/run_step29_klein_512_2ref_s4_cond256_fa.log`
  - `sampling 89.45s`, `total 100.21s`
- Accepted run:
  - log: `exp_20260218_klein_q40/step29_edit_512_2ref/run_step29_klein_512_2ref_s4_cond256_fa_noresize512_optmem.log`
  - `sampling 89.24s`, `decode 2.61s`, `total 98.93s`
  - image: `exp_20260218_klein_q40/step29_edit_512_2ref/step29_klein_512_2ref_s4_cond256_fa_noresize512_optmem.png`
- Full step log:
  - `docs/adreno/steps/step29.md`

### Step 30 - Fork/PR Packaging (Docs + Naming Cleanup) (Passed)
- Goal:
  - make fork README/records more presentation-ready (performance and image evidence visible at top level);
  - prepare upstream-friendly branch (`pr/main`) with neutral naming and complete switch guidance.
- Deliverables:
  - fork showcase docs:
    - `README.md`
    - `docs/adreno/README.md`
    - `docs/adreno/flags.md`
  - step report:
    - `docs/adreno/steps/step30.md`
  - branch policy:
    - `work/main`: performance + debug traceability
    - `pr/main`: merge-oriented patch set (clean names, minimal knobs, no debug-only wording)
  - pr/main clean-naming commit:
    - `d822d2f` (`pr: switch to replay-only naming for attention and vae knobs`)

## Tag Map

Tag policy:
- Legacy index tags: `adreno-step01` ... `adreno-step18` (doc index only)
- Canonical source tags (engineering): `adreno-stepXX-src`
  - first canonical source tag: `adreno-step18-src` (`b07d269`)
  - current: `adreno-step29-src` (Step29 accepted source snapshot)
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
