# Adreno Step Tags

This fork uses two tag classes:

1) `adreno-stepXX` (index tags, legacy)
- Documentation index tags created in one batch.
- They record method/result text, but are not per-step source snapshots.

2) `adreno-stepXX-src` (engineering tags, canonical)
- Source snapshot tags.
- Each tag points to the real code state that passed (or was accepted for) that step.
- Start policy: from Step18 onward.

## Current canonical tags

- `adreno-step18-src`
  - commit: `b07d269`
  - scope: z-image 1024 flash-on + h30 mldrift (4096+128) integration path
  - submodule: ggml commit `788031c1` on `happyyzy/ggml` branch `adreno-step18-src`
  - artifacts: `exp_20260216_zimage_q40/step18_1024_flashon_mldrift/`

- `adreno-step19-src`
  - commit: tag target (Step19 accepted source snapshot on `work/main`)
  - scope: z-image 1024 Step19 acceptance (`single-step <52s + finite`) with default `iofirst_chunk64` route
  - submodule: ggml commit `4cbc58d2` on `happyyzy/ggml` branch `adreno-step19-src`
  - artifacts:
    - step1: `exp_20260216_zimage_q40/step19_1024_opt/qmul_scan_20260218c/run_iofirst_chunk64.log`
    - 4-step host decode: `exp_20260216_zimage_q40/step19_1024_opt/step19_sub52_s4_hostdecode_20260218/`

- `adreno-step20-src`
  - commit: `4e7167f`
  - scope: qcom_ml VAE host-attn OpenCL backend path, z-image 1024 decode `<10s`
  - artifacts:
    - decode log: `exp_20260216_zimage_q40/step20_vae_1024_opt/run_step20_qcomml_1024_hostattn_backendggml_t40_o0.log`
    - decode image: `exp_20260216_zimage_q40/step20_vae_1024_opt/step20_hostattn_backendggml_t40_o0.png`

- `adreno-step21-src`
  - commit: `4e7167f`
  - scope: z-image 1024 8-step final gate (`<480s`) on Step19+Step20 combined path
  - artifacts:
    - attempt-1: `exp_20260216_zimage_q40/step21_final_8step/run_step21_zimg_1024_s8_qcomml.log`
    - pass rerun: `exp_20260216_zimage_q40/step21_final_8step/run_step21_zimg_1024_s8_qcomml_rerun.log`
    - pass image: `exp_20260216_zimage_q40/step21_final_8step/step21_zimg_1024_s8_qcomml_rerun.png`

- `adreno-step22-src`
  - commit: `64545f6`
  - scope: klein 1024 qcom_ml VAE decode gate (`<10s`) with host-attn ggml backend + prepare path
  - artifacts:
    - decode baseline: `exp_20260218_klein_q40/step22_vae_1024_opt/run_step22_flux2_1024_qcomml_t32_o0_v30_optmem.log`
    - pass runs: `.../run_step22_flux2_1024_qcomml_t32_o0_v33_optmem_prepare.log`, `...v34...`, `...v35...`
    - image/metrics: `exp_20260218_klein_q40/step22_vae_1024_opt/repeatability_metrics.md`

- `adreno-step23-src`
  - commit: `3f57c76`
  - scope: klein 1024 full decode qcom_ml host-attn(mldrift backend route) final gate (`<142s`)
  - artifacts:
    - full-chain pass: `exp_20260218_klein_q40/step23_full_decode_mldrift/run_step23_flux2_1024_s4_qcomml_mldfull3.log`
    - image: `exp_20260218_klein_q40/step23_full_decode_mldrift/images/step23_flux2_klein_1024_s4_qcomml_mldfull3.png`
    - detail: `exp_20260218_klein_q40/step23_full_decode_mldrift/README.md`

## Index tags (legacy)

- `adreno-step01` ... `adreno-step18`
- `adreno-step19-wip`

## WIP source tags

- `adreno-step24-wip`
  - branch: `milestone/step24-wip-source`
  - scope: Step24 diagnostic checkpoint (`<=2s` host-attn path verified; native no-host still blocked)
  - expected artifacts:
    - `exp_20260216_zimage_q40/step24_vae_512_opt/nohost_recheck_20260219.md`
    - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_scan_nohattn_wt0/summary.md`
    - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_scan_nohattn_fullgrid/summary.md`
    - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_force_heads_scan/summary.md`
    - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_desc_scan_v2/summary.md`
    - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_desc_scan_v2/summary_note.md`
    - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_desc_scan_v2/repro_nohost_opt0_disconnect_status.txt`
    - `exp_20260216_zimage_q40/step24_vae_512_opt/mha_desc_scan_v2/repro_nohost_opt0_dbg.log`

## Step20/21/22/23 status

- Step20 passed (`<10s` VAE decode gate met).
- Step21 passed in rerun (`458.41s < 480s`).
- Step22 passed (`7.98~8.68s < 10s` decode gate met).
- Step23 passed (`129.73s < 142s`, full decode path).
- canonical source tags should be maintained as `adreno-step20-src`, `adreno-step21-src`, `adreno-step22-src`, `adreno-step23-src`.

See full detail: `docs/adreno/README.md`.
