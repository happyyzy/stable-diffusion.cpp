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

## Index tags (legacy)

- `adreno-step01` ... `adreno-step18`
- `adreno-step19-wip`

## Step20 status

- Step20 is in progress (no acceptance tag yet).
- Next canonical source tag will be `adreno-step20-src` only after `<10s` VAE decode + numeric/image gate is passed.

See full detail: `docs/adreno/README.md`.
