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

## Index tags (legacy)

- `adreno-step01` ... `adreno-step18`
- `adreno-step19-wip`

See full detail: `docs/adreno/README.md`.
