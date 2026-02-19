# Step30 - Fork Showcase + PR Packaging

## Goal

- Make the fork entry docs more compelling (performance + image evidence at top-level).
- Prepare an upstream-friendly packaging path on `pr/main`:
  - neutral naming for attention/VAE acceleration terms;
  - complete switch documentation and migration guidance.

## What Was Updated

- Top-level showcase:
  - `README.md`
    - expanded Adreno fork section (performance scoreboard + before/after image table + branch model).
- Consolidated flag guide:
  - `docs/adreno/flags.md`
    - preferred neutral names (`REPLAY_*`) + legacy compatibility mapping (`MLDRIFT_* -> REPLAY_*`).
    - accepted presets for Step27/28/29.
- Logbook index:
  - `docs/adreno/README.md`
    - GOAL range updated to 1-30.
    - added Step30 packaging section and naming policy.
- Step30 visual assets:
  - `docs/adreno/assets/step30/step30_case1_before.png`
  - `docs/adreno/assets/step30/step30_case1_after.png`
  - `docs/adreno/assets/step30/step30_case2_before.png`
  - `docs/adreno/assets/step30/step30_case2_after.png`

## Before / After (Doc-side measurable deltas)

- `README.md` Adreno section lines (from title to `Important News`):
  - before: 11 lines
  - after: 39 lines
- `README.md` embedded before/after gallery images:
  - before: 0
  - after: 6
- `docs/adreno/flags.md` documented runtime/build rows:
  - before: 27
  - after: 36
- Legacy-to-neutral switch mapping entries:
  - added: 16

## Visual Proof

| Case | Before | After |
|---|---|---|
| Flux2 Klein 512 final gate | ![](../assets/step30/step30_case1_before.png) | ![](../assets/step30/step30_case1_after.png) |
| Flux2 Klein 512 edit 2-ref | ![](../assets/step30/step30_case2_before.png) | ![](../assets/step30/step30_case2_after.png) |

## Notes

- This step is repository packaging/documentation-focused; model kernels and acceptance numbers from Step27/28/29 remain unchanged.
- Runtime naming migration is documented so scripts can move from `work/main` style to `pr/main` style without ambiguity.
