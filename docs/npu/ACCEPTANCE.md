# Acceptance Targets (Current Plan)

These are the practical acceptance thresholds for the current `stable-diffusion.cpp-npu` implementation phase.

## Primary Performance Targets

- **HMX full-offload + flash on path**
  - 512 single UNet forward: `<= 8 s`
  - 1024 single UNet forward: `<= 30 s`

## Primary Numeric Targets (against HIP baseline)

- UNet single-step latent MAE: `<= 3e-2`
- UNet 8-step latent MAE: `<= 2e-1`

## Observability Requirements

Each acceptance run must produce:

1. **Op-time profile (full graph)**
   - `GGML_OP_PROFILE=1`
   - CSV output (`GGML_OP_PROFILE_CSV`, optional shape CSV)

2. **HTP fallback accounting**
   - `GGML_HTP_FALLBACK_STATS=1`
   - optional reason CSV via `GGML_HTP_FALLBACK_CSV`

3. **Contract metadata traceability**
   - exported GGUF includes `sd.npu.contract.*` keys
   - runtime can enforce expected contract ID with:
     - `GGML_HTP_CONTRACT_EXPECT=<id>`
     - `GGML_HTP_CONTRACT_STRICT=1`

## Run Classification

When reporting results, always include:
- backend (`SD_ACCEL_BACKEND` / initialized backend name),
- flash mode on/off,
- offloaded/fallbacked op distribution,
- model/export contract ID.

## Phase0 Fixed Baseline Template

- Fixed reference command generator:
  - `tools/npu/run_phase0_baseline.py`
- Fixed reference knobs (HIP baseline):
  - `seed=42`
  - `sampling_method=euler`
  - `scheduler=discrete`
  - `cfg_scale=7.0`
- Fixed case matrix:
  - `512x512, steps=1`
  - `512x512, steps=8`
  - `1024x1024, steps=1`
  - `1024x1024, steps=8`
