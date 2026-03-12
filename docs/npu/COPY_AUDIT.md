# COPY_AUDIT (Phase 5)

## 1) Policy lock (Phase 5)

- Prohibited by default: repeated step-time runtime prepack (`permute/repack`) for the same weight buffer.
- Allowed: one-shot load-time prepack and pointer-cache reuse.
- Debug-only escape hatch: `GGML_HTP_DEBUG_DISABLE_PREPACK_CACHE=1` (forces no-cache behavior for diagnosis).

Implementation anchor:
- `ggml/src/ggml-htp/htp-ops.cc`
  - prepack cache guards in `htp_permute_quant_weight_inplace_if_needed`, `htp_repack_quant_weight_inplace_if_needed`, `htp_permute_f16_weight_inplace_if_needed`
  - prepack stats dump line: `GGML_HTP_PREPACK: ...`

## 2) Copy source inventory and elimination status

| Source category | Where measured | Current status | Notes |
|---|---|---|---|
| Offload I/O copy estimate (`offload_copy_bytes`) | `GGML_OP_PROFILE_CSV` | Measured, not eliminated | Dominated by HMX/HVX offload tensors (`dst + src` estimate per offloaded op call). |
| CPU-side copy estimate (`cpu_copy_bytes`) | `GGML_OP_PROFILE_CSV` | Measured | Tracks bytes for CPU-path profile records. |
| Runtime quant permute (`Q4/Q8/IQ4`) | `GGML_HTP_PREPACK` (`permute_*`) | Eliminated to one-shot per weight pointer | Cached unless debug cache-disable env is set. |
| Runtime quant repack (`Q4/Q8/IQ4`) | `GGML_HTP_PREPACK` (`repack_*`) | Eliminated to one-shot per weight pointer | Cached unless debug cache-disable env is set. |
| Runtime F16 permute | `GGML_HTP_PREPACK` (`permute_*`) | Eliminated to one-shot per weight pointer | Same pointer-cache rule. |
| CPU fallback compute cost | `GGML_OP_PROFILE_CSV` (`cpu_us`) | Measured, ongoing optimization target | Main indicator for non-offloaded hotspot pressure. |

## 3) Phase 5 metric definitions

Primary metrics:
- `copy_bytes_per_step = sum(total_copy_bytes) / steps`
- `fallback_cpu_time_per_step_ms = sum(cpu_us) / 1000 / steps`
- `repack_time_per_step_ms = prepack.repack_time_s * 1000 / steps`

Supplemental (from same pipeline):
- `permute_time_per_step_ms = prepack.permute_time_s * 1000 / steps`
- `prepack_total_time_per_step_ms = prepack.total_time_s * 1000 / steps`

Computation script:
- `tools/npu/calc_phase5_metrics.py`

## 4) Collection recipe (single run)

Phone run (example):

```bash
adb -s 8fd321f3 shell "cd /data/local/tmp/s07_step31 && \
export LD_LIBRARY_PATH=/data/local/tmp/s07_step31:/vendor/lib64:/system/lib64 && \
export ADSP_LIBRARY_PATH=/data/local/tmp/s07_step31\;/vendor/lib/rfsa/adsp\;/vendor/dsp && \
export SD_ACCEL_BACKEND=HTP && \
export SD_SKIP_DECODE=1 && \
export GGML_HTP_STATS=1 && \
export GGML_HTP_PREPACK_STATS=1 && \
export GGML_OP_PROFILE=1 && \
export GGML_OP_PROFILE_CSV=/data/local/tmp/s07_step31/phase5_min_op_profile.csv && \
./sd-cli --diffusion-model z_image_turbo_fromq4_q4_0_adaln0q8_outq8_core_cap_permrepack_fix1.gguf \
  --vae ae.safetensors --cond-crossattn zimg_cond.tensor --cfg-scale 1.0 --steps 1 --seed 42 \
  -W 64 -H 64 --diffusion-fa --output /data/local/tmp/s07_step31/phase5_min.png"
```

Pull outputs:

```bash
adb -s 8fd321f3 pull /data/local/tmp/s07_step31/phase5_min_op_profile.csv /tmp/phase5_min_op_profile.csv
adb -s 8fd321f3 pull /data/local/tmp/s07_step31/phase5_min_op_profile.csv_detail.csv /tmp/phase5_min_op_profile_detail.csv
```

Compute metrics:

```bash
python3 tools/npu/calc_phase5_metrics.py \
  --op-csv /tmp/phase5_min_op_profile.csv \
  --steps 1 \
  --prepack-log /tmp/sdcpp_phase5_prepack_min.log \
  --output-json /tmp/phase5_metrics_min.json
```

## 5) Snapshot (2026-03-02, validation run)

Artifacts:
- `/tmp/sdcpp_phase5_prepack_min.log`
- `/tmp/phase5_min_op_profile.csv`
- `/tmp/phase5_metrics_min.json`

Observed values:
- `copy_bytes_per_step = 4373926912`
- `fallback_cpu_time_per_step_ms = 290.009`
- `repack_time_per_step_ms = 1.0`
- `permute_time_per_step_ms = 0.0`
- prepack line present:
  - `GGML_HTP_PREPACK: permute_calls=0 ... repack_calls=1 ...`

Interpretation:
- Phase 5 instrumentation path is live end-to-end (runtime -> csv/log -> metrics json).
- Copy/fallback/prepack baselines are now auditable and scriptable for later optimization gates.
