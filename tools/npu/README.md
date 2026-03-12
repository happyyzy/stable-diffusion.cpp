# NPU Tools

## `export_hmx_gguf.py`

Generate (and optionally execute) a reproducible multi-variant GGUF export plan.

### Dry run

```bash
python3 tools/npu/export_hmx_gguf.py \
  --model /path/to/model.safetensors \
  --output-dir /path/to/output
```

### Execute

```bash
python3 tools/npu/export_hmx_gguf.py \
  --model /path/to/model.safetensors \
  --output-dir /path/to/output \
  --cli-bin ./build-android-hex-step31/examples/cli/sdcpp \
  --execute
```

The script writes `export_manifest.json` under `--output-dir`.
Each job stores command, variant contract ID, and output path.
It also writes per-variant sidecar metadata:
- `<output-dir>/<variant>/model.npu_manifest.json`

## `run_phase0_baseline.py`

Generate or run the fixed Phase0 HIP baseline matrix:
- 512 step=1
- 512 step=8
- 1024 step=1
- 1024 step=8

```bash
python3 tools/npu/run_phase0_baseline.py \
  --bin ./build-hip/bin/sd-cli \
  --model /path/to/model.gguf \
  --out-dir /tmp/sdcpp_phase0_hip
```

Add `--execute` to run all cases and write per-case logs.

## `calc_phase5_metrics.py`

Compute Phase5 indicators from op-profile CSV and optional prepack log:
- `copy_bytes_per_step`
- `fallback_cpu_time_per_step_ms`
- `repack_time_per_step_ms`

```bash
python3 tools/npu/calc_phase5_metrics.py \
  --op-csv /path/to/op_profile.csv \
  --steps 1 \
  --prepack-log /path/to/run.log \
  --output-json /tmp/phase5_metrics.json
```
