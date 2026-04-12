# LLaMA.cpp-NPU Pattern Applied to Diffusion

This project follows the same engineering pattern that worked for LLM eager HMX integration.

## Core Pattern

1. **External high-performance op library**
   - keep HMX/HVX kernels in dedicated ops backend (`libhtp_ops.so`).

2. **Thin runtime routing layer in ggml**
   - per-op capability check,
   - strict input/output/layout contracts,
   - reliable CPU fallback.

3. **Contract-first model export**
   - exporter writes contract metadata (`sd.npu.contract.*`),
   - runtime validates contract before execution in strict mode.

4. **Always-on observability**
   - op-time profile for full graph,
   - fallback reason accounting by category and op.

## Why this matters for diffusion

Diffusion graphs have many layout-sensitive lightweight ops around heavy GEMM/attention kernels.
A stable integration requires:
- contract clarity (layout + quant/dequant semantics),
- explicit fallback accounting,
- controlled expansion from heavy-op offload to long-tail op coverage.

This minimizes blind debugging and makes portability to new DiT-family models predictable.
