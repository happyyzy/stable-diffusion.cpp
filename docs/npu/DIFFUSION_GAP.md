# Diffusion NPU Gap List (Current)

## Already implemented in this slice

- Centralized sd.cpp-side NPU policy mapping (`SD_NPU_*` -> GGML/HTP env knobs).
- HTP fallback reason taxonomy + counters + optional CSV output.
- GGUF contract metadata write/read path with optional strict validation.

## Still missing for target-level performance

1. **More offloaded non-GEMM hotspots**
   - repeated elementwise/layout ops (for example RoPE decomposition path pieces).

2. **Layout-stable fallback strategy per hotspot family**
   - avoid costly roundtrip reorder/copy patterns between offloaded and CPU ops.

3. **Diffusion-specific fused kernels**
   - q/k norm + RoPE + pack/unpack style sequences,
   - attention pre/post transforms with minimal host overhead.

4. **Exporter variants for multiple HMX contracts**
   - canonical named variants with reproducible manifests,
   - contract ID tied to exact export knobs.

## Practical next milestones

- M1: lock one reproducible contract + runtime policy for z_image UNet.
- M2: shrink CPU fallback dominant clusters with HVX kernels.
- M3: push full-offload path to acceptance metrics at 512/1024.
