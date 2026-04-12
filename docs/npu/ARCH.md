# stable-diffusion.cpp-npu Architecture (Draft)

This document defines the first implementation slice for the `stable-diffusion.cpp-npu` direction:
- keep eager execution semantics,
- prioritize HMX for heavy ops,
- keep deterministic CPU fallback for unsupported ops,
- keep the path extensible for other diffusion models.

## Runtime Layers

1. **Model/graph construction (sd.cpp)**
   - Builds GGML graph from diffusion runtime (`z_image`, `flux`, etc.).
   - Remains backend-agnostic.

2. **CPU executor + HTP hook (`ggml-cpu`)**
   - Every op is scheduled by CPU backend.
   - `ggml_htp_try_compute(...)` attempts HTP offload per op.
   - On unsupported/failure paths, op runs on CPU with the same graph semantics.

3. **HTP eager offload (`ggml-htp`)**
   - Current heavy-op support focus:
     - `GGML_OP_MUL_MAT` (HMX path for supported type/shape contracts)
     - `GGML_OP_FLASH_ATTN_EXT` (current HTP flash path)
   - Fallback reason accounting is emitted via `GGML_HTP_FALLBACK_*`.

4. **External DSP/HTP op library (`libhtp_ops.so`)**
   - Real device kernels and RPC message handling.
   - sd.cpp does not own kernel internals; it owns routing policy and observability.

## Contract Boundary

The NPU contract is now explicit at GGUF metadata level:
- `sd.npu.contract.id`
- `sd.npu.contract.layout.matmul.{f16,q4_0,q8_0,iq4_nl}`
- `sd.npu.contract.layout.flash_attn_ext`
- `sd.npu.contract.export.{permute,repack}`

`ModelLoader` validates contract ID when `GGML_HTP_CONTRACT_EXPECT` is set.
Strict mode is controlled by `GGML_HTP_CONTRACT_STRICT=1`.

## Fallback Semantics

Fallback reasons are categorized in `ggml-htp` (examples):
- backend unavailable (`skip-by-env`, `backend-uninitialized`)
- matmul gating failures (`shape`, `dtype`, `alignment`, policy filters)
- flash contract failures
- RPC compute failures in single-thread mode

Counts are available in stderr and optional CSV (`GGML_HTP_FALLBACK_CSV`).

## Near-Term Direction

- Keep heavy ops on HMX first (matmul and attention path).
- Reduce CPU fallback overhead by:
  - adding HVX kernels for repetitive elementwise/layout-heavy hotspots,
  - minimizing unnecessary layout transforms and host copies,
  - keeping contracts explicit in exporter + runtime.
