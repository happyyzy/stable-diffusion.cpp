# Layout Contract v1

This document defines the v1 runtime/export contract used by `stable-diffusion.cpp-npu`.

## 1. Contract Keys in GGUF

Required model-level keys:
- `sd.npu.contract.id`
- `sd.npu.contract.layout.matmul.f16`
- `sd.npu.contract.layout.matmul.q4_0`
- `sd.npu.contract.layout.matmul.q8_0`
- `sd.npu.contract.layout.matmul.iq4_nl`
- `sd.npu.contract.layout.flash_attn_ext`
- `sd.npu.contract.export.permute`
- `sd.npu.contract.export.repack`

Current default values written by exporter:
- `sd.npu.contract.id = sdcpp.hmx.v1` (overridable by `SD_NPU_CONTRACT_ID`)
- `sd.npu.contract.layout.matmul.f16 = permute_32x32.v1`
- `sd.npu.contract.layout.matmul.{q4_0,q8_0,iq4_nl} = common_deq.v1`
- `sd.npu.contract.layout.flash_attn_ext = f32_q_f16_kv.v1`

## 2. Runtime Validation

Validation knobs:
- `GGML_HTP_CONTRACT_EXPECT=<contract_id>`
- `GGML_HTP_CONTRACT_STRICT=1`

Behavior:
- Strict mode: missing/mismatched `sd.npu.contract.id` fails load.
- Non-strict mode: runtime warns and continues.

Runtime tracking:
- Loader exports active contract into process env:
  - `GGML_HTP_CONTRACT_ACTIVE=<actual-id|<missing>>`

## 3. Offload/Fallback Traceability (Gate Requirement)

Per-op trace includes both contract version and fallback reason.

### 3.1 Op profile CSV
When `GGML_OP_PROFILE=1`:
- summary csv contains `contract_id` and `fallback_reason`
- detail csv (`GGML_OP_PROFILE_DETAIL_CSV` or default `*_detail.csv`) contains:
  - `op,device,calls,total_us,total_ms,avg_ms,total_pct,copy_bytes,contract_id,fallback_reason`

### 3.2 HTP fallback CSV
When `GGML_HTP_FALLBACK_STATS=1`:
- fallback csv includes a top line with active contract id:
  - `contract_id,<...>`
- reason counts remain per reason and per op+reason.

## 4. v1 Fallback Reason Set

Current reason taxonomy:
- `skip-by-env`
- `backend-uninitialized`
- `rms-norm-disabled`
- `unknown-op`
- `matmul-name-filter`
- `matmul-adaln-disabled`
- `matmul-min-shape`
- `matmul-contiguous-or-align`
- `matmul-f16-disabled`
- `matmul-f16-shape`
- `matmul-quant-disabled`
- `matmul-q4-shape`
- `matmul-q8-shape`
- `matmul-iq4-shape`
- `matmul-type-unsupported`
- `flash-disabled`
- `flash-contiguous-or-align`
- `flash-type-contract`
- `compute-failed-single-thread`

## 5. Notes for v1

- `copy_bytes` is deterministic offload I/O estimate (`dst + src tensors`) per op call.
- `RMS_NORM` remains CPU fallback in v1 by policy.
- This contract is intentionally narrow and must evolve with kernel support.
