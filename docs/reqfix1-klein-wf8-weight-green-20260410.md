# reqfix1-klein-wf8-weight-green-20260410

Date: 2026-04-10

Goal:

- record the exact commands that produced the current Klein 4-step Chinese-prompt compare artifacts
- tag the current dirty-source closure after confirming Klein WF8 image quality is normal

Current source closure relative to base tag:

- base tag:
  `reqfix1-v79-zimg-klein-wf8raw-green-20260409`
- current dirty delta in `sdcpp`:
  - `model.cpp`
  - allow `GGML_TYPE_WF8_HMX_PREPACK` exports to cast skipped `BF16` tensors to `F16`
- current dirty delta in `htp_ops`:
  - `src/dsp/ops/mat_mul.c`
  - pad `M < 32` WF8 HMX rows up to tile height and zero the padded activation tail before HMX core launch

## Prompt / Cond

Prompt text file:

- `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/prompt_zh_complex.txt`

Generated Flux2-Klein cond tensor:

- `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/prompt_zh_complex_flux2_klein.cond.tensor`
- generation log:
  `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/prompt_zh_complex_flux2_klein.cond.log`

Host cond generation command:

```sh
OUT='/media/happyyzy/Elements SE/klein_q80_wf8_cmp_20260410'
HOST_BIN='/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/sdcpp_build_host_convert/bin/sd-cli'
LLM='/media/happyyzy/Data/ComfyUI_Zluda_New/models/text_encoders/qwen_3_4b.safetensors'
PROMPT=$(cat "$OUT/prompt_zh_complex.txt")

"$HOST_BIN" -v \
  --llm-forward-only \
  --llm-version flux2_klein \
  --llm "$LLM" \
  --prompt "$PROMPT" \
  --llm-forward-dump "$OUT/prompt_zh_complex_flux2_klein.cond.tensor" \
  > "$OUT/prompt_zh_complex_flux2_klein.cond.log" 2>&1
```

## Q80 Compare Image

Produced image:

- `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/q80_4step.hip.png`
- latent:
  `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/q80_4step.latent.tensor`
- device log:
  `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/q80_4step.run.log`
- decode log:
  `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/q80_4step.hip.log`

Phone-side q80 4-step command:

```sh
HOST_OUT='/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410'
REMOTE='/data/local/tmp/klein_q80_wf8_cmp_4step_20260410'
BIN='/data/local/tmp/current_source_reval_reqfix1_20260409_rel/bench_wf8_nodump'
Q80='/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf'
VAE='/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors'
COND="$REMOTE/prompt_zh_complex_flux2_klein.cond.tensor"

adb shell "rm -rf '$REMOTE' && mkdir -p '$REMOTE' '$REMOTE/q80' '$REMOTE/wf8'"
adb push "$HOST_OUT/prompt_zh_complex_flux2_klein.cond.tensor" "$COND"
adb shell "pkill -9 sd-cli || true"
adb shell "cd '$REMOTE/q80' && env \
  LD_LIBRARY_PATH='$BIN:/vendor/lib64:/system/lib64' \
  ADSP_LIBRARY_PATH='$BIN;/vendor/lib/rfsa/adsp;/vendor/dsp' \
  SD_ACCEL_BACKEND=HTP \
  SD_NPU_BACKEND=HTP \
  SD_SKIP_DECODE=1 \
  SD_NPU_OP_PROFILE=1 \
  SD_NPU_HTP_STATS=1 \
  GGML_HTP_ENABLE_F16_MATMUL=0 \
  GGML_HTP_ENABLE_QUANT_MATMUL=1 \
  GGML_HTP_ENABLE_FLASH_ATTN=1 \
  GGML_HTP_ENABLE_ADALN_MATMUL=1 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_ZIMG_ROPE=1 \
  GGML_HTP_FLUX_SS_LINEAR2_FUSED=1 \
  GGML_HTP_FLUX_QKNORM_ROPE=1 \
  GGML_HTP_FLUX_QKNORM_ROPE_DIRECT_INPUT=1 \
  GGML_HTP_FLASH_QKNORM_DIRECT=1 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_MAP_MIB=4608 \
  GGML_HTP_MAX_ACTIVE_MAPS=8 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  GGML_HTP_MATMUL_SPLIT_M=0 \
  GGML_HTP_MATMUL_SPLIT_M_QKV=0 \
  GGML_HTP_MATMUL_SPLIT_M_OUT=0 \
  GGML_HTP_MATMUL_SPLIT_M_W2=0 \
  GGML_HTP_MATMUL_SPLIT_M_W13=0 \
  GGML_HTP_MATMUL_SPLIT_M_ADALN=0 \
  SD_DUMP_LATENT='$REMOTE/q80/q80_4step.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/q80/q80_4step.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/q80/q80_4step.shape.csv' \
  '$BIN/sd-cli' -v \
    --diffusion-model '$Q80' \
    --vae '$VAE' \
    --cond-crossattn '$COND' \
    --cfg-scale 1.0 --steps 4 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output '$REMOTE/q80/q80_4step.out.png' \
    > '$REMOTE/q80/q80_4step.run.log' 2>&1"

adb pull "$REMOTE/q80/q80_4step.latent.tensor" "$HOST_OUT/q80_4step.latent.tensor"
adb pull "$REMOTE/q80/q80_4step.run.log" "$HOST_OUT/q80_4step.run.log"
```

Host decode command that produced `q80_4step.hip.png`:

```sh
OUT='/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410'
HOST_BIN='/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/sdcpp_build_host_convert/bin/sd-cli'
VAE='/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/models/flux2-vae.safetensors'
Q80='/media/happyyzy/Data/flux2_klein_q80_ref_20260410/flux-2-klein-4b-q8_0-hmx-q8only.gguf'
COND="$OUT/prompt_zh_complex_flux2_klein.cond.tensor"

"$HOST_BIN" -v \
  --diffusion-model "$Q80" \
  --vae "$VAE" \
  --cond-crossattn "$COND" \
  --load-latent "$OUT/q80_4step.latent.tensor" \
  --vae-tiling \
  --output "$OUT/q80_4step.hip.png" \
  > "$OUT/q80_4step.hip.log" 2>&1
```

Observed q80 result summary:

- `sampling completed, taking 50.53s`
- `GGML_HTP_STATS: offloaded_ops=672 mul_mat=252 flash_attn=100 other=320`
- `flash_seen=400 flash_mask_null=400`

## WF8 Compare Image

Produced image:

- `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/wf8_4step_refcmd.step30hip.png`
- latent:
  `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/wf8_4step_refcmd.latent.tensor`
- device log:
  `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/wf8_4step_refcmd.run.log`
- decode log:
  `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/wf8_4step_refcmd.step30hip.log`

Phone-side WF8 4-step command.
This is the exact `flux2_wf8_fixbuild_qkrope_perf_20260410` runtime closure,
with only the model / cond / steps changed:

```sh
BASE=/data/local/tmp/flux2_wf8_fixbuild_qkrope_perf_20260410
REMOTE=/data/local/tmp/flux2_wf8_fixbuild_qkrope_perf_20260410_zh4
COND=/data/local/tmp/klein_q80_wf8_cmp_4step_20260410/prompt_zh_complex_flux2_klein.cond.tensor
HOST_OUT='/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410'

adb shell "pkill -9 sd-cli || true"
adb shell "rm -rf '$REMOTE' && mkdir -p '$REMOTE'"
adb shell "cp '$BASE/sd-cli' '$BASE/libhtp_ops.so' '$BASE/libhtp_ops_skel.so' '$BASE/libggml-htp-v79.so' '$BASE/libomp.so' '$REMOTE/'"
adb shell "chmod 755 '$REMOTE/sd-cli' '$REMOTE/libhtp_ops.so' '$REMOTE/libhtp_ops_skel.so' '$REMOTE/libggml-htp-v79.so' '$REMOTE/libomp.so'"

adb shell "cd '$REMOTE' && env \
  LD_LIBRARY_PATH='$REMOTE:/vendor/lib64:/system/lib64' \
  ADSP_LIBRARY_PATH='$REMOTE;/vendor/lib/rfsa/adsp;/vendor/dsp' \
  SD_ACCEL_BACKEND=HTP \
  SD_NPU_BACKEND=HTP \
  SD_SKIP_DECODE=1 \
  SD_NPU_OP_PROFILE=1 \
  SD_NPU_HTP_STATS=1 \
  GGML_HTP_ENABLE_F16_MATMUL=0 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_ZIMG_ROPE=1 \
  GGML_HTP_FLUX_SS_LINEAR2_FUSED=1 \
  GGML_HTP_FLUX_QKNORM_ROPE=1 \
  GGML_HTP_FLUX_QKNORM_ROPE_DIRECT_INPUT=1 \
  GGML_HTP_FLASH_QKNORM_DIRECT=1 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_MAP_MIB=3072 \
  GGML_HTP_MAX_ACTIVE_MAPS=16 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  SD_DUMP_LATENT='$REMOTE/latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/shape.csv' \
  ./sd-cli -v \
    --diffusion-model /data/local/tmp/current_source_reval_reqfix1_20260409_rel/models/flux-2-klein-4b-wf8_hmx-nobf16-fixbuild-20260409.gguf \
    --vae /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors \
    --cond-crossattn '$COND' \
    --cfg-scale 1.0 --steps 4 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output '$REMOTE/out.png' \
    > '$REMOTE/run.log' 2>&1"

adb pull "$REMOTE/latent.tensor" "$HOST_OUT/wf8_4step_refcmd.latent.tensor"
adb pull "$REMOTE/run.log" "$HOST_OUT/wf8_4step_refcmd.run.log"
adb pull "$REMOTE/op.csv" "$HOST_OUT/wf8_4step_refcmd.op.csv"
adb pull "$REMOTE/shape.csv" "$HOST_OUT/wf8_4step_refcmd.shape.csv"
```

Host HIP decode command that produced `wf8_4step_refcmd.step30hip.png`:

```sh
OUT='/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410'
HIP_BIN='/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/stable-diffusion.cpp-step30/build-hip/bin/sd-cli'
Q80='/media/happyyzy/Data/flux2_klein_q80_ref_20260410/flux-2-klein-4b-q8_0-hmx-q8only.gguf'
VAE='/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/models/flux2-vae.safetensors'
COND="$OUT/prompt_zh_complex_flux2_klein.cond.tensor"

"$HIP_BIN" -v \
  --diffusion-model "$Q80" \
  --vae "$VAE" \
  --cond-crossattn "$COND" \
  --load-latent "$OUT/wf8_4step_refcmd.latent.tensor" \
  --vae-tiling \
  --output "$OUT/wf8_4step_refcmd.step30hip.png" \
  > "$OUT/wf8_4step_refcmd.step30hip.log" 2>&1
```

Observed WF8 result summary:

- `sampling completed, taking 54.31s`
- `GGML_HTP_STATS: offloaded_ops=672 mul_mat=332 flash_attn=100 other=240`
- `flash_seen=400 flash_mask_null=400`
- host decode backend:
  - `Using CUDA backend`
  - `ggml_cuda_init: found 1 ROCm devices`
  - `computing vae decode graph completed, taking 12.51s`
  - `save result image 0 to '/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/wf8_4step_refcmd.step30hip.png' (success)`

## Acceptance Artifact

WF8 image accepted as normal:

- `/media/happyyzy/Elements SE/klein_q80_wf8_cmp_4step_20260410/wf8_4step_refcmd.step30hip.png`
