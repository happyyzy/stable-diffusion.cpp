# reqfix1-current-source-6case-single-step-20260412

Date: 2026-04-12

Goal:

- record the 8 current-source single-step acceptance commands used in this rerun
- keep one exact rerun closure for:
  - `z-image q40+q80 v75`
  - `z-image q40+q80 v79`
  - `flux2 klein 4b q80 v75`
  - `flux2 klein 4b q80 v79`
  - `z-image wf8 v79`
  - `flux2 klein 4b wf8 v79`
  - `flux2 klein 4b q80 v75 2k`
  - `flux2 klein 4b wf8 v79 2k`

Current source closure:

- `htp_ops`:
  `$WORKTREE_ROOT/htp_ops`
- `sdcpp`:
  `$WORKTREE_ROOT/sdcpp`
- current build output:
  `$VALIDATE_OUT/build`

Variables used below:

- `$WORKTREE_ROOT`: checked-out worktree root
- `$VALIDATE_OUT`: host-side build/output root for this validation batch
- `$RESULTS_ROOT`: host-side results root for this validation batch
- `$PHONE_RUNTIME_ROOT`: phone-side runtime root
- `$PHONE_MODEL_PACK`: phone-side 4-weight model pack
- `$PHONE_AUX`: phone-side auxiliary weights dir
- `$PHONE_ZIMG_COND`: phone-side Z-Image cond tensor
- `$PHONE_KLEIN_COND`: phone-side Klein cond tensor
- `$LIBOMP_PATH`: host-side Android `libomp.so`

Important runtime note:

- old `zimg/flux qknorm+rope` env names were cleaned up in current source
- these reruns use:
  - `GGML_HTP_ZIMG_ROPE=1`
  - `GGML_HTP_DIT_QKNORM_ROPE=1`
- removed old debug/dirty envs are not used here:
  - no `GGML_HTP_MATMUL_SPLIT_M*`
  - no `GGML_HTP_FLUX_QKNORM_ROPE_DIRECT_INPUT`
  - no `GGML_HTP_FLASH_QKNORM_DIRECT`
  - no runtime 33-tensor f16 override env
- `2k` `klein` runs need a smaller map window to let mappings recycle earlier:
  - `GGML_HTP_DEFER_UNMAP=0`
  - reduced `GGML_HTP_MAX_MAP_MIB`
  - reduced `GGML_HTP_MAX_ACTIVE_MAPS`

## Runtime Staging

Host build root:

- `$VALIDATE_OUT/build`

Phone runtime root:

- `$PHONE_RUNTIME_ROOT`

Phone model pack:

- `$PHONE_MODEL_PACK`

Phone aux weights:

- `$PHONE_AUX`

Push runtime command:

```sh
set -euo pipefail

SER=8fd321f3
VAL='$VALIDATE_OUT'
BASE=$PHONE_RUNTIME_ROOT
LIBOMP=$LIBOMP_PATH

adb -s $SER shell "rm -rf $BASE && mkdir -p $BASE/v75 $BASE/v79"

for arch in v75 v79; do
  adb -s $SER push "$VAL/build/sdcpp_android_rel/bin/sd-cli" $BASE/$arch/sd-cli
  adb -s $SER push "$VAL/build/htp_android_rel/libhtp_ops.so" $BASE/$arch/libhtp_ops.so
  adb -s $SER push "$VAL/build/htp_hex_${arch}_rel/libhtp_ops_skel.so" $BASE/$arch/libhtp_ops_skel.so
  adb -s $SER push "$VAL/build/sdcpp_android_rel/ggml/src/ggml-hexagon/libggml-htp-v79.so" $BASE/$arch/libggml-htp-v79.so
  adb -s $SER push "$LIBOMP" $BASE/$arch/libomp.so
  adb -s $SER shell "chmod 755 $BASE/$arch/sd-cli $BASE/$arch/libhtp_ops.so $BASE/$arch/libhtp_ops_skel.so $BASE/$arch/libggml-htp-v79.so $BASE/$arch/libomp.so"
done
```

## `z-image q40+q80 v75`

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=$PHONE_RUNTIME_ROOT/v75
OUT='$RESULTS_ROOT/zimg_v75'

mkdir -p "$OUT"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/zimg.*"

adb -s $SER shell "cd $REMOTE && env \
  LD_LIBRARY_PATH='$REMOTE:/vendor/lib64:/system/lib64' \
  ADSP_LIBRARY_PATH='$REMOTE;/vendor/lib/rfsa/adsp;/vendor/dsp' \
  SD_ACCEL_BACKEND=HTP \
  SD_NPU_BACKEND=HTP \
  SD_SKIP_DECODE=1 \
  SD_NPU_OP_PROFILE=1 \
  SD_NPU_HTP_STATS=1 \
  GGML_HTP_ENABLE_F16_MATMUL=0 \
  GGML_HTP_ENABLE_QUANT_MATMUL=1 \
  GGML_HTP_ENABLE_FLASH_ATTN=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=1 \
  GGML_HTP_ENABLE_ADALN_MATMUL=1 \
  GGML_HTP_ENABLE_CAP_EMBED_MATMUL=1 \
  GGML_HTP_ENABLE_NOISE_REFINER_W2_MATMUL=1 \
  GGML_HTP_ZIMG_ROPE=1 \
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_MAX_MAP_MIB=3072 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_ACTIVE_MAPS=16 \
  GGML_HTP_SKIP_CORE_PERMUTE_REPACK=0 \
  GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS=0 \
  GGML_HTP_RUNTIME_REPACK_QWEIGHTS=0 \
  SD_DUMP_LATENT='$REMOTE/zimg.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/zimg.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/zimg.shape.csv' \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/fix1_w2q8_exportfix_20260319.gguf \
    --vae $PHONE_AUX/ae.safetensors \
    --cond-crossattn $PHONE_ZIMG_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/zimg.out.png \
    > $REMOTE/zimg.run.log 2>&1"

adb -s $SER pull "$REMOTE/zimg.latent.tensor" "$OUT/zimg.latent.tensor"
adb -s $SER pull "$REMOTE/zimg.run.log" "$OUT/run.log"
adb -s $SER pull "$REMOTE/zimg.op.csv" "$OUT/op.csv"
adb -s $SER pull "$REMOTE/zimg.shape.csv" "$OUT/shape.csv"
```

Observed result:

- `mae vs doc latent = 0.0018010765`
- `sampling completed = 16.18s`
- `GGML_OP_PROFILE total = 15.621s`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`

## `z-image q40+q80 v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=$PHONE_RUNTIME_ROOT/v79
OUT='$RESULTS_ROOT/zimg_v79'

mkdir -p "$OUT"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/zimg.*"

adb -s $SER shell "cd $REMOTE && env \
  LD_LIBRARY_PATH='$REMOTE:/vendor/lib64:/system/lib64' \
  ADSP_LIBRARY_PATH='$REMOTE;/vendor/lib/rfsa/adsp;/vendor/dsp' \
  SD_ACCEL_BACKEND=HTP \
  SD_NPU_BACKEND=HTP \
  SD_SKIP_DECODE=1 \
  SD_NPU_OP_PROFILE=1 \
  SD_NPU_HTP_STATS=1 \
  GGML_HTP_ENABLE_F16_MATMUL=0 \
  GGML_HTP_ENABLE_QUANT_MATMUL=1 \
  GGML_HTP_ENABLE_FLASH_ATTN=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=1 \
  GGML_HTP_ENABLE_ADALN_MATMUL=1 \
  GGML_HTP_ENABLE_CAP_EMBED_MATMUL=1 \
  GGML_HTP_ENABLE_NOISE_REFINER_W2_MATMUL=1 \
  GGML_HTP_ZIMG_ROPE=1 \
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_MAX_MAP_MIB=3072 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_ACTIVE_MAPS=16 \
  GGML_HTP_SKIP_CORE_PERMUTE_REPACK=0 \
  GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS=0 \
  GGML_HTP_RUNTIME_REPACK_QWEIGHTS=0 \
  SD_DUMP_LATENT='$REMOTE/zimg.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/zimg.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/zimg.shape.csv' \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/fix1_w2q8_exportfix_20260319.gguf \
    --vae $PHONE_AUX/ae.safetensors \
    --cond-crossattn $PHONE_ZIMG_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/zimg.out.png \
    > $REMOTE/zimg.run.log 2>&1"

adb -s $SER pull "$REMOTE/zimg.latent.tensor" "$OUT/zimg.latent.tensor"
adb -s $SER pull "$REMOTE/zimg.run.log" "$OUT/run.log"
adb -s $SER pull "$REMOTE/zimg.op.csv" "$OUT/op.csv"
adb -s $SER pull "$REMOTE/zimg.shape.csv" "$OUT/shape.csv"
```

Observed result:

- `mae vs doc latent = 0.0017842893`
- `sampling completed = 16.27s`
- `generating 1 latent images completed = 16.31s`
- `GGML_OP_PROFILE total = 15.751s`
- `offload = 11.416s`
- `cpu_fallback = 4.335s`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`

## `flux2 klein 4b q80 v75`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=$PHONE_RUNTIME_ROOT
SRC=$BASE/v75
REMOTE=$BASE/klein_v75
OUT='$RESULTS_ROOT/klein_v75'

mkdir -p "$OUT"
adb -s $SER shell "rm -rf $REMOTE && mkdir -p $REMOTE && cp $SRC/sd-cli $SRC/libhtp_ops.so $SRC/libhtp_ops_skel.so $SRC/libggml-htp-v79.so $SRC/libomp.so $REMOTE/"
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/latent.tensor $REMOTE/run.log $REMOTE/op.csv $REMOTE/shape.csv"

adb -s $SER shell "cd $REMOTE && env \
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
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  SD_DUMP_LATENT='$REMOTE/latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/shape.csv' \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/flux-2-klein-4b-q8_0-hmx-q8only.gguf \
    --vae $PHONE_AUX/flux2-vae.safetensors \
    --cond-crossattn $PHONE_KLEIN_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/out.png \
    > $REMOTE/run.log 2>&1"

adb -s $SER pull "$REMOTE/latent.tensor" "$OUT/latent.tensor"
adb -s $SER pull "$REMOTE/run.log" "$OUT/run.log"
adb -s $SER pull "$REMOTE/op.csv" "$OUT/op.csv"
adb -s $SER pull "$REMOTE/shape.csv" "$OUT/shape.csv"
```

Observed result:

- `mae vs doc latent = 0`
- `sampling completed = 12.36s`
- `GGML_OP_PROFILE total = 11.582s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- `flash_seen=100 flash_mask_null=100`

## `flux2 klein 4b q80 v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=$PHONE_RUNTIME_ROOT
SRC=$BASE/v79
REMOTE=$BASE/klein_v79
OUT='$RESULTS_ROOT/klein_v79'

mkdir -p "$OUT"
adb -s $SER shell "rm -rf $REMOTE && mkdir -p $REMOTE && cp $SRC/sd-cli $SRC/libhtp_ops.so $SRC/libhtp_ops_skel.so $SRC/libggml-htp-v79.so $SRC/libomp.so $REMOTE/"
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/latent.tensor $REMOTE/run.log $REMOTE/op.csv $REMOTE/shape.csv"

adb -s $SER shell "cd $REMOTE && env \
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
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  SD_DUMP_LATENT='$REMOTE/latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/shape.csv' \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/flux-2-klein-4b-q8_0-hmx-q8only.gguf \
    --vae $PHONE_AUX/flux2-vae.safetensors \
    --cond-crossattn $PHONE_KLEIN_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/out.png \
    > $REMOTE/run.log 2>&1"

adb -s $SER pull "$REMOTE/latent.tensor" "$OUT/latent.tensor"
adb -s $SER pull "$REMOTE/run.log" "$OUT/run.log"
adb -s $SER pull "$REMOTE/op.csv" "$OUT/op.csv"
adb -s $SER pull "$REMOTE/shape.csv" "$OUT/shape.csv"
```

Observed result:

- `mae vs doc latent = 0.0020561247`
- `sampling completed = 12.00s`
- `generating 1 latent images completed = 12.04s`
- `GGML_OP_PROFILE total = 11.156s`
- `offload = 8.480s`
- `cpu_fallback = 2.676s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- `flash_seen=100 flash_mask_null=100`

## `z-image wf8 v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=$PHONE_RUNTIME_ROOT
REMOTE=$BASE/wf8raw_repro_current
SRC=$BASE/v79
OUT='$RESULTS_ROOT/zimg_wf8raw_s1'

mkdir -p "$OUT"
adb -s $SER shell "rm -rf $REMOTE && mkdir -p $REMOTE && cp $SRC/sd-cli $SRC/libhtp_ops.so $SRC/libhtp_ops_skel.so $SRC/libggml-htp-v79.so $SRC/libomp.so $REMOTE/"
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/wf8raw_s1_current.*"

adb -s $SER shell "cd $REMOTE && env \
  LD_LIBRARY_PATH='$REMOTE:/vendor/lib64:/system/lib64' \
  ADSP_LIBRARY_PATH='$REMOTE;/vendor/lib/rfsa/adsp;/vendor/dsp' \
  SD_ACCEL_BACKEND=HTP \
  SD_NPU_BACKEND=HTP \
  SD_SKIP_DECODE=1 \
  SD_NPU_OP_PROFILE=1 \
  SD_NPU_HTP_STATS=1 \
  SD_NPU_HTP_FALLBACK=1 \
  GGML_HTP_ENABLE_F16_MATMUL=0 \
  GGML_HTP_ENABLE_QUANT_MATMUL=1 \
  GGML_HTP_ENABLE_FLASH_ATTN=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=1 \
  GGML_HTP_ENABLE_ADALN_MATMUL=1 \
  GGML_HTP_ENABLE_CAP_EMBED_MATMUL=1 \
  GGML_HTP_ENABLE_NOISE_REFINER_W2_MATMUL=1 \
  GGML_HTP_ZIMG_ROPE=1 \
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_MAX_MAP_MIB=3072 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_ACTIVE_MAPS=16 \
  GGML_HTP_SKIP_CORE_PERMUTE_REPACK=0 \
  GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS=0 \
  GGML_HTP_RUNTIME_REPACK_QWEIGHTS=0 \
  SD_DUMP_LATENT='$REMOTE/wf8raw_s1_current.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/wf8raw_s1_current.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/wf8raw_s1_current.shape.csv' \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/z_image_turbo_f16base3_wf8hmx_20260409.gguf \
    --vae $PHONE_AUX/ae.safetensors \
    --cond-crossattn $PHONE_ZIMG_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/wf8raw_s1_current.out.png \
    > $REMOTE/wf8raw_s1_current.run.log 2>&1"

adb -s $SER pull "$REMOTE/wf8raw_s1_current.latent.tensor" "$OUT/latent.tensor"
adb -s $SER pull "$REMOTE/wf8raw_s1_current.run.log" "$OUT/run.log"
adb -s $SER pull "$REMOTE/wf8raw_s1_current.op.csv" "$OUT/op.csv"
adb -s $SER pull "$REMOTE/wf8raw_s1_current.shape.csv" "$OUT/shape.csv"
```

Observed result from the 6-case validation run:

- `mae vs wf8 reference latent = 0.0348836660`
- `sampling completed = 17.24s`
- `GGML_OP_PROFILE total = 16.756s`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`

Latest same-closure rerun:

- `sampling completed = 15.29s`
- `generating 1 latent images completed = 15.34s`
- `GGML_OP_PROFILE total = 14.757s`
- `offload = 10.372s`
- `cpu_fallback = 4.385s`
- log:
  `$RESULTS_ROOT/zimg_v79_pair_retime_20260412/wf8.run.log`

## `flux2 klein 4b wf8 v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=$PHONE_RUNTIME_ROOT
SRC=$BASE/v79
REMOTE=$BASE/klein_wf8_1step
OUT='$RESULTS_ROOT/klein_wf8_1step'

mkdir -p "$OUT"
adb -s $SER shell "rm -rf $REMOTE && mkdir -p $REMOTE && cp $SRC/sd-cli $SRC/libhtp_ops.so $SRC/libhtp_ops_skel.so $SRC/libggml-htp-v79.so $SRC/libomp.so $REMOTE/"
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/latent.tensor $REMOTE/run.log $REMOTE/op.csv $REMOTE/shape.csv"

adb -s $SER shell "cd $REMOTE && env \
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
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  SD_DUMP_LATENT='$REMOTE/latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/shape.csv' \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/flux-2-klein-4b-wf8_hmx-nobf16-fixbuild-20260409.gguf \
    --vae $PHONE_AUX/flux2-vae.safetensors \
    --cond-crossattn $PHONE_KLEIN_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/out.png \
    > $REMOTE/run.log 2>&1"

adb -s $SER pull "$REMOTE/latent.tensor" "$OUT/latent.tensor"
adb -s $SER pull "$REMOTE/run.log" "$OUT/run.log"
adb -s $SER pull "$REMOTE/op.csv" "$OUT/op.csv"
adb -s $SER pull "$REMOTE/shape.csv" "$OUT/shape.csv"
```

Observed result:

- `mae vs current q80 v79 latent = 0.0568612415`
- `sampling completed = 12.56s`
- `generating 1 latent images completed = 12.60s`
- `GGML_OP_PROFILE total = 11.824s`
- `offload = 7.743s`
- `cpu_fallback = 4.081s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=83 flash_attn=25 other=60`
- `flash_seen=100 flash_mask_null=100`

## `flux2 klein 4b q80 v75 2k`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=$PHONE_RUNTIME_ROOT
SRC=$BASE/v75
REMOTE=$BASE/klein_v75_q80_2k_tuned
RUN=$REMOTE/run
OUT='$RESULTS_ROOT/klein_2k_retry_mapmib_20260412/v75_q80_2k_m384_a4'

mkdir -p "$OUT"
adb -s $SER shell "pkill -9 sd-cli || true; rm -rf $REMOTE && mkdir -p $RUN && cp $SRC/sd-cli $SRC/libhtp_ops.so $SRC/libhtp_ops_skel.so $SRC/libggml-htp-v79.so $SRC/libomp.so $REMOTE/"
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $RUN/*"

adb -s $SER shell "cd $REMOTE && env \
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
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_DEFER_UNMAP=0 \
  GGML_HTP_MAX_MAP_MIB=384 \
  GGML_HTP_MAX_ACTIVE_MAPS=4 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/flux-2-klein-4b-q8_0-hmx-q8only.gguf \
    --vae $PHONE_AUX/flux2-vae.safetensors \
    --cond-crossattn $PHONE_KLEIN_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 2048 -H 2048 -t 4 --diffusion-fa \
    --output $RUN/out.png \
    > $RUN/run.log 2>&1"
```

Observed result:

- `sampling completed = 81.17s`
- `generating 1 latent images completed = 81.22s`
- `GGML_OP_PROFILE total = 78.129s`
- `offload = 68.325s`
- `cpu_fallback = 9.804s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- `flash_seen=100 flash_mask_null=100`
- log:
  `$RESULTS_ROOT/klein_2k_retry_mapmib_20260412/v75_q80_2k_m384_a4/run.log`

## `flux2 klein 4b wf8 v79 2k`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=$PHONE_RUNTIME_ROOT
SRC=$BASE/v79
REMOTE=$BASE/klein_v79_wf8_2k_tuned2
RUN=$REMOTE/run
OUT='$RESULTS_ROOT/klein_2k_retry_mapmib_20260412/v79_wf8_2k_m384_a6'

mkdir -p "$OUT"
adb -s $SER shell "pkill -9 sd-cli || true; rm -rf $REMOTE && mkdir -p $RUN && cp $SRC/sd-cli $SRC/libhtp_ops.so $SRC/libhtp_ops_skel.so $SRC/libggml-htp-v79.so $SRC/libomp.so $REMOTE/"
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"
adb -s $SER shell "pkill -9 sd-cli || true; rm -f $RUN/*"

adb -s $SER shell "cd $REMOTE && env \
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
  GGML_HTP_DIT_QKNORM_ROPE=1 \
  GGML_HTP_DEFER_UNMAP=0 \
  GGML_HTP_MAX_MAP_MIB=384 \
  GGML_HTP_MAX_ACTIVE_MAPS=6 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  ./sd-cli -v \
    --diffusion-model $PHONE_MODEL_PACK/flux-2-klein-4b-wf8_hmx-nobf16-fixbuild-20260409.gguf \
    --vae $PHONE_AUX/flux2-vae.safetensors \
    --cond-crossattn $PHONE_KLEIN_COND \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 2048 -H 2048 -t 4 --diffusion-fa \
    --output $RUN/out.png \
    > $RUN/run.log 2>&1"
```

Observed result:

- `sampling completed = 86.57s`
- `generating 1 latent images completed = 86.63s`
- `GGML_OP_PROFILE total = 83.453s`
- `offload = 69.462s`
- `cpu_fallback = 13.991s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=83 flash_attn=25 other=60`
- `flash_seen=100 flash_mask_null=100`
- log:
  `$RESULTS_ROOT/klein_2k_retry_mapmib_20260412/v79_wf8_2k_m384_a6/run.log`
