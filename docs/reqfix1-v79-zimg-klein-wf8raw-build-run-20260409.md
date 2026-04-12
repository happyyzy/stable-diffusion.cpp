# reqfix1-v79-zimg-klein-wf8raw-build-run-20260409

Date: 2026-04-09

Goal:

- record one exact current-source fresh-build closure for:
  - `z-image v75`
  - `z-image v79`
  - `flux2 klein 4b v75`
  - `flux2 klein 4b v79`
- record one exact current-binary repro of:
  - `/media/happyyzy/Elements SE/current_dirty_zimg_wf8_debug_20260409/cpu_decode_4step/wf8raw_s1.run.log`

Current source closure before commit/tag:

- `sdcpp`: `3ab41d2`
- `ggml`: `1431c412`
- `htp_ops`: `6460b21`
- current closure is dirty, and this is intentional

Important build note:

- this closure must be configured with `-DCMAKE_BUILD_TYPE=Release`
- if `CMAKE_BUILD_TYPE` is left empty for `htp_hex_v75/v79`, generated `flags.make`
  will drop `-mv75/-mv79`, and Hexagon compile will fail on HMX codepaths

Host paths used in this rerun:

- `ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main`
- `VAL='/media/happyyzy/Elements SE/current_source_reval_reqfix1_20260409'`
- `SDK=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/Hexagon_SDK/6.5.0.0`
- `TOOLS=$SDK/tools/HEXAGON_Tools/19.0.07`
- `NDK=/home/happyyzy/Android/Sdk/ndk/25.2.9519653`
- `SER=8fd321f3`

## Configure

```sh
set -euo pipefail

ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main
VAL='/media/happyyzy/Elements SE/current_source_reval_reqfix1_20260409'
SDK=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/Hexagon_SDK/6.5.0.0
TOOLS=$SDK/tools/HEXAGON_Tools/19.0.07
NDK=/home/happyyzy/Android/Sdk/ndk/25.2.9519653

mkdir -p "$VAL/build"

cmake -S $ROOT/htp_ops -B "$VAL/build/htp_android_rel" \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/android_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NDK=$NDK \
  -DANDROID_NATIVE_API_LEVEL=26 \
  -DANDROID_STL=none \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$VAL/build/htp_android_rel/ship" \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DOS_TYPE=HLOS \
  -DDSP_TYPE=3 \
  -DPREBUILT_LIB_DIR=android_aarch64 \
  -DV=android_ReleaseG_aarch64 \
  > "$VAL/cmake_htp_android_rel.log" 2>&1

cmake -S $ROOT/htp_ops -B "$VAL/build/htp_hex_v75_rel" \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/hexagon_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DHEXAGON_TOOLS_ROOT=$TOOLS \
  -DDSP_VERSION=v75 \
  -DPREBUILT_LIB_DIR=hexagon_toolv19_v75 \
  -DV=hexagon_ReleaseG_toolv19_v75 \
  > "$VAL/cmake_htp_hex_v75_rel.log" 2>&1

cmake -S $ROOT/htp_ops -B "$VAL/build/htp_hex_v79_rel" \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/hexagon_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DHEXAGON_TOOLS_ROOT=$TOOLS \
  -DDSP_VERSION=v79 \
  -DPREBUILT_LIB_DIR=hexagon_toolv19_v79 \
  -DV=hexagon_ReleaseG_toolv19_v79 \
  > "$VAL/cmake_htp_hex_v79_rel.log" 2>&1

cmake -S $ROOT/sdcpp -B "$VAL/build/sdcpp_android_rel" \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DSD_HEXAGON=ON \
  -DSD_BUILD_EXAMPLES=ON \
  -DSD_BUILD_SHARED_LIBS=OFF \
  -DGGML_OPENMP=ON \
  -DGGML_HTP=ON \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DHEXAGON_TOOLS_ROOT=$TOOLS \
  -DPREBUILT_LIB_DIR=android_aarch64 \
  -DV=android_ReleaseG_aarch64 \
  > "$VAL/cmake_sdcpp_android_rel.log" 2>&1
```

## Build

```sh
set -euo pipefail

VAL='/media/happyyzy/Elements SE/current_source_reval_reqfix1_20260409'

cmake --build "$VAL/build/htp_android_rel" -j$(nproc)
cmake --build "$VAL/build/htp_hex_v75_rel" -j$(nproc)
cmake --build "$VAL/build/htp_hex_v79_rel" -j$(nproc)
cmake --build "$VAL/build/sdcpp_android_rel" -j$(nproc)
```

Built runtime artifacts used below:

- `sd-cli`:
  `8cc3b6a056ac4e448c07d15e31d9700d3e868fcd`
- `libhtp_ops.so`:
  `757251bad41f0724b5ffc9cc23ec744d85ee51f3`
- `libhtp_ops_skel.so` `v75`:
  `c4a9dcce488143b19199132d0f13e9d565136ee9`
- `libhtp_ops_skel.so` `v79`:
  `0d71830116a058cf6f3e979f4eb1c8f1ebe62ed2`
- `libggml-htp-v79.so`:
  `17a989dcfeda60068e916e8ca0257520e9829629`

## Push Runtime To Phone

```sh
set -euo pipefail

SER=8fd321f3
VAL='/media/happyyzy/Elements SE/current_source_reval_reqfix1_20260409'
BASE=/data/local/tmp/current_source_reval_reqfix1_20260409_rel
LIBOMP=/home/happyyzy/Android/Sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7/lib/linux/aarch64/libomp.so

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

## `z-image v75`

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=/data/local/tmp/current_source_reval_reqfix1_20260409_rel/v75

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
  GGML_HTP_ZIMG_QKNORM_ROPE=1 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_MAX_MAP_MIB=3072 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_ACTIVE_MAPS=16 \
  GGML_HTP_SKIP_CORE_PERMUTE_REPACK=0 \
  GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS=0 \
  GGML_HTP_RUNTIME_REPACK_QWEIGHTS=0 \
  GGML_HTP_MATMUL_SPLIT_M=0 \
  GGML_HTP_MATMUL_SPLIT_M_QKV=0 \
  GGML_HTP_MATMUL_SPLIT_M_OUT=0 \
  GGML_HTP_MATMUL_SPLIT_M_W2=0 \
  GGML_HTP_MATMUL_SPLIT_M_W13=0 \
  GGML_HTP_MATMUL_SPLIT_M_ADALN=0 \
  SD_DUMP_LATENT='$REMOTE/zimg.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/zimg.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/zimg.shape.csv' \
  ./sd-cli -v \
    --diffusion-model /data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf \
    --vae /data/local/tmp/s07_step31/ae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/zimg_cond.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/zimg.out.png \
    > $REMOTE/zimg.run.log 2>&1"
```

Actual result:

- latent sha1:
  `3d7e244f8090c63a98703751be6be7bec83270aa`
- `sampling completed = 16.15s`
- `GGML_OP_PROFILE total = 15.673s`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`
- log:
  `worktrees/reqfix1_hvx_rope_main/scratch/current_source_reval_reqfix1_20260409_rel/zimg_v75/run.log`

## `z-image v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=/data/local/tmp/current_source_reval_reqfix1_20260409_rel/v79

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
  GGML_HTP_ZIMG_QKNORM_ROPE=1 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_MAX_MAP_MIB=3072 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_ACTIVE_MAPS=16 \
  GGML_HTP_SKIP_CORE_PERMUTE_REPACK=0 \
  GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS=0 \
  GGML_HTP_RUNTIME_REPACK_QWEIGHTS=0 \
  GGML_HTP_MATMUL_SPLIT_M=0 \
  GGML_HTP_MATMUL_SPLIT_M_QKV=0 \
  GGML_HTP_MATMUL_SPLIT_M_OUT=0 \
  GGML_HTP_MATMUL_SPLIT_M_W2=0 \
  GGML_HTP_MATMUL_SPLIT_M_W13=0 \
  GGML_HTP_MATMUL_SPLIT_M_ADALN=0 \
  SD_DUMP_LATENT='$REMOTE/zimg.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/zimg.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/zimg.shape.csv' \
  ./sd-cli -v \
    --diffusion-model /data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf \
    --vae /data/local/tmp/s07_step31/ae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/zimg_cond.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/zimg.out.png \
    > $REMOTE/zimg.run.log 2>&1"
```

Actual result:

- latent sha1:
  `8f7861083018d53444a9ca31791f5bf90034f268`
- `sampling completed = 17.10s`
- `GGML_OP_PROFILE total = 16.435s`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`
- log:
  `worktrees/reqfix1_hvx_rope_main/scratch/current_source_reval_reqfix1_20260409_rel/zimg_v79/run.log`

Compare-to-`v75` result:

- `MAE = 0.0025281315289857886`
- `RMSE = 0.0050870283055255445`
- `max_abs = 0.11392068862915039`
- `cos = 0.9999884100480394`

## `flux2 klein 4b v75`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=/data/local/tmp/current_source_reval_reqfix1_20260409_rel
SRC=$BASE/v75
REMOTE=$BASE/klein_v75

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
  GGML_HTP_FLUX_QKNORM_ROPE=1 \
  GGML_HTP_FLUX_QKNORM_ROPE_DIRECT_INPUT=1 \
  GGML_HTP_FLASH_QKNORM_DIRECT=1 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  SD_DUMP_LATENT='$REMOTE/latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/shape.csv' \
  ./sd-cli -v \
    --diffusion-model /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf \
    --vae /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/out.png \
    > $REMOTE/run.log 2>&1"
```

Actual result:

- latent sha1:
  `4d23130976c84a4d8666897aaaf448e02b6247d7`
- `sampling completed = 11.26s`
- `GGML_OP_PROFILE total = 10.502s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- `flash_seen=100 flash_mask_null=100`
- log:
  `worktrees/reqfix1_hvx_rope_main/scratch/current_source_reval_reqfix1_20260409_rel/klein_v75/run.log`

## `flux2 klein 4b v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
BASE=/data/local/tmp/current_source_reval_reqfix1_20260409_rel
SRC=$BASE/v79
REMOTE=$BASE/klein_v79

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
  GGML_HTP_FLUX_QKNORM_ROPE=1 \
  GGML_HTP_FLUX_QKNORM_ROPE_DIRECT_INPUT=1 \
  GGML_HTP_FLASH_QKNORM_DIRECT=1 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_FLASH_PREP_IN_KERNEL=0 \
  SD_DUMP_LATENT='$REMOTE/latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/shape.csv' \
  ./sd-cli -v \
    --diffusion-model /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf \
    --vae /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/out.png \
    > $REMOTE/run.log 2>&1"
```

Actual result:

- latent sha1:
  `48912698e4ce84836a6433505284e44eb518237c`
- `sampling completed = 11.73s`
- `GGML_OP_PROFILE total = 10.944s`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- `flash_seen=100 flash_mask_null=100`
- log:
  `worktrees/reqfix1_hvx_rope_main/scratch/current_source_reval_reqfix1_20260409_rel/klein_v79/run.log`

Compare-to-`v75` result:

- `MAE = 0.004023044801144238`
- `RMSE = 0.006549814428856361`
- `max_abs = 0.12334740161895752`
- `cos = 0.9999707980749146`

## `z-image wf8raw_s1` current-binary repro

Reference:

- reference log:
  `/media/happyyzy/Elements SE/current_dirty_zimg_wf8_debug_20260409/cpu_decode_4step/wf8raw_s1.run.log`
- reference latent used for exact compare:
  `/data/local/tmp/current_dirty_zimg_wf8_debug_20260409/v79_run1_retest/wf8raw_s1.latent.tensor`
- reference latent sha1:
  `ce599346a14f9e42e2f559fe0436fe9df7606cee`

Run with current fresh-built binary:

```sh
set -euo pipefail

SER=8fd321f3
BASE=/data/local/tmp/current_source_reval_reqfix1_20260409_rel
SRC=$BASE/v79
REMOTE=$BASE/wf8raw_repro_current

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
  GGML_HTP_ZIMG_QKNORM_ROPE=1 \
  GGML_HTP_Q8_OUTSTATIONARY=1 \
  GGML_HTP_MAX_MAP_MIB=3072 \
  GGML_HTP_DEFER_UNMAP=1 \
  GGML_HTP_MAX_ACTIVE_MAPS=16 \
  GGML_HTP_SKIP_CORE_PERMUTE_REPACK=0 \
  GGML_HTP_RUNTIME_PERMUTE_QWEIGHTS=0 \
  GGML_HTP_RUNTIME_REPACK_QWEIGHTS=0 \
  GGML_HTP_MATMUL_SPLIT_M=0 \
  GGML_HTP_MATMUL_SPLIT_M_QKV=0 \
  GGML_HTP_MATMUL_SPLIT_M_OUT=0 \
  GGML_HTP_MATMUL_SPLIT_M_W2=0 \
  GGML_HTP_MATMUL_SPLIT_M_W13=0 \
  GGML_HTP_MATMUL_SPLIT_M_ADALN=0 \
  SD_DUMP_LATENT='$REMOTE/wf8raw_s1_current.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/wf8raw_s1_current.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/wf8raw_s1_current.shape.csv' \
  ./sd-cli -v \
    --diffusion-model /data/local/tmp/current_dirty_zimg_wf8_debug_20260409/v79/z_image_turbo_f16base3_wf8hmx_20260409.gguf \
    --vae /data/local/tmp/s07_step31/ae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/zimg_cond.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/wf8raw_s1_current.out.png \
    > $REMOTE/wf8raw_s1_current.run.log 2>&1"
```

Current result:

- current latent sha1:
  `ce599346a14f9e42e2f559fe0436fe9df7606cee`
- compare against reference latent:
  - `mae = 0`
  - `rmse = 0`
  - `max_abs = 0`
  - `cos = 1`
- `z_image runtime override summary: 0 tensors forced to f16`
- `Weight type stat: f32: 244 | f16: 35 | bf16: 248 | wf8_hmx: 170`
- `z_image compute buffer size = 592.79 MB`
- `sampling completed = 15.50s`
- `generating 1 latent images completed = 15.52s`
- `GGML_HTP_STATS: offloaded_ops=272 mul_mat=170 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`
- `GGML_OP_PROFILE total = 14.987s`
- `MUL_MAT,208 = 6313.342 ms`
- `FLASH_ATTN_EXT,34 = 3272.258 ms`
- `MAP_CUSTOM3,68 = 1186.063 ms`
- `matmul-f16-disabled count = 140`
- current log:
  `worktrees/reqfix1_hvx_rope_main/scratch/current_source_reval_reqfix1_20260409_rel/wf8raw_repro/wf8raw_s1_current.run.log`

Acceptance rule for this doc:

- the 4 baseline doc runs must continue to reproduce the documented latent sha1 values and `v79 vs v75` metrics
- the current fresh-built binary must exactly reproduce `wf8raw_s1` latent sha1 `ce599346a14f9e42e2f559fe0436fe9df7606cee`
- hard pass condition is:
  - the listed latent sha1 values
  - the listed `offloaded_ops/mul_mat/flash_attn/other` counters
  - the listed `flash_seen/flash_mask_null` counters
  - exact `wf8raw_s1` latent match (`mae=0`, `rmse=0`, `max_abs=0`, `cos=1`)
