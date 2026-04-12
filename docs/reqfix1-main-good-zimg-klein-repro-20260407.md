# reqfix1-main-good-zimg-klein-repro-20260407

Date: 2026-04-07

Goal:

- record the exact build and run commands that reproduce both `z-image` and
  `flux2 klein 4b` on the current good main worktree
- pin the current good source closure that passes both latents on device

Current good source closure:

- `sdcpp`: `6e5ec0b`
- `ggml`: `bba4b4fa`
- `htp_ops`: `d31589f`
- common tag anchor:
  `reqfix1-q80-outstat-flashprep-directinput-cold1460-20260323`
- describe:
  - `sdcpp`:
    `reqfix1-q80-outstat-flashprep-directinput-cold1460-20260323-3-g6e5ec0b-dirty`
  - `ggml`:
    `reqfix1-q80-outstat-flashprep-directinput-cold1460-20260323-3-gbba4b4fa`
  - `htp_ops`:
    `reqfix1-q80-outstat-flashprep-directinput-cold1460-20260323-3-gd31589f`

Host paths used in the verified rerun:

- `ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main`
- `VAL=$ROOT/scratch/main_good_validate_20260407`
- `SDK=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/Hexagon_SDK/6.5.0.0`
- `TOOLS=$SDK/tools/HEXAGON_Tools/19.0.07`
- `NDK=/home/happyyzy/Android/Sdk/ndk/25.2.9519653`
- `SER=8fd321f3`
- `REMOTE=/data/local/tmp/main_good_validate_20260407`

## Build

Configure:

```sh
set -euo pipefail

ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main
VAL=$ROOT/scratch/main_good_validate_20260407
SDK=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/Hexagon_SDK/6.5.0.0
TOOLS=$SDK/tools/HEXAGON_Tools/19.0.07
NDK=/home/happyyzy/Android/Sdk/ndk/25.2.9519653

mkdir -p $VAL/build

cmake -S $ROOT/htp_ops -B $VAL/build/htp_android \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/android_toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NDK=$NDK \
  -DANDROID_NATIVE_API_LEVEL=26 \
  -DANDROID_STL=none \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=$VAL/build/htp_android/ship \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DOS_TYPE=HLOS \
  -DDSP_TYPE=3 \
  -DPREBUILT_LIB_DIR=android_aarch64 \
  -DV=android_ReleaseG_aarch64 \
  > $VAL/cmake_htp_android.log 2>&1

cmake -S $ROOT/htp_ops -B $VAL/build/htp_hex_v75 \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/hexagon_toolchain.cmake \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DHEXAGON_TOOLS_ROOT=$TOOLS \
  -DDSP_VERSION=v75 \
  -DPREBUILT_LIB_DIR=hexagon_toolv19_v75 \
  -DV=hexagon_ReleaseG_toolv19_v75 \
  > $VAL/cmake_htp_hex_v75.log 2>&1

cmake -S $ROOT/sdcpp -B $VAL/build/sdcpp_android \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
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
  > $VAL/cmake_sdcpp_android.log 2>&1
```

Build:

```sh
set -euo pipefail

ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main
VAL=$ROOT/scratch/main_good_validate_20260407

cmake --build $VAL/build/htp_android -j$(nproc)
cmake --build $VAL/build/htp_hex_v75 -j$(nproc)
cmake --build $VAL/build/sdcpp_android -j$(nproc)
```

Built runtime artifacts used in the verified rerun:

- `sd-cli`:
  `$VAL/build/sdcpp_android/bin/sd-cli`
  `3532ae36b80c6710ce60ad845be1c818ffe0fa67`
- `libggml-htp-v79.so`:
  `$VAL/build/sdcpp_android/ggml/src/ggml-hexagon/libggml-htp-v79.so`
  `993213ee1150bbfff60f7f2f84fbcd63e1f3fb83`
- `libhtp_ops.so`:
  `$VAL/build/htp_android/libhtp_ops.so`
  `cee4dba075c64d3ae5aeecde43ffe6ded91d20a6`
- `libhtp_ops_skel.so`:
  `$VAL/build/htp_hex_v75/libhtp_ops_skel.so`
  `7ed6966469715729274e96ba37f41f599831de46`
- `libomp.so`:
  `$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7/lib/linux/aarch64/libomp.so`
  `fc95f5c5317678f222768966ead63ea2154c19a2`

## Phone Assets

The verified rerun used these existing phone-side weights:

### z-image

- model:
  `/data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf`
- vae:
  `/data/local/tmp/s07_step31/ae.safetensors`
- cond:
  `/data/local/tmp/s07_step31/zimg_cond.tensor`

### flux2 klein 4b

- model:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf`
- vae:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors`
- cond:
  `/data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor`

## Push Runtime To Phone

```sh
set -euo pipefail

SER=8fd321f3
ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main
VAL=$ROOT/scratch/main_good_validate_20260407
REMOTE=/data/local/tmp/main_good_validate_20260407
NDK=/home/happyyzy/Android/Sdk/ndk/25.2.9519653
LIBOMP=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7/lib/linux/aarch64/libomp.so

adb -s $SER shell "rm -rf $REMOTE && mkdir -p $REMOTE"
adb -s $SER push $VAL/build/sdcpp_android/bin/sd-cli $REMOTE/sd-cli
adb -s $SER push $VAL/build/sdcpp_android/ggml/src/ggml-hexagon/libggml-htp-v79.so $REMOTE/libggml-htp-v79.so
adb -s $SER push $VAL/build/htp_android/libhtp_ops.so $REMOTE/libhtp_ops.so
adb -s $SER push $VAL/build/htp_hex_v75/libhtp_ops_skel.so $REMOTE/libhtp_ops_skel.so
adb -s $SER push $LIBOMP $REMOTE/libomp.so
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libggml-htp-v79.so $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libomp.so"
```

## z-image Repro Command

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=/data/local/tmp/main_good_validate_20260407

adb -s $SER shell 'pkill -9 sd-cli || true; rm -f /data/local/tmp/main_good_validate_20260407/zimg.*'

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

Pull and check:

```sh
set -euo pipefail

SER=8fd321f3
ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main
OUT=$ROOT/scratch/main_good_validate_20260407/runs_main/zimg
REMOTE=/data/local/tmp/main_good_validate_20260407

mkdir -p $OUT
adb -s $SER pull $REMOTE/zimg.run.log $OUT/run.log
adb -s $SER pull $REMOTE/zimg.latent.tensor $OUT/latent.tensor
adb -s $SER pull $REMOTE/zimg.op.csv $OUT/op.csv
adb -s $SER pull $REMOTE/zimg.shape.csv $OUT/shape.csv
sha1sum $OUT/latent.tensor
rg -n 'sampling completed|GGML_HTP_STATS: offloaded_ops|GGML_OP_PROFILE: total_calls' $OUT/run.log
```

Expected hard result:

- latent sha1:
  `3d7e244f8090c63a98703751be6be7bec83270aa`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`

Verified rerun on this exact doc closure:

- `sampling completed = 15.55s`
- `GGML_OP_PROFILE total = 15.077s`
- log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/main_good_validate_20260407/runs_main/zimg/run.log`

## flux2 klein 4b Repro Command

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=/data/local/tmp/main_good_validate_20260407

adb -s $SER shell 'pkill -9 sd-cli || true; rm -f /data/local/tmp/main_good_validate_20260407/klein4b.*'

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
  SD_DUMP_LATENT='$REMOTE/klein4b.latent.tensor' \
  SD_NPU_OP_PROFILE_CSV='$REMOTE/klein4b.op.csv' \
  SD_NPU_OP_PROFILE_SHAPE_CSV='$REMOTE/klein4b.shape.csv' \
  ./sd-cli -v \
    --diffusion-model /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf \
    --vae /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/klein4b.out.png \
    > $REMOTE/klein4b.run.log 2>&1"
```

Pull and check:

```sh
set -euo pipefail

SER=8fd321f3
ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main
OUT=$ROOT/scratch/main_good_validate_20260407/runs_main/klein4b
REMOTE=/data/local/tmp/main_good_validate_20260407

mkdir -p $OUT
adb -s $SER pull $REMOTE/klein4b.run.log $OUT/run.log
adb -s $SER pull $REMOTE/klein4b.latent.tensor $OUT/latent.tensor
adb -s $SER pull $REMOTE/klein4b.op.csv $OUT/op.csv
adb -s $SER pull $REMOTE/klein4b.shape.csv $OUT/shape.csv
sha1sum $OUT/latent.tensor
rg -n 'sampling completed|GGML_HTP_STATS: offloaded_ops|GGML_OP_PROFILE: total_calls' $OUT/run.log
```

Expected hard result:

- latent sha1:
  `4d23130976c84a4d8666897aaaf448e02b6247d7`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- critical gate:
  `GGML_HTP_ENABLE_F16_MATMUL=0`

Verified rerun on this exact doc closure:

- `sampling completed = 11.29s`
- `GGML_OP_PROFILE total = 10.521s`
- log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/main_good_validate_20260407/runs_main/klein4b/run.log`

Acceptance rule for this doc:

- time can drift with thermals
- latent sha1 and offload counters above are the hard pass condition
- if `klein4b` misses the target latent, check `GGML_HTP_ENABLE_F16_MATMUL=0`
  first before changing any source
