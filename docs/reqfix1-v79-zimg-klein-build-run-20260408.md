# reqfix1-v79-zimg-klein-build-run-20260408

Date: 2026-04-08

Goal:

- record the exact current dirty-source closure that fixes `v79` `z-image`
  `rms_norm + flash_attn`
- record one complete fresh-build command set for both `v75` and `v79`
- record the exact run commands that revalidated:
  - `z-image v75`
  - `z-image v79`
  - `flux2 klein 4b v75`
  - `flux2 klein 4b v79`

Current source closure:

- `sdcpp`: `822db48`
- `ggml`: `bba4b4fa`
- `htp_ops`: `d31589f`
- current closure is dirty, and this is intentional
- dirty files in this closure:
  - `sdcpp/ggml/src/ggml-cpu/ggml-cpu.c`
  - `sdcpp/ggml/src/ggml-htp/htp-ops.cc`
  - `sdcpp/ggml/src/ggml-htp/op_reg.h`
  - `htp_ops/include/dsp/hvx_math.h`
  - `htp_ops/include/op_reg.h`
  - `htp_ops/src/dsp/op_executor.cc`
  - `htp_ops/src/dsp/ops/flash_attn.c`
  - `htp_ops/src/dsp/ops/rms_norm.c`
  - `htp_ops/src/dsp/ops/dit_qknorm_rope.c`

Tag anchor under this closure:

- `reqfix1-zimg-klein-mainfix-20260407`

Host paths used in the verified rerun:

- `ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main`
- `VAL=/media/happyyzy/Elements SE/ggufhtp_offload_20260407/tag_rebuild_reqfix1_zimg_klein_mainfix_20260407_rel`
- `SDK=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/Hexagon_SDK/6.5.0.0`
- `TOOLS=$SDK/tools/HEXAGON_Tools/19.0.07`
- `NDK=/home/happyyzy/Android/Sdk/ndk/25.2.9519653`
- `SER=8fd321f3`

## Configure

```sh
set -euo pipefail

ROOT=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main
VAL='/media/happyyzy/Elements SE/ggufhtp_offload_20260407/tag_rebuild_reqfix1_zimg_klein_mainfix_20260407_rel'
SDK=/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/Hexagon_SDK/6.5.0.0
TOOLS=$SDK/tools/HEXAGON_Tools/19.0.07
NDK=/home/happyyzy/Android/Sdk/ndk/25.2.9519653

mkdir -p "$VAL/build"

cmake -S $ROOT/htp_ops -B "$VAL/build/htp_android" \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/android_toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NDK=$NDK \
  -DANDROID_NATIVE_API_LEVEL=26 \
  -DANDROID_STL=none \
  -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY="$VAL/build/htp_android/ship" \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DOS_TYPE=HLOS \
  -DDSP_TYPE=3 \
  -DPREBUILT_LIB_DIR=android_aarch64 \
  -DV=android_ReleaseG_aarch64 \
  > "$VAL/cmake_htp_android.log" 2>&1

cmake -S $ROOT/htp_ops -B "$VAL/build/htp_hex_v75" \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/hexagon_toolchain.cmake \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DHEXAGON_TOOLS_ROOT=$TOOLS \
  -DDSP_VERSION=v75 \
  -DPREBUILT_LIB_DIR=hexagon_toolv19_v75 \
  -DV=hexagon_ReleaseG_toolv19_v75 \
  > "$VAL/cmake_htp_hex_v75.log" 2>&1

cmake -S $ROOT/htp_ops -B "$VAL/build/htp_hex_v79" \
  -DCMAKE_TOOLCHAIN_FILE=$SDK/build/cmake/hexagon_toolchain.cmake \
  -DHEXAGON_SDK_ROOT=$SDK \
  -DHEXAGON_TOOLS_ROOT=$TOOLS \
  -DDSP_VERSION=v79 \
  -DPREBUILT_LIB_DIR=hexagon_toolv19_v79 \
  -DV=hexagon_ReleaseG_toolv19_v79 \
  > "$VAL/cmake_htp_hex_v79.log" 2>&1

cmake -S $ROOT/sdcpp -B "$VAL/build/sdcpp_android" \
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
  > "$VAL/cmake_sdcpp_android.log" 2>&1
```

## Build

```sh
set -euo pipefail

VAL='/media/happyyzy/Elements SE/ggufhtp_offload_20260407/tag_rebuild_reqfix1_zimg_klein_mainfix_20260407_rel'

cmake --build "$VAL/build/htp_android" -j$(nproc)
cmake --build "$VAL/build/htp_hex_v75" -j$(nproc)
cmake --build "$VAL/build/htp_hex_v79" -j$(nproc)
cmake --build "$VAL/build/sdcpp_android" -j$(nproc)
```

Built runtime artifacts used in the verified rerun:

- `sd-cli`:
  `c46f4eb90f414011500c9a633b49fab2b1fd49eb`
- `libhtp_ops.so`:
  `835e1ac82377ce1b502f746ed5a03a25be9453f9`
- `libhtp_ops_skel.so` `v75`:
  `4bad2a0360c14cadaba971b8cd5e32eccb2b63b6`
- `libhtp_ops_skel.so` `v79`:
  `bc0bfdcb10f8fe5d8fc2b4e5dab4b0563435022b`
- `libggml-htp-v79.so`:
  `993213ee1150bbfff60f7f2f84fbcd63e1f3fb83`
- `libomp.so`:
  `fc95f5c5317678f222768966ead63ea2154c19a2`

Note:

- `libggml-htp-v79.so` is the same for both runs
- the actual `v75/v79` runtime switch is the `libhtp_ops_skel.so` pushed into
  each per-arch remote directory

## Phone Assets

`z-image`:

- model:
  `/data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf`
- vae:
  `/data/local/tmp/s07_step31/ae.safetensors`
- cond:
  `/data/local/tmp/s07_step31/zimg_cond.tensor`

`flux2 klein 4b`:

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
VAL='/media/happyyzy/Elements SE/ggufhtp_offload_20260407/tag_rebuild_reqfix1_zimg_klein_mainfix_20260407_rel'
BASE=/data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407
LIBOMP=/home/happyyzy/Android/Sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7/lib/linux/aarch64/libomp.so

adb -s $SER shell "rm -rf $BASE && mkdir -p $BASE/v75 $BASE/v79"

for arch in v75 v79; do
  adb -s $SER push "$VAL/build/sdcpp_android/bin/sd-cli" $BASE/$arch/sd-cli
  adb -s $SER push "$VAL/build/htp_android/libhtp_ops.so" $BASE/$arch/libhtp_ops.so
  adb -s $SER push "$VAL/build/htp_hex_$arch/libhtp_ops_skel.so" $BASE/$arch/libhtp_ops_skel.so
  adb -s $SER push "$VAL/build/sdcpp_android/ggml/src/ggml-hexagon/libggml-htp-v79.so" $BASE/$arch/libggml-htp-v79.so
  adb -s $SER push "$LIBOMP" $BASE/$arch/libomp.so
  adb -s $SER shell "chmod 755 $BASE/$arch/sd-cli $BASE/$arch/libhtp_ops.so $BASE/$arch/libhtp_ops_skel.so $BASE/$arch/libggml-htp-v79.so $BASE/$arch/libomp.so"
done
```

## `z-image v75`

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=/data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407/v75

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
    --diffusion-model /data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf \
    --vae /data/local/tmp/s07_step31/ae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/zimg_cond.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/zimg.out.png \
    > $REMOTE/zimg.run.log 2>&1"
```

Expected hard result:

- latent sha1:
  `3d7e244f8090c63a98703751be6be7bec83270aa`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`

Verified rerun:

- `sampling completed = 16.52s`
- `GGML_OP_PROFILE total = 15.968s`
- log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/build_verify_zimg_v75_v79_after_v79mods_20260407/v75/run.log`

## `z-image v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=/data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407/v79

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
    --diffusion-model /data/local/tmp/s07_step31/fix1_w2q8_exportfix_20260319.gguf \
    --vae /data/local/tmp/s07_step31/ae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/zimg_cond.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/zimg.out.png \
    > $REMOTE/zimg.run.log 2>&1"
```

Expected hard result:

- latent sha1:
  `8f7861083018d53444a9ca31791f5bf90034f268`
- `GGML_HTP_STATS: offloaded_ops=305 mul_mat=203 flash_attn=34 other=68`
- `flash_seen=136 flash_mask_null=136`

Expected compare-to-`v75` result:

- `MAE = 0.0025281315289857886`
- `RMSE = 0.0050870283055255445`
- `max_abs = 0.11392068862915039`
- `cos = 0.9999884100480394`

Verified rerun:

- `sampling completed = 17.57s`
- `GGML_OP_PROFILE total = 17.058s`
- log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/build_verify_zimg_v75_v79_after_v79mods_20260407/v79/run.log`

## `flux2 klein 4b v75`

Run:

```sh
set -euo pipefail

SER=8fd321f3
REMOTE=/data/local/tmp/build_verify_klein4b_v75_after_v79mods_20260407

adb -s $SER shell "rm -rf $REMOTE && mkdir -p $REMOTE"
adb -s $SER push /data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407/v75/sd-cli $REMOTE/sd-cli
adb -s $SER push /data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407/v75/libhtp_ops.so $REMOTE/libhtp_ops.so
adb -s $SER push /data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407/v75/libhtp_ops_skel.so $REMOTE/libhtp_ops_skel.so
adb -s $SER push /data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407/v75/libggml-htp-v79.so $REMOTE/libggml-htp-v79.so
adb -s $SER push /data/local/tmp/build_verify_zimg_v75_v79_after_v79mods_20260407/v75/libomp.so $REMOTE/libomp.so
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"

adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/klein4b.*"

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

Expected hard result:

- latent sha1:
  `4d23130976c84a4d8666897aaaf448e02b6247d7`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- `flash_seen=100 flash_mask_null=100`
- critical gate:
  `GGML_HTP_ENABLE_F16_MATMUL=0`

Verified rerun:

- `sampling completed = 11.60s`
- `GGML_OP_PROFILE total = 10.783s`
- log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/build_verify_klein4b_v75_after_v79mods_20260407/run.log`

## `flux2 klein 4b v79`

Run:

```sh
set -euo pipefail

SER=8fd321f3
VAL='/media/happyyzy/Elements SE/ggufhtp_offload_20260407/tag_rebuild_reqfix1_zimg_klein_mainfix_20260407_rel'
REMOTE=/data/local/tmp/confirm_klein_v79_full_20260408

adb -s $SER shell "rm -rf $REMOTE && mkdir -p $REMOTE"
adb -s $SER push "$VAL/build/sdcpp_android/bin/sd-cli" $REMOTE/sd-cli
adb -s $SER push "$VAL/build/htp_android/libhtp_ops.so" $REMOTE/libhtp_ops.so
adb -s $SER push "$VAL/build/htp_hex_v79/libhtp_ops_skel.so" $REMOTE/libhtp_ops_skel.so
adb -s $SER push "$VAL/build/sdcpp_android/ggml/src/ggml-hexagon/libggml-htp-v79.so" $REMOTE/libggml-htp-v79.so
adb -s $SER push /home/happyyzy/Android/Sdk/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/14.0.7/lib/linux/aarch64/libomp.so $REMOTE/libomp.so
adb -s $SER shell "chmod 755 $REMOTE/sd-cli $REMOTE/libhtp_ops.so $REMOTE/libhtp_ops_skel.so $REMOTE/libggml-htp-v79.so $REMOTE/libomp.so"

adb -s $SER shell "pkill -9 sd-cli || true; rm -f $REMOTE/*"

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
    --diffusion-model /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux-2-klein-4b-q8_0-hmx-q8only.gguf \
    --vae /data/local/tmp/s07_step31/flux2_klein_q80_20260324/flux2-vae.safetensors \
    --cond-crossattn /data/local/tmp/s07_step31/flux2_klein_q80_20260324/cond256_a_lovely_cat.tensor \
    --cfg-scale 1.0 --steps 1 --seed 42 \
    -W 1024 -H 1024 -t 4 --diffusion-fa \
    --output $REMOTE/out.png \
    > $REMOTE/run.log 2>&1"
```

Expected hard result:

- latent sha1:
  `48912698e4ce84836a6433505284e44eb518237c`
- `GGML_HTP_STATS: offloaded_ops=168 mul_mat=63 flash_attn=25 other=80`
- `flash_seen=100 flash_mask_null=100`

Expected compare-to-`v75` result:

- `MAE = 0.004023044801144238`
- `RMSE = 0.006549814428856361`
- `max_abs = 0.12334740161895752`
- `cos = 0.9999707980749146`

Verified rerun:

- `sampling completed = 12.09s`
- `GGML_OP_PROFILE total = 11.283s`
- log:
  `/media/happyyzy/Windows/pyproject_work_20260202/work_sd_cpp/ggufhtp/worktrees/reqfix1_hvx_rope_main/scratch/confirm_klein_v79_full_20260408/run.log`

Acceptance rule for this doc:

- this doc records the current dirty-source `v79` fix closure
- `v75/v79` `z-image` and `v75/v79` `klein4b` all run with `nullmask` flash
- hard pass condition is:
  - the listed latent sha1 values
  - the listed `offloaded_ops/mul_mat/flash_attn/other` counters
  - the listed `v79 vs v75` latent metrics
