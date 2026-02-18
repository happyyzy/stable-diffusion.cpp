cmake_minimum_required(VERSION 3.13)
project(sd_qcom_ml_vae_bridge VERSION 0.1 LANGUAGES C CXX)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE is required (Android NDK toolchain).")
endif()

if (NOT DEFINED ANDROID_ABI)
    message(FATAL_ERROR "ANDROID_ABI is required (arm64-v8a or armeabi-v7a).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CLML_SDK_ROOT "" CACHE PATH "Path to CLML SDK root (contains src/, lib/, lib64/)")
if (NOT CLML_SDK_ROOT)
    message(FATAL_ERROR "CLML_SDK_ROOT is not set.")
endif()

set(CLML_SDK_SRC "${CLML_SDK_ROOT}/src")
if (NOT EXISTS "${CLML_SDK_SRC}/nn_framework/clml_base_nn.cpp")
    message(FATAL_ERROR "Invalid CLML_SDK_ROOT: ${CLML_SDK_ROOT}")
endif()

if (NOT DEFINED LIB_OPENCL)
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        set(LIB_OPENCL "${CLML_SDK_ROOT}/lib64/libOpenCL.so")
    else()
        set(LIB_OPENCL "${CLML_SDK_ROOT}/lib/libOpenCL.so")
    endif()
endif()
if (NOT EXISTS "${LIB_OPENCL}")
    message(FATAL_ERROR "LIB_OPENCL not found: ${LIB_OPENCL}")
endif()

add_library(common_utils STATIC
    "${CLML_SDK_SRC}/utils/clml_fp_conv_utils.cpp"
    "${CLML_SDK_SRC}/utils/clml_utils.cpp"
    "${CLML_SDK_SRC}/utils/clml_train_utils.cpp"
)

add_library(nn_framework STATIC
    "${CLML_SDK_SRC}/nn_framework/clml_utils_nn.cpp"
    "${CLML_SDK_SRC}/nn_framework/clml_tensor_nn.cpp"
    "${CLML_SDK_SRC}/nn_framework/clml_log_nn.cpp"
    "${CLML_SDK_SRC}/nn_framework/clml_base_nn.cpp"
)
target_compile_options(nn_framework PRIVATE -Wno-unused-but-set-variable)

target_include_directories(common_utils PUBLIC "${CLML_SDK_ROOT}" "${CLML_SDK_SRC}")
target_include_directories(nn_framework PUBLIC "${CLML_SDK_ROOT}" "${CLML_SDK_SRC}")

add_library(sd_qcom_ml_vae SHARED
    clml_decoder.cpp
    sd_qcom_ml_vae_bridge.cpp
)
target_include_directories(sd_qcom_ml_vae PRIVATE . "${CLML_SDK_ROOT}" "${CLML_SDK_SRC}")
target_link_options(sd_qcom_ml_vae PRIVATE -Wl,--unresolved-symbols=ignore-in-shared-libs)
target_link_libraries(sd_qcom_ml_vae PRIVATE "${LIB_OPENCL}" common_utils nn_framework)

set_target_properties(sd_qcom_ml_vae PROPERTIES OUTPUT_NAME "sd_qcom_ml_vae")
