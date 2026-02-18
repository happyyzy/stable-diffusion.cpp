// Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//******************************************************************************************************************************

/**
********************************************************************************************************************************
* @file
*     clml_decoder.cpp
* @brief
*     Image Decode Model class implementation in CLML. Implements Decoder Model used in Stable Diffusion 2.1.
*
********************************************************************************************************************************
*/

#include "clml_decoder.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static bool env_enabled(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

static bool env_parse_int(const char* name, int* out) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || *end != '\0') {
        return false;
    }
    *out = static_cast<int>(parsed);
    return true;
}

static cl_arithmetic_mode_qcom resolve_mha_arithmetic_mode(cl_arithmetic_mode_qcom fallback) {
    int mode = -1;
    if (!env_parse_int("SD_QCOM_ML_VAE_MHA_ARITH", &mode)) {
        return fallback;
    }
    switch (mode) {
        case 0: return CL_ARITHMETIC_MODE_FP16_QCOM;
        case 1: return CL_ARITHMETIC_MODE_FP16_ACC32_QCOM;
        case 2: return CL_ARITHMETIC_MODE_FP32_QCOM;
        default: return fallback;
    }
}

static cl_softmax_mode_qcom resolve_mha_softmax_mode(cl_softmax_mode_qcom fallback) {
    int mode = -1;
    if (!env_parse_int("SD_QCOM_ML_VAE_MHA_SOFTMAX", &mode)) {
        return fallback;
    }
    switch (mode) {
        case 0: return CL_SOFTMAX_MODE_INSTANCE_QCOM;
        case 1: return CL_SOFTMAX_MODE_CHANNEL_QCOM;
        case 2: return CL_SOFTMAX_MODE_SPATIAL_QCOM;
        case 3: return CL_SOFTMAX_MODE_WIDTH_QCOM;
        default: return fallback;
    }
}

static cl_multi_head_attn_weights_transform_qcom resolve_mha_weight_transform(
    cl_multi_head_attn_weights_transform_qcom fallback) {
    int mode = -1;
    if (!env_parse_int("SD_QCOM_ML_VAE_MHA_WT", &mode)) {
        return fallback;
    }
    switch (mode) {
        case 0: return CL_MULTI_HEAD_ATTN_WEIGHTS_TRANSFORM_NONE_QCOM;
        case 1: return CL_MULTI_HEAD_ATTN_WEIGHTS_TRANSFORM_TRANSPOSE_QCOM;
        default: return fallback;
    }
}

struct CpuAttentionHostOpData {
    MLTensor q;
    MLTensor k;
    MLTensor v;
    MLTensor out;
    size_t element_count = 0;
    cl_uint batch = 0;
    cl_uint seq = 0;
    cl_uint channels = 0;
    cl_uint num_heads = 0;
    cl_uint head_dim = 0;
    float scale = 1.0f;
    bool debug = false;
    bool use_opencl_backend = false;
    bool profile_opencl_backend = false;
};

struct OpenCLAttentionEngine {
    bool ready = false;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel qk_kernel = nullptr;
    cl_kernel softmax_kernel = nullptr;
    cl_kernel pv_kernel = nullptr;
    cl_mem q_buf = nullptr;
    cl_mem k_buf = nullptr;
    cl_mem v_buf = nullptr;
    cl_mem p_buf = nullptr;
    cl_mem o_buf = nullptr;
    size_t cap_seq = 0;
    size_t cap_dim = 0;
};

static OpenCLAttentionEngine& get_opencl_attention_engine() {
    static OpenCLAttentionEngine engine;
    return engine;
}

static void release_opencl_attention_buffer(cl_mem* mem) {
    if (mem != nullptr && *mem != nullptr) {
        clReleaseMemObject(*mem);
        *mem = nullptr;
    }
}

static bool parse_backend_env(const char* name, std::string* out) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    if (out != nullptr) {
        *out = std::string(v);
        std::transform(out->begin(), out->end(), out->begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    }
    return true;
}

static const char* kOpenClAttentionKernelSrc = R"CLC(
__kernel void sd_qcom_qk_dot(
    __global const float* q,
    __global const float* k,
    __global float* p,
    const int seq,
    const int channels,
    const int head_dim,
    const int head_offset,
    const float scale) {
    const int qi = get_global_id(0);
    const int kj = get_global_id(1);
    if (qi >= seq || kj >= seq) {
        return;
    }

    const int qoff = qi * channels + head_offset;
    const int koff = kj * channels + head_offset;
    float acc = 0.0f;
    int d = 0;
    for (; d + 3 < head_dim; d += 4) {
        const float4 qv = vload4(0, q + qoff + d);
        const float4 kv = vload4(0, k + koff + d);
        acc += dot(qv, kv);
    }
    for (; d < head_dim; ++d) {
        acc += q[qoff + d] * k[koff + d];
    }
    p[qi * seq + kj] = acc * scale;
}

__kernel void sd_qcom_softmax_inplace(
    __global float* p,
    const int seq) {
    const int qi = get_global_id(0);
    if (qi >= seq) {
        return;
    }

    __global float* row = p + qi * seq;
    float maxv = -MAXFLOAT;
    for (int j = 0; j < seq; ++j) {
        maxv = fmax(maxv, row[j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < seq; ++j) {
        const float e = exp(row[j] - maxv);
        row[j] = e;
        sum += e;
    }

    const float inv = (sum > 0.0f && isfinite(sum)) ? (1.0f / sum) : 0.0f;
    for (int j = 0; j < seq; ++j) {
        row[j] *= inv;
    }
}

__kernel void sd_qcom_pv_out(
    __global const float* p,
    __global const float* v,
    __global float* o,
    const int seq,
    const int channels,
    const int head_dim,
    const int head_offset) {
    const int qi = get_global_id(0);
    const int d = get_global_id(1);
    if (qi >= seq || d >= head_dim) {
        return;
    }

    float acc = 0.0f;
    const int out_off = qi * channels + head_offset + d;
    __global const float* prow = p + qi * seq;
    for (int j = 0; j < seq; ++j) {
        const float w = prow[j];
        const float vv = v[j * channels + head_offset + d];
        acc = mad(w, vv, acc);
    }
    o[out_off] = acc;
}
)CLC";

static bool opencl_attention_init(OpenCLAttentionEngine* engine) {
    if (engine == nullptr) {
        return false;
    }
    if (engine->ready) {
        return true;
    }

    cl_int err = CL_SUCCESS;
    cl_uint n_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) {
        return false;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    err = clGetPlatformIDs(n_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        return false;
    }

    cl_platform_id chosen_platform = nullptr;
    cl_device_id chosen_device = nullptr;
    for (cl_platform_id p : platforms) {
        cl_uint n_devices = 0;
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices);
        if (err != CL_SUCCESS || n_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(n_devices);
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, n_devices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            continue;
        }
        chosen_platform = p;
        chosen_device = devices[0];
        break;
    }
    if (chosen_platform == nullptr || chosen_device == nullptr) {
        return false;
    }

    cl_context context = clCreateContext(nullptr, 1, &chosen_device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || context == nullptr) {
        return false;
    }

#if defined(CL_VERSION_2_0)
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, chosen_device, nullptr, &err);
#else
    cl_command_queue queue = clCreateCommandQueue(context, chosen_device, 0, &err);
#endif
    if (err != CL_SUCCESS || queue == nullptr) {
        clReleaseContext(context);
        return false;
    }

    const char* src = kOpenClAttentionKernelSrc;
    size_t src_len = std::strlen(src);
    cl_program program = clCreateProgramWithSource(context, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS || program == nullptr) {
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    err = clBuildProgram(program, 1, &chosen_device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        if (log_size > 0) {
            clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        }
        std::fprintf(stderr, "[QCOM_ML_VAE][GGML_ATTN] clBuildProgram failed: %s\n", log.data());
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    cl_kernel qk_kernel = clCreateKernel(program, "sd_qcom_qk_dot", &err);
    if (err != CL_SUCCESS || qk_kernel == nullptr) {
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    cl_kernel softmax_kernel = clCreateKernel(program, "sd_qcom_softmax_inplace", &err);
    if (err != CL_SUCCESS || softmax_kernel == nullptr) {
        clReleaseKernel(qk_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }
    cl_kernel pv_kernel = clCreateKernel(program, "sd_qcom_pv_out", &err);
    if (err != CL_SUCCESS || pv_kernel == nullptr) {
        clReleaseKernel(softmax_kernel);
        clReleaseKernel(qk_kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return false;
    }

    engine->platform = chosen_platform;
    engine->device = chosen_device;
    engine->context = context;
    engine->queue = queue;
    engine->program = program;
    engine->qk_kernel = qk_kernel;
    engine->softmax_kernel = softmax_kernel;
    engine->pv_kernel = pv_kernel;
    engine->ready = true;
    return true;
}

static bool opencl_attention_ensure_buffers(OpenCLAttentionEngine* engine, size_t seq, size_t dim) {
    if (engine == nullptr || !engine->ready || seq == 0 || dim == 0) {
        return false;
    }
    if (seq <= engine->cap_seq && dim <= engine->cap_dim &&
        engine->q_buf != nullptr && engine->k_buf != nullptr && engine->v_buf != nullptr &&
        engine->p_buf != nullptr && engine->o_buf != nullptr) {
        return true;
    }

    release_opencl_attention_buffer(&engine->q_buf);
    release_opencl_attention_buffer(&engine->k_buf);
    release_opencl_attention_buffer(&engine->v_buf);
    release_opencl_attention_buffer(&engine->p_buf);
    release_opencl_attention_buffer(&engine->o_buf);

    cl_int err = CL_SUCCESS;
    const size_t elem_qkv = seq * dim;
    const size_t elem_scores = seq * seq;
    engine->q_buf = clCreateBuffer(engine->context, CL_MEM_READ_WRITE, elem_qkv * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || engine->q_buf == nullptr) {
        return false;
    }
    engine->k_buf = clCreateBuffer(engine->context, CL_MEM_READ_WRITE, elem_qkv * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || engine->k_buf == nullptr) {
        return false;
    }
    engine->v_buf = clCreateBuffer(engine->context, CL_MEM_READ_WRITE, elem_qkv * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || engine->v_buf == nullptr) {
        return false;
    }
    engine->p_buf = clCreateBuffer(engine->context, CL_MEM_READ_WRITE, elem_scores * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || engine->p_buf == nullptr) {
        return false;
    }
    engine->o_buf = clCreateBuffer(engine->context, CL_MEM_READ_WRITE, elem_qkv * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS || engine->o_buf == nullptr) {
        return false;
    }

    engine->cap_seq = seq;
    engine->cap_dim = dim;
    return true;
}

static size_t roundup(size_t x, size_t m) {
    return ((x + m - 1) / m) * m;
}

static bool run_opencl_attention_backend(
    CpuAttentionHostOpData* data,
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    std::vector<float>* out) {
    if (data == nullptr || out == nullptr || data->seq == 0 || data->channels == 0 || data->num_heads == 0) {
        return false;
    }

    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto t0 = std::chrono::steady_clock::now();

    OpenCLAttentionEngine& engine = get_opencl_attention_engine();
    if (!opencl_attention_init(&engine)) {
        return false;
    }
    if (!opencl_attention_ensure_buffers(&engine, data->seq, data->channels)) {
        return false;
    }

    const size_t elem_qkv = static_cast<size_t>(data->seq) * data->channels;
    const size_t elem_batch = static_cast<size_t>(data->batch) * elem_qkv;
    if (q.size() != elem_qkv || k.size() != elem_qkv || v.size() != elem_qkv) {
        if (q.size() != elem_batch || k.size() != elem_batch || v.size() != elem_batch) {
            return false;
        }
    }
    out->assign(elem_batch, 0.0f);

    cl_int err = CL_SUCCESS;
    const int seq_i = static_cast<int>(data->seq);
    const int channels_i = static_cast<int>(data->channels);
    const int head_dim_i = static_cast<int>(data->head_dim);
    const float scale = data->scale;
    const size_t qk_lws[2] = {8, 8};
    const size_t qk_gws[2] = {roundup(data->seq, qk_lws[0]), roundup(data->seq, qk_lws[1])};
    const size_t sf_lws[1] = {64};
    const size_t sf_gws[1] = {roundup(data->seq, sf_lws[0])};
    const size_t pv_lws[2] = {8, 8};
    const size_t pv_gws[2] = {roundup(data->seq, pv_lws[0]), roundup(data->head_dim, pv_lws[1])};

    for (cl_uint b = 0; b < data->batch; ++b) {
        const size_t b_off = static_cast<size_t>(b) * elem_qkv;
        err = clEnqueueWriteBuffer(engine.queue, engine.q_buf, CL_TRUE, 0, elem_qkv * sizeof(float), q.data() + b_off, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;
        err = clEnqueueWriteBuffer(engine.queue, engine.k_buf, CL_TRUE, 0, elem_qkv * sizeof(float), k.data() + b_off, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;
        err = clEnqueueWriteBuffer(engine.queue, engine.v_buf, CL_TRUE, 0, elem_qkv * sizeof(float), v.data() + b_off, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;

        for (cl_uint h = 0; h < data->num_heads; ++h) {
            const int head_offset_i = static_cast<int>(h * data->head_dim);

            err = clSetKernelArg(engine.qk_kernel, 0, sizeof(cl_mem), &engine.q_buf);
            err |= clSetKernelArg(engine.qk_kernel, 1, sizeof(cl_mem), &engine.k_buf);
            err |= clSetKernelArg(engine.qk_kernel, 2, sizeof(cl_mem), &engine.p_buf);
            err |= clSetKernelArg(engine.qk_kernel, 3, sizeof(int), &seq_i);
            err |= clSetKernelArg(engine.qk_kernel, 4, sizeof(int), &channels_i);
            err |= clSetKernelArg(engine.qk_kernel, 5, sizeof(int), &head_dim_i);
            err |= clSetKernelArg(engine.qk_kernel, 6, sizeof(int), &head_offset_i);
            err |= clSetKernelArg(engine.qk_kernel, 7, sizeof(float), &scale);
            if (err != CL_SUCCESS) return false;
            err = clEnqueueNDRangeKernel(engine.queue, engine.qk_kernel, 2, nullptr, qk_gws, qk_lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return false;

            err = clSetKernelArg(engine.softmax_kernel, 0, sizeof(cl_mem), &engine.p_buf);
            err |= clSetKernelArg(engine.softmax_kernel, 1, sizeof(int), &seq_i);
            if (err != CL_SUCCESS) return false;
            err = clEnqueueNDRangeKernel(engine.queue, engine.softmax_kernel, 1, nullptr, sf_gws, sf_lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return false;

            err = clSetKernelArg(engine.pv_kernel, 0, sizeof(cl_mem), &engine.p_buf);
            err |= clSetKernelArg(engine.pv_kernel, 1, sizeof(cl_mem), &engine.v_buf);
            err |= clSetKernelArg(engine.pv_kernel, 2, sizeof(cl_mem), &engine.o_buf);
            err |= clSetKernelArg(engine.pv_kernel, 3, sizeof(int), &seq_i);
            err |= clSetKernelArg(engine.pv_kernel, 4, sizeof(int), &channels_i);
            err |= clSetKernelArg(engine.pv_kernel, 5, sizeof(int), &head_dim_i);
            err |= clSetKernelArg(engine.pv_kernel, 6, sizeof(int), &head_offset_i);
            if (err != CL_SUCCESS) return false;
            err = clEnqueueNDRangeKernel(engine.queue, engine.pv_kernel, 2, nullptr, pv_gws, pv_lws, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) return false;
        }

        err = clEnqueueReadBuffer(engine.queue, engine.o_buf, CL_TRUE, 0, elem_qkv * sizeof(float), out->data() + b_off, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) return false;
    }
    clFinish(engine.queue);

    if (data->profile_opencl_backend) {
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr,
                     "[QCOM_ML_VAE][GGML_ATTN] seq=%u dim=%u time=%.3fms\n",
                     data->seq,
                     data->channels,
                     ms);
    }
    return true;
}

static void upload_fp32_to_tensor(const MLTensor& tensor, const std::vector<float>& data) {
    const cl_ml_tensor_desc_qcom desc = tensor.getDesc();
    if (desc.data_type == CL_FLOAT) {
        tensor.uploadDataIntoGPUMem(data.data());
        return;
    }
    if (desc.data_type == CL_HALF_FLOAT) {
        std::vector<cl_half> fp16(data.size(), fp32ToFP16(0.0f));
        for (size_t i = 0; i < data.size(); ++i) {
            fp16[i] = fp32ToFP16(data[i]);
        }
        tensor.uploadDataIntoGPUMem(fp16.data());
        return;
    }
    if (desc.data_type == CL_ML_BFLOAT16_QCOM) {
        std::vector<cl_ml_bfloat16_qcom> bf16(data.size(), fp32ToBFP16(0.0f));
        for (size_t i = 0; i < data.size(); ++i) {
            bf16[i] = fp32ToBFP16(data[i]);
        }
        tensor.uploadDataIntoGPUMem(bf16.data());
        return;
    }

    // Keep behavior deterministic for unexpected tensor dtypes.
    tensor.uploadDataIntoGPUMem(data.data());
}

static std::unique_ptr<CpuAttentionHostOpData> create_cpu_attention_host_op_data(
    const MLTensor& q,
    const MLTensor& k,
    const MLTensor& v,
    const MLTensor& out,
    cl_uint num_heads) {
    const auto dims = q.getDims();
    const cl_uint batch = dims.n;
    const cl_uint seq = dims.h;
    const cl_uint channels = dims.w;
    assert(num_heads > 0 && channels % num_heads == 0);

    std::unique_ptr<CpuAttentionHostOpData> data(new CpuAttentionHostOpData());
    data->q = q;
    data->k = k;
    data->v = v;
    data->out = out;
    data->batch = batch;
    data->seq = seq;
    data->channels = channels;
    data->num_heads = num_heads;
    data->head_dim = channels / num_heads;
    data->element_count = static_cast<size_t>(batch) * seq * channels;
    data->scale = 1.0f / std::sqrt(static_cast<float>(data->head_dim));
    data->debug = env_enabled("SD_QCOM_ML_VAE_HOST_ATTN_DEBUG");
    std::string backend;
    if (parse_backend_env("SD_QCOM_ML_VAE_HOST_ATTN_BACKEND", &backend)) {
        data->use_opencl_backend = (backend == "ggml" || backend == "opencl" || backend == "ocl");
    }
    data->profile_opencl_backend = env_enabled("SD_QCOM_ML_VAE_HOST_ATTN_BACKEND_PROFILE");
    return data;
}

static void run_cpu_attention_host_op(void* user_data) {
    auto* data = reinterpret_cast<CpuAttentionHostOpData*>(user_data);
    if (data == nullptr) {
        return;
    }

    std::vector<float> q = data->q.downloadDataFromGPUMem();
    std::vector<float> k = data->k.downloadDataFromGPUMem();
    std::vector<float> v = data->v.downloadDataFromGPUMem();
    if (q.size() != data->element_count || k.size() != data->element_count || v.size() != data->element_count) {
        std::vector<float> out_zeros(data->element_count, 0.0f);
        upload_fp32_to_tensor(data->out, out_zeros);
        return;
    }

    if (data->use_opencl_backend) {
        std::vector<float> out_opencl;
        if (run_opencl_attention_backend(data, q, k, v, &out_opencl)) {
            if (data->debug) {
                std::fprintf(stderr,
                             "[QCOM_ML_VAE][HOST_ATTN] backend=opencl batch=%u seq=%u channels=%u heads=%u\n",
                             data->batch,
                             data->seq,
                             data->channels,
                             data->num_heads);
            }
            upload_fp32_to_tensor(data->out, out_opencl);
            return;
        }
        if (data->debug) {
            std::fprintf(stderr, "[QCOM_ML_VAE][HOST_ATTN] backend=opencl failed, fallback=cpu\n");
        }
    }

    std::vector<float> out(data->element_count, 0.0f);
    std::vector<float> scores(data->seq, 0.0f);

    const size_t bc_stride = static_cast<size_t>(data->seq) * data->channels;
    for (cl_uint b = 0; b < data->batch; ++b) {
        const size_t b_off = static_cast<size_t>(b) * bc_stride;
        for (cl_uint h = 0; h < data->num_heads; ++h) {
            const size_t h_off = static_cast<size_t>(h) * data->head_dim;
            for (cl_uint i = 0; i < data->seq; ++i) {
                const float* qi = q.data() + b_off + static_cast<size_t>(i) * data->channels + h_off;

                float row_max = -std::numeric_limits<float>::infinity();
                for (cl_uint j = 0; j < data->seq; ++j) {
                    const float* kj = k.data() + b_off + static_cast<size_t>(j) * data->channels + h_off;
                    float dot = 0.0f;
                    for (cl_uint d = 0; d < data->head_dim; ++d) {
                        dot += qi[d] * kj[d];
                    }
                    float score = dot * data->scale;
                    scores[j] = score;
                    row_max = std::max(row_max, score);
                }

                float denom = 0.0f;
                for (cl_uint j = 0; j < data->seq; ++j) {
                    float e = std::exp(scores[j] - row_max);
                    scores[j] = e;
                    denom += e;
                }
                if (denom <= 0.0f || !std::isfinite(denom)) {
                    denom = 1.0f;
                }
                const float inv_denom = 1.0f / denom;

                float* oi = out.data() + b_off + static_cast<size_t>(i) * data->channels + h_off;
                for (cl_uint d = 0; d < data->head_dim; ++d) {
                    oi[d] = 0.0f;
                }
                for (cl_uint j = 0; j < data->seq; ++j) {
                    const float p = scores[j] * inv_denom;
                    const float* vj = v.data() + b_off + static_cast<size_t>(j) * data->channels + h_off;
                    for (cl_uint d = 0; d < data->head_dim; ++d) {
                        oi[d] += p * vj[d];
                    }
                }
            }
        }
    }

    if (data->debug) {
        std::fprintf(stderr,
                     "[QCOM_ML_VAE][HOST_ATTN] done batch=%u seq=%u channels=%u heads=%u\n",
                     data->batch,
                     data->seq,
                     data->channels,
                     data->num_heads);
    }
    upload_fp32_to_tensor(data->out, out);
}

static bool weight_file_exists(const std::string& prefix) {
    const std::string f16 = prefix + ".qfp16";
    FILE* fp              = std::fopen(f16.c_str(), "rb");
    if (fp != nullptr) {
        std::fclose(fp);
        return true;
    }
    const std::string f32 = prefix + ".qfp32";
    fp                    = std::fopen(f32.c_str(), "rb");
    if (fp != nullptr) {
        std::fclose(fp);
        return true;
    }
    return false;
}

/*******************************************************************************************************************************
*   Decoder::Decoder
*
*   @brief
*       Parameterized Constructor for Decoder model.
*   @note
*       Its caller's responsibility to release the cl environment and tuning cache.
*
*******************************************************************************************************************************/
Decoder::Decoder(
    const NNModelDesc&    desc,     ///< Model descriptor
    const CLEnvironment&  cl_env)   ///< OpenCL environment
    :
    CLMLBaseNN(
        desc,
        cl_env),
    m_desc(desc)
{
    m_desc.pretrained_model_path += "weights/decoder/";
}

/*******************************************************************************************************************************
*   Decoder::inference
*
*   @brief
*       Creates Decoder model in CLML and returns model outputs.
*
*******************************************************************************************************************************/
std::vector<MLTensor> Decoder::create(
    const std::vector<MLTensor>&   inputs,   ///< Inputs to this model
    const std::vector<MLTensor>&   targets)  ///< Optional target tensor (Only for training mode)
{
    (void)(targets);
    assert(inputs.size() == 1 || !"Decoder expects one input tensor");

    // Set model input tensors
    setModelInputs(inputs);

    // Flux.1 AE decoder path:
    // - latent channels: 16
    // - no post_quant_conv node
    MLTensor layer_input = inputs[0];

    // Decoder conv in
    layer_input = createConvOp(
        layer_input,
        { 512, 16, 3, 3 }, // filter dims
        {
            CL_CONVOLUTION_MODE_CONVOLUTION_QCOM,
            1, // group count
            4, // num_dimensions
            { 1, 1, 0 }, // padding before
            { 1, 1, 0 }, // padding after
            { 1, 1, 1 }, // stride
            { 1, 1, 1 }, // dilation
            0,  // bit field
            getModelArithmeticMode(m_desc.model_dtype)
        },
        true, // If conv has bias
        m_desc.pretrained_model_path + "decoder_conv_in_weight",
        m_desc.pretrained_model_path + "decoder_conv_in_bias");

    // Decoder mid block
    layer_input = createMidBlockOps(
        layer_input,
        512, // channels
        2, // num_blocks
        1e-6f, // eps
        CL_ACTIVATION_SILU,
        32, // norm_groups,
        1, // num_attn_heads
        m_desc.pretrained_model_path + "decoder_mid");

    // Decoder up blocks
    const cl_uint num_up_blocks                             = 4;
    const std::vector<cl_uint> reversed_block_out_channels = { 512, 512, 256, 128 };
    const std::vector<std::string> up_block_prefixes       = {
        m_desc.pretrained_model_path + "decoder_up_3",
        m_desc.pretrained_model_path + "decoder_up_2",
        m_desc.pretrained_model_path + "decoder_up_1",
        m_desc.pretrained_model_path + "decoder_up_0",
    };
    cl_uint out_channels = reversed_block_out_channels[0];
    for (cl_uint i = 0; i < num_up_blocks; i++)
    {
        cl_uint in_channels = out_channels;
        out_channels = reversed_block_out_channels[i];
        layer_input = createUpBlockOps(
            layer_input,
            in_channels,
            out_channels,
            3, // num_blocks
            1e-6f, // eps
            CL_ACTIVATION_SILU,
            32, // norm_groups,
            i == (num_up_blocks - 1) ? false : true, // add_down_sample,
            up_block_prefixes[i]);
    }

    // Output
    layer_input = createGroupNormOp(
        layer_input,
        {
            CL_GROUPNORM_MODE_INSTANCE_QCOM,
            32, //norm_groups
            { { 1e-6f }, CL_FLOAT },
            getModelArithmeticMode(m_desc.model_dtype)
        },
        m_desc.pretrained_model_path + "decoder_norm_out_weight",
        m_desc.pretrained_model_path + "decoder_norm_out_bias");

    layer_input = createActivationOp(
        layer_input,
        {
            CL_ACTIVATION_SILU,
            CL_PROPAGATE_NAN_QCOM,
            getModelArithmeticMode(m_desc.model_dtype)
        },
        layer_input.getDims().c, // param_size
        ""); // param filename prefix

    layer_input = createConvOp(
        layer_input,
        { 3, 128, 3, 3 }, // filter dims
        {
            CL_CONVOLUTION_MODE_CONVOLUTION_QCOM,
            1, // group count
            4, // num_dimensions
            { 1, 1, 0 }, // padding before
            { 1, 1, 0 }, // padding after
            { 1, 1, 1 }, // stride
            { 1, 1, 1 }, // dilation
            0,  // bit field
            getModelArithmeticMode(m_desc.model_dtype)
        },
        true, // If conv has bias
        m_desc.pretrained_model_path + "decoder_conv_out_weight",
        m_desc.pretrained_model_path + "decoder_conv_out_bias",
        true); // defer_output_mem

    // Set the model output
    setModelOutputs({ layer_input });

    // Returns model output
    return getModelOutputs();
}

/*******************************************************************************************************************************
*   Decoder::~Decoder
*
*   @brief
*       Decoder class destructor.
*
*******************************************************************************************************************************/
Decoder::~Decoder()
{
    tearDown();
}

/*******************************************************************************************************************************
*   Decoder::createMidBlockOps
*
*   @brief
*       Creates UNet mid block operations.
*
*******************************************************************************************************************************/
MLTensor Decoder::createMidBlockOps(
    const MLTensor&                     input,                 ///< Input tensor to the block
    cl_uint                             channels,              ///< The number of channels in the input and output
    cl_uint                             num_blocks,            ///< Number of resnet blocks and transformer blocks
    float                               eps,                   ///< Group Norm Epsilon value
    const cl_activation_function_qcom&  act_fxn,               ///< Activation fxn type in ResNet block
    cl_uint                             norm_groups,           ///< Num of groups in group normalization
    cl_uint                             num_attn_heads,        ///< Number of attn heads
    const std::string&                  param_filename_prefix, ///< Param filename prefix to load pretrained param data
    bool                                defer_output_mem)      ///< Defer output tensor backing device memory creation
{
    (void)num_blocks;
    MLTensor layer_input = input;
    layer_input          = createResNetBlockOps(
        layer_input,
        channels,
        channels,
        eps,
        act_fxn,
        norm_groups,
        param_filename_prefix + "_block_1",
        defer_output_mem);

    const char* disable_attn = std::getenv("SD_QCOM_ML_VAE_DISABLE_ATTN");
    if (!(disable_attn != nullptr && disable_attn[0] != '\0' && disable_attn[0] != '0')) {
        layer_input = createAttentionBlockOps(
            layer_input,
            num_attn_heads,
            channels,
            eps,
            norm_groups,
            param_filename_prefix + "_attn_1");
    }

    return createResNetBlockOps(
        layer_input,
        channels,
        channels,
        eps,
        act_fxn,
        norm_groups,
        param_filename_prefix + "_block_2",
        defer_output_mem);
}

/*******************************************************************************************************************************
*   Decoder::createUpBlockOps
*
*   @brief
*       Creates Decoder up block operations.
*
*******************************************************************************************************************************/
MLTensor Decoder::createUpBlockOps(
    const MLTensor&                      input,                  ///< Input tensor to the block
    cl_uint                              in_channels,            ///< Number of input features to the block
    cl_uint                              out_channels,           ///< Number of output features to the block
    cl_uint                              num_blocks,             ///< Number of resnet blocks
    float                                eps,                    ///< Group Norm Epsilon value
    const cl_activation_function_qcom&   act_fxn,                ///< Activation fxn type in ResNet block
    cl_uint                              norm_groups,            ///< Num of groups in group normalization
    bool                                 add_up_sample,          ///< Add up sample block if any.
    const std::string&                   param_filename_prefix,  ///< Param filename prefix to load pretrained param data
    bool                                 defer_output_mem)       ///< Defer output tensor backing device memory creation
{
    MLTensor layer_input = input;
    for (cl_uint i = 0; i < num_blocks; i++)
    {
        layer_input = createResNetBlockOps(
            layer_input,
            i == 0 ? in_channels : out_channels, // in_channels
            out_channels,
            eps,
            act_fxn,
            norm_groups,
            param_filename_prefix + "_block_" + std::to_string(i),
            defer_output_mem);
    }

    if (add_up_sample == true)
    {
        layer_input = createUpSampleOp(
            layer_input,
            out_channels,
            out_channels,
            param_filename_prefix + "_upsample",
            defer_output_mem);
    }

    return layer_input;
}

/*******************************************************************************************************************************
*   Decoder::createResNetBlockOps
*
*   @brief
*       Creates Decoder ResNet block operations.
*
*******************************************************************************************************************************/
MLTensor Decoder::createResNetBlockOps(
    const MLTensor&                      input,                  ///< Input tensor to the block
    cl_uint                              in_channels,            ///< The number of channels in input
    cl_uint                              out_channels,           ///< The number of channels in output
    float                                eps,                    ///< Group Norm Epsilon value
    const cl_activation_function_qcom&   act_fxn,                ///< Activation fxn type in ResNet block
    cl_uint                              norm_groups,            ///< Num of groups in group normalization
    const std::string&                   param_filename_prefix,  ///< Param filename prefix to load pretrained param data
    bool                                 defer_output_mem)       ///< Defer output tensor backing device memory creation
{
    // Save block input for residual connection
    MLTensor skip_connection = input;

    MLTensor layer_input = input;
    layer_input = createGroupNormOp(
        layer_input,
        {
            CL_GROUPNORM_MODE_INSTANCE_QCOM,
            norm_groups,
            { { eps }, CL_FLOAT },
            getModelArithmeticMode(m_desc.model_dtype)
        },
        param_filename_prefix +"_norm1_weight",
        param_filename_prefix +"_norm1_bias");

    layer_input = createActivationOp(
        layer_input,
        {
            act_fxn,
            CL_PROPAGATE_NAN_QCOM,
            getModelArithmeticMode(m_desc.model_dtype)
        },
        in_channels,
        ""); // Param filename prefix

    layer_input = createConvOp(
        layer_input,
        { out_channels, in_channels, 3, 3}, // filter dims
        {
            CL_CONVOLUTION_MODE_CONVOLUTION_QCOM,
            1, // group count
            4, // num_dimensions
            { 1, 1, 0 }, // padding before
            { 1, 1, 0 }, // padding after
            { 1, 1, 1 }, // stride
            { 1, 1, 1 }, // dilation
            0,  // bit field
            getModelArithmeticMode(m_desc.model_dtype)
        },
        true, // If conv has bias
        param_filename_prefix +"_conv1_weight",
        param_filename_prefix +"_conv1_bias");

    layer_input = createGroupNormOp(
        layer_input,
        {
            CL_GROUPNORM_MODE_INSTANCE_QCOM,
            norm_groups,
            { { eps }, CL_FLOAT },
            getModelArithmeticMode(m_desc.model_dtype)
        },
        param_filename_prefix +"_norm2_weight",
        param_filename_prefix +"_norm2_bias");

    layer_input = createActivationOp(
        layer_input,
        {
            act_fxn,
            CL_PROPAGATE_NAN_QCOM,
            getModelArithmeticMode(m_desc.model_dtype)
        },
        out_channels,
        ""); // Param filename prefix

    layer_input = createConvOp(
        layer_input,
        { out_channels, out_channels, 3, 3}, // filter dims
        {
            CL_CONVOLUTION_MODE_CONVOLUTION_QCOM,
            1, // group count
            4, // num_dimensions
            { 1, 1, 0 }, // padding before
            { 1, 1, 0 }, // padding after
            { 1, 1, 1 }, // stride
            { 1, 1, 1 }, // dilation
            0,  // bit field
            getModelArithmeticMode(m_desc.model_dtype)
        },
        true, // If conv has bias
        param_filename_prefix +"_conv2_weight",
        param_filename_prefix +"_conv2_bias");

    if (in_channels != out_channels)
    {
        std::string shortcut_prefix = param_filename_prefix + "_nin_shortcut";
        if (!weight_file_exists(shortcut_prefix + "_weight")) {
            shortcut_prefix = param_filename_prefix + "_conv_shortcut";
        }
        skip_connection = createConvOp(
            skip_connection,
            { out_channels, in_channels, 1, 1}, // filter dims
            {
                CL_CONVOLUTION_MODE_CONVOLUTION_QCOM,
                1, // group count
                4, // num_dimensions
                { 0, 0, 0 }, // padding before
                { 0, 0, 0 }, // padding after
                { 1, 1, 1 }, // stride
                { 1, 1, 1 }, // dilation
                0,  // bit field
                getModelArithmeticMode(m_desc.model_dtype)
            },
            true, // If conv has bias
            shortcut_prefix + "_weight",
            shortcut_prefix + "_bias");
    }

    layer_input = createBinaryOp(
        skip_connection,
        layer_input,
        {
            CL_TENSOR_OP_ADD_QCOM,
            { { 1.0 }, CL_FLOAT }, // alpha
            { { 1.0 }, CL_FLOAT }, // beta
            { { 0.0 }, CL_FLOAT }, // gamma
            getModelArithmeticMode(m_desc.model_dtype),
        },
        defer_output_mem);

    return layer_input;
}

/*******************************************************************************************************************************
*   Decoder::createAttentionBlockOps
*
*   @brief
*       Creates Decoder Attention block operations.
*
*******************************************************************************************************************************/
MLTensor Decoder::createAttentionBlockOps(
    const MLTensor&      input,                  ///< Input tensor to the block
    cl_uint              num_attn_heads,         ///< Number of attention heads
    cl_uint              channels,               ///< The number of channels in the input and output
    float                eps,                    ///< Group norm epsilon value
    cl_uint              norm_groups,            ///< The number of groups in group norm
    const std::string&   param_filename_prefix,  ///< Param filename prefix to load pretrained param data
    bool                 defer_output_mem)       ///< Defer output tensor backing device memory creation
{
    // Save block input for residual connection
    MLTensor skip_connection = input;

    MLTensor layer_input = input;
    layer_input = createGroupNormOp(
        layer_input,
        {
            CL_GROUPNORM_MODE_INSTANCE_QCOM,
            norm_groups,
            { { eps }, CL_FLOAT },
            getModelArithmeticMode(m_desc.model_dtype)
        },
        param_filename_prefix + "_norm_weight",
        param_filename_prefix + "_norm_bias");

    // Border layer: Transition from Convolution Network to Transformer Network
    layer_input = createFusedReshapeAndPermuteOp(
        layer_input,
        {
            4, // order len
            {0, 3, 2, 1}, // order
            getModelArithmeticMode(m_desc.model_dtype)
        },
        CL_TENSOR_USAGE_TNN_QCOM, // output usage
        {
            layer_input.getDims().n, 1,
            layer_input.getDims().h * layer_input.getDims().w, layer_input.getDims().c
        }); // output_dims

    const bool use_host_attn = env_enabled("SD_QCOM_ML_VAE_HOST_ATTN");
    if (use_host_attn) {
        MLTensor attn_q = createFullyConnectedOp(
            layer_input,
            {
                0, // flatten_axis
                CL_FC_WEIGHT_TRANSFORM_TRANSPOSE_QCOM,
                getModelArithmeticMode(m_desc.model_dtype)
            },
            channels,
            param_filename_prefix + "_q_weight",
            param_filename_prefix + "_q_bias");
        MLTensor attn_k = createFullyConnectedOp(
            layer_input,
            {
                0, // flatten_axis
                CL_FC_WEIGHT_TRANSFORM_TRANSPOSE_QCOM,
                getModelArithmeticMode(m_desc.model_dtype)
            },
            channels,
            param_filename_prefix + "_k_weight",
            param_filename_prefix + "_k_bias");
        MLTensor attn_v = createFullyConnectedOp(
            layer_input,
            {
                0, // flatten_axis
                CL_FC_WEIGHT_TRANSFORM_TRANSPOSE_QCOM,
                getModelArithmeticMode(m_desc.model_dtype)
            },
            channels,
            param_filename_prefix + "_v_weight",
            param_filename_prefix + "_v_bias");

        MLTensor attn_out = createTensor(
            { layer_input.getDims().n, 1, layer_input.getDims().h, channels },
            m_desc.model_dtype,
            CL_TENSOR_LAYOUT_OPTIMAL_QCOM,
            CL_TENSOR_USAGE_TNN_QCOM,
            "",
            m_desc.tensor_data_src_layout,
            true,
            MLTensorOrigin::NNFramework);

        auto host_attn_data = create_cpu_attention_host_op_data(attn_q, attn_k, attn_v, attn_out, num_attn_heads);
        CustomHostOp host_op = {};
        host_op.callback = run_cpu_attention_host_op;
        host_op.user_data = host_attn_data.get();
        createHostOp(
            "CPU_Attention",
            { attn_q, attn_k, attn_v },
            { attn_out },
            host_op);
        m_host_attn_ops.push_back(std::move(host_attn_data));

        layer_input = createFullyConnectedOp(
            attn_out,
            {
                0, // flatten_axis
                CL_FC_WEIGHT_TRANSFORM_TRANSPOSE_QCOM,
                getModelArithmeticMode(m_desc.model_dtype)
            },
            channels,
            param_filename_prefix + "_proj_out_weight",
            param_filename_prefix + "_proj_out_bias");
    } else {
        cl_arithmetic_mode_qcom mha_arith_mode = resolve_mha_arithmetic_mode(getModelArithmeticMode(m_desc.model_dtype));
        cl_softmax_mode_qcom mha_softmax_mode = resolve_mha_softmax_mode(CL_SOFTMAX_MODE_WIDTH_QCOM);
        cl_multi_head_attn_weights_transform_qcom mha_weight_transform =
            resolve_mha_weight_transform(CL_MULTI_HEAD_ATTN_WEIGHTS_TRANSFORM_TRANSPOSE_QCOM);
        const bool mha_attn_has_bias = !env_enabled("SD_QCOM_ML_VAE_MHA_NO_ATTN_BIAS");
        const bool mha_out_has_bias = !env_enabled("SD_QCOM_ML_VAE_MHA_NO_OUT_BIAS");

        layer_input = createMultiHeadAttentionOp(
            layer_input,
            layer_input,
            layer_input,
            {
                num_attn_heads,
                channels / num_attn_heads,
                channels / num_attn_heads,
                mha_softmax_mode,
                mha_weight_transform,
                false, // is_causal
                mha_arith_mode
            },
            channels,
            mha_attn_has_bias,
            mha_out_has_bias,
            param_filename_prefix + "_q_weight",
            param_filename_prefix + "_k_weight",
            param_filename_prefix + "_v_weight",
            param_filename_prefix + "_proj_out_weight",
            param_filename_prefix + "_q_bias",
            param_filename_prefix + "_k_bias",
            param_filename_prefix + "_v_bias",
            param_filename_prefix + "_proj_out_bias",
            NULL,
            NULL,
            defer_output_mem);
    }

    // Border layer: Transition from Transformer Network to Convolution Network
    layer_input = createFusedReshapeAndPermuteOp(
        layer_input,
        {
            4, // order len
            {0, 3, 1, 2}, // order
            getModelArithmeticMode(m_desc.model_dtype)
        },
        CL_TENSOR_USAGE_CNN_QCOM, // output usage
        {
            layer_input.getDims().n, layer_input.getDims().w,
            input.getDims().h, input.getDims().w
        }); // output_dims

    layer_input = createBinaryOp(
        skip_connection,
        layer_input,
        {
            CL_TENSOR_OP_ADD_QCOM,
            { { 1.0 }, CL_FLOAT }, // alpha
            { { 1.0 }, CL_FLOAT }, // beta
            { { 0.0 }, CL_FLOAT }, // gamma
            getModelArithmeticMode(m_desc.model_dtype),
        },
        defer_output_mem);

    return layer_input;
}

/*******************************************************************************************************************************
*   Decoder::createUpSampleOp
*
*   @brief
*       Creates Decoder upsample block operation.
*
*******************************************************************************************************************************/
MLTensor Decoder::createUpSampleOp(
    const MLTensor&                        input,                 ///< Input tensor to the block
    cl_uint                                in_channels,           ///< Number of input features to the block
    cl_uint                                out_channels,          ///< Number of output features to the block
    const std::string&                     param_filename_prefix, ///< Param filename prefix to load pretrained param data
    bool                                   defer_output_mem)      ///< Defer output tensor backing device memory creation
{
    MLTensor layer_input = input;

    layer_input = createResizeOp(
        layer_input,
        {
            CL_RESIZE_NEAREST_MODE_QCOM,
            false,
            false,
            getModelArithmeticMode(m_desc.model_dtype)
        },
        {
            input.getDims().n, input.getDims().c,
            input.getDims().h*2, input.getDims().w*2
        });

    layer_input = createConvOp(
        layer_input,
        { out_channels, in_channels, 3, 3 },
        {
            CL_CONVOLUTION_MODE_CONVOLUTION_QCOM,
            1, // group count
            4, // num_dimensions
            { 1, 1, 0 }, // padding before
            { 1, 1, 0 }, // padding after
            { 1, 1, 1 }, // stride
            { 1, 1, 1 }, // dilation
            0,  // bit field
            getModelArithmeticMode(m_desc.model_dtype)
        },
        true, // If conv has bias
        param_filename_prefix + "_conv_weight",
        param_filename_prefix + "_conv_bias",
        defer_output_mem);

    return layer_input;
}

/*******************************************************************************************************************************
*   Decoder::tearDown
*
*   @brief
*       Releases resources used in the model.
*   @note
*       This class does not release tensor and its memory with retain_data (True). These tensors will be used in other
*       models and must be released by them.
*
*******************************************************************************************************************************/
void Decoder::tearDown()
{
}
