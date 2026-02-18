#include "clml_decoder.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include "nn_framework/clml_utils_nn.h"
#include "utils/clml_fp_conv_utils.h"
#include "utils/clml_utils.h"

namespace {

static bool bridge_debug_enabled() {
    const char* v = std::getenv("SD_QCOM_ML_VAE_DEBUG");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

static void bridge_log(const char* fmt, ...) {
    if (!bridge_debug_enabled()) {
        return;
    }
    std::fprintf(stderr, "[QCOM_ML_VAE] ");
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

static constexpr int kFluxAeLatentChannels      = 16;
static constexpr int kFlux2PackedLatentChannels = 128;
static constexpr int kFlux2LatentChannels       = 32;

struct SdQcomMlVaeCtx {
    CLEnvironment cl_env = {};
    CLMLInterfaceV4QCOM* clml_intf = nullptr;

    NNModelDesc model_desc = {
        NNModelMode::INFERENCE,
        CL_HALF_FLOAT,
        CL_TENSOR_LAYOUT_NCHW_QCOM,
        true,
        "",
        false,
        "",
        false,
        false,
        false,
        false,
        false,
        true,
        false,
    };

    std::unique_ptr<Decoder> decoder;
    MLTensor latent;
    std::vector<MLTensor> outputs;

    int latent_w = 0;
    int latent_h = 0;
    int latent_c = 0;
    int batch = 0;

    bool disable_mnn_attention = true;
    bool fallback_attn_len_16384 = true;
};

static void set_err(char* err_buf, int err_buf_cap, const std::string& msg) {
    if (err_buf == nullptr || err_buf_cap <= 0) {
        return;
    }
    std::snprintf(err_buf, static_cast<size_t>(err_buf_cap), "%s", msg.c_str());
}

static std::vector<cl_half> fp32_to_fp16(const float* src, size_t count) {
    std::vector<cl_half> dst(count);
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp32ToFP16(src[i]);
    }
    return dst;
}

static bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

static inline size_t whcn_index(int x, int y, int c, int n, int w, int h, int channels) {
    return (((size_t)n * (size_t)channels + (size_t)c) * (size_t)h + (size_t)y) * (size_t)w + (size_t)x;
}

static bool unpack_flux2_latent_whcn(const float* src,
                                     int in_w,
                                     int in_h,
                                     int in_c,
                                     int batch,
                                     std::vector<float>& dst,
                                     int& out_w,
                                     int& out_h,
                                     int& out_c) {
    if (src == nullptr || in_w <= 0 || in_h <= 0 || in_c <= 0 || batch <= 0) {
        return false;
    }
    if (in_c % 4 != 0) {
        return false;
    }

    out_w = in_w * 2;
    out_h = in_h * 2;
    out_c = in_c / 4;
    dst.assign((size_t)out_w * (size_t)out_h * (size_t)out_c * (size_t)batch, 0.0f);

    for (int n = 0; n < batch; ++n) {
        for (int c = 0; c < out_c; ++c) {
            const int c4 = c * 4;
            for (int y = 0; y < in_h; ++y) {
                for (int x = 0; x < in_w; ++x) {
                    // Matches AutoencodingEngine::decode flux2 path:
                    // in_ch = c * 4 + bx + 2 * by, out_xy = (2*x + bx, 2*y + by)
                    const float v00 = src[whcn_index(x, y, c4 + 0, n, in_w, in_h, in_c)];
                    const float v10 = src[whcn_index(x, y, c4 + 1, n, in_w, in_h, in_c)];
                    const float v01 = src[whcn_index(x, y, c4 + 2, n, in_w, in_h, in_c)];
                    const float v11 = src[whcn_index(x, y, c4 + 3, n, in_w, in_h, in_c)];

                    dst[whcn_index(2 * x + 0, 2 * y + 0, c, n, out_w, out_h, out_c)] = v00;
                    dst[whcn_index(2 * x + 1, 2 * y + 0, c, n, out_w, out_h, out_c)] = v10;
                    dst[whcn_index(2 * x + 0, 2 * y + 1, c, n, out_w, out_h, out_c)] = v01;
                    dst[whcn_index(2 * x + 1, 2 * y + 1, c, n, out_w, out_h, out_c)] = v11;
                }
            }
        }
    }
    return true;
}

static bool tensor_prefix_exists(const std::string& prefix) {
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

static bool validate_flux_decoder_weight_layout(const SdQcomMlVaeCtx* ctx, int latent_c, std::string& err) {
    if (ctx == nullptr) {
        err = "null ctx";
        return false;
    }
    const std::string root = ctx->model_desc.pretrained_model_path + "weights/decoder/";
    const char* required[] = {
        "decoder_conv_in_weight",
        "decoder_mid_attn_1_q_weight",
        "decoder_mid_block_1_conv1_weight",
        "decoder_up_3_block_0_conv1_weight",
        "decoder_up_0_block_0_conv1_weight",
        "decoder_norm_out_weight",
        "decoder_conv_out_weight",
    };
    for (const char* name : required) {
        if (!tensor_prefix_exists(root + name)) {
            err = "missing Flux decoder tensor: " + std::string(name);
            return false;
        }
    }
    if (latent_c == kFlux2LatentChannels) {
        const char* required_flux2[] = {
            "post_quant_conv_weight",
            "post_quant_conv_bias",
        };
        for (const char* name : required_flux2) {
            if (!tensor_prefix_exists(root + name)) {
                err = "missing Flux2 decoder tensor: " + std::string(name);
                return false;
            }
        }
    }
    return true;
}

static bool ensure_graph_for_shape(SdQcomMlVaeCtx* ctx,
                                   int latent_w,
                                   int latent_h,
                                   int latent_c,
                                   int batch,
                                   std::string& err) {
    if (ctx == nullptr) {
        err = "null ctx";
        return false;
    }

    if (latent_c != kFluxAeLatentChannels && latent_c != kFlux2LatentChannels) {
        err = "unsupported latent_c=" + std::to_string(latent_c) +
              " (qcom_ml bridge currently expects latent_c=16 or 32)";
        return false;
    }

    const int64_t seq_len = static_cast<int64_t>(latent_w) * static_cast<int64_t>(latent_h);
    const bool host_attn_enabled = env_flag_enabled("SD_QCOM_ML_VAE_HOST_ATTN");
    if (ctx->fallback_attn_len_16384 && seq_len >= 16384 && !host_attn_enabled) {
        err = "attention sequence length " + std::to_string(seq_len) +
              " is not supported by current qcom_ml attention path";
        return false;
    }

    if (!validate_flux_decoder_weight_layout(ctx, latent_c, err)) {
        return false;
    }

    if (ctx->decoder && ctx->latent_w == latent_w && ctx->latent_h == latent_h && ctx->latent_c == latent_c &&
        ctx->batch == batch) {
        bridge_log("reuse graph latent=%dx%dx%dx%d", latent_w, latent_h, latent_c, batch);
        return true;
    }

    ctx->decoder.reset();
    ctx->outputs.clear();

    tensor_dims_t latent_dims = {
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(latent_c),
        static_cast<uint32_t>(latent_h),
        static_cast<uint32_t>(latent_w),
    };

    ctx->latent = createMLTensor(ctx->cl_env,
                                 ctx->clml_intf,
                                 latent_dims,
                                 ctx->model_desc.model_dtype,
                                 CL_TENSOR_LAYOUT_NCHW_QCOM,
                                 CL_TENSOR_USAGE_CNN_QCOM);
    bridge_log("create decoder graph latent=%dx%dx%dx%d", latent_w, latent_h, latent_c, batch);
    ctx->decoder.reset(new Decoder(ctx->model_desc, ctx->cl_env));
    bridge_log("decoder create begin");
    ctx->outputs = ctx->decoder->create({ctx->latent});
    bridge_log("decoder create done (outputs=%zu)", ctx->outputs.size());
    if (ctx->outputs.size() != 1) {
        err = "decoder output size is not 1";
        return false;
    }
    bridge_log("decoder initParams begin");
    ctx->decoder->initParams();
    bridge_log("decoder initParams done");

    ctx->latent_w = latent_w;
    ctx->latent_h = latent_h;
    ctx->latent_c = latent_c;
    ctx->batch = batch;
    return true;
}

}  // namespace

extern "C" int sd_qcom_ml_vae_create(const char* model_dir,
                                     int disable_mnn_attention,
                                     int fallback_attn_len_16384,
                                     void** out_ctx,
                                     char* err_buf,
                                     int err_buf_cap) {
    if (out_ctx == nullptr) {
        set_err(err_buf, err_buf_cap, "out_ctx is null");
        return -1;
    }
    *out_ctx = nullptr;

    if (model_dir == nullptr || model_dir[0] == '\0') {
        set_err(err_buf, err_buf_cap, "model_dir is empty");
        return -1;
    }

    std::unique_ptr<SdQcomMlVaeCtx> ctx(new SdQcomMlVaeCtx());
    bridge_log("bridge build id: step22-flux2-unpack-20260219");
    ctx->disable_mnn_attention = disable_mnn_attention != 0;
    ctx->fallback_attn_len_16384 = fallback_attn_len_16384 != 0;
    const char* fp32_env = std::getenv("SD_QCOM_ML_VAE_FP32");
    if (fp32_env != nullptr && fp32_env[0] != '\0' && fp32_env[0] != '0') {
        ctx->model_desc.model_dtype = CL_FLOAT;
        bridge_log("force model dtype to CL_FLOAT");
    }
    if (env_flag_enabled("SD_QCOM_ML_VAE_OPTIMIZE_MEM")) {
        ctx->model_desc.optimize_device_mem = true;
        bridge_log("enable optimize_device_mem");
    }
    if (env_flag_enabled("SD_QCOM_ML_VAE_RECORDABLE_QUEUE")) {
        ctx->model_desc.use_recordable_queue = true;
        bridge_log("enable use_recordable_queue");
    }
    if (env_flag_enabled("SD_QCOM_ML_VAE_GMEM")) {
        ctx->model_desc.enable_gmem_buffers = true;
        bridge_log("enable gmem buffers");
    }

    // Decoder source has MNN host-op path removed. Keep this flag for ABI compatibility.
    if (ctx->disable_mnn_attention) {
        setenv("CLML_DISABLE_MNN_ATTN", "1", 1);
    }

    std::string base_dir = model_dir;
    if (!base_dir.empty() && base_dir.back() != '/') {
        base_dir.push_back('/');
    }
    ctx->model_desc.pretrained_model_path = base_dir;

    ctx->cl_env = createCLEnvironment(false);
    const cl_int minor_version = 0;
    ctx->clml_intf = clGetMLInterfaceV4QCOM(minor_version);
    if (ctx->clml_intf == nullptr) {
        set_err(err_buf, err_buf_cap, "clGetMLInterfaceV4QCOM failed");
        releaseCLEnvironment(ctx->cl_env);
        return -1;
    }

    *out_ctx = ctx.release();
    return 0;
}

extern "C" int sd_qcom_ml_vae_decode(void* raw_ctx,
                                     const float* latent_nchw,
                                     int latent_w,
                                     int latent_h,
                                     int latent_c,
                                     int batch,
                                     float* out_nchw,
                                     int out_w,
                                     int out_h,
                                     int out_c,
                                     int* used_attn_fallback_16384,
                                     char* err_buf,
                                     int err_buf_cap) {
    SdQcomMlVaeCtx* ctx = reinterpret_cast<SdQcomMlVaeCtx*>(raw_ctx);
    if (ctx == nullptr) {
        set_err(err_buf, err_buf_cap, "ctx is null");
        return -1;
    }
    if (used_attn_fallback_16384 != nullptr) {
        *used_attn_fallback_16384 = 0;
    }
    if (latent_nchw == nullptr || out_nchw == nullptr) {
        set_err(err_buf, err_buf_cap, "latent_nchw/out_nchw is null");
        return -1;
    }
    if (latent_w <= 0 || latent_h <= 0 || latent_c <= 0 || batch <= 0 ||
        out_w <= 0 || out_h <= 0 || out_c <= 0) {
        set_err(err_buf, err_buf_cap, "invalid tensor shape");
        return -1;
    }

    int run_w = latent_w;
    int run_h = latent_h;
    int run_c = latent_c;
    std::vector<float> latent_unpacked;
    const float* latent_run = latent_nchw;
    if (latent_c == kFlux2PackedLatentChannels) {
        if (!unpack_flux2_latent_whcn(latent_nchw, latent_w, latent_h, latent_c, batch, latent_unpacked, run_w, run_h, run_c)) {
            set_err(err_buf, err_buf_cap, "failed to unpack flux2 latent (expected c%4==0)");
            return -1;
        }
        latent_run = latent_unpacked.data();
        bridge_log("flux2 latent unpack: %dx%dx%dx%d -> %dx%dx%dx%d",
                   latent_w, latent_h, latent_c, batch,
                   run_w, run_h, run_c, batch);
    }

    std::string err_msg;
    const auto t_graph_start = std::chrono::steady_clock::now();
    if (!ensure_graph_for_shape(ctx, run_w, run_h, run_c, batch, err_msg)) {
        bridge_log("ensure_graph_for_shape failed: %s", err_msg.c_str());
        set_err(err_buf, err_buf_cap, err_msg);
        return -1;
    }
    const auto t_graph_end = std::chrono::steady_clock::now();
    bridge_log("timing ensure_graph_for_shape: %.3f ms",
               std::chrono::duration<double, std::milli>(t_graph_end - t_graph_start).count());

    const size_t latent_count = static_cast<size_t>(run_w) * run_h * run_c * batch;
    const std::vector<cl_half> latent_fp16 = fp32_to_fp16(latent_run, latent_count);
    const auto t_upload_start = std::chrono::steady_clock::now();
    bridge_log("upload latent elems=%zu", latent_count);
    ctx->latent.uploadDataIntoGPUMem(latent_fp16.data(), ctx->model_desc.tensor_data_src_layout);
    const auto t_upload_end = std::chrono::steady_clock::now();
    bridge_log("timing upload_latent: %.3f ms",
               std::chrono::duration<double, std::milli>(t_upload_end - t_upload_start).count());

    const auto t_forward_start = std::chrono::steady_clock::now();
    bridge_log("decoder forward start");
    ctx->decoder->forward();
    bridge_log("decoder forward done");
    const auto t_forward_end = std::chrono::steady_clock::now();
    bridge_log("timing decoder_forward: %.3f ms",
               std::chrono::duration<double, std::milli>(t_forward_end - t_forward_start).count());
    const auto t_finish_start = std::chrono::steady_clock::now();
    clFinish(ctx->cl_env.queue);
    bridge_log("queue finish done");
    const auto t_finish_end = std::chrono::steady_clock::now();
    bridge_log("timing clFinish: %.3f ms",
               std::chrono::duration<double, std::milli>(t_finish_end - t_finish_start).count());

    const auto t_download_start = std::chrono::steady_clock::now();
    bridge_log("download output start");
    std::vector<float> out = ctx->outputs[0].downloadDataFromGPUMem();
    bridge_log("download output done elems=%zu", out.size());
    const auto t_download_end = std::chrono::steady_clock::now();
    bridge_log("timing download_output: %.3f ms",
               std::chrono::duration<double, std::milli>(t_download_end - t_download_start).count());
    const size_t out_count = static_cast<size_t>(out_w) * out_h * out_c * batch;
    if (out.size() != out_count) {
        set_err(err_buf, err_buf_cap, "output size mismatch");
        return -1;
    }
    std::memcpy(out_nchw, out.data(), out_count * sizeof(float));
    return 0;
}

extern "C" int sd_qcom_ml_vae_prepare(void* raw_ctx,
                                      int latent_w,
                                      int latent_h,
                                      int latent_c,
                                      int batch,
                                      char* err_buf,
                                      int err_buf_cap) {
    SdQcomMlVaeCtx* ctx = reinterpret_cast<SdQcomMlVaeCtx*>(raw_ctx);
    if (ctx == nullptr) {
        set_err(err_buf, err_buf_cap, "ctx is null");
        return -1;
    }
    if (latent_w <= 0 || latent_h <= 0 || latent_c <= 0 || batch <= 0) {
        set_err(err_buf, err_buf_cap, "invalid tensor shape");
        return -1;
    }

    int run_w = latent_w;
    int run_h = latent_h;
    int run_c = latent_c;
    if (latent_c == kFlux2PackedLatentChannels) {
        run_w = latent_w * 2;
        run_h = latent_h * 2;
        run_c = kFlux2LatentChannels;
        bridge_log("prepare flux2 unpack-shape: %dx%dx%dx%d -> %dx%dx%dx%d",
                   latent_w, latent_h, latent_c, batch,
                   run_w, run_h, run_c, batch);
    }

    std::string err_msg;
    const auto t_graph_start = std::chrono::steady_clock::now();
    if (!ensure_graph_for_shape(ctx, run_w, run_h, run_c, batch, err_msg)) {
        bridge_log("prepare ensure_graph_for_shape failed: %s", err_msg.c_str());
        set_err(err_buf, err_buf_cap, err_msg);
        return -1;
    }
    const auto t_graph_end = std::chrono::steady_clock::now();
    bridge_log("timing prepare_graph_for_shape: %.3f ms",
               std::chrono::duration<double, std::milli>(t_graph_end - t_graph_start).count());
    return 0;
}

extern "C" void sd_qcom_ml_vae_destroy(void* raw_ctx) {
    SdQcomMlVaeCtx* ctx = reinterpret_cast<SdQcomMlVaeCtx*>(raw_ctx);
    if (ctx == nullptr) {
        return;
    }
    ctx->decoder.reset();
    ctx->outputs.clear();
    releaseCLEnvironment(ctx->cl_env);
    delete ctx;
}
