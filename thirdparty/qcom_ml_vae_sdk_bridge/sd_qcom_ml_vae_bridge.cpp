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

static constexpr int kFluxAeLatentChannels = 16;

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

static bool validate_flux_decoder_weight_layout(const SdQcomMlVaeCtx* ctx, std::string& err) {
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

    if (latent_c != kFluxAeLatentChannels) {
        err = "unsupported latent_c=" + std::to_string(latent_c) +
              " (qcom_ml bridge currently expects Flux AE latent_c=16)";
        return false;
    }

    const int64_t seq_len = static_cast<int64_t>(latent_w) * static_cast<int64_t>(latent_h);
    if (ctx->fallback_attn_len_16384 && seq_len >= 16384) {
        err = "attention sequence length " + std::to_string(seq_len) +
              " is not supported by current qcom_ml attention path";
        return false;
    }

    if (!validate_flux_decoder_weight_layout(ctx, err)) {
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
    ctx->outputs = ctx->decoder->create({ctx->latent});
    if (ctx->outputs.size() != 1) {
        err = "decoder output size is not 1";
        return false;
    }
    ctx->decoder->initParams();

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
    ctx->disable_mnn_attention = disable_mnn_attention != 0;
    ctx->fallback_attn_len_16384 = fallback_attn_len_16384 != 0;
    const char* fp32_env = std::getenv("SD_QCOM_ML_VAE_FP32");
    if (fp32_env != nullptr && fp32_env[0] != '\0' && fp32_env[0] != '0') {
        ctx->model_desc.model_dtype = CL_FLOAT;
        bridge_log("force model dtype to CL_FLOAT");
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

    std::string err_msg;
    if (!ensure_graph_for_shape(ctx, latent_w, latent_h, latent_c, batch, err_msg)) {
        bridge_log("ensure_graph_for_shape failed: %s", err_msg.c_str());
        set_err(err_buf, err_buf_cap, err_msg);
        return -1;
    }

    const size_t latent_count = static_cast<size_t>(latent_w) * latent_h * latent_c * batch;
    const std::vector<cl_half> latent_fp16 = fp32_to_fp16(latent_nchw, latent_count);
    bridge_log("upload latent elems=%zu", latent_count);
    ctx->latent.uploadDataIntoGPUMem(latent_fp16.data(), ctx->model_desc.tensor_data_src_layout);

    bridge_log("decoder forward start");
    ctx->decoder->forward();
    bridge_log("decoder forward done");
    clFinish(ctx->cl_env.queue);
    bridge_log("queue finish done");

    bridge_log("download output start");
    std::vector<float> out = ctx->outputs[0].downloadDataFromGPUMem();
    bridge_log("download output done elems=%zu", out.size());
    const size_t out_count = static_cast<size_t>(out_w) * out_h * out_c * batch;
    if (out.size() != out_count) {
        set_err(err_buf, err_buf_cap, "output size mismatch");
        return -1;
    }
    std::memcpy(out_nchw, out.data(), out_count * sizeof(float));
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
