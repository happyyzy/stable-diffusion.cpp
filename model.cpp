#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gguf_reader.hpp"
#include "model.h"
#include "stable-diffusion.h"
#include "util.h"
#include "vocab.hpp"
#include "vocab_mistral.hpp"
#include "vocab_qwen.hpp"
#include "vocab_umt5.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "name_conversion.h"
#include "stable-diffusion.h"

#ifdef SD_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef SD_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef SD_USE_OPENCL
#include "ggml-opencl.h"
#endif

#define ST_HEADER_SIZE_LEN 8

uint64_t read_u64(uint8_t* buffer) {
    // little endian
    uint64_t value = 0;
    value |= static_cast<int64_t>(buffer[7]) << 56;
    value |= static_cast<int64_t>(buffer[6]) << 48;
    value |= static_cast<int64_t>(buffer[5]) << 40;
    value |= static_cast<int64_t>(buffer[4]) << 32;
    value |= static_cast<int64_t>(buffer[3]) << 24;
    value |= static_cast<int64_t>(buffer[2]) << 16;
    value |= static_cast<int64_t>(buffer[1]) << 8;
    value |= static_cast<int64_t>(buffer[0]);
    return value;
}

int32_t read_int(uint8_t* buffer) {
    // little endian
    int value = 0;
    value |= buffer[3] << 24;
    value |= buffer[2] << 16;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

uint16_t read_short(uint8_t* buffer) {
    // little endian
    uint16_t value = 0;
    value |= buffer[1] << 8;
    value |= buffer[0];
    return value;
}

/*================================================= Preprocess ==================================================*/

const char* unused_tensors[] = {
    "betas",
    "alphas_cumprod_prev",
    "sqrt_alphas_cumprod",
    "sqrt_one_minus_alphas_cumprod",
    "log_one_minus_alphas_cumprod",
    "sqrt_recip_alphas_cumprod",
    "sqrt_recipm1_alphas_cumprod",
    "posterior_variance",
    "posterior_log_variance_clipped",
    "posterior_mean_coef1",
    "posterior_mean_coef2",
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "cond_stage_model.1.model.text_model.embeddings.position_ids",
    "cond_stage_model.transformer.vision_model.embeddings.position_ids",
    "cond_stage_model.model.logit_scale",
    "conditioner.embedders.0.transformer.text_model.embeddings.position_ids",
    "conditioner.embedders.0.model.logit_scale",
    "conditioner.embedders.1.model.logit_scale",
    "model.diffusion_model.time_embedding.cond_proj.weight",
    "unet.time_embedding.cond_proj.weight",
    "model_ema.decay",
    "model_ema.num_updates",
    "model_ema.diffusion_model",
    "embedding_manager",
    "denoiser.sigmas",
    "text_encoders.t5xxl.transformer.encoder.embed_tokens.weight",  // only used during training
    "ztsnr",                                                        // Found in some SDXL vpred models
    "edm_vpred.sigma_min",                                          // Found in CosXL
    // TODO: find another way to avoid the "unknown tensor" for these two
    // "edm_vpred.sigma_max", // Used to detect CosXL
    // "v_pred", // Used to detect SDXL vpred models
    "text_encoders.llm.output.weight",
    "text_encoders.llm.lm_head.",
    "first_stage_model.bn.",
};

bool is_unused_tensor(std::string name) {
    for (size_t i = 0; i < sizeof(unused_tensors) / sizeof(const char*); i++) {
        if (starts_with(name, unused_tensors[i])) {
            return true;
        }
    }
    return false;
}

uint16_t f8_e4m3_to_f16(uint8_t f8) {
    // do we need to support uz?

    const uint32_t exponent_bias = 7;
    if (f8 == 0xff) {
        return ggml_fp32_to_fp16(-NAN);
    } else if (f8 == 0x7f) {
        return ggml_fp32_to_fp16(NAN);
    }

    uint32_t sign     = f8 & 0x80;
    uint32_t exponent = (f8 & 0x78) >> 3;
    uint32_t mantissa = f8 & 0x07;
    uint32_t result   = sign << 24;
    if (exponent == 0) {
        if (mantissa > 0) {
            exponent = 0x7f - exponent_bias;

            // yes, 2 times
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }
            if ((mantissa & 0x04) == 0) {
                mantissa &= 0x03;
                mantissa <<= 1;
                exponent -= 1;
            }

            result |= (mantissa & 0x03) << 21;
            result |= exponent << 23;
        }
    } else {
        result |= mantissa << 20;
        exponent += 0x7f - exponent_bias;
        result |= exponent << 23;
    }

    return ggml_fp32_to_fp16(*reinterpret_cast<const float*>(&result));
}

uint16_t f8_e5m2_to_f16(uint8_t fp8) {
    uint8_t sign     = (fp8 >> 7) & 0x1;
    uint8_t exponent = (fp8 >> 2) & 0x1F;
    uint8_t mantissa = fp8 & 0x3;

    uint16_t fp16_sign = sign << 15;
    uint16_t fp16_exponent;
    uint16_t fp16_mantissa;

    if (exponent == 0 && mantissa == 0) {  // zero
        return fp16_sign;
    }

    if (exponent == 0x1F) {  // NAN and INF
        fp16_exponent = 0x1F;
        fp16_mantissa = mantissa ? (mantissa << 8) : 0;
        return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
    }

    if (exponent == 0) {  // subnormal numbers
        fp16_mantissa = (mantissa << 8);
        return fp16_sign | fp16_mantissa;
    }

    // normal numbers
    int16_t true_exponent = (int16_t)exponent - 15 + 15;
    if (true_exponent <= 0) {
        fp16_exponent = 0;
        fp16_mantissa = (mantissa << 8);
    } else if (true_exponent >= 0x1F) {
        fp16_exponent = 0x1F;
        fp16_mantissa = 0;
    } else {
        fp16_exponent = (uint16_t)true_exponent;
        fp16_mantissa = mantissa << 8;
    }

    return fp16_sign | (fp16_exponent << 10) | fp16_mantissa;
}

void f8_e4m3_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e4m3_to_f16(src[i]);
    }
}

void f8_e5m2_to_f16_vec(uint8_t* src, uint16_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = n - 1; i >= 0; i--) {
        dst[i] = f8_e5m2_to_f16(src[i]);
    }
}

void f64_to_f32_vec(double* src, float* dst, int64_t n) {
    // support inplace op
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (float)src[i];
    }
}

void i64_to_i32_vec(int64_t* src, int32_t* dst, int64_t n) {
    // support inplace op
    for (int64_t i = 0; i < n; i++) {
        dst[i] = (int32_t)src[i];
    }
}

void convert_tensor(void* src,
                    ggml_type src_type,
                    void* dst,
                    ggml_type dst_type,
                    int nrows,
                    int n_per_row) {
    int n = nrows * n_per_row;
    if (src_type == dst_type) {
        size_t nbytes = n * ggml_type_size(src_type) / ggml_blck_size(src_type);
        memcpy(((char*)dst), ((char*)src), nbytes);
    } else if (src_type == GGML_TYPE_F32) {
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src, dst, 0, nrows, n_per_row, im);
        }
    } else if (dst_type == GGML_TYPE_F32) {
        if (src_type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)src, (float*)dst, n);
        } else {
            auto qtype = ggml_get_type_traits(src_type);
            if (qtype->to_float == nullptr) {
                throw std::runtime_error(sd_format("type %s unsupported for integer quantization: no dequantization available",
                                                   ggml_type_name(src_type)));
            }
            qtype->to_float(src, (float*)dst, n);
        }
    } else {
        // src_type == GGML_TYPE_F16 => dst_type is quantized
        // src_type is quantized => dst_type == GGML_TYPE_F16 or dst_type is quantized
        auto qtype = ggml_get_type_traits(src_type);
        if (qtype->to_float == nullptr) {
            throw std::runtime_error(sd_format("type %s unsupported for integer quantization: no dequantization available",
                                               ggml_type_name(src_type)));
        }
        std::vector<char> buf;
        buf.resize(sizeof(float) * n);
        char* src_data_f32 = buf.data();
        qtype->to_float(src, (float*)src_data_f32, n);
        if (dst_type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)src_data_f32, (ggml_fp16_t*)dst, n);
        } else {
            std::vector<float> imatrix(n_per_row, 1.0f);  // dummy importance matrix
            const float* im = imatrix.data();
            ggml_quantize_chunk(dst_type, (float*)src_data_f32, dst, 0, nrows, n_per_row, im);
        }
    }
}

static bool sd_convert_htp_permute_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SD_GGUF_CONVERT_HTP_PERMUTE");
        enabled         = (env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// Inverse of the HMX 32x32 weight layout permute, used to build host-side CPU
// references from a permuted+repacked GGUF. Mutually exclusive with PERMUTE.
static bool sd_convert_htp_invpermute_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SD_GGUF_CONVERT_HTP_INVPERMUTE");
        enabled         = (env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static bool sd_convert_htp_permute_core_only() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SD_GGUF_CONVERT_HTP_PERMUTE_CORE_ONLY");
        enabled         = (env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static bool sd_convert_htp_permute_log_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SD_GGUF_CONVERT_HTP_PERMUTE_LOG");
        enabled         = (env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

static std::atomic<int>& sd_convert_htp_permute_count() {
    static std::atomic<int> count(0);
    return count;
}

static bool sd_convert_htp_repack_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SD_GGUF_CONVERT_HTP_REPACK");
        enabled         = (env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return enabled != 0;
}

// When enabled, only weights that were permuted in this conversion run will be
// repacked into the HMX packed-quant layout. This keeps small CPU-only matmuls
// (e.g. cap_embed / adaLN) compatible with vanilla ggml execution.
static bool sd_convert_htp_repack_only_permuted() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SD_GGUF_CONVERT_HTP_REPACK_ONLY_PERMUTED");
        if (env != nullptr && env[0] != '\0') {
            enabled = (strcmp(env, "0") != 0) ? 1 : 0;
        } else {
            const char* overlay = getenv("SD_DIFFUSION_MODEL_OVERLAY");
            // Overlay convert starts from an already packed baseline GGUF and replaces
            // only a subset of tensors from the source model. Repacking untouched
            // baseline qweights again corrupts their on-disk layout, so in overlay mode
            // only tensors permuted in this conversion pass may be repacked.
            enabled = (overlay != nullptr && overlay[0] != '\0') ? 1 : 0;
        }
    }
    return enabled != 0;
}

// Optional override for the GGUF metadata flag sd.npu.contract.export.repack.
// This can be used to keep CPU fallback in vanilla mode even if we repack a
// subset of weights for the HMX fast-path.
static bool sd_convert_htp_repack_metadata_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SD_GGUF_CONVERT_HTP_REPACK_META");
        if (env != nullptr && env[0] != '\0') {
            enabled = (strcmp(env, "0") != 0 && strcmp(env, "false") != 0) ? 1 : 0;
        } else {
            enabled = sd_convert_htp_repack_enabled() ? 1 : 0;
        }
    }
    return enabled != 0;
}

static std::atomic<int>& sd_convert_htp_repack_count() {
    static std::atomic<int> count(0);
    return count;
}

static std::vector<std::string> sd_parse_csv_env(const char* name) {
    std::vector<std::string> out;
    const char* env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return out;
    }
    std::string s(env);
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        if (end == std::string::npos) {
            end = s.size();
        }
        size_t b = start;
        while (b < end && std::isspace((unsigned char)s[b])) {
            ++b;
        }
        size_t e = end;
        while (e > b && std::isspace((unsigned char)s[e - 1])) {
            --e;
        }
        if (e > b) {
            out.emplace_back(s.substr(b, e - b));
        }
        start = end + 1;
    }
    return out;
}

static bool sd_name_match_any(const std::string& name, const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (!k.empty() && name.find(k) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static bool sd_convert_htp_permute_name_allowed(const std::string& name) {
    static std::vector<std::string> include = sd_parse_csv_env("SD_GGUF_CONVERT_HTP_PERMUTE_INCLUDE");
    static std::vector<std::string> exclude = sd_parse_csv_env("SD_GGUF_CONVERT_HTP_PERMUTE_EXCLUDE");

    if (!include.empty() && !sd_name_match_any(name, include)) {
        return false;
    }
    if (!exclude.empty() && sd_name_match_any(name, exclude)) {
        return false;
    }
    return true;
}

static bool sd_convert_should_hmx_permute(const TensorStorage& tensor_storage, const ggml_tensor* dst_tensor) {
    const bool do_perm = sd_convert_htp_permute_enabled();
    const bool do_inv  = sd_convert_htp_invpermute_enabled();
    if (do_perm && do_inv) {
        LOG_ERROR("SD_GGUF_CONVERT_HTP_PERMUTE and SD_GGUF_CONVERT_HTP_INVPERMUTE are both enabled; pick one");
        return false;
    }
    if (!do_perm && !do_inv) {
        return false;
    }
    if (dst_tensor == nullptr) {
        return false;
    }
    const bool supported_dst_type =
        ggml_is_quantized(dst_tensor->type) ||
        dst_tensor->type == GGML_TYPE_F16 ||
        dst_tensor->type == GGML_TYPE_BF16 ||
        dst_tensor->type == GGML_TYPE_F32;
    if (!supported_dst_type) {
        return false;
    }
    if (tensor_storage.n_dims != 2) {
        return false;
    }
    const int64_t k = tensor_storage.ne[0];
    const int64_t n = tensor_storage.ne[1];
    if ((n % 32) != 0) {
        return false;
    }
    if (do_inv) {
        // For INVPERMUTE, only handle quantized weights. In our diffusion exports, fp16 tensors are
        // kept in vanilla basis (CPU-friendly) and should not be unpermuted.
        if (!ggml_is_quantized(tensor_storage.type)) {
            return false;
        }
        if ((k % 256) != 0) {
            return false;
        }
    } else {
        // Match HTP matmul constraints for the target weight dtype:
        // - Quant (q4/q8/iq4): k % 256 == 0, n % 32 == 0
        // - F16 weight matmul: k % 32 == 0, n % 32 == 0
        if (ggml_is_quantized(dst_tensor->type)) {
            if ((k % 256) != 0) {
                return false;
            }
        } else if (dst_tensor->type == GGML_TYPE_F16) {
            if ((k % 32) != 0) {
                return false;
            }
        } else {
            // NOTE: HTP offload currently consumes permuted weights for quant/f16 matmul only.
            // Keep other dtypes conservative to avoid breaking CPU fallback correctness.
            return false;
        }
    }
    if (ends_with(tensor_storage.name, ".bias") || ends_with(tensor_storage.name, ".scale")) {
        return false;
    }
    if (sd_convert_htp_permute_core_only()) {
        const std::string& name = tensor_storage.name;
        const bool core_linear =
            ends_with(name, ".attention.qkv.weight") ||
            ends_with(name, ".attention.out.weight") ||
            ends_with(name, ".feed_forward.w1.weight") ||
            ends_with(name, ".feed_forward.w2.weight") ||
            ends_with(name, ".feed_forward.w3.weight");
        if (!core_linear) {
            return false;
        }
    }
    if (!sd_convert_htp_permute_name_allowed(tensor_storage.name)) {
        return false;
    }
    return true;
}

static void sd_hmx_permute_linear_weight_f32(const float* src, float* dst, int n, int k) {
    const int n_chunks = n / 32;
    const int k_chunks = k / 32;
    const int k_stride = k_chunks * 32;

    for (int r = 0; r < n; ++r) {
        const int64_t row_base = (int64_t)r * k_stride;
        for (int c = 0; c < k_stride; ++c) {
            int64_t t = row_base + c;
            int u     = (int)(t & 1);
            t >>= 1;
            int i = (int)(t & 31);
            t >>= 5;
            int g = (int)(t & 15);
            t >>= 4;
            int b = (int)(t % k_chunks);
            int a = (int)(t / k_chunks);

            if (a < 0 || a >= n_chunks) {
                continue;
            }

            int src_row = a * 32 + i;
            int src_col = b * 32 + g * 2 + u;
            dst[row_base + c] = src[(int64_t)src_row * k + src_col];
        }
    }
}

// Inverse mapping of sd_hmx_permute_linear_weight_f32():
// - src: permuted basis (as stored for HMX matmul)
// - dst: vanilla basis (row-major [n,k])
static void sd_hmx_invpermute_linear_weight_f32(const float* src, float* dst, int n, int k) {
    const int n_chunks = n / 32;
    const int k_chunks = k / 32;
    const int k_stride = k_chunks * 32;

    for (int r = 0; r < n; ++r) {
        const int64_t row_base = (int64_t)r * k_stride;
        for (int c = 0; c < k_stride; ++c) {
            int64_t t = row_base + c;
            int u     = (int)(t & 1);
            t >>= 1;
            int i = (int)(t & 31);
            t >>= 5;
            int g = (int)(t & 15);
            t >>= 4;
            int b = (int)(t % k_chunks);
            int a = (int)(t / k_chunks);

            if (a < 0 || a >= n_chunks) {
                continue;
            }

            int src_row = a * 32 + i;
            int src_col = b * 32 + g * 2 + u;
            dst[(int64_t)src_row * k + src_col] = src[row_base + c];
        }
    }
}

struct sd_std_block_q4_like {
    uint16_t d;
    uint8_t  qs[16];
} __attribute__((packed));
static_assert(sizeof(sd_std_block_q4_like) == 18, "unexpected q4-like block size");

struct sd_std_block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
} __attribute__((packed));
static_assert(sizeof(sd_std_block_q8_0) == 34, "unexpected q8_0 block size");

struct sd_my_block_q4_0_like {
    uint16_t scales[8];
    uint8_t  quants[8 * 16];
} __attribute__((packed));
static_assert(sizeof(sd_my_block_q4_0_like) == 144, "unexpected packed q4-like super-block size");

struct sd_my_block_q8_0 {
    uint16_t scales[8];
    int8_t   quants[8 * 32];
} __attribute__((packed));
static_assert(sizeof(sd_my_block_q8_0) == 272, "unexpected packed q8_0 super-block size");

static void sd_hmx_repack_row_q4_like_inplace(uint8_t* row, size_t n_super_blocks) {
    for (size_t sb = 0; sb < n_super_blocks; ++sb) {
        const size_t src_off = sb * 8 * sizeof(sd_std_block_q4_like);
        const auto* src_ptr  = reinterpret_cast<const sd_std_block_q4_like*>(row + src_off);

        sd_std_block_q4_like src_blocks[8];
        std::memcpy(src_blocks, src_ptr, sizeof(src_blocks));

        sd_my_block_q4_0_like dst_block{};
        uint8_t unpacked_qs[256];
        for (size_t bi = 0; bi < 8; ++bi) {
            dst_block.scales[bi] = src_blocks[bi].d;
            for (size_t j = 0; j < sizeof(src_blocks[bi].qs); ++j) {
                uint8_t q = src_blocks[bi].qs[j];
                unpacked_qs[bi * 32 + j + 0]  = q & 0x0F;
                unpacked_qs[bi * 32 + j + 16] = q >> 4;
            }
        }
        for (size_t j = 0; j < sizeof(dst_block.quants) / 2; ++j) {
            dst_block.quants[j * 2 + 0] = (uint8_t)((unpacked_qs[j + 128] << 4) | unpacked_qs[j + 0]);
            dst_block.quants[j * 2 + 1] = (uint8_t)((unpacked_qs[j + 192] << 4) | unpacked_qs[j + 64]);
        }

        const size_t dst_off = sb * sizeof(sd_my_block_q4_0_like);
        std::memcpy(row + dst_off, &dst_block, sizeof(dst_block));
    }
}

static void sd_hmx_unrepack_row_q4_like_inplace(uint8_t* row, size_t n_super_blocks) {
    for (size_t sb = 0; sb < n_super_blocks; ++sb) {
        const size_t src_off = sb * sizeof(sd_my_block_q4_0_like);
        const auto* src_ptr  = reinterpret_cast<const sd_my_block_q4_0_like*>(row + src_off);

        sd_my_block_q4_0_like src_block{};
        std::memcpy(&src_block, src_ptr, sizeof(src_block));

        uint8_t unpacked_qs[256];
        for (size_t j = 0; j < sizeof(src_block.quants) / 2; ++j) {
            const uint8_t q0 = src_block.quants[j * 2 + 0];
            const uint8_t q1 = src_block.quants[j * 2 + 1];

            unpacked_qs[j + 0]   = q0 & 0x0F;
            unpacked_qs[j + 128] = q0 >> 4;
            unpacked_qs[j + 64]  = q1 & 0x0F;
            unpacked_qs[j + 192] = q1 >> 4;
        }

        sd_std_block_q4_like dst_blocks[8];
        for (size_t bi = 0; bi < 8; ++bi) {
            dst_blocks[bi].d = src_block.scales[bi];
            for (size_t j = 0; j < sizeof(dst_blocks[bi].qs); ++j) {
                const uint8_t lo = unpacked_qs[bi * 32 + j + 0];
                const uint8_t hi = unpacked_qs[bi * 32 + j + 16];
                dst_blocks[bi].qs[j] = (uint8_t)((hi << 4) | lo);
            }
        }

        const size_t dst_off = sb * 8 * sizeof(sd_std_block_q4_like);
        std::memcpy(row + dst_off, dst_blocks, sizeof(dst_blocks));
    }
}

static void sd_hmx_repack_row_q8_0_inplace(uint8_t* row, size_t n_super_blocks) {
    for (size_t sb = 0; sb < n_super_blocks; ++sb) {
        const size_t src_off = sb * 8 * sizeof(sd_std_block_q8_0);
        const auto* src_ptr  = reinterpret_cast<const sd_std_block_q8_0*>(row + src_off);

        sd_std_block_q8_0 src_blocks[8];
        std::memcpy(src_blocks, src_ptr, sizeof(src_blocks));

        sd_my_block_q8_0 dst_block{};
        for (size_t i = 0; i < 8; ++i) {
            dst_block.scales[i] = src_blocks[i].d;
            std::memcpy(dst_block.quants + i * sizeof(src_blocks[i].qs), src_blocks[i].qs, sizeof(src_blocks[i].qs));
        }

        const size_t dst_off = sb * sizeof(sd_my_block_q8_0);
        std::memcpy(row + dst_off, &dst_block, sizeof(dst_block));
    }
}

static void sd_hmx_unrepack_row_q8_0_inplace(uint8_t* row, size_t n_super_blocks) {
    for (size_t sb = 0; sb < n_super_blocks; ++sb) {
        const size_t src_off = sb * sizeof(sd_my_block_q8_0);
        const auto* src_ptr  = reinterpret_cast<const sd_my_block_q8_0*>(row + src_off);

        sd_my_block_q8_0 src_block{};
        std::memcpy(&src_block, src_ptr, sizeof(src_block));

        sd_std_block_q8_0 dst_blocks[8];
        for (size_t bi = 0; bi < 8; ++bi) {
            dst_blocks[bi].d = src_block.scales[bi];
            std::memcpy(dst_blocks[bi].qs,
                        src_block.quants + bi * sizeof(dst_blocks[bi].qs),
                        sizeof(dst_blocks[bi].qs));
        }

        const size_t dst_off = sb * 8 * sizeof(sd_std_block_q8_0);
        std::memcpy(row + dst_off, dst_blocks, sizeof(dst_blocks));
    }
}

static void sd_hmx_repack_quant_weight_inplace_if_needed(char* data, ggml_type type, int nrows, int n_per_row, const std::string& name) {
    if (!sd_convert_htp_repack_enabled()) {
        return;
    }
    const bool is_q4_like = type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL;
    const bool is_q8_0    = type == GGML_TYPE_Q8_0;
    if (!is_q4_like && !is_q8_0) {
        return;
    }
    if (n_per_row <= 0 || (n_per_row % 256) != 0) {
        return;
    }
    const size_t n_super_blocks = (size_t)n_per_row / 256;
    const size_t row_size       = ggml_row_size(type, n_per_row);
    const size_t expected_row_size =
        is_q4_like ? n_super_blocks * sizeof(sd_my_block_q4_0_like) : n_super_blocks * sizeof(sd_my_block_q8_0);
    if (row_size != expected_row_size) {
        if (sd_convert_htp_permute_log_enabled()) {
            LOG_WARN("hmx repack skip (row-size mismatch): %s type=%s row=%zu expect=%zu",
                     name.c_str(), ggml_type_name(type), row_size, expected_row_size);
        }
        return;
    }
    for (int r = 0; r < nrows; ++r) {
        uint8_t* row = reinterpret_cast<uint8_t*>(data + (size_t)r * row_size);
        if (is_q4_like) {
            sd_hmx_repack_row_q4_like_inplace(row, n_super_blocks);
        } else {
            sd_hmx_repack_row_q8_0_inplace(row, n_super_blocks);
        }
    }
    const int repack_id = sd_convert_htp_repack_count().fetch_add(1) + 1;
    if (sd_convert_htp_permute_log_enabled() && repack_id <= 128) {
        LOG_INFO("hmx repack #%d: %s type=%s shape=[%d,%d]", repack_id, name.c_str(), ggml_type_name(type), n_per_row, nrows);
    }
}

static void sd_hmx_unrepack_quant_weight_inplace_if_needed(char* data, ggml_type type, int nrows, int n_per_row, const std::string& name) {
    // Unrepack is only used in GGUF conversion when reading a mixed-layout model exported
    // with "repack-only-permuted": permuted 2D weights are repacked for HMX fast-path,
    // while 1D weights remain vanilla ggml layout for CPU correctness.
    const bool is_q4_like = type == GGML_TYPE_Q4_0 || type == GGML_TYPE_IQ4_NL;
    const bool is_q8_0    = type == GGML_TYPE_Q8_0;
    if (!is_q4_like && !is_q8_0) {
        return;
    }
    if (n_per_row <= 0 || (n_per_row % 256) != 0) {
        return;
    }
    const size_t n_super_blocks = (size_t)n_per_row / 256;
    const size_t row_size       = ggml_row_size(type, n_per_row);
    const size_t expected_row_size =
        is_q4_like ? n_super_blocks * sizeof(sd_my_block_q4_0_like) : n_super_blocks * sizeof(sd_my_block_q8_0);
    if (row_size != expected_row_size) {
        if (sd_convert_htp_permute_log_enabled()) {
            LOG_WARN("hmx unrepack skip (row-size mismatch): %s type=%s row=%zu expect=%zu",
                     name.c_str(), ggml_type_name(type), row_size, expected_row_size);
        }
        return;
    }
    for (int r = 0; r < nrows; ++r) {
        uint8_t* row = reinterpret_cast<uint8_t*>(data + (size_t)r * row_size);
        if (is_q4_like) {
            sd_hmx_unrepack_row_q4_like_inplace(row, n_super_blocks);
        } else {
            sd_hmx_unrepack_row_q8_0_inplace(row, n_super_blocks);
        }
    }
    if (sd_convert_htp_permute_log_enabled()) {
        LOG_INFO("hmx unrepack: %s type=%s shape=[%d,%d]", name.c_str(), ggml_type_name(type), n_per_row, nrows);
    }
}

/*================================================= ModelLoader ==================================================*/

void ModelLoader::add_tensor_storage(const TensorStorage& tensor_storage) {
    tensor_storage_map[tensor_storage.name] = tensor_storage;
}

void ModelLoader::overlay_tensor_storage(const TensorStorage& tensor_storage, const std::string& file_path) {
    size_t file_index = 0;
    auto it = std::find(file_paths_.begin(), file_paths_.end(), file_path);
    if (it == file_paths_.end()) {
        file_paths_.push_back(file_path);
        file_index = file_paths_.size() - 1;
    } else {
        file_index = static_cast<size_t>(std::distance(file_paths_.begin(), it));
    }

    TensorStorage remapped = tensor_storage;
    remapped.file_index = file_index;
    add_tensor_storage(remapped);
}

bool is_zip_file(const std::string& file_path) {
    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == nullptr) {
        return false;
    }
    zip_close(zip);
    return true;
}

bool is_gguf_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[4];

    file.read(magic, sizeof(magic));
    if (!file) {
        return false;
    }
    for (uint32_t i = 0; i < sizeof(magic); i++) {
        if (magic[i] != GGUF_MAGIC[i]) {
            return false;
        }
    }

    return true;
}

bool is_safetensors_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        return false;
    }

    size_t header_size_ = read_u64(header_size_buf);
    if (header_size_ >= file_size_ || header_size_ <= 2) {
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        return false;
    }
    nlohmann::json header_ = nlohmann::json::parse(header_buf.data());
    if (header_.is_discarded()) {
        return false;
    }
    return true;
}

bool ModelLoader::init_from_file(const std::string& file_path, const std::string& prefix) {
    if (is_directory(file_path)) {
        LOG_INFO("load %s using diffusers format", file_path.c_str());
        return init_from_diffusers_file(file_path, prefix);
    } else if (is_gguf_file(file_path)) {
        LOG_INFO("load %s using gguf format", file_path.c_str());
        return init_from_gguf_file(file_path, prefix);
    } else if (is_safetensors_file(file_path)) {
        LOG_INFO("load %s using safetensors format", file_path.c_str());
        return init_from_safetensors_file(file_path, prefix);
    } else if (is_zip_file(file_path)) {
        LOG_INFO("load %s using checkpoint format", file_path.c_str());
        return init_from_ckpt_file(file_path, prefix);
    } else {
        if (file_exists(file_path)) {
            LOG_WARN("unknown format %s", file_path.c_str());
        } else {
            LOG_WARN("file %s not found", file_path.c_str());
        }
        return false;
    }
}

void ModelLoader::convert_tensors_name() {
    SDVersion version = (version_ == VERSION_COUNT) ? get_sd_version() : version_;
    String2TensorStorage new_map;

    for (auto& [_, tensor_storage] : tensor_storage_map) {
        auto new_name = convert_tensor_name(tensor_storage.name, version);
        // LOG_DEBUG("%s -> %s", tensor_storage.name.c_str(), new_name.c_str());
        tensor_storage.name = new_name;
        new_map[new_name]   = std::move(tensor_storage);
    }

    tensor_storage_map.swap(new_map);
}

bool ModelLoader::init_from_file_and_convert_name(const std::string& file_path, const std::string& prefix, SDVersion version) {
    if (version_ == VERSION_COUNT && version != VERSION_COUNT) {
        version_ = version;
    }
    if (!init_from_file(file_path, prefix)) {
        return false;
    }
    convert_tensors_name();
    return true;
}

/*================================================= GGUFModelLoader ==================================================*/

static bool sd_env_truthy(const char* key, bool default_value = false) {
    const char* env = std::getenv(key);
    if (env == nullptr || env[0] == '\0') {
        return default_value;
    }
    return std::strcmp(env, "0") != 0;
}

static void model_set_env_overwrite(const char* key, const std::string& value) {
#ifdef _WIN32
    _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}

static std::string gguf_scalar_to_string(const gguf_context* ctx, int64_t key_id, gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8: return std::to_string(gguf_get_val_u8(ctx, key_id));
        case GGUF_TYPE_INT8: return std::to_string(gguf_get_val_i8(ctx, key_id));
        case GGUF_TYPE_UINT16: return std::to_string(gguf_get_val_u16(ctx, key_id));
        case GGUF_TYPE_INT16: return std::to_string(gguf_get_val_i16(ctx, key_id));
        case GGUF_TYPE_UINT32: return std::to_string(gguf_get_val_u32(ctx, key_id));
        case GGUF_TYPE_INT32: return std::to_string(gguf_get_val_i32(ctx, key_id));
        case GGUF_TYPE_FLOAT32: return std::to_string(gguf_get_val_f32(ctx, key_id));
        case GGUF_TYPE_UINT64: return std::to_string(gguf_get_val_u64(ctx, key_id));
        case GGUF_TYPE_INT64: return std::to_string(gguf_get_val_i64(ctx, key_id));
        case GGUF_TYPE_FLOAT64: return std::to_string(gguf_get_val_f64(ctx, key_id));
        case GGUF_TYPE_BOOL: return gguf_get_val_bool(ctx, key_id) ? "true" : "false";
        case GGUF_TYPE_STRING: {
            const char* s = gguf_get_val_str(ctx, key_id);
            return s ? std::string(s) : std::string();
        }
        default:
            return std::string("<unsupported>");
    }
}

static void model_loader_read_gguf_metadata(const gguf_context* ctx, std::map<std::string, std::string>& out) {
    out.clear();
    if (ctx == nullptr) {
        return;
    }
    const int64_t n_kv = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        if (key == nullptr || key[0] == '\0') {
            continue;
        }
        const gguf_type type = gguf_get_kv_type(ctx, i);
        if (type == GGUF_TYPE_ARRAY) {
            continue;
        }
        out[std::string(key)] = gguf_scalar_to_string(ctx, i, type);
    }
}

static bool model_loader_validate_npu_contract(const std::map<std::string, std::string>& metadata,
                                               const std::string& file_path) {
    const char* expected_env = std::getenv("GGML_HTP_CONTRACT_EXPECT");
    if (expected_env == nullptr || expected_env[0] == '\0') {
        return true;
    }

    const bool strict = sd_env_truthy("GGML_HTP_CONTRACT_STRICT", false);
    auto it = metadata.find("sd.npu.contract.id");
    if (it == metadata.end()) {
        model_set_env_overwrite("GGML_HTP_CONTRACT_ACTIVE", "<missing>");
        if (strict) {
            LOG_ERROR("GGUF '%s' missing required key sd.npu.contract.id (expected '%s')", file_path.c_str(), expected_env);
            return false;
        }
        LOG_WARN("GGUF '%s' has no sd.npu.contract.id; expected '%s'", file_path.c_str(), expected_env);
        return true;
    }

    if (it->second != expected_env) {
        model_set_env_overwrite("GGML_HTP_CONTRACT_ACTIVE", it->second);
        if (strict) {
            LOG_ERROR("GGUF '%s' contract mismatch: got '%s', expected '%s'", file_path.c_str(), it->second.c_str(),
                      expected_env);
            return false;
        }
        LOG_WARN("GGUF '%s' contract mismatch: got '%s', expected '%s'", file_path.c_str(), it->second.c_str(),
                 expected_env);
        return true;
    }

    model_set_env_overwrite("GGML_HTP_CONTRACT_ACTIVE", it->second);
    LOG_INFO("GGUF '%s' contract id = %s", file_path.c_str(), it->second.c_str());
    return true;
}

static void model_loader_write_npu_contract_metadata(gguf_context* gguf_ctx) {
    if (gguf_ctx == nullptr) {
        return;
    }

    const char* contract_id = std::getenv("SD_NPU_CONTRACT_ID");
    if (contract_id == nullptr || contract_id[0] == '\0') {
        contract_id = "sdcpp.hmx.v1";
    }

    gguf_set_val_str(gguf_ctx, "sd.npu.contract.id", contract_id);
    gguf_set_val_str(gguf_ctx, "sd.npu.contract.layout.matmul.f16", "permute_32x32.v1");
    gguf_set_val_str(gguf_ctx, "sd.npu.contract.layout.matmul.q4_0", "common_deq.v1");
    gguf_set_val_str(gguf_ctx, "sd.npu.contract.layout.matmul.q8_0", "common_deq.v1");
    gguf_set_val_str(gguf_ctx, "sd.npu.contract.layout.matmul.iq4_nl", "common_deq.v1");
    gguf_set_val_str(gguf_ctx, "sd.npu.contract.layout.flash_attn_ext", "f32_q_f16_kv.v1");
    gguf_set_val_bool(gguf_ctx, "sd.npu.contract.export.permute", sd_convert_htp_permute_enabled());
    gguf_set_val_bool(gguf_ctx, "sd.npu.contract.export.repack", sd_convert_htp_repack_metadata_enabled());
}

static void model_loader_write_existing_npu_contract_metadata(gguf_context* gguf_ctx,
                                                              const std::map<std::string, std::string>& metadata) {
    if (gguf_ctx == nullptr) {
        return;
    }

    auto write_str = [&](const char* key) {
        auto it = metadata.find(key);
        if (it != metadata.end()) {
            gguf_set_val_str(gguf_ctx, key, it->second.c_str());
        }
    };
    auto write_bool = [&](const char* key) {
        auto it = metadata.find(key);
        if (it != metadata.end()) {
            gguf_set_val_bool(gguf_ctx, key, it->second == "true");
        }
    };

    write_str("sd.npu.contract.id");
    write_str("sd.npu.contract.layout.matmul.f16");
    write_str("sd.npu.contract.layout.matmul.q4_0");
    write_str("sd.npu.contract.layout.matmul.q8_0");
    write_str("sd.npu.contract.layout.matmul.iq4_nl");
    write_str("sd.npu.contract.layout.flash_attn_ext");
    write_bool("sd.npu.contract.export.permute");
    write_bool("sd.npu.contract.export.repack");
}

bool ModelLoader::init_from_gguf_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;
    gguf_metadata_.clear();

    gguf_context* ctx_gguf_ = nullptr;
    ggml_context* ctx_meta_ = nullptr;

    ctx_gguf_ = gguf_init_from_file(file_path.c_str(), {true, &ctx_meta_});
    if (!ctx_gguf_) {
        LOG_ERROR("failed to open '%s' with gguf_init_from_file. Try to open it with GGUFReader.", file_path.c_str());
        GGUFReader gguf_reader;
        if (!gguf_reader.load(file_path)) {
            LOG_ERROR("failed to open '%s' with GGUFReader.", file_path.c_str());
            return false;
        }

        size_t data_offset = gguf_reader.data_offset();
        for (const auto& gguf_tensor_info : gguf_reader.tensors()) {
            std::string name = gguf_tensor_info.name;
            if (!starts_with(name, prefix)) {
                name = prefix + name;
            }

            TensorStorage tensor_storage(
                name,
                gguf_tensor_info.type,
                gguf_tensor_info.shape.data(),
                static_cast<int>(gguf_tensor_info.shape.size()),
                file_index,
                data_offset + gguf_tensor_info.offset);

            // LOG_DEBUG("%s %s", name.c_str(), tensor_storage.to_string().c_str());

            add_tensor_storage(tensor_storage);
        }

        LOG_WARN("loaded GGUF tensors via fallback reader; GGUF metadata is not available for contract validation");

        return true;
    }

    model_loader_read_gguf_metadata(ctx_gguf_, gguf_metadata_);
    auto cid_it = gguf_metadata_.find("sd.npu.contract.id");
    if (cid_it != gguf_metadata_.end()) {
        model_set_env_overwrite("GGML_HTP_CONTRACT_ACTIVE", cid_it->second);
    } else {
        model_set_env_overwrite("GGML_HTP_CONTRACT_ACTIVE", "<missing>");
    }

    // Export contract booleans are needed for CPU fallback to interpret prepacked
    // qweights correctly (e.g. hmx repack layout).
    //
    // NOTE: model_loader_read_gguf_metadata() stores bools as "true"/"false".
    const auto perm_it = gguf_metadata_.find("sd.npu.contract.export.permute");
    if (perm_it != gguf_metadata_.end()) {
        model_set_env_overwrite("GGML_HTP_CONTRACT_EXPORT_PERMUTE", perm_it->second == "true" ? "1" : "0");
    }
    const auto repack_it = gguf_metadata_.find("sd.npu.contract.export.repack");
    if (repack_it != gguf_metadata_.end()) {
        model_set_env_overwrite("GGML_HTP_CONTRACT_EXPORT_REPACK", repack_it->second == "true" ? "1" : "0");
    }
    if (!model_loader_validate_npu_contract(gguf_metadata_, file_path)) {
        gguf_free(ctx_gguf_);
        ggml_free(ctx_meta_);
        return false;
    }

    int n_tensors = static_cast<int>(gguf_get_n_tensors(ctx_gguf_));

    size_t total_size  = 0;
    size_t data_offset = gguf_get_data_offset(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name          = gguf_get_tensor_name(ctx_gguf_, i);
        struct ggml_tensor* dummy = ggml_get_tensor(ctx_meta_, name.c_str());
        size_t offset             = data_offset + gguf_get_tensor_offset(ctx_gguf_, i);

        // LOG_DEBUG("%s", name.c_str());

        if (!starts_with(name, prefix)) {
            name = prefix + name;
        }

        TensorStorage tensor_storage(name, dummy->type, dummy->ne, ggml_n_dims(dummy), file_index, offset);

        GGML_ASSERT(ggml_nbytes(dummy) == tensor_storage.nbytes());

        add_tensor_storage(tensor_storage);
    }

    gguf_free(ctx_gguf_);
    ggml_free(ctx_meta_);

    return true;
}

/*================================================= SafeTensorsModelLoader ==================================================*/

ggml_type str_to_ggml_type(const std::string& dtype) {
    ggml_type ttype = GGML_TYPE_COUNT;
    if (dtype == "F16") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "BF16") {
        ttype = GGML_TYPE_BF16;
    } else if (dtype == "F32") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F64") {
        ttype = GGML_TYPE_F32;
    } else if (dtype == "F8_E4M3") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "F8_E5M2") {
        ttype = GGML_TYPE_F16;
    } else if (dtype == "I64") {
        ttype = GGML_TYPE_I32;
    }
    return ttype;
}

// https://huggingface.co/docs/safetensors/index
bool ModelLoader::init_from_safetensors_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s', prefix = '%s'", file_path.c_str(), prefix.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    // get file size
    file.seekg(0, file.end);
    size_t file_size_ = file.tellg();
    file.seekg(0, file.beg);

    // read header size
    if (file_size_ <= ST_HEADER_SIZE_LEN) {
        LOG_ERROR("invalid safetensor file '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    uint8_t header_size_buf[ST_HEADER_SIZE_LEN];
    file.read((char*)header_size_buf, ST_HEADER_SIZE_LEN);
    if (!file) {
        LOG_ERROR("read safetensors header size failed: '%s'", file_path.c_str());
        return false;
    }

    size_t header_size_ = read_u64(header_size_buf);
    if (header_size_ >= file_size_) {
        LOG_ERROR("invalid safetensor file '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    // read header
    std::vector<char> header_buf;
    header_buf.resize(header_size_ + 1);
    header_buf[header_size_] = '\0';
    file.read(header_buf.data(), header_size_);
    if (!file) {
        LOG_ERROR("read safetensors header failed: '%s'", file_path.c_str());
        file_paths_.pop_back();
        return false;
    }

    nlohmann::json header_ = nlohmann::json::parse(header_buf.data());

    for (auto& item : header_.items()) {
        std::string name           = item.key();
        nlohmann::json tensor_info = item.value();
        // LOG_DEBUG("%s %s\n", name.c_str(), tensor_info.dump().c_str());

        if (name == "__metadata__") {
            continue;
        }

        if (is_unused_tensor(name)) {
            continue;
        }

        std::string dtype    = tensor_info["dtype"];
        nlohmann::json shape = tensor_info["shape"];

        if (dtype == "U8") {
            continue;
        }

        size_t begin = tensor_info["data_offsets"][0].get<size_t>();
        size_t end   = tensor_info["data_offsets"][1].get<size_t>();

        ggml_type type = str_to_ggml_type(dtype);
        if (type == GGML_TYPE_COUNT) {
            LOG_ERROR("unsupported dtype '%s' (tensor '%s')", dtype.c_str(), name.c_str());
            return false;
        }

        if (shape.size() > SD_MAX_DIMS) {
            LOG_ERROR("invalid tensor '%s'", name.c_str());
            return false;
        }

        int n_dims              = (int)shape.size();
        int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};
        for (int i = 0; i < n_dims; i++) {
            ne[i] = shape[i].get<int64_t>();
        }

        if (n_dims == 5) {
            n_dims = 4;
            ne[0]  = ne[0] * ne[1];
            ne[1]  = ne[2];
            ne[2]  = ne[3];
            ne[3]  = ne[4];
        }

        // ggml_n_dims returns 1 for scalars
        if (n_dims == 0) {
            n_dims = 1;
        }

        if (!starts_with(name, prefix)) {
            name = prefix + name;
        }

        TensorStorage tensor_storage(name, type, ne, n_dims, file_index, ST_HEADER_SIZE_LEN + header_size_ + begin);
        tensor_storage.reverse_ne();

        size_t tensor_data_size = end - begin;

        if (dtype == "F8_E4M3") {
            tensor_storage.is_f8_e4m3 = true;
            // f8 -> f16
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F8_E5M2") {
            tensor_storage.is_f8_e5m2 = true;
            // f8 -> f16
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size * 2);
        } else if (dtype == "F64") {
            tensor_storage.is_f64 = true;
            // f64 -> f32
            GGML_ASSERT(tensor_storage.nbytes() * 2 == tensor_data_size);
        } else if (dtype == "I64") {
            tensor_storage.is_i64 = true;
            // i64 -> i32
            GGML_ASSERT(tensor_storage.nbytes() * 2 == tensor_data_size);
        } else {
            GGML_ASSERT(tensor_storage.nbytes() == tensor_data_size);
        }

        add_tensor_storage(tensor_storage);

        // LOG_DEBUG("%s %s", tensor_storage.to_string().c_str(), dtype.c_str());
    }

    return true;
}

/*================================================= DiffusersModelLoader ==================================================*/

bool ModelLoader::init_from_diffusers_file(const std::string& file_path, const std::string& prefix) {
    std::string unet_path   = path_join(file_path, "unet/diffusion_pytorch_model.safetensors");
    std::string vae_path    = path_join(file_path, "vae/diffusion_pytorch_model.safetensors");
    std::string clip_path   = path_join(file_path, "text_encoder/model.safetensors");
    std::string clip_g_path = path_join(file_path, "text_encoder_2/model.safetensors");

    if (!init_from_safetensors_file(unet_path, "unet.")) {
        return false;
    }

    if (!init_from_safetensors_file(vae_path, "vae.")) {
        LOG_WARN("Couldn't find working VAE in %s", file_path.c_str());
        // return false;
    }
    if (!init_from_safetensors_file(clip_path, "te.")) {
        LOG_WARN("Couldn't find working text encoder in %s", file_path.c_str());
        // return false;
    }
    if (!init_from_safetensors_file(clip_g_path, "te.1.")) {
        LOG_DEBUG("Couldn't find working second text encoder in %s", file_path.c_str());
    }
    return true;
}

/*================================================= CkptModelLoader ==================================================*/

// $ python -m pickletools sd-v1-4/archive/data.pkl | head -n 100
//     0: \x80 PROTO      2
//     2: }    EMPTY_DICT
//     3: q    BINPUT     0
//     5: (    MARK
//     6: X        BINUNICODE 'epoch'
//    16: q        BINPUT     1
//    18: K        BININT1    6
//    20: X        BINUNICODE 'global_step'
//    36: q        BINPUT     2
//    38: J        BININT     470000
//    43: X        BINUNICODE 'pytorch-lightning_version'
//    73: q        BINPUT     3
//    75: X        BINUNICODE '1.4.2'
//    85: q        BINPUT     4
//    87: X        BINUNICODE 'state_dict'
//   102: q        BINPUT     5
//   104: }        EMPTY_DICT
//   105: q        BINPUT     6
//   107: (        MARK
//   108: X            BINUNICODE 'betas'
//   118: q            BINPUT     7
//   120: c            GLOBAL     'torch._utils _rebuild_tensor_v2'
//   153: q            BINPUT     8
//   155: (            MARK
//   156: (                MARK
//   157: X                    BINUNICODE 'storage'
//   169: q                    BINPUT     9
//   171: c                    GLOBAL     'torch FloatStorage'
//   191: q                    BINPUT     10
//   193: X                    BINUNICODE '0'
//   199: q                    BINPUT     11
//   201: X                    BINUNICODE 'cpu'
//   209: q                    BINPUT     12
//   211: M                    BININT2    1000
//   214: t                    TUPLE      (MARK at 156)
//   215: q                BINPUT     13
//   217: Q                BINPERSID
//   218: K                BININT1    0
//   220: M                BININT2    1000
//  ...............................
//  3201: q            BINPUT     250
//  3203: R            REDUCE
//  3204: q            BINPUT     251
//  3206: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.weight'
//  3264: q            BINPUT     252
//  3266: h            BINGET     8
//  3268: (            MARK
//  3269: (                MARK
//  3270: h                    BINGET     9
//  3272: h                    BINGET     10
//  3274: X                    BINUNICODE '30'
//  3281: q                    BINPUT     253
//  3283: h                    BINGET     12
//  3285: J                    BININT     102400
//  3290: t                    TUPLE      (MARK at 3269)
//  3291: q                BINPUT     254
//  3293: Q                BINPERSID
//  3294: K                BININT1    0
//  3296: (                MARK
//  3297: M                    BININT2    320
//  3300: M                    BININT2    320
//  3303: K                    BININT1    1
//  3305: K                    BININT1    1
//  3307: t                    TUPLE      (MARK at 3296)
//  3308: q                BINPUT     255
//  3310: (                MARK
//  3311: M                    BININT2    320
//  3314: K                    BININT1    1
//  3316: K                    BININT1    1
//  3318: K                    BININT1    1
//  3320: t                    TUPLE      (MARK at 3310)
//  3321: r                LONG_BINPUT 256
//  3326: \x89             NEWFALSE
//  3327: h                BINGET     16
//  3329: )                EMPTY_TUPLE
//  3330: R                REDUCE
//  3331: r                LONG_BINPUT 257
//  3336: t                TUPLE      (MARK at 3268)
//  3337: r            LONG_BINPUT 258
//  3342: R            REDUCE
//  3343: r            LONG_BINPUT 259
//  3348: X            BINUNICODE 'model.diffusion_model.input_blocks.1.1.proj_in.bias'
//  3404: r            LONG_BINPUT 260
//  3409: h            BINGET     8
//  3411: (            MARK
//  3412: (                MARK
//  3413: h                    BINGET     9
//  3415: h                    BINGET     10
//  3417: X                    BINUNICODE '31'

struct PickleTensorReader {
    enum ReadPhase {
        READ_NAME,
        READ_DATA,
        CHECK_SIZE,
        READ_DIMENS
    };
    ReadPhase phase   = READ_NAME;
    size_t entry_size = 0;
    int32_t nelements = 0;

    TensorStorage tensor_storage;

    static ggml_type global_type;  // all pickle_tensors data type
    static bool read_global_type;

    bool read_int_value(uint32_t value) {
        if (phase == CHECK_SIZE) {
            if (entry_size == value * ggml_type_size(tensor_storage.type)) {
                nelements = value;
                phase     = READ_DIMENS;
                return true;
            } else {
                phase = READ_NAME;
            }
        } else if (phase == READ_DIMENS) {
            if (tensor_storage.n_dims + 1 > SD_MAX_DIMS) {  // too many dimens
                phase                 = READ_NAME;
                tensor_storage.n_dims = 0;
            }
            if (nelements % value == 0) {
                tensor_storage.ne[tensor_storage.n_dims] = value;
                tensor_storage.n_dims++;
            }
        }
        return false;
    }

    void read_global(const std::string& str) {
        if (str == "FloatStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F32;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F32;
        } else if (str == "HalfStorage") {
            if (read_global_type) {
                global_type      = GGML_TYPE_F16;
                read_global_type = false;
            }
            tensor_storage.type = GGML_TYPE_F16;
        }
    }

    void read_string(const std::string& str, struct zip_t* zip, std::string dir) {
        if (str == "storage") {
            read_global_type = true;
        } else if (str != "state_dict") {
            if (phase == READ_DATA) {
                std::string entry_name = dir + "data/" + std::string(str);

                size_t i, n = zip_entries_total(zip);
                for (i = 0; i < n; ++i) {
                    zip_entry_openbyindex(zip, i);
                    {
                        std::string name = zip_entry_name(zip);
                        if (name == entry_name) {
                            tensor_storage.index_in_zip = (int)i;
                            entry_size                  = zip_entry_size(zip);
                            zip_entry_close(zip);
                            break;
                        }
                    }
                    zip_entry_close(zip);
                }

                phase = entry_size > 0 ? CHECK_SIZE : READ_NAME;
            }
            if (!read_global_type && phase == READ_NAME) {
                tensor_storage.name = str;
                phase               = READ_DATA;
                tensor_storage.type = global_type;
            }
        }
    }
};

ggml_type PickleTensorReader::global_type = GGML_TYPE_F32;  // all pickle_tensors data type
bool PickleTensorReader::read_global_type = false;

int find_char(uint8_t* buffer, int len, char c) {
    for (int pos = 0; pos < len; pos++) {
        if (buffer[pos] == c) {
            return pos;
        }
    }
    return -1;
}

#define MAX_STRING_BUFFER 512

bool ModelLoader::parse_data_pkl(uint8_t* buffer,
                                 size_t buffer_size,
                                 zip_t* zip,
                                 std::string dir,
                                 size_t file_index,
                                 const std::string prefix) {
    uint8_t* buffer_end = buffer + buffer_size;
    if (buffer[0] == 0x80) {  // proto
        if (buffer[1] != 2) {
            LOG_ERROR("Unsupported protocol\n");
            return false;
        }
        buffer += 2;  // 0x80 and version
        char string_buffer[MAX_STRING_BUFFER];
        bool finish = false;
        PickleTensorReader reader;
        // read pickle binary file
        while (!finish && buffer < buffer_end) {
            uint8_t opcode = *buffer;
            buffer++;
            // https://github.com/python/cpython/blob/3.7/Lib/pickletools.py#L1048
            // https://github.com/python/cpython/blob/main/Lib/pickle.py#L105
            switch (opcode) {
                case '}':  // EMPTY_DICT     = b'}'   # push empty dict
                    break;
                case ']':  // EMPTY_LIST     = b']'   # push empty list
                    break;
                // skip unused sections
                case 'h':  // BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
                case 'q':  // BINPUT         = b'q'   #   "     "    "   "   " ;   "    " 1-byte arg
                case 'Q':  // BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
                    buffer++;
                    break;
                case 'r':  // LONG_BINPUT    = b'r'   #   "     "    "   "   " ;   "    " 4-byte arg
                    buffer += 4;
                    break;
                case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
                    buffer += 8;
                    break;
                case 0x94:  // MEMOIZE          = b'\x94'  # store top of the stack in memo
                    break;
                case '(':  // MARK           = b'('   # push special markobject on stack
                    break;
                case 'K':  // BININT1        = b'K'   # push 1-byte unsigned int
                {
                    uint8_t value = *buffer;
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer++;
                } break;
                case 'M':  // BININT2        = b'M'   # push 2-byte unsigned int
                {
                    uint16_t value = read_short(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;
                    }
                    buffer += 2;
                } break;
                case 'J':  // BININT         = b'J'   # push four-byte signed int
                {
                    const int32_t value = read_int(buffer);
                    if (reader.read_int_value(value)) {
                        buffer++;  // skip tuple after read num_elements
                    }
                    buffer += 4;
                } break;
                case 'X':  // BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
                {
                    const int32_t len = read_int(buffer);
                    buffer += 4;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    if (len > MAX_STRING_BUFFER) {
                        LOG_WARN("tensor name very large");
                    }
                    memcpy(string_buffer, buffer, len < MAX_STRING_BUFFER ? len : (MAX_STRING_BUFFER - 1));
                    buffer += len;
                    reader.read_string(string_buffer, zip, dir);
                } break;
                case 0x8C:  // SHORT_BINUNICODE = b'\x8c'  # push short string; UTF-8 length < 256 bytes
                {
                    const int8_t len = *buffer;
                    buffer++;
                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len;
                    // printf("String: '%s'\n", string_buffer);
                } break;
                case 'c':  // GLOBAL         = b'c'   # push self.find_class(modname, name); 2 string args
                {
                    int len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    buffer += len + 1;
                    len = find_char(buffer, MAX_STRING_BUFFER, '\n');

                    memset(string_buffer, 0, MAX_STRING_BUFFER);
                    memcpy(string_buffer, buffer, len);
                    buffer += len + 1;
                    reader.read_global(string_buffer);
                } break;
                case 0x86:  // TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
                case 0x85:  // TUPLE1         = b'\x85'  # build 1-tuple from stack top
                case 't':   // TUPLE          = b't'   # build tuple from topmost stack items
                    if (reader.phase == PickleTensorReader::READ_DIMENS) {
                        reader.tensor_storage.reverse_ne();
                        reader.tensor_storage.file_index = file_index;
                        // if(strcmp(prefix.c_str(), "scarlett") == 0)
                        // printf(" ZIP got tensor %s \n ", reader.tensor_storage.name.c_str());
                        std::string name = reader.tensor_storage.name;
                        if (!starts_with(name, prefix)) {
                            name = prefix + name;
                        }
                        reader.tensor_storage.name = name;
                        add_tensor_storage(reader.tensor_storage);

                        // LOG_DEBUG("%s", reader.tensor_storage.name.c_str());
                        // reset
                        reader = PickleTensorReader();
                    }
                    break;
                case '.':  // STOP           = b'.'   # every pickle ends with STOP
                    finish = true;
                    break;
                default:
                    break;
            }
        }
    }
    return true;
}

bool ModelLoader::init_from_ckpt_file(const std::string& file_path, const std::string& prefix) {
    LOG_DEBUG("init from '%s'", file_path.c_str());
    file_paths_.push_back(file_path);
    size_t file_index = file_paths_.size() - 1;

    struct zip_t* zip = zip_open(file_path.c_str(), 0, 'r');
    if (zip == nullptr) {
        LOG_ERROR("failed to open '%s'", file_path.c_str());
        return false;
    }
    int n = (int)zip_entries_total(zip);
    for (int i = 0; i < n; ++i) {
        zip_entry_openbyindex(zip, i);
        {
            std::string name = zip_entry_name(zip);
            size_t pos       = name.find("data.pkl");
            if (pos != std::string::npos) {
                std::string dir = name.substr(0, pos);
                printf("ZIP %d, name = %s, dir = %s \n", i, name.c_str(), dir.c_str());
                void* pkl_data = nullptr;
                size_t pkl_size;
                zip_entry_read(zip, &pkl_data, &pkl_size);

                // LOG_DEBUG("%lld", pkl_size);

                parse_data_pkl((uint8_t*)pkl_data, pkl_size, zip, dir, file_index, prefix);

                free(pkl_data);
            }
        }
        zip_entry_close(zip);
    }
    zip_close(zip);
    return true;
}

SDVersion ModelLoader::get_sd_version() {
    TensorStorage token_embedding_weight, input_block_weight;

    bool has_multiple_encoders = false;
    bool is_unet               = false;

    bool is_xl                       = false;
    bool is_flux                     = false;
    bool is_flux2                    = false;
    bool has_single_block_47         = false;
    bool is_wan                      = false;
    int64_t patch_embedding_channels = 0;
    bool has_img_emb                 = false;
    bool has_middle_block_1          = false;
    bool has_output_block_311        = false;
    bool has_output_block_71         = false;

    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (!(is_xl)) {
            if (tensor_storage.name.find("model.diffusion_model.double_blocks.") != std::string::npos) {
                is_flux = true;
            }
            if (tensor_storage.name.find("model.diffusion_model.nerf_final_layer_conv.") != std::string::npos) {
                return VERSION_CHROMA_RADIANCE;
            }
            if (tensor_storage.name.find("model.diffusion_model.joint_blocks.") != std::string::npos) {
                return VERSION_SD3;
            }
            if (tensor_storage.name.find("model.diffusion_model.transformer_blocks.0.img_mod.1.weight") != std::string::npos) {
                return VERSION_QWEN_IMAGE;
            }
            if (tensor_storage.name.find("model.diffusion_model.double_stream_modulation_img.lin.weight") != std::string::npos) {
                is_flux2 = true;
            }
            if (tensor_storage.name.find("single_blocks.47.linear1.weight") != std::string::npos) {
                has_single_block_47 = true;
            }
            if (tensor_storage.name.find("model.diffusion_model.double_blocks.0.img_mlp.gate_proj.weight") != std::string::npos) {
                return VERSION_OVIS_IMAGE;
            }
            if (tensor_storage.name.find("model.diffusion_model.cap_embedder.0.weight") != std::string::npos) {
                return VERSION_Z_IMAGE;
            }
            if (tensor_storage.name.find("model.diffusion_model.blocks.0.cross_attn.norm_k.weight") != std::string::npos) {
                is_wan = true;
            }
            if (tensor_storage.name.find("model.diffusion_model.patch_embedding.weight") != std::string::npos) {
                patch_embedding_channels = tensor_storage.ne[3];
            }
            if (tensor_storage.name.find("model.diffusion_model.img_emb") != std::string::npos) {
                has_img_emb = true;
            }
            if (tensor_storage.name.find("model.diffusion_model.input_blocks.") != std::string::npos ||
                tensor_storage.name.find("unet.down_blocks.") != std::string::npos) {
                is_unet = true;
                if (has_multiple_encoders) {
                    is_xl = true;
                }
            }
            if (tensor_storage.name.find("conditioner.embedders.1") != std::string::npos ||
                tensor_storage.name.find("cond_stage_model.1") != std::string::npos ||
                tensor_storage.name.find("te.1") != std::string::npos) {
                has_multiple_encoders = true;
                if (is_unet) {
                    is_xl = true;
                }
            }
            if (tensor_storage.name.find("model.diffusion_model.input_blocks.8.0.time_mixer.mix_factor") != std::string::npos) {
                return VERSION_SVD;
            }
        }
        if (tensor_storage.name.find("model.diffusion_model.middle_block.1.") != std::string::npos ||
            tensor_storage.name.find("unet.mid_block.resnets.1.") != std::string::npos) {
            has_middle_block_1 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.output_blocks.3.1.transformer_blocks.1") != std::string::npos) {
            has_output_block_311 = true;
        }
        if (tensor_storage.name.find("model.diffusion_model.output_blocks.7.1") != std::string::npos) {
            has_output_block_71 = true;
        }
        if (tensor_storage.name == "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "cond_stage_model.model.token_embedding.weight" ||
            tensor_storage.name == "text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "te.text_model.embeddings.token_embedding.weight" ||
            tensor_storage.name == "conditioner.embedders.0.model.token_embedding.weight" ||
            tensor_storage.name == "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight") {
            token_embedding_weight = tensor_storage;
            // break;
        }
        if (tensor_storage.name == "model.diffusion_model.input_blocks.0.0.weight" ||
            tensor_storage.name == "model.diffusion_model.img_in.weight" ||
            tensor_storage.name == "unet.conv_in.weight") {
            input_block_weight = tensor_storage;
        }
    }
    if (is_wan) {
        LOG_DEBUG("patch_embedding_channels %d", patch_embedding_channels);
        if (patch_embedding_channels == 184320 && !has_img_emb) {
            return VERSION_WAN2_2_I2V;
        }
        if (patch_embedding_channels == 147456 && !has_img_emb) {
            return VERSION_WAN2_2_TI2V;
        }
        return VERSION_WAN2;
    }
    bool is_inpaint = input_block_weight.ne[2] == 9;
    bool is_ip2p    = input_block_weight.ne[2] == 8;
    if (is_xl) {
        if (is_inpaint) {
            return VERSION_SDXL_INPAINT;
        }
        if (is_ip2p) {
            return VERSION_SDXL_PIX2PIX;
        }
        if (!has_middle_block_1) {
            if (!has_output_block_311) {
                return VERSION_SDXL_VEGA;
            }
            return VERSION_SDXL_SSD1B;
        }
        return VERSION_SDXL;
    }

    if (is_flux && !is_flux2) {
        if (input_block_weight.ne[0] == 384) {
            return VERSION_FLUX_FILL;
        }
        if (input_block_weight.ne[0] == 128) {
            return VERSION_FLUX_CONTROLS;
        }
        if (input_block_weight.ne[0] == 196) {
            return VERSION_FLEX_2;
        }
        return VERSION_FLUX;
    }

    if (is_flux2) {
        if (has_single_block_47) {
            return VERSION_FLUX2;
        }
        return VERSION_FLUX2_KLEIN;
    }

    if (token_embedding_weight.ne[0] == 768) {
        if (is_inpaint) {
            return VERSION_SD1_INPAINT;
        }
        if (is_ip2p) {
            return VERSION_SD1_PIX2PIX;
        }
        if (!has_middle_block_1) {
            if (!has_output_block_71) {
                return VERSION_SDXS;
            }
            return VERSION_SD1_TINY_UNET;
        }
        return VERSION_SD1;
    } else if (token_embedding_weight.ne[0] == 1024) {
        if (is_inpaint) {
            return VERSION_SD2_INPAINT;
        }
        if (!has_middle_block_1) {
            return VERSION_SD2_TINY_UNET;
        }
        return VERSION_SD2;
    }
    return VERSION_COUNT;
}

std::map<ggml_type, uint32_t> ModelLoader::get_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_conditioner_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if ((tensor_storage.name.find("text_encoders") == std::string::npos &&
             tensor_storage.name.find("cond_stage_model") == std::string::npos &&
             tensor_storage.name.find("te.text_model.") == std::string::npos &&
             tensor_storage.name.find("conditioner") == std::string::npos)) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_diffusion_model_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("model.diffusion_model.") == std::string::npos && tensor_storage.name.find("unet.") == std::string::npos) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

std::map<ggml_type, uint32_t> ModelLoader::get_vae_wtype_stat() {
    std::map<ggml_type, uint32_t> wtype_stat;
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        if (tensor_storage.name.find("vae.") == std::string::npos &&
            tensor_storage.name.find("first_stage_model") == std::string::npos) {
            continue;
        }

        auto iter = wtype_stat.find(tensor_storage.type);
        if (iter != wtype_stat.end()) {
            iter->second++;
        } else {
            wtype_stat[tensor_storage.type] = 1;
        }
    }
    return wtype_stat;
}

static std::vector<std::pair<std::string, ggml_type>> parse_tensor_type_rules(const std::string& tensor_type_rules) {
    std::vector<std::pair<std::string, ggml_type>> result;
    for (const auto& item : split_string(tensor_type_rules, ',')) {
        if (item.size() == 0)
            continue;
        std::string::size_type pos = item.find('=');
        if (pos == std::string::npos) {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
            continue;
        }
        std::string tensor_pattern = item.substr(0, pos);
        std::string type_name      = item.substr(pos + 1);

        ggml_type tensor_type = GGML_TYPE_COUNT;

        if (type_name == "f32") {
            tensor_type = GGML_TYPE_F32;
        } else {
            for (size_t i = 0; i < GGML_TYPE_COUNT; i++) {
                auto trait = ggml_get_type_traits((ggml_type)i);
                if (trait->to_float && trait->type_size && type_name == trait->type_name) {
                    tensor_type = (ggml_type)i;
                }
            }
        }

        if (tensor_type != GGML_TYPE_COUNT) {
            result.emplace_back(tensor_pattern, tensor_type);
        } else {
            LOG_WARN("ignoring invalid quant override \"%s\"", item.c_str());
        }
    }
    return result;
}

void ModelLoader::set_wtype_override(ggml_type wtype, std::string tensor_type_rules) {
    auto map_rules = parse_tensor_type_rules(tensor_type_rules);
    for (auto& [name, tensor_storage] : tensor_storage_map) {
        ggml_type dst_type = wtype;
        for (const auto& tensor_type_rule : map_rules) {
            std::regex pattern(tensor_type_rule.first);
            if (std::regex_search(name, pattern)) {
                dst_type = tensor_type_rule.second;
                break;
            }
        }
        if (dst_type == GGML_TYPE_COUNT) {
            continue;
        }
        if (!tensor_should_be_converted(tensor_storage, dst_type)) {
            continue;
        }
        tensor_storage.expected_type = dst_type;
    }
}

std::string ModelLoader::load_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(merges_utf8_c_str), sizeof(merges_utf8_c_str));
    return merges_utf8_str;
}

std::string ModelLoader::load_qwen2_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(qwen2_merges_utf8_c_str), sizeof(qwen2_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string ModelLoader::load_mistral_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(mistral_merges_utf8_c_str), sizeof(mistral_merges_utf8_c_str));
    return merges_utf8_str;
}

std::string ModelLoader::load_mistral_vocab_json() {
    std::string json_str(reinterpret_cast<const char*>(mistral_vocab_json_utf8_c_str), sizeof(mistral_vocab_json_utf8_c_str));
    return json_str;
}

std::string ModelLoader::load_t5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(t5_tokenizer_json_str), sizeof(t5_tokenizer_json_str));
    return json_str;
}

std::string ModelLoader::load_umt5_tokenizer_json() {
    std::string json_str(reinterpret_cast<const char*>(umt5_tokenizer_json_str), sizeof(umt5_tokenizer_json_str));
    return json_str;
}

bool ModelLoader::load_tensors(on_new_tensor_cb_t on_new_tensor_cb, int n_threads_p, bool enable_mmap) {
    int64_t process_time_ms = 0;
    std::atomic<int64_t> read_time_ms(0);
    std::atomic<int64_t> memcpy_time_ms(0);
    std::atomic<int64_t> copy_to_backend_time_ms(0);
    std::atomic<int64_t> convert_time_ms(0);

    int num_threads_to_use = n_threads_p > 0 ? n_threads_p : sd_get_num_physical_cores();
    LOG_DEBUG("using %d threads for model loading", num_threads_to_use);

    int64_t start_time = ggml_time_ms();
    if (sd_convert_htp_permute_log_enabled()) {
        sd_convert_htp_permute_count().store(0);
    }

    std::vector<TensorStorage> processed_tensor_storages;
    for (const auto& [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }
        processed_tensor_storages.push_back(tensor_storage);
    }

    process_time_ms = ggml_time_ms() - start_time;

    bool success                          = true;
    size_t total_tensors_processed        = 0;
    const size_t total_tensors_to_process = processed_tensor_storages.size();
    const int64_t t_start                 = ggml_time_ms();
    int last_n_threads                    = 1;

    for (size_t file_index = 0; file_index < file_paths_.size(); file_index++) {
        std::string file_path = file_paths_[file_index];
        LOG_DEBUG("loading tensors from %s", file_path.c_str());

        std::vector<const TensorStorage*> file_tensors;
        for (const auto& ts : processed_tensor_storages) {
            if (ts.file_index == file_index) {
                file_tensors.push_back(&ts);
            }
        }
        if (file_tensors.empty()) {
            continue;
        }

        bool is_zip = false;
        for (auto const& ts : file_tensors) {
            if (ts->index_in_zip >= 0) {
                is_zip = true;
                break;
            }
        }

        std::unique_ptr<MmapWrapper> mmapped;
        if (enable_mmap && !is_zip) {
            LOG_DEBUG("using mmap for I/O");
            mmapped = MmapWrapper::create(file_path);
            if (!mmapped) {
                LOG_WARN("failed to memory-map '%s'", file_path.c_str());
            }
        }

        int n_threads = is_zip ? 1 : std::min(num_threads_to_use, (int)file_tensors.size());
        if (n_threads < 1) {
            n_threads = 1;
        }
        last_n_threads = n_threads;

        std::atomic<size_t> tensor_idx(0);
        std::atomic<bool> failed(false);
        std::vector<std::thread> workers;

        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back([&, file_path, is_zip]() {
                std::ifstream file;
                struct zip_t* zip = nullptr;
                if (is_zip) {
                    zip = zip_open(file_path.c_str(), 0, 'r');
                    if (zip == nullptr) {
                        LOG_ERROR("failed to open zip '%s'", file_path.c_str());
                        failed = true;
                        return;
                    }
                } else if (!mmapped) {
                    file.open(file_path, std::ios::binary);
                    if (!file.is_open()) {
                        LOG_ERROR("failed to open '%s'", file_path.c_str());
                        failed = true;
                        return;
                    }
                }

                std::vector<uint8_t> read_buffer;
                std::vector<uint8_t> convert_buffer;

                while (true) {
                    int64_t t0, t1;
                    size_t idx = tensor_idx.fetch_add(1);
                    if (idx >= file_tensors.size() || failed) {
                        break;
                    }

                    const TensorStorage& tensor_storage = *file_tensors[idx];
                    ggml_tensor* dst_tensor             = nullptr;

                    t0 = ggml_time_ms();

                    if (!on_new_tensor_cb(tensor_storage, &dst_tensor)) {
                        LOG_WARN("process tensor failed: '%s'", tensor_storage.name.c_str());
                        failed = true;
                        break;
                    }

                    if (dst_tensor == nullptr) {
                        t1 = ggml_time_ms();
                        read_time_ms.fetch_add(t1 - t0);
                        continue;
                    }

                    size_t nbytes_to_read = tensor_storage.nbytes_to_read();

                    auto read_data = [&](char* buf, size_t n) {
                        if (zip != nullptr) {
                            zip_entry_openbyindex(zip, tensor_storage.index_in_zip);
                            size_t entry_size = zip_entry_size(zip);
                            if (entry_size != n) {
                                int64_t t_memcpy_start;
                                read_buffer.resize(entry_size);
                                zip_entry_noallocread(zip, (void*)read_buffer.data(), entry_size);
                                t_memcpy_start = ggml_time_ms();
                                memcpy((void*)buf, (void*)(read_buffer.data() + tensor_storage.offset), n);
                                memcpy_time_ms.fetch_add(ggml_time_ms() - t_memcpy_start);
                            } else {
                                zip_entry_noallocread(zip, (void*)buf, n);
                            }
                            zip_entry_close(zip);
                        } else if (mmapped) {
                            if (!mmapped->copy_data(buf, n, tensor_storage.offset)) {
                                LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                                failed = true;
                            }
                        } else {
                            file.seekg(tensor_storage.offset);
                            file.read(buf, n);
                            if (!file) {
                                LOG_ERROR("read tensor data failed: '%s'", file_path.c_str());
                                failed = true;
                            }
                        }
                    };

                    char* read_buf    = nullptr;
                    char* target_buf  = nullptr;
                    char* convert_buf = nullptr;
                    if (dst_tensor->buffer == nullptr || ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        if (tensor_storage.type == dst_tensor->type) {
                            GGML_ASSERT(ggml_nbytes(dst_tensor) == tensor_storage.nbytes());
                            if (tensor_storage.is_f64 || tensor_storage.is_i64) {
                                read_buffer.resize(tensor_storage.nbytes_to_read());
                                read_buf = (char*)read_buffer.data();
                            } else {
                                read_buf = (char*)dst_tensor->data;
                            }
                            target_buf = (char*)dst_tensor->data;
                        } else {
                            read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                            read_buf    = (char*)read_buffer.data();
                            target_buf  = read_buf;
                            convert_buf = (char*)dst_tensor->data;
                        }
                    } else {
                        read_buffer.resize(std::max(tensor_storage.nbytes(), tensor_storage.nbytes_to_read()));
                        read_buf   = (char*)read_buffer.data();
                        target_buf = read_buf;

                        if (tensor_storage.type != dst_tensor->type) {
                            convert_buffer.resize(ggml_nbytes(dst_tensor));
                            convert_buf = (char*)convert_buffer.data();
                        }
                    }

                    t0 = ggml_time_ms();
                    read_data(read_buf, nbytes_to_read);
                    t1 = ggml_time_ms();
                    read_time_ms.fetch_add(t1 - t0);

                    t0 = ggml_time_ms();
                    if (tensor_storage.is_f8_e4m3) {
                        f8_e4m3_to_f16_vec((uint8_t*)read_buf, (uint16_t*)target_buf, tensor_storage.nelements());
                    } else if (tensor_storage.is_f8_e5m2) {
                        f8_e5m2_to_f16_vec((uint8_t*)read_buf, (uint16_t*)target_buf, tensor_storage.nelements());
                    } else if (tensor_storage.is_f64) {
                        f64_to_f32_vec((double*)read_buf, (float*)target_buf, tensor_storage.nelements());
                    } else if (tensor_storage.is_i64) {
                        i64_to_i32_vec((int64_t*)read_buf, (int32_t*)target_buf, tensor_storage.nelements());
                    }
                    const int nrows     = (int)tensor_storage.nelements() / (int)tensor_storage.ne[0];
                    const int n_per_row = (int)tensor_storage.ne[0];
                    bool did_hmx_permute = false;
                    if (sd_convert_should_hmx_permute(tensor_storage, dst_tensor)) {
                        did_hmx_permute = true;
                        if (convert_buf == nullptr) {
                            // Type may be unchanged (e.g. q4_0 -> q4_0): permute still needs a temp output slot.
                            convert_buffer.resize(ggml_nbytes(dst_tensor));
                            convert_buf = (char*)convert_buffer.data();
                        }
                        const int permute_id = sd_convert_htp_permute_count().fetch_add(1) + 1;
                        if (sd_convert_htp_permute_log_enabled() && permute_id <= 128) {
                            LOG_INFO("hmx permute #%d: %s shape=[%lld,%lld] src=%s dst=%s",
                                     permute_id,
                                     tensor_storage.name.c_str(),
                                     (long long)tensor_storage.ne[0],
                                     (long long)tensor_storage.ne[1],
                                     ggml_type_name(tensor_storage.type),
                                     ggml_type_name(dst_tensor->type));
                        }
                        if (sd_convert_htp_invpermute_enabled()) {
                            // Build a vanilla-basis tensor from a permuted one (input is assumed to already be in
                            // HMX permuted basis). NOTE: some exported GGUFs are mixed-layout (repack-only-permuted):
                            // permuted 2D weights are repacked for HMX, while 1D weights remain vanilla ggml blocks.
                            // For INVPERMUTE we unrepack the permuted weights into vanilla blocks, then dequantize.
                            std::vector<float> src_f32((size_t)nrows * n_per_row);
                            std::vector<float> inv_f32((size_t)nrows * n_per_row);
                            if (ggml_is_quantized(tensor_storage.type)) {
                                // If export.repack is enabled, ggml dequant handles repacked rows directly.
                                // If it's disabled (mixed-layout export), unrepack permuted weights to vanilla blocks first.
                                const char* env_repack = std::getenv("GGML_HTP_CONTRACT_EXPORT_REPACK");
                                const bool export_repack =
                                    (env_repack != nullptr && env_repack[0] != '\0' && std::strcmp(env_repack, "0") != 0 && std::strcmp(env_repack, "false") != 0);
                                if (!export_repack) {
                                    sd_hmx_unrepack_quant_weight_inplace_if_needed(target_buf, tensor_storage.type, nrows, n_per_row, tensor_storage.name);
                                }
                            }
                            convert_tensor((void*)target_buf,
                                           tensor_storage.type,
                                           (void*)src_f32.data(),
                                           GGML_TYPE_F32,
                                           nrows,
                                           n_per_row);
                            sd_hmx_invpermute_linear_weight_f32(src_f32.data(), inv_f32.data(), nrows, n_per_row);
                            convert_tensor((void*)inv_f32.data(),
                                           GGML_TYPE_F32,
                                           convert_buf,
                                           dst_tensor->type,
                                           nrows,
                                           n_per_row);
                        } else {
                            // IMPORTANT: for quant weights, match runtime prepack semantics:
                            //   (1) quantize (vanilla) -> (2) dequant -> (3) permute -> (4) requant -> (5) repack.
                            // Direct "permute then quantize from bf16/f16" produces different qblocks and can
                            // break HMX matmul correctness for diffusion.
                            if (ggml_is_quantized(dst_tensor->type)) {
                                std::vector<char> q0;
                                q0.resize(ggml_nbytes(dst_tensor));
                                convert_tensor((void*)target_buf,
                                               tensor_storage.type,
                                               (void*)q0.data(),
                                               dst_tensor->type,
                                               nrows,
                                               n_per_row);

                                auto qtype = ggml_get_type_traits(dst_tensor->type);
                                GGML_ASSERT(qtype != nullptr && qtype->to_float != nullptr);

                                std::vector<float> q0_f32((size_t)nrows * n_per_row);
                                std::vector<float> permuted_f32((size_t)nrows * n_per_row);
                                qtype->to_float((const void*)q0.data(), q0_f32.data(), (int64_t)nrows * n_per_row);
                                sd_hmx_permute_linear_weight_f32(q0_f32.data(), permuted_f32.data(), nrows, n_per_row);
                                convert_tensor((void*)permuted_f32.data(),
                                               GGML_TYPE_F32,
                                               convert_buf,
                                               dst_tensor->type,
                                               nrows,
                                               n_per_row);
                            } else {
                                std::vector<float> src_f32((size_t)nrows * n_per_row);
                                std::vector<float> permuted_f32((size_t)nrows * n_per_row);
                                convert_tensor((void*)target_buf,
                                               tensor_storage.type,
                                               (void*)src_f32.data(),
                                               GGML_TYPE_F32,
                                               nrows,
                                               n_per_row);
                                sd_hmx_permute_linear_weight_f32(src_f32.data(), permuted_f32.data(), nrows, n_per_row);
                                convert_tensor((void*)permuted_f32.data(),
                                               GGML_TYPE_F32,
                                               convert_buf,
                                               dst_tensor->type,
                                               nrows,
                                               n_per_row);
                            }
                            sd_hmx_repack_quant_weight_inplace_if_needed(convert_buf, dst_tensor->type, nrows, n_per_row, tensor_storage.name);
                        }
                    } else if (tensor_storage.type != dst_tensor->type) {
                        if (convert_buf == nullptr) {
                            LOG_ERROR("read tensor data failed: too less memory for conversion");
                            failed = true;
                            return;
                        }
                        convert_tensor((void*)target_buf,
                                       tensor_storage.type,
                                       convert_buf,
                                       dst_tensor->type,
                                       nrows,
                                       n_per_row);
                    } else {
                        convert_buf = read_buf;
                    }

                    // Optional HMX repack without permute (repack-only) for CPU fallback correctness.
                    // This is required when exporting a mixed offload model: some qweights stay in
                    // vanilla basis (no permute), but we still want a single on-disk repack contract.
                    if (!did_hmx_permute) {
                        if (!sd_convert_htp_repack_only_permuted()) {
                            sd_hmx_repack_quant_weight_inplace_if_needed(convert_buf, dst_tensor->type, nrows, n_per_row, tensor_storage.name);
                        }
                    }
                    t1 = ggml_time_ms();
                    convert_time_ms.fetch_add(t1 - t0);

                    if (dst_tensor->buffer != nullptr && !ggml_backend_buffer_is_host(dst_tensor->buffer)) {
                        t0 = ggml_time_ms();
                        ggml_backend_tensor_set(dst_tensor, convert_buf, 0, ggml_nbytes(dst_tensor));
                        t1 = ggml_time_ms();
                        copy_to_backend_time_ms.fetch_add(t1 - t0);
                    } else if (convert_buf != (char*)dst_tensor->data) {
                        // Same-type convert path (e.g. q4_0->q4_0 with HMX pre-permute) uses a temp buffer.
                        // Copy back so converted bytes are persisted before GGUF save.
                        t0 = ggml_time_ms();
                        memcpy(dst_tensor->data, convert_buf, ggml_nbytes(dst_tensor));
                        t1 = ggml_time_ms();
                        memcpy_time_ms.fetch_add(t1 - t0);
                    }
                }
                if (zip != nullptr) {
                    zip_close(zip);
                }
            });
        }

        while (true) {
            size_t current_idx = tensor_idx.load();
            if (current_idx >= file_tensors.size() || failed) {
                break;
            }
            size_t curr_num = total_tensors_processed + current_idx;
            pretty_progress(static_cast<int>(curr_num), static_cast<int>(total_tensors_to_process), (ggml_time_ms() - t_start) / 1000.0f / (curr_num + 1e-6f));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        for (auto& w : workers) {
            w.join();
        }

        if (failed) {
            success = false;
            break;
        }
        total_tensors_processed += file_tensors.size();
        pretty_progress(static_cast<int>(total_tensors_processed), static_cast<int>(total_tensors_to_process), (ggml_time_ms() - t_start) / 1000.0f / (total_tensors_processed + 1e-6f));
        if (total_tensors_processed < total_tensors_to_process) {
            printf("\n");
        }
    }

    int64_t end_time = ggml_time_ms();
    LOG_INFO("loading tensors completed, taking %.2fs (process: %.2fs, read: %.2fs, memcpy: %.2fs, convert: %.2fs, copy_to_backend: %.2fs)",
             (end_time - start_time) / 1000.f,
             process_time_ms / 1000.f,
             (read_time_ms.load() / (float)last_n_threads) / 1000.f,
             (memcpy_time_ms.load() / (float)last_n_threads) / 1000.f,
             (convert_time_ms.load() / (float)last_n_threads) / 1000.f,
             (copy_to_backend_time_ms.load() / (float)last_n_threads) / 1000.f);
    if (sd_convert_htp_permute_log_enabled()) {
        LOG_INFO("hmx permute total tensors: %d", sd_convert_htp_permute_count().load());
        LOG_INFO("hmx repack total tensors: %d", sd_convert_htp_repack_count().load());
    }
    return success;
}

bool ModelLoader::load_tensors(std::map<std::string, struct ggml_tensor*>& tensors,
                               std::set<std::string> ignore_tensors,
                               int n_threads,
                               bool enable_mmap) {
    std::set<std::string> tensor_names_in_file;
    std::mutex tensor_names_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        // LOG_DEBUG("%s", tensor_storage.to_string().c_str());
        {
            std::lock_guard<std::mutex> lock(tensor_names_mutex);
            tensor_names_in_file.insert(name);
        }

        struct ggml_tensor* real;
        if (tensors.find(name) != tensors.end()) {
            real = tensors[name];
        } else {
            for (auto& ignore_tensor : ignore_tensors) {
                if (starts_with(name, ignore_tensor)) {
                    return true;
                }
            }
            LOG_INFO("unknown tensor '%s' in model file", tensor_storage.to_string().c_str());
            return true;
        }

        if (
            real->ne[0] != tensor_storage.ne[0] ||
            real->ne[1] != tensor_storage.ne[1] ||
            real->ne[2] != tensor_storage.ne[2] ||
            real->ne[3] != tensor_storage.ne[3]) {
            LOG_ERROR(
                "tensor '%s' has wrong shape in model file: "
                "got [%d, %d, %d, %d], expected [%d, %d, %d, %d]",
                name.c_str(),
                (int)tensor_storage.ne[0], (int)tensor_storage.ne[1], (int)tensor_storage.ne[2], (int)tensor_storage.ne[3],
                (int)real->ne[0], (int)real->ne[1], (int)real->ne[2], (int)real->ne[3]);
            return false;
        }

        *dst_tensor = real;

        return true;
    };

    bool success = load_tensors(on_new_tensor_cb, n_threads, enable_mmap);
    if (!success) {
        LOG_ERROR("load tensors from file failed");
        return false;
    }

    bool some_tensor_not_init = false;

    for (auto pair : tensors) {
        if (pair.first.find("cond_stage_model.transformer.text_model.encoder.layers.23") != std::string::npos) {
            continue;
        }

        if (pair.first.find("alphas_cumprod") != std::string::npos) {
            continue;
        }

        if (tensor_names_in_file.find(pair.first) == tensor_names_in_file.end()) {
            LOG_ERROR("tensor '%s' not in model file", pair.first.c_str());
            some_tensor_not_init = true;
        }
    }

    if (some_tensor_not_init) {
        return false;
    }
    return true;
}

bool ModelLoader::tensor_should_be_converted(const TensorStorage& tensor_storage, ggml_type type) {
    const std::string& name = tensor_storage.name;
    if (type != GGML_TYPE_COUNT) {
        if (ggml_is_quantized(type) && tensor_storage.ne[0] % ggml_blck_size(type) != 0) {
            // Pass, do not convert
        } else if (ends_with(name, ".bias")) {
            // Pass, do not convert
        } else if (ends_with(name, ".scale")) {
            // Pass, do not convert
        } else if (contains(name, "img_in.") ||
                   contains(name, "txt_in.") ||
                   contains(name, "time_in.") ||
                   contains(name, "vector_in.") ||
                   contains(name, "guidance_in.") ||
                   contains(name, "final_layer.")) {
            // Pass, do not convert. For FLUX
        } else if (contains(name, "x_embedder.") ||
                   contains(name, "t_embedder.") ||
                   contains(name, "y_embedder.") ||
                   contains(name, "pos_embed") ||
                   contains(name, "context_embedder.")) {
            // Pass, do not convert. For MMDiT
        } else if (contains(name, "time_embed.") || contains(name, "label_emb.")) {
            // Pass, do not convert. For Unet
        } else if (contains(name, "embedding")) {
            // Pass, do not convert embedding
        } else {
            return true;
        }
    }
    return false;
}

bool ModelLoader::save_to_gguf_file(const std::string& file_path, ggml_type type, const std::string& tensor_type_rules_str) {
    auto tensor_type_rules = parse_tensor_type_rules(tensor_type_rules_str);

    auto backend    = ggml_backend_cpu_init();
    size_t mem_size = 1 * 1024 * 1024;  // for padding
    mem_size += tensor_storage_map.size() * ggml_tensor_overhead();
    mem_size += get_params_mem_size(backend, type, tensor_type_rules);
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});

    gguf_context* gguf_ctx = gguf_init_empty();
    const char* overlay_path = std::getenv("SD_DIFFUSION_MODEL_OVERLAY");
    if (overlay_path != nullptr && overlay_path[0] != '\0') {
        model_loader_write_existing_npu_contract_metadata(gguf_ctx, gguf_metadata_);
    } else {
        model_loader_write_npu_contract_metadata(gguf_ctx);
    }

    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        ggml_type tensor_type   = tensor_storage.type;
        ggml_type dst_type      = type;

        for (const auto& tensor_type_rule : tensor_type_rules) {
            std::regex pattern(tensor_type_rule.first);
            if (std::regex_search(name, pattern)) {
                dst_type = tensor_type_rule.second;
                break;
            }
        }

        const bool should_convert = tensor_should_be_converted(tensor_storage, dst_type);
        if (should_convert) {
            tensor_type = dst_type;
        }

        // Keep quantized exports BF16-free: tensors skipped from quantization
        // are still cast to F16 to avoid BF16-only runtime gaps on some backends.
        if (!should_convert &&
            tensor_type == GGML_TYPE_BF16 &&
            (dst_type == GGML_TYPE_F16 || ggml_is_quantized(dst_type))) {
            tensor_type = GGML_TYPE_F16;
        }

        std::lock_guard<std::mutex> lock(tensor_mutex);
        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, tensor_storage.n_dims, tensor_storage.ne);
        if (tensor == nullptr) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        // LOG_DEBUG("%s %d %s %d[%d %d %d %d] %d[%d %d %d %d]", name.c_str(),
        // ggml_nbytes(tensor), ggml_type_name(tensor_type),
        // tensor_storage.n_dims,
        // tensor_storage.ne[0], tensor_storage.ne[1], tensor_storage.ne[2], tensor_storage.ne[3],
        // tensor->n_dims, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

        if (!tensor->data) {
            GGML_ASSERT(ggml_nelements(tensor) == 0);
            // avoid crashing the gguf writer by setting a dummy pointer for zero-sized tensors
            LOG_DEBUG("setting dummy pointer for zero-sized tensor %s", name.c_str());
            tensor->data = ggml_get_mem_buffer(ggml_ctx);
        }

        *dst_tensor = tensor;

        gguf_add_tensor(gguf_ctx, tensor);

        return true;
    };

    // Export correctness matters more than host-side conversion throughput.
    // The convert path mutates tensor bytes in-place (cast / quantize / permute / repack)
    // before handing them to the GGUF writer, and we have observed nondeterministic full-model
    // corruption when many tensors are processed concurrently. Keep GGUF export single-threaded
    // so repeated source->GGUF conversions are byte-stable.
    bool success = load_tensors(on_new_tensor_cb, 1, false);
    ggml_backend_free(backend);
    LOG_INFO("load tensors done");
    LOG_INFO("trying to save tensors to %s", file_path.c_str());
    if (success) {
        gguf_write_to_file(gguf_ctx, file_path.c_str(), false);
    }
    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);
    return success;
}

int64_t ModelLoader::get_params_mem_size(ggml_backend_t backend,
                                         ggml_type type,
                                         const std::vector<std::pair<std::string, ggml_type>>& tensor_type_rules) {
    size_t alignment = 128;
    if (backend != nullptr) {
        alignment = ggml_backend_get_alignment(backend);
    }
    int64_t mem_size = 0;

    for (auto [name, tensor_storage] : tensor_storage_map) {
        if (is_unused_tensor(tensor_storage.name)) {
            continue;
        }

        ggml_type dst_type = type;
        for (const auto& tensor_type_rule : tensor_type_rules) {
            std::regex pattern(tensor_type_rule.first);
            if (std::regex_search(name, pattern)) {
                dst_type = tensor_type_rule.second;
                break;
            }
        }

        const bool should_convert = tensor_should_be_converted(tensor_storage, dst_type);
        if (should_convert) {
            tensor_storage.type = dst_type;
        } else if (tensor_storage.type == GGML_TYPE_BF16 &&
                   (dst_type == GGML_TYPE_F16 || ggml_is_quantized(dst_type))) {
            // Keep quantized exports BF16-free: tensors skipped from quantization
            // are still cast to F16 to avoid BF16-only runtime gaps on some backends.
            tensor_storage.type = GGML_TYPE_F16;
        }
        mem_size += tensor_storage.nbytes() + alignment;
    }

    return mem_size;
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    ModelLoader model_loader;

    if (!model_loader.init_from_file(input_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    if (const char* overlay_path = std::getenv("SD_DIFFUSION_MODEL_OVERLAY");
        overlay_path != nullptr && overlay_path[0] != '\0') {
        struct SavedEnv {
            bool had = false;
            std::string value;
        };
        std::map<std::string, SavedEnv> saved_env;
        for (const char* key : {
                 "GGML_HTP_CONTRACT_ACTIVE",
                 "GGML_HTP_CONTRACT_EXPORT_PERMUTE",
                 "GGML_HTP_CONTRACT_EXPORT_REPACK",
             }) {
            const char* cur = std::getenv(key);
            SavedEnv s;
            s.had = (cur != nullptr);
            if (cur != nullptr) {
                s.value = cur;
            }
            saved_env.emplace(key, std::move(s));
        }

        LOG_INFO("loading diffusion model overlay for convert from '%s'", overlay_path);
        ModelLoader overlay_loader;
        if (!overlay_loader.init_from_file(overlay_path)) {
            LOG_ERROR("init overlay model loader from file failed: '%s'", overlay_path);
            return false;
        }

        const auto include = sd_parse_csv_env("SD_DIFFUSION_MODEL_OVERLAY_INCLUDE");
        size_t overlay_count = 0;
        for (const auto& kv : overlay_loader.get_tensor_storage_map()) {
            TensorStorage ts = kv.second;
            // The source overlay may come either from raw safetensors names
            // ("layers.0....") or from a prefixed GGUF view
            // ("model.diffusion_model.layers.0...."). Match whichever naming
            // convention the current baseline loader uses so we replace tensors
            // instead of appending duplicate ones.
            const std::string raw_name = ts.name;
            const std::string prefixed_name =
                (raw_name.rfind("model.diffusion_model.", 0) == 0) ? raw_name : "model.diffusion_model." + raw_name;
            const auto& base_map = model_loader.get_tensor_storage_map();
            if (base_map.find(raw_name) != base_map.end()) {
                ts.name = raw_name;
            } else if (base_map.find(prefixed_name) != base_map.end()) {
                ts.name = prefixed_name;
            } else {
                ts.name = raw_name;
            }
            if (ts.name.find(".feed_forward.w2.weight") == std::string::npos) {
                continue;
            }
            if (!include.empty() && !sd_name_match_any(ts.name, include)) {
                continue;
            }
            model_loader.overlay_tensor_storage(ts, overlay_path);
            overlay_count++;
        }
        LOG_INFO("applied convert overlay tensors: %zu", overlay_count);

        for (const auto& kv : saved_env) {
            if (kv.second.had) {
                model_set_env_overwrite(kv.first.c_str(), kv.second.value);
            } else {
#ifdef _WIN32
                _putenv_s(kv.first.c_str(), "");
#else
                unsetenv(kv.first.c_str());
#endif
            }
        }
    }

    if (vae_path != nullptr && strlen(vae_path) > 0) {
        if (!model_loader.init_from_file(vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", vae_path);
            return false;
        }
    }
    if (convert_name) {
        model_loader.convert_tensors_name();
    }
    bool success = model_loader.save_to_gguf_file(output_path, (ggml_type)output_type, tensor_type_rules);
    return success;
}
