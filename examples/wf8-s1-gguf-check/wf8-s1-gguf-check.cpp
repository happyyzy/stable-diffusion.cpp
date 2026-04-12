#include "ggml.h"
#include "gguf.h"
#include "../../ggml/src/ggml-quants.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <numeric>
#include <vector>

#if defined(__ANDROID__) && defined(_LIBCPP_VERSION)
namespace fs = std::__fs::filesystem;
#else
namespace fs = std::filesystem;
#endif

extern "C" void repack_q4_0_super_block_hvx(const block_q4_0 * src, void * dst, size_t size);

static size_t permuted_weight_index(int k, int i, int j) {
    static const uint8_t lane_perm[4] = { 0, 2, 1, 3 };
    const int i0 = i / 32;
    const int i1 = i % 32;
    const int j0 = j / 32;
    const int j1 = j % 32;
    const size_t tile_idx = (size_t) j0 * (size_t) (k / 32) + (size_t) i0;
    const size_t tile_row = (size_t) (i1 / 4) * 4 + (size_t) (j1 / 8);
    const size_t tile_col = (size_t) (j1 % 8) * 4 + (size_t) lane_perm[i1 % 4];
    return tile_idx * 1024 + tile_row * 32 + tile_col;
}

static float decode_fp8_e4m3fn(uint8_t code) {
    const int sign = (code & 0x80) ? -1 : 1;
    const int exp  = (code >> 3) & 0x0f;
    const int mant = code & 0x07;
    if (exp == 0x0f && mant == 0x07) {
        return 0.0f;
    }
    if (exp == 0) {
        if (mant == 0) {
            return 0.0f;
        }
        return sign * std::ldexp((float) mant, -9);
    }
    return sign * std::ldexp(1.0f + (float) mant / 8.0f, exp - 7);
}

static uint8_t encode_fp8_e4m3fn_best(float x, float scale = 1.0f) {
    x *= scale;
    if (x == 0.0f) {
        return 0x00;
    }

    uint8_t best_code = 0x00;
    float best_err = 1.0e30f;
    for (int code = 0; code < 256; ++code) {
        if (((code >> 3) & 0x0f) == 0x0f && (code & 0x07) == 0x07) {
            continue;
        }
        if (code == 0x80) {
            continue;
        }
        const float y = decode_fp8_e4m3fn((uint8_t) code);
        const float err = std::fabs(y - x);
        if (err < best_err) {
            best_err = err;
            best_code = (uint8_t) code;
        }
    }
    return best_code;
}

static const std::array<uint8_t, 65536> & fp8_encode_lut_f16(float scale) {
    static const std::array<uint8_t, 65536> lut_scale1 = []() {
        std::array<uint8_t, 65536> lut{};
        for (size_t i = 0; i < lut.size(); ++i) {
            lut[i] = encode_fp8_e4m3fn_best(ggml_fp16_to_fp32((ggml_fp16_t) i), 1.0f);
        }
        return lut;
    }();
    static const std::array<uint8_t, 65536> lut_scale256 = []() {
        std::array<uint8_t, 65536> lut{};
        for (size_t i = 0; i < lut.size(); ++i) {
            lut[i] = encode_fp8_e4m3fn_best(ggml_fp16_to_fp32((ggml_fp16_t) i), 256.0f);
        }
        return lut;
    }();

    if (scale == 1.0f) {
        return lut_scale1;
    }
    if (scale == 256.0f) {
        return lut_scale256;
    }
    GGML_ABORT("unsupported scale for fp8 LUT");
}

static inline uint8_t encode_fp8_e4m3fn_best_lut(float x, float scale) {
    if (x == 0.0f) {
        return 0x00;
    }
    const auto & lut = fp8_encode_lut_f16(scale);
    return lut[(uint16_t) ggml_fp32_to_fp16(x)];
}

static std::array<uint8_t, 16> build_periodic16_small_code_lut(float scale) {
    std::array<uint8_t, 16> lut{};
    static const std::array<float, 16> values = {
        0.0f,   0.5f,   1.0f,   1.5f,
       -0.5f,  -1.0f,  -1.5f,   0.25f,
       -0.25f,  0.75f, -0.75f,  1.25f,
       -1.25f,  0.125f,-0.125f, 0.0f,
    };
    for (int idx = 0; idx < 16; ++idx) {
        lut[idx] = encode_fp8_e4m3fn_best(values[idx], scale);
    }
    return lut;
}

static void pack_wf8_from_kn_rowmajor(
        std::vector<uint8_t> & dst,
        const std::vector<float> & weight_kn,
        int k,
        int n,
        float scale) {
    dst.assign((size_t) k * (size_t) n, 0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            dst[permuted_weight_index(k, i, j)] =
                encode_fp8_e4m3fn_best(weight_kn[(size_t) i * (size_t) n + (size_t) j], scale);
        }
    }
}

static void pack_wf8_from_ggml_layout(
        std::vector<uint8_t> & dst,
        const std::vector<float> & weight_ggml,
        int k,
        int n,
        float scale) {
    dst.assign((size_t) k * (size_t) n, 0);
    for (int j = 0; j < n; ++j) {
        const size_t row_off = (size_t) j * (size_t) k;
        for (int i = 0; i < k; ++i) {
            dst[permuted_weight_index(k, i, j)] =
                encode_fp8_e4m3fn_best(weight_ggml[row_off + (size_t) i], scale);
        }
    }
}

static void pack_wf8_from_ggml_layout_periodic16_small(
        std::vector<uint8_t> & dst,
        int k,
        int n,
        float scale) {
    dst.assign((size_t) k * (size_t) n, 0);
    const auto lut = build_periodic16_small_code_lut(scale);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            const int idx = (((i & 15) * 7) + ((j & 15) * 3)) & 15;
            dst[permuted_weight_index(k, i, j)] = lut[idx];
        }
    }
}

static void pack_wf8_constant(
        std::vector<uint8_t> & dst,
        int k,
        int n,
        float value,
        float scale) {
    dst.assign((size_t) k * (size_t) n, encode_fp8_e4m3fn_best(value, scale));
}

static void write_tensor_dump_meta(const fs::path & path, const char * type, int64_t ne0, int64_t ne1, size_t nbytes) {
    std::ofstream ofs(path, std::ios::binary);
    ofs << "name=dump\n";
    ofs << "op=NONE\n";
    ofs << "src0=\n";
    ofs << "src1=\n";
    ofs << "type=" << type << "\n";
    ofs << "nbytes=" << nbytes << "\n";
    ofs << "ne=" << ne0 << "," << ne1 << ",1,1\n";
    ofs << "nb=4," << (ne0 * 4) << ",0,0\n";
}

static float make_activation_value(size_t idx) {
    const int v = (int) ((idx * 13 + 5) % 29) - 14;
    return 0.0625f + 0.125f * (float) v;
}

static float make_weight_value_pattern(int i, int j) {
    const int v = (int) (((size_t) i * 17 + (size_t) j * 31 + 7) % 37) - 18;
    return 0.125f + 0.25f * (float) v;
}

static float make_weight_value_periodic16_small(int i, int j) {
    static const std::array<float, 16> lut = {
        0.0f,   0.5f,   1.0f,   1.5f,
       -0.5f,  -1.0f,  -1.5f,   0.25f,
       -0.25f,  0.75f, -0.75f,  1.25f,
       -1.25f,  0.125f,-0.125f, 0.0f,
    };
    return lut[(((i & 15) * 7) + ((j & 15) * 3)) & 15];
}

int main(int argc, char ** argv) {
    int argi = 1;
    bool const_w1 = false;
    bool hmx_contract = false;
    bool periodic16_small = false;
    bool wf8_only = false;
    while (argi < argc && argv[argi][0] == '-') {
        if (std::strcmp(argv[argi], "--const-w1") == 0) {
            const_w1 = true;
        } else if (std::strcmp(argv[argi], "--hmx-contract") == 0) {
            hmx_contract = true;
        } else if (std::strcmp(argv[argi], "--periodic16-small") == 0) {
            periodic16_small = true;
        } else if (std::strcmp(argv[argi], "--wf8-only") == 0) {
            wf8_only = true;
        } else {
            std::fprintf(stderr, "unknown flag: %s\n", argv[argi]);
            return 1;
        }
        ++argi;
    }

    const int m = argc > argi + 0 ? std::atoi(argv[argi + 0]) : 4096;
    const int k = argc > argi + 1 ? std::atoi(argv[argi + 1]) : 4096;
    const int n = argc > argi + 2 ? std::atoi(argv[argi + 2]) : 12288;
    const fs::path out_dir =
        argc > argi + 3 ? fs::path(argv[argi + 3]) : fs::path("/tmp/wf8_s1_gguf_check");

    if (m <= 0 || k <= 0 || n <= 0 || (k % 32) != 0 || (n % 32) != 0) {
        std::fprintf(stderr, "usage: %s [--const-w1] [--periodic16-small] [--hmx-contract] [m=4096] [k=4096] [n=12288] [out_dir]\n", argv[0]);
        return 1;
    }

    fs::create_directories(out_dir);

    std::vector<float> weight_kn;
    std::vector<float> weight_ggml((size_t) k * (size_t) n, 0.0f);
    if (const_w1) {
        std::fill(weight_ggml.begin(), weight_ggml.end(), 1.0f);
    } else {
        weight_kn.assign((size_t) k * (size_t) n, 0.0f);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                const float val = periodic16_small ? make_weight_value_periodic16_small(i, j)
                                                   : make_weight_value_pattern(i, j);
                weight_kn[(size_t) i * (size_t) n + (size_t) j] = val;
                // ggml [k, n] tensor physical layout: data[j*k + i] = logical W[k_idx=i, n_idx=j].
                weight_ggml[(size_t) j * (size_t) k + (size_t) i] = val;
            }
        }
    }

    const float wf8_scale = hmx_contract ? 256.0f : 1.0f;
    std::vector<uint8_t> wf8_from_kn;
    std::vector<uint8_t> wf8_from_ggml;
    if (const_w1) {
        pack_wf8_constant(wf8_from_kn, k, n, 1.0f, wf8_scale);
        pack_wf8_constant(wf8_from_ggml, k, n, 1.0f, wf8_scale);
    } else if (periodic16_small) {
        pack_wf8_from_ggml_layout_periodic16_small(wf8_from_kn, k, n, wf8_scale);
        pack_wf8_from_ggml_layout_periodic16_small(wf8_from_ggml, k, n, wf8_scale);
    } else {
        pack_wf8_from_kn_rowmajor(wf8_from_kn, weight_kn, k, n, wf8_scale);
        pack_wf8_from_ggml_layout(wf8_from_ggml, weight_ggml, k, n, wf8_scale);
    }
    if (wf8_from_kn != wf8_from_ggml) {
        std::fprintf(stderr, "wf8 contract mismatch between KN and ggml views\n");
        return 5;
    }
    std::vector<uint8_t> & wf8 = wf8_from_kn;

    std::vector<float> activation;
    std::vector<float> ref;
    if (!wf8_only) {
        activation.resize((size_t) m * (size_t) k);
        for (size_t idx = 0; idx < activation.size(); ++idx) {
            activation[idx] = make_activation_value(idx);
        }

        ref.assign((size_t) m * (size_t) n, 0.0f);
        if (const_w1) {
            for (int row = 0; row < m; ++row) {
                const float * act_row = activation.data() + (size_t) row * (size_t) k;
                const float sum = std::accumulate(act_row, act_row + k, 0.0f);
                std::fill(ref.begin() + (size_t) row * (size_t) n,
                          ref.begin() + (size_t) (row + 1) * (size_t) n,
                          sum);
            }
        } else if (periodic16_small) {
            for (int row = 0; row < m; ++row) {
                const float * act_row = activation.data() + (size_t) row * (size_t) k;
                float sum_mod16[16] = {0};
                for (int kk = 0; kk < k; ++kk) {
                    sum_mod16[kk & 15] += act_row[kk];
                }
                float out_mod16[16] = {0};
                for (int jm = 0; jm < 16; ++jm) {
                    float sum = 0.0f;
                    for (int km = 0; km < 16; ++km) {
                        sum += sum_mod16[km] * make_weight_value_periodic16_small(km, jm);
                    }
                    out_mod16[jm] = sum;
                }
                float * ref_row = ref.data() + (size_t) row * (size_t) n;
                for (int col = 0; col < n; ++col) {
                    ref_row[col] = out_mod16[col & 15];
                }
            }
        } else {
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < n; ++col) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < k; ++kk) {
                        sum += activation[(size_t) row * (size_t) k + (size_t) kk] *
                               weight_kn[(size_t) kk * (size_t) n + (size_t) col];
                    }
                    ref[(size_t) row * (size_t) n + (size_t) col] = sum;
                }
            }
        }
    }

    const fs::path gguf_path = out_dir / "s1_wf8_hmx.gguf";
    const fs::path gguf_f16_src_path = out_dir / "s1_f16_src.gguf";
    const fs::path gguf_q4_hvx_path = out_dir / "s1_q4_0_hvx.gguf";
    const fs::path wf8_bin_path = out_dir / "s1_wf8_hmx.weight.bin";
    const fs::path act_bin_path = out_dir / "s1_wf8_hmx.act.bin";
    const fs::path act_meta_path = out_dir / "s1_wf8_hmx.act.meta.txt";
    const fs::path ref_bin_path = out_dir / "s1_wf8_hmx.ref.bin";
    const fs::path ref_meta_path = out_dir / "s1_wf8_hmx.ref.meta.txt";
    std::vector<ggml_fp16_t> weight_f16_ggml((size_t) k * (size_t) n);
    for (size_t idx = 0; idx < weight_ggml.size(); ++idx) {
        weight_f16_ggml[idx] = ggml_fp32_to_fp16(weight_ggml[idx]);
    }

    std::vector<uint8_t> weight_q4_hvx;
    if (!wf8_only) {
        const size_t q4_std_bytes = ggml_row_size(GGML_TYPE_Q4_0, k) * (size_t) n;
        std::vector<uint8_t> weight_q4_std(q4_std_bytes);
        weight_q4_hvx.resize(q4_std_bytes);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, weight_ggml.data(), weight_q4_std.data(),
                            0, (int64_t) weight_ggml.size(), k, nullptr);
        repack_q4_0_super_block_hvx((const block_q4_0 *) weight_q4_std.data(), weight_q4_hvx.data(), q4_std_bytes);
    }

    {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        gguf_context * gguf = gguf_init_empty();

        gguf_set_val_str(gguf, "general.architecture", "llama");
        gguf_set_val_str(gguf, "general.type", "test");
        gguf_set_val_str(gguf, "general.name", "S1_WF8_HMX_GEMM");
        gguf_set_val_u32(gguf, "general.alignment", GGUF_DEFAULT_ALIGNMENT);
        gguf_set_val_u32(gguf, "llama.context_length", (uint32_t) 32);
        gguf_set_val_u32(gguf, "llama.embedding_length", (uint32_t) 32);
        gguf_set_val_u32(gguf, "llama.block_count", (uint32_t) 1);
        gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1.0e-5f);
        gguf_set_val_u32(gguf, "s1.m", (uint32_t) m);
        gguf_set_val_u32(gguf, "s1.k", (uint32_t) k);
        gguf_set_val_u32(gguf, "s1.n", (uint32_t) n);

        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_WF8_HMX_PREPACK, k, n);
        ggml_set_name(tensor, "wf8.test.weight");
        tensor->data = wf8.data();
        gguf_add_tensor(gguf, tensor);
        gguf_set_tensor_data(gguf, tensor->name, wf8.data());
        gguf_write_to_file(gguf, gguf_path.string().c_str(), false);

        gguf_free(gguf);
        ggml_free(ctx);
    }

    if (!wf8_only) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        gguf_context * gguf = gguf_init_empty();

        gguf_set_val_str(gguf, "general.architecture", "llama");
        gguf_set_val_str(gguf, "general.type", "test");
        gguf_set_val_str(gguf, "general.name", "S1_Q4_0_HVX_GEMM");
        gguf_set_val_u32(gguf, "general.alignment", GGUF_DEFAULT_ALIGNMENT);
        gguf_set_val_u32(gguf, "llama.context_length", (uint32_t) 32);
        gguf_set_val_u32(gguf, "llama.embedding_length", (uint32_t) 32);
        gguf_set_val_u32(gguf, "llama.block_count", (uint32_t) 1);
        gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1.0e-5f);
        gguf_set_val_u32(gguf, "s1.m", (uint32_t) m);
        gguf_set_val_u32(gguf, "s1.k", (uint32_t) k);
        gguf_set_val_u32(gguf, "s1.n", (uint32_t) n);

        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, n);
        ggml_set_name(tensor, "wf8.test.weight");
        tensor->data = weight_q4_hvx.data();
        gguf_add_tensor(gguf, tensor);
        gguf_set_tensor_data(gguf, tensor->name, weight_q4_hvx.data());
        gguf_write_to_file(gguf, gguf_q4_hvx_path.string().c_str(), false);

        gguf_free(gguf);
        ggml_free(ctx);
    }

    if (!wf8_only) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        gguf_context * gguf = gguf_init_empty();

        gguf_set_val_str(gguf, "general.architecture", "llama");
        gguf_set_val_str(gguf, "general.type", "test");
        gguf_set_val_str(gguf, "general.name", "S1_F16_SRC_GEMM");
        gguf_set_val_u32(gguf, "general.alignment", GGUF_DEFAULT_ALIGNMENT);
        gguf_set_val_u32(gguf, "llama.context_length", (uint32_t) 32);
        gguf_set_val_u32(gguf, "llama.embedding_length", (uint32_t) 32);
        gguf_set_val_u32(gguf, "llama.block_count", (uint32_t) 1);
        gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1.0e-5f);
        gguf_set_val_u32(gguf, "s1.m", (uint32_t) m);
        gguf_set_val_u32(gguf, "s1.k", (uint32_t) k);
        gguf_set_val_u32(gguf, "s1.n", (uint32_t) n);

        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
        ggml_set_name(tensor, "wf8.test.weight");
        tensor->data = weight_f16_ggml.data();
        gguf_add_tensor(gguf, tensor);
        gguf_set_tensor_data(gguf, tensor->name, weight_f16_ggml.data());
        gguf_write_to_file(gguf, gguf_f16_src_path.string().c_str(), false);

        gguf_free(gguf);
        ggml_free(ctx);
    }

    if (!wf8_only) {
        gguf_init_params params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ nullptr,
        };
        ggml_context * data_ctx = nullptr;
        params.ctx = &data_ctx;
        gguf_context * gguf = gguf_init_from_file(gguf_path.string().c_str(), params);
        if (gguf == nullptr || data_ctx == nullptr) {
            std::fprintf(stderr, "failed to reload gguf: %s\n", gguf_path.string().c_str());
            return 2;
        }
        const int tid = gguf_find_tensor(gguf, "wf8.test.weight");
        if (tid < 0) {
            std::fprintf(stderr, "tensor not found in gguf\n");
            gguf_free(gguf);
            ggml_free(data_ctx);
            return 3;
        }
        ggml_tensor * tensor = ggml_get_first_tensor(data_ctx);
        while (tensor && std::strcmp(tensor->name, "wf8.test.weight") != 0) {
            tensor = ggml_get_next_tensor(data_ctx, tensor);
        }
        if (tensor == nullptr || tensor->data == nullptr) {
            std::fprintf(stderr, "failed to find tensor data in reloaded gguf\n");
            gguf_free(gguf);
            ggml_free(data_ctx);
            return 4;
        }
        std::ofstream wf8_out(wf8_bin_path, std::ios::binary);
        wf8_out.write((const char *) tensor->data, (std::streamsize) wf8.size());
        wf8_out.close();
        gguf_free(gguf);
        ggml_free(data_ctx);
    }

    if (!wf8_only) {
        std::ofstream ofs(act_bin_path, std::ios::binary);
        ofs.write((const char *) activation.data(), (std::streamsize) (activation.size() * sizeof(float)));
    }
    if (!wf8_only) {
        write_tensor_dump_meta(act_meta_path, "f32", k, m, activation.size() * sizeof(float));
    }

    if (!wf8_only) {
        std::ofstream ofs(ref_bin_path, std::ios::binary);
        ofs.write((const char *) ref.data(), (std::streamsize) (ref.size() * sizeof(float)));
    }
    if (!wf8_only) {
        write_tensor_dump_meta(ref_meta_path, "f32", n, m, ref.size() * sizeof(float));
    }

    const uintmax_t gguf_size = fs::file_size(gguf_path);
    const double fp16_bytes = (double) k * (double) n * (double) sizeof(ggml_fp16_t);
    std::printf("wf8_s1_gguf wf8_gguf=%s q4_hvx_gguf=%s f16_src_gguf=%s weight_bin=%s act_bin=%s ref_bin=%s gguf_bytes=%ju fp16_weight_bytes=%.0f compression=%.6fx first_ref=%g wf8_only=%d\n",
                gguf_path.string().c_str(),
                gguf_q4_hvx_path.string().c_str(),
                gguf_f16_src_path.string().c_str(),
                wf8_bin_path.string().c_str(),
                act_bin_path.string().c_str(),
                ref_bin_path.string().c_str(),
                (uintmax_t) gguf_size,
                fp16_bytes,
                fp16_bytes / (double) gguf_size,
                ref.empty() ? 0.0f : ref[0],
                wf8_only ? 1 : 0);
    return 0;
}
