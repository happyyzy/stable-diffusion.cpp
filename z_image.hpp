#ifndef __Z_IMAGE_HPP__
#define __Z_IMAGE_HPP__

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "flux.hpp"
#include "ggml_extend.hpp"
#include "mmdit.hpp"

// Ref: https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/models/model.py
// Ref: https://github.com/huggingface/diffusers/pull/12703

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace ZImage {
    constexpr int Z_IMAGE_GRAPH_SIZE = 20480;
    constexpr int ADALN_EMBED_DIM    = 256;
    constexpr int SEQ_MULTI_OF       = 32;

    static inline bool zimg_dump_l0_enabled() {
        static int enabled = -1;
        if (enabled < 0) {
            const char* env = std::getenv("SD_ZIMG_DUMP_L0");
            enabled         = (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
        }
        return enabled != 0;
    }

    static inline bool zimg_dump_snapshot_enabled() {
        static int enabled = -1;
        if (enabled < 0) {
            const char* env = std::getenv("SD_ZIMG_DUMP_SNAPSHOT");
            enabled         = (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
        }
        return enabled != 0;
    }

    static inline bool zimg_force_split_qk_proj_enabled() {
        static int enabled = -1;
        if (enabled < 0) {
            const char* env = std::getenv("SD_ZIMG_FORCE_SPLIT_QK_PROJ");
            enabled         = (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
        }
        return enabled != 0;
    }

    static inline int zimg_max_main_layers() {
        static int max_layers = -2;
        if (max_layers == -2) {
            max_layers = -1;
            const char* env = std::getenv("SD_ZIMG_MAX_MAIN_LAYERS");
            if (env != nullptr && env[0] != '\0') {
                int v = std::atoi(env);
                if (v > 0) {
                    max_layers = v;
                    LOG_INFO("z_image: SD_ZIMG_MAX_MAIN_LAYERS=%d", max_layers);
                }
            }
        }
        return max_layers;
    }

    static inline int zimg_dump_main_layer() {
        // Default to layer0 to preserve legacy behavior; allow overriding for late-layer drift debugging.
        static int dump_layer = -2;
        if (dump_layer == -2) {
            dump_layer = 0;
            const char* env = std::getenv("SD_ZIMG_DUMP_MAIN_LAYER");
            if (env != nullptr && env[0] != '\0') {
                dump_layer = std::atoi(env);
                if (dump_layer < 0) {
                    dump_layer = 0;
                }
                LOG_INFO("z_image: SD_ZIMG_DUMP_MAIN_LAYER=%d", dump_layer);
            }
        }
        return dump_layer;
    }

    static inline const std::vector<std::string>& zimg_dump_filter_tokens() {
        static const std::vector<std::string> tokens = []() {
            std::vector<std::string> out;
            const char* env = std::getenv("SD_ZIMG_DUMP_FILTER");
            if (env == nullptr || env[0] == '\0') {
                return out;
            }
            std::string s(env);
            for (char& c : s) {
                if (c == ';' || c == ' ') {
                    c = ',';
                } else {
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
            }
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) {
                    out.push_back(tok);
                }
            }
            return out;
        }();
        return tokens;
    }

    static inline bool zimg_dump_name_allowed(const std::string& name) {
        const auto& tokens = zimg_dump_filter_tokens();
        if (tokens.empty()) {
            return true;
        }
        std::string lower(name);
        for (char& c : lower) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        for (const auto& t : tokens) {
            if (!t.empty() && lower.find(t) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    static inline void zimg_cache_tensor(GGMLRunnerContext* ctx, const std::string& name, struct ggml_tensor* tensor) {
        if (!zimg_dump_l0_enabled() || ctx == nullptr || ctx->runner == nullptr || tensor == nullptr ||
            !zimg_dump_name_allowed(name)) {
            return;
        }
        struct ggml_tensor* to_cache = tensor;
        if (zimg_dump_snapshot_enabled() && ctx->ggml_ctx != nullptr) {
            // Snapshot: force a materialized copy node so we can dump stable intermediate values.
            to_cache = ggml_dup_tensor(ctx->ggml_ctx, tensor);
            GGML_ASSERT(to_cache != nullptr);
            to_cache = ggml_cpy(ctx->ggml_ctx, tensor, to_cache);
            GGML_ASSERT(to_cache != nullptr);
        }
        ctx->runner->cache(name, to_cache);
    }

    static inline bool zimg_dump_tensor_to_file(const std::string& path, const char* name, const ggml_tensor* tensor) {
        if (tensor == nullptr) {
            LOG_ERROR("z_image dump tensor failed: tensor is null");
            return false;
        }

        FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            LOG_ERROR("z_image dump tensor failed: cannot open '%s'", path.c_str());
            return false;
        }

        const int32_t n_dims   = ggml_n_dims(tensor);
        const int32_t name_len = name ? static_cast<int32_t>(std::strlen(name)) : 0;
        const int32_t ttype    = static_cast<int32_t>(tensor->type);

        std::fwrite(&n_dims, sizeof(n_dims), 1, f);
        std::fwrite(&name_len, sizeof(name_len), 1, f);
        std::fwrite(&ttype, sizeof(ttype), 1, f);
        for (int i = 0; i < n_dims; ++i) {
            const int32_t ne = static_cast<int32_t>(tensor->ne[i]);
            std::fwrite(&ne, sizeof(ne), 1, f);
        }
        if (name_len > 0) {
            std::fwrite(name, 1, name_len, f);
        }

        const size_t nbytes = ggml_nbytes(tensor);
        std::vector<uint8_t> tmp;
        const bool want_dense = (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16);
        if (want_dense && !ggml_is_contiguous(tensor)) {
            // Dump as a dense contiguous array even when the runtime tensor uses a strided layout.
            // This keeps the on-disk format stable (dims+raw dense data) for Python tooling.
            const bool has_backend_buf = (tensor->buffer != nullptr && tensor->data == nullptr);
            const bool has_host_data   = (tensor->data != nullptr);
            if (!has_backend_buf && !has_host_data) {
                std::fclose(f);
                LOG_ERROR("z_image dump tensor failed: tensor has no data");
                return false;
            }
            const size_t elem_size = ggml_type_size(tensor->type);
            const int64_t ne0 = tensor->ne[0];
            const int64_t ne1 = tensor->ne[1];
            const int64_t ne2 = tensor->ne[2];
            const int64_t ne3 = tensor->ne[3];
            const size_t row_bytes = static_cast<size_t>(ne0) * elem_size;
            const size_t dense_bytes = static_cast<size_t>(ggml_nelements(tensor)) * elem_size;
            tmp.resize(dense_bytes);

            // Fast-path: dim0 contiguous.
            const bool row_contig = (tensor->nb[0] == elem_size);
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        const size_t src_off =
                            static_cast<size_t>(i1) * tensor->nb[1] +
                            static_cast<size_t>(i2) * tensor->nb[2] +
                            static_cast<size_t>(i3) * tensor->nb[3];
                        const size_t dst_off =
                            static_cast<size_t>((i3 * ne2 * ne1 + i2 * ne1 + i1) * ne0) * elem_size;

                        if (row_contig) {
                            if (has_backend_buf) {
                                ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), tmp.data() + dst_off, src_off, row_bytes);
                            } else if (has_host_data) {
                                const uint8_t * src = static_cast<const uint8_t *>(tensor->data) + src_off;
                                std::memcpy(tmp.data() + dst_off, src, row_bytes);
                            }
                        } else {
                            // Slow path: gather element-by-element when even dim0 is strided.
                            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                                const size_t src_off_i0 = src_off + static_cast<size_t>(i0) * tensor->nb[0];
                                const size_t dst_off_i0 = dst_off + static_cast<size_t>(i0) * elem_size;
                                if (has_backend_buf) {
                                    ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), tmp.data() + dst_off_i0, src_off_i0, elem_size);
                                } else if (has_host_data) {
                                    const uint8_t * src = static_cast<const uint8_t *>(tensor->data) + src_off_i0;
                                    std::memcpy(tmp.data() + dst_off_i0, src, elem_size);
                                }
                            }
                        }
                    }
                }
            }

            std::fwrite(tmp.data(), 1, dense_bytes, f);
        } else if (tensor->buffer != nullptr && tensor->data == nullptr) {
            tmp.resize(nbytes);
            ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), tmp.data(), 0, nbytes);
            std::fwrite(tmp.data(), 1, nbytes, f);
        } else if (tensor->data != nullptr) {
            std::fwrite(tensor->data, 1, nbytes, f);
        } else {
            std::fclose(f);
            LOG_ERROR("z_image dump tensor failed: tensor has no data");
            return false;
        }

        std::fflush(f);
        std::fclose(f);
        return true;
    }

    struct JointAttention : public GGMLBlock {
    protected:
        int64_t hidden_size;
        int64_t head_dim;
        int64_t num_heads;
        int64_t num_kv_heads;
        bool qk_norm;
        bool use_split_qkv;
        bool use_split_qk_v;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            blocks.clear();
            use_split_qkv = false;

            const bool has_q_proj = tensor_storage_map.find(prefix + "q_proj.weight") != tensor_storage_map.end();
            const bool has_k_proj = tensor_storage_map.find(prefix + "k_proj.weight") != tensor_storage_map.end();
            const bool has_v_proj = tensor_storage_map.find(prefix + "v_proj.weight") != tensor_storage_map.end();
            const bool has_qk_proj = tensor_storage_map.find(prefix + "qk_proj.weight") != tensor_storage_map.end();
            if (has_q_proj && has_k_proj && has_v_proj) {
                use_split_qkv   = true;
                use_split_qk_v  = false;
                blocks["q_proj"] = std::make_shared<Linear>(hidden_size, num_heads * head_dim, false);
                blocks["k_proj"] = std::make_shared<Linear>(hidden_size, num_kv_heads * head_dim, false);
                blocks["v_proj"] = std::make_shared<Linear>(hidden_size, num_kv_heads * head_dim, false);
            } else if (has_qk_proj && has_v_proj) {
                use_split_qkv  = false;
                use_split_qk_v = true;
                blocks["qk_proj"] = std::make_shared<Linear>(hidden_size, (num_heads + num_kv_heads) * head_dim, false);
                blocks["v_proj"]  = std::make_shared<Linear>(hidden_size, num_kv_heads * head_dim, false);
            } else {
                use_split_qk_v = false;
                blocks["qkv"] = std::make_shared<Linear>(hidden_size, (num_heads + num_kv_heads * 2) * head_dim, false);
            }

            float scale = 1.f;
#if GGML_USE_HIP
            // Prevent NaN issues with certain ROCm setups
            scale = 1.f / 16.f;
#endif
            blocks["out"] = std::make_shared<Linear>(num_heads * head_dim, hidden_size, false, false, false, scale);
            if (qk_norm) {
                blocks["q_norm"] = std::make_shared<RMSNorm>(head_dim);
                blocks["k_norm"] = std::make_shared<RMSNorm>(head_dim);
            }
        }

    public:
        JointAttention(int64_t hidden_size, int64_t head_dim, int64_t num_heads, int64_t num_kv_heads, bool qk_norm)
            : hidden_size(hidden_size),
              head_dim(head_dim),
              num_heads(num_heads),
              num_kv_heads(num_kv_heads),
              qk_norm(qk_norm),
              use_split_qkv(false),
              use_split_qk_v(false) {}

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* pe_pack = nullptr,
                                    struct ggml_tensor* mask = nullptr,
                                    const std::string& dump_prefix = "") {
            // x: [N, n_token, hidden_size]
            int64_t n_token = x->ne[1];
            int64_t N       = x->ne[2];
            auto out_proj   = std::dynamic_pointer_cast<Linear>(blocks["out"]);

            struct ggml_tensor* q = nullptr;
            struct ggml_tensor* k = nullptr;
            struct ggml_tensor* v = nullptr;

            if (use_split_qkv) {
                auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q_proj"]);
                auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k_proj"]);
                auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
                GGML_ASSERT(q_proj != nullptr && k_proj != nullptr && v_proj != nullptr);

                auto q_linear = q_proj->forward(ctx, x);  // [N, n_token, num_heads*head_dim]
                auto k_linear = k_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
                auto v_linear = v_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]

                if (!dump_prefix.empty()) {
                    auto qk = ggml_concat(ctx->ggml_ctx, q_linear, k_linear, 0);
                    auto qkv_linear = ggml_concat(ctx->ggml_ctx, qk, v_linear, 0);
                    zimg_cache_tensor(ctx, dump_prefix + "qkv_linear", qkv_linear);
                }

                q = ggml_reshape_4d(ctx->ggml_ctx, q_linear, head_dim, num_heads, n_token, N);        // [N, n_token, num_heads, head_dim]
                k = ggml_reshape_4d(ctx->ggml_ctx, k_linear, head_dim, num_kv_heads, n_token, N);     // [N, n_token, num_kv_heads, head_dim]
                v = ggml_reshape_4d(ctx->ggml_ctx, v_linear, head_dim, num_kv_heads, n_token, N);     // [N, n_token, num_kv_heads, head_dim]
            } else if (use_split_qk_v) {
                auto qk_proj = std::dynamic_pointer_cast<Linear>(blocks["qk_proj"]);
                auto v_proj  = std::dynamic_pointer_cast<Linear>(blocks["v_proj"]);
                GGML_ASSERT(qk_proj != nullptr && v_proj != nullptr);

                struct ggml_tensor* qk_linear = nullptr;
                struct ggml_tensor* v_linear  = nullptr;

                // Debug/bring-up switch: split qk_proj matmul into q_proj + k_proj matmuls.
                // This bypasses the fused qk_proj(q4) path while preserving math equivalence.
                if (zimg_force_split_qk_proj_enabled() && ctx->weight_adapter == nullptr) {
                    std::map<std::string, struct ggml_tensor*> qk_params;
                    qk_proj->get_param_tensors(qk_params);
                    auto w_it = qk_params.find("weight");
                    GGML_ASSERT(w_it != qk_params.end() && w_it->second != nullptr);
                    auto* qk_weight = w_it->second;

                    const int64_t q_out = num_heads * head_dim;
                    const int64_t k_out = num_kv_heads * head_dim;
                    GGML_ASSERT(qk_weight->ne[1] == q_out + k_out);

                    auto q_weight = ggml_view_2d(ctx->ggml_ctx,
                                                 qk_weight,
                                                 qk_weight->ne[0],
                                                 q_out,
                                                 qk_weight->nb[1],
                                                 0);
                    auto k_weight = ggml_view_2d(ctx->ggml_ctx,
                                                 qk_weight,
                                                 qk_weight->ne[0],
                                                 k_out,
                                                 qk_weight->nb[1],
                                                 q_out * qk_weight->nb[1]);
                    if (qk_weight->name[0] != '\0') {
                        std::string q_name(qk_weight->name);
                        std::string k_name(qk_weight->name);
                        const std::string key = ".attention.qk_proj.weight";
                        auto pos = q_name.find(key);
                        if (pos != std::string::npos) {
                            q_name.replace(pos, key.size(), ".attention.q_proj.weight");
                            k_name.replace(pos, key.size(), ".attention.k_proj.weight");
                        } else {
                            q_name += ".q_split";
                            k_name += ".k_split";
                        }
                        ggml_set_name(q_weight, q_name.c_str());
                        ggml_set_name(k_weight, k_name.c_str());
                    }

                    auto q_linear = ggml_ext_linear(ctx->ggml_ctx, x, q_weight, nullptr);
                    auto k_linear = ggml_ext_linear(ctx->ggml_ctx, x, k_weight, nullptr);
                    qk_linear     = ggml_concat(ctx->ggml_ctx, q_linear, k_linear, 0);
                    v_linear      = v_proj->forward(ctx, x);  // [N, n_token, num_kv_heads*head_dim]
                } else {
                    qk_linear = qk_proj->forward(ctx, x);  // [N, n_token, (num_heads + num_kv_heads)*head_dim]
                    v_linear  = v_proj->forward(ctx, x);   // [N, n_token, num_kv_heads*head_dim]
                }

                if (!dump_prefix.empty()) {
                    auto qkv_linear = ggml_concat(ctx->ggml_ctx, qk_linear, v_linear, 0);
                    zimg_cache_tensor(ctx, dump_prefix + "qkv_linear", qkv_linear);
                }

                auto qk = ggml_reshape_4d(ctx->ggml_ctx, qk_linear, head_dim, num_heads + num_kv_heads, n_token, N);
                q       = ggml_view_4d(ctx->ggml_ctx,
                                 qk,
                                 qk->ne[0],
                                 num_heads,
                                 qk->ne[2],
                                 qk->ne[3],
                                 qk->nb[1],
                                 qk->nb[2],
                                 qk->nb[3],
                                 0);
                k       = ggml_view_4d(ctx->ggml_ctx,
                                 qk,
                                 qk->ne[0],
                                 num_kv_heads,
                                 qk->ne[2],
                                 qk->ne[3],
                                 qk->nb[1],
                                 qk->nb[2],
                                 qk->nb[3],
                                 num_heads * qk->nb[1]);
                v       = ggml_reshape_4d(ctx->ggml_ctx, v_linear, head_dim, num_kv_heads, n_token, N);
            } else {
                auto qkv_proj = std::dynamic_pointer_cast<Linear>(blocks["qkv"]);
                GGML_ASSERT(qkv_proj != nullptr);
                auto qkv = qkv_proj->forward(ctx, x);                                                                            // [N, n_token, (num_heads + num_kv_heads*2)*head_dim]
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "qkv_linear", qkv);
                }
                qkv      = ggml_reshape_4d(ctx->ggml_ctx, qkv, head_dim, num_heads + num_kv_heads * 2, qkv->ne[1], qkv->ne[2]);  // [N, n_token, num_heads + num_kv_heads*2, head_dim]

                q = ggml_view_4d(ctx->ggml_ctx,
                                 qkv,
                                 qkv->ne[0],
                                 num_heads,
                                 qkv->ne[2],
                                 qkv->ne[3],
                                 qkv->nb[1],
                                 qkv->nb[2],
                                 qkv->nb[3],
                                 0);  // [N, n_token, num_heads, head_dim]
                k = ggml_view_4d(ctx->ggml_ctx,
                                 qkv,
                                 qkv->ne[0],
                                 num_kv_heads,
                                 qkv->ne[2],
                                 qkv->ne[3],
                                 qkv->nb[1],
                                 qkv->nb[2],
                                 qkv->nb[3],
                                 num_heads * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]
                v = ggml_view_4d(ctx->ggml_ctx,
                                 qkv,
                                 qkv->ne[0],
                                 num_kv_heads,
                                 qkv->ne[2],
                                 qkv->ne[3],
                                 qkv->nb[1],
                                 qkv->nb[2],
                                 qkv->nb[3],
                                 (num_heads + num_kv_heads) * qkv->nb[1]);  // [N, n_token, num_kv_heads, head_dim]
            }
            if (!dump_prefix.empty()) {
                zimg_cache_tensor(ctx, dump_prefix + "q_raw", q);
                zimg_cache_tensor(ctx, dump_prefix + "k_raw", k);
                zimg_cache_tensor(ctx, dump_prefix + "v_raw", v);
            }

            if (qk_norm) {
                auto q_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"]);
                auto k_norm = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"]);

                q = q_norm->forward(ctx, q);
                k = k_norm->forward(ctx, k);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "q_post_norm", q);
                    zimg_cache_tensor(ctx, dump_prefix + "k_post_norm", k);
                }
            }

            x = Rope::attention(ctx, q, k, v, pe, pe_pack, mask, 1.f / 128.f);  // [N, n_token, num_heads * head_dim]
            if (!dump_prefix.empty()) {
                zimg_cache_tensor(ctx, dump_prefix + "attn", x);
            }

            x = out_proj->forward(ctx, x);  // [N, n_token, hidden_size]
            if (!dump_prefix.empty()) {
                zimg_cache_tensor(ctx, dump_prefix + "attn_proj", x);
            }
            return x;
        }
    };

    class FeedForward : public GGMLBlock {
    public:
        FeedForward(int64_t dim,
                    int64_t hidden_dim,
                    int64_t multiple_of,
                    float ffn_dim_multiplier = 0.f) {
            if (ffn_dim_multiplier > 0.f) {
                hidden_dim = static_cast<int64_t>(ffn_dim_multiplier * hidden_dim);
            }
            hidden_dim   = multiple_of * ((hidden_dim + multiple_of - 1) / multiple_of);
            blocks["w1"] = std::make_shared<Linear>(dim, hidden_dim, false);

            bool force_prec_f32 = false;
            float scale         = 1.f / 128.f;
#ifdef SD_USE_VULKAN
            force_prec_f32 = true;
#endif
            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example, when using CUDA but the weights are k-quants.
            blocks["w2"] = std::make_shared<Linear>(hidden_dim, dim, false, false, force_prec_f32, scale);
            blocks["w3"] = std::make_shared<Linear>(dim, hidden_dim, false);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx, struct ggml_tensor* x, const std::string& dump_prefix = "") {
            auto w1 = std::dynamic_pointer_cast<Linear>(blocks["w1"]);
            auto w2 = std::dynamic_pointer_cast<Linear>(blocks["w2"]);
            auto w3 = std::dynamic_pointer_cast<Linear>(blocks["w3"]);

            auto x1 = w1->forward(ctx, x);
            auto x3 = w3->forward(ctx, x);
            if (!dump_prefix.empty()) {
                zimg_cache_tensor(ctx, dump_prefix + "w1", x1);
                zimg_cache_tensor(ctx, dump_prefix + "w3", x3);
            }
            x       = ggml_mul(ctx->ggml_ctx, ggml_silu(ctx->ggml_ctx, x1), x3);
            if (!dump_prefix.empty()) {
                zimg_cache_tensor(ctx, dump_prefix + "w1_silu_mul_w3", x);
            }
            x       = w2->forward(ctx, x);
            if (!dump_prefix.empty()) {
                zimg_cache_tensor(ctx, dump_prefix + "w2", x);
            }

            return x;
        }
    };

    __STATIC_INLINE__ struct ggml_tensor* modulate(struct ggml_context* ctx,
                                                   struct ggml_tensor* x,
                                                   struct ggml_tensor* scale) {
        // x: [N, L, C]
        // scale: [N, C]
        scale = ggml_reshape_3d(ctx, scale, scale->ne[0], 1, scale->ne[1]);  // [N, 1, C]
        x     = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        return x;
    }

    struct JointTransformerBlock : public GGMLBlock {
    protected:
        bool modulation;

    public:
        JointTransformerBlock(int layer_id,
                              int64_t hidden_size,
                              int64_t head_dim,
                              int64_t num_heads,
                              int64_t num_kv_heads,
                              int64_t multiple_of,
                              float ffn_dim_multiplier,
                              float norm_eps,
                              bool qk_norm,
                              bool modulation = true)
            : modulation(modulation) {
            blocks["attention"]       = std::make_shared<JointAttention>(hidden_size, head_dim, num_heads, num_kv_heads, qk_norm);
            blocks["feed_forward"]    = std::make_shared<FeedForward>(hidden_size, hidden_size, multiple_of, ffn_dim_multiplier);
            blocks["attention_norm1"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm1"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["attention_norm2"] = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            blocks["ffn_norm2"]       = std::make_shared<RMSNorm>(hidden_size, norm_eps);
            if (modulation) {
                blocks["adaLN_modulation.0"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), 4 * hidden_size);
            }
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* pe_pack    = nullptr,
                                    struct ggml_tensor* mask        = nullptr,
                                    struct ggml_tensor* adaln_input = nullptr,
                                    const std::string& dump_prefix  = "") {
            auto attention       = std::dynamic_pointer_cast<JointAttention>(blocks["attention"]);
            auto feed_forward    = std::dynamic_pointer_cast<FeedForward>(blocks["feed_forward"]);
            auto attention_norm1 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm1"]);
            auto ffn_norm1       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm1"]);
            auto attention_norm2 = std::dynamic_pointer_cast<RMSNorm>(blocks["attention_norm2"]);
            auto ffn_norm2       = std::dynamic_pointer_cast<RMSNorm>(blocks["ffn_norm2"]);

            if (modulation) {
                GGML_ASSERT(adaln_input != nullptr);
                auto adaLN_modulation_0 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.0"]);

                auto m         = adaLN_modulation_0->forward(ctx, adaln_input);  // [N, 4 * hidden_size]
                auto mods      = ggml_ext_chunk(ctx->ggml_ctx, m, 4, 0);
                auto scale_msa = mods[0];
                auto gate_msa  = mods[1];
                auto scale_mlp = mods[2];
                auto gate_mlp  = mods[3];
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "adaln_linear", m);
                    zimg_cache_tensor(ctx, dump_prefix + "scale_msa", scale_msa);
                    zimg_cache_tensor(ctx, dump_prefix + "gate_msa", gate_msa);
                    zimg_cache_tensor(ctx, dump_prefix + "scale_mlp", scale_mlp);
                    zimg_cache_tensor(ctx, dump_prefix + "gate_mlp", gate_mlp);
                }

                auto residual = x;
                x             = modulate(ctx->ggml_ctx, attention_norm1->forward(ctx, x), scale_msa);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "attn_mod_in", x);
                }
                x             = attention->forward(ctx, x, pe, pe_pack, mask, dump_prefix.empty() ? "" : (dump_prefix + "attn_"));
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_msa));
                x             = ggml_add(ctx->ggml_ctx, x, residual);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "attn_res", x);
                }

                residual = x;
                x        = modulate(ctx->ggml_ctx, ffn_norm1->forward(ctx, x), scale_mlp);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "ffn_mod_in", x);
                }
                x        = feed_forward->forward(ctx, x, dump_prefix.empty() ? "" : (dump_prefix + "ffn_"));
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_mul(ctx->ggml_ctx, x, ggml_tanh(ctx->ggml_ctx, gate_mlp));
                x        = ggml_add(ctx->ggml_ctx, x, residual);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "ffn_res", x);
                }
            } else {
                GGML_ASSERT(adaln_input == nullptr);

                auto residual = x;
                x             = attention_norm1->forward(ctx, x);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "attn_norm_in", x);
                }
                x             = attention->forward(ctx, x, pe, pe_pack, mask, dump_prefix.empty() ? "" : (dump_prefix + "attn_"));
                x             = attention_norm2->forward(ctx, x);
                x             = ggml_add(ctx->ggml_ctx, x, residual);

                residual = x;
                x        = ffn_norm1->forward(ctx, x);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "ffn_norm_in", x);
                }
                x        = feed_forward->forward(ctx, x, dump_prefix.empty() ? "" : (dump_prefix + "ffn_"));
                x        = ffn_norm2->forward(ctx, x);
                x        = ggml_add(ctx->ggml_ctx, x, residual);
            }

            if (!dump_prefix.empty()) {
                zimg_cache_tensor(ctx, dump_prefix + "out", x);
            }
            return x;
        }
    };

    struct FinalLayer : public GGMLBlock {
    public:
        FinalLayer(int64_t hidden_size,
                   int64_t patch_size,
                   int64_t out_channels) {
            blocks["norm_final"]         = std::make_shared<LayerNorm>(hidden_size, 1e-06f, false);
            blocks["linear"]             = std::make_shared<Linear>(hidden_size, patch_size * patch_size * out_channels, true, true);
            blocks["adaLN_modulation.1"] = std::make_shared<Linear>(MIN(hidden_size, ADALN_EMBED_DIM), hidden_size);
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]
            auto norm_final         = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            auto linear             = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto adaLN_modulation_1 = std::dynamic_pointer_cast<Linear>(blocks["adaLN_modulation.1"]);

            auto scale = adaLN_modulation_1->forward(ctx, ggml_silu(ctx->ggml_ctx, c));  // [N, hidden_size]
            x          = norm_final->forward(ctx, x);
            x          = modulate(ctx->ggml_ctx, x, scale);
            x          = linear->forward(ctx, x);

            return x;
        }
    };

    struct ZImageParams {
        int patch_size             = 2;
        int64_t hidden_size        = 3840;
        int64_t in_channels        = 16;
        int64_t out_channels       = 16;
        int64_t num_layers         = 30;
        int64_t num_refiner_layers = 2;
        int64_t head_dim           = 128;
        int64_t num_heads          = 30;
        int64_t num_kv_heads       = 30;
        int64_t multiple_of        = 256;
        float ffn_dim_multiplier   = 8.0f / 3.0f;
        float norm_eps             = 1e-5f;
        bool qk_norm               = true;
        int64_t cap_feat_dim       = 2560;
        int theta                  = 256;
        std::vector<int> axes_dim  = {32, 48, 48};
        int64_t axes_dim_sum       = 128;
    };

    class ZImageModel : public GGMLBlock {
    protected:
        ZImageParams z_image_params;

        void init_params(struct ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            params["cap_pad_token"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
            params["x_pad_token"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, z_image_params.hidden_size);
        }

    public:
        ZImageModel() = default;
        ZImageModel(ZImageParams z_image_params)
            : z_image_params(z_image_params) {
            blocks["x_embedder"]     = std::make_shared<Linear>(z_image_params.patch_size * z_image_params.patch_size * z_image_params.in_channels, z_image_params.hidden_size);
            blocks["t_embedder"]     = std::make_shared<TimestepEmbedder>(MIN(z_image_params.hidden_size, 1024), 256, 256);
            blocks["cap_embedder.0"] = std::make_shared<RMSNorm>(z_image_params.cap_feat_dim, z_image_params.norm_eps);
            blocks["cap_embedder.1"] = std::make_shared<Linear>(z_image_params.cap_feat_dim, z_image_params.hidden_size);

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["noise_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     false);

                blocks["context_refiner." + std::to_string(i)] = block;
            }

            for (int i = 0; i < z_image_params.num_layers; i++) {
                auto block = std::make_shared<JointTransformerBlock>(i,
                                                                     z_image_params.hidden_size,
                                                                     z_image_params.head_dim,
                                                                     z_image_params.num_heads,
                                                                     z_image_params.num_kv_heads,
                                                                     z_image_params.multiple_of,
                                                                     z_image_params.ffn_dim_multiplier,
                                                                     z_image_params.norm_eps,
                                                                     z_image_params.qk_norm,
                                                                     true);

                blocks["layers." + std::to_string(i)] = block;
            }

            blocks["final_layer"] = std::make_shared<FinalLayer>(z_image_params.hidden_size, z_image_params.patch_size, z_image_params.out_channels);
        }

        struct ggml_tensor* pad_to_patch_size(GGMLRunnerContext* ctx,
                                              struct ggml_tensor* x) {
            int64_t W = x->ne[0];
            int64_t H = x->ne[1];

            int pad_h = (z_image_params.patch_size - H % z_image_params.patch_size) % z_image_params.patch_size;
            int pad_w = (z_image_params.patch_size - W % z_image_params.patch_size) % z_image_params.patch_size;
            x         = ggml_ext_pad(ctx->ggml_ctx, x, pad_w, pad_h, 0, 0, ctx->circular_x_enabled, ctx->circular_y_enabled);
            return x;
        }

        struct ggml_tensor* patchify(struct ggml_context* ctx,
                                     struct ggml_tensor* x) {
            // x: [N, C, H, W]
            // return: [N, h*w, patch_size*patch_size*C]
            int64_t N = x->ne[3];
            int64_t C = x->ne[2];
            int64_t H = x->ne[1];
            int64_t W = x->ne[0];
            int64_t p = z_image_params.patch_size;
            int64_t h = H / z_image_params.patch_size;
            int64_t w = W / z_image_params.patch_size;

            GGML_ASSERT(h * p == H && w * p == W);

            x = ggml_reshape_4d(ctx, x, p, w, p, h * C * N);                 // [N*C*h, p, w, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));            // [N*C*h, w, p, p]
            x = ggml_reshape_4d(ctx, x, p * p, w * h, C, N);                 // [N, C, h*w, p*p]
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 2, 0, 1, 3));  // [N, h*w, C, p*p]
            x = ggml_reshape_3d(ctx, x, C * p * p, w * h, N);                // [N, h*w, p*p*C]
            return x;
        }

        struct ggml_tensor* process_img(GGMLRunnerContext* ctx,
                                        struct ggml_tensor* x) {
            x = pad_to_patch_size(ctx, x);
            x = patchify(ctx->ggml_ctx, x);
            return x;
        }

        struct ggml_tensor* unpatchify(struct ggml_context* ctx,
                                       struct ggml_tensor* x,
                                       int64_t h,
                                       int64_t w) {
            // x: [N, h*w, patch_size*patch_size*C]
            // return: [N, C, H, W]
            int64_t N = x->ne[2];
            int64_t C = x->ne[0] / z_image_params.patch_size / z_image_params.patch_size;
            int64_t H = h * z_image_params.patch_size;
            int64_t W = w * z_image_params.patch_size;
            int64_t p = z_image_params.patch_size;

            GGML_ASSERT(C * p * p == x->ne[0]);

            x = ggml_reshape_4d(ctx, x, C, p * p, w * h, N);                 // [N, h*w, p*p, C]
            x = ggml_cont(ctx, ggml_ext_torch_permute(ctx, x, 1, 2, 0, 3));  // [N, C, h*w, p*p]
            x = ggml_reshape_4d(ctx, x, p, p, w, h * C * N);                 // [N*C*h, w, p, p]
            x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));            // [N*C*h, p, w, p]
            x = ggml_reshape_4d(ctx, x, W, H, C, N);                         // [N, C, h*p, w*p]

            return x;
        }

        struct ggml_tensor* forward_core(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* timestep,
                                         struct ggml_tensor* context,
                                         struct ggml_tensor* pe,
                                         struct ggml_tensor* pe_pack = nullptr) {
            auto x_embedder     = std::dynamic_pointer_cast<Linear>(blocks["x_embedder"]);
            auto t_embedder     = std::dynamic_pointer_cast<TimestepEmbedder>(blocks["t_embedder"]);
            auto cap_embedder_0 = std::dynamic_pointer_cast<RMSNorm>(blocks["cap_embedder.0"]);
            auto cap_embedder_1 = std::dynamic_pointer_cast<Linear>(blocks["cap_embedder.1"]);
            auto norm_final     = std::dynamic_pointer_cast<RMSNorm>(blocks["norm_final"]);
            auto final_layer    = std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"]);

            auto txt_pad_token = params["cap_pad_token"];
            auto img_pad_token = params["x_pad_token"];

            int64_t N           = x->ne[2];
            int64_t n_img_token = x->ne[1];
            int64_t n_txt_token = context->ne[1];

            auto t_emb = t_embedder->forward(ctx, timestep);

            auto txt_norm = cap_embedder_0->forward(ctx, context);
            auto txt      = cap_embedder_1->forward(ctx, txt_norm);  // [N, n_txt_token, hidden_size]
            auto img      = x_embedder->forward(ctx, x);             // [N, n_img_token, hidden_size]
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_l0_txt_norm", txt_norm);
                zimg_cache_tensor(ctx, "zimg_l0_txt_embed", txt);
                zimg_cache_tensor(ctx, "zimg_l0_img_embed", img);
            }

            int64_t n_txt_pad_token = Rope::bound_mod(static_cast<int>(n_txt_token), SEQ_MULTI_OF);
            if (n_txt_pad_token > 0) {
                auto txt_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, txt_pad_token, txt_pad_token->ne[0], n_txt_pad_token, N, 1);
                txt                 = ggml_concat(ctx->ggml_ctx, txt, txt_pad_tokens, 1);  // [N, n_txt_token + n_txt_pad_token, hidden_size]
            }

            int64_t n_img_pad_token = Rope::bound_mod(static_cast<int>(n_img_token), SEQ_MULTI_OF);
            if (n_img_pad_token > 0) {
                auto img_pad_tokens = ggml_repeat_4d(ctx->ggml_ctx, img_pad_token, img_pad_token->ne[0], n_img_pad_token, N, 1);
                img                 = ggml_concat(ctx->ggml_ctx, img, img_pad_tokens, 1);  // [N, n_img_token + n_img_pad_token, hidden_size]
            }

            GGML_ASSERT(txt->ne[1] + img->ne[1] == pe->ne[3]);
            if (pe_pack != nullptr) {
                const int64_t pe_pack_token_dim = pe_pack->ne[1] == 2 ? 3 : 2;
                GGML_ASSERT(txt->ne[1] + img->ne[1] == pe_pack->ne[pe_pack_token_dim]);
            }

            auto txt_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, txt->ne[1]);
            auto img_pe = ggml_ext_slice(ctx->ggml_ctx, pe, 3, txt->ne[1], pe->ne[3]);
            auto txt_pe_pack = pe_pack ? ggml_ext_slice(ctx->ggml_ctx, pe_pack, pe_pack->ne[1] == 2 ? 3 : 2, 0, txt->ne[1]) : nullptr;
            auto img_pe_pack = pe_pack ? ggml_ext_slice(ctx->ggml_ctx, pe_pack, pe_pack->ne[1] == 2 ? 3 : 2,
                                                        txt->ne[1], txt->ne[1] + img->ne[1]) : nullptr;

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["context_refiner." + std::to_string(i)]);
                const std::string ref_dump = (zimg_dump_l0_enabled() && i == 0) ? "zimg_l0_ctx_ref0_" : "";
                txt                        = block->forward(ctx, txt, txt_pe, txt_pe_pack, nullptr, nullptr, ref_dump);
                if (zimg_dump_l0_enabled()) {
                    zimg_cache_tensor(ctx, "zimg_l0_ctx_refiner_" + std::to_string(i) + "_out", txt);
                }
            }

            for (int i = 0; i < z_image_params.num_refiner_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["noise_refiner." + std::to_string(i)]);
                const std::string ref_dump = (zimg_dump_l0_enabled() && i == 0) ? "zimg_l0_noise_ref0_" : "";
                img                        = block->forward(ctx, img, img_pe, img_pe_pack, nullptr, t_emb, ref_dump);
                if (zimg_dump_l0_enabled()) {
                    zimg_cache_tensor(ctx, "zimg_l0_noise_refiner_" + std::to_string(i) + "_out", img);
                }
            }

            auto txt_img = ggml_concat(ctx->ggml_ctx, txt, img, 1);  // [N, n_txt_token + n_txt_pad_token + n_img_token + n_img_pad_token, hidden_size]
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_l0_txt_img_in", txt_img);
            }

            const int max_main_layers = zimg_max_main_layers();
            const int total_main      = static_cast<int>(z_image_params.num_layers);
            const int run_main_layers = (max_main_layers > 0) ? std::min(total_main, max_main_layers) : total_main;
            for (int i = 0; i < run_main_layers; i++) {
                auto block = std::dynamic_pointer_cast<JointTransformerBlock>(blocks["layers." + std::to_string(i)]);
                const int dump_layer          = zimg_dump_main_layer();
                const std::string dump_prefix =
                    (zimg_dump_l0_enabled() && i == dump_layer) ? ("zimg_l" + std::to_string(i) + "_") : "";
                txt_img                       = block->forward(ctx, txt_img, pe, pe_pack, nullptr, t_emb, dump_prefix);
                if (!dump_prefix.empty()) {
                    zimg_cache_tensor(ctx, dump_prefix + "txt_img_out", txt_img);
                }
            }

            txt_img = final_layer->forward(ctx, txt_img, t_emb);  // [N, n_txt_token + n_txt_pad_token + n_img_token + n_img_pad_token, ph*pw*C]
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_post_final_layer", txt_img);
            }

            img = ggml_ext_slice(ctx->ggml_ctx, txt_img, 1, n_txt_token + n_txt_pad_token, n_txt_token + n_txt_pad_token + n_img_token);  // [N, n_img_token, ph*pw*C]
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_post_img_slice", img);
            }

            return img;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe,
                                    struct ggml_tensor* pe_pack = nullptr,
                                    std::vector<ggml_tensor*> ref_latents = {}) {
            // Forward pass of DiT.
            // x: [N, C, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // return: [N, C, H, W]

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            auto img             = process_img(ctx, x);
            uint64_t n_img_token = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = process_img(ctx, ref);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            int64_t h_len = ((H + (z_image_params.patch_size / 2)) / z_image_params.patch_size);
            int64_t w_len = ((W + (z_image_params.patch_size / 2)) / z_image_params.patch_size);

            auto out = forward_core(ctx, img, timestep, context, pe, pe_pack);
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_post_forward_core_out", out);
            }

            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, n_img_token);  // [N, n_img_token, ph*pw*C]
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_post_img_token_slice", out);
            }
            out = unpatchify(ctx->ggml_ctx, out, h_len, w_len);           // [N, C, H + pad_h, W + pad_w]
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_post_unpatchify", out);
            }

            // slice
            out = ggml_ext_slice(ctx->ggml_ctx, out, 1, 0, H);  // [N, C, H, W + pad_w]
            out = ggml_ext_slice(ctx->ggml_ctx, out, 0, 0, W);  // [N, C, H, W]
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_post_hw_slice", out);
            }

            out = ggml_ext_scale(ctx->ggml_ctx, out, -1.f);
            if (zimg_dump_l0_enabled()) {
                zimg_cache_tensor(ctx, "zimg_post_scale", out);
            }

            return out;
        }
    };

    struct ZImageRunner : public GGMLRunner {
    protected:
        bool dump_cached_tensors_if_needed() {
            if (!zimg_dump_l0_enabled()) {
                return false;
            }
            const char* dump_dir = std::getenv("SD_DUMP_TENSOR_DIR");
            if (dump_dir == nullptr || dump_dir[0] == '\0') {
                return false;
            }
            std::string base(dump_dir);
            if (!base.empty() && base.back() != '/') {
                base.push_back('/');
            }
            const char* tag = std::getenv("SD_DUMP_TENSOR_TAG");
            if (tag == nullptr || tag[0] == '\0') {
                tag = "zimg";
            }

            bool wrote = false;
            for (ggml_tensor* t = ggml_get_first_tensor(cache_ctx); t != nullptr; t = ggml_get_next_tensor(cache_ctx, t)) {
                const char* name = ggml_get_name(t);
                if (name == nullptr || name[0] == '\0') {
                    continue;
                }
                std::string fname = base + std::string(tag) + "_" + name + ".tensor";
                wrote |= zimg_dump_tensor_to_file(fname, name, t);
            }
            if (wrote) {
                LOG_INFO("z_image dumped cached tensors to '%s' with tag '%s'", dump_dir, tag);
            }
            return wrote;
        }

    public:
        ZImageParams z_image_params;
        ZImageModel z_image;
        std::vector<float> pe_vec;
        std::vector<float> timestep_vec;
        SDVersion version;

        ZImageRunner(ggml_backend_t backend,
                     bool offload_params_to_cpu,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "",
                     SDVersion version                              = VERSION_Z_IMAGE)
            : GGMLRunner(backend, offload_params_to_cpu) {
            z_image = ZImageModel(z_image_params);
            z_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "z_image";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            z_image.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        std::vector<ggml_tensor*> ref_latents = {},
                                        bool increase_ref_index               = false) {
            GGML_ASSERT(x->ne[3] == 1);
            struct ggml_cgraph* gf = new_graph_custom(Z_IMAGE_GRAPH_SIZE);

            x         = to_backend(x);
            context   = to_backend(context);
            timesteps = to_backend(timesteps);

            for (int i = 0; i < ref_latents.size(); i++) {
                ref_latents[i] = to_backend(ref_latents[i]);
            }

            pe_vec      = Rope::gen_z_image_pe(static_cast<int>(x->ne[1]),
                                               static_cast<int>(x->ne[0]),
                                               z_image_params.patch_size,
                                               static_cast<int>(x->ne[3]),
                                               static_cast<int>(context->ne[1]),
                                               SEQ_MULTI_OF,
                                               ref_latents,
                                               increase_ref_index,
                                               z_image_params.theta,
                                               circular_y_enabled,
                                               circular_x_enabled,
                                               z_image_params.axes_dim);
            int pos_len = static_cast<int>(pe_vec.size() / z_image_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe      = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, z_image_params.axes_dim_sum / 2, pos_len);
            struct ggml_tensor* pe_pack = nullptr;
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe, true, "pe");
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_vec.data());
            if (Rope::zimg_htp_rope_requested(runtime_backend)) {
                pe_pack = pe;
            }
            auto runner_ctx = get_context();

            struct ggml_tensor* out = z_image.forward(&runner_ctx,
                                                      x,
                                                      timesteps,
                                                      context,
                                                      pe,
                                                      pe_pack,
                                                      ref_latents);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     std::vector<ggml_tensor*> ref_latents = {},
                     bool increase_ref_index               = false,
                     struct ggml_tensor** output           = nullptr,
                     struct ggml_context* output_ctx       = nullptr) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index);
            };

            bool ok = GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
            if (ok) {
                dump_cached_tensors_if_needed();
            }
            return ok;
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);

            {
                // auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = load_tensor_from_file(work_ctx, "./z_image_x.bin");
                print_ggml_tensor(x);

                std::vector<float> timesteps_vec(1, 0.f);
                auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);

                // auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 2560, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = load_tensor_from_file(work_ctx, "./z_image_context.bin");
                print_ggml_tensor(context);

                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                compute(8, x, timesteps, context, {}, false, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("z_image test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cuda q8: pass
            // cuda q8 fa: pass
            // ggml_backend_t backend = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            if (model_data_type != GGML_TYPE_COUNT) {
                for (auto& [name, tensor_storage] : tensor_storage_map) {
                    if (ends_with(name, "weight")) {
                        tensor_storage.expected_type = model_data_type;
                    }
                }
            }

            std::shared_ptr<ZImageRunner> z_image = std::make_shared<ZImageRunner>(backend,
                                                                                   false,
                                                                                   tensor_storage_map,
                                                                                   "model.diffusion_model",
                                                                                   VERSION_QWEN_IMAGE);

            z_image->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            z_image->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("z_image model loaded");
            z_image->test();
        }
    };

}  // namespace ZImage

#endif  // __Z_IMAGE_HPP__
