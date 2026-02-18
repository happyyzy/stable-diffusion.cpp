#ifndef SD_QCOM_ML_VAE_BRIDGE_HPP
#define SD_QCOM_ML_VAE_BRIDGE_HPP

#include <string>
#include <vector>

struct QcomMlVaeConfig {
    std::string model_dir;
    bool disable_mnn_attention   = true;
    bool fallback_attn_len_16384 = true;
};

class QcomMlVaeBridge {
public:
    QcomMlVaeBridge() = default;
    ~QcomMlVaeBridge();

    bool init(const std::string& shared_lib_path, const QcomMlVaeConfig& cfg, std::string* err_msg);

    bool decode(const std::vector<float>& latent_nchw,
                int latent_w,
                int latent_h,
                int latent_c,
                int batch,
                std::vector<float>* out_nchw,
                int out_w,
                int out_h,
                int out_c,
                std::string* err_msg,
                bool* used_attn_fallback_16384);

    bool ready() const;

private:
    QcomMlVaeBridge(const QcomMlVaeBridge&)            = delete;
    QcomMlVaeBridge& operator=(const QcomMlVaeBridge&) = delete;

    void* m_lib_handle = nullptr;
    void* m_ctx_handle = nullptr;

    typedef int (*CreateFn)(const char* model_dir,
                            int disable_mnn_attention,
                            int fallback_attn_len_16384,
                            void** out_ctx,
                            char* err_buf,
                            int err_buf_cap);
    typedef int (*DecodeFn)(void* ctx,
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
                            int err_buf_cap);
    typedef void (*DestroyFn)(void* ctx);

    CreateFn  m_create_fn  = nullptr;
    DecodeFn  m_decode_fn  = nullptr;
    DestroyFn m_destroy_fn = nullptr;
};

#endif  // SD_QCOM_ML_VAE_BRIDGE_HPP
