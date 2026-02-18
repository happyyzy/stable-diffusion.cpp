#include "qcom_ml_vae_bridge.hpp"

#include <cstring>

#ifdef SD_USE_QCOM_ML_VAE
#include <dlfcn.h>
#endif

QcomMlVaeBridge::~QcomMlVaeBridge() {
#ifdef SD_USE_QCOM_ML_VAE
    if (m_destroy_fn != nullptr && m_ctx_handle != nullptr) {
        m_destroy_fn(m_ctx_handle);
        m_ctx_handle = nullptr;
    }
    if (m_lib_handle != nullptr) {
        dlclose(m_lib_handle);
        m_lib_handle = nullptr;
    }
#endif
}

bool QcomMlVaeBridge::init(const std::string& shared_lib_path, const QcomMlVaeConfig& cfg, std::string* err_msg) {
#ifndef SD_USE_QCOM_ML_VAE
    (void)shared_lib_path;
    (void)cfg;
    if (err_msg != nullptr) {
        *err_msg = "SD_USE_QCOM_ML_VAE is OFF";
    }
    return false;
#else
    if (shared_lib_path.empty()) {
        if (err_msg != nullptr) {
            *err_msg = "missing SD_QCOM_ML_VAE_LIB path";
        }
        return false;
    }
    if (cfg.model_dir.empty()) {
        if (err_msg != nullptr) {
            *err_msg = "missing qcom_ml VAE model dir";
        }
        return false;
    }

    if (m_lib_handle != nullptr || m_ctx_handle != nullptr) {
        if (err_msg != nullptr) {
            *err_msg = "bridge already initialized";
        }
        return false;
    }

    m_lib_handle = dlopen(shared_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (m_lib_handle == nullptr) {
        if (err_msg != nullptr) {
            const char* dl_err = dlerror();
            *err_msg           = (dl_err != nullptr && dl_err[0] != '\0') ? dl_err : "dlopen failed";
        }
        return false;
    }

    m_create_fn  = reinterpret_cast<CreateFn>(dlsym(m_lib_handle, "sd_qcom_ml_vae_create"));
    m_decode_fn  = reinterpret_cast<DecodeFn>(dlsym(m_lib_handle, "sd_qcom_ml_vae_decode"));
    m_prepare_fn = reinterpret_cast<PrepareFn>(dlsym(m_lib_handle, "sd_qcom_ml_vae_prepare"));
    m_destroy_fn = reinterpret_cast<DestroyFn>(dlsym(m_lib_handle, "sd_qcom_ml_vae_destroy"));

    if (m_create_fn == nullptr || m_decode_fn == nullptr || m_destroy_fn == nullptr) {
        if (err_msg != nullptr) {
            *err_msg = "missing required symbols: sd_qcom_ml_vae_create/decode/destroy";
        }
        dlclose(m_lib_handle);
        m_lib_handle = nullptr;
        m_create_fn  = nullptr;
        m_decode_fn  = nullptr;
        m_prepare_fn = nullptr;
        m_destroy_fn = nullptr;
        return false;
    }

    char err_buf[512];
    std::memset(err_buf, 0, sizeof(err_buf));
    void* out_ctx = nullptr;
    const int rc  = m_create_fn(cfg.model_dir.c_str(),
                               cfg.disable_mnn_attention ? 1 : 0,
                               cfg.fallback_attn_len_16384 ? 1 : 0,
                               &out_ctx,
                               err_buf,
                               static_cast<int>(sizeof(err_buf)));
    if (rc != 0 || out_ctx == nullptr) {
        if (err_msg != nullptr) {
            *err_msg = err_buf[0] != '\0' ? err_buf : "sd_qcom_ml_vae_create failed";
        }
        dlclose(m_lib_handle);
        m_lib_handle = nullptr;
        m_create_fn  = nullptr;
        m_decode_fn  = nullptr;
        m_destroy_fn = nullptr;
        return false;
    }

    m_ctx_handle = out_ctx;
    return true;
#endif
}

bool QcomMlVaeBridge::decode(const std::vector<float>& latent_nchw,
                             int latent_w,
                             int latent_h,
                             int latent_c,
                             int batch,
                             std::vector<float>* out_nchw,
                             int out_w,
                             int out_h,
                             int out_c,
                             std::string* err_msg,
                             bool* used_attn_fallback_16384) {
#ifndef SD_USE_QCOM_ML_VAE
    (void)latent_nchw;
    (void)latent_w;
    (void)latent_h;
    (void)latent_c;
    (void)batch;
    (void)out_nchw;
    (void)out_w;
    (void)out_h;
    (void)out_c;
    (void)used_attn_fallback_16384;
    if (err_msg != nullptr) {
        *err_msg = "SD_USE_QCOM_ML_VAE is OFF";
    }
    return false;
#else
    if (m_decode_fn == nullptr || m_ctx_handle == nullptr) {
        if (err_msg != nullptr) {
            *err_msg = "bridge is not initialized";
        }
        return false;
    }
    if (out_nchw == nullptr) {
        if (err_msg != nullptr) {
            *err_msg = "null out_nchw";
        }
        return false;
    }

    const size_t out_elem_count = static_cast<size_t>(out_w) * out_h * out_c * batch;
    out_nchw->assign(out_elem_count, 0.0f);

    char err_buf[512];
    std::memset(err_buf, 0, sizeof(err_buf));
    int fallback_used = 0;
    const int rc      = m_decode_fn(m_ctx_handle,
                               latent_nchw.data(),
                               latent_w,
                               latent_h,
                               latent_c,
                               batch,
                               out_nchw->data(),
                               out_w,
                               out_h,
                               out_c,
                               &fallback_used,
                               err_buf,
                               static_cast<int>(sizeof(err_buf)));
    if (used_attn_fallback_16384 != nullptr) {
        *used_attn_fallback_16384 = fallback_used != 0;
    }
    if (rc != 0) {
        if (err_msg != nullptr) {
            *err_msg = err_buf[0] != '\0' ? err_buf : "sd_qcom_ml_vae_decode failed";
        }
        return false;
    }
    return true;
#endif
}

bool QcomMlVaeBridge::prepare(int latent_w,
                              int latent_h,
                              int latent_c,
                              int batch,
                              std::string* err_msg) {
#ifndef SD_USE_QCOM_ML_VAE
    (void)latent_w;
    (void)latent_h;
    (void)latent_c;
    (void)batch;
    if (err_msg != nullptr) {
        *err_msg = "SD_USE_QCOM_ML_VAE is OFF";
    }
    return false;
#else
    if (m_decode_fn == nullptr || m_ctx_handle == nullptr) {
        if (err_msg != nullptr) {
            *err_msg = "bridge is not initialized";
        }
        return false;
    }
    // Older bridge library may not export prepare symbol; treat as no-op success.
    if (m_prepare_fn == nullptr) {
        return true;
    }

    char err_buf[512];
    std::memset(err_buf, 0, sizeof(err_buf));
    const int rc = m_prepare_fn(m_ctx_handle,
                                latent_w,
                                latent_h,
                                latent_c,
                                batch,
                                err_buf,
                                static_cast<int>(sizeof(err_buf)));
    if (rc != 0) {
        if (err_msg != nullptr) {
            *err_msg = err_buf[0] != '\0' ? err_buf : "sd_qcom_ml_vae_prepare failed";
        }
        return false;
    }
    return true;
#endif
}

bool QcomMlVaeBridge::ready() const {
    return m_decode_fn != nullptr && m_ctx_handle != nullptr;
}
