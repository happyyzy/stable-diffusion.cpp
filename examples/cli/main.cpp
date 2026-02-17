#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cctype>
#include <inttypes.h>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

#include "common/common.hpp"
#include "llm.hpp"
#include "conditioner.hpp"
#include "flux.hpp"

#ifdef SD_USE_HEXAGON
#include "ggml-hexagon.h"
#endif

#include "avi_writer.h"

const char* previews_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

std::regex format_specifier_regex("(?:[^%]|^)(?:%%)*(%\\d{0,3}d)");

struct SDCliParams {
    SDMode mode             = IMG_GEN;
    std::string output_path = "output.png";
    int output_begin_idx    = -1;
    std::string load_latent_path;

    bool verbose          = false;
    bool canny_preprocess = false;
    bool convert_name     = false;

    preview_t preview_method = PREVIEW_NONE;
    int preview_interval     = 1;
    std::string preview_path = "preview.png";
    int preview_fps          = 16;
    bool taesd_preview       = false;
    bool preview_noisy       = false;
    bool color               = false;

    bool llm_embed_only       = false;
    int llm_embed_id          = -1;
    int llm_embed_n           = 1;
    std::string llm_embed_dump = "llm_embed_only.tensor";
    std::string llm_embed_arch = "qwen3";
    bool llm_forward_only       = false;
    bool llm_generate_only      = false;
    int llm_max_new_tokens      = 32;
    std::string llm_forward_dump;
    std::string llm_generate_dump;
    std::string llm_version      = "flux2_klein";
    bool q4_matmul_bench         = false;
    std::string q4_weight       = "q_proj";
    int q4_layer                = 0;
    int q4_n_token              = 1;
    std::string q4_input;
    std::string q4_dump         = "q4_matmul.tensor";
    bool rmsnorm_bench          = false;
    std::string rmsnorm_weight  = "q_norm";
    std::string rmsnorm_input;
    std::string rmsnorm_dump    = "rmsnorm.tensor";
    float rmsnorm_eps           = 1e-6f;
    bool attn_bench             = false;
    std::string attn_q_input;
    std::string attn_k_input;
    std::string attn_v_input;
    int attn_n_head             = 32;
    std::string attn_dump        = "attn.tensor";
    bool flux_fwd_bench         = false;
    int flux_bench_runs         = 1;
    int flux_bench_warmup       = 0;
    int flux_bench_ctx          = 256;

    bool normal_exit = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-o",
             "--output",
             "path to write result image to. you can use printf-style %d format specifiers for image sequences (default: ./output.png) (eg. output_%03d.png)",
             &output_path},
            {"",
             "--load-latent",
             "path to latent tensor dump to decode with VAE (ggml .tensor format)",
             &load_latent_path},
            {"",
             "--preview-path",
             "path to write preview image to (default: ./preview.png)",
             &preview_path},
            {"",
             "--llm-embed-dump",
             "path to write embedding tensor dump (default: ./llm_embed_only.tensor)",
             &llm_embed_dump},
            {"",
             "--llm-embed-arch",
             "llm arch for embed-only: qwen3 | mistral-small3.2 (default: qwen3)",
             &llm_embed_arch},
            {"",
             "--llm-forward-dump",
             "path to write llm hidden_states dump (c_crossattn) for llm-forward-only",
             &llm_forward_dump},
            {"",
             "--llm-generate-dump",
             "path to write llm generated text for llm-generate-only",
             &llm_generate_dump},
            {"",
             "--llm-version",
             "llm version for llm-forward-only: flux2_klein | flux2 | z_image | ovis_image | qwen_image (default: flux2_klein)",
             &llm_version},
            {"",
             "--q4-weight",
             "q4 matmul weight name: q_proj | k_proj | v_proj | gate_proj | up_proj | down_proj (default: q_proj)",
             &q4_weight},
            {"",
             "--q4-dump",
             "path to write q4 matmul output tensor (default: ./q4_matmul.tensor)",
             &q4_dump},
            {"",
             "--q4-input",
             "path to load input tensor dump for q4 matmul (overrides --q4-n-token)",
             &q4_input},
            {"",
             "--rmsnorm-weight",
             "rmsnorm weight name: q_norm | k_norm | input_layernorm | post_attention_layernorm (default: q_norm)",
             &rmsnorm_weight},
            {"",
             "--rmsnorm-input",
             "path to load input tensor dump for rmsnorm",
             &rmsnorm_input},
            {"",
             "--rmsnorm-dump",
             "path to write rmsnorm output tensor (default: ./rmsnorm.tensor)",
             &rmsnorm_dump},
            {"",
             "--attn-q",
             "path to load attention q tensor dump",
             &attn_q_input},
            {"",
             "--attn-k",
             "path to load attention k tensor dump",
             &attn_k_input},
            {"",
             "--attn-v",
             "path to load attention v tensor dump",
             &attn_v_input},
            {"",
             "--attn-dump",
             "path to write attention output tensor (default: ./attn.tensor)",
             &attn_dump},
        };

        options.int_options = {
            {"",
             "--preview-interval",
             "interval in denoising steps between consecutive updates of the image preview file (default is 1, meaning updating at every step)",
             &preview_interval},
            {"",
             "--output-begin-idx",
             "starting index for output image sequence, must be non-negative (default 0 if specified %d in output path, 1 otherwise)",
             &output_begin_idx},
            {"",
             "--llm-embed-id",
             "token id to lookup for embed-only",
             &llm_embed_id},
            {"",
             "--llm-embed-n",
             "number of tokens to repeat for embed-only (default: 1)",
             &llm_embed_n},
            {"",
             "--llm-max-new-tokens",
             "max new tokens for llm-generate-only (default: 32)",
             &llm_max_new_tokens},
            {"",
             "--q4-n-token",
             "q4 matmul input token count (default: 1)",
             &q4_n_token},
            {"",
             "--q4-layer",
             "q4 matmul layer index (default: 0)",
             &q4_layer},
            {"",
             "--attn-n-head",
             "attention num heads (default: 32)",
             &attn_n_head},
            {"",
             "--flux-bench-runs",
             "number of timed runs for flux forward bench (default: 1)",
             &flux_bench_runs},
            {"",
             "--flux-bench-warmup",
             "number of warmup runs for flux forward bench (default: 0)",
             &flux_bench_warmup},
            {"",
             "--flux-bench-ctx",
             "context length for flux forward bench (default: 256)",
             &flux_bench_ctx},
        };

        options.bool_options = {
            {"",
             "--canny",
             "apply canny preprocessor (edge detection)",
             true, &canny_preprocess},
            {"",
             "--convert-name",
             "convert tensor name (for convert mode)",
             true, &convert_name},
            {"-v",
             "--verbose",
             "print extra info",
             true, &verbose},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
            {"",
             "--taesd-preview-only",
             std::string("prevents usage of taesd for decoding the final image. (for use with --preview ") + previews_str[PREVIEW_TAE] + ")",
             true, &taesd_preview},
            {"",
             "--preview-noisy",
             "enables previewing noisy inputs of the models rather than the denoised outputs",
             true, &preview_noisy},
            {"",
             "--llm-embed-only",
             "run llm embedding-only test and exit",
             true, &llm_embed_only},
            {"",
             "--llm-forward-only",
             "run llm forward-only (prompt to hidden_states) and exit",
             true, &llm_forward_only},
            {"",
             "--llm-generate-only",
             "run llm greedy generation test and exit",
             true, &llm_generate_only},
            {"",
             "--q4-matmul-bench",
             "run q4 matmul microbench and exit",
             true, &q4_matmul_bench},
            {"",
             "--rmsnorm-bench",
             "run rmsnorm microbench and exit",
             true, &rmsnorm_bench},
            {"",
             "--attn-bench",
             "run attention microbench and exit",
             true, &attn_bench},
            {"",
             "--flux-fwd-bench",
             "run flux forward microbench and exit",
             true, &flux_fwd_bench},

        };

        auto on_mode_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* mode_c_str = argv[index];
            if (mode_c_str != nullptr) {
                int mode_found = -1;
                for (int i = 0; i < MODE_COUNT; i++) {
                    if (!strcmp(mode_c_str, modes_str[i])) {
                        mode_found = i;
                    }
                }
                if (mode_found == -1) {
                    LOG_ERROR("error: invalid mode %s, must be one of [%s]\n",
                              mode_c_str, SD_ALL_MODES_STR);
                    exit(1);
                }
                mode = (SDMode)mode_found;
            }
            return 1;
        };

        auto on_preview_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* preview = argv[index];
            int preview_found   = -1;
            for (int m = 0; m < PREVIEW_COUNT; m++) {
                if (!strcmp(preview, previews_str[m])) {
                    preview_found = m;
                }
            }
            if (preview_found == -1) {
                LOG_ERROR("error: preview method %s", preview);
                return -1;
            }
            preview_method = (preview_t)preview_found;
            return 1;
        };

        auto on_help_arg = [&](int argc, const char** argv, int index) {
            normal_exit = true;
            return -1;
        };

        options.manual_options = {
            {"-M",
             "--mode",
             "run mode, one of [img_gen, vid_gen, upscale, convert], default: img_gen",
             on_mode_arg},
            {"",
             "--preview",
             std::string("preview method. must be one of the following [") + previews_str[0] + ", " + previews_str[1] + ", " + previews_str[2] + ", " + previews_str[3] + "] (default is " + previews_str[PREVIEW_NONE] + ")",
             on_preview_arg},
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
        };

        return options;
    };

    bool process_and_check() {
        if (output_path.length() == 0) {
            LOG_ERROR("error: the following arguments are required: output_path");
            return false;
        }
        if (flux_bench_runs <= 0) {
            LOG_ERROR("error: --flux-bench-runs must be > 0");
            return false;
        }
        if (flux_bench_warmup < 0) {
            LOG_ERROR("error: --flux-bench-warmup must be >= 0");
            return false;
        }
        if (flux_bench_ctx <= 0) {
            LOG_ERROR("error: --flux-bench-ctx must be > 0");
            return false;
        }

        if (mode == CONVERT) {
            if (output_path == "output.png") {
                output_path = "output.gguf";
            }
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDCliParams {\n"
            << "  mode: " << modes_str[mode] << ",\n"
            << "  output_path: \"" << output_path << "\",\n"
            << "  load_latent_path: \"" << load_latent_path << "\",\n"
            << "  verbose: " << (verbose ? "true" : "false") << ",\n"
            << "  color: " << (color ? "true" : "false") << ",\n"
            << "  canny_preprocess: " << (canny_preprocess ? "true" : "false") << ",\n"
            << "  convert_name: " << (convert_name ? "true" : "false") << ",\n"
            << "  preview_method: " << previews_str[preview_method] << ",\n"
            << "  preview_interval: " << preview_interval << ",\n"
            << "  preview_path: \"" << preview_path << "\",\n"
            << "  preview_fps: " << preview_fps << ",\n"
            << "  taesd_preview: " << (taesd_preview ? "true" : "false") << ",\n"
            << "  preview_noisy: " << (preview_noisy ? "true" : "false") << ",\n"
            << "  llm_embed_only: " << (llm_embed_only ? "true" : "false") << ",\n"
            << "  llm_embed_id: " << llm_embed_id << ",\n"
            << "  llm_embed_n: " << llm_embed_n << ",\n"
            << "  llm_embed_dump: \"" << llm_embed_dump << "\",\n"
            << "  llm_embed_arch: \"" << llm_embed_arch << "\",\n"
            << "  llm_forward_only: " << (llm_forward_only ? "true" : "false") << ",\n"
            << "  llm_generate_only: " << (llm_generate_only ? "true" : "false") << ",\n"
            << "  llm_max_new_tokens: " << llm_max_new_tokens << ",\n"
            << "  llm_forward_dump: \"" << llm_forward_dump << "\",\n"
            << "  llm_generate_dump: \"" << llm_generate_dump << "\",\n"
            << "  llm_version: \"" << llm_version << "\",\n"
            << "  q4_matmul_bench: " << (q4_matmul_bench ? "true" : "false") << ",\n"
            << "  q4_weight: \"" << q4_weight << "\",\n"
            << "  q4_n_token: " << q4_n_token << ",\n"
            << "  q4_input: \"" << q4_input << "\",\n"
            << "  q4_dump: \"" << q4_dump << "\",\n"
            << "  rmsnorm_bench: " << (rmsnorm_bench ? "true" : "false") << ",\n"
            << "  rmsnorm_weight: \"" << rmsnorm_weight << "\",\n"
            << "  rmsnorm_input: \"" << rmsnorm_input << "\",\n"
            << "  rmsnorm_dump: \"" << rmsnorm_dump << "\",\n"
            << "  rmsnorm_eps: " << rmsnorm_eps << ",\n"
            << "  attn_bench: " << (attn_bench ? "true" : "false") << ",\n"
            << "  attn_q_input: \"" << attn_q_input << "\",\n"
            << "  attn_k_input: \"" << attn_k_input << "\",\n"
            << "  attn_v_input: \"" << attn_v_input << "\",\n"
            << "  attn_n_head: " << attn_n_head << ",\n"
            << "  attn_dump: \"" << attn_dump << "\",\n"
            << "  flux_fwd_bench: " << (flux_fwd_bench ? "true" : "false") << ",\n"
            << "  flux_bench_runs: " << flux_bench_runs << ",\n"
            << "  flux_bench_warmup: " << flux_bench_warmup << ",\n"
            << "  flux_bench_ctx: " << flux_bench_ctx << "\n"
            << "}";
        return oss.str();
    }
};

static ggml_backend_t init_llm_embed_backend() {
    ggml_backend_t backend = nullptr;
#ifdef SD_USE_CUDA
    LOG_DEBUG("Using CUDA backend");
    backend = ggml_backend_cuda_init(0);
#endif
#ifdef SD_USE_METAL
    LOG_DEBUG("Using Metal backend");
    backend = ggml_backend_metal_init();
#endif
#ifdef SD_USE_VULKAN
    LOG_DEBUG("Using Vulkan backend");
    size_t device          = 0;
    const int device_count = ggml_backend_vk_get_device_count();
    if (device_count) {
        const char* SD_VK_DEVICE = getenv("SD_VK_DEVICE");
        if (SD_VK_DEVICE != nullptr) {
            std::string sd_vk_device_str = SD_VK_DEVICE;
            try {
                device = std::stoull(sd_vk_device_str);
            } catch (...) {
                device = 0;
            }
            if (device >= static_cast<size_t>(device_count)) {
                device = 0;
            }
        }
        backend = ggml_backend_vk_init(device);
    }
#endif
#ifdef SD_USE_OPENCL
    LOG_DEBUG("Using OpenCL backend");
    backend = ggml_backend_opencl_init();
#endif
#ifdef SD_USE_SYCL
    LOG_DEBUG("Using SYCL backend");
    backend = ggml_backend_sycl_init(0);
#endif
#ifdef SD_USE_HEXAGON
    LOG_DEBUG("Using Hexagon backend");
    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL, nullptr);
#endif
    if (!backend) {
        LOG_DEBUG("Using CPU backend");
        backend = ggml_backend_cpu_init();
    }
    return backend;
}

static bool parse_llm_embed_arch(const std::string& arch_str, LLM::LLMArch* arch_out) {
    if (arch_out == nullptr) {
        return false;
    }
    std::string lower = arch_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "mistral-small3.2" || lower == "mistral_small_3_2" || lower == "mistral") {
        *arch_out = LLM::LLMArch::MISTRAL_SMALL_3_2;
        return true;
    }
    if (lower == "qwen3" || lower == "qwen") {
        *arch_out = LLM::LLMArch::QWEN3;
        return true;
    }
    return false;
}

static bool parse_llm_version(const std::string& version_str, SDVersion* version_out) {
    if (version_out == nullptr) {
        return false;
    }
    std::string lower = version_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "flux2_klein" || lower == "flux2-klein") {
        *version_out = VERSION_FLUX2_KLEIN;
        return true;
    }
    if (lower == "flux2") {
        *version_out = VERSION_FLUX2;
        return true;
    }
    if (lower == "z_image" || lower == "z-image") {
        *version_out = VERSION_Z_IMAGE;
        return true;
    }
    if (lower == "ovis_image" || lower == "ovis-image") {
        *version_out = VERSION_OVIS_IMAGE;
        return true;
    }
    if (lower == "qwen_image" || lower == "qwen-image") {
        *version_out = VERSION_QWEN_IMAGE;
        return true;
    }
    return false;
}

static int run_llm_embed_only(const SDCliParams& cli_params, const SDContextParams& ctx_params) {
    if (ctx_params.llm_path.empty()) {
        LOG_ERROR("llm embed-only requires --llm path");
        return 1;
    }
    if (cli_params.llm_embed_id < 0) {
        LOG_ERROR("llm embed-only requires --llm-embed-id >= 0");
        return 1;
    }
    if (cli_params.llm_embed_n <= 0) {
        LOG_ERROR("llm embed-only requires --llm-embed-n > 0");
        return 1;
    }

    LLM::LLMArch arch = LLM::LLMArch::QWEN3;
    if (!parse_llm_embed_arch(cli_params.llm_embed_arch, &arch)) {
        LOG_ERROR("unknown --llm-embed-arch '%s'", cli_params.llm_embed_arch.c_str());
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(ctx_params.llm_path, "text_encoders.llm.")) {
        LOG_ERROR("init model loader from file failed: '%s'", ctx_params.llm_path.c_str());
        return 1;
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();

    ggml_backend_t backend = init_llm_embed_backend();
    if (!backend) {
        LOG_ERROR("failed to init backend");
        return 1;
    }

    int n_threads = ctx_params.n_threads > 0 ? ctx_params.n_threads : std::max(1u, std::thread::hardware_concurrency());

    LLM::LLMRunner llm(arch,
                       backend,
                       ctx_params.offload_params_to_cpu,
                       tensor_storage_map,
                       "text_encoders.llm",
                       false);

    if (!llm.alloc_params_buffer()) {
        LOG_ERROR("llm alloc params buffer failed");
        ggml_backend_free(backend);
        return 1;
    }

    std::map<std::string, ggml_tensor*> tensors;
    llm.get_param_tensors(tensors, "text_encoders.llm");
    if (!model_loader.load_tensors(tensors, {}, n_threads, ctx_params.enable_mmap)) {
        LOG_ERROR("load tensors from model loader failed");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(256) * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init failed");
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<int> ids(static_cast<size_t>(cli_params.llm_embed_n), cli_params.llm_embed_id);
    ggml_tensor* input_ids = vector_to_ggml_tensor_i32(work_ctx, ids);
    input_ids              = ggml_reshape_2d(work_ctx, input_ids, input_ids->ne[0], 1);

    ggml_tensor* output = nullptr;
    if (!llm.compute_embed(n_threads, input_ids, &output, work_ctx)) {
        LOG_ERROR("llm embed-only compute failed");
        ggml_free(work_ctx);
        ggml_backend_free(backend);
        return 1;
    }

    LOG_INFO("llm embed-only output: type=%s ne=[%" PRId64 ", %" PRId64 ", %" PRId64 "]",
             ggml_type_name(output->type),
             output->ne[0], output->ne[1], output->ne[2]);

    if (!LLM::LLMRunner::dump_tensor_to_file(cli_params.llm_embed_dump, "llm_embed_only", output)) {
        LOG_WARN("failed to dump embedding tensor to '%s'", cli_params.llm_embed_dump.c_str());
    } else {
        LOG_INFO("saved embedding tensor to '%s'", cli_params.llm_embed_dump.c_str());
    }

    ggml_free(work_ctx);
    ggml_backend_free(backend);
    return 0;
}

static int run_llm_forward_only(const SDCliParams& cli_params,
                                const SDContextParams& ctx_params,
                                const SDGenerationParams& gen_params) {
    if (ctx_params.llm_path.empty()) {
        LOG_ERROR("llm forward-only requires --llm path");
        return 1;
    }
    if (gen_params.prompt.empty()) {
        LOG_ERROR("llm forward-only requires --prompt");
        return 1;
    }

    SDVersion version = VERSION_FLUX2_KLEIN;
    if (!parse_llm_version(cli_params.llm_version, &version)) {
        LOG_ERROR("unknown --llm-version '%s'", cli_params.llm_version.c_str());
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(ctx_params.llm_path, "text_encoders.llm.")) {
        LOG_ERROR("init model loader from file failed: '%s'", ctx_params.llm_path.c_str());
        return 1;
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();

    ggml_backend_t backend = init_llm_embed_backend();
    if (!backend) {
        LOG_ERROR("failed to init backend");
        return 1;
    }

    int n_threads = ctx_params.n_threads > 0 ? ctx_params.n_threads : std::max(1u, std::thread::hardware_concurrency());

    std::shared_ptr<LLMEmbedder> llm = std::make_shared<LLMEmbedder>(backend,
                                                                     ctx_params.offload_params_to_cpu,
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "text_encoders.llm",
                                                                     false);

    llm->alloc_params_buffer();

    std::map<std::string, ggml_tensor*> tensors;
    llm->get_param_tensors(tensors);
    if (!model_loader.load_tensors(tensors, {}, n_threads, ctx_params.enable_mmap)) {
        LOG_ERROR("load tensors from model loader failed");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params params;
    params.mem_size   = static_cast<size_t>(512) * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc   = false;
    ggml_context* work_ctx = ggml_init(params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init failed");
        ggml_backend_free(backend);
        return 1;
    }

    ConditionerParams cond_params;
    cond_params.text      = gen_params.prompt;
    cond_params.clip_skip = gen_params.clip_skip;

    SDCondition cond = llm->get_learned_condition(work_ctx, n_threads, cond_params);

    if (!cli_params.llm_forward_dump.empty()) {
        if (!LLM::LLMRunner::dump_tensor_to_file(cli_params.llm_forward_dump.c_str(), "llm_c_crossattn", cond.c_crossattn)) {
            LOG_WARN("failed to dump llm hidden_states to '%s'", cli_params.llm_forward_dump.c_str());
        } else {
            LOG_INFO("saved llm hidden_states to '%s'", cli_params.llm_forward_dump.c_str());
        }
    }

    ggml_free(work_ctx);
    ggml_backend_free(backend);
    return 0;
}

static std::string build_llm_prompt_for_version(SDVersion version, const std::string& user_text) {
    if (version == VERSION_FLUX2) {
        std::string prompt = "[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]";
        prompt += user_text;
        prompt += "[/INST]";
        return prompt;
    }
    if (version == VERSION_FLUX2_KLEIN) {
        std::string prompt = "<|im_start|>user\n";
        prompt += user_text;
        prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        return prompt;
    }
    if (sd_version_is_z_image(version)) {
        std::string prompt = "<|im_start|>user\n";
        prompt += user_text;
        prompt += "<|im_end|>\n<|im_start|>assistant\n";
        return prompt;
    }
    if (version == VERSION_OVIS_IMAGE) {
        std::string prompt = "<|im_start|>user\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background: ";
        prompt += user_text;
        prompt += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        return prompt;
    }

    std::string prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
    prompt += user_text;
    prompt += "<|im_end|>\n<|im_start|>assistant\n";
    return prompt;
}

static std::string escape_for_log(std::string s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

static int run_llm_generate_only(const SDCliParams& cli_params,
                                 const SDContextParams& ctx_params,
                                 const SDGenerationParams& gen_params) {
    if (ctx_params.llm_path.empty()) {
        LOG_ERROR("llm generate-only requires --llm path");
        return 1;
    }
    if (gen_params.prompt.empty()) {
        LOG_ERROR("llm generate-only requires --prompt");
        return 1;
    }
    if (cli_params.llm_max_new_tokens <= 0) {
        LOG_ERROR("llm generate-only requires --llm-max-new-tokens > 0");
        return 1;
    }

    SDVersion version = VERSION_FLUX2_KLEIN;
    if (!parse_llm_version(cli_params.llm_version, &version)) {
        LOG_ERROR("unknown --llm-version '%s'", cli_params.llm_version.c_str());
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(ctx_params.llm_path, "text_encoders.llm.")) {
        LOG_ERROR("init model loader from file failed: '%s'", ctx_params.llm_path.c_str());
        return 1;
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();

    ggml_backend_t backend = init_llm_embed_backend();
    if (!backend) {
        LOG_ERROR("failed to init backend");
        return 1;
    }

    int n_threads = ctx_params.n_threads > 0 ? ctx_params.n_threads : std::max(1u, std::thread::hardware_concurrency());

    std::shared_ptr<LLMEmbedder> llm = std::make_shared<LLMEmbedder>(backend,
                                                                     ctx_params.offload_params_to_cpu,
                                                                     tensor_storage_map,
                                                                     version,
                                                                     "text_encoders.llm",
                                                                     false);

    llm->alloc_params_buffer();

    std::map<std::string, ggml_tensor*> tensors;
    llm->get_param_tensors(tensors);
    if (!model_loader.load_tensors(tensors, {}, n_threads, ctx_params.enable_mmap)) {
        LOG_ERROR("load tensors from model loader failed");
        ggml_backend_free(backend);
        return 1;
    }

    std::string prompt = build_llm_prompt_for_version(version, gen_params.prompt);
    std::vector<int> all_tokens = llm->tokenizer->tokenize(prompt, nullptr);
    std::vector<int> generated_tokens;
    generated_tokens.reserve(cli_params.llm_max_new_tokens);

    LOG_INFO("llm generate-only initial tokens: %zu", all_tokens.size());

    for (int step = 0; step < cli_params.llm_max_new_tokens; ++step) {
        ggml_init_params params;
        params.mem_size   = static_cast<size_t>(512) * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc   = false;
        ggml_context* work_ctx = ggml_init(params);
        if (!work_ctx) {
            LOG_ERROR("ggml_init failed at llm generation step %d", step + 1);
            ggml_backend_free(backend);
            return 1;
        }

        ggml_tensor* input_ids = vector_to_ggml_tensor_i32(work_ctx, all_tokens);
        ggml_tensor* logits    = nullptr;

        const int64_t t0 = ggml_time_ms();
        const bool ok    = llm->llm->compute_last_logits(n_threads,
                                                      input_ids,
                                                      nullptr,
                                                      {},
                                                      &logits,
                                                      work_ctx);
        const int64_t t1 = ggml_time_ms();
        if (!ok || logits == nullptr || logits->data == nullptr) {
            LOG_ERROR("llm generate-only failed at step %d", step + 1);
            ggml_free(work_ctx);
            ggml_backend_free(backend);
            return 1;
        }

        const float* logits_data = static_cast<const float*>(logits->data);
        const int64_t vocab_size = logits->ne[0] * logits->ne[1];
        if (vocab_size <= 0) {
            LOG_ERROR("invalid logits shape at step %d", step + 1);
            ggml_free(work_ctx);
            ggml_backend_free(backend);
            return 1;
        }

        int best_id      = 0;
        float best_logit = logits_data[0];
        for (int64_t i = 1; i < vocab_size; ++i) {
            if (logits_data[i] > best_logit) {
                best_logit = logits_data[i];
                best_id    = static_cast<int>(i);
            }
        }

        generated_tokens.push_back(best_id);
        all_tokens.push_back(best_id);

        const std::string token_text = llm->tokenizer->decode(std::vector<int>{best_id}, false);
        LOG_INFO("llm step %d: token=%d logit=%.6f time=%" PRId64 "ms piece='%s'",
                 step + 1,
                 best_id,
                 best_logit,
                 (t1 - t0),
                 escape_for_log(token_text).c_str());

        ggml_free(work_ctx);

        if (best_id == llm->tokenizer->eos_token_id()) {
            LOG_INFO("llm generation hit eos at step %d", step + 1);
            break;
        }
    }

    const std::string generated_text = llm->tokenizer->decode(generated_tokens, false);
    LOG_INFO("llm generated text: %s", generated_text.c_str());

    if (!cli_params.llm_generate_dump.empty()) {
        std::ofstream out(cli_params.llm_generate_dump);
        if (!out.is_open()) {
            LOG_WARN("failed to open llm generate dump file '%s'", cli_params.llm_generate_dump.c_str());
        } else {
            out << generated_text << "\n";
            out << "token_ids:";
            for (int id : generated_tokens) {
                out << " " << id;
            }
            out << "\n";
            out.flush();
            LOG_INFO("saved llm generated text to '%s'", cli_params.llm_generate_dump.c_str());
        }
    }

    ggml_backend_free(backend);
    return 0;
}

static bool parse_q4_weight_name(const std::string& weight, std::string* weight_out) {
    if (weight_out == nullptr) {
        return false;
    }
    std::string lower = weight;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    auto strip_prefix = [](std::string& s, const std::string& prefix) {
        if (s.rfind(prefix, 0) == 0) {
            s = s.substr(prefix.size());
        }
    };
    strip_prefix(lower, "text_encoders.llm.layers.0.");
    strip_prefix(lower, "layers.0.");
    strip_prefix(lower, "llm.layers.0.");

    if (lower == "q_proj" || lower == "qproj" || lower == "self_attn.q_proj" || lower == "self_attn.q_proj.weight") {
        *weight_out = "self_attn.q_proj.weight";
        return true;
    }
    if (lower == "k_proj" || lower == "kproj" || lower == "self_attn.k_proj" || lower == "self_attn.k_proj.weight") {
        *weight_out = "self_attn.k_proj.weight";
        return true;
    }
    if (lower == "v_proj" || lower == "vproj" || lower == "self_attn.v_proj" || lower == "self_attn.v_proj.weight") {
        *weight_out = "self_attn.v_proj.weight";
        return true;
    }
    if (lower == "gate_proj" || lower == "mlp_gate" || lower == "mlp.gate_proj" || lower == "mlp.gate_proj.weight") {
        *weight_out = "mlp.gate_proj.weight";
        return true;
    }
    if (lower == "up_proj" || lower == "mlp_up" || lower == "mlp.up_proj" || lower == "mlp.up_proj.weight") {
        *weight_out = "mlp.up_proj.weight";
        return true;
    }
    if (lower == "down_proj" || lower == "mlp_down" || lower == "mlp.down_proj" || lower == "mlp.down_proj.weight") {
        *weight_out = "mlp.down_proj.weight";
        return true;
    }
    if (lower == "embed_tokens" || lower == "embed_tokens.weight" || lower == "lm_head" || lower == "lm_head.weight") {
        *weight_out = "embed_tokens.weight";
        return true;
    }
    return false;
}

static bool parse_rmsnorm_weight_name(const std::string& weight, std::string* weight_out) {
    if (weight_out == nullptr) {
        return false;
    }
    std::string lower = weight;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "q_norm" || lower == "qnorm") {
        *weight_out = "q_norm";
        return true;
    }
    if (lower == "k_norm" || lower == "knorm") {
        *weight_out = "k_norm";
        return true;
    }
    if (lower == "input_layernorm" || lower == "input_layer_norm" || lower == "ln1") {
        *weight_out = "input_layernorm";
        return true;
    }
    if (lower == "post_attention_layernorm" || lower == "post_attention_layer_norm" || lower == "ln2") {
        *weight_out = "post_attention_layernorm";
        return true;
    }
    return false;
}

static int run_q4_matmul_bench(const SDCliParams& cli_params,
                               const SDContextParams& ctx_params) {
    if (ctx_params.llm_path.empty()) {
        LOG_ERROR("q4 matmul bench requires --llm path");
        return 1;
    }
    if (cli_params.q4_n_token <= 0) {
        LOG_ERROR("q4 matmul bench requires --q4-n-token > 0");
        return 1;
    }
    if (cli_params.q4_layer < 0) {
        LOG_ERROR("q4 matmul bench requires --q4-layer >= 0");
        return 1;
    }
    std::string weight_key;
    if (!parse_q4_weight_name(cli_params.q4_weight, &weight_key)) {
        LOG_ERROR("unknown --q4-weight '%s'", cli_params.q4_weight.c_str());
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(ctx_params.llm_path, "text_encoders.llm.")) {
        LOG_ERROR("init model loader from file failed: '%s'", ctx_params.llm_path.c_str());
        return 1;
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    std::string weight_name;
    auto find_name = [&](const std::string& candidate) -> bool {
        auto iter = tensor_storage_map.find(candidate);
        if (iter == tensor_storage_map.end()) {
            return false;
        }
        weight_name = candidate;
        return true;
    };
    const std::string layer_str = std::to_string(cli_params.q4_layer);
    if (weight_key == "embed_tokens.weight") {
        if (!find_name("text_encoders.llm.embed_tokens.weight") &&
            !find_name("text_encoders.llm.model.embed_tokens.weight") &&
            !find_name("model.embed_tokens.weight") &&
            !find_name("embed_tokens.weight")) {
            for (auto iter = tensor_storage_map.begin(); iter != tensor_storage_map.end(); ++iter) {
                const std::string& name = iter->first;
                if (name.size() >= std::strlen("embed_tokens.weight") &&
                    name.compare(name.size() - std::strlen("embed_tokens.weight"),
                                 std::strlen("embed_tokens.weight"),
                                 "embed_tokens.weight") == 0) {
                    weight_name = name;
                    break;
                }
            }
        }
    } else if (!find_name("text_encoders.llm.layers." + layer_str + "." + weight_key) &&
               !find_name("text_encoders.llm.model.layers." + layer_str + "." + weight_key) &&
               !find_name("model.layers." + layer_str + "." + weight_key) &&
               !find_name("layers." + layer_str + "." + weight_key)) {
        const std::string suffix0 = ".layers." + layer_str + "." + weight_key;
        const std::string suffix1 = ".model.layers." + layer_str + "." + weight_key;
        for (auto iter = tensor_storage_map.begin(); iter != tensor_storage_map.end(); ++iter) {
            const std::string& name = iter->first;
            if ((name.size() >= suffix0.size() && name.compare(name.size() - suffix0.size(), suffix0.size(), suffix0) == 0) ||
                (name.size() >= suffix1.size() && name.compare(name.size() - suffix1.size(), suffix1.size(), suffix1) == 0)) {
                weight_name = name;
                break;
            }
        }
    }
    auto it = weight_name.empty() ? tensor_storage_map.end() : tensor_storage_map.find(weight_name);
    if (it == tensor_storage_map.end()) {
        LOG_ERROR("weight '%s' not found in model", weight_name.c_str());
        return 1;
    }
    const TensorStorage& ts = it->second;
    const ggml_type wtype   = ts.expected_type != GGML_TYPE_COUNT ? ts.expected_type : ts.type;
    if (wtype != GGML_TYPE_Q4_0) {
        LOG_ERROR("weight '%s' type is %s, expected q4_0", weight_name.c_str(), ggml_type_name(wtype));
        return 1;
    }

    ggml_backend_t backend = init_llm_embed_backend();
    if (!backend) {
        LOG_ERROR("failed to init backend");
        return 1;
    }

    ggml_init_params wparams;
    wparams.mem_size   = static_cast<size_t>(256) * 1024 * 1024;
    wparams.mem_buffer = nullptr;
    wparams.no_alloc   = true;
    ggml_context* wctx = ggml_init(wparams);
    if (!wctx) {
        LOG_ERROR("ggml_init failed for weights");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* w = ggml_new_tensor_2d(wctx, wtype, ts.ne[0], ts.ne[1]);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) {
        LOG_ERROR("alloc weight buffer failed");
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::map<std::string, ggml_tensor*> tmap;
    tmap[weight_name] = w;
    if (!model_loader.load_tensors(tmap)) {
        LOG_ERROR("load tensors from model loader failed");
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params cparams;
    cparams.mem_size   = static_cast<size_t>(256) * 1024 * 1024;
    cparams.mem_buffer = nullptr;
    cparams.no_alloc   = true;
    ggml_context* cctx = ggml_init(cparams);
    if (!cctx) {
        LOG_ERROR("ggml_init failed for compute");
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    const int64_t k = ts.ne[0];
    ggml_tensor* x  = nullptr;
    std::vector<float> x_host;
    if (!cli_params.q4_input.empty()) {
        std::ifstream fin(cli_params.q4_input, std::ios::binary);
        if (!fin.is_open()) {
            LOG_ERROR("failed to open q4 input '%s'", cli_params.q4_input.c_str());
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(backend);
            return 1;
        }
        int32_t n_dims = 0;
        int32_t name_len = 0;
        int32_t ttype = 0;
        fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        fin.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));
        if (!fin || n_dims <= 0 || n_dims > 4) {
            LOG_ERROR("invalid q4 input header");
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(backend);
            return 1;
        }
        std::vector<int32_t> ne(static_cast<size_t>(n_dims));
        fin.read(reinterpret_cast<char*>(ne.data()), sizeof(int32_t) * ne.size());
        if (name_len > 0) {
            fin.ignore(name_len);
        }
        if (ttype != GGML_TYPE_F32) {
            LOG_ERROR("q4 input type is %s, expected f32", ggml_type_name((ggml_type)ttype));
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(backend);
            return 1;
        }
        if (ne[0] != k) {
            LOG_ERROR("q4 input ne0=%d does not match K=%lld", ne[0], (long long)k);
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(backend);
            return 1;
        }
        int64_t n = ne.size() > 1 ? ne[1] : 1;
        x = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, k, n);
        x_host.resize(ggml_nelements(x));
        fin.read(reinterpret_cast<char*>(x_host.data()), sizeof(float) * x_host.size());
        if (!fin) {
            LOG_ERROR("failed to read q4 input payload");
            ggml_backend_buffer_free(wbuf);
            ggml_free(wctx);
            ggml_backend_free(backend);
            return 1;
        }
    } else {
        const int64_t n = cli_params.q4_n_token;
        x  = ggml_new_tensor_3d(cctx, GGML_TYPE_F32, k, n, 1);
        x_host.resize(ggml_nelements(x));
        for (size_t i = 0; i < x_host.size(); ++i) {
            uint32_t v = static_cast<uint32_t>(i * 1664525u + 1013904223u);
            float f    = (float)(v % 1024) / 512.0f - 1.0f;
            x_host[i]  = f;
        }
    }

    ggml_tensor* y = ggml_mul_mat(cctx, w, x);
    ggml_cgraph* gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    if (!cbuf) {
        LOG_ERROR("alloc compute buffer failed");
        ggml_free(cctx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_tensor_set(x, x_host.data(), 0, ggml_nbytes(x));
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERROR("q4 matmul compute failed: %s", ggml_status_to_string(status));
        ggml_backend_buffer_free(cbuf);
        ggml_free(cctx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    if (!LLM::LLMRunner::dump_tensor_to_file(cli_params.q4_dump.c_str(), "q4_matmul", y)) {
        LOG_WARN("failed to dump q4 matmul output to '%s'", cli_params.q4_dump.c_str());
    } else {
        LOG_INFO("saved q4 matmul output to '%s'", cli_params.q4_dump.c_str());
    }

    ggml_backend_buffer_free(cbuf);
    ggml_free(cctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}

static int run_rmsnorm_bench(const SDCliParams& cli_params,
                             const SDContextParams& ctx_params) {
    if (ctx_params.llm_path.empty()) {
        LOG_ERROR("rmsnorm bench requires --llm path");
        return 1;
    }
    if (cli_params.rmsnorm_input.empty()) {
        LOG_ERROR("rmsnorm bench requires --rmsnorm-input");
        return 1;
    }

    std::string weight_key;
    if (!parse_rmsnorm_weight_name(cli_params.rmsnorm_weight, &weight_key)) {
        LOG_ERROR("unknown --rmsnorm-weight '%s'", cli_params.rmsnorm_weight.c_str());
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(ctx_params.llm_path, "text_encoders.llm.")) {
        LOG_ERROR("init model loader from file failed: '%s'", ctx_params.llm_path.c_str());
        return 1;
    }

    auto& tensor_storage_map = model_loader.get_tensor_storage_map();
    std::string weight_name;
    if (weight_key == "q_norm" || weight_key == "k_norm") {
        weight_name = "text_encoders.llm.layers.0.self_attn." + weight_key + ".weight";
    } else {
        weight_name = "text_encoders.llm.layers.0." + weight_key + ".weight";
    }
    auto it = tensor_storage_map.find(weight_name);
    if (it == tensor_storage_map.end()) {
        std::string suffix;
        if (weight_key == "q_norm" || weight_key == "k_norm") {
            suffix = ".layers.0.self_attn." + weight_key + ".weight";
        } else {
            suffix = ".layers.0." + weight_key + ".weight";
        }
        for (auto iter = tensor_storage_map.begin(); iter != tensor_storage_map.end(); ++iter) {
            const std::string& name = iter->first;
            if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                weight_name = name;
                it          = iter;
                break;
            }
        }
    }
    if (it == tensor_storage_map.end()) {
        LOG_ERROR("weight '%s' not found in model", weight_name.c_str());
        return 1;
    }
    const TensorStorage& ts = it->second;

    ggml_backend_t backend = init_llm_embed_backend();
    if (!backend) {
        LOG_ERROR("failed to init backend");
        return 1;
    }

    ggml_init_params wparams;
    wparams.mem_size   = static_cast<size_t>(64) * 1024 * 1024;
    wparams.mem_buffer = nullptr;
    wparams.no_alloc   = true;
    ggml_context* wctx = ggml_init(wparams);
    if (!wctx) {
        LOG_ERROR("ggml_init failed for weights");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* w = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, ts.ne[0]);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) {
        LOG_ERROR("alloc weight buffer failed");
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::map<std::string, ggml_tensor*> tmap;
    tmap[weight_name] = w;
    if (!model_loader.load_tensors(tmap)) {
        LOG_ERROR("load tensors from model loader failed");
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    // load input tensor
    std::ifstream fin(cli_params.rmsnorm_input, std::ios::binary);
    if (!fin.is_open()) {
        LOG_ERROR("failed to open rmsnorm input '%s'", cli_params.rmsnorm_input.c_str());
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }
    int32_t n_dims = 0;
    int32_t name_len = 0;
    int32_t ttype = 0;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    fin.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));
    if (!fin || n_dims <= 0 || n_dims > 4) {
        LOG_ERROR("invalid rmsnorm input header");
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }
    std::vector<int32_t> ne(static_cast<size_t>(n_dims));
    fin.read(reinterpret_cast<char*>(ne.data()), sizeof(int32_t) * ne.size());
    if (name_len > 0) {
        fin.ignore(name_len);
    }
    if (ttype != GGML_TYPE_F32) {
        LOG_ERROR("rmsnorm input type is %s, expected f32", ggml_type_name((ggml_type)ttype));
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }
    if (ne[0] != ts.ne[0]) {
        LOG_ERROR("rmsnorm input ne0=%d does not match weight=%lld", ne[0], (long long)ts.ne[0]);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params cparams;
    cparams.mem_size   = static_cast<size_t>(256) * 1024 * 1024;
    cparams.mem_buffer = nullptr;
    cparams.no_alloc   = true;
    ggml_context* cctx = ggml_init(cparams);
    if (!cctx) {
        LOG_ERROR("ggml_init failed for compute");
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<int64_t> ne64(ne.begin(), ne.end());
    ggml_tensor* x = ggml_new_tensor(cctx, GGML_TYPE_F32, n_dims, ne64.data());
    std::vector<float> x_host(ggml_nelements(x));
    fin.read(reinterpret_cast<char*>(x_host.data()), sizeof(float) * x_host.size());
    if (!fin) {
        LOG_ERROR("failed to read rmsnorm input payload");
        ggml_free(cctx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* y = ggml_rms_norm(cctx, x, cli_params.rmsnorm_eps);
    y = ggml_mul(cctx, y, w);

    ggml_cgraph* gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    if (!cbuf) {
        LOG_ERROR("alloc compute buffer failed");
        ggml_free(cctx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_tensor_set(x, x_host.data(), 0, ggml_nbytes(x));
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERROR("rmsnorm compute failed: %s", ggml_status_to_string(status));
        ggml_backend_buffer_free(cbuf);
        ggml_free(cctx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_backend_free(backend);
        return 1;
    }

    if (!LLM::LLMRunner::dump_tensor_to_file(cli_params.rmsnorm_dump.c_str(), "rmsnorm", y)) {
        LOG_WARN("failed to dump rmsnorm output to '%s'", cli_params.rmsnorm_dump.c_str());
    } else {
        LOG_INFO("saved rmsnorm output to '%s'", cli_params.rmsnorm_dump.c_str());
    }

    ggml_backend_buffer_free(cbuf);
    ggml_free(cctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}

struct DumpTensorData {
    int32_t n_dims = 0;
    std::vector<int64_t> ne;
    std::vector<float> data;
};

static bool load_f32_tensor_dump(const std::string& path, DumpTensorData* out) {
    if (out == nullptr) {
        return false;
    }
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        return false;
    }
    int32_t n_dims = 0;
    int32_t name_len = 0;
    int32_t ttype = 0;
    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
    fin.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));
    if (!fin || n_dims <= 0 || n_dims > 4) {
        return false;
    }
    std::vector<int32_t> ne_i32(static_cast<size_t>(n_dims));
    fin.read(reinterpret_cast<char*>(ne_i32.data()), sizeof(int32_t) * ne_i32.size());
    if (name_len > 0) {
        fin.ignore(name_len);
    }
    if (ttype != GGML_TYPE_F32) {
        return false;
    }
    out->n_dims = n_dims;
    out->ne.assign(ne_i32.begin(), ne_i32.end());
    size_t n_elem = 1;
    for (int32_t v : ne_i32) {
        n_elem *= static_cast<size_t>(v);
    }
    out->data.resize(n_elem);
    fin.read(reinterpret_cast<char*>(out->data.data()), sizeof(float) * out->data.size());
    if (!fin) {
        return false;
    }
    return true;
}

static int run_attn_bench(const SDCliParams& cli_params,
                          const SDContextParams& ctx_params) {
    if (ctx_params.llm_path.empty()) {
        LOG_ERROR("attention bench requires --llm path");
        return 1;
    }
    if (cli_params.attn_q_input.empty() || cli_params.attn_k_input.empty() || cli_params.attn_v_input.empty()) {
        LOG_ERROR("attention bench requires --attn-q/--attn-k/--attn-v");
        return 1;
    }
    if (cli_params.attn_n_head <= 0) {
        LOG_ERROR("attention bench requires --attn-n-head > 0");
        return 1;
    }

    DumpTensorData qd, kd, vd;
    if (!load_f32_tensor_dump(cli_params.attn_q_input, &qd)) {
        LOG_ERROR("failed to load attn q dump '%s'", cli_params.attn_q_input.c_str());
        return 1;
    }
    if (!load_f32_tensor_dump(cli_params.attn_k_input, &kd)) {
        LOG_ERROR("failed to load attn k dump '%s'", cli_params.attn_k_input.c_str());
        return 1;
    }
    if (!load_f32_tensor_dump(cli_params.attn_v_input, &vd)) {
        LOG_ERROR("failed to load attn v dump '%s'", cli_params.attn_v_input.c_str());
        return 1;
    }

    ggml_backend_t backend = init_llm_embed_backend();
    if (!backend) {
        LOG_ERROR("failed to init backend");
        return 1;
    }

    ggml_init_params cparams;
    cparams.mem_size   = static_cast<size_t>(512) * 1024 * 1024;
    cparams.mem_buffer = nullptr;
    cparams.no_alloc   = true;
    ggml_context* cctx = ggml_init(cparams);
    if (!cctx) {
        LOG_ERROR("ggml_init failed for compute");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* q = ggml_new_tensor(cctx, GGML_TYPE_F32, qd.n_dims, qd.ne.data());
    ggml_tensor* k = ggml_new_tensor(cctx, GGML_TYPE_F32, kd.n_dims, kd.ne.data());
    ggml_tensor* v = ggml_new_tensor(cctx, GGML_TYPE_F32, vd.n_dims, vd.ne.data());

    ggml_tensor* y = ggml_ext_attention_ext(cctx, backend, q, k, v, cli_params.attn_n_head, nullptr, true, false);

    ggml_cgraph* gf = ggml_new_graph(cctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    if (!cbuf) {
        LOG_ERROR("alloc compute buffer failed");
        ggml_free(cctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_tensor_set(q, qd.data.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kd.data.data(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, vd.data.data(), 0, ggml_nbytes(v));

    ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERROR("attention compute failed: %s", ggml_status_to_string(status));
        ggml_backend_buffer_free(cbuf);
        ggml_free(cctx);
        ggml_backend_free(backend);
        return 1;
    }

    if (!LLM::LLMRunner::dump_tensor_to_file(cli_params.attn_dump.c_str(), "attn", y)) {
        LOG_WARN("failed to dump attention output to '%s'", cli_params.attn_dump.c_str());
    } else {
        LOG_INFO("saved attention output to '%s'", cli_params.attn_dump.c_str());
    }

    ggml_backend_buffer_free(cbuf);
    ggml_free(cctx);
    ggml_backend_free(backend);
    return 0;
}

static int run_flux_fwd_bench(const SDCliParams& cli_params,
                              const SDContextParams& ctx_params,
                              const SDGenerationParams& gen_params) {
    const std::string diffusion_model_path =
        !ctx_params.diffusion_model_path.empty() ? ctx_params.diffusion_model_path : ctx_params.model_path;
    if (diffusion_model_path.empty()) {
        LOG_ERROR("flux forward bench requires --diffusion-model (or --model)");
        return 1;
    }
    if (gen_params.width % 16 != 0 || gen_params.height % 16 != 0) {
        LOG_ERROR("flux forward bench requires width/height divisible by 16, got %dx%d", gen_params.width, gen_params.height);
        return 1;
    }

    ModelLoader model_loader;
    if (!model_loader.init_from_file_and_convert_name(diffusion_model_path, "model.diffusion_model.")) {
        LOG_ERROR("init model loader from file failed: '%s'", diffusion_model_path.c_str());
        return 1;
    }

    const SDVersion model_version = model_loader.get_sd_version();
    if (!sd_version_is_flux(model_version) && !sd_version_is_flux2(model_version)) {
        LOG_ERROR("flux forward bench only supports Flux/Flux2 diffusion models, got version=%d", model_version);
        return 1;
    }

    ggml_backend_t backend = init_llm_embed_backend();
    if (!backend) {
        LOG_ERROR("failed to init backend");
        return 1;
    }

    const bool circular_x = ctx_params.circular || ctx_params.circular_x;
    const bool circular_y = ctx_params.circular || ctx_params.circular_y;

    auto flux = std::make_shared<Flux::FluxRunner>(backend,
                                                   ctx_params.offload_params_to_cpu,
                                                   model_loader.get_tensor_storage_map(),
                                                   "model.diffusion_model",
                                                   model_version,
                                                   false);
    flux->set_flash_attention_enabled(ctx_params.flash_attn || ctx_params.diffusion_flash_attn);
    flux->set_conv2d_direct_enabled(ctx_params.diffusion_conv_direct);
    flux->set_circular_axes(circular_x, circular_y);

    if (!flux->alloc_params_buffer()) {
        LOG_ERROR("alloc flux params buffer failed");
        ggml_backend_free(backend);
        return 1;
    }

    std::map<std::string, ggml_tensor*> tensors;
    flux->get_param_tensors(tensors, "model.diffusion_model");
    if (!model_loader.load_tensors(tensors)) {
        LOG_ERROR("load diffusion tensors failed");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params work_params;
    work_params.mem_size   = static_cast<size_t>(64) * 1024 * 1024;
    work_params.mem_buffer = nullptr;
    work_params.no_alloc   = false;
    ggml_context* work_ctx = ggml_init(work_params);
    if (!work_ctx) {
        LOG_ERROR("ggml_init failed for flux bench work context");
        ggml_backend_free(backend);
        return 1;
    }

    const int64_t latent_w  = gen_params.width / 16;
    const int64_t latent_h  = gen_params.height / 16;
    const int64_t channels  = flux->flux_params.in_channels;
    const int64_t ctx_len   = cli_params.flux_bench_ctx;
    const int64_t ctx_width = flux->flux_params.context_in_dim;
    if (channels <= 0 || ctx_width <= 0) {
        LOG_ERROR("invalid flux bench shape, channels=%lld, context_width=%lld", (long long)channels, (long long)ctx_width);
        ggml_free(work_ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_tensor* x         = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, latent_w, latent_h, channels, 1);
    ggml_tensor* timesteps = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
    ggml_tensor* context   = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, ctx_width, ctx_len, 1);
    ggml_tensor* guidance  = ggml_new_tensor_1d(work_ctx, GGML_TYPE_F32, 1);
    ggml_tensor* y         = nullptr;
    if (flux->flux_params.vec_in_dim > 0) {
        y = ggml_new_tensor_2d(work_ctx, GGML_TYPE_F32, flux->flux_params.vec_in_dim, 1);
    }

    auto fill_tensor = [](ggml_tensor* t, uint32_t seed) {
        std::vector<float> host(ggml_nelements(t));
        uint32_t state = seed;
        for (size_t i = 0; i < host.size(); ++i) {
            state       = state * 1664525u + 1013904223u;
            host[i]     = (float)(state & 0xFFFF) / 32768.0f - 1.0f;
        }
        memcpy(t->data, host.data(), host.size() * sizeof(float));
    };
    fill_tensor(x, 0x1234u);
    fill_tensor(context, 0x5678u);
    if (y != nullptr) {
        memset(y->data, 0, ggml_nbytes(y));
    }
    ggml_set_f32(timesteps, 1000.0f);
    ggml_set_f32(guidance, 3.5f);

    const int n_threads = ctx_params.n_threads > 0 ? ctx_params.n_threads : sd_get_num_physical_cores();
    for (int i = 0; i < cli_params.flux_bench_warmup; ++i) {
        if (!flux->compute(n_threads, x, timesteps, context, nullptr, y, guidance, {}, false, nullptr, nullptr, {})) {
            LOG_ERROR("flux warmup run %d failed", i);
            ggml_free(work_ctx);
            ggml_backend_free(backend);
            return 1;
        }
    }

    int64_t t_start = ggml_time_ms();
    for (int i = 0; i < cli_params.flux_bench_runs; ++i) {
        if (!flux->compute(n_threads, x, timesteps, context, nullptr, y, guidance, {}, false, nullptr, nullptr, {})) {
            LOG_ERROR("flux timed run %d failed", i);
            ggml_free(work_ctx);
            ggml_backend_free(backend);
            return 1;
        }
    }
    int64_t t_end = ggml_time_ms();
    const double avg_ms = static_cast<double>(t_end - t_start) / static_cast<double>(cli_params.flux_bench_runs);

    LOG_INFO("flux step: avg %.3f ms (runs=%d warmup=%d) latent=%lldx%lld ctx=%lld flash_attn=%s",
             avg_ms,
             cli_params.flux_bench_runs,
             cli_params.flux_bench_warmup,
             (long long)latent_w,
             (long long)latent_h,
             (long long)ctx_len,
             (ctx_params.flash_attn || ctx_params.diffusion_flash_attn) ? "on" : "off");

    ggml_free(work_ctx);
    ggml_backend_free(backend);
    return 0;
}

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "CLI Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nGeneration Options:\n";
    options_list[2].print();
}

void parse_args(int argc, const char** argv, SDCliParams& cli_params, SDContextParams& ctx_params, SDGenerationParams& gen_params) {
    std::vector<ArgOptions> options_vec = {cli_params.get_options(), ctx_params.get_options(), gen_params.get_options()};

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        exit(cli_params.normal_exit ? 0 : 1);
    }

    if (!cli_params.process_and_check() ||
        (!cli_params.llm_embed_only && !cli_params.llm_forward_only && !cli_params.llm_generate_only && !cli_params.q4_matmul_bench && !cli_params.rmsnorm_bench && !cli_params.attn_bench && !cli_params.flux_fwd_bench && !ctx_params.process_and_check(cli_params.mode)) ||
        (!cli_params.llm_embed_only && !cli_params.llm_forward_only && !cli_params.llm_generate_only && !cli_params.q4_matmul_bench && !cli_params.rmsnorm_bench && !cli_params.attn_bench && !cli_params.flux_fwd_bench && !gen_params.process_and_check(cli_params.mode, ctx_params.lora_model_dir))) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

std::string get_image_params(const SDCliParams& cli_params, const SDContextParams& ctx_params, const SDGenerationParams& gen_params, int64_t seed) {
    std::string parameter_string = gen_params.prompt_with_lora + "\n";
    if (gen_params.negative_prompt.size() != 0) {
        parameter_string += "Negative prompt: " + gen_params.negative_prompt + "\n";
    }
    parameter_string += "Steps: " + std::to_string(gen_params.sample_params.sample_steps) + ", ";
    parameter_string += "CFG scale: " + std::to_string(gen_params.sample_params.guidance.txt_cfg) + ", ";
    if (gen_params.sample_params.guidance.slg.scale != 0 && gen_params.skip_layers.size() != 0) {
        parameter_string += "SLG scale: " + std::to_string(gen_params.sample_params.guidance.txt_cfg) + ", ";
        parameter_string += "Skip layers: [";
        for (const auto& layer : gen_params.skip_layers) {
            parameter_string += std::to_string(layer) + ", ";
        }
        parameter_string += "], ";
        parameter_string += "Skip layer start: " + std::to_string(gen_params.sample_params.guidance.slg.layer_start) + ", ";
        parameter_string += "Skip layer end: " + std::to_string(gen_params.sample_params.guidance.slg.layer_end) + ", ";
    }
    parameter_string += "Guidance: " + std::to_string(gen_params.sample_params.guidance.distilled_guidance) + ", ";
    parameter_string += "Eta: " + std::to_string(gen_params.sample_params.eta) + ", ";
    parameter_string += "Seed: " + std::to_string(seed) + ", ";
    parameter_string += "Size: " + std::to_string(gen_params.get_resolved_width()) + "x" + std::to_string(gen_params.get_resolved_height()) + ", ";
    parameter_string += "Model: " + sd_basename(ctx_params.model_path) + ", ";
    parameter_string += "RNG: " + std::string(sd_rng_type_name(ctx_params.rng_type)) + ", ";
    if (ctx_params.sampler_rng_type != RNG_TYPE_COUNT) {
        parameter_string += "Sampler RNG: " + std::string(sd_rng_type_name(ctx_params.sampler_rng_type)) + ", ";
    }
    parameter_string += "Sampler: " + std::string(sd_sample_method_name(gen_params.sample_params.sample_method));
    if (!gen_params.custom_sigmas.empty()) {
        parameter_string += ", Custom Sigmas: [";
        for (size_t i = 0; i < gen_params.custom_sigmas.size(); ++i) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << gen_params.custom_sigmas[i];
            parameter_string += oss.str() + (i == gen_params.custom_sigmas.size() - 1 ? "" : ", ");
        }
        parameter_string += "]";
    } else if (gen_params.sample_params.scheduler != SCHEDULER_COUNT) {  // Only show schedule if not using custom sigmas
        parameter_string += " " + std::string(sd_scheduler_name(gen_params.sample_params.scheduler));
    }
    parameter_string += ", ";
    for (const auto& te : {ctx_params.clip_l_path, ctx_params.clip_g_path, ctx_params.t5xxl_path, ctx_params.llm_path, ctx_params.llm_vision_path}) {
        if (!te.empty()) {
            parameter_string += "TE: " + sd_basename(te) + ", ";
        }
    }
    if (!ctx_params.diffusion_model_path.empty()) {
        parameter_string += "Unet: " + sd_basename(ctx_params.diffusion_model_path) + ", ";
    }
    if (!ctx_params.vae_path.empty()) {
        parameter_string += "VAE: " + sd_basename(ctx_params.vae_path) + ", ";
    }
    if (gen_params.clip_skip != -1) {
        parameter_string += "Clip skip: " + std::to_string(gen_params.clip_skip) + ", ";
    }
    parameter_string += "Version: stable-diffusion.cpp";
    return parameter_string;
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDCliParams* cli_params = (SDCliParams*)data;
    log_print(level, log, cli_params->verbose, cli_params->color);
}

bool load_images_from_dir(const std::string dir,
                          std::vector<sd_image_t>& images,
                          int expected_width  = 0,
                          int expected_height = 0,
                          int max_image_num   = 0,
                          bool verbose        = false) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        LOG_ERROR("'%s' is not a valid directory\n", dir.c_str());
        return false;
    }

    std::vector<fs::directory_entry> entries;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const fs::directory_entry& a, const fs::directory_entry& b) {
                  return a.path().filename().string() < b.path().filename().string();
              });

    for (const auto& entry : entries) {
        std::string path = entry.path().string();
        std::string ext  = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
            LOG_DEBUG("load image %zu from '%s'", images.size(), path.c_str());
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height, expected_width, expected_height);
            if (image_buffer == nullptr) {
                LOG_ERROR("load image from '%s' failed", path.c_str());
                return false;
            }

            images.push_back({(uint32_t)width,
                              (uint32_t)height,
                              3,
                              image_buffer});

            if (max_image_num > 0 && images.size() >= max_image_num) {
                break;
            }
        }
    }
    return true;
}

void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data) {
    (void)step;
    (void)is_noisy;
    SDCliParams* cli_params = (SDCliParams*)data;
    // is_noisy is set to true if the preview corresponds to noisy latents, false if it's denoised latents
    // unused in this app, it will either be always noisy or always denoised here
    if (frame_count == 1) {
        stbi_write_png(cli_params->preview_path.c_str(), image->width, image->height, image->channel, image->data, 0);
    } else {
        create_mjpg_avi_from_sd_images(cli_params->preview_path.c_str(), image, frame_count, cli_params->preview_fps);
    }
}

std::string format_frame_idx(std::string pattern, int frame_idx) {
    std::smatch match;
    std::string result = pattern;
    while (std::regex_search(result, match, format_specifier_regex)) {
        std::string specifier = match.str(1);
        char buffer[32];
        snprintf(buffer, sizeof(buffer), specifier.c_str(), frame_idx);
        result.replace(match.position(1), match.length(1), buffer);
    }

    // Then replace all '%%' with '%'
    size_t pos = 0;
    while ((pos = result.find("%%", pos)) != std::string::npos) {
        result.replace(pos, 2, "%");
        pos += 1;
    }
    return result;
}

bool save_results(const SDCliParams& cli_params,
                  const SDContextParams& ctx_params,
                  const SDGenerationParams& gen_params,
                  sd_image_t* results,
                  int num_results) {
    if (results == nullptr || num_results <= 0) {
        return false;
    }

    namespace fs      = std::filesystem;
    fs::path out_path = cli_params.output_path;

    if (!out_path.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
        if (ec) {
            LOG_ERROR("failed to create directory '%s': %s",
                      out_path.parent_path().string().c_str(), ec.message().c_str());
            return false;
        }
    }

    fs::path base_path = out_path;
    fs::path ext       = out_path.has_extension() ? out_path.extension() : fs::path{};
    if (!ext.empty())
        base_path.replace_extension();

    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    bool is_jpg = (ext_lower == ".jpg" || ext_lower == ".jpeg" || ext_lower == ".jpe");

    int output_begin_idx = cli_params.output_begin_idx;
    if (output_begin_idx < 0) {
        output_begin_idx = 0;
    }

    auto write_image = [&](const fs::path& path, int idx) {
        const sd_image_t& img = results[idx];
        if (!img.data)
            return;

        std::string params = get_image_params(cli_params, ctx_params, gen_params, gen_params.seed + idx);
        int ok             = 0;
        if (is_jpg) {
            ok = stbi_write_jpg(path.string().c_str(), img.width, img.height, img.channel, img.data, 90, params.c_str());
        } else {
            ok = stbi_write_png(path.string().c_str(), img.width, img.height, img.channel, img.data, 0, params.c_str());
        }
        LOG_INFO("save result image %d to '%s' (%s)", idx, path.string().c_str(), ok ? "success" : "failure");
    };

    if (std::regex_search(cli_params.output_path, format_specifier_regex)) {
        if (!is_jpg && ext_lower != ".png")
            ext = ".png";
        fs::path pattern = base_path;
        pattern += ext;

        for (int i = 0; i < num_results; ++i) {
            fs::path img_path = format_frame_idx(pattern.string(), output_begin_idx + i);
            write_image(img_path, i);
        }
        return true;
    }

    if (cli_params.mode == VID_GEN && num_results > 1) {
        if (ext_lower != ".avi")
            ext = ".avi";
        fs::path video_path = base_path;
        video_path += ext;
        create_mjpg_avi_from_sd_images(video_path.string().c_str(), results, num_results, gen_params.fps);
        LOG_INFO("save result MJPG AVI video to '%s'", video_path.string().c_str());
        return true;
    }

    if (!is_jpg && ext_lower != ".png")
        ext = ".png";

    for (int i = 0; i < num_results; ++i) {
        fs::path img_path = base_path;
        if (num_results > 1) {
            img_path += "_" + std::to_string(output_begin_idx + i);
        }
        img_path += ext;
        write_image(img_path, i);
    }

    return true;
}

int main(int argc, const char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

    SDCliParams cli_params;
    SDContextParams ctx_params;
    SDGenerationParams gen_params;

    parse_args(argc, argv, cli_params, ctx_params, gen_params);
    if (gen_params.video_frames > 4) {
        size_t last_dot_pos   = cli_params.preview_path.find_last_of(".");
        std::string base_path = cli_params.preview_path;
        std::string file_ext  = "";
        if (last_dot_pos != std::string::npos) {  // filename has extension
            base_path = cli_params.preview_path.substr(0, last_dot_pos);
            file_ext  = cli_params.preview_path.substr(last_dot_pos);
            std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);
        }
        if (file_ext == ".png") {
            cli_params.preview_path = base_path + ".avi";
        }
    }
    cli_params.preview_fps = gen_params.fps;
    if (cli_params.preview_method == PREVIEW_PROJ)
        cli_params.preview_fps /= 4;

    sd_set_log_callback(sd_log_cb, (void*)&cli_params);
    log_verbose = cli_params.verbose;
    log_color   = cli_params.color;
    sd_set_preview_callback(step_callback,
                            cli_params.preview_method,
                            cli_params.preview_interval,
                            !cli_params.preview_noisy,
                            cli_params.preview_noisy,
                            (void*)&cli_params);

    LOG_DEBUG("version: %s", version_string().c_str());
    LOG_DEBUG("%s", sd_get_system_info());
    LOG_DEBUG("%s", cli_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", gen_params.to_string().c_str());

    if (cli_params.llm_embed_only) {
        return run_llm_embed_only(cli_params, ctx_params);
    }
    if (cli_params.llm_forward_only) {
        return run_llm_forward_only(cli_params, ctx_params, gen_params);
    }
    if (cli_params.llm_generate_only) {
        return run_llm_generate_only(cli_params, ctx_params, gen_params);
    }
    if (cli_params.q4_matmul_bench) {
        return run_q4_matmul_bench(cli_params, ctx_params);
    }
    if (cli_params.rmsnorm_bench) {
        return run_rmsnorm_bench(cli_params, ctx_params);
    }
    if (cli_params.attn_bench) {
        return run_attn_bench(cli_params, ctx_params);
    }
    if (cli_params.flux_fwd_bench) {
        return run_flux_fwd_bench(cli_params, ctx_params, gen_params);
    }

    if (cli_params.mode == CONVERT) {
        bool success = convert(ctx_params.model_path.c_str(),
                               ctx_params.vae_path.c_str(),
                               cli_params.output_path.c_str(),
                               ctx_params.wtype,
                               ctx_params.tensor_type_rules.c_str(),
                               cli_params.convert_name);
        if (!success) {
            LOG_ERROR("convert '%s'/'%s' to '%s' failed",
                      ctx_params.model_path.c_str(),
                      ctx_params.vae_path.c_str(),
                      cli_params.output_path.c_str());
            return 1;
        } else {
            LOG_INFO("convert '%s'/'%s' to '%s' success",
                     ctx_params.model_path.c_str(),
                     ctx_params.vae_path.c_str(),
                     cli_params.output_path.c_str());
            return 0;
        }
    }

    bool vae_decode_only     = true;
    sd_image_t init_image    = {0, 0, 3, nullptr};
    sd_image_t end_image     = {0, 0, 3, nullptr};
    sd_image_t control_image = {0, 0, 3, nullptr};
    sd_image_t mask_image    = {0, 0, 1, nullptr};
    std::vector<sd_image_t> ref_images;
    std::vector<sd_image_t> pmid_images;
    std::vector<sd_image_t> control_frames;

    auto release_all_resources = [&]() {
        free(init_image.data);
        free(end_image.data);
        free(control_image.data);
        free(mask_image.data);
        for (auto image : ref_images) {
            free(image.data);
            image.data = nullptr;
        }
        ref_images.clear();
        for (auto image : pmid_images) {
            free(image.data);
            image.data = nullptr;
        }
        pmid_images.clear();
        for (auto image : control_frames) {
            free(image.data);
            image.data = nullptr;
        }
        control_frames.clear();
    };

    auto load_image_and_update_size = [&](const std::string& path,
                                          sd_image_t& image,
                                          bool resize_image    = true,
                                          int expected_channel = 3) -> bool {
        int expected_width  = 0;
        int expected_height = 0;
        if (resize_image && gen_params.width_and_height_are_set()) {
            expected_width  = gen_params.width;
            expected_height = gen_params.height;
        }

        if (!load_sd_image_from_file(&image, path.c_str(), expected_width, expected_height, expected_channel)) {
            LOG_ERROR("load image from '%s' failed", path.c_str());
            release_all_resources();
            return false;
        }

        gen_params.set_width_and_height_if_unset(image.width, image.height);
        return true;
    };

    if (gen_params.init_image_path.size() > 0) {
        vae_decode_only = false;
        if (!load_image_and_update_size(gen_params.init_image_path, init_image)) {
            return 1;
        }
    }

    if (gen_params.end_image_path.size() > 0) {
        vae_decode_only = false;
        if (!load_image_and_update_size(gen_params.init_image_path, end_image)) {
            return 1;
        }
    }

    if (gen_params.ref_image_paths.size() > 0) {
        vae_decode_only = false;
        for (auto& path : gen_params.ref_image_paths) {
            sd_image_t ref_image = {0, 0, 3, nullptr};
            if (!load_image_and_update_size(path, ref_image, false)) {
                return 1;
            }
            ref_images.push_back(ref_image);
        }
    }

    if (gen_params.mask_image_path.size() > 0) {
        if (!load_sd_image_from_file(&mask_image,
                                     gen_params.mask_image_path.c_str(),
                                     gen_params.get_resolved_width(),
                                     gen_params.get_resolved_height(),
                                     1)) {
            LOG_ERROR("load image from '%s' failed", gen_params.mask_image_path.c_str());
            release_all_resources();
            return 1;
        }
    } else {
        mask_image.data = (uint8_t*)malloc(gen_params.get_resolved_width() * gen_params.get_resolved_height());
        if (mask_image.data == nullptr) {
            LOG_ERROR("malloc mask image failed");
            release_all_resources();
            return 1;
        }
        mask_image.width  = gen_params.get_resolved_width();
        mask_image.height = gen_params.get_resolved_height();
        memset(mask_image.data, 255, gen_params.get_resolved_width() * gen_params.get_resolved_height());
    }

    if (gen_params.control_image_path.size() > 0) {
        if (!load_sd_image_from_file(&control_image,
                                     gen_params.control_image_path.c_str(),
                                     gen_params.get_resolved_width(),
                                     gen_params.get_resolved_height())) {
            LOG_ERROR("load image from '%s' failed", gen_params.control_image_path.c_str());
            release_all_resources();
            return 1;
        }
        if (cli_params.canny_preprocess) {  // apply preprocessor
            preprocess_canny(control_image,
                             0.08f,
                             0.08f,
                             0.8f,
                             1.0f,
                             false);
        }
    }

    if (!gen_params.control_video_path.empty()) {
        if (!load_images_from_dir(gen_params.control_video_path,
                                  control_frames,
                                  gen_params.get_resolved_width(),
                                  gen_params.get_resolved_height(),
                                  gen_params.video_frames,
                                  cli_params.verbose)) {
            release_all_resources();
            return 1;
        }
    }

    if (!gen_params.pm_id_images_dir.empty()) {
        if (!load_images_from_dir(gen_params.pm_id_images_dir,
                                  pmid_images,
                                  0,
                                  0,
                                  0,
                                  cli_params.verbose)) {
            release_all_resources();
            return 1;
        }
    }

    if (cli_params.mode == VID_GEN) {
        vae_decode_only = false;
    }
    if (!cli_params.load_latent_path.empty()) {
        vae_decode_only = true;
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(vae_decode_only, true, cli_params.taesd_preview);

    sd_image_t* results = nullptr;
    int num_results     = 0;

    if (cli_params.mode == UPSCALE) {
        num_results = 1;
        results     = (sd_image_t*)calloc(num_results, sizeof(sd_image_t));
        if (results == nullptr) {
            LOG_INFO("failed to allocate results array");
            release_all_resources();
            return 1;
        }

        results[0]      = init_image;
        init_image.data = nullptr;
    } else {
        sd_ctx_t* sd_ctx = new_sd_ctx(&sd_ctx_params);

        if (sd_ctx == nullptr) {
            LOG_INFO("new_sd_ctx_t failed");
            release_all_resources();
            return 1;
        }

        if (!cli_params.load_latent_path.empty()) {
            results = decode_latent_file(sd_ctx, cli_params.load_latent_path.c_str());
            if (results == nullptr) {
                LOG_ERROR("decode_latent_file failed");
                free_sd_ctx(sd_ctx);
                release_all_resources();
                return 1;
            }
            num_results = 1;
            free_sd_ctx(sd_ctx);
        } else {
            if (gen_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
                gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
            }

            if (gen_params.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
                gen_params.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx);
            }

            if (gen_params.sample_params.scheduler == SCHEDULER_COUNT) {
                gen_params.sample_params.scheduler = sd_get_default_scheduler(sd_ctx, gen_params.sample_params.sample_method);
            }

            if (cli_params.mode == IMG_GEN) {
                sd_img_gen_params_t img_gen_params = {
                    gen_params.lora_vec.data(),
                    static_cast<uint32_t>(gen_params.lora_vec.size()),
                    gen_params.prompt.c_str(),
                    gen_params.negative_prompt.c_str(),
                    gen_params.clip_skip,
                    init_image,
                    ref_images.data(),
                    (int)ref_images.size(),
                    gen_params.auto_resize_ref_image,
                    gen_params.increase_ref_index,
                    mask_image,
                    gen_params.get_resolved_width(),
                    gen_params.get_resolved_height(),
                    gen_params.sample_params,
                    gen_params.strength,
                    gen_params.seed,
                    gen_params.batch_count,
                    control_image,
                    gen_params.control_strength,
                    {
                        pmid_images.data(),
                        (int)pmid_images.size(),
                        gen_params.pm_id_embed_path.c_str(),
                        gen_params.pm_style_strength,
                    },  // pm_params
                    ctx_params.vae_tiling_params,
                    gen_params.cache_params,
                };

                results     = generate_image(sd_ctx, &img_gen_params);
                num_results = gen_params.batch_count;
            } else if (cli_params.mode == VID_GEN) {
                sd_vid_gen_params_t vid_gen_params = {
                    gen_params.lora_vec.data(),
                    static_cast<uint32_t>(gen_params.lora_vec.size()),
                    gen_params.prompt.c_str(),
                    gen_params.negative_prompt.c_str(),
                    gen_params.clip_skip,
                    init_image,
                    end_image,
                    control_frames.data(),
                    (int)control_frames.size(),
                    gen_params.get_resolved_width(),
                    gen_params.get_resolved_height(),
                    gen_params.sample_params,
                    gen_params.high_noise_sample_params,
                    gen_params.moe_boundary,
                    gen_params.strength,
                    gen_params.seed,
                    gen_params.video_frames,
                    gen_params.vace_strength,
                    ctx_params.vae_tiling_params,
                    gen_params.cache_params,
                };

                results = generate_video(sd_ctx, &vid_gen_params, &num_results);
            }

            if (results == nullptr) {
                LOG_ERROR("generate failed");
                free_sd_ctx(sd_ctx);
                return 1;
            }

            free_sd_ctx(sd_ctx);
        }
    }

    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (ctx_params.esrgan_path.size() > 0 && gen_params.upscale_repeats > 0) {
        upscaler_ctx_t* upscaler_ctx = new_upscaler_ctx(ctx_params.esrgan_path.c_str(),
                                                        ctx_params.offload_params_to_cpu,
                                                        ctx_params.diffusion_conv_direct,
                                                        ctx_params.n_threads,
                                                        gen_params.upscale_tile_size);

        if (upscaler_ctx == nullptr) {
            LOG_ERROR("new_upscaler_ctx failed");
        } else {
            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                sd_image_t current_image = results[i];
                for (int u = 0; u < gen_params.upscale_repeats; ++u) {
                    sd_image_t upscaled_image = upscale(upscaler_ctx, current_image, upscale_factor);
                    if (upscaled_image.data == nullptr) {
                        LOG_ERROR("upscale failed");
                        break;
                    }
                    free(current_image.data);
                    current_image = upscaled_image;
                }
                results[i] = current_image;  // Set the final upscaled image as the result
            }
        }
    }

    if (!save_results(cli_params, ctx_params, gen_params, results, num_results)) {
        return 1;
    }

    for (int i = 0; i < num_results; i++) {
        free(results[i].data);
        results[i].data = nullptr;
    }
    free(results);

    release_all_resources();

    return 0;
}
