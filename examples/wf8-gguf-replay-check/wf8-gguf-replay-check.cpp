#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#if defined(__ANDROID__)
#include <dlfcn.h>
#include "../../ggml/src/ggml-htp/dsprpc_interface.h"
#include "../../ggml/src/ggml-htp/message.h"
#include "../../ggml/src/ggml-htp/op_reg.h"

extern "C" {
struct ggml_compute_params {
    int ith, nth;
    size_t wsize;
    void * wdata;
    struct ggml_threadpool * threadpool;
};

bool htp_ops_support_op(const struct ggml_tensor * dst);
int  htp_ops_compute_op(struct ggml_compute_params * params, struct ggml_tensor * dst);
}
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if defined(__ANDROID__) && defined(_LIBCPP_VERSION)
namespace fs = std::__fs::filesystem;
#else
namespace fs = std::filesystem;
#endif

struct TensorDumpMeta {
    char type[16] = {0};
    int ne[4] = {0, 0, 0, 0};
    int n_dims = 0;
    size_t nbytes = 0;
};

static ggml_backend_t init_backend_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_dev_name(dev), name) == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

static int load_tensor_dump_meta(const fs::path & path, TensorDumpMeta & meta) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::fprintf(stderr, "failed to open meta: %s\n", path.string().c_str());
        return -1;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (std::sscanf(line.c_str(), "type=%15s", meta.type) == 1) {
            continue;
        }
        if (std::sscanf(line.c_str(), "nbytes=%zu", &meta.nbytes) == 1) {
            continue;
        }
        if (line.rfind("ne=", 0) == 0) {
            const char * p = line.c_str() + 3;
            meta.n_dims = 0;
            while (*p && meta.n_dims < 4) {
                meta.ne[meta.n_dims++] = std::atoi(p);
                while (*p && *p != ',') ++p;
                if (*p == ',') ++p;
            }
        }
    }
    if (meta.n_dims < 2 || std::strcmp(meta.type, "f32") != 0) {
        std::fprintf(stderr, "bad meta: %s\n", path.string().c_str());
        return -1;
    }
    return 0;
}

static std::vector<uint8_t> load_bytes(const fs::path & path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("failed to open file");
    }
    ifs.seekg(0, std::ios::end);
    const size_t size = (size_t) ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(size);
    ifs.read((char *) out.data(), (std::streamsize) out.size());
    return out;
}

#if defined(__ANDROID__)
struct DirectRpcmemBuf {
    void * ptr = nullptr;
    int fd = -1;
    size_t bytes = 0;
    bool mapped = false;
};

static size_t align_up_128(size_t v) {
    return (v + 127u) & ~size_t(127u);
}

static void alloc_direct_rpcmem(DirectRpcmemBuf & buf, size_t bytes) {
    const bool replay_debug = std::getenv("REPLAY_DEBUG") != nullptr;
    buf.bytes = align_up_128(bytes);
    if (replay_debug) {
        std::fprintf(stderr, "replay_debug: rpcmem_alloc request=%zu aligned=%zu\n", bytes, buf.bytes);
    }
    buf.ptr = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_UNCACHED, (int) buf.bytes);
    if (!buf.ptr) {
        throw std::runtime_error("rpcmem_alloc failed");
    }
    if (replay_debug) {
        std::fprintf(stderr, "replay_debug: rpcmem_alloc ok ptr=%p\n", buf.ptr);
    }
    buf.fd = rpcmem_to_fd(buf.ptr);
    if (buf.fd < 0) {
        rpcmem_free(buf.ptr);
        buf.ptr = nullptr;
        throw std::runtime_error("rpcmem_to_fd failed");
    }
    if (replay_debug) {
        std::fprintf(stderr, "replay_debug: rpcmem_to_fd ok fd=%d\n", buf.fd);
    }
    if (fastrpc_mmap(CDSP_DOMAIN_ID, buf.fd, buf.ptr, 0, buf.bytes, FASTRPC_MAP_FD) != 0) {
        rpcmem_free(buf.ptr);
        buf.ptr = nullptr;
        buf.fd = -1;
        throw std::runtime_error("fastrpc_mmap failed");
    }
    if (replay_debug) {
        std::fprintf(stderr, "replay_debug: fastrpc_mmap ok fd=%d bytes=%zu\n", buf.fd, buf.bytes);
    }
    buf.mapped = true;
}

static void free_direct_rpcmem(DirectRpcmemBuf & buf) {
    if (buf.mapped && buf.ptr && buf.fd >= 0 && buf.bytes > 0) {
        (void) fastrpc_munmap(CDSP_DOMAIN_ID, buf.fd, buf.ptr, buf.bytes);
    }
    if (buf.ptr) {
        rpcmem_free(buf.ptr);
    }
    buf = {};
}

struct HtpOpsFns {
    void * dl = nullptr;
    int (*open_dsp_session)(int, int) = nullptr;
    void (*close_dsp_session)() = nullptr;
    void (*init_htp_backend)() = nullptr;
    int (*create_htp_message_channel)(int, unsigned int) = nullptr;
    int (*htp_ops_destroy_channel)(long long) = nullptr;
    long long (*get_global_handle)() = nullptr;
};

static void close_htp_ops_fns(HtpOpsFns & fns) {
    if (fns.dl) {
        dlclose(fns.dl);
    }
    fns = {};
}

static void open_htp_ops_fns(HtpOpsFns & fns) {
    fns.dl = dlopen("libhtp_ops.so", RTLD_LAZY | RTLD_LOCAL);
    if (!fns.dl) {
        throw std::runtime_error("dlopen(libhtp_ops.so) failed");
    }

    fns.open_dsp_session = reinterpret_cast<int (*)(int, int)>(dlsym(fns.dl, "open_dsp_session"));
    fns.close_dsp_session = reinterpret_cast<void (*)()>(dlsym(fns.dl, "close_dsp_session"));
    fns.init_htp_backend = reinterpret_cast<void (*)()>(dlsym(fns.dl, "init_htp_backend"));
    fns.create_htp_message_channel =
        reinterpret_cast<int (*)(int, unsigned int)>(dlsym(fns.dl, "create_htp_message_channel"));
    fns.htp_ops_destroy_channel = reinterpret_cast<int (*)(long long)>(dlsym(fns.dl, "htp_ops_destroy_channel"));
    fns.get_global_handle = reinterpret_cast<long long (*)()>(dlsym(fns.dl, "get_global_handle"));

    if (!fns.open_dsp_session || !fns.close_dsp_session || !fns.init_htp_backend ||
        !fns.create_htp_message_channel || !fns.htp_ops_destroy_channel || !fns.get_global_handle) {
        close_htp_ops_fns(fns);
        throw std::runtime_error("libhtp_ops.so symbols missing");
    }
}

static int run_single_wf8_request_via_channel(
        void * chan_ptr,
        int out_fd,
        int act_fd,
        int weight_fd,
        int m,
        int k,
        int n,
        uint32_t flags,
        int64_t * elapsed_us) {
    auto * msg = reinterpret_cast<MessageHeader *>(chan_ptr);

    const RequestHeader req_hdr {
        .state = 0,
        .type  = REQUEST_TYPE_OP_COMPUTE,
    };
    const OpComputeRequest op_req {
        .op = HTP_OPS_MAT_MUL_PERMUTED_WF8A16,
    };
    const MatMulParams mm {
        .output     = { out_fd, 0 },
        .activation = { act_fd, 0 },
        .weight     = { weight_fd, 0 },
        .m          = m,
        .k          = k,
        .n          = n,
        .flags      = flags,
    };

    const size_t req_size = sizeof(req_hdr) + sizeof(op_req) + sizeof(mm);
    msg->state.d = 0;
    msg->checksum = 0;
    msg->n_reqs = 1;
    msg->req_offsets[0] = (int32_t) message_header_size(msg);
    msg->req_offsets[1] = (int32_t) (msg->req_offsets[0] + req_size);

    uint8_t * p = reinterpret_cast<uint8_t *>(message_header_get_request_ptr(msg, 0));
    std::memcpy(p, &req_hdr, sizeof(req_hdr));
    p += sizeof(req_hdr);
    std::memcpy(p, &op_req, sizeof(op_req));
    p += sizeof(op_req);
    std::memcpy(p, &mm, sizeof(mm));

    const int64_t t0 = ggml_time_us();
    msg->state.v[0] = 1;
    while (msg->state.v[1] != 1) {
    }
    const int64_t t1 = ggml_time_us();
    if (elapsed_us != nullptr) {
        *elapsed_us = t1 - t0;
    }
    return message_header_get_request_ptr(msg, 0)->state;
}

static int replay_wf8_direct_via_host_session(
        const uint8_t * weight_bytes,
        size_t weight_nbytes,
        const uint8_t * act_bytes,
        size_t act_nbytes,
        const uint8_t * ref_bytes,
        size_t ref_nbytes,
        int m,
        int k,
        int n,
        int repeat,
        const char * out_dump_path,
        const char * tensor_name,
        const char * gguf_path) {
    HtpOpsFns fns;
    DirectRpcmemBuf chan_buf, weight_buf, act_buf, out_buf;
    const bool replay_debug = std::getenv("REPLAY_DEBUG") != nullptr;

    try {
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: host_session open symbols\n");
        }
        rpcmem_init();
        open_htp_ops_fns(fns);
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: host_session open_dsp_session\n");
        }
        if (fns.open_dsp_session(CDSP_DOMAIN_ID, 1) != 0) {
            throw std::runtime_error("open_dsp_session failed");
        }
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: host_session init_backend\n");
        }
        fns.init_htp_backend();

        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: host_session alloc rpcmem\n");
        }
        alloc_direct_rpcmem(chan_buf, 4096);
        alloc_direct_rpcmem(weight_buf, weight_nbytes);
        alloc_direct_rpcmem(act_buf, act_nbytes);
        alloc_direct_rpcmem(out_buf, ref_nbytes);

        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: host_session create channel\n");
        }
        if (fns.create_htp_message_channel(chan_buf.fd, (unsigned int) chan_buf.bytes) != 0) {
            throw std::runtime_error("create_htp_message_channel failed");
        }

        std::memcpy(weight_buf.ptr, weight_bytes, weight_nbytes);
        std::memcpy(act_buf.ptr, act_bytes, act_nbytes);
        std::memset(out_buf.ptr, 0, out_buf.bytes);

        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: host_session warmup\n");
        }
        int64_t warmup_us = 0;
        if (run_single_wf8_request_via_channel(chan_buf.ptr, out_buf.fd, act_buf.fd, weight_buf.fd, m, k, n, 0, &warmup_us) != 0) {
            throw std::runtime_error("warmup failed");
        }

        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: host_session timed loop\n");
        }
        int64_t total_us = 0;
        for (int r = 0; r < repeat; ++r) {
            int64_t elapsed_us = 0;
            if (run_single_wf8_request_via_channel(chan_buf.ptr, out_buf.fd, act_buf.fd, weight_buf.fd, m, k, n, 0, &elapsed_us) != 0) {
                throw std::runtime_error("compute failed");
            }
            total_us += elapsed_us;
        }

        std::vector<float> out((size_t) m * (size_t) n, 0.0f);
        std::memcpy(out.data(), out_buf.ptr, out.size() * sizeof(float));
        const float * ref = reinterpret_cast<const float *>(ref_bytes);

        if (out_dump_path != nullptr && out_dump_path[0] != '\0') {
            std::ofstream ofs(out_dump_path, std::ios::binary);
            if (!ofs.is_open()) {
                throw std::runtime_error("failed to open out dump");
            }
            ofs.write(reinterpret_cast<const char *>(out.data()), (std::streamsize) (out.size() * sizeof(float)));
            if (!ofs) {
                throw std::runtime_error("failed to write out dump");
            }
        }

        int bad = 0;
        float mae = 0.0f;
        float mse = 0.0f;
        float max_abs = 0.0f;
        double dot = 0.0;
        double na = 0.0;
        double nb = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            const float a = out[i];
            const float b = ref[i];
            const float d = a - b;
            const float ad = std::fabs(d);
            mae += ad;
            mse += d * d;
            if (ad > max_abs) {
                max_abs = ad;
            }
            if (ad > 1e-2f) {
                ++bad;
            }
            dot += (double) a * (double) b;
            na += (double) a * (double) a;
            nb += (double) b * (double) b;
        }

        mae /= (float) out.size();
        mse /= (float) out.size();
        const float rmse = std::sqrt(mse);
        const double cos = (na == 0.0 || nb == 0.0) ? 0.0 : dot / std::sqrt(na * nb);
        const double us_per_run = (double) total_us / (double) repeat;
        const double tflops = 2.0 * (double) m * (double) k * (double) n / us_per_run / 1.0e6;

        std::printf("replay/gguf tensor=%s wtype=wf8_hmx m=%d k=%d n=%d repeat=%d bad=%d/%zu mae=%g rmse=%g max_abs=%g cos=%.9f us_per_run=%g tflops=%g warmup_us=%lld first=%g/%g gguf=%s\n",
                    tensor_name, m, k, n, repeat, bad, out.size(), mae, rmse, max_abs, cos,
                    us_per_run, tflops, (long long) warmup_us, out[0], ref[0], gguf_path);

        (void) fns.htp_ops_destroy_channel(fns.get_global_handle());
        free_direct_rpcmem(out_buf);
        free_direct_rpcmem(act_buf);
        free_direct_rpcmem(weight_buf);
        free_direct_rpcmem(chan_buf);
        fns.close_dsp_session();
        close_htp_ops_fns(fns);
        rpcmem_deinit();
        return 0;
    } catch (const std::exception &) {
        if (fns.htp_ops_destroy_channel && fns.get_global_handle) {
            (void) fns.htp_ops_destroy_channel(fns.get_global_handle());
        }
        free_direct_rpcmem(out_buf);
        free_direct_rpcmem(act_buf);
        free_direct_rpcmem(weight_buf);
        free_direct_rpcmem(chan_buf);
        if (fns.close_dsp_session) {
            fns.close_dsp_session();
        }
        close_htp_ops_fns(fns);
        rpcmem_deinit();
        throw;
    }
}
#endif

struct RunCtx {
    gguf_context * gguf = nullptr;
    ggml_context * gguf_data_ctx = nullptr;
    ggml_context * weight_ctx = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * out = nullptr;
    ggml_cgraph * gf = nullptr;
};

static void free_run(RunCtx & run) {
    if (run.threadpool) ggml_threadpool_free(run.threadpool);
    if (run.weight_buf) ggml_backend_buffer_free(run.weight_buf);
    if (run.buf) ggml_backend_buffer_free(run.buf);
    if (run.ctx) ggml_free(run.ctx);
    if (run.weight_ctx) ggml_free(run.weight_ctx);
    if (run.gguf) gguf_free(run.gguf);
    if (run.gguf_data_ctx) ggml_free(run.gguf_data_ctx);
    if (run.backend) ggml_backend_free(run.backend);
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s gguf tensor_name act_meta act_bin ref_meta ref_bin [repeat=1] [out_dump_bin]\n", argv[0]);
        return 1;
    }

    const fs::path gguf_path = argv[1];
    const std::string tensor_name = argv[2];
    const fs::path act_meta_path = argv[3];
    const fs::path act_bin_path = argv[4];
    const fs::path ref_meta_path = argv[5];
    const fs::path ref_bin_path = argv[6];
    const int repeat = argc > 7 ? std::max(1, std::atoi(argv[7])) : 1;
    const char * out_dump_path = argc > 8 ? argv[8] : nullptr;

    RunCtx run;

    try {
        const bool replay_debug = std::getenv("REPLAY_DEBUG") != nullptr;
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: main entered\n");
        }
        TensorDumpMeta act_meta, ref_meta;
        if (load_tensor_dump_meta(act_meta_path, act_meta) != 0 ||
            load_tensor_dump_meta(ref_meta_path, ref_meta) != 0) {
            throw std::runtime_error("failed to parse dump meta");
        }
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: meta loaded\n");
        }

        const int k = act_meta.ne[0];
        const int m = act_meta.ne[1];
        const int n = ref_meta.ne[0];
        if (ref_meta.ne[1] != m) {
            throw std::runtime_error("ref/act shape mismatch");
        }

        gguf_init_params params = {
            /*.no_alloc =*/ false,
            /*.ctx      =*/ &run.gguf_data_ctx,
        };
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: before gguf_init_from_file\n");
        }
        run.gguf = gguf_init_from_file(gguf_path.string().c_str(), params);
        if (run.gguf == nullptr || run.gguf_data_ctx == nullptr) {
            throw std::runtime_error("failed to load gguf");
        }
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: gguf loaded path=%s\n", gguf_path.string().c_str());
        }

        ggml_tensor * src_weight = ggml_get_first_tensor(run.gguf_data_ctx);
        while (src_weight != nullptr && std::strcmp(src_weight->name, tensor_name.c_str()) != 0) {
            src_weight = ggml_get_next_tensor(run.gguf_data_ctx, src_weight);
        }
        if (src_weight == nullptr) {
            throw std::runtime_error("tensor not found in gguf");
        }
        if (src_weight->ne[0] != k || src_weight->ne[1] != n) {
            throw std::runtime_error("tensor shape mismatch");
        }
        if (replay_debug) {
            std::fprintf(stderr, "replay_debug: tensor type=%d k=%d n=%d\n", (int) src_weight->type, k, n);
        }

        const std::vector<uint8_t> act_bytes = load_bytes(act_bin_path);
        const std::vector<uint8_t> ref_bytes = load_bytes(ref_bin_path);
        if (act_bytes.size() != (size_t) m * k * sizeof(float) ||
            ref_bytes.size() != (size_t) m * n * sizeof(float)) {
            throw std::runtime_error("bad act/ref file sizes");
        }

#if defined(__ANDROID__)
        const bool force_graph = std::getenv("REPLAY_FORCE_GRAPH") != nullptr;
        if (!force_graph && src_weight->type == GGML_TYPE_WF8_HMX_PREPACK) {
            if (replay_debug) {
                std::fprintf(stderr, "replay_debug: entering host-session wf8 direct path data=%p nbytes=%zu\n",
                             src_weight->data, ggml_nbytes(src_weight));
            }
            const int rc = replay_wf8_direct_via_host_session(
                reinterpret_cast<const uint8_t *>(src_weight->data),
                ggml_nbytes(src_weight),
                act_bytes.data(),
                act_bytes.size(),
                ref_bytes.data(),
                ref_bytes.size(),
                m,
                k,
                n,
                repeat,
                out_dump_path,
                tensor_name.c_str(),
                gguf_path.string().c_str());
            free_run(run);
            return rc;
        }
#endif

        ggml_backend_load_all();
        run.backend = init_backend_by_name("HTP");
        if (run.backend == nullptr) {
            throw std::runtime_error("failed to init HTP backend");
        }
        int replay_threads = 1;
        if (const char * env = std::getenv("REPLAY_THREADS")) {
            replay_threads = std::max(1, std::atoi(env));
        }
        if (ggml_backend_dev_t dev = ggml_backend_get_device(run.backend)) {
            if (ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev)) {
                auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (fn) {
                    fn(run.backend, replay_threads);
                }
            }
        }

        ggml_init_params ggml_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_init_params weight_params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 2,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        run.weight_ctx = ggml_init(weight_params);
        run.ctx = ggml_init(ggml_params);
        run.weight = ggml_new_tensor_2d(run.weight_ctx, src_weight->type, k, n);
        run.activation = ggml_new_tensor_2d(run.ctx, GGML_TYPE_F32, k, m);
        ggml_set_name(run.weight, tensor_name.c_str());
        ggml_set_name(run.activation, "replay.activation");
        run.out = ggml_mul_mat(run.ctx, run.weight, run.activation);
        ggml_set_name(run.out, "replay.out");
        run.gf = ggml_new_graph_custom(run.ctx, 16, false);
        ggml_build_forward_expand(run.gf, run.out);
        run.weight_buf = ggml_backend_alloc_ctx_tensors(run.weight_ctx, run.backend);
        run.buf = ggml_backend_alloc_ctx_tensors(run.ctx, run.backend);
        if (run.weight_buf == nullptr || run.buf == nullptr) {
            throw std::runtime_error("alloc ctx tensors failed");
        }

        ggml_backend_tensor_set(run.weight, src_weight->data, 0, ggml_nbytes(src_weight));
        ggml_backend_tensor_set(run.activation, act_bytes.data(), 0, act_bytes.size());

        if (!ggml_backend_supports_op(run.backend, run.out)) {
            throw std::runtime_error("backend does not support replay op");
        }

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(replay_threads);
        run.threadpool = ggml_threadpool_new(&ttp);
        struct ggml_cplan cplan = ggml_graph_plan(run.gf, replay_threads, run.threadpool);
        std::vector<uint8_t> work(cplan.work_size);
        cplan.work_data = work.empty() ? nullptr : work.data();
        auto compute_once = [&]() -> ggml_status {
#if defined(__ANDROID__)
            static struct ggml_compute_params direct_params = {
                /*.ith        =*/ 0,
                /*.nth        =*/ 1,
                /*.wsize      =*/ 0,
                /*.wdata      =*/ nullptr,
                /*.threadpool =*/ nullptr,
            };
            if (htp_ops_support_op(run.out)) {
                return htp_ops_compute_op(&direct_params, run.out) == 0 ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
            }
#endif
            return ggml_graph_compute(run.gf, &cplan);
        };

        const ggml_status warm = compute_once();
        if (warm != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("warmup failed");
        }

        const int64_t t0 = ggml_time_us();
        for (int r = 0; r < repeat; ++r) {
            const ggml_status st = compute_once();
            if (st != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("compute failed");
            }
        }
        const int64_t t1 = ggml_time_us();

        std::vector<float> out((size_t) m * (size_t) n, 0.0f);
        ggml_backend_tensor_get(run.out, out.data(), 0, out.size() * sizeof(float));
        if (out_dump_path != nullptr && out_dump_path[0] != '\0') {
            std::ofstream ofs(out_dump_path, std::ios::binary);
            if (!ofs.is_open()) {
                throw std::runtime_error("failed to open out dump");
            }
            ofs.write(reinterpret_cast<const char *>(out.data()), (std::streamsize) (out.size() * sizeof(float)));
            if (!ofs) {
                throw std::runtime_error("failed to write out dump");
            }
        }
        const float * ref = reinterpret_cast<const float *>(ref_bytes.data());

        const bool print_nz = std::getenv("REPLAY_PRINT_NZ") != nullptr;
        int first_nz_r = -1;
        int first_nz_c = -1;
        float first_nz_v = 0.0f;
        float nz_eps = 1.0e-6f;
        if (const char * eps_env = std::getenv("REPLAY_NZ_EPS")) {
            nz_eps = std::strtof(eps_env, nullptr);
        }
        if (print_nz) {
            for (int rr = 0; rr < m && first_nz_r < 0; ++rr) {
                for (int cc = 0; cc < n; ++cc) {
                    const float v = out[(size_t) rr * (size_t) n + (size_t) cc];
                    if (std::fabs(v) > nz_eps) {
                        first_nz_r = rr;
                        first_nz_c = cc;
                        first_nz_v = v;
                        break;
                    }
                }
            }
        }

        int bad = 0;
        float mae = 0.0f;
        float mse = 0.0f;
        float max_abs = 0.0f;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < out.size(); ++i) {
            const float a = out[i];
            const float b = ref[i];
            const float d = a - b;
            const float ad = std::fabs(d);
            mae += ad;
            mse += d * d;
            if (ad > max_abs) max_abs = ad;
            if (ad > 1e-2f) ++bad;
            dot += (double) a * (double) b;
            na += (double) a * (double) a;
            nb += (double) b * (double) b;
        }
        mae /= (float) out.size();
        mse /= (float) out.size();
        const float rmse = std::sqrt(mse);
        const double cos = (na == 0.0 || nb == 0.0) ? 0.0 : dot / std::sqrt(na * nb);
        const double us_per_run = (double) (t1 - t0) / (double) repeat;

        std::printf("replay/gguf tensor=%s wtype=%s m=%d k=%d n=%d repeat=%d bad=%d/%zu mae=%g rmse=%g max_abs=%g cos=%.9f us_per_run=%g first=%g/%g gguf=%s\n",
                    tensor_name.c_str(), ggml_type_name(src_weight->type), m, k, n, repeat, bad, out.size(), mae, rmse, max_abs, cos,
                    us_per_run, out[0], ref[0], gguf_path.string().c_str());
        if (print_nz) {
            std::printf("replay/nz first_r=%d first_c=%d first_v=%g eps=%g\n",
                        first_nz_r, first_nz_c, first_nz_v, nz_eps);
        }

        free_run(run);
        return 0;
    } catch (const std::exception & err) {
        std::fprintf(stderr, "wf8_gguf_replay_check failed: %s\n", err.what());
        free_run(run);
        return 3;
    }
}
