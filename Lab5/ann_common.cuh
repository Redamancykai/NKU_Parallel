constexpr int kTile = 16;
constexpr int kMaxResultK = 64;
constexpr int kMaxTopK = 256;
constexpr int kTopKThreads = 128;
constexpr int kPQTopKThreads = 64;
constexpr int kPQLocalTopK = 16;
constexpr float kDeviceInf = 3.402823466e+38F;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            throw std::runtime_error(std::string("CUDA error at ") +           \
                                     __FILE__ + ":" + std::to_string(__LINE__) +\
                                     ": " + cudaGetErrorString(err__));        \
        }                                                                      \
    } while (0)

template <typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin) {
        throw std::runtime_error("failed to open data file: " + data_path);
    }

    uint32_t n32 = 0;
    uint32_t d32 = 0;
    fin.read(reinterpret_cast<char*>(&n32), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&d32), sizeof(uint32_t));
    if (!fin) {
        throw std::runtime_error("failed to read fbin/bin header: " + data_path);
    }

    n = n32;
    d = d32;
    T* data = new T[n * d];
    fin.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(n * d * sizeof(T)));
    if (!fin) {
        delete[] data;
        throw std::runtime_error("failed to read data payload: " + data_path);
    }

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number: " << n
              << "  size_per_element: " << sizeof(T) << "\n";
    return data;
}

void LoadDataHeader(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin) {
        throw std::runtime_error("failed to open data file: " + data_path);
    }
    uint32_t n32 = 0;
    uint32_t d32 = 0;
    fin.read(reinterpret_cast<char*>(&n32), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&d32), sizeof(uint32_t));
    if (!fin) {
        throw std::runtime_error("failed to read fbin/bin header: " + data_path);
    }
    n = n32;
    d = d32;
    std::cerr << "load header " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number: " << n << "\n";
}

std::string JoinPath(const std::string& dir, const std::string& file) {
    if (dir.empty()) {
        return file;
    }
    char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + file;
    }
    return dir + "/" + file;
}

struct Options {
    std::string data_path = "/anndata";
    std::string index = "flat";
    std::string mode = "baseline";
    size_t query_count = 2000;
    size_t batch_size = 64;
    size_t dims = 0;
    size_t rerank = 0;
    size_t k = 10;
    size_t nlist = 128;
    size_t nprobe = 16;
    size_t train_iters = 10;
    size_t pq_m = 8;
    size_t pq_ks = 16;
    std::string group_strategy = "none";
};

void PrintUsage(const char* program) {
    std::cerr
        << "Usage: " << program << " [--data PATH] [--index flat|ivf|ivf-group|ivf-fused|ivf-union|ivfpq] "
        << "[--mode baseline|topk|fp16|int8] [--queries N] [--batch N] "
        << "[--dims N] [--rerank P] [--k N] [--nlist N] [--nprobe N] "
        << "[--pq-m M] [--pq-ks Ks] [--group-strategy none|primary|signature]\n"
        << "\n"
        << "Example:\n"
        << "  " << program << " --data /anndata --mode baseline --queries 2000 --batch 64 --k 10\n";
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--data") {
            options.data_path = need_value(arg);
        } else if (arg == "--index") {
            options.index = need_value(arg);
        } else if (arg == "--mode") {
            options.mode = need_value(arg);
        } else if (arg == "--queries") {
            options.query_count = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--batch") {
            options.batch_size = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--dims") {
            options.dims = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--rerank") {
            options.rerank = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--k") {
            options.k = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--nlist") {
            options.nlist = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--nprobe") {
            options.nprobe = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--train-iters") {
            options.train_iters = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--pq-m") {
            options.pq_m = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--pq-ks") {
            options.pq_ks = static_cast<size_t>(std::stoull(need_value(arg)));
        } else if (arg == "--group-strategy") {
            options.group_strategy = need_value(arg);
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (options.batch_size == 0) {
        throw std::runtime_error("--batch must be positive");
    }
    if (options.k == 0 || options.k > kMaxResultK) {
        throw std::runtime_error("--k must be in [1, 64]");
    }
    if (options.rerank != 0 && (options.rerank < options.k || options.rerank > kMaxTopK)) {
        throw std::runtime_error("--rerank must be 0 or in [k, 256]");
    }
    if (options.mode != "baseline" && options.mode != "topk" &&
        options.mode != "fp16" && options.mode != "int8") {
        throw std::runtime_error("--mode must be one of: baseline, topk, fp16, int8");
    }
    if (options.index != "flat" && options.index != "ivf" &&
        options.index != "ivf-group" && options.index != "ivf-fused" &&
        options.index != "ivf-union" &&
        options.index != "ivfpq") {
        throw std::runtime_error("--index must be one of: flat, ivf, ivf-group, ivf-fused, ivf-union, ivfpq");
    }
    if (options.nlist == 0 || options.nprobe == 0) {
        throw std::runtime_error("--nlist and --nprobe must be positive");
    }
    if (options.pq_m == 0 || options.pq_ks == 0 || options.pq_ks > 256) {
        throw std::runtime_error("--pq-m must be positive and --pq-ks must be in [1, 256]");
    }
    if (options.group_strategy != "none" &&
        options.group_strategy != "primary" &&
        options.group_strategy != "signature") {
        throw std::runtime_error("--group-strategy must be one of: none, primary, signature");
    }
    return options;
}
