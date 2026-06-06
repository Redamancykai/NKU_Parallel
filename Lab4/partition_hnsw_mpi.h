#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include "mpi_hnsw_common.h"

// ================================================================
// Partition-HNSW + MPI
// ================================================================

#ifndef PART_HNSW_MPI_INVALID_ID
#define PART_HNSW_MPI_INVALID_ID 0xffffffffu
#endif

// 分片方式
enum PartitionHNSWPartitionStrategy {
    PART_HNSW_PARTITION_CONTIGUOUS = 0,        // 连续划分：0..N/P, N/P..2N/P, ...
    PART_HNSW_PARTITION_ROUND_ROBIN = 1,       // 轮转划分：id % P
    PART_HNSW_PARTITION_RANDOM_BALANCED = 2,   // 按 hash 打乱后均衡切分，推荐主实验使用
    PART_HNSW_PARTITION_NORM_SORTED = 3        // 按向量 L2 范数排序后均衡切分，启发式划分
};

enum PartitionHNSWSearchMode {
    PART_HNSW_SEARCH_HIERARCHICAL = 0,     // hnswlib 标准 searchKnn
    PART_HNSW_SEARCH_LAYER0 = 1,           // 单入口 layer0 搜索
    PART_HNSW_SEARCH_LAYER0_OPENMP = 2,    // 多入口 OpenMP 搜索
    PART_HNSW_SEARCH_LAYER0_PTHREAD = 3    // 多入口 Pthread 搜索
};

struct PartitionHNSWMPIIndex {
    const float* base = nullptr;
    size_t base_number = 0;
    size_t vecdim = 0;

    size_t part_num = 0;
    size_t hnsw_M = 16;
    size_t ef_construction = 150;
    size_t random_seed = 100;
    PartitionHNSWPartitionStrategy partition_strategy =
        PART_HNSW_PARTITION_RANDOM_BALANCED;

    bool trained = false;

    // partitions[p][i] 是第 p 个分片中第 i 个向量的全局 base id。
    std::vector<std::vector<uint32_t> > partitions;

    std::vector<HNSWIndex*> part_hnsw;

    PartitionHNSWMPIIndex() {}

    ~PartitionHNSWMPIIndex() {
        clear();
    }

    PartitionHNSWMPIIndex(const PartitionHNSWMPIIndex&) = delete;
    PartitionHNSWMPIIndex& operator=(const PartitionHNSWMPIIndex&) = delete;

    void clear() {
        for (size_t i = 0; i < part_hnsw.size(); ++i) {
            delete part_hnsw[i];
        }
        part_hnsw.clear();
        partitions.clear();
        trained = false;
        base = nullptr;
        base_number = 0;
        vecdim = 0;
        part_num = 0;
    }
};

static inline bool part_hnsw_mkdir_if_needed(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }

    if (mkdir(dir.c_str(), 0777) == 0) {
        return true;
    }

    return errno == EEXIST;
}

static inline uint64_t part_hnsw_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline float part_hnsw_l2_norm2(
    const float* x,
    size_t dim
) {
    float s = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        s += x[d] * x[d];
    }
    return s;
}

static inline std::string part_hnsw_strategy_name(
    PartitionHNSWPartitionStrategy strategy
) {
    switch (strategy) {
        case PART_HNSW_PARTITION_CONTIGUOUS:
            return "contiguous";
        case PART_HNSW_PARTITION_ROUND_ROBIN:
            return "roundrobin";
        case PART_HNSW_PARTITION_RANDOM_BALANCED:
            return "randombalanced";
        case PART_HNSW_PARTITION_NORM_SORTED:
            return "normsorted";
        default:
            return "unknown";
    }
}

static inline int part_hnsw_strategy_code(
    PartitionHNSWPartitionStrategy strategy
) {
    return static_cast<int>(strategy);
}

static inline std::string part_hnsw_default_prefix(
    size_t base_number,
    size_t vecdim,
    size_t part_num,
    size_t hnsw_M,
    size_t ef_construction,
    PartitionHNSWPartitionStrategy strategy,
    size_t random_seed,
    const std::string& dir = "files"
) {
    std::ostringstream oss;
    oss << dir
        << "/partition_hnsw_base_" << static_cast<unsigned long long>(base_number)
        << "_dim_" << static_cast<unsigned long long>(vecdim)
        << "_P_" << static_cast<unsigned long long>(part_num)
        << "_M_" << static_cast<unsigned long long>(hnsw_M)
        << "_efc_" << static_cast<unsigned long long>(ef_construction)
        << "_strategy_" << part_hnsw_strategy_name(strategy)
        << "_seed_" << static_cast<unsigned long long>(random_seed);
    return oss.str();
}

static inline std::string part_hnsw_meta_path(const std::string& prefix) {
    return prefix + ".meta";
}

static inline std::string part_hnsw_partition_path(
    const std::string& prefix,
    size_t part_id
) {
    std::ostringstream oss;
    oss << prefix << ".part_" << static_cast<unsigned long long>(part_id) << ".hnsw";
    return oss.str();
}

static inline std::string part_hnsw_bundle_path(const std::string& prefix) {
    return prefix + ".bundle";
}

static inline void part_hnsw_push_topk(
    std::priority_queue<std::pair<float, uint32_t> >& heap,
    float dis,
    uint32_t id,
    size_t k
) {
    if (k == 0 || id == PART_HNSW_MPI_INVALID_ID) {
        return;
    }

    if (heap.size() < k) {
        heap.emplace(dis, id);
    } else if (dis < heap.top().first) {
        heap.pop();
        heap.emplace(dis, id);
    }
}

static inline void part_hnsw_merge_topk(
    std::priority_queue<std::pair<float, uint32_t> >& dst,
    std::priority_queue<std::pair<float, uint32_t> >& src,
    size_t k
) {
    while (!src.empty()) {
        const std::pair<float, uint32_t> item = src.top();
        src.pop();
        part_hnsw_push_topk(dst, item.first, item.second, k);
    }
}

static inline void part_hnsw_build_partitions(
    PartitionHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t part_num,
    PartitionHNSWPartitionStrategy strategy,
    size_t random_seed
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(part_num > 0);

    index.partitions.clear();
    index.partitions.resize(part_num);

    if (strategy == PART_HNSW_PARTITION_ROUND_ROBIN) {
        for (size_t i = 0; i < base_number; ++i) {
            const size_t p = i % part_num;
            index.partitions[p].push_back(static_cast<uint32_t>(i));
        }
        return;
    }

    if (strategy == PART_HNSW_PARTITION_CONTIGUOUS) {
        for (size_t p = 0; p < part_num; ++p) {
            const size_t start = p * base_number / part_num;
            const size_t end = (p + 1) * base_number / part_num;
            index.partitions[p].reserve(end - start);
            for (size_t i = start; i < end; ++i) {
                index.partitions[p].push_back(static_cast<uint32_t>(i));
            }
        }
        return;
    }

    if (strategy == PART_HNSW_PARTITION_NORM_SORTED) {
        struct NormItem {
            float norm2;
            uint32_t id;
        };

        std::vector<NormItem> items;
        items.reserve(base_number);
        for (size_t i = 0; i < base_number; ++i) {
            NormItem item;
            item.norm2 = part_hnsw_l2_norm2(base + i * vecdim, vecdim);
            item.id = static_cast<uint32_t>(i);
            items.push_back(item);
        }

        std::sort(
            items.begin(),
            items.end(),
            [](const NormItem& a, const NormItem& b) {
                if (a.norm2 != b.norm2) {
                    return a.norm2 < b.norm2;
                }
                return a.id < b.id;
            }
        );

        for (size_t p = 0; p < part_num; ++p) {
            const size_t start = p * base_number / part_num;
            const size_t end = (p + 1) * base_number / part_num;
            index.partitions[p].reserve(end - start);
            for (size_t pos = start; pos < end; ++pos) {
                index.partitions[p].push_back(items[pos].id);
            }
        }
        return;
    }

    struct HashItem {
        uint64_t key;
        uint32_t id;
    };

    std::vector<HashItem> items;
    items.reserve(base_number);
    for (size_t i = 0; i < base_number; ++i) {
        HashItem item;
        item.key = part_hnsw_mix64(
            static_cast<uint64_t>(i) ^
            (static_cast<uint64_t>(random_seed) + 0x9e3779b97f4a7c15ULL)
        );
        item.id = static_cast<uint32_t>(i);
        items.push_back(item);
    }

    std::sort(
        items.begin(),
        items.end(),
        [](const HashItem& a, const HashItem& b) {
            if (a.key != b.key) {
                return a.key < b.key;
            }
            return a.id < b.id;
        }
    );

    for (size_t p = 0; p < part_num; ++p) {
        const size_t start = p * base_number / part_num;
        const size_t end = (p + 1) * base_number / part_num;
        index.partitions[p].reserve(end - start);
        for (size_t pos = start; pos < end; ++pos) {
            index.partitions[p].push_back(items[pos].id);
        }
    }
}

static inline bool part_hnsw_build_one_partition(
    PartitionHNSWMPIIndex& index,
    size_t part_id
) {
    assert(index.base != nullptr);
    assert(part_id < index.partitions.size());

    const std::vector<uint32_t>& ids = index.partitions[part_id];
    if (ids.empty()) {
        index.part_hnsw[part_id] = nullptr;
        return true;
    }

    HNSWIndex* sub = new HNSWIndex();
    try {
        if (!mpi_hnsw_build_from_global_ids(
                *sub,
                index.base,
                index.vecdim,
                ids,
                index.hnsw_M,
                index.ef_construction,
                index.random_seed + part_id
            )) {
            throw std::runtime_error("common HNSW builder returned false");
        }
        index.part_hnsw[part_id] = sub;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Partition-HNSW] Build part " << part_id
                  << " failed: " << e.what() << std::endl;
        delete sub;
        index.part_hnsw[part_id] = nullptr;
        return false;
    }
}

static inline bool part_hnsw_save_meta(
    const PartitionHNSWMPIIndex& index,
    const std::string& prefix
) {
    if (index.partitions.empty()) {
        return false;
    }

    part_hnsw_mkdir_if_needed("files");

    std::ofstream out(part_hnsw_meta_path(prefix).c_str(), std::ios::binary);
    if (!out) {
        std::cerr << "[Partition-HNSW] Cannot open meta for save: "
                  << part_hnsw_meta_path(prefix) << std::endl;
        return false;
    }

    const uint64_t magic = 0x50484e53574d5049ULL; // "PHNSWMPI"
    const uint64_t version = 1;
    const uint64_t base_number = static_cast<uint64_t>(index.base_number);
    const uint64_t vecdim = static_cast<uint64_t>(index.vecdim);
    const uint64_t part_num = static_cast<uint64_t>(index.part_num);
    const uint64_t hnsw_M = static_cast<uint64_t>(index.hnsw_M);
    const uint64_t ef_construction = static_cast<uint64_t>(index.ef_construction);
    const uint64_t random_seed = static_cast<uint64_t>(index.random_seed);
    const uint64_t strategy = static_cast<uint64_t>(
        part_hnsw_strategy_code(index.partition_strategy)
    );

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&base_number), sizeof(base_number));
    out.write(reinterpret_cast<const char*>(&vecdim), sizeof(vecdim));
    out.write(reinterpret_cast<const char*>(&part_num), sizeof(part_num));
    out.write(reinterpret_cast<const char*>(&hnsw_M), sizeof(hnsw_M));
    out.write(reinterpret_cast<const char*>(&ef_construction), sizeof(ef_construction));
    out.write(reinterpret_cast<const char*>(&random_seed), sizeof(random_seed));
    out.write(reinterpret_cast<const char*>(&strategy), sizeof(strategy));

    for (size_t p = 0; p < index.part_num; ++p) {
        const uint64_t sz = static_cast<uint64_t>(index.partitions[p].size());
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        if (sz > 0) {
            out.write(
                reinterpret_cast<const char*>(index.partitions[p].data()),
                sizeof(uint32_t) * static_cast<size_t>(sz)
            );
        }
    }

    return static_cast<bool>(out);
}

static inline bool part_hnsw_load_one_partition(
    PartitionHNSWMPIIndex& index,
    const std::string& prefix,
    size_t part_id
) {
    assert(part_id < index.partitions.size());

    const std::vector<uint32_t>& ids = index.partitions[part_id];
    if (ids.empty()) {
        index.part_hnsw[part_id] = nullptr;
        return true;
    }

    HNSWIndex* sub = new HNSWIndex();
    const std::string path = part_hnsw_partition_path(prefix, part_id);

    if (!sub->load_index(path, ids.size(), index.vecdim)) {
        delete sub;
        index.part_hnsw[part_id] = nullptr;
        return false;
    }

    index.part_hnsw[part_id] = sub;
    return true;
}

static inline bool part_hnsw_save_one_partition(
    const PartitionHNSWMPIIndex& index,
    const std::string& prefix,
    size_t part_id
) {
    assert(part_id < index.partitions.size());

    if (index.partitions[part_id].empty()) {
        return true;
    }

    if (index.part_hnsw[part_id] == nullptr ||
        !index.part_hnsw[part_id]->trained) {
        return false;
    }

    const std::string path = part_hnsw_partition_path(prefix, part_id);
    return index.part_hnsw[part_id]->save_index(path);
}

static inline std::vector<int> part_hnsw_assign_partition_owner_greedy(
    const PartitionHNSWMPIIndex& index,
    int world_size
) {
    assert(world_size > 0);

    std::vector<int> owner(index.part_num, 0);

    struct PartItem {
        size_t part_size;
        size_t part_id;
    };

    std::vector<PartItem> items;
    items.reserve(index.part_num);
    for (size_t p = 0; p < index.part_num; ++p) {
        PartItem item;
        item.part_size = index.partitions[p].size();
        item.part_id = p;
        items.push_back(item);
    }

    std::sort(
        items.begin(),
        items.end(),
        [](const PartItem& a, const PartItem& b) {
            if (a.part_size != b.part_size) {
                return a.part_size > b.part_size;
            }
            return a.part_id < b.part_id;
        }
    );

    std::vector<size_t> load(static_cast<size_t>(world_size), 0);

    for (size_t i = 0; i < items.size(); ++i) {
        int best_rank = 0;
        size_t best_load = load[0];

        for (int r = 1; r < world_size; ++r) {
            if (load[static_cast<size_t>(r)] < best_load) {
                best_load = load[static_cast<size_t>(r)];
                best_rank = r;
            }
        }

        owner[items[i].part_id] = best_rank;
        load[static_cast<size_t>(best_rank)] += items[i].part_size;
    }

    return owner;
}

static inline bool part_hnsw_init_empty_index(
    PartitionHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t part_num,
    size_t hnsw_M,
    size_t ef_construction,
    PartitionHNSWPartitionStrategy strategy,
    size_t random_seed
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(part_num > 0);

    index.clear();
    index.base = base;
    index.base_number = base_number;
    index.vecdim = vecdim;
    index.part_num = std::min(part_num, base_number);
    index.hnsw_M = hnsw_M;
    index.ef_construction = ef_construction;
    index.random_seed = random_seed;
    index.partition_strategy = strategy;

    part_hnsw_build_partitions(
        index,
        base,
        base_number,
        vecdim,
        index.part_num,
        strategy,
        random_seed
    );

    index.part_hnsw.assign(index.part_num, static_cast<HNSWIndex*>(nullptr));
    return true;
}

static inline bool part_hnsw_build_all_partitions(
    PartitionHNSWMPIIndex& index
) {
    assert(index.base != nullptr);
    assert(index.part_hnsw.size() == index.part_num);

    for (size_t p = 0; p < index.part_num; ++p) {
        if (!part_hnsw_build_one_partition(index, p)) {
            index.trained = false;
            return false;
        }
        std::cerr << "[Partition-HNSW] Built part " << (p + 1)
                  << " / " << index.part_num
                  << ", size=" << index.partitions[p].size()
                  << std::endl;
    }

    index.trained = true;
    return true;
}

static inline bool part_hnsw_build_assigned_partitions(
    PartitionHNSWMPIIndex& index,
    const std::vector<int>& owner,
    int rank
) {
    assert(index.part_hnsw.size() == index.part_num);
    assert(owner.size() == index.part_num);

    for (size_t p = 0; p < index.part_num; ++p) {
        if (owner[p] != rank) {
            continue;
        }

        if (!part_hnsw_build_one_partition(index, p)) {
            index.trained = false;
            return false;
        }

        std::cerr << "[Partition-HNSW][MPI] Rank " << rank
                  << " built part " << p
                  << ", size=" << index.partitions[p].size()
                  << std::endl;
    }

    index.trained = true;
    return true;
}

static inline bool part_hnsw_load_all_partitions(
    PartitionHNSWMPIIndex& index,
    const std::string& prefix
) {
    assert(index.part_hnsw.size() == index.part_num);

    const std::string bundle_path = part_hnsw_bundle_path(prefix);
    std::ifstream in(bundle_path.c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t base_number = 0;
    uint64_t vecdim = 0;
    uint64_t part_num = 0;
    uint64_t hnsw_M = 0;
    uint64_t ef_construction = 0;
    uint64_t random_seed = 0;
    uint64_t strategy = 0;

    mpi_hnsw_read_u64(in, magic);
    mpi_hnsw_read_u64(in, version);
    mpi_hnsw_read_u64(in, base_number);
    mpi_hnsw_read_u64(in, vecdim);
    mpi_hnsw_read_u64(in, part_num);
    mpi_hnsw_read_u64(in, hnsw_M);
    mpi_hnsw_read_u64(in, ef_construction);
    mpi_hnsw_read_u64(in, random_seed);
    mpi_hnsw_read_u64(in, strategy);

    const uint64_t expected_magic = 0x50484e535742554eULL; // "PHNSWBUN"
    if (!in ||
        magic != expected_magic ||
        version != 1 ||
        base_number != static_cast<uint64_t>(index.base_number) ||
        vecdim != static_cast<uint64_t>(index.vecdim) ||
        part_num != static_cast<uint64_t>(index.part_num) ||
        hnsw_M != static_cast<uint64_t>(index.hnsw_M) ||
        ef_construction != static_cast<uint64_t>(index.ef_construction) ||
        random_seed != static_cast<uint64_t>(index.random_seed) ||
        strategy != static_cast<uint64_t>(part_hnsw_strategy_code(index.partition_strategy))) {
        return false;
    }

    std::vector<std::vector<uint32_t> > loaded_partitions(index.part_num);
    for (size_t p = 0; p < index.part_num; ++p) {
        uint64_t sz = 0;
        mpi_hnsw_read_u64(in, sz);
        if (!in) {
            return false;
        }
        loaded_partitions[p].resize(static_cast<size_t>(sz));
        if (sz > 0) {
            in.read(
                reinterpret_cast<char*>(loaded_partitions[p].data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * static_cast<size_t>(sz))
            );
        }
    }

    if (!in || loaded_partitions != index.partitions) {
        return false;
    }

    for (size_t p = 0; p < index.part_num; ++p) {
        if (index.partitions[p].empty()) {
            continue;
        }

        uint64_t pid = 0;
        uint64_t blob_size = 0;
        mpi_hnsw_read_u64(in, pid);
        mpi_hnsw_read_u64(in, blob_size);
        if (!in || pid != static_cast<uint64_t>(p) || blob_size == 0) {
            index.trained = false;
            return false;
        }

        std::vector<char> blob(static_cast<size_t>(blob_size));
        in.read(blob.data(), static_cast<std::streamsize>(blob.size()));
        if (!in) {
            index.trained = false;
            return false;
        }

        const std::string tmp_path = mpi_hnsw_tmp_path(prefix, "part", 0, p);
        if (!mpi_hnsw_write_file_blob(tmp_path, blob)) {
            index.trained = false;
            return false;
        }

        HNSWIndex* sub = new HNSWIndex();
        if (!sub->load_index(tmp_path, index.partitions[p].size(), index.vecdim)) {
            mpi_hnsw_remove_file(tmp_path);
            delete sub;
            index.trained = false;
            return false;
        }
        mpi_hnsw_remove_file(tmp_path);
        index.part_hnsw[p] = sub;
    }

    index.trained = true;
    std::cerr << "[Partition-HNSW] Loaded bundle: " << bundle_path << std::endl;
    return true;
}

static inline bool part_hnsw_load_assigned_partitions(
    PartitionHNSWMPIIndex& index,
    const std::string& prefix,
    const std::vector<int>& owner,
    int rank
) {
    assert(index.part_hnsw.size() == index.part_num);
    assert(owner.size() == index.part_num);

    const std::string bundle_path = part_hnsw_bundle_path(prefix);
    std::ifstream in(bundle_path.c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t base_number = 0;
    uint64_t vecdim = 0;
    uint64_t part_num = 0;
    uint64_t hnsw_M = 0;
    uint64_t ef_construction = 0;
    uint64_t random_seed = 0;
    uint64_t strategy = 0;

    mpi_hnsw_read_u64(in, magic);
    mpi_hnsw_read_u64(in, version);
    mpi_hnsw_read_u64(in, base_number);
    mpi_hnsw_read_u64(in, vecdim);
    mpi_hnsw_read_u64(in, part_num);
    mpi_hnsw_read_u64(in, hnsw_M);
    mpi_hnsw_read_u64(in, ef_construction);
    mpi_hnsw_read_u64(in, random_seed);
    mpi_hnsw_read_u64(in, strategy);

    const uint64_t expected_magic = 0x50484e535742554eULL;
    if (!in ||
        magic != expected_magic ||
        version != 1 ||
        base_number != static_cast<uint64_t>(index.base_number) ||
        vecdim != static_cast<uint64_t>(index.vecdim) ||
        part_num != static_cast<uint64_t>(index.part_num) ||
        hnsw_M != static_cast<uint64_t>(index.hnsw_M) ||
        ef_construction != static_cast<uint64_t>(index.ef_construction) ||
        random_seed != static_cast<uint64_t>(index.random_seed) ||
        strategy != static_cast<uint64_t>(part_hnsw_strategy_code(index.partition_strategy))) {
        return false;
    }

    std::vector<std::vector<uint32_t> > loaded_partitions(index.part_num);
    for (size_t p = 0; p < index.part_num; ++p) {
        uint64_t sz = 0;
        mpi_hnsw_read_u64(in, sz);
        if (!in) {
            return false;
        }
        loaded_partitions[p].resize(static_cast<size_t>(sz));
        if (sz > 0) {
            in.read(
                reinterpret_cast<char*>(loaded_partitions[p].data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * static_cast<size_t>(sz))
            );
        }
    }

    if (!in || loaded_partitions != index.partitions) {
        return false;
    }

    for (size_t p = 0; p < index.part_num; ++p) {
        if (index.partitions[p].empty()) {
            continue;
        }

        uint64_t pid = 0;
        uint64_t blob_size = 0;
        mpi_hnsw_read_u64(in, pid);
        mpi_hnsw_read_u64(in, blob_size);
        if (!in || pid != static_cast<uint64_t>(p) || blob_size == 0) {
            index.trained = false;
            return false;
        }

        std::vector<char> blob(static_cast<size_t>(blob_size));
        in.read(blob.data(), static_cast<std::streamsize>(blob.size()));
        if (!in) {
            index.trained = false;
            return false;
        }

        if (owner[p] != rank) {
            continue;
        }

        const std::string tmp_path = mpi_hnsw_tmp_path(prefix, "part", rank, p);
        if (!mpi_hnsw_write_file_blob(tmp_path, blob)) {
            index.trained = false;
            return false;
        }

        HNSWIndex* sub = new HNSWIndex();
        if (!sub->load_index(tmp_path, index.partitions[p].size(), index.vecdim)) {
            mpi_hnsw_remove_file(tmp_path);
            delete sub;
            index.trained = false;
            return false;
        }
        mpi_hnsw_remove_file(tmp_path);
        index.part_hnsw[p] = sub;
    }

    index.trained = true;
    return true;
}

static inline bool part_hnsw_save_all_partitions(
    const PartitionHNSWMPIIndex& index,
    const std::string& prefix,
    int rank_for_tmp = 0
) {
    if (!index.trained) {
        return false;
    }

    part_hnsw_mkdir_if_needed("files");
    const std::string bundle_path = part_hnsw_bundle_path(prefix);
    std::vector<std::vector<char> > part_blobs(index.part_num);

    for (size_t p = 0; p < index.part_num; ++p) {
        if (index.partitions[p].empty()) {
            continue;
        }
        if (index.part_hnsw[p] == nullptr || !index.part_hnsw[p]->trained) {
            return false;
        }

        const std::string tmp_path = mpi_hnsw_tmp_path(
            prefix,
            "part",
            rank_for_tmp,
            p
        );
        if (!index.part_hnsw[p]->save_index(tmp_path)) {
            mpi_hnsw_remove_file(tmp_path);
            std::cerr << "[Partition-HNSW] Save temp part failed: "
                      << tmp_path << std::endl;
            return false;
        }
        if (!mpi_hnsw_read_file_blob(tmp_path, part_blobs[p])) {
            mpi_hnsw_remove_file(tmp_path);
            return false;
        }
        mpi_hnsw_remove_file(tmp_path);
    }

    std::ofstream out(bundle_path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[Partition-HNSW] Cannot open bundle for save: "
                  << bundle_path << std::endl;
        return false;
    }

    const uint64_t magic = 0x50484e535742554eULL; // "PHNSWBUN"
    const uint64_t version = 1;
    mpi_hnsw_write_u64(out, magic);
    mpi_hnsw_write_u64(out, version);
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.base_number));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.vecdim));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.part_num));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.hnsw_M));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.ef_construction));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.random_seed));
    mpi_hnsw_write_u64(
        out,
        static_cast<uint64_t>(part_hnsw_strategy_code(index.partition_strategy))
    );

    for (size_t p = 0; p < index.part_num; ++p) {
        const std::vector<uint32_t>& ids = index.partitions[p];
        mpi_hnsw_write_u64(out, static_cast<uint64_t>(ids.size()));
        if (!ids.empty()) {
            out.write(
                reinterpret_cast<const char*>(ids.data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * ids.size())
            );
        }
    }

    for (size_t p = 0; p < index.part_num; ++p) {
        if (index.partitions[p].empty()) {
            continue;
        }
        mpi_hnsw_write_u64(out, static_cast<uint64_t>(p));
        mpi_hnsw_write_u64(out, static_cast<uint64_t>(part_blobs[p].size()));
        if (!part_blobs[p].empty()) {
            out.write(
                part_blobs[p].data(),
                static_cast<std::streamsize>(part_blobs[p].size())
            );
        }
    }

    if (!out) {
        return false;
    }

    std::cerr << "[Partition-HNSW] Saved bundle: " << bundle_path << std::endl;
    return true;
}

static inline bool part_hnsw_save_assigned_partitions(
    const PartitionHNSWMPIIndex& index,
    const std::string& prefix,
    const std::vector<int>& owner,
    int rank
) {
    if (!index.trained) {
        return false;
    }

    bool ok = true;
    for (size_t p = 0; p < index.part_num; ++p) {
        if (owner[p] != rank) {
            continue;
        }
        if (!part_hnsw_save_one_partition(index, prefix, p)) {
            std::cerr << "[Partition-HNSW][MPI] Rank " << rank
                      << " save part failed: "
                      << part_hnsw_partition_path(prefix, p) << std::endl;
            ok = false;
        }
    }
    return ok;
}

static inline bool part_hnsw_build_or_load_index(
    PartitionHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t part_num = 4,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    PartitionHNSWPartitionStrategy strategy = PART_HNSW_PARTITION_RANDOM_BALANCED,
    const std::string& prefix = "",
    size_t random_seed = 100
) {
    const std::string real_prefix = prefix.empty()
        ? part_hnsw_default_prefix(
              base_number,
              vecdim,
              part_num,
              hnsw_M,
              ef_construction,
              strategy,
              random_seed
          )
        : prefix;

    part_hnsw_init_empty_index(
        index,
        base,
        base_number,
        vecdim,
        part_num,
        hnsw_M,
        ef_construction,
        strategy,
        random_seed
    );

    if (part_hnsw_load_all_partitions(index, real_prefix)) {
        return true;
    }

    std::cerr << "[Partition-HNSW] No valid saved index, build new one" << std::endl;

    if (!part_hnsw_build_all_partitions(index)) {
        return false;
    }

    part_hnsw_save_all_partitions(index, real_prefix);
    return true;
}

static inline bool part_hnsw_mpi_build_or_load_index(
    PartitionHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t part_num = 4,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    PartitionHNSWPartitionStrategy strategy = PART_HNSW_PARTITION_RANDOM_BALANCED,
    const std::string& prefix = "",
    size_t random_seed = 100,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const std::string real_prefix = prefix.empty()
        ? part_hnsw_default_prefix(
              base_number,
              vecdim,
              part_num,
              hnsw_M,
              ef_construction,
              strategy,
              random_seed
          )
        : prefix;

    part_hnsw_init_empty_index(
        index,
        base,
        base_number,
        vecdim,
        part_num,
        hnsw_M,
        ef_construction,
        strategy,
        random_seed
    );

    const std::vector<int> owner =
        part_hnsw_assign_partition_owner_greedy(index, world_size);

    bool loaded = part_hnsw_load_assigned_partitions(
        index,
        real_prefix,
        owner,
        rank
    );

    const int local_loaded = loaded ? 1 : 0;
    int all_loaded = 0;
    MPI_Allreduce(&local_loaded, &all_loaded, 1, MPI_INT, MPI_MIN, comm);

    if (all_loaded) {
        if (rank == 0) {
            std::cerr << "[Partition-HNSW][MPI] Loaded saved index prefix: "
                      << real_prefix << std::endl;
        }
        return true;
    }

    // 单 bundle 模式下，首次缺索引时由 rank 0 构建完整 bundle；
    // 其他 rank 等待后再从 bundle 中只加载自己负责的分片。
    if (rank == 0) {
        std::cerr << "[Partition-HNSW][MPI] Bundle missing; rank 0 builds full bundle"
                  << std::endl;
        for (size_t p = 0; p < index.part_hnsw.size(); ++p) {
            if (index.part_hnsw[p] != nullptr) {
                delete index.part_hnsw[p];
                index.part_hnsw[p] = nullptr;
            }
        }
        if (part_hnsw_build_all_partitions(index)) {
            part_hnsw_save_all_partitions(index, real_prefix, rank);
        }
    }

    MPI_Barrier(comm);

    for (size_t p = 0; p < index.part_hnsw.size(); ++p) {
        if (index.part_hnsw[p] != nullptr) {
            delete index.part_hnsw[p];
            index.part_hnsw[p] = nullptr;
        }
    }
    index.trained = false;

    loaded = part_hnsw_load_assigned_partitions(index, real_prefix, owner, rank);

    const int local_ok = loaded ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    return all_ok != 0;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
part_hnsw_search_one_partition(
    const PartitionHNSWMPIIndex& index,
    size_t part_id,
    const float* query,
    size_t k,
    size_t ef,
    PartitionHNSWSearchMode mode,
    int entry_count,
    int thread_num
) {
    std::priority_queue<std::pair<float, uint32_t> > empty;

    if (part_id >= index.part_num) {
        return empty;
    }

    const size_t part_size = index.partitions[part_id].size();
    if (part_size == 0 || index.part_hnsw[part_id] == nullptr) {
        return empty;
    }

    const size_t real_k = std::min(k, part_size);
    const size_t real_ef = std::max(real_k, ef);
    if (real_k == 0) {
        return empty;
    }

    const HNSWIndex& sub = *(index.part_hnsw[part_id]);

    if (mode == PART_HNSW_SEARCH_LAYER0) {
        return hnsw_search_layer0(sub, query, real_k, real_ef);
    }

    if (mode == PART_HNSW_SEARCH_LAYER0_OPENMP) {
        return hnsw_search_layer0_multi_entry_openmp(
            sub,
            query,
            real_k,
            real_ef,
            entry_count,
            thread_num
        );
    }

    if (mode == PART_HNSW_SEARCH_LAYER0_PTHREAD) {
        return hnsw_search_layer0_multi_entry_pthread(
            sub,
            query,
            real_k,
            real_ef,
            entry_count,
            thread_num
        );
    }

    return hnsw_search_hierarchical(sub, query, real_k, real_ef);
}

static inline std::priority_queue<std::pair<float, uint32_t> >
part_hnsw_search_local_assigned_partitions(
    const PartitionHNSWMPIIndex& index,
    const float* query,
    size_t k,
    const std::vector<int>& owner,
    int rank,
    size_t ef,
    PartitionHNSWSearchMode mode,
    int entry_count,
    int thread_num
) {
    std::priority_queue<std::pair<float, uint32_t> > local_result;

    for (size_t p = 0; p < index.part_num; ++p) {
        if (owner[p] != rank) {
            continue;
        }

        std::priority_queue<std::pair<float, uint32_t> > part_result =
            part_hnsw_search_one_partition(
                index,
                p,
                query,
                k,
                ef,
                mode,
                entry_count,
                thread_num
            );

        part_hnsw_merge_topk(local_result, part_result, k);
    }

    return local_result;
}

static inline void part_hnsw_heap_to_fixed_arrays(
    std::priority_queue<std::pair<float, uint32_t> > heap,
    size_t k,
    std::vector<float>& dist,
    std::vector<uint32_t>& ids
) {
    mpi_hnsw_heap_to_fixed_arrays(heap, k, PART_HNSW_MPI_INVALID_ID, dist, ids);
}

static inline std::priority_queue<std::pair<float, uint32_t> >
part_hnsw_mpi_merge_all_local_results(
    std::priority_queue<std::pair<float, uint32_t> > local_result,
    size_t k,
    MPI_Comm comm
) {
    return mpi_hnsw_merge_all_local_results(
        local_result,
        k,
        comm,
        PART_HNSW_MPI_INVALID_ID
    );
}

static inline std::priority_queue<std::pair<float, uint32_t> >
part_hnsw_search_mpi(
    const PartitionHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64,
    PartitionHNSWSearchMode mode = PART_HNSW_SEARCH_HIERARCHICAL,
    int entry_count = 4,
    int thread_num = 1,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const std::vector<int> owner =
        part_hnsw_assign_partition_owner_greedy(index, world_size);

    std::priority_queue<std::pair<float, uint32_t> > local_result =
        part_hnsw_search_local_assigned_partitions(
            index,
            query,
            k,
            owner,
            rank,
            ef,
            mode,
            entry_count,
            thread_num
        );

    return part_hnsw_mpi_merge_all_local_results(local_result, k, comm);
}

static inline std::priority_queue<std::pair<float, uint32_t> >
part_hnsw_search_single_process(
    const PartitionHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64,
    PartitionHNSWSearchMode mode = PART_HNSW_SEARCH_HIERARCHICAL,
    int entry_count = 4,
    int thread_num = 1
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);

    std::vector<int> owner(index.part_num, 0);
    return part_hnsw_search_local_assigned_partitions(
        index,
        query,
        k,
        owner,
        0,
        ef,
        mode,
        entry_count,
        thread_num
    );
}

static inline std::priority_queue<std::pair<float, uint32_t> >
partition_hnsw_search(
    const PartitionHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64,
    PartitionHNSWSearchMode mode = PART_HNSW_SEARCH_HIERARCHICAL,
    int entry_count = 4,
    int thread_num = 1
) {
    return part_hnsw_search_single_process(
        index,
        query,
        k,
        ef,
        mode,
        entry_count,
        thread_num
    );
}

// 兼容命名：MPI 正式测试版本
static inline std::priority_queue<std::pair<float, uint32_t> >
partition_hnsw_search_mpi(
    const PartitionHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64,
    PartitionHNSWSearchMode mode = PART_HNSW_SEARCH_HIERARCHICAL,
    int entry_count = 4,
    int thread_num = 1,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    return part_hnsw_search_mpi(
        index,
        query,
        k,
        ef,
        mode,
        entry_count,
        thread_num,
        comm
    );
}

static inline bool partition_hnsw_load_or_build(
    PartitionHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t part_num = 4,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    PartitionHNSWPartitionStrategy strategy = PART_HNSW_PARTITION_RANDOM_BALANCED,
    const std::string& prefix = "",
    size_t random_seed = 100
) {
    return part_hnsw_build_or_load_index(
        index,
        base,
        base_number,
        vecdim,
        part_num,
        hnsw_M,
        ef_construction,
        strategy,
        prefix,
        random_seed
    );
}

static inline bool partition_hnsw_mpi_load_or_build(
    PartitionHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t part_num = 4,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    PartitionHNSWPartitionStrategy strategy = PART_HNSW_PARTITION_RANDOM_BALANCED,
    const std::string& prefix = "",
    size_t random_seed = 100,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    return part_hnsw_mpi_build_or_load_index(
        index,
        base,
        base_number,
        vecdim,
        part_num,
        hnsw_M,
        ef_construction,
        strategy,
        prefix,
        random_seed,
        comm
    );
}
