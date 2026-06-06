#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cfloat>
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
// IVF + HNSW + MPI
// ================================================================

#ifndef IVFHNSW_MPI_INVALID_ID
#define IVFHNSW_MPI_INVALID_ID 0xffffffffu
#endif

enum IVFHNSWSearchMode {
    IVFHNSW_SEARCH_HIERARCHICAL = 0,      // hnswlib 标准 searchKnn
    IVFHNSW_SEARCH_LAYER0 = 1,            // 单入口 layer0 搜索
    IVFHNSW_SEARCH_LAYER0_OPENMP = 2,     // 簇内多入口 OpenMP 搜索
    IVFHNSW_SEARCH_LAYER0_PTHREAD = 3     // 簇内多入口 Pthread 搜索
};

struct IVFHNSWMPIIndex {
    const float* base = nullptr;
    size_t base_number = 0;
    size_t vecdim = 0;

    size_t nlist = 0;
    size_t ivf_train_iters = 10;

    size_t hnsw_M = 16;
    size_t ef_construction = 150;
    size_t random_seed = 100;

    bool trained = false;

    // centroids[c * vecdim + d]
    std::vector<float> centroids;

    // inverted_lists[c][j] 是第 c 个 IVF 簇内第 j 个向量的全局 base id
    std::vector<std::vector<uint32_t> > inverted_lists;

    // 每个 IVF 簇一个 HNSW 子索引。空簇对应 nullptr。
    std::vector<HNSWIndex*> cluster_hnsw;

    IVFHNSWMPIIndex() {}

    ~IVFHNSWMPIIndex() {
        clear();
    }

    IVFHNSWMPIIndex(const IVFHNSWMPIIndex&) = delete;
    IVFHNSWMPIIndex& operator=(const IVFHNSWMPIIndex&) = delete;

    void clear() {
        for (size_t i = 0; i < cluster_hnsw.size(); ++i) {
            delete cluster_hnsw[i];
        }
        cluster_hnsw.clear();
        centroids.clear();
        inverted_lists.clear();
        trained = false;
        base = nullptr;
        base_number = 0;
        vecdim = 0;
    }
};

static inline bool ivfhnsw_mkdir_if_needed(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }

    if (mkdir(dir.c_str(), 0777) == 0) {
        return true;
    }

    return errno == EEXIST;
}

static inline float ivfhnsw_inner_product_scalar(
    const float* a,
    const float* b,
    size_t dim
) {
    return mpi_hnsw_inner_product_scalar(a, b, dim);
}

static inline float ivfhnsw_ip_distance(
    const float* a,
    const float* b,
    size_t dim
) {
    return mpi_hnsw_ip_distance(a, b, dim);
}

static inline std::string ivfhnsw_to_string_u64(size_t x) {
    std::ostringstream oss;
    oss << static_cast<unsigned long long>(x);
    return oss.str();
}

static inline std::string ivfhnsw_default_prefix(
    size_t base_number,
    size_t vecdim,
    size_t nlist,
    size_t hnsw_M,
    size_t ef_construction,
    size_t ivf_train_iters,
    const std::string& dir = "files"
) {
    std::ostringstream oss;
    oss << dir
        << "/ivf_hnsw_base_" << static_cast<unsigned long long>(base_number)
        << "_dim_" << static_cast<unsigned long long>(vecdim)
        << "_nlist_" << static_cast<unsigned long long>(nlist)
        << "_M_" << static_cast<unsigned long long>(hnsw_M)
        << "_efc_" << static_cast<unsigned long long>(ef_construction)
        << "_ivfiter_" << static_cast<unsigned long long>(ivf_train_iters);
    return oss.str();
}

static inline std::string ivfhnsw_meta_path(const std::string& prefix) {
    return prefix + ".meta";
}

static inline std::string ivfhnsw_cluster_path(
    const std::string& prefix,
    size_t cid
) {
    std::ostringstream oss;
    oss << prefix << ".cluster_" << static_cast<unsigned long long>(cid) << ".hnsw";
    return oss.str();
}

static inline std::string ivfhnsw_bundle_path(const std::string& prefix) {
    return prefix + ".bundle";
}

static inline void ivfhnsw_push_topk(
    std::priority_queue<std::pair<float, uint32_t> >& heap,
    float dis,
    uint32_t id,
    size_t k
) {
    if (k == 0 || id == IVFHNSW_MPI_INVALID_ID) {
        return;
    }

    if (heap.size() < k) {
        heap.emplace(dis, id);
    } else if (dis < heap.top().first) {
        heap.pop();
        heap.emplace(dis, id);
    }
}

static inline void ivfhnsw_merge_topk(
    std::priority_queue<std::pair<float, uint32_t> >& dst,
    std::priority_queue<std::pair<float, uint32_t> >& src,
    size_t k
) {
    while (!src.empty()) {
        const std::pair<float, uint32_t> item = src.top();
        src.pop();
        ivfhnsw_push_topk(dst, item.first, item.second, k);
    }
}

static inline void ivfhnsw_train_ivf_centroids(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist,
    size_t train_iters
) {
    mpi_hnsw_train_ivf_centroids(
        index.centroids,
        base,
        base_number,
        vecdim,
        nlist,
        train_iters
    );
}

static inline uint32_t ivfhnsw_find_nearest_centroid(
    const IVFHNSWMPIIndex& index,
    const float* query
) {
    return mpi_hnsw_find_nearest_centroid(
        index.centroids,
        index.nlist,
        index.vecdim,
        query
    );
}

static inline void ivfhnsw_assign_inverted_lists(
    IVFHNSWMPIIndex& index
) {
    assert(index.base != nullptr);
    assert(index.base_number > 0);
    assert(index.vecdim > 0);
    assert(index.nlist > 0);

    mpi_hnsw_assign_ivf_lists(
        index.inverted_lists,
        index.centroids,
        index.base,
        index.base_number,
        index.vecdim,
        index.nlist
    );
}

static inline bool ivfhnsw_build_cluster_hnsw(
    IVFHNSWMPIIndex& index,
    size_t cid
) {
    assert(index.base != nullptr);
    assert(cid < index.inverted_lists.size());

    const std::vector<uint32_t>& ids = index.inverted_lists[cid];
    if (ids.empty()) {
        index.cluster_hnsw[cid] = nullptr;
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
                index.random_seed + cid
            )) {
            throw std::runtime_error("common HNSW builder returned false");
        }
        index.cluster_hnsw[cid] = sub;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[IVF-HNSW] Build cluster " << cid
                  << " failed: " << e.what() << std::endl;
        delete sub;
        index.cluster_hnsw[cid] = nullptr;
        return false;
    }
}

static inline bool ivfhnsw_build_index(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist = 1024,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    size_t ivf_train_iters = 10,
    size_t random_seed = 100
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(nlist > 0);

    index.clear();

    index.base = base;
    index.base_number = base_number;
    index.vecdim = vecdim;
    index.nlist = std::min(nlist, base_number);
    index.ivf_train_iters = ivf_train_iters;
    index.hnsw_M = hnsw_M;
    index.ef_construction = ef_construction;
    index.random_seed = random_seed;

    std::cerr << "[IVF-HNSW] Train IVF centroids: nlist=" << index.nlist
              << ", iter=" << index.ivf_train_iters << std::endl;

    ivfhnsw_train_ivf_centroids(
        index,
        base,
        base_number,
        vecdim,
        index.nlist,
        index.ivf_train_iters
    );

    std::cerr << "[IVF-HNSW] Assign base vectors to IVF lists" << std::endl;
    ivfhnsw_assign_inverted_lists(index);

    index.cluster_hnsw.assign(index.nlist, static_cast<HNSWIndex*>(nullptr));

    std::cerr << "[IVF-HNSW] Build HNSW for each non-empty IVF list" << std::endl;
    for (size_t c = 0; c < index.nlist; ++c) {
        if (!ivfhnsw_build_cluster_hnsw(index, c)) {
            index.clear();
            return false;
        }

        if ((c + 1) % 128 == 0 || c + 1 == index.nlist) {
            std::cerr << "[IVF-HNSW] Built clusters " << (c + 1)
                      << " / " << index.nlist << std::endl;
        }
    }

    index.trained = true;
    return true;
}

static inline bool ivfhnsw_save_meta(
    const IVFHNSWMPIIndex& index,
    const std::string& prefix
) {
    assert(index.trained);

    ivfhnsw_mkdir_if_needed("files");

    std::ofstream out(ivfhnsw_meta_path(prefix).c_str(), std::ios::binary);
    if (!out) {
        std::cerr << "[IVF-HNSW] Cannot open meta for save: "
                  << ivfhnsw_meta_path(prefix) << std::endl;
        return false;
    }

    const uint64_t magic = 0x495646484e53574dULL; // "IVFHNSWM" shortened
    const uint64_t version = 1;

    const uint64_t base_number = static_cast<uint64_t>(index.base_number);
    const uint64_t vecdim = static_cast<uint64_t>(index.vecdim);
    const uint64_t nlist = static_cast<uint64_t>(index.nlist);
    const uint64_t ivf_train_iters = static_cast<uint64_t>(index.ivf_train_iters);
    const uint64_t hnsw_M = static_cast<uint64_t>(index.hnsw_M);
    const uint64_t ef_construction = static_cast<uint64_t>(index.ef_construction);
    const uint64_t random_seed = static_cast<uint64_t>(index.random_seed);

    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&base_number), sizeof(base_number));
    out.write(reinterpret_cast<const char*>(&vecdim), sizeof(vecdim));
    out.write(reinterpret_cast<const char*>(&nlist), sizeof(nlist));
    out.write(reinterpret_cast<const char*>(&ivf_train_iters), sizeof(ivf_train_iters));
    out.write(reinterpret_cast<const char*>(&hnsw_M), sizeof(hnsw_M));
    out.write(reinterpret_cast<const char*>(&ef_construction), sizeof(ef_construction));
    out.write(reinterpret_cast<const char*>(&random_seed), sizeof(random_seed));

    if (!index.centroids.empty()) {
        out.write(
            reinterpret_cast<const char*>(index.centroids.data()),
            sizeof(float) * index.centroids.size()
        );
    }

    for (size_t c = 0; c < index.nlist; ++c) {
        const uint64_t sz = static_cast<uint64_t>(index.inverted_lists[c].size());
        out.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        if (sz > 0) {
            out.write(
                reinterpret_cast<const char*>(index.inverted_lists[c].data()),
                sizeof(uint32_t) * static_cast<size_t>(sz)
            );
        }
    }

    return static_cast<bool>(out);
}

static inline bool ivfhnsw_save_index(
    const IVFHNSWMPIIndex& index,
    const std::string& prefix,
    int rank_for_tmp = 0
) {
    if (!index.trained) {
        return false;
    }

    ivfhnsw_mkdir_if_needed("files");
    const std::string bundle_path = ivfhnsw_bundle_path(prefix);
    std::vector<std::vector<char> > cluster_blobs(index.nlist);

    for (size_t c = 0; c < index.nlist; ++c) {
        if (index.inverted_lists[c].empty()) {
            continue;
        }

        if (index.cluster_hnsw[c] == nullptr || !index.cluster_hnsw[c]->trained) {
            return false;
        }

        const std::string tmp_path = mpi_hnsw_tmp_path(
            prefix,
            "ivf_cluster",
            rank_for_tmp,
            c
        );
        if (!index.cluster_hnsw[c]->save_index(tmp_path)) {
            mpi_hnsw_remove_file(tmp_path);
            std::cerr << "[IVF-HNSW] Save cluster failed: " << tmp_path << std::endl;
            return false;
        }

        if (!mpi_hnsw_read_file_blob(tmp_path, cluster_blobs[c])) {
            mpi_hnsw_remove_file(tmp_path);
            std::cerr << "[IVF-HNSW] Read temp cluster failed: "
                      << tmp_path << std::endl;
            return false;
        }
        mpi_hnsw_remove_file(tmp_path);
    }

    std::ofstream out(bundle_path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[IVF-HNSW] Cannot open bundle for save: "
                  << bundle_path << std::endl;
        return false;
    }

    const uint64_t magic = 0x49564648484e4255ULL; // "IVFHHNBU"
    const uint64_t version = 1;
    mpi_hnsw_write_u64(out, magic);
    mpi_hnsw_write_u64(out, version);
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.base_number));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.vecdim));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.nlist));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.ivf_train_iters));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.hnsw_M));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.ef_construction));
    mpi_hnsw_write_u64(out, static_cast<uint64_t>(index.random_seed));

    if (!index.centroids.empty()) {
        out.write(
            reinterpret_cast<const char*>(index.centroids.data()),
            static_cast<std::streamsize>(sizeof(float) * index.centroids.size())
        );
    }

    for (size_t c = 0; c < index.nlist; ++c) {
        const std::vector<uint32_t>& ids = index.inverted_lists[c];
        mpi_hnsw_write_u64(out, static_cast<uint64_t>(ids.size()));
        if (!ids.empty()) {
            out.write(
                reinterpret_cast<const char*>(ids.data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * ids.size())
            );
        }
    }

    for (size_t c = 0; c < index.nlist; ++c) {
        if (index.inverted_lists[c].empty()) {
            continue;
        }
        mpi_hnsw_write_u64(out, static_cast<uint64_t>(c));
        mpi_hnsw_write_u64(out, static_cast<uint64_t>(cluster_blobs[c].size()));
        if (!cluster_blobs[c].empty()) {
            out.write(
                cluster_blobs[c].data(),
                static_cast<std::streamsize>(cluster_blobs[c].size())
            );
        }
    }

    if (!out) {
        return false;
    }

    std::cerr << "[IVF-HNSW] Saved bundle: " << bundle_path << std::endl;
    return true;
}

static inline bool ivfhnsw_load_meta(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t expected_base_number,
    size_t expected_vecdim,
    size_t expected_nlist,
    size_t expected_hnsw_M,
    size_t expected_ef_construction,
    size_t expected_ivf_train_iters,
    const std::string& prefix
) {
    std::ifstream in(ivfhnsw_meta_path(prefix).c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t base_number = 0;
    uint64_t vecdim = 0;
    uint64_t nlist = 0;
    uint64_t ivf_train_iters = 0;
    uint64_t hnsw_M = 0;
    uint64_t ef_construction = 0;
    uint64_t random_seed = 0;

    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&base_number), sizeof(base_number));
    in.read(reinterpret_cast<char*>(&vecdim), sizeof(vecdim));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(nlist));
    in.read(reinterpret_cast<char*>(&ivf_train_iters), sizeof(ivf_train_iters));
    in.read(reinterpret_cast<char*>(&hnsw_M), sizeof(hnsw_M));
    in.read(reinterpret_cast<char*>(&ef_construction), sizeof(ef_construction));
    in.read(reinterpret_cast<char*>(&random_seed), sizeof(random_seed));

    const uint64_t expected_magic = 0x495646484e53574dULL;
    if (!in || magic != expected_magic || version != 1) {
        return false;
    }

    if (base_number != static_cast<uint64_t>(expected_base_number) ||
        vecdim != static_cast<uint64_t>(expected_vecdim) ||
        nlist != static_cast<uint64_t>(std::min(expected_nlist, expected_base_number)) ||
        hnsw_M != static_cast<uint64_t>(expected_hnsw_M) ||
        ef_construction != static_cast<uint64_t>(expected_ef_construction) ||
        ivf_train_iters != static_cast<uint64_t>(expected_ivf_train_iters)) {
        return false;
    }

    index.clear();
    index.base = base;
    index.base_number = static_cast<size_t>(base_number);
    index.vecdim = static_cast<size_t>(vecdim);
    index.nlist = static_cast<size_t>(nlist);
    index.ivf_train_iters = static_cast<size_t>(ivf_train_iters);
    index.hnsw_M = static_cast<size_t>(hnsw_M);
    index.ef_construction = static_cast<size_t>(ef_construction);
    index.random_seed = static_cast<size_t>(random_seed);

    index.centroids.resize(index.nlist * index.vecdim);
    if (!index.centroids.empty()) {
        in.read(
            reinterpret_cast<char*>(index.centroids.data()),
            sizeof(float) * index.centroids.size()
        );
    }

    index.inverted_lists.clear();
    index.inverted_lists.resize(index.nlist);

    for (size_t c = 0; c < index.nlist; ++c) {
        uint64_t sz = 0;
        in.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        if (!in) {
            index.clear();
            return false;
        }

        index.inverted_lists[c].resize(static_cast<size_t>(sz));
        if (sz > 0) {
            in.read(
                reinterpret_cast<char*>(index.inverted_lists[c].data()),
                sizeof(uint32_t) * static_cast<size_t>(sz)
            );
        }
    }

    if (!in) {
        index.clear();
        return false;
    }

    return true;
}

static inline bool ivfhnsw_load_index(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t expected_base_number,
    size_t expected_vecdim,
    size_t expected_nlist,
    size_t expected_hnsw_M,
    size_t expected_ef_construction,
    size_t expected_ivf_train_iters,
    const std::string& prefix,
    int rank_for_tmp = 0
) {
    const std::string bundle_path = ivfhnsw_bundle_path(prefix);
    std::ifstream in(bundle_path.c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t base_number = 0;
    uint64_t vecdim = 0;
    uint64_t nlist = 0;
    uint64_t ivf_train_iters = 0;
    uint64_t hnsw_M = 0;
    uint64_t ef_construction = 0;
    uint64_t random_seed = 0;

    mpi_hnsw_read_u64(in, magic);
    mpi_hnsw_read_u64(in, version);
    mpi_hnsw_read_u64(in, base_number);
    mpi_hnsw_read_u64(in, vecdim);
    mpi_hnsw_read_u64(in, nlist);
    mpi_hnsw_read_u64(in, ivf_train_iters);
    mpi_hnsw_read_u64(in, hnsw_M);
    mpi_hnsw_read_u64(in, ef_construction);
    mpi_hnsw_read_u64(in, random_seed);

    const uint64_t expected_magic = 0x49564648484e4255ULL;
    if (!in || magic != expected_magic || version != 1) {
        return false;
    }

    if (base_number != static_cast<uint64_t>(expected_base_number) ||
        vecdim != static_cast<uint64_t>(expected_vecdim) ||
        nlist != static_cast<uint64_t>(std::min(expected_nlist, expected_base_number)) ||
        hnsw_M != static_cast<uint64_t>(expected_hnsw_M) ||
        ef_construction != static_cast<uint64_t>(expected_ef_construction) ||
        ivf_train_iters != static_cast<uint64_t>(expected_ivf_train_iters)) {
        return false;
    }

    index.clear();
    index.base = base;
    index.base_number = static_cast<size_t>(base_number);
    index.vecdim = static_cast<size_t>(vecdim);
    index.nlist = static_cast<size_t>(nlist);
    index.ivf_train_iters = static_cast<size_t>(ivf_train_iters);
    index.hnsw_M = static_cast<size_t>(hnsw_M);
    index.ef_construction = static_cast<size_t>(ef_construction);
    index.random_seed = static_cast<size_t>(random_seed);

    index.centroids.resize(index.nlist * index.vecdim);
    if (!index.centroids.empty()) {
        in.read(
            reinterpret_cast<char*>(index.centroids.data()),
            static_cast<std::streamsize>(sizeof(float) * index.centroids.size())
        );
    }

    index.inverted_lists.clear();
    index.inverted_lists.resize(index.nlist);
    for (size_t c = 0; c < index.nlist; ++c) {
        uint64_t sz = 0;
        mpi_hnsw_read_u64(in, sz);
        if (!in) {
            index.clear();
            return false;
        }
        index.inverted_lists[c].resize(static_cast<size_t>(sz));
        if (sz > 0) {
            in.read(
                reinterpret_cast<char*>(index.inverted_lists[c].data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * static_cast<size_t>(sz))
            );
        }
    }

    if (!in) {
        index.clear();
        return false;
    }

    index.cluster_hnsw.assign(index.nlist, static_cast<HNSWIndex*>(nullptr));

    for (size_t c = 0; c < index.nlist; ++c) {
        const size_t sz = index.inverted_lists[c].size();
        if (sz == 0) {
            continue;
        }

        uint64_t cid_from_file = 0;
        uint64_t blob_size = 0;
        mpi_hnsw_read_u64(in, cid_from_file);
        mpi_hnsw_read_u64(in, blob_size);
        if (!in || cid_from_file != static_cast<uint64_t>(c) || blob_size == 0) {
            index.clear();
            return false;
        }

        std::vector<char> blob(static_cast<size_t>(blob_size));
        in.read(blob.data(), static_cast<std::streamsize>(blob.size()));
        if (!in) {
            index.clear();
            return false;
        }

        HNSWIndex* sub = new HNSWIndex();
        const std::string tmp_path = mpi_hnsw_tmp_path(
            prefix,
            "ivf_cluster",
            rank_for_tmp,
            c
        );

        if (!mpi_hnsw_write_file_blob(tmp_path, blob)) {
            delete sub;
            index.clear();
            return false;
        }

        if (!sub->load_index(tmp_path, sz, index.vecdim, sz)) {
            mpi_hnsw_remove_file(tmp_path);
            delete sub;
            index.clear();
            return false;
        }
        mpi_hnsw_remove_file(tmp_path);

        index.cluster_hnsw[c] = sub;
    }

    index.trained = true;
    std::cerr << "[IVF-HNSW] Loaded bundle: " << bundle_path << std::endl;
    return true;
}

static inline bool ivfhnsw_build_or_load_index(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist = 1024,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    size_t ivf_train_iters = 10,
    const std::string& prefix = "",
    size_t random_seed = 100
) {
    const std::string real_prefix = prefix.empty()
        ? ivfhnsw_default_prefix(
              base_number,
              vecdim,
              nlist,
              hnsw_M,
              ef_construction,
              ivf_train_iters
          )
        : prefix;

    if (ivfhnsw_load_index(
            index,
            base,
            base_number,
            vecdim,
            nlist,
            hnsw_M,
            ef_construction,
            ivf_train_iters,
            real_prefix)) {
        return true;
    }

    std::cerr << "[IVF-HNSW] No valid saved index, build new one" << std::endl;

    if (!ivfhnsw_build_index(
            index,
            base,
            base_number,
            vecdim,
            nlist,
            hnsw_M,
            ef_construction,
            ivf_train_iters,
            random_seed)) {
        return false;
    }

    ivfhnsw_save_index(index, real_prefix);
    return true;
}

static inline bool ivfhnsw_mpi_build_or_load_index(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist = 1024,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    size_t ivf_train_iters = 10,
    const std::string& prefix = "",
    size_t random_seed = 100,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const std::string real_prefix = prefix.empty()
        ? ivfhnsw_default_prefix(
              base_number,
              vecdim,
              nlist,
              hnsw_M,
              ef_construction,
              ivf_train_iters
          )
        : prefix;

    bool loaded = ivfhnsw_load_index(
        index,
        base,
        base_number,
        vecdim,
        nlist,
        hnsw_M,
        ef_construction,
        ivf_train_iters,
        real_prefix,
        rank
    );

    const int local_loaded = loaded ? 1 : 0;
    int all_loaded = 0;
    MPI_Allreduce(&local_loaded, &all_loaded, 1, MPI_INT, MPI_MIN, comm);

    if (all_loaded) {
        return true;
    }

    // 如果不是所有 rank 都能加载，则先让 root 构建并保存。
    if (rank == root) {
        if (!loaded) {
            std::cerr << "[IVF-HNSW][MPI] Root builds index" << std::endl;
            loaded = ivfhnsw_build_index(
                index,
                base,
                base_number,
                vecdim,
                nlist,
                hnsw_M,
                ef_construction,
                ivf_train_iters,
                random_seed
            );
            if (loaded) {
                ivfhnsw_save_index(index, real_prefix, rank);
            }
        } else {
            // root 已加载，但其他 rank 可能没有本地文件。仍等待后续 barrier。
            std::cerr << "[IVF-HNSW][MPI] Root has loaded index; other ranks will retry load"
                      << std::endl;
        }
    }

    MPI_Barrier(comm);

    if (rank != root) {
        loaded = ivfhnsw_load_index(
            index,
            base,
            base_number,
            vecdim,
            nlist,
            hnsw_M,
            ef_construction,
            ivf_train_iters,
            real_prefix,
            rank
        );

        // 如果集群不是共享文件系统，root 保存后其他节点仍然读不到，
        // 则允许本 rank 本地构建一次，保证程序可以继续运行。
        if (!loaded) {
            std::cerr << "[IVF-HNSW][MPI] Rank " << rank
                      << " cannot load root-saved index; build locally" << std::endl;
            loaded = ivfhnsw_build_index(
                index,
                base,
                base_number,
                vecdim,
                nlist,
                hnsw_M,
                ef_construction,
                ivf_train_iters,
                random_seed
            );
            if (loaded) {
                ivfhnsw_save_index(index, real_prefix, rank);
            }
        }
    }

    const int ok = loaded ? 1 : 0;
    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    return all_ok != 0;
}

static inline std::vector<uint32_t> ivfhnsw_select_probe_lists(
    const IVFHNSWMPIIndex& index,
    const float* query,
    size_t nprobe
) {
    assert(index.trained);
    assert(query != nullptr);

    nprobe = std::min(nprobe, index.nlist);
    std::priority_queue<std::pair<float, uint32_t> > heap;

    for (size_t c = 0; c < index.nlist; ++c) {
        const float* centroid = &index.centroids[c * index.vecdim];
        const float dis = ivfhnsw_ip_distance(query, centroid, index.vecdim);

        if (heap.size() < nprobe) {
            heap.emplace(dis, static_cast<uint32_t>(c));
        } else if (dis < heap.top().first) {
            heap.pop();
            heap.emplace(dis, static_cast<uint32_t>(c));
        }
    }

    std::vector<std::pair<float, uint32_t> > tmp;
    tmp.reserve(heap.size());
    while (!heap.empty()) {
        tmp.push_back(heap.top());
        heap.pop();
    }

    std::sort(tmp.begin(), tmp.end());

    std::vector<uint32_t> probe_lists;
    probe_lists.reserve(tmp.size());
    for (size_t i = 0; i < tmp.size(); ++i) {
        probe_lists.push_back(tmp[i].second);
    }

    return probe_lists;
}

static inline std::vector<int> ivfhnsw_assign_probe_owner_greedy(
    const IVFHNSWMPIIndex& index,
    const std::vector<uint32_t>& probe_lists,
    int world_size
) {
    assert(world_size > 0);

    std::vector<int> owner(probe_lists.size(), 0);
    if (probe_lists.empty()) {
        return owner;
    }

    struct ProbeItem {
        size_t list_size;
        size_t probe_pos;
        uint32_t cid;
    };

    std::vector<ProbeItem> items;
    items.reserve(probe_lists.size());

    for (size_t i = 0; i < probe_lists.size(); ++i) {
        const uint32_t cid = probe_lists[i];
        ProbeItem item;
        item.list_size = index.inverted_lists[cid].size();
        item.probe_pos = i;
        item.cid = cid;
        items.push_back(item);
    }

    std::sort(
        items.begin(),
        items.end(),
        [](const ProbeItem& a, const ProbeItem& b) {
            if (a.list_size != b.list_size) {
                return a.list_size > b.list_size;
            }
            return a.cid < b.cid;
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

        owner[items[i].probe_pos] = best_rank;
        load[static_cast<size_t>(best_rank)] += items[i].list_size;
    }

    return owner;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
ivfhnsw_search_one_cluster(
    const IVFHNSWMPIIndex& index,
    uint32_t cid,
    const float* query,
    size_t k,
    size_t ef,
    IVFHNSWSearchMode mode,
    int entry_count,
    int thread_num
) {
    std::priority_queue<std::pair<float, uint32_t> > empty;

    if (cid >= index.nlist) {
        return empty;
    }

    const size_t cluster_size = index.inverted_lists[cid].size();
    if (cluster_size == 0 || index.cluster_hnsw[cid] == nullptr) {
        return empty;
    }

    const size_t real_k = std::min(k, cluster_size);
    const size_t real_ef = std::max(real_k, ef);
    if (real_k == 0) {
        return empty;
    }

    const HNSWIndex& sub = *(index.cluster_hnsw[cid]);

    if (mode == IVFHNSW_SEARCH_LAYER0) {
        return hnsw_search_layer0(sub, query, real_k, real_ef);
    }

    if (mode == IVFHNSW_SEARCH_LAYER0_OPENMP) {
        return hnsw_search_layer0_multi_entry_openmp(
            sub,
            query,
            real_k,
            real_ef,
            entry_count,
            thread_num
        );
    }

    if (mode == IVFHNSW_SEARCH_LAYER0_PTHREAD) {
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
ivfhnsw_search_local_assigned_clusters(
    const IVFHNSWMPIIndex& index,
    const float* query,
    size_t k,
    const std::vector<uint32_t>& probe_lists,
    const std::vector<int>& owner,
    int rank,
    size_t ef,
    IVFHNSWSearchMode mode,
    int entry_count,
    int thread_num
) {
    std::priority_queue<std::pair<float, uint32_t> > local_result;

    for (size_t i = 0; i < probe_lists.size(); ++i) {
        if (owner[i] != rank) {
            continue;
        }

        const uint32_t cid = probe_lists[i];
        std::priority_queue<std::pair<float, uint32_t> > cluster_result =
            ivfhnsw_search_one_cluster(
                index,
                cid,
                query,
                k,
                ef,
                mode,
                entry_count,
                thread_num
            );

        ivfhnsw_merge_topk(local_result, cluster_result, k);
    }

    return local_result;
}

static inline void ivfhnsw_heap_to_fixed_arrays(
    std::priority_queue<std::pair<float, uint32_t> > heap,
    size_t k,
    std::vector<float>& dist,
    std::vector<uint32_t>& ids
) {
    mpi_hnsw_heap_to_fixed_arrays(heap, k, IVFHNSW_MPI_INVALID_ID, dist, ids);
}

static inline std::priority_queue<std::pair<float, uint32_t> >
ivfhnsw_mpi_merge_all_local_results(
    std::priority_queue<std::pair<float, uint32_t> > local_result,
    size_t k,
    MPI_Comm comm
) {
    return mpi_hnsw_merge_all_local_results(
        local_result,
        k,
        comm,
        IVFHNSW_MPI_INVALID_ID
    );
}

static inline std::priority_queue<std::pair<float, uint32_t> >
ivfhnsw_search_mpi(
    const IVFHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t nprobe = 64,
    size_t ef = 64,
    IVFHNSWSearchMode mode = IVFHNSW_SEARCH_HIERARCHICAL,
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

    const std::vector<uint32_t> probe_lists =
        ivfhnsw_select_probe_lists(index, query, nprobe);

    const std::vector<int> owner =
        ivfhnsw_assign_probe_owner_greedy(index, probe_lists, world_size);

    std::priority_queue<std::pair<float, uint32_t> > local_result =
        ivfhnsw_search_local_assigned_clusters(
            index,
            query,
            k,
            probe_lists,
            owner,
            rank,
            ef,
            mode,
            entry_count,
            thread_num
        );

    return ivfhnsw_mpi_merge_all_local_results(local_result, k, comm);
}

static inline std::priority_queue<std::pair<float, uint32_t> >
ivfhnsw_search_single_process(
    const IVFHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t nprobe = 64,
    size_t ef = 64,
    IVFHNSWSearchMode mode = IVFHNSW_SEARCH_HIERARCHICAL,
    int entry_count = 4,
    int thread_num = 1
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);

    const std::vector<uint32_t> probe_lists =
        ivfhnsw_select_probe_lists(index, query, nprobe);

    std::vector<int> owner(probe_lists.size(), 0);

    return ivfhnsw_search_local_assigned_clusters(
        index,
        query,
        k,
        probe_lists,
        owner,
        0,
        ef,
        mode,
        entry_count,
        thread_num
    );
}

// 兼容你前面实验中常用的函数命名风格：
// - 单进程调试：ivf_hnsw_search(...)
// - MPI 正式测试：ivf_hnsw_search_mpi(...)
static inline std::priority_queue<std::pair<float, uint32_t> >
ivf_hnsw_search(
    const IVFHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t nprobe = 64,
    size_t ef = 64,
    IVFHNSWSearchMode mode = IVFHNSW_SEARCH_HIERARCHICAL,
    int entry_count = 4,
    int thread_num = 1
) {
    return ivfhnsw_search_single_process(
        index,
        query,
        k,
        nprobe,
        ef,
        mode,
        entry_count,
        thread_num
    );
}

static inline bool ivf_hnsw_mpi_load_or_build(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist = 1024,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    size_t ivf_train_iters = 10,
    const std::string& prefix = "",
    size_t random_seed = 100,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    return ivfhnsw_mpi_build_or_load_index(
        index,
        base,
        base_number,
        vecdim,
        nlist,
        hnsw_M,
        ef_construction,
        ivf_train_iters,
        prefix,
        random_seed,
        comm,
        root
    );
}

static inline bool ivf_hnsw_load_or_build(
    IVFHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist = 1024,
    size_t hnsw_M = 16,
    size_t ef_construction = 150,
    size_t ivf_train_iters = 10,
    const std::string& prefix = "",
    size_t random_seed = 100
) {
    return ivfhnsw_build_or_load_index(
        index,
        base,
        base_number,
        vecdim,
        nlist,
        hnsw_M,
        ef_construction,
        ivf_train_iters,
        prefix,
        random_seed
    );
}
