#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include "hnsw_search.h"

#ifndef MPI_HNSW_INVALID_ID
#define MPI_HNSW_INVALID_ID 0xffffffffu
#endif

static inline bool mpi_hnsw_file_exists(const std::string& path) {
    std::ifstream fin(path.c_str(), std::ios::binary);
    return fin.good();
}

static inline bool mpi_hnsw_mkdir_if_needed(const std::string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        return S_ISDIR(st.st_mode);
    }

    if (mkdir(dir.c_str(), 0777) == 0) {
        return true;
    }

    return errno == EEXIST;
}

static inline std::string mpi_hnsw_tmp_path(
    const std::string& prefix,
    const std::string& name,
    int rank,
    size_t id = 0
) {
    std::ostringstream oss;
    oss << prefix << ".tmp.rank_" << rank << "." << name << "_" << id << ".hnsw";
    return oss.str();
}

static inline void mpi_hnsw_remove_file(const std::string& path) {
    std::remove(path.c_str());
}

static inline void mpi_hnsw_write_u64(std::ofstream& out, uint64_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}

static inline bool mpi_hnsw_read_u64(std::ifstream& in, uint64_t& x) {
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    return static_cast<bool>(in);
}

static inline bool mpi_hnsw_read_file_blob(
    const std::string& path,
    std::vector<char>& blob
) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    if (size < 0) {
        return false;
    }
    in.seekg(0, std::ios::beg);

    blob.resize(static_cast<size_t>(size));
    if (!blob.empty()) {
        in.read(blob.data(), size);
    }
    return static_cast<bool>(in);
}

static inline bool mpi_hnsw_write_file_blob(
    const std::string& path,
    const std::vector<char>& blob
) {
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    if (!blob.empty()) {
        out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    }
    return static_cast<bool>(out);
}

static inline float mpi_hnsw_inner_product_scalar(
    const float* a,
    const float* b,
    size_t dim
) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline float mpi_hnsw_ip_distance(
    const float* a,
    const float* b,
    size_t dim
) {
    return 1.0f - mpi_hnsw_inner_product_scalar(a, b, dim);
}

static inline void mpi_hnsw_train_ivf_centroids(
    std::vector<float>& centroids,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist,
    size_t train_iters
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(nlist > 0);

    centroids.assign(nlist * vecdim, 0.0f);
    for (size_t c = 0; c < nlist; ++c) {
        const size_t id = (c * base_number) / nlist;
        std::memcpy(
            &centroids[c * vecdim],
            base + id * vecdim,
            sizeof(float) * vecdim
        );
    }

    std::vector<float> new_centroids(nlist * vecdim, 0.0f);
    std::vector<size_t> counts(nlist, 0);

    for (size_t iter = 0; iter < train_iters; ++iter) {
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), static_cast<size_t>(0));

        for (size_t i = 0; i < base_number; ++i) {
            const float* x = base + i * vecdim;
            size_t best_c = 0;
            float best_dis = std::numeric_limits<float>::max();

            for (size_t c = 0; c < nlist; ++c) {
                const float* centroid = &centroids[c * vecdim];
                const float dis = mpi_hnsw_ip_distance(x, centroid, vecdim);
                if (dis < best_dis) {
                    best_dis = dis;
                    best_c = c;
                }
            }

            ++counts[best_c];
            float* sum = &new_centroids[best_c * vecdim];
            for (size_t d = 0; d < vecdim; ++d) {
                sum[d] += x[d];
            }
        }

        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }
            const float inv_cnt = 1.0f / static_cast<float>(counts[c]);
            float* dst = &centroids[c * vecdim];
            float* sum = &new_centroids[c * vecdim];
            for (size_t d = 0; d < vecdim; ++d) {
                dst[d] = sum[d] * inv_cnt;
            }
        }
    }
}

static inline uint32_t mpi_hnsw_find_nearest_centroid(
    const std::vector<float>& centroids,
    size_t nlist,
    size_t vecdim,
    const float* query
) {
    assert(centroids.size() == nlist * vecdim);
    assert(query != nullptr);

    size_t best_c = 0;
    float best_dis = std::numeric_limits<float>::max();
    for (size_t c = 0; c < nlist; ++c) {
        const float* centroid = &centroids[c * vecdim];
        const float dis = mpi_hnsw_ip_distance(query, centroid, vecdim);
        if (dis < best_dis) {
            best_dis = dis;
            best_c = c;
        }
    }
    return static_cast<uint32_t>(best_c);
}

static inline void mpi_hnsw_assign_ivf_lists(
    std::vector<std::vector<uint32_t> >& lists,
    const std::vector<float>& centroids,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist
) {
    assert(base != nullptr);
    lists.clear();
    lists.resize(nlist);

    for (size_t i = 0; i < base_number; ++i) {
        const float* x = base + i * vecdim;
        const uint32_t cid = mpi_hnsw_find_nearest_centroid(
            centroids,
            nlist,
            vecdim,
            x
        );
        lists[cid].push_back(static_cast<uint32_t>(i));
    }
}

static inline bool mpi_hnsw_build_from_global_ids(
    HNSWIndex& index,
    const float* base,
    size_t vecdim,
    const std::vector<uint32_t>& ids,
    size_t M,
    size_t ef_construction,
    size_t random_seed
) {
    assert(base != nullptr);
    assert(vecdim > 0);

    if (ids.empty()) {
        index.graph.reset();
        index.space.reset();
        index.base_number = 0;
        index.vecdim = vecdim;
        index.trained = false;
        return true;
    }

    try {
        index.base_number = ids.size();
        index.vecdim = vecdim;
        index.M = M;
        index.ef_construction = ef_construction;
        index.random_seed = random_seed;
        index.trained = false;
        index.space.reset(new hnswlib::InnerProductSpace(vecdim));
        index.graph.reset(new hnswlib::HierarchicalNSW<float>(
            index.space.get(),
            ids.size(),
            M,
            ef_construction,
            random_seed
        ));

        for (size_t i = 0; i < ids.size(); ++i) {
            const uint32_t gid = ids[i];
            index.graph->addPoint(
                base + static_cast<size_t>(gid) * vecdim,
                static_cast<hnswlib::labeltype>(gid)
            );
        }

        index.trained = true;
        return true;
    } catch (...) {
        index.graph.reset();
        index.space.reset();
        index.trained = false;
        return false;
    }
}

static inline bool mpi_hnsw_build_from_labeled_vectors(
    HNSWIndex& index,
    const float* vectors,
    const std::vector<uint32_t>& labels,
    size_t vecdim,
    size_t M,
    size_t ef_construction,
    size_t random_seed
) {
    assert(vectors != nullptr);
    assert(vecdim > 0);
    assert(!labels.empty());

    try {
        index.base_number = labels.size();
        index.vecdim = vecdim;
        index.M = M;
        index.ef_construction = ef_construction;
        index.random_seed = random_seed;
        index.trained = false;
        index.space.reset(new hnswlib::InnerProductSpace(vecdim));
        index.graph.reset(new hnswlib::HierarchicalNSW<float>(
            index.space.get(),
            labels.size(),
            M,
            ef_construction,
            random_seed
        ));

        for (size_t i = 0; i < labels.size(); ++i) {
            index.graph->addPoint(
                vectors + i * vecdim,
                static_cast<hnswlib::labeltype>(labels[i])
            );
        }

        index.trained = true;
        return true;
    } catch (...) {
        index.graph.reset();
        index.space.reset();
        index.trained = false;
        return false;
    }
}

static inline void mpi_hnsw_heap_to_fixed_arrays(
    std::priority_queue<std::pair<float, uint32_t> > heap,
    size_t k,
    uint32_t invalid_id,
    std::vector<float>& dist,
    std::vector<uint32_t>& ids
) {
    dist.assign(k, std::numeric_limits<float>::infinity());
    ids.assign(k, invalid_id);

    size_t pos = 0;
    while (!heap.empty() && pos < k) {
        const std::pair<float, uint32_t> item = heap.top();
        heap.pop();
        dist[pos] = item.first;
        ids[pos] = item.second;
        ++pos;
    }
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_hnsw_merge_all_local_results(
    std::priority_queue<std::pair<float, uint32_t> > local_result,
    size_t k,
    MPI_Comm comm,
    uint32_t invalid_id = MPI_HNSW_INVALID_ID
) {
    std::priority_queue<std::pair<float, uint32_t> > global_result;
    if (k == 0) {
        return global_result;
    }

    int world_size = 1;
    MPI_Comm_size(comm, &world_size);
    assert(k <= static_cast<size_t>(std::numeric_limits<int>::max()));
    const int count = static_cast<int>(k);

    std::vector<float> send_dist;
    std::vector<uint32_t> send_ids;
    mpi_hnsw_heap_to_fixed_arrays(local_result, k, invalid_id, send_dist, send_ids);

    std::vector<float> all_dist(static_cast<size_t>(world_size) * k);
    std::vector<uint32_t> all_ids(static_cast<size_t>(world_size) * k);

    MPI_Allgather(
        send_dist.data(),
        count,
        MPI_FLOAT,
        all_dist.data(),
        count,
        MPI_FLOAT,
        comm
    );

    MPI_Allgather(
        send_ids.data(),
        count,
        MPI_UNSIGNED,
        all_ids.data(),
        count,
        MPI_UNSIGNED,
        comm
    );

    for (size_t i = 0; i < all_ids.size(); ++i) {
        if (all_ids[i] == invalid_id || !std::isfinite(all_dist[i])) {
            continue;
        }
        hnsw_push_topk(global_result, all_dist[i], all_ids[i], k);
    }

    return global_result;
}