#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mpi_hnsw_common.h"

// ================================================================
// HNSW + HNSW + MPI
// ================================================================

struct HNSWHNSWMPIIndex {
    const float* base = nullptr;
    size_t base_number = 0;
    size_t vecdim = 0;

    size_t subgraph_num = 0;
    size_t ivf_train_iters = 10;
    size_t random_seed = 100;

    size_t top_M = 8;
    size_t top_ef_construction = 80;

    size_t bottom_M = 16;
    size_t bottom_ef_construction = 150;

    bool trained = false;

    std::vector<float> centroids;
    std::vector<std::vector<uint32_t> > inverted_lists;
    std::vector<uint32_t> non_empty_subgraphs;

    // top_hnsw label = IVF list/subgraph id
    HNSWIndex top_hnsw;

    // bottom_hnsw[c] label = global base id
    std::vector<HNSWIndex> bottom_hnsw;
};

static inline std::string hnsw_hnsw_prefix(
    size_t base_number,
    size_t vecdim,
    size_t subgraph_num,
    size_t top_M,
    size_t top_ef_construction,
    size_t bottom_M,
    size_t bottom_ef_construction,
    size_t random_seed,
    size_t ivf_train_iters
) {
    std::ostringstream oss;
    oss << "files/hnsw_hnsw"
        << "_base_" << base_number
        << "_dim_" << vecdim
        << "_P_" << subgraph_num
        << "_topM_" << top_M
        << "_topEfc_" << top_ef_construction
        << "_botM_" << bottom_M
        << "_botEfc_" << bottom_ef_construction
        << "_seed_" << random_seed
        << "_ivfiter_" << ivf_train_iters;
    return oss.str();
}

static inline std::string hnsw_hnsw_bundle_path(const std::string& prefix) {
    return prefix + ".bundle";
}

static inline std::string hnsw_hnsw_tmp_path(
    const std::string& prefix,
    const std::string& name,
    int rank,
    size_t id = 0
) {
    std::ostringstream oss;
    oss << prefix << ".tmp.rank_" << rank << "." << name << "_" << id << ".hnsw";
    return oss.str();
}

static inline bool hnsw_hnsw_read_file_blob(
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

static inline bool hnsw_hnsw_write_file_blob(
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

static inline void hnsw_hnsw_remove_file(const std::string& path) {
    std::remove(path.c_str());
}

static inline void hnsw_hnsw_write_u64(std::ofstream& out, uint64_t x) {
    out.write(reinterpret_cast<const char*>(&x), sizeof(x));
}

static inline bool hnsw_hnsw_read_u64(std::ifstream& in, uint64_t& x) {
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    return static_cast<bool>(in);
}

static inline void hnsw_hnsw_prepare_ivf_layout(
    HNSWHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t subgraph_num,
    size_t top_M,
    size_t top_ef_construction,
    size_t bottom_M,
    size_t bottom_ef_construction,
    size_t random_seed,
    size_t ivf_train_iters
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(subgraph_num > 0);

    index.base = base;
    index.base_number = base_number;
    index.vecdim = vecdim;
    index.subgraph_num = std::min(subgraph_num, base_number);
    index.top_M = top_M;
    index.top_ef_construction = top_ef_construction;
    index.bottom_M = bottom_M;
    index.bottom_ef_construction = bottom_ef_construction;
    index.random_seed = random_seed;
    index.ivf_train_iters = ivf_train_iters;
    index.trained = false;

    std::cerr << "[HNSW+HNSW] Train IVF layout: nlist="
              << index.subgraph_num
              << ", iter=" << index.ivf_train_iters << std::endl;

    mpi_hnsw_train_ivf_centroids(
        index.centroids,
        base,
        base_number,
        vecdim,
        index.subgraph_num,
        ivf_train_iters
    );

    mpi_hnsw_assign_ivf_lists(
        index.inverted_lists,
        index.centroids,
        base,
        base_number,
        vecdim,
        index.subgraph_num
    );

    index.non_empty_subgraphs.clear();
    for (size_t c = 0; c < index.subgraph_num; ++c) {
        if (!index.inverted_lists[c].empty()) {
            index.non_empty_subgraphs.push_back(static_cast<uint32_t>(c));
        }
    }

    index.bottom_hnsw.clear();
    index.bottom_hnsw.resize(index.subgraph_num);
}

static inline bool hnsw_hnsw_load_index(
    HNSWHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t subgraph_num,
    size_t top_M,
    size_t top_ef_construction,
    size_t bottom_M,
    size_t bottom_ef_construction,
    size_t random_seed,
    size_t ivf_train_iters,
    int rank_for_tmp = 0
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(subgraph_num > 0);

    const size_t real_subgraph_num = std::min(subgraph_num, base_number);

    const std::string prefix = hnsw_hnsw_prefix(
        base_number,
        vecdim,
        real_subgraph_num,
        top_M,
        top_ef_construction,
        bottom_M,
        bottom_ef_construction,
        random_seed,
        ivf_train_iters
    );
    const std::string bundle_path = hnsw_hnsw_bundle_path(prefix);

    std::ifstream in(bundle_path.c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    uint64_t magic = 0;
    uint64_t version = 0;
    uint64_t file_base_number = 0;
    uint64_t file_vecdim = 0;
    uint64_t file_subgraph_num = 0;
    uint64_t file_top_M = 0;
    uint64_t file_top_ef_construction = 0;
    uint64_t file_bottom_M = 0;
    uint64_t file_bottom_ef_construction = 0;
    uint64_t file_random_seed = 0;
    uint64_t file_ivf_train_iters = 0;

    hnsw_hnsw_read_u64(in, magic);
    hnsw_hnsw_read_u64(in, version);
    hnsw_hnsw_read_u64(in, file_base_number);
    hnsw_hnsw_read_u64(in, file_vecdim);
    hnsw_hnsw_read_u64(in, file_subgraph_num);
    hnsw_hnsw_read_u64(in, file_top_M);
    hnsw_hnsw_read_u64(in, file_top_ef_construction);
    hnsw_hnsw_read_u64(in, file_bottom_M);
    hnsw_hnsw_read_u64(in, file_bottom_ef_construction);
    hnsw_hnsw_read_u64(in, file_random_seed);
    hnsw_hnsw_read_u64(in, file_ivf_train_iters);

    const uint64_t expected_magic = 0x48484842554e444cULL; // "HHHBUNDL"
    if (!in ||
        magic != expected_magic ||
        version != 1 ||
        file_base_number != static_cast<uint64_t>(base_number) ||
        file_vecdim != static_cast<uint64_t>(vecdim) ||
        file_subgraph_num != static_cast<uint64_t>(real_subgraph_num) ||
        file_top_M != static_cast<uint64_t>(top_M) ||
        file_top_ef_construction != static_cast<uint64_t>(top_ef_construction) ||
        file_bottom_M != static_cast<uint64_t>(bottom_M) ||
        file_bottom_ef_construction != static_cast<uint64_t>(bottom_ef_construction) ||
        file_random_seed != static_cast<uint64_t>(random_seed) ||
        file_ivf_train_iters != static_cast<uint64_t>(ivf_train_iters)) {
        return false;
    }

    index.base = base;
    index.base_number = base_number;
    index.vecdim = vecdim;
    index.subgraph_num = real_subgraph_num;
    index.top_M = top_M;
    index.top_ef_construction = top_ef_construction;
    index.bottom_M = bottom_M;
    index.bottom_ef_construction = bottom_ef_construction;
    index.random_seed = random_seed;
    index.ivf_train_iters = ivf_train_iters;
    index.trained = false;

    index.centroids.resize(index.subgraph_num * index.vecdim);
    if (!index.centroids.empty()) {
        in.read(
            reinterpret_cast<char*>(index.centroids.data()),
            static_cast<std::streamsize>(sizeof(float) * index.centroids.size())
        );
    }

    uint64_t list_num = 0;
    hnsw_hnsw_read_u64(in, list_num);
    if (!in || list_num != static_cast<uint64_t>(index.subgraph_num)) {
        return false;
    }

    index.inverted_lists.clear();
    index.inverted_lists.resize(index.subgraph_num);
    for (size_t c = 0; c < index.subgraph_num; ++c) {
        uint64_t sz = 0;
        hnsw_hnsw_read_u64(in, sz);
        if (!in) {
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

    uint64_t non_empty_count = 0;
    hnsw_hnsw_read_u64(in, non_empty_count);
    if (!in || non_empty_count == 0) {
        return false;
    }

    index.non_empty_subgraphs.resize(static_cast<size_t>(non_empty_count));
    in.read(
        reinterpret_cast<char*>(index.non_empty_subgraphs.data()),
        static_cast<std::streamsize>(
            sizeof(uint32_t) * index.non_empty_subgraphs.size()
        )
    );
    if (!in) {
        return false;
    }

    uint64_t top_blob_size = 0;
    hnsw_hnsw_read_u64(in, top_blob_size);
    std::vector<char> top_blob(static_cast<size_t>(top_blob_size));
    if (top_blob_size > 0) {
        in.read(top_blob.data(), static_cast<std::streamsize>(top_blob.size()));
    }
    if (!in || top_blob.empty()) {
        return false;
    }

    index.bottom_hnsw.clear();
    index.bottom_hnsw.resize(index.subgraph_num);

    const std::string top_tmp = hnsw_hnsw_tmp_path(
        prefix,
        "top",
        rank_for_tmp
    );
    if (!hnsw_hnsw_write_file_blob(top_tmp, top_blob)) {
        return false;
    }
    if (!index.top_hnsw.load_index(
            top_tmp,
            index.non_empty_subgraphs.size(),
            vecdim
        )) {
        hnsw_hnsw_remove_file(top_tmp);
        return false;
    }
    hnsw_hnsw_remove_file(top_tmp);

    for (size_t i = 0; i < index.non_empty_subgraphs.size(); ++i) {
        uint64_t cid_from_file = 0;
        uint64_t bottom_blob_size = 0;
        hnsw_hnsw_read_u64(in, cid_from_file);
        hnsw_hnsw_read_u64(in, bottom_blob_size);
        if (!in || cid_from_file != static_cast<uint64_t>(index.non_empty_subgraphs[i])) {
            return false;
        }

        std::vector<char> bottom_blob(static_cast<size_t>(bottom_blob_size));
        if (bottom_blob_size > 0) {
            in.read(
                bottom_blob.data(),
                static_cast<std::streamsize>(bottom_blob.size())
            );
        }
        if (!in || bottom_blob.empty()) {
            return false;
        }

        const size_t cid = index.non_empty_subgraphs[i];
        const size_t expected_size = index.inverted_lists[cid].size();
        const std::string bottom_tmp = hnsw_hnsw_tmp_path(
            prefix,
            "bottom",
            rank_for_tmp,
            cid
        );

        if (!hnsw_hnsw_write_file_blob(bottom_tmp, bottom_blob)) {
            return false;
        }

        if (!index.bottom_hnsw[cid].load_index(
                bottom_tmp,
                expected_size,
                vecdim
            )) {
            hnsw_hnsw_remove_file(bottom_tmp);
            return false;
        }
        hnsw_hnsw_remove_file(bottom_tmp);
    }

    index.trained = true;
    std::cerr << "[HNSW+HNSW] Loaded bundle index: " << bundle_path << std::endl;
    return true;
}

static inline bool hnsw_hnsw_build_and_save_index(
    HNSWHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t subgraph_num,
    size_t top_M,
    size_t top_ef_construction,
    size_t bottom_M,
    size_t bottom_ef_construction,
    size_t random_seed,
    size_t ivf_train_iters,
    int rank_for_tmp = 0
) {
    hnsw_hnsw_prepare_ivf_layout(
        index,
        base,
        base_number,
        vecdim,
        subgraph_num,
        top_M,
        top_ef_construction,
        bottom_M,
        bottom_ef_construction,
        random_seed,
        ivf_train_iters
    );

    const std::string prefix = hnsw_hnsw_prefix(
        base_number,
        vecdim,
        index.subgraph_num,
        top_M,
        top_ef_construction,
        bottom_M,
        bottom_ef_construction,
        random_seed,
        ivf_train_iters
    );

    mpi_hnsw_mkdir_if_needed("files");

    const std::string bundle_path = hnsw_hnsw_bundle_path(prefix);

    std::cerr << "[HNSW+HNSW] Build IVF-routed bundle index: "
              << bundle_path << std::endl;

    std::vector<std::vector<char> > bottom_blobs(index.non_empty_subgraphs.size());

    for (size_t i = 0; i < index.non_empty_subgraphs.size(); ++i) {
        const size_t cid = index.non_empty_subgraphs[i];
        const std::vector<uint32_t>& ids = index.inverted_lists[cid];

        std::cerr << "[HNSW+HNSW] Build bottom " << cid
                  << ", size=" << ids.size() << std::endl;

        if (!mpi_hnsw_build_from_global_ids(
                index.bottom_hnsw[cid],
                base,
                vecdim,
                ids,
                bottom_M,
                bottom_ef_construction,
                random_seed + cid + 1
            )) {
            index.trained = false;
            return false;
        }

        const std::string bottom_tmp = hnsw_hnsw_tmp_path(
            prefix,
            "bottom",
            rank_for_tmp,
            cid
        );
        if (!index.bottom_hnsw[cid].save_index(bottom_tmp)) {
            hnsw_hnsw_remove_file(bottom_tmp);
            index.trained = false;
            return false;
        }

        if (!hnsw_hnsw_read_file_blob(bottom_tmp, bottom_blobs[i])) {
            hnsw_hnsw_remove_file(bottom_tmp);
            index.trained = false;
            return false;
        }
        hnsw_hnsw_remove_file(bottom_tmp);
    }

    std::vector<float> top_vectors;
    top_vectors.reserve(index.non_empty_subgraphs.size() * vecdim);
    for (size_t i = 0; i < index.non_empty_subgraphs.size(); ++i) {
        const size_t cid = index.non_empty_subgraphs[i];
        const float* centroid = &index.centroids[cid * vecdim];
        top_vectors.insert(top_vectors.end(), centroid, centroid + vecdim);
    }

    if (!mpi_hnsw_build_from_labeled_vectors(
            index.top_hnsw,
            top_vectors.data(),
            index.non_empty_subgraphs,
            vecdim,
            top_M,
            top_ef_construction,
            random_seed + 99991
        )) {
        index.trained = false;
        return false;
    }

    const std::string top_tmp = hnsw_hnsw_tmp_path(prefix, "top", rank_for_tmp);
    if (!index.top_hnsw.save_index(top_tmp)) {
        hnsw_hnsw_remove_file(top_tmp);
        index.trained = false;
        return false;
    }

    std::vector<char> top_blob;
    if (!hnsw_hnsw_read_file_blob(top_tmp, top_blob)) {
        hnsw_hnsw_remove_file(top_tmp);
        index.trained = false;
        return false;
    }
    hnsw_hnsw_remove_file(top_tmp);

    std::ofstream out(bundle_path.c_str(), std::ios::binary | std::ios::trunc);
    if (!out) {
        index.trained = false;
        return false;
    }

    const uint64_t magic = 0x48484842554e444cULL; // "HHHBUNDL"
    const uint64_t version = 1;
    hnsw_hnsw_write_u64(out, magic);
    hnsw_hnsw_write_u64(out, version);
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.base_number));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.vecdim));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.subgraph_num));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.top_M));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.top_ef_construction));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.bottom_M));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.bottom_ef_construction));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.random_seed));
    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.ivf_train_iters));

    if (!index.centroids.empty()) {
        out.write(
            reinterpret_cast<const char*>(index.centroids.data()),
            static_cast<std::streamsize>(sizeof(float) * index.centroids.size())
        );
    }

    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.inverted_lists.size()));
    for (size_t c = 0; c < index.inverted_lists.size(); ++c) {
        const std::vector<uint32_t>& ids = index.inverted_lists[c];
        hnsw_hnsw_write_u64(out, static_cast<uint64_t>(ids.size()));
        if (!ids.empty()) {
            out.write(
                reinterpret_cast<const char*>(ids.data()),
                static_cast<std::streamsize>(sizeof(uint32_t) * ids.size())
            );
        }
    }

    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(index.non_empty_subgraphs.size()));
    out.write(
        reinterpret_cast<const char*>(index.non_empty_subgraphs.data()),
        static_cast<std::streamsize>(
            sizeof(uint32_t) * index.non_empty_subgraphs.size()
        )
    );

    hnsw_hnsw_write_u64(out, static_cast<uint64_t>(top_blob.size()));
    if (!top_blob.empty()) {
        out.write(top_blob.data(), static_cast<std::streamsize>(top_blob.size()));
    }

    for (size_t i = 0; i < index.non_empty_subgraphs.size(); ++i) {
        hnsw_hnsw_write_u64(
            out,
            static_cast<uint64_t>(index.non_empty_subgraphs[i])
        );
        hnsw_hnsw_write_u64(out, static_cast<uint64_t>(bottom_blobs[i].size()));
        if (!bottom_blobs[i].empty()) {
            out.write(
                bottom_blobs[i].data(),
                static_cast<std::streamsize>(bottom_blobs[i].size())
            );
        }
    }

    if (!out) {
        index.trained = false;
        return false;
    }

    index.trained = true;
    std::cerr << "[HNSW+HNSW] Bundle build done." << std::endl;
    return true;
}

static inline bool hnsw_hnsw_mpi_load_or_build(
    HNSWHNSWMPIIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t subgraph_num = 16,
    size_t top_M = 8,
    size_t top_ef_construction = 80,
    size_t bottom_M = 16,
    size_t bottom_ef_construction = 150,
    size_t random_seed = 100,
    size_t ivf_train_iters = 10,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (hnsw_hnsw_load_index(
            index,
            base,
            base_number,
            vecdim,
            subgraph_num,
            top_M,
            top_ef_construction,
            bottom_M,
            bottom_ef_construction,
            random_seed,
            ivf_train_iters,
            rank
        )) {
        MPI_Barrier(comm);
        return true;
    }

    if (rank == 0) {
        hnsw_hnsw_build_and_save_index(
            index,
            base,
            base_number,
            vecdim,
            subgraph_num,
            top_M,
            top_ef_construction,
            bottom_M,
            bottom_ef_construction,
            random_seed,
            ivf_train_iters,
            rank
        );
    }

    MPI_Barrier(comm);

    if (rank == 0 && index.trained) {
        return true;
    }

    if (hnsw_hnsw_load_index(
            index,
            base,
            base_number,
            vecdim,
            subgraph_num,
            top_M,
            top_ef_construction,
            bottom_M,
            bottom_ef_construction,
            random_seed,
            ivf_train_iters,
            rank
        )) {
        return true;
    }

    std::cerr << "[HNSW+HNSW] Rank local load failed; build locally." << std::endl;
    return hnsw_hnsw_build_and_save_index(
        index,
        base,
        base_number,
        vecdim,
        subgraph_num,
        top_M,
        top_ef_construction,
        bottom_M,
        bottom_ef_construction,
        random_seed,
        ivf_train_iters,
        rank
    );
}

static inline std::vector<uint32_t> hnsw_hnsw_select_subgraphs(
    const HNSWHNSWMPIIndex& index,
    const float* query,
    size_t upper_probe,
    size_t upper_ef_search
) {
    assert(index.trained);
    assert(index.top_hnsw.trained);
    assert(query != nullptr);

    upper_probe = std::min<size_t>(
        std::max<size_t>(upper_probe, 1),
        index.non_empty_subgraphs.size()
    );

    index.top_hnsw.graph->setEf(std::max(upper_probe, upper_ef_search));
    std::priority_queue<std::pair<float, hnswlib::labeltype> > raw =
        index.top_hnsw.graph->searchKnn(query, upper_probe);

    std::vector<uint32_t> subgraphs;
    subgraphs.reserve(upper_probe);
    while (!raw.empty()) {
        const hnswlib::labeltype label = raw.top().second;
        raw.pop();
        const uint32_t cid = static_cast<uint32_t>(label);
        if (cid < index.subgraph_num && !index.inverted_lists[cid].empty()) {
            subgraphs.push_back(cid);
        }
    }

    std::sort(subgraphs.begin(), subgraphs.end());
    subgraphs.erase(std::unique(subgraphs.begin(), subgraphs.end()), subgraphs.end());
    return subgraphs;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
hnsw_hnsw_search_mpi(
    const HNSWHNSWMPIIndex& index,
    const float* query,
    size_t k,
    size_t upper_probe = 4,
    size_t upper_ef_search = 32,
    size_t bottom_ef_search = 64,
    MPI_Comm comm = MPI_COMM_WORLD
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const std::vector<uint32_t> subgraphs = hnsw_hnsw_select_subgraphs(
        index,
        query,
        upper_probe,
        upper_ef_search
    );

    std::priority_queue<std::pair<float, uint32_t> > local_heap;

    for (size_t i = 0; i < subgraphs.size(); ++i) {
        if (static_cast<int>(i % static_cast<size_t>(world_size)) != rank) {
            continue;
        }

        const uint32_t cid = subgraphs[i];
        const HNSWIndex& bottom = index.bottom_hnsw[cid];
        if (!bottom.trained || bottom.graph == nullptr || bottom.base_number == 0) {
            continue;
        }

        const size_t real_k = std::min(k, bottom.base_number);
        const size_t real_ef = std::max(real_k, bottom_ef_search);
        std::priority_queue<std::pair<float, uint32_t> > one =
            hnsw_search_hierarchical(bottom, query, real_k, real_ef);
        hnsw_merge_topk(local_heap, one, k);
    }

    return mpi_hnsw_merge_all_local_results(local_heap, k, comm);
}