#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include <fstream>
#include <string>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>

#include "flat_simd.h"
#include "ivf_simd.h"
#include "pq_simd.h"

static inline void ivffirstpq_push_topk(
    std::priority_queue<std::pair<float, uint32_t>>& heap,
    float dis,
    uint32_t id,
    size_t k
) {
    if (heap.size() < k) {
        heap.push(std::make_pair(dis, id));
    } else if (dis < heap.top().first) {
        heap.pop();
        heap.push(std::make_pair(dis, id));
    }
}

static inline void ivffirstpq_ensure_files_dir() {
    struct stat st;
    if (stat("files", &st) != 0) {
        if (mkdir("files", 0755) != 0 && errno != EEXIST) {
            std::cerr << "[IVF-first-PQ] Cannot create files directory, errno = "
                      << errno << ", reason = " << std::strerror(errno) << std::endl;
        }
    }
}

struct IVFFirstPQIndex {
    const float* base = nullptr;
    size_t base_number = 0;
    size_t vecdim = 0;

    // IVF 参数
    size_t nlist = 0;

    // PQ 参数
    size_t M = 4;
    size_t Ks = 256;
    size_t dsub = 0;
    size_t pq_train_iters = 10;

    bool trained = false;

    // IVF centroid: centroids[cid * vecdim + d]
    std::vector<float> centroids;

    // 每个簇保存原始 base id
    std::vector<std::vector<uint32_t>> inverted_lists;

    // 每个簇保存连续 PQ code:
    // inverted_codes[cid][pos * M + m]
    std::vector<std::vector<uint8_t>> inverted_codes;

    // 每个簇一套 PQ codebook:
    // cluster_codebooks[((cid * M + m) * Ks + c) * dsub + j]
    std::vector<float> cluster_codebooks;

    bool save_index(const std::string& path) const {
        errno = 0;

        std::ofstream ofs(
            path,
            std::ios::out | std::ios::binary | std::ios::trunc
        );

        if (!ofs.is_open()) {
            std::cerr << "[IVF-first-PQ] Cannot open index file for writing: "
                      << path << std::endl;
            std::cerr << "[IVF-first-PQ] errno = " << errno
                      << ", reason = " << std::strerror(errno) << std::endl;
            return false;
        }

        ofs.write(reinterpret_cast<const char*>(&base_number), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&vecdim), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&nlist), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&M), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&Ks), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&dsub), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&pq_train_iters), sizeof(size_t));
        ofs.write(reinterpret_cast<const char*>(&trained), sizeof(bool));

        size_t centroid_size = centroids.size();
        ofs.write(reinterpret_cast<const char*>(&centroid_size), sizeof(size_t));
        if (centroid_size > 0) {
            ofs.write(reinterpret_cast<const char*>(centroids.data()),
                      centroid_size * sizeof(float));
        }

        size_t codebook_size = cluster_codebooks.size();
        ofs.write(reinterpret_cast<const char*>(&codebook_size), sizeof(size_t));
        if (codebook_size > 0) {
            ofs.write(reinterpret_cast<const char*>(cluster_codebooks.data()),
                      codebook_size * sizeof(float));
        }

        size_t list_num = inverted_lists.size();
        ofs.write(reinterpret_cast<const char*>(&list_num), sizeof(size_t));

        for (size_t cid = 0; cid < list_num; ++cid) {
            size_t list_size = inverted_lists[cid].size();
            ofs.write(reinterpret_cast<const char*>(&list_size), sizeof(size_t));

            if (list_size > 0) {
                ofs.write(reinterpret_cast<const char*>(inverted_lists[cid].data()),
                          list_size * sizeof(uint32_t));
            }

            if (!ofs.good()) {
                std::cerr << "[IVF-first-PQ] Write failed at list "
                          << cid << std::endl;
                return false;
            }
        }

        size_t code_list_num = inverted_codes.size();
        ofs.write(reinterpret_cast<const char*>(&code_list_num), sizeof(size_t));

        for (size_t cid = 0; cid < code_list_num; ++cid) {
            size_t code_size = inverted_codes[cid].size();
            ofs.write(reinterpret_cast<const char*>(&code_size), sizeof(size_t));

            if (code_size > 0) {
                ofs.write(reinterpret_cast<const char*>(inverted_codes[cid].data()),
                          code_size * sizeof(uint8_t));
            }

            if (!ofs.good()) {
                std::cerr << "[IVF-first-PQ] Write failed at code list "
                          << cid << std::endl;
                return false;
            }
        }

        ofs.flush();
        if (!ofs.good()) {
            std::cerr << "[IVF-first-PQ] Flush failed: " << path << std::endl;
            return false;
        }

        ofs.close();
        if (ofs.fail()) {
            std::cerr << "[IVF-first-PQ] Close failed: " << path << std::endl;
            return false;
        }

        return true;
    }

    bool load_index(
        const std::string& path,
        const float* base_ptr,
        size_t expected_base_number,
        size_t expected_vecdim,
        size_t expected_nlist,
        size_t expected_M,
        size_t expected_Ks
    ) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.is_open()) {
            return false;
        }

        size_t file_base_number = 0;
        size_t file_vecdim = 0;
        size_t file_nlist = 0;
        size_t file_M = 0;
        size_t file_Ks = 0;
        size_t file_dsub = 0;
        size_t file_pq_train_iters = 0;
        bool file_trained = false;

        ifs.read(reinterpret_cast<char*>(&file_base_number), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_vecdim), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_nlist), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_M), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_Ks), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_dsub), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_pq_train_iters), sizeof(size_t));
        ifs.read(reinterpret_cast<char*>(&file_trained), sizeof(bool));

        if (!ifs.good()) {
            return false;
        }

        if (file_base_number != expected_base_number ||
            file_vecdim != expected_vecdim ||
            file_nlist != expected_nlist ||
            file_M != expected_M ||
            file_Ks != expected_Ks ||
            file_dsub != expected_vecdim / expected_M ||
            !file_trained) {
            return false;
        }

        size_t centroid_size = 0;
        ifs.read(reinterpret_cast<char*>(&centroid_size), sizeof(size_t));
        if (centroid_size != expected_nlist * expected_vecdim) {
            return false;
        }

        std::vector<float> loaded_centroids(centroid_size);
        if (centroid_size > 0) {
            ifs.read(reinterpret_cast<char*>(loaded_centroids.data()),
                     centroid_size * sizeof(float));
        }

        size_t codebook_size = 0;
        ifs.read(reinterpret_cast<char*>(&codebook_size), sizeof(size_t));
        if (codebook_size != expected_nlist * expected_M * expected_Ks * file_dsub) {
            return false;
        }

        std::vector<float> loaded_codebooks(codebook_size);
        if (codebook_size > 0) {
            ifs.read(reinterpret_cast<char*>(loaded_codebooks.data()),
                     codebook_size * sizeof(float));
        }

        size_t list_num = 0;
        ifs.read(reinterpret_cast<char*>(&list_num), sizeof(size_t));
        if (list_num != expected_nlist) {
            return false;
        }

        std::vector<std::vector<uint32_t>> loaded_lists(list_num);
        for (size_t cid = 0; cid < list_num; ++cid) {
            size_t list_size = 0;
            ifs.read(reinterpret_cast<char*>(&list_size), sizeof(size_t));

            loaded_lists[cid].resize(list_size);
            if (list_size > 0) {
                ifs.read(reinterpret_cast<char*>(loaded_lists[cid].data()),
                         list_size * sizeof(uint32_t));
            }

            if (!ifs.good()) {
                return false;
            }
        }

        size_t code_list_num = 0;
        ifs.read(reinterpret_cast<char*>(&code_list_num), sizeof(size_t));
        if (!ifs.good() || code_list_num != expected_nlist) {
            return false;
        }

        std::vector<std::vector<uint8_t>> loaded_codes(code_list_num);
        for (size_t cid = 0; cid < code_list_num; ++cid) {
            size_t code_size = 0;
            ifs.read(reinterpret_cast<char*>(&code_size), sizeof(size_t));

            if (code_size != loaded_lists[cid].size() * expected_M) {
                return false;
            }

            loaded_codes[cid].resize(code_size);
            if (code_size > 0) {
                ifs.read(reinterpret_cast<char*>(loaded_codes[cid].data()),
                         code_size * sizeof(uint8_t));
            }

            if (!ifs.good()) {
                return false;
            }
        }

        base = base_ptr;
        base_number = file_base_number;
        vecdim = file_vecdim;
        nlist = file_nlist;
        M = file_M;
        Ks = file_Ks;
        dsub = file_dsub;
        pq_train_iters = file_pq_train_iters;
        trained = true;

        centroids.swap(loaded_centroids);
        cluster_codebooks.swap(loaded_codebooks);
        inverted_lists.swap(loaded_lists);
        inverted_codes.swap(loaded_codes);

        return true;
    }
};

static inline float* ivffirstpq_codebook_ptr(
    IVFFirstPQIndex& index,
    size_t cid,
    size_t m,
    size_t c
) {
    return index.cluster_codebooks.data()
        + ((cid * index.M + m) * index.Ks + c) * index.dsub;
}

static inline const float* ivffirstpq_codebook_ptr(
    const IVFFirstPQIndex& index,
    size_t cid,
    size_t m,
    size_t c
) {
    return index.cluster_codebooks.data()
        + ((cid * index.M + m) * index.Ks + c) * index.dsub;
}

static inline const float* ivffirstpq_base_subvec_ptr(
    const IVFFirstPQIndex& index,
    uint32_t id,
    size_t m
) {
    return index.base + static_cast<size_t>(id) * index.vecdim + m * index.dsub;
}

static inline uint32_t ivffirstpq_nearest_ivf_centroid(
    const IVFFirstPQIndex& index,
    const float* x
) {
    uint32_t best_cid = 0;
    float best_dis = std::numeric_limits<float>::max();

    for (size_t cid = 0; cid < index.nlist; ++cid) {
        const float* center = index.centroids.data() + cid * index.vecdim;
        float dis = 1.0f - InnerProductSIMD(x, center, index.vecdim);

        if (dis < best_dis) {
            best_dis = dis;
            best_cid = static_cast<uint32_t>(cid);
        }
    }

    return best_cid;
}

static inline void ivffirstpq_init_ivf_centroids(IVFFirstPQIndex& index) {
    index.centroids.assign(index.nlist * index.vecdim, 0.0f);

    for (size_t cid = 0; cid < index.nlist; ++cid) {
        size_t id = cid * index.base_number / index.nlist;
        if (id >= index.base_number) {
            id = index.base_number - 1;
        }

        std::memcpy(
            index.centroids.data() + cid * index.vecdim,
            index.base + id * index.vecdim,
            index.vecdim * sizeof(float)
        );
    }
}

static inline void ivffirstpq_build_ivf_lists(
    IVFFirstPQIndex& index,
    size_t ivf_train_iters
) {
    ivffirstpq_init_ivf_centroids(index);

    std::vector<uint32_t> assign(index.base_number, 0);
    std::vector<float> sums(index.nlist * index.vecdim, 0.0f);
    std::vector<uint32_t> counts(index.nlist, 0);

    for (size_t it = 0; it < ivf_train_iters; ++it) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < index.base_number; ++i) {
            const float* x = index.base + i * index.vecdim;
            uint32_t cid = ivffirstpq_nearest_ivf_centroid(index, x);

            assign[i] = cid;
            counts[cid]++;

            float* sum = sums.data() + static_cast<size_t>(cid) * index.vecdim;
            for (size_t d = 0; d < index.vecdim; ++d) {
                sum[d] += x[d];
            }
        }

        for (size_t cid = 0; cid < index.nlist; ++cid) {
            if (counts[cid] == 0) {
                continue;
            }

            float* dst = index.centroids.data() + cid * index.vecdim;
            const float* sum = sums.data() + cid * index.vecdim;
            float inv = 1.0f / static_cast<float>(counts[cid]);

            for (size_t d = 0; d < index.vecdim; ++d) {
                dst[d] = sum[d] * inv;
            }
        }
    }

    std::fill(counts.begin(), counts.end(), 0);

    for (size_t i = 0; i < index.base_number; ++i) {
        uint32_t cid = ivffirstpq_nearest_ivf_centroid(
            index,
            index.base + i * index.vecdim
        );
        assign[i] = cid;
        counts[cid]++;
    }

    index.inverted_lists.assign(index.nlist, std::vector<uint32_t>());

    for (size_t cid = 0; cid < index.nlist; ++cid) {
        index.inverted_lists[cid].reserve(counts[cid]);
    }

    for (size_t i = 0; i < index.base_number; ++i) {
        index.inverted_lists[assign[i]].push_back(static_cast<uint32_t>(i));
    }
}

static inline void ivffirstpq_init_cluster_pq_centroids(
    IVFFirstPQIndex& index,
    size_t cid,
    size_t m
) {
    const std::vector<uint32_t>& ids = index.inverted_lists[cid];

    for (size_t c = 0; c < index.Ks; ++c) {
        float* center = ivffirstpq_codebook_ptr(index, cid, m, c);

        if (ids.empty()) {
            std::fill(center, center + index.dsub, 0.0f);
            continue;
        }

        size_t pos = c * ids.size() / index.Ks;
        if (pos >= ids.size()) {
            pos = ids.size() - 1;
        }

        uint32_t id = ids[pos];
        std::memcpy(
            center,
            ivffirstpq_base_subvec_ptr(index, id, m),
            index.dsub * sizeof(float)
        );
    }
}

static inline uint8_t ivffirstpq_nearest_cluster_pq_centroid(
    const IVFFirstPQIndex& index,
    size_t cid,
    const float* x,
    size_t m
) {
    float best_dis = std::numeric_limits<float>::max();
    uint8_t best_code = 0;

    for (size_t c = 0; c < index.Ks; ++c) {
        const float* center = ivffirstpq_codebook_ptr(index, cid, m, c);
        float dis = l2_distance_simd(x, center, index.dsub);

        if (dis < best_dis) {
            best_dis = dis;
            best_code = static_cast<uint8_t>(c);
        }
    }

    return best_code;
}

static inline void ivffirstpq_train_one_cluster(
    IVFFirstPQIndex& index,
    size_t cid
) {
    const std::vector<uint32_t>& ids = index.inverted_lists[cid];

    if (ids.empty()) {
        for (size_t m = 0; m < index.M; ++m) {
            for (size_t c = 0; c < index.Ks; ++c) {
                float* center = ivffirstpq_codebook_ptr(index, cid, m, c);
                std::fill(center, center + index.dsub, 0.0f);
            }
        }
        return;
    }

    std::vector<uint8_t> assign(ids.size(), 0);
    std::vector<float> sums(index.Ks * index.dsub, 0.0f);
    std::vector<uint32_t> counts(index.Ks, 0);

    for (size_t m = 0; m < index.M; ++m) {
        ivffirstpq_init_cluster_pq_centroids(index, cid, m);

        for (size_t it = 0; it < index.pq_train_iters; ++it) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t pos = 0; pos < ids.size(); ++pos) {
                uint32_t id = ids[pos];
                const float* x = ivffirstpq_base_subvec_ptr(index, id, m);

                uint8_t code = ivffirstpq_nearest_cluster_pq_centroid(
                    index,
                    cid,
                    x,
                    m
                );

                assign[pos] = code;
                counts[code]++;

                float* sum = sums.data() + static_cast<size_t>(code) * index.dsub;
                for (size_t j = 0; j < index.dsub; ++j) {
                    sum[j] += x[j];
                }
            }

            for (size_t c = 0; c < index.Ks; ++c) {
                float* center = ivffirstpq_codebook_ptr(index, cid, m, c);

                if (counts[c] == 0) {
                    size_t pos = (c * 9973 + it * 7919 + m * 104729 + cid * 17)
                               % ids.size();
                    uint32_t id = ids[pos];

                    std::memcpy(
                        center,
                        ivffirstpq_base_subvec_ptr(index, id, m),
                        index.dsub * sizeof(float)
                    );
                    continue;
                }

                const float inv = 1.0f / static_cast<float>(counts[c]);
                const float* sum = sums.data() + c * index.dsub;

                for (size_t j = 0; j < index.dsub; ++j) {
                    center[j] = sum[j] * inv;
                }
            }
        }
    }
}

static inline void ivffirstpq_train_all_cluster_pq(IVFFirstPQIndex& index) {
    index.cluster_codebooks.assign(
        index.nlist * index.M * index.Ks * index.dsub,
        0.0f
    );

    for (size_t cid = 0; cid < index.nlist; ++cid) {
        ivffirstpq_train_one_cluster(index, cid);
    }
}

static inline void ivffirstpq_encode_all_clusters(IVFFirstPQIndex& index) {
    index.inverted_codes.assign(index.nlist, std::vector<uint8_t>());

    for (size_t cid = 0; cid < index.nlist; ++cid) {
        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        std::vector<uint8_t>& codes = index.inverted_codes[cid];

        codes.assign(ids.size() * index.M, 0);

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            uint32_t id = ids[pos];

            for (size_t m = 0; m < index.M; ++m) {
                const float* x = ivffirstpq_base_subvec_ptr(index, id, m);
                codes[pos * index.M + m] =
                    ivffirstpq_nearest_cluster_pq_centroid(
                        index,
                        cid,
                        x,
                        m
                    );
            }
        }
    }
}

static inline std::string ivffirstpq_default_index_path(
    size_t base_number,
    size_t vecdim,
    size_t nlist,
    size_t M,
    size_t Ks,
    size_t pq_train_iters,
    size_t ivf_train_iters
) {
    return std::string("files/ivffirst_pq_index_base_") + std::to_string(base_number) +
           "_dim_" + std::to_string(vecdim) +
           "_nlist_" + std::to_string(nlist) +
           "_M_" + std::to_string(M) +
           "_Ks_" + std::to_string(Ks) +
           "_pqiter_" + std::to_string(pq_train_iters) +
           "_ivfiter_" + std::to_string(ivf_train_iters) +
           "_cluster_pq_layout_codes.bin";
}

static inline void ivffirstpq_build(
    IVFFirstPQIndex& index,
    const float* base,
    size_t base_number,
    size_t vecdim,
    size_t nlist,
    size_t M = 4,
    size_t Ks = 256,
    size_t pq_train_iters = 10,
    size_t ivf_train_iters = 10,
    const std::string& index_path = ""
) {
    assert(base != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(vecdim % 32 == 0);
    assert(vecdim % M == 0);
    assert(nlist > 0 && nlist <= base_number);
    assert(Ks > 0 && Ks <= 256);

    std::string path = index_path;
    if (path.empty()) {
        path = ivffirstpq_default_index_path(
            base_number,
            vecdim,
            nlist,
            M,
            Ks,
            pq_train_iters,
            ivf_train_iters
        );
    }

    if (index.load_index(path, base, base_number, vecdim, nlist, M, Ks)) {
        std::cerr << "[IVF-first-PQ] Load index from: " << path << std::endl;
        return;
    }

    std::cerr << "[IVF-first-PQ] No valid saved index, build new index..."
              << std::endl;

    index.base = base;
    index.base_number = base_number;
    index.vecdim = vecdim;
    index.nlist = nlist;
    index.M = M;
    index.Ks = Ks;
    index.dsub = vecdim / M;
    index.pq_train_iters = pq_train_iters;
    index.trained = false;

    // 先训练 IVF，并构建倒排表
    ivffirstpq_build_ivf_lists(index, ivf_train_iters);

    // 每个 IVF 簇内部单独训练 PQ
    ivffirstpq_train_all_cluster_pq(index);

    // 每个簇内部单独编码，并连续保存 code
    ivffirstpq_encode_all_clusters(index);

    index.trained = true;

    ivffirstpq_ensure_files_dir();

    std::cerr << "[IVF-first-PQ] Try saving index to: "
              << path << std::endl;

    if (!index.save_index(path)) {
        std::cerr << "[IVF-first-PQ] Failed to save index to: "
                  << path << std::endl;
    } else {
        std::cerr << "[IVF-first-PQ] Saved index to: "
                  << path << std::endl;
    }
}

static inline void ivffirstpq_select_probe_lists(
    const IVFFirstPQIndex& index,
    const float* query,
    size_t nprobe,
    std::vector<uint32_t>& probe_lists
) {
    assert(index.trained);
    assert(query != nullptr);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    std::priority_queue<std::pair<float, uint32_t>> heap;

    for (size_t cid = 0; cid < index.nlist; ++cid) {
        const float* center = index.centroids.data() + cid * index.vecdim;
        float dis = 1.0f - InnerProductSIMD(query, center, index.vecdim);

        ivffirstpq_push_topk(heap, dis, static_cast<uint32_t>(cid), nprobe);
    }

    probe_lists.clear();
    probe_lists.reserve(nprobe);

    while (!heap.empty()) {
        probe_lists.push_back(heap.top().second);
        heap.pop();
    }
}

static inline void ivffirstpq_build_cluster_lut(
    const IVFFirstPQIndex& index,
    uint32_t cid,
    const float* query,
    std::vector<float>& lut
) {
    assert(index.trained);
    assert(query != nullptr);

    lut.assign(index.M * index.Ks, 0.0f);

    for (size_t m = 0; m < index.M; ++m) {
        const float* qsub = query + m * index.dsub;
        float* lut_m = lut.data() + m * index.Ks;

        size_t c = 0;

        for (; c + 4 <= index.Ks; c += 4) {
            const float* center0 = ivffirstpq_codebook_ptr(index, cid, m, c + 0);
            const float* center1 = ivffirstpq_codebook_ptr(index, cid, m, c + 1);
            const float* center2 = ivffirstpq_codebook_ptr(index, cid, m, c + 2);
            const float* center3 = ivffirstpq_codebook_ptr(index, cid, m, c + 3);

            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            size_t j = 0;
            for (; j + 4 <= index.dsub; j += 4) {
                float32x4_t qv = vld1q_f32(qsub + j);

                float32x4_t c0 = vld1q_f32(center0 + j);
                float32x4_t c1 = vld1q_f32(center1 + j);
                float32x4_t c2 = vld1q_f32(center2 + j);
                float32x4_t c3 = vld1q_f32(center3 + j);

                float32x4_t d0 = vsubq_f32(qv, c0);
                float32x4_t d1 = vsubq_f32(qv, c1);
                float32x4_t d2 = vsubq_f32(qv, c2);
                float32x4_t d3 = vsubq_f32(qv, c3);

                acc0 = vfmaq_f32(acc0, d0, d0);
                acc1 = vfmaq_f32(acc1, d1, d1);
                acc2 = vfmaq_f32(acc2, d2, d2);
                acc3 = vfmaq_f32(acc3, d3, d3);
            }

            float dis0 = vaddvq_f32(acc0);
            float dis1 = vaddvq_f32(acc1);
            float dis2 = vaddvq_f32(acc2);
            float dis3 = vaddvq_f32(acc3);

            for (; j < index.dsub; ++j) {
                float q = qsub[j];

                float t0 = q - center0[j];
                float t1 = q - center1[j];
                float t2 = q - center2[j];
                float t3 = q - center3[j];

                dis0 += t0 * t0;
                dis1 += t1 * t1;
                dis2 += t2 * t2;
                dis3 += t3 * t3;
            }

            lut_m[c + 0] = dis0;
            lut_m[c + 1] = dis1;
            lut_m[c + 2] = dis2;
            lut_m[c + 3] = dis3;
        }

        for (; c < index.Ks; ++c) {
            const float* center = ivffirstpq_codebook_ptr(index, cid, m, c);
            lut_m[c] = l2_distance_simd(qsub, center, index.dsub);
        }
    }
}

static inline float ivffirstpq_adc_distance_code(
    const IVFFirstPQIndex& index,
    const uint8_t* code,
    const std::vector<float>& lut
) {
    float dis = 0.0f;

    for (size_t m = 0; m < index.M; ++m) {
        dis += lut[m * index.Ks + code[m]];
    }

    return dis;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivffirstpq_rerank_flat(
    const IVFFirstPQIndex& index,
    const float* query,
    std::priority_queue<std::pair<float, uint32_t>> candidates,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t>> result;

    while (!candidates.empty()) {
        uint32_t id = candidates.top().second;
        candidates.pop();

        const float* x = index.base + static_cast<size_t>(id) * index.vecdim;
        float dis = 1.0f - InnerProductSIMD(query, x, index.vecdim);

        ivffirstpq_push_topk(result, dis, id, k);
    }

    return result;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivffirstpq_search_simd(
    const IVFFirstPQIndex& index,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }

    std::vector<uint32_t> probe_lists;
    ivffirstpq_select_probe_lists(index, query, nprobe, probe_lists);

    std::priority_queue<std::pair<float, uint32_t>> candidates;
    std::vector<float> lut;

    for (uint32_t cid : probe_lists) {
        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<uint8_t>& codes = index.inverted_codes[cid];

        if (ids.empty()) {
            continue;
        }

        ivffirstpq_build_cluster_lut(index, cid, query, lut);

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            const uint8_t* code = codes.data() + pos * index.M;

            float dis = ivffirstpq_adc_distance_code(index, code, lut);
            ivffirstpq_push_topk(candidates, dis, ids[pos], top_p);
        }
    }

    return ivffirstpq_rerank_flat(index, query, candidates, k);
}