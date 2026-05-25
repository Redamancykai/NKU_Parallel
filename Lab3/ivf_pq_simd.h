#pragma once

#include <pthread.h>
#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
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

struct IVFPQIndex {
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

    // IVF centroid：centroids[c * vecdim + d]
    std::vector<float> centroids;

    // 每个簇保存原始 base id
    std::vector<std::vector<uint32_t>> inverted_lists;

    // 每个簇连续保存对应向量的 PQ code
    std::vector<std::vector<uint8_t>> inverted_codes;

    // PQ codebook：codebooks[(m * Ks + c) * dsub + j]
    std::vector<float> codebooks;

    // 全部 base 的 PQ 编码：codes[i * M + m]
    std::vector<uint8_t> codes;

    bool save_index(const std::string& path) const {
        errno = 0;

        std::ofstream ofs(
            path,
            std::ios::out | std::ios::binary | std::ios::trunc
        );

        if (!ofs.is_open()) {
            std::cerr << "[IVF-PQ] Cannot open index file for writing: "
                      << path << std::endl;
            std::cerr << "[IVF-PQ] errno = " << errno
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

        size_t codebook_size = codebooks.size();
        ofs.write(reinterpret_cast<const char*>(&codebook_size), sizeof(size_t));
        if (codebook_size > 0) {
            ofs.write(reinterpret_cast<const char*>(codebooks.data()),
                      codebook_size * sizeof(float));
        }

        size_t code_size = codes.size();
        ofs.write(reinterpret_cast<const char*>(&code_size), sizeof(size_t));
        if (code_size > 0) {
            ofs.write(reinterpret_cast<const char*>(codes.data()),
                      code_size * sizeof(uint8_t));
        }

        size_t list_num = inverted_lists.size();
        ofs.write(reinterpret_cast<const char*>(&list_num), sizeof(size_t));

        for (size_t i = 0; i < list_num; ++i) {
            size_t list_size = inverted_lists[i].size();
            ofs.write(reinterpret_cast<const char*>(&list_size), sizeof(size_t));

            if (list_size > 0) {
                ofs.write(reinterpret_cast<const char*>(inverted_lists[i].data()),
                          list_size * sizeof(uint32_t));
            }

            if (!ofs.good()) {
                std::cerr << "[IVF-PQ] Write failed at inverted list "
                          << i << ", list_size = " << list_size << std::endl;
                return false;
            }
        }

        // 保存倒排表内连续 PQ code
        size_t inv_code_list_num = inverted_codes.size();
        ofs.write(reinterpret_cast<const char*>(&inv_code_list_num), sizeof(size_t));

        for (size_t i = 0; i < inv_code_list_num; ++i) {
            size_t code_list_size = inverted_codes[i].size();
            ofs.write(reinterpret_cast<const char*>(&code_list_size), sizeof(size_t));

            if (code_list_size > 0) {
                ofs.write(reinterpret_cast<const char*>(inverted_codes[i].data()),
                          code_list_size * sizeof(uint8_t));
            }

            if (!ofs.good()) {
                std::cerr << "[IVF-PQ] Write failed at inverted code list "
                          << i << ", code_list_size = " << code_list_size << std::endl;
                return false;
            }
        }

        ofs.flush();
        if (!ofs.good()) {
            std::cerr << "[IVF-PQ] Flush failed when saving index: "
                      << path << std::endl;
            return false;
        }

        ofs.close();
        if (ofs.fail()) {
            std::cerr << "[IVF-PQ] Close failed when saving index: "
                      << path << std::endl;
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
        if (codebook_size != expected_M * expected_Ks * file_dsub) {
            return false;
        }

        std::vector<float> loaded_codebooks(codebook_size);
        if (codebook_size > 0) {
            ifs.read(reinterpret_cast<char*>(loaded_codebooks.data()),
                     codebook_size * sizeof(float));
        }

        size_t code_size = 0;
        ifs.read(reinterpret_cast<char*>(&code_size), sizeof(size_t));
        if (code_size != expected_base_number * expected_M) {
            return false;
        }

        std::vector<uint8_t> loaded_codes(code_size);
        if (code_size > 0) {
            ifs.read(reinterpret_cast<char*>(loaded_codes.data()),
                     code_size * sizeof(uint8_t));
        }

        size_t list_num = 0;
        ifs.read(reinterpret_cast<char*>(&list_num), sizeof(size_t));
        if (list_num != expected_nlist) {
            return false;
        }

        std::vector<std::vector<uint32_t>> loaded_lists(list_num);
        for (size_t i = 0; i < list_num; ++i) {
            size_t list_size = 0;
            ifs.read(reinterpret_cast<char*>(&list_size), sizeof(size_t));

            loaded_lists[i].resize(list_size);
            if (list_size > 0) {
                ifs.read(reinterpret_cast<char*>(loaded_lists[i].data()),
                         list_size * sizeof(uint32_t));
            }

            if (!ifs.good()) {
                return false;
            }
        }

        size_t inv_code_list_num = 0;
        ifs.read(reinterpret_cast<char*>(&inv_code_list_num), sizeof(size_t));
        if (!ifs.good() || inv_code_list_num != expected_nlist) {
            return false;
        }

        std::vector<std::vector<uint8_t>> loaded_inverted_codes(inv_code_list_num);
        for (size_t i = 0; i < inv_code_list_num; ++i) {
            size_t code_list_size = 0;
            ifs.read(reinterpret_cast<char*>(&code_list_size), sizeof(size_t));

            if (code_list_size != loaded_lists[i].size() * expected_M) {
                return false;
            }

            loaded_inverted_codes[i].resize(code_list_size);
            if (code_list_size > 0) {
                ifs.read(reinterpret_cast<char*>(loaded_inverted_codes[i].data()),
                         code_list_size * sizeof(uint8_t));
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
        codebooks.swap(loaded_codebooks);
        codes.swap(loaded_codes);
        inverted_lists.swap(loaded_lists);
        inverted_codes.swap(loaded_inverted_codes);

        return true;
    }
};

struct IVFPQRefineThreadParam {
    int tid = 0;
    int thread_num = 1;

    const IVFPQIndex* index = nullptr;
    const float* query = nullptr;
    const std::vector<uint32_t>* probe_lists = nullptr;
    const std::vector<float>* lut = nullptr;

    size_t local_p = 0;

    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    size_t scanned_lists = 0;
    size_t scanned_points = 0;
};

struct IVFPQDynamicThreadParam {
    int tid = 0;

    const IVFPQIndex* index = nullptr;
    const float* query = nullptr;
    const std::vector<uint32_t>* probe_lists = nullptr;
    const std::vector<float>* lut = nullptr;

    size_t local_p = 0;

    int* next_task = nullptr;
    pthread_mutex_t* task_mutex = nullptr;

    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    size_t scanned_lists = 0;
    size_t scanned_points = 0;
};

static inline void ivfpq_push_topk(
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

static inline const float* ivfpq_subvec_ptr(
    const IVFPQIndex& index,
    size_t vec_id,
    size_t m
) {
    return index.base + vec_id * index.vecdim + m * index.dsub;
}

static inline float* ivfpq_centroid_ptr(
    IVFPQIndex& index,
    size_t m,
    size_t c
) {
    return index.codebooks.data() + (m * index.Ks + c) * index.dsub;
}

static inline const float* ivfpq_centroid_ptr(
    const IVFPQIndex& index,
    size_t m,
    size_t c
) {
    return index.codebooks.data() + (m * index.Ks + c) * index.dsub;
}

static inline void ivfpq_init_pq_centroids(IVFPQIndex& index, size_t m) {
    for (size_t c = 0; c < index.Ks; ++c) {
        size_t id = c * index.base_number / index.Ks;
        if (id >= index.base_number) {
            id = index.base_number - 1;
        }
        std::memcpy(
            ivfpq_centroid_ptr(index, m, c),
            ivfpq_subvec_ptr(index, id, m),
            index.dsub * sizeof(float)
        );
    }
}

static inline uint8_t ivfpq_nearest_pq_centroid(
    const IVFPQIndex& index,
    const float* x,
    size_t m
) {
    float best_dis = std::numeric_limits<float>::max();
    uint8_t best_id = 0;

    for (size_t c = 0; c < index.Ks; ++c) {
        float dis = l2_distance_simd(x, ivfpq_centroid_ptr(index, m, c), index.dsub);
        if (dis < best_dis) {
            best_dis = dis;
            best_id = static_cast<uint8_t>(c);
        }
    }

    return best_id;
}

static inline void ivfpq_train_codebooks(IVFPQIndex& index) {
    std::vector<uint8_t> assign(index.base_number, 0);
    std::vector<float> sums(index.Ks * index.dsub, 0.0f);
    std::vector<uint32_t> counts(index.Ks, 0);

    for (size_t m = 0; m < index.M; ++m) {
        ivfpq_init_pq_centroids(index, m);

        for (size_t it = 0; it < index.pq_train_iters; ++it) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t i = 0; i < index.base_number; ++i) {
                const float* x = ivfpq_subvec_ptr(index, i, m);
                uint8_t cid = ivfpq_nearest_pq_centroid(index, x, m);

                assign[i] = cid;
                counts[cid]++;

                float* sum = sums.data() + static_cast<size_t>(cid) * index.dsub;
                for (size_t j = 0; j < index.dsub; ++j) {
                    sum[j] += x[j];
                }
            }

            for (size_t c = 0; c < index.Ks; ++c) {
                float* center = ivfpq_centroid_ptr(index, m, c);

                if (counts[c] == 0) {
                    size_t id = (c * 9973 + it * 7919 + m * 104729) % index.base_number;
                    std::memcpy(center, ivfpq_subvec_ptr(index, id, m), index.dsub * sizeof(float));
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

static inline void ivfpq_encode_base(IVFPQIndex& index) {
    index.codes.assign(index.base_number * index.M, 0);

    for (size_t i = 0; i < index.base_number; ++i) {
        for (size_t m = 0; m < index.M; ++m) {
            const float* x = ivfpq_subvec_ptr(index, i, m);
            index.codes[i * index.M + m] = ivfpq_nearest_pq_centroid(index, x, m);
        }
    }
}

static inline void ivfpq_init_ivf_centroids(IVFPQIndex& index) {
    index.centroids.assign(index.nlist * index.vecdim, 0.0f);

    for (size_t c = 0; c < index.nlist; ++c) {
        size_t id = c * index.base_number / index.nlist;
        if (id >= index.base_number) {
            id = index.base_number - 1;
        }

        std::memcpy(
            index.centroids.data() + c * index.vecdim,
            index.base + id * index.vecdim,
            index.vecdim * sizeof(float)
        );
    }
}

static inline uint32_t ivfpq_nearest_ivf_centroid(
    const IVFPQIndex& index,
    const float* x
) {
    uint32_t best_c = 0;
    float best_dis = std::numeric_limits<float>::max();

    for (size_t c = 0; c < index.nlist; ++c) {
        const float* centroid = index.centroids.data() + c * index.vecdim;
        float dis = 1.0f - InnerProductSIMD(x, centroid, index.vecdim);

        if (dis < best_dis) {
            best_dis = dis;
            best_c = static_cast<uint32_t>(c);
        }
    }

    return best_c;
}

static inline void ivfpq_build_ivf_lists(
    IVFPQIndex& index,
    size_t ivf_train_iters
) {
    ivfpq_init_ivf_centroids(index);

    std::vector<uint32_t> assign(index.base_number, 0);
    std::vector<float> new_centroids(index.nlist * index.vecdim, 0.0f);
    std::vector<uint32_t> counts(index.nlist, 0);

    for (size_t it = 0; it < ivf_train_iters; ++it) {
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < index.base_number; ++i) {
            const float* x = index.base + i * index.vecdim;
            uint32_t cid = ivfpq_nearest_ivf_centroid(index, x);
            assign[i] = cid;
            counts[cid]++;

            float* sum = new_centroids.data() + static_cast<size_t>(cid) * index.vecdim;
            for (size_t d = 0; d < index.vecdim; ++d) {
                sum[d] += x[d];
            }
        }

        for (size_t c = 0; c < index.nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }

            float* dst = index.centroids.data() + c * index.vecdim;
            const float* sum = new_centroids.data() + c * index.vecdim;
            float inv = 1.0f / static_cast<float>(counts[c]);
            for (size_t d = 0; d < index.vecdim; ++d) {
                dst[d] = sum[d] * inv;
            }
        }
    }

    std::fill(counts.begin(), counts.end(), 0);
    for (size_t i = 0; i < index.base_number; ++i) {
        uint32_t cid = ivfpq_nearest_ivf_centroid(index, index.base + i * index.vecdim);
        assign[i] = cid;
        counts[cid]++;
    }

    index.inverted_lists.assign(index.nlist, std::vector<uint32_t>());
    for (size_t c = 0; c < index.nlist; ++c) {
        index.inverted_lists[c].reserve(counts[c]);
    }

    for (size_t i = 0; i < index.base_number; ++i) {
        index.inverted_lists[assign[i]].push_back(static_cast<uint32_t>(i));
    }

    // 为每个倒排表构建连续 PQ code 布局
    index.inverted_codes.assign(index.nlist, std::vector<uint8_t>());
    for (size_t c = 0; c < index.nlist; ++c) {
        const std::vector<uint32_t>& ids = index.inverted_lists[c];
        std::vector<uint8_t>& codes_in_list = index.inverted_codes[c];

        codes_in_list.resize(ids.size() * index.M);

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            uint32_t id = ids[pos];
            const uint8_t* src = index.codes.data() + static_cast<size_t>(id) * index.M;
            uint8_t* dst = codes_in_list.data() + pos * index.M;
            std::memcpy(dst, src, index.M * sizeof(uint8_t));
        }
    }
}

static inline void ivfpq_ensure_files_dir() {
    struct stat st;
    if (stat("files", &st) != 0) {
        if (mkdir("files", 0755) != 0 && errno != EEXIST) {
            std::cerr << "[IVF-PQ] Cannot create files directory, errno = "
                      << errno << ", reason = " << std::strerror(errno) << std::endl;
        }
    }
}

static inline std::string ivfpq_default_index_path(
    size_t base_number,
    size_t vecdim,
    size_t nlist,
    size_t M,
    size_t Ks,
    size_t pq_train_iters,
    size_t ivf_train_iters
) {
    return std::string("files/ivfpq_index_base_") + std::to_string(base_number) +
           "_dim_" + std::to_string(vecdim) +
           "_nlist_" + std::to_string(nlist) +
           "_M_" + std::to_string(M) +
           "_Ks_" + std::to_string(Ks) +
           "_pqiter_" + std::to_string(pq_train_iters) +
           "_ivfiter_" + std::to_string(ivf_train_iters) +
           "_layout_codes.bin";
}

static inline void ivfpq_build(
    IVFPQIndex& index,
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
        path = ivfpq_default_index_path(
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
        std::cerr << "[IVF-PQ] Load index from: " << path << std::endl;
        return;
    }

    std::cerr << "[IVF-PQ] No valid saved index, build new index..." << std::endl;

    index.base = base;
    index.base_number = base_number;
    index.vecdim = vecdim;
    index.nlist = nlist;
    index.M = M;
    index.Ks = Ks;
    index.dsub = vecdim / M;
    index.pq_train_iters = pq_train_iters;
    index.trained = false;

    index.codebooks.assign(M * Ks * index.dsub, 0.0f);
    ivfpq_train_codebooks(index);
    ivfpq_encode_base(index);

    ivfpq_build_ivf_lists(index, ivf_train_iters);

    index.trained = true;

    ivfpq_ensure_files_dir();
    std::cerr << "[IVF-PQ] Try saving index to: " << path << std::endl;
    if (!index.save_index(path)) {
        std::cerr << "[IVF-PQ] Failed to save index to: " << path << std::endl;
    } else {
        std::cerr << "[IVF-PQ] Saved index to: " << path << std::endl;
    }
}

static inline void ivfpq_select_probe_lists(
    const IVFPQIndex& index,
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

    for (size_t c = 0; c < index.nlist; ++c) {
        const float* centroid = index.centroids.data() + c * index.vecdim;
        float dis = 1.0f - InnerProductSIMD(query, centroid, index.vecdim);

        ivfpq_push_topk(heap, dis, static_cast<uint32_t>(c), nprobe);
    }

    probe_lists.clear();
    probe_lists.reserve(nprobe);
    while (!heap.empty()) {
        probe_lists.push_back(heap.top().second);
        heap.pop();
    }
}

static inline void ivfpq_build_lut(
    const IVFPQIndex& index,
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
            const float* center0 = ivfpq_centroid_ptr(index, m, c + 0);
            const float* center1 = ivfpq_centroid_ptr(index, m, c + 1);
            const float* center2 = ivfpq_centroid_ptr(index, m, c + 2);
            const float* center3 = ivfpq_centroid_ptr(index, m, c + 3);

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
            lut_m[c] = l2_distance_simd(qsub, ivfpq_centroid_ptr(index, m, c), index.dsub);
        }
    }
}

static inline float ivfpq_adc_distance(
    const IVFPQIndex& index,
    uint32_t id,
    const std::vector<float>& lut
) {
    const uint8_t* code = index.codes.data() + static_cast<size_t>(id) * index.M;
    float dis = 0.0f;

    for (size_t m = 0; m < index.M; ++m) {
        dis += lut[m * index.Ks + code[m]];
    }

    return dis;
}

static inline float ivfpq_adc_distance_code(
    const IVFPQIndex& index,
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
ivfpq_rerank_flat(
    const IVFPQIndex& index,
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
        ivfpq_push_topk(result, dis, id, k);
    }

    return result;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd(
    const IVFPQIndex& index,
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
    ivfpq_select_probe_lists(index, query, nprobe, probe_lists);

    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (uint32_t cid : probe_lists) {
        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<uint8_t>& codes_in_list = index.inverted_codes[cid];

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            const uint8_t* code = codes_in_list.data() + pos * index.M;
            float dis = ivfpq_adc_distance_code(index, code, lut);
            ivfpq_push_topk(candidates, dis, ids[pos], top_p);
        }
    }

    return ivfpq_rerank_flat(index, query, candidates, k);
}

static void* ivfpq_refine_cluster_thread_func(void* arg) {
    IVFPQRefineThreadParam* param = static_cast<IVFPQRefineThreadParam*>(arg);

    const IVFPQIndex& index = *(param->index);
    const std::vector<uint32_t>& probe_lists = *(param->probe_lists);
    const std::vector<float>& lut = *(param->lut);

    for (size_t i = static_cast<size_t>(param->tid);
         i < probe_lists.size();
         i += static_cast<size_t>(param->thread_num)) {
        uint32_t cid = probe_lists[i];
        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<uint8_t>& codes_in_list = index.inverted_codes[cid];

        param->scanned_lists++;
        param->scanned_points += ids.size();

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            const uint8_t* code = codes_in_list.data() + pos * index.M;
            float dis = ivfpq_adc_distance_code(index, code, lut);
            ivfpq_push_topk(param->local_heap, dis, ids[pos], param->local_p);
        }
    }

    return nullptr;
}

static void* ivfpq_refine_dynamic_thread_func(void* arg) {
    IVFPQDynamicThreadParam* param = static_cast<IVFPQDynamicThreadParam*>(arg);

    const IVFPQIndex& index = *(param->index);
    const std::vector<uint32_t>& probe_lists = *(param->probe_lists);
    const std::vector<float>& lut = *(param->lut);

    while (true) {
        int task_id = 0;

        pthread_mutex_lock(param->task_mutex);
        task_id = *(param->next_task);
        (*(param->next_task))++;
        pthread_mutex_unlock(param->task_mutex);

        if (task_id >= static_cast<int>(probe_lists.size())) {
            break;
        }

        uint32_t cid = probe_lists[static_cast<size_t>(task_id)];
        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<uint8_t>& codes_in_list = index.inverted_codes[cid];

        param->scanned_lists++;
        param->scanned_points += ids.size();

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            const uint8_t* code = codes_in_list.data() + pos * index.M;
            float dis = ivfpq_adc_distance_code(index, code, lut);
            ivfpq_push_topk(param->local_heap, dis, ids[pos], param->local_p);
        }
    }

    return nullptr;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_pthread_refine_cluster(
    const IVFPQIndex& index,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }
    if (thread_num <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(index, query, nprobe, probe_lists);

    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    std::vector<pthread_t> threads(static_cast<size_t>(thread_num));
    std::vector<IVFPQRefineThreadParam> params(static_cast<size_t>(thread_num));

    for (int t = 0; t < thread_num; ++t) {
        params[t].tid = t;
        params[t].thread_num = thread_num;
        params[t].index = &index;
        params[t].query = query;
        params[t].probe_lists = &probe_lists;
        params[t].lut = &lut;
        params[t].local_p = top_p;

        pthread_create(&threads[t], nullptr, ivfpq_refine_cluster_thread_func, &params[t]);
    }

    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (int t = 0; t < thread_num; ++t) {
        pthread_join(threads[t], nullptr);

        while (!params[t].local_heap.empty()) {
            auto item = params[t].local_heap.top();
            params[t].local_heap.pop();
            ivfpq_push_topk(candidates, item.first, item.second, top_p);
        }
    }

    return ivfpq_rerank_flat(index, query, candidates, k);
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_pthread_refine_dynamic(
    const IVFPQIndex& index,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }
    if (thread_num <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(index, query, nprobe, probe_lists);

    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    int next_task = 0;
    pthread_mutex_t task_mutex;
    pthread_mutex_init(&task_mutex, nullptr);

    std::vector<pthread_t> threads(static_cast<size_t>(thread_num));
    std::vector<IVFPQDynamicThreadParam> params(static_cast<size_t>(thread_num));

    for (int t = 0; t < thread_num; ++t) {
        params[t].tid = t;
        params[t].index = &index;
        params[t].query = query;
        params[t].probe_lists = &probe_lists;
        params[t].lut = &lut;
        params[t].local_p = top_p;
        params[t].next_task = &next_task;
        params[t].task_mutex = &task_mutex;

        pthread_create(&threads[t], nullptr, ivfpq_refine_dynamic_thread_func, &params[t]);
    }

    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (int t = 0; t < thread_num; ++t) {
        pthread_join(threads[t], nullptr);

        while (!params[t].local_heap.empty()) {
            auto item = params[t].local_heap.top();
            params[t].local_heap.pop();
            ivfpq_push_topk(candidates, item.first, item.second, top_p);
        }
    }

    pthread_mutex_destroy(&task_mutex);

    return ivfpq_rerank_flat(index, query, candidates, k);
}

struct IVFPQBatchQueryPthreadParam {
    int tid = 0;

    const IVFPQIndex* index = nullptr;
    const float* queries = nullptr;

    size_t query_number = 0;
    size_t vecdim = 0;
    size_t k = 0;
    size_t top_p = 0;
    size_t nprobe = 0;

    size_t query_chunk_size = 1;

    size_t* next_query = nullptr;
    pthread_mutex_t* task_mutex = nullptr;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>>* cache_results = nullptr;

    size_t processed_queries = 0;
};

static void* ivfpq_batch_query_pthread_func(void* arg) {
    IVFPQBatchQueryPthreadParam* param =
        static_cast<IVFPQBatchQueryPthreadParam*>(arg);

    const IVFPQIndex& index = *(param->index);
    const float* queries = param->queries;

    while (true) {
        size_t begin_q = 0;
        size_t end_q = 0;

        pthread_mutex_lock(param->task_mutex);

        begin_q = *(param->next_query);
        *(param->next_query) += param->query_chunk_size;

        pthread_mutex_unlock(param->task_mutex);

        if (begin_q >= param->query_number) {
            break;
        }

        end_q = std::min(
            begin_q + param->query_chunk_size,
            param->query_number
        );

        for (size_t qi = begin_q; qi < end_q; ++qi) {
            const float* query =
                queries + qi * param->vecdim;

            // 每个 query 内部使用串行 IVF-PQ-SIMD。
            // 不要在这里调用 pthread refine 版本，否则会变成嵌套并行。
            (*param->cache_results)[qi] =
                ivfpq_search_simd(
                    index,
                    query,
                    param->k,
                    param->top_p,
                    param->nprobe
                );

            param->processed_queries++;
        }
    }

    return nullptr;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_batch_query_pthread_cached_optimized(
    const IVFPQIndex& index,
    const float* queries,
    size_t query_id,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    size_t query_chunk_size = 1
) {
    assert(index.trained);
    assert(queries != nullptr);
    assert(query_id < query_number);
    assert(query_number > 0);
    assert(vecdim == index.vecdim);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (top_p < k) {
        top_p = k;
    }

    if (thread_num > static_cast<int>(query_number)) {
        thread_num = static_cast<int>(query_number);
    }

    if (thread_num <= 1) {
        const float* query = queries + query_id * vecdim;
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    if (query_chunk_size == 0) {
        query_chunk_size = 1;
    }

    static std::vector<std::priority_queue<std::pair<float, uint32_t>>> cache_results;

    static bool cache_ready = false;
    static const IVFPQIndex* cached_index = nullptr;
    static const float* cached_queries = nullptr;
    static size_t cached_query_number = 0;
    static size_t cached_vecdim = 0;
    static size_t cached_k = 0;
    static size_t cached_top_p = 0;
    static size_t cached_nprobe = 0;

    bool need_rebuild =
        !cache_ready ||
        cached_index != &index ||
        cached_queries != queries ||
        cached_query_number != query_number ||
        cached_vecdim != vecdim ||
        cached_k != k ||
        cached_top_p != top_p ||
        cached_nprobe != nprobe;

    if (need_rebuild) {
        cache_results.clear();
        cache_results.resize(query_number);

        size_t next_query = 0;
        pthread_mutex_t task_mutex;
        pthread_mutex_init(&task_mutex, nullptr);

        std::vector<pthread_t> threads(static_cast<size_t>(thread_num));
        std::vector<IVFPQBatchQueryPthreadParam> params(
            static_cast<size_t>(thread_num)
        );

        for (int t = 0; t < thread_num; ++t) {
            params[t].tid = t;
            params[t].index = &index;
            params[t].queries = queries;
            params[t].query_number = query_number;
            params[t].vecdim = vecdim;
            params[t].k = k;
            params[t].top_p = top_p;
            params[t].nprobe = nprobe;
            params[t].query_chunk_size = query_chunk_size;
            params[t].next_query = &next_query;
            params[t].task_mutex = &task_mutex;
            params[t].cache_results = &cache_results;

            pthread_create(
                &threads[static_cast<size_t>(t)],
                nullptr,
                ivfpq_batch_query_pthread_func,
                &params[static_cast<size_t>(t)]
            );
        }

        for (int t = 0; t < thread_num; ++t) {
            pthread_join(threads[static_cast<size_t>(t)], nullptr);
        }

        pthread_mutex_destroy(&task_mutex);

        cached_index = &index;
        cached_queries = queries;
        cached_query_number = query_number;
        cached_vecdim = vecdim;
        cached_k = k;
        cached_top_p = top_p;
        cached_nprobe = nprobe;
        cache_ready = true;
    }

    return cache_results[query_id];
}