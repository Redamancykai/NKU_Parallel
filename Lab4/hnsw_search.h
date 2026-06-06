#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#include <pthread.h>

#include "hnswlib/hnswlib/hnswlib.h"

struct HNSWIndex {
    size_t base_number = 0;
    size_t vecdim = 0;
    size_t M = 16;
    size_t ef_construction = 150;
    size_t random_seed = 100;
    bool trained = false;

    std::unique_ptr<hnswlib::InnerProductSpace> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> graph;

    bool load_index(
        const std::string& path,
        size_t expected_base_number,
        size_t expected_vecdim,
        size_t max_elements = 0
    ) {
        try {
            space.reset(new hnswlib::InnerProductSpace(expected_vecdim));
            graph.reset(new hnswlib::HierarchicalNSW<float>(
                space.get(),
                path,
                false,
                max_elements == 0 ? expected_base_number : max_elements
            ));

            if (graph->getCurrentElementCount() != expected_base_number) {
                graph.reset();
                space.reset();
                trained = false;
                return false;
            }

            base_number = expected_base_number;
            vecdim = expected_vecdim;
            M = graph->M_;
            ef_construction = graph->ef_construction_;
            trained = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[HNSW] Load failed: " << e.what() << std::endl;
            graph.reset();
            space.reset();
            trained = false;
            return false;
        }
    }

    bool save_index(const std::string& path) const {
        if (!trained || !graph) {
            return false;
        }

        try {
            graph->saveIndex(path);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[HNSW] Save failed: " << e.what() << std::endl;
            return false;
        }
    }
};

static inline void hnsw_push_topk(
    std::priority_queue<std::pair<float, uint32_t>>& heap,
    float dis,
    uint32_t id,
    size_t k
) {
    if (heap.size() < k) {
        heap.emplace(dis, id);
    } else if (dis < heap.top().first) {
        heap.pop();
        heap.emplace(dis, id);
    }
}

static inline void hnsw_merge_topk(
    std::priority_queue<std::pair<float, uint32_t>>& dst,
    std::priority_queue<std::pair<float, uint32_t>>& src,
    size_t k
) {
    while (!src.empty()) {
        const auto item = src.top();
        src.pop();
        hnsw_push_topk(dst, item.first, item.second, k);
    }
}

static inline uint64_t hnsw_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static inline uint32_t hnsw_pick_entry_by_query(
    const HNSWIndex& index,
    const float* query,
    size_t salt = 0
) {
    assert(index.trained);
    assert(query != nullptr);

    const size_t n = index.graph->getCurrentElementCount();
    if (n == 0) {
        return 0;
    }

    uint64_t h = 1469598103934665603ULL ^ (salt + 0x9e3779b97f4a7c15ULL);
    const size_t sample_dim = std::min(index.vecdim, static_cast<size_t>(16));
    for (size_t i = 0; i < sample_dim; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, query + i, sizeof(uint32_t));
        h ^= bits;
        h *= 1099511628211ULL;
    }

    return static_cast<uint32_t>(hnsw_mix64(h) % n);
}

static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_convert_internal_result(
    const HNSWIndex& index,
    std::priority_queue<
        std::pair<float, hnswlib::tableint>,
        std::vector<std::pair<float, hnswlib::tableint>>,
        hnswlib::HierarchicalNSW<float>::CompareByFirst
    >& internal_heap,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t>> result;

    while (!internal_heap.empty()) {
        const auto item = internal_heap.top();
        internal_heap.pop();

        const hnswlib::labeltype label =
            index.graph->getExternalLabel(item.second);

        hnsw_push_topk(
            result,
            item.first,
            static_cast<uint32_t>(label),
            k
        );
    }

    return result;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_convert_label_result(
    std::priority_queue<std::pair<float, hnswlib::labeltype>>& label_heap,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t>> result;

    while (!label_heap.empty()) {
        const auto item = label_heap.top();
        label_heap.pop();
        hnsw_push_topk(
            result,
            item.first,
            static_cast<uint32_t>(item.second),
            k
        );
    }

    return result;
}

static inline bool hnsw_load(
    HNSWIndex& index,
    size_t base_number,
    size_t vecdim,
    const std::string& index_path = "files/hnsw.index"
) {
    assert(base_number > 0);
    assert(vecdim > 0);

    if (index.load_index(index_path, base_number, vecdim)) {
        std::cerr << "[HNSW] Loaded saved index: " << index_path << std::endl;
        return true;
    }

    std::cerr << "[HNSW] Cannot load index: " << index_path
              << ". Please call build_index(base, base_number, vecdim) first."
              << std::endl;
    return false;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_hierarchical(
    const HNSWIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64
) {
    assert(index.trained);
    assert(index.graph != nullptr);
    assert(query != nullptr);
    assert(k > 0);

    index.graph->setEf(std::max(k, ef));
    auto raw = index.graph->searchKnn(query, k);
    return hnsw_convert_label_result(raw, k);
}

// Search only the bottom graph from a single entry point.
// If entry_id is negative, a deterministic pseudo-random entry is derived from
// the query. This makes repeated runs reproducible while still avoiding the
// upper HNSW layers.
static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_layer0(
    const HNSWIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64,
    int entry_id = -1
) {
    assert(index.trained);
    assert(index.graph != nullptr);
    assert(query != nullptr);
    assert(k > 0);

    const size_t real_ef = std::max(k, ef);
    uint32_t entry = 0;
    if (entry_id >= 0) {
        entry = static_cast<uint32_t>(
            static_cast<size_t>(entry_id) %
            index.graph->getCurrentElementCount()
        );
    } else {
        entry = hnsw_pick_entry_by_query(index, query);
    }

    auto raw = index.graph->searchBaseLayerST<true>(
        static_cast<hnswlib::tableint>(entry),
        query,
        real_ef
    );

    return hnsw_convert_internal_result(index, raw, k);
}

static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_layer0_multi_entry_openmp(
    const HNSWIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64,
    int entry_count = 4,
    int thread_num = 4
) {
    assert(index.trained);
    assert(index.graph != nullptr);
    assert(query != nullptr);
    assert(k > 0);

    const size_t n = index.graph->getCurrentElementCount();
    if (n == 0) {
        return {};
    }

    if (entry_count <= 1 || thread_num <= 1) {
        return hnsw_search_layer0(index, query, k, ef);
    }

    entry_count = std::min<int>(entry_count, static_cast<int>(n));
    thread_num = std::min(thread_num, entry_count);

    const size_t real_ef = std::max(k, ef);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(
        static_cast<size_t>(entry_count)
    );

#pragma omp parallel for num_threads(thread_num) schedule(static)
    for (int i = 0; i < entry_count; ++i) {
        const uint32_t entry =
            hnsw_pick_entry_by_query(index, query, static_cast<size_t>(i));

        auto raw = index.graph->searchBaseLayerST<true>(
            static_cast<hnswlib::tableint>(entry),
            query,
            real_ef
        );

        local_results[static_cast<size_t>(i)] =
            hnsw_convert_internal_result(index, raw, k);
    }

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (int i = 0; i < entry_count; ++i) {
        hnsw_merge_topk(result, local_results[static_cast<size_t>(i)], k);
    }

    return result;
}

struct HNSWMultiEntryPthreadParam {
    int tid = 0;

    const HNSWIndex* index = nullptr;
    const float* query = nullptr;

    size_t k = 0;
    size_t ef = 0;

    int entry_start = 0;
    int entry_end = 0;

    std::priority_queue<std::pair<float, uint32_t>>* local_result = nullptr;
};

static inline void hnsw_multi_entry_pthread_compute(
    HNSWMultiEntryPthreadParam* param
) {
    const HNSWIndex& index = *(param->index);
    auto& local_result = *(param->local_result);

    for (int i = param->entry_start; i < param->entry_end; ++i) {
        const uint32_t entry =
            hnsw_pick_entry_by_query(index, param->query, static_cast<size_t>(i));

        auto raw = index.graph->searchBaseLayerST<true>(
            static_cast<hnswlib::tableint>(entry),
            param->query,
            param->ef
        );

        auto one_entry_result =
            hnsw_convert_internal_result(index, raw, param->k);

        hnsw_merge_topk(local_result, one_entry_result, param->k);
    }
}

static void* hnsw_multi_entry_pthread_worker(void* arg) {
    HNSWMultiEntryPthreadParam* param =
        reinterpret_cast<HNSWMultiEntryPthreadParam*>(arg);
    hnsw_multi_entry_pthread_compute(param);
    return nullptr;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
hnsw_search_layer0_multi_entry_pthread(
    const HNSWIndex& index,
    const float* query,
    size_t k,
    size_t ef = 64,
    int entry_count = 4,
    int thread_num = 4
) {
    assert(index.trained);
    assert(index.graph != nullptr);
    assert(query != nullptr);
    assert(k > 0);

    const size_t n = index.graph->getCurrentElementCount();
    if (n == 0) {
        return {};
    }

    if (entry_count <= 1 || thread_num <= 1) {
        return hnsw_search_layer0(index, query, k, ef);
    }

    entry_count = std::min<int>(entry_count, static_cast<int>(n));
    thread_num = std::min(thread_num, entry_count);

    const size_t real_ef = std::max(k, ef);
    const int child_threads = thread_num - 1;

    std::vector<pthread_t> threads(static_cast<size_t>(child_threads));
    std::vector<HNSWMultiEntryPthreadParam> params(
        static_cast<size_t>(thread_num)
    );
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_results(
        static_cast<size_t>(thread_num)
    );

    const int block = (entry_count + thread_num - 1) / thread_num;

    for (int t = 0; t < thread_num; ++t) {
        const int start = t * block;
        const int end = std::min(start + block, entry_count);

        params[static_cast<size_t>(t)].tid = t;
        params[static_cast<size_t>(t)].index = &index;
        params[static_cast<size_t>(t)].query = query;
        params[static_cast<size_t>(t)].k = k;
        params[static_cast<size_t>(t)].ef = real_ef;
        params[static_cast<size_t>(t)].entry_start = start;
        params[static_cast<size_t>(t)].entry_end = end;
        params[static_cast<size_t>(t)].local_result =
            &local_results[static_cast<size_t>(t)];
    }

    for (int t = 1; t < thread_num; ++t) {
        pthread_create(
            &threads[static_cast<size_t>(t - 1)],
            nullptr,
            hnsw_multi_entry_pthread_worker,
            &params[static_cast<size_t>(t)]
        );
    }

    hnsw_multi_entry_pthread_compute(&params[0]);

    for (int t = 1; t < thread_num; ++t) {
        pthread_join(threads[static_cast<size_t>(t - 1)], nullptr);
    }

    std::priority_queue<std::pair<float, uint32_t>> result;
    for (int t = 0; t < thread_num; ++t) {
        hnsw_merge_topk(result, local_results[static_cast<size_t>(t)], k);
    }

    return result;
}
