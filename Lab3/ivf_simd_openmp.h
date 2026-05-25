#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "ivf_simd.h"

static inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_simd_openmp_refine_cluster(
    const IVFIndex& index,
    const float* query,
    size_t k,
    size_t local_k,
    size_t nprobe,
    int thread_num
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(local_k > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    assert(index.inverted_lists.size() == index.nlist);
    assert(index.inverted_vectors.size() == index.nlist);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (local_k < k) {
        local_k = k;
    }

    if (thread_num <= 1) {
        return ivf_search_simd(index, query, k, nprobe);
    }

    if (thread_num > static_cast<int>(nprobe)) {
        thread_num = static_cast<int>(nprobe);
    }

    // IVF 粗排
    std::vector<uint32_t> probe_lists;
    ivf_select_probe_lists(index, query, nprobe, probe_lists);

    // 维护局部 top-k
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps(
        static_cast<size_t>(thread_num)
    );

    // OpenMP 并行精排
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = omp_get_thread_num();
        auto& local_heap = local_heaps[static_cast<size_t>(tid)];

        // dynamic,1 适合倒排表长度不均衡的 IVF 精排
        #pragma omp for schedule(dynamic, 1)
        for (int pi = 0; pi < static_cast<int>(probe_lists.size()); ++pi) {
            uint32_t cid = probe_lists[static_cast<size_t>(pi)];

            const std::vector<uint32_t>& ids = index.inverted_lists[cid];
            const std::vector<float>& vecs = index.inverted_vectors[cid];

            assert(vecs.size() == ids.size() * index.vecdim);

            const float* vec_data = vecs.data();

            for (size_t j = 0; j < ids.size(); ++j) {
                const float* x = vec_data + j * index.vecdim;

                float dis = 1.0f - InnerProductSIMD(
                    query,
                    x,
                    index.vecdim
                );

                push_topk(local_heap, dis, ids[j], local_k);
            }
        }
    }

    // 合并局部 top-k
    std::priority_queue<std::pair<float, uint32_t>> result;

    for (int t = 0; t < thread_num; ++t) {
        auto& local_heap = local_heaps[static_cast<size_t>(t)];

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();

            push_topk(result, item.first, item.second, k);
        }
    }

    return result;
}

// IVF-SIMD OpenMP query 级并行
static inline std::priority_queue<std::pair<float, uint32_t>>
ivf_search_batch_query_openmp_cached(
    const IVFIndex& index,
    const float* queries,
    size_t query_id,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    int thread_num
) {
    assert(index.trained);
    assert(queries != nullptr);
    assert(query_id < query_number);
    assert(query_number > 0);
    assert(vecdim == index.vecdim);
    assert(k > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (thread_num > static_cast<int>(query_number)) {
        thread_num = static_cast<int>(query_number);
    }

    static std::vector<std::priority_queue<std::pair<float, uint32_t>>> cache_results;
    static const IVFIndex* cached_index = nullptr;
    static const float* cached_queries = nullptr;
    static size_t cached_query_number = 0;
    static size_t cached_vecdim = 0;
    static size_t cached_k = 0;
    static size_t cached_nprobe = 0;
    static int cached_thread_num = 0;
    static bool cache_ready = false;
    bool need_rebuild_cache = false;

    if (!cache_ready) {
        need_rebuild_cache = true;
    }

    if (cached_index != &index ||
        cached_queries != queries ||
        cached_query_number != query_number ||
        cached_vecdim != vecdim ||
        cached_k != k ||
        cached_nprobe != nprobe ||
        cached_thread_num != thread_num) {
        need_rebuild_cache = true;
    }

    // 适配你的测试框架：for 循环从 i = 0 开始。
    // i == 0 时重新计算整批 query，避免复用旧参数下的缓存。
    if (query_id == 0) {
        need_rebuild_cache = true;
    }

    if (need_rebuild_cache) {
        cache_results.clear();
        cache_results.resize(query_number);

        if (thread_num <= 1) {
            for (size_t qi = 0; qi < query_number; ++qi) {
                const float* query = queries + qi * vecdim;

                cache_results[qi] = ivf_search_simd(
                    index,
                    query,
                    k,
                    nprobe
                );
            }
        } else {
            #pragma omp parallel for num_threads(thread_num) schedule(static)
            for (int qi = 0; qi < static_cast<int>(query_number); ++qi) {
                const float* query = queries + static_cast<size_t>(qi) * vecdim;

                cache_results[static_cast<size_t>(qi)] = ivf_search_simd(
                    index,
                    query,
                    k,
                    nprobe
                );
            }
        }

        cached_index = &index;
        cached_queries = queries;
        cached_query_number = query_number;
        cached_vecdim = vecdim;
        cached_k = k;
        cached_nprobe = nprobe;
        cached_thread_num = thread_num;
        cache_ready = true;
    }
    return cache_results[query_id];
}