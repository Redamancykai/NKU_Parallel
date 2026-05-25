#pragma once

#include <omp.h>
#include <queue>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cassert>

#include "flat_simd.h"

// 维护 top-k 最大堆
static inline void flat_omp_push_topk(
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

// Flat-SIMD + OpenMP
// base 向量划分并行
static inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_openmp_simd(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t local_p,
    int num_threads = 8
) {
    assert(base != nullptr);
    assert(query != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(vecdim % 32 == 0);
    assert(k > 0);
    assert(local_p > 0);

    if (num_threads <= 0) {
        num_threads = 1;
    }

    if (static_cast<size_t>(num_threads) > base_number) {
        num_threads = static_cast<int>(base_number);
    }

    // 每个 OpenMP 线程维护一个局部 top-p 堆
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps(num_threads);

    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_heap = local_heaps[tid];

        // base 向量划分并行
        // schedule(static) 表示静态连续/近似连续划分任务
#pragma omp for schedule(static)
        for (size_t i = 0; i < base_number; ++i) {
            const float* base_vec = base + i * vecdim;

            float dot = InnerProductSIMD(base_vec, query, vecdim);
            float dis = 1.0f - dot;

            flat_omp_push_topk(
                local_heap,
                dis,
                static_cast<uint32_t>(i),
                local_p
            );
        }
    }

    // reduce：合并每个线程的 local top-p，得到 global top-k
    std::priority_queue<std::pair<float, uint32_t>> global_heap;

    for (int t = 0; t < num_threads; ++t) {
        auto& local_heap = local_heaps[t];

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();

            flat_omp_push_topk(
                global_heap,
                item.first,
                item.second,
                k
            );
        }
    }

    return global_heap;
}