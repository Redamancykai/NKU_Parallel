#pragma once
#include <pthread.h>
#include <queue>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include "flat_simd.h"

struct FlatThreadParam {
    int tid;                    // 多线程编号，0 表示主线程，1~7 表示 pthread 子线程

    const float* base;          // base 向量首地址
    const float* query;         // 当前 query 向量首地址

    size_t base_number;         // base 向量数量
    size_t vecdim;              // 向量维度

    size_t start_id;            // 当前线程负责的 base 起始编号
    size_t end_id;              // 当前线程负责的 base 结束编号，左闭右开 [start_id, end_id)

    size_t local_p;             // top-p 参数

    // 每个线程维护自己的局部 top-p 堆。
    // pair.first  = distance
    // pair.second = base id
    // priority_queue 默认是大根堆，因此堆顶是当前 top-p 中距离最大的点。
    std::priority_queue<std::pair<float, uint32_t>>* local_heap;
};

// 维护最大堆，向 top-k 堆中插入一个候选点
static inline void flat_push_topk(
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

// 线程计算函数
static inline void flat_worker_compute(FlatThreadParam* param) {
    const float* base = param->base;
    const float* query = param->query;
    const size_t vecdim = param->vecdim;
    const size_t local_p = param->local_p;

    auto& heap = *(param->local_heap);

    // 计算内积并维护局部 top-p 堆
    for (size_t i = param->start_id; i < param->end_id; ++i) {
        const float* base_vec = base + i * vecdim;
        float dot = InnerProductSIMD(base_vec, query, vecdim);
        float dis = 1.0f - dot;
        flat_push_topk(heap, dis, static_cast<uint32_t>(i), local_p);
    }
}

// pthread 子线程入口函数
static void* flat_pthread_worker(void* arg) {
    FlatThreadParam* param = reinterpret_cast<FlatThreadParam*>(arg);
    flat_worker_compute(param);
    return nullptr;
}

// 采用 7 个 pthread 子线程 + 主线程参与计算，共 8 个逻辑计算线程。
static inline std::priority_queue<std::pair<float, uint32_t>>
flat_search_pthread_simd_8threads(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t local_p
) {
    assert(base != nullptr);
    assert(query != nullptr);
    assert(base_number > 0);
    assert(vecdim > 0);
    assert(k > 0);

    constexpr int TOTAL_THREADS = 8;   // 总线程数
    constexpr int CHILD_THREADS = 7;   // 子线程数

    int actual_threads = TOTAL_THREADS;

    int actual_child_threads = actual_threads - 1;

    std::vector<pthread_t> threads(actual_child_threads);
    std::vector<FlatThreadParam> params(actual_threads);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps(actual_threads);

    // 连续静态分块：
    // 第 t 个逻辑线程负责 [t * block, min((t + 1) * block, base_number))
    size_t block = (base_number + actual_threads - 1) / actual_threads;

    for (int t = 0; t < actual_threads; ++t) {
        size_t start = static_cast<size_t>(t) * block;
        size_t end = std::min(start + block, base_number);

        params[t].tid = t;
        params[t].base = base;
        params[t].query = query;
        params[t].base_number = base_number;
        params[t].vecdim = vecdim;
        params[t].start_id = start;
        params[t].end_id = end;
        params[t].local_p = local_p;
        params[t].local_heap = &local_heaps[t];
    }

    // 创建子线程
    for (int t = 1; t < actual_threads; ++t) {
        pthread_create(&threads[t - 1], nullptr, flat_pthread_worker, &params[t]);
    }

    // 主线程作为线程 0
    flat_worker_compute(&params[0]);

    // 等待所有子线程完成
    for (int t = 1; t < actual_threads; ++t) {
        pthread_join(threads[t - 1], nullptr);
    }

    // 合并所有线程的局部 top-k。
    std::priority_queue<std::pair<float, uint32_t>> global_heap;

    for (int t = 0; t < actual_threads; ++t) {
        auto& local_heap = local_heaps[t];

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();

            flat_push_topk(global_heap, item.first, item.second, k);
        }
    }

    return global_heap;
}