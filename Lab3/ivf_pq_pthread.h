#pragma once

#include <pthread.h>
#include <atomic>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>
#include <limits>

#include "ivf_pq_simd.h"

// IVF-PQ Pthread 实现了 centroid partition 并行和 cluster partition 并行
struct IVFPQCentroidPthreadParam {
    int tid = 0;
    int thread_num = 1;

    const IVFPQIndex* index = nullptr;
    const float* query = nullptr;

    size_t nprobe = 0;

    std::priority_queue<std::pair<float, uint32_t>> local_heap;
};

struct IVFPQClusterPthreadParam {
    int tid = 0;

    const IVFPQIndex* index = nullptr;
    const std::vector<uint32_t>* probe_lists = nullptr;
    const std::vector<float>* lut = nullptr;

    std::atomic<size_t>* next_task = nullptr;

    size_t local_p = 0;

    std::priority_queue<std::pair<float, uint32_t>> local_heap;

    size_t scanned_lists = 0;
    size_t scanned_points = 0;
};

static inline void ivfpq_pthread_push_topk(
    std::priority_queue<std::pair<float, uint32_t>>& heap,
    float dis,
    uint32_t id,
    size_t k
) {
    if (k == 0) return;

    if (heap.size() < k) {
        heap.push(std::make_pair(dis, id));
    } else if (dis < heap.top().first) {
        heap.pop();
        heap.push(std::make_pair(dis, id));
    }
}

// IVF centroid partition 并行
static void* ivfpq_centroid_pthread_func(void* arg) {
    IVFPQCentroidPthreadParam* param =
        static_cast<IVFPQCentroidPthreadParam*>(arg);

    const IVFPQIndex& index = *(param->index);
    const float* query = param->query;

    const size_t nlist = index.nlist;
    const size_t vecdim = index.vecdim;
    const size_t nprobe = param->nprobe;

    const int tid = param->tid;
    const int thread_num = param->thread_num;

    const size_t start = nlist * static_cast<size_t>(tid)
                       / static_cast<size_t>(thread_num);
    const size_t end = nlist * static_cast<size_t>(tid + 1)
                     / static_cast<size_t>(thread_num);

    for (size_t c = start; c < end; ++c) {
        const float* centroid = index.centroids.data() + c * vecdim;

        float dis = 1.0f - InnerProductSIMD(query, centroid, vecdim);

        ivfpq_pthread_push_topk(
            param->local_heap,
            dis,
            static_cast<uint32_t>(c),
            nprobe
        );
    }

    return nullptr;
}

static inline void ivfpq_select_probe_lists_pthread_centroid(
    const IVFPQIndex& index,
    const float* query,
    size_t nprobe,
    int thread_num,
    std::vector<uint32_t>& probe_lists
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(nprobe > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (thread_num <= 1 || index.nlist <= 1) {
        ivfpq_select_probe_lists(index, query, nprobe, probe_lists);
        return;
    }

    if (thread_num > static_cast<int>(index.nlist)) {
        thread_num = static_cast<int>(index.nlist);
    }

    std::vector<pthread_t> threads(static_cast<size_t>(thread_num));
    std::vector<IVFPQCentroidPthreadParam> params(static_cast<size_t>(thread_num));

    for (int t = 0; t < thread_num; ++t) {
        params[t].tid = t;
        params[t].thread_num = thread_num;
        params[t].index = &index;
        params[t].query = query;
        params[t].nprobe = nprobe;

        pthread_create(
            &threads[static_cast<size_t>(t)],
            nullptr,
            ivfpq_centroid_pthread_func,
            &params[static_cast<size_t>(t)]
        );
    }

    std::priority_queue<std::pair<float, uint32_t>> global_heap;

    for (int t = 0; t < thread_num; ++t) {
        pthread_join(threads[static_cast<size_t>(t)], nullptr);

        while (!params[static_cast<size_t>(t)].local_heap.empty()) {
            auto item = params[static_cast<size_t>(t)].local_heap.top();
            params[static_cast<size_t>(t)].local_heap.pop();

            ivfpq_pthread_push_topk(
                global_heap,
                item.first,
                item.second,
                nprobe
            );
        }
    }

    probe_lists.clear();
    probe_lists.reserve(nprobe);

    while (!global_heap.empty()) {
        probe_lists.push_back(global_heap.top().second);
        global_heap.pop();
    }

    std::reverse(probe_lists.begin(), probe_lists.end());
}

// ADC 距离计算
static inline float ivfpq_adc_distance_inverted_code_pthread(
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

// selected cluster partition 并行
static void* ivfpq_cluster_pthread_dynamic_func(void* arg) {
    IVFPQClusterPthreadParam* param =
        static_cast<IVFPQClusterPthreadParam*>(arg);

    const IVFPQIndex& index = *(param->index);
    const std::vector<uint32_t>& probe_lists = *(param->probe_lists);
    const std::vector<float>& lut = *(param->lut);

    while (true) {
        size_t task_id = param->next_task->fetch_add(1);

        if (task_id >= probe_lists.size()) {
            break;
        }

        uint32_t cid = probe_lists[task_id];

        const std::vector<uint32_t>& ids = index.inverted_lists[cid];
        const std::vector<uint8_t>& codes_in_list = index.inverted_codes[cid];

        param->scanned_lists++;
        param->scanned_points += ids.size();

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            const uint8_t* code = codes_in_list.data() + pos * index.M;

            float dis = ivfpq_adc_distance_inverted_code_pthread(
                index,
                code,
                lut
            );

            ivfpq_pthread_push_topk(
                param->local_heap,
                dis,
                ids[pos],
                param->local_p
            );
        }
    }

    return nullptr;
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_pthread_centroid_cluster(
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

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    // centroid partition 并行粗排
    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists_pthread_centroid(
        index,
        query,
        nprobe,
        thread_num,
        probe_lists
    );

    // 构建 PQ LUT
    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    // cluster partition 并行 ADC 扫描
    int real_thread_num = thread_num;
    if (real_thread_num > static_cast<int>(probe_lists.size())) {
        real_thread_num = static_cast<int>(probe_lists.size());
    }
    if (real_thread_num <= 1) {
        std::priority_queue<std::pair<float, uint32_t>> candidates;

        for (uint32_t cid : probe_lists) {
            const std::vector<uint32_t>& ids = index.inverted_lists[cid];
            const std::vector<uint8_t>& codes_in_list = index.inverted_codes[cid];

            for (size_t pos = 0; pos < ids.size(); ++pos) {
                const uint8_t* code = codes_in_list.data() + pos * index.M;
                float dis = ivfpq_adc_distance_inverted_code_pthread(index, code, lut);
                ivfpq_pthread_push_topk(candidates, dis, ids[pos], top_p);
            }
        }

        return ivfpq_rerank_flat(index, query, candidates, k);
    }

    std::atomic<size_t> next_task(0);

    std::vector<pthread_t> threads(static_cast<size_t>(real_thread_num));
    std::vector<IVFPQClusterPthreadParam> params(
        static_cast<size_t>(real_thread_num)
    );

    for (int t = 0; t < real_thread_num; ++t) {
        params[static_cast<size_t>(t)].tid = t;
        params[static_cast<size_t>(t)].index = &index;
        params[static_cast<size_t>(t)].probe_lists = &probe_lists;
        params[static_cast<size_t>(t)].lut = &lut;
        params[static_cast<size_t>(t)].next_task = &next_task;
        params[static_cast<size_t>(t)].local_p = top_p;

        pthread_create(
            &threads[static_cast<size_t>(t)],
            nullptr,
            ivfpq_cluster_pthread_dynamic_func,
            &params[static_cast<size_t>(t)]
        );
    }

    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (int t = 0; t < real_thread_num; ++t) {
        pthread_join(threads[static_cast<size_t>(t)], nullptr);

        while (!params[static_cast<size_t>(t)].local_heap.empty()) {
            auto item = params[static_cast<size_t>(t)].local_heap.top();
            params[static_cast<size_t>(t)].local_heap.pop();

            ivfpq_pthread_push_topk(
                candidates,
                item.first,
                item.second,
                top_p
            );
        }
    }

    // Flat-SIMD rerank
    return ivfpq_rerank_flat(index, query, candidates, k);
}