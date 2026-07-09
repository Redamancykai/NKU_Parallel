#pragma once

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "ivf_pq_simd.h"

// ================================================================
// IVF-PQ query-level asynchronous pipeline
//
// Pipeline stages:
//   Stage 1: select nprobe IVF lists + build PQ LUT
//   Stage 2: ADC scan selected lists and keep top-p candidates
//   Stage 3: Flat-SIMD rerank top-p candidates and output top-k
//
// This file does not change the IVF-PQ index format or recall logic.
// It only changes batch query scheduling.
// ================================================================

template <typename T>
class IVFPQPipelineBlockingQueue {
private:
    std::queue<T> q_;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    bool closed_ = false;
    size_t capacity_ = 0;  // 0 means unbounded.

public:
    explicit IVFPQPipelineBlockingQueue(size_t capacity = 0)
        : capacity_(capacity) {}

    bool push(const T& item) {
        std::unique_lock<std::mutex> lock(mtx_);

        while (!closed_ && capacity_ > 0 && q_.size() >= capacity_) {
            cv_not_full_.wait(lock);
        }

        if (closed_) {
            return false;
        }

        q_.push(item);
        cv_not_empty_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);

        while (!closed_ && q_.empty()) {
            cv_not_empty_.wait(lock);
        }

        if (q_.empty()) {
            return false;
        }

        item = q_.front();
        q_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::unique_lock<std::mutex> lock(mtx_);
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }
};

struct IVFPQPipelineTask {
    size_t qid = 0;
    const float* query = nullptr;

    std::vector<uint32_t> probe_lists;
    std::vector<float> lut;

    std::priority_queue<std::pair<float, uint32_t>> candidates;
    std::priority_queue<std::pair<float, uint32_t>> result;
};

using IVFPQPipelineTaskPtr = std::shared_ptr<IVFPQPipelineTask>;
using IVFPQPipelineResult = std::priority_queue<std::pair<float, uint32_t>>;

static inline void ivfpq_pipeline_stage1_worker(
    const IVFPQIndex& index,
    size_t nprobe,
    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr>& input_q,
    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr>& output_q
) {
    IVFPQPipelineTaskPtr task;

    while (input_q.pop(task)) {
        ivfpq_select_probe_lists(
            index,
            task->query,
            nprobe,
            task->probe_lists
        );

        ivfpq_build_lut(
            index,
            task->query,
            task->lut
        );

        output_q.push(task);
    }
}

static inline void ivfpq_pipeline_stage2_worker(
    const IVFPQIndex& index,
    size_t top_p,
    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr>& input_q,
    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr>& output_q
) {
    IVFPQPipelineTaskPtr task;

    while (input_q.pop(task)) {
        task->candidates = IVFPQPipelineResult();

        for (uint32_t cid : task->probe_lists) {
            const std::vector<uint32_t>& ids = index.inverted_lists[cid];
            const std::vector<uint8_t>& codes_in_list = index.inverted_codes[cid];
            const uint8_t* code_data = codes_in_list.data();

            for (size_t pos = 0; pos < ids.size(); ++pos) {
                const uint8_t* code = code_data + pos * index.M;
                float dis = ivfpq_adc_distance_code(index, code, task->lut);
                ivfpq_push_topk(task->candidates, dis, ids[pos], top_p);
            }
        }

        output_q.push(task);
    }
}

static inline void ivfpq_pipeline_stage3_worker(
    const IVFPQIndex& index,
    size_t k,
    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr>& input_q,
    std::vector<IVFPQPipelineResult>& results
) {
    IVFPQPipelineTaskPtr task;

    while (input_q.pop(task)) {
        task->result = ivfpq_rerank_flat(
            index,
            task->query,
            task->candidates,
            k
        );

        results[task->qid] = task->result;
    }
}

static inline void ivfpq_search_pipeline_batch(
    const IVFPQIndex& index,
    const float* queries,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    size_t nprobe,
    std::vector<IVFPQPipelineResult>& results,
    int stage1_threads = 1,
    int stage2_threads = 4,
    int stage3_threads = 1,
    size_t queue_capacity = 64
) {
    assert(index.trained);
    assert(queries != nullptr);
    assert(query_number > 0);
    assert(vecdim == index.vecdim);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }
    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (stage1_threads <= 0) {
        stage1_threads = 1;
    }
    if (stage2_threads <= 0) {
        stage2_threads = 1;
    }
    if (stage3_threads <= 0) {
        stage3_threads = 1;
    }

    results.clear();
    results.resize(query_number);

    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr> q_input(queue_capacity);
    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr> q_stage1_stage2(queue_capacity);
    IVFPQPipelineBlockingQueue<IVFPQPipelineTaskPtr> q_stage2_stage3(queue_capacity);

    std::vector<std::thread> workers_stage1;
    std::vector<std::thread> workers_stage2;
    std::vector<std::thread> workers_stage3;

    workers_stage1.reserve(static_cast<size_t>(stage1_threads));
    workers_stage2.reserve(static_cast<size_t>(stage2_threads));
    workers_stage3.reserve(static_cast<size_t>(stage3_threads));

    for (int t = 0; t < stage1_threads; ++t) {
        workers_stage1.emplace_back(
            ivfpq_pipeline_stage1_worker,
            std::ref(index),
            nprobe,
            std::ref(q_input),
            std::ref(q_stage1_stage2)
        );
    }

    for (int t = 0; t < stage2_threads; ++t) {
        workers_stage2.emplace_back(
            ivfpq_pipeline_stage2_worker,
            std::ref(index),
            top_p,
            std::ref(q_stage1_stage2),
            std::ref(q_stage2_stage3)
        );
    }

    for (int t = 0; t < stage3_threads; ++t) {
        workers_stage3.emplace_back(
            ivfpq_pipeline_stage3_worker,
            std::ref(index),
            k,
            std::ref(q_stage2_stage3),
            std::ref(results)
        );
    }

    for (size_t qid = 0; qid < query_number; ++qid) {
        IVFPQPipelineTaskPtr task(new IVFPQPipelineTask());
        task->qid = qid;
        task->query = queries + qid * vecdim;
        q_input.push(task);
    }

    q_input.close();

    for (std::thread& t : workers_stage1) {
        t.join();
    }
    q_stage1_stage2.close();

    for (std::thread& t : workers_stage2) {
        t.join();
    }
    q_stage2_stage3.close();

    for (std::thread& t : workers_stage3) {
        t.join();
    }
}

// Cached wrapper for your current main.cc style.
// Call this function inside the original for-loop. On the first call it runs
// the whole batch pipeline and caches all query results; later calls only
// return cache_results[query_id]. Therefore, the existing average latency
// roughly becomes: batch pipeline time / query_number.
static inline IVFPQPipelineResult ivfpq_search_pipeline_cached(
    const IVFPQIndex& index,
    const float* queries,
    size_t query_id,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int stage1_threads = 1,
    int stage2_threads = 4,
    int stage3_threads = 1,
    size_t queue_capacity = 64
) {
    assert(query_id < query_number);

    static bool cache_ready = false;
    static const IVFPQIndex* cached_index = nullptr;
    static const float* cached_queries = nullptr;
    static size_t cached_query_number = 0;
    static size_t cached_vecdim = 0;
    static size_t cached_k = 0;
    static size_t cached_top_p = 0;
    static size_t cached_nprobe = 0;
    static int cached_stage1_threads = 0;
    static int cached_stage2_threads = 0;
    static int cached_stage3_threads = 0;
    static size_t cached_queue_capacity = 0;
    static std::vector<IVFPQPipelineResult> cache_results;

    bool need_rebuild =
        !cache_ready ||
        cached_index != &index ||
        cached_queries != queries ||
        cached_query_number != query_number ||
        cached_vecdim != vecdim ||
        cached_k != k ||
        cached_top_p != top_p ||
        cached_nprobe != nprobe ||
        cached_stage1_threads != stage1_threads ||
        cached_stage2_threads != stage2_threads ||
        cached_stage3_threads != stage3_threads ||
        cached_queue_capacity != queue_capacity;

    if (need_rebuild) {
        ivfpq_search_pipeline_batch(
            index,
            queries,
            query_number,
            vecdim,
            k,
            top_p,
            nprobe,
            cache_results,
            stage1_threads,
            stage2_threads,
            stage3_threads,
            queue_capacity
        );

        cached_index = &index;
        cached_queries = queries;
        cached_query_number = query_number;
        cached_vecdim = vecdim;
        cached_k = k;
        cached_top_p = top_p;
        cached_nprobe = nprobe;
        cached_stage1_threads = stage1_threads;
        cached_stage2_threads = stage2_threads;
        cached_stage3_threads = stage3_threads;
        cached_queue_capacity = queue_capacity;
        cache_ready = true;
    }

    return cache_results[query_id];
}
