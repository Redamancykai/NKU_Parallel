#pragma once

#include <mpi.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "ivf_pq_mpi.h"

// ================================================================
// IVF-PQ-MPI-Pthread
// ================================================================

struct IVFPQMPIPackedItem {
    float dist;
    uint32_t id;
};

// ================================================================
// root 端 merge 开销统计
// ================================================================

struct IVFPQMPIProfileMergeStats {
    unsigned long long query_count = 0;
    unsigned long long total_candidates = 0;

    double total_gather_us = 0.0;
    double total_merge_us = 0.0;
};

static IVFPQMPIProfileMergeStats g_ivfpq_mpi_merge_stats;

static inline void ivfpq_mpi_merge_profile_reset() {
    g_ivfpq_mpi_merge_stats = IVFPQMPIProfileMergeStats();
}

static inline void ivfpq_mpi_merge_profile_update(
    size_t candidates,
    double gather_us,
    double merge_us
) {
    IVFPQMPIProfileMergeStats& s = g_ivfpq_mpi_merge_stats;

    s.query_count += 1;
    s.total_candidates += static_cast<unsigned long long>(candidates);
    s.total_gather_us += gather_us;
    s.total_merge_us += merge_us;
}

static inline void ivfpq_mpi_merge_profile_dump(
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank != root) {
        return;
    }

    const IVFPQMPIProfileMergeStats& s = g_ivfpq_mpi_merge_stats;

    const double q = static_cast<double>(s.query_count);

    const double avg_candidates =
        q == 0.0 ? 0.0 : static_cast<double>(s.total_candidates) / q;

    const double avg_gather_us =
        q == 0.0 ? 0.0 : s.total_gather_us / q;

    const double avg_merge_us =
        q == 0.0 ? 0.0 : s.total_merge_us / q;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "query_count,avg_candidates,avg_gather_us,avg_merge_us\n";

    std::cout << static_cast<unsigned long long>(s.query_count) << ","
              << avg_candidates << ","
              << avg_gather_us << ","
              << avg_merge_us << "\n";

    std::cout.flush();
}

// ================================================================
// MPI 通信 profile 统计
// 用于统计 Bcast、Gather、root merge 的平均开销
// ================================================================

struct IVFPQMPICommProfileStats {
    unsigned long long query_count = 0;
    unsigned long long total_candidates = 0;

    double total_bcast_us = 0.0;
    double total_local_us = 0.0;
    double total_gather_us = 0.0;
    double total_merge_us = 0.0;
    double total_query_us = 0.0;
};

static IVFPQMPICommProfileStats g_ivfpq_mpi_comm_profile_stats;

static inline void ivfpq_mpi_comm_profile_reset() {
    g_ivfpq_mpi_comm_profile_stats = IVFPQMPICommProfileStats();
}

static inline void ivfpq_mpi_comm_profile_update(
    size_t candidates,
    double bcast_us,
    double local_us,
    double gather_us,
    double merge_us,
    double total_us
) {
    IVFPQMPICommProfileStats& s =
        g_ivfpq_mpi_comm_profile_stats;

    s.query_count += 1;
    s.total_candidates += static_cast<unsigned long long>(candidates);

    s.total_bcast_us += bcast_us;
    s.total_local_us += local_us;
    s.total_gather_us += gather_us;
    s.total_merge_us += merge_us;
    s.total_query_us += total_us;
}

static inline void ivfpq_mpi_comm_profile_dump(
    const std::string& method_name,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if (rank != root) {
        return;
    }

    const IVFPQMPICommProfileStats& s =
        g_ivfpq_mpi_comm_profile_stats;

    const double q = static_cast<double>(s.query_count);

    const double avg_candidates =
        q == 0.0 ? 0.0 : static_cast<double>(s.total_candidates) / q;

    const double avg_bcast_us =
        q == 0.0 ? 0.0 : s.total_bcast_us / q;

    const double avg_local_us =
        q == 0.0 ? 0.0 : s.total_local_us / q;

    const double avg_gather_us =
        q == 0.0 ? 0.0 : s.total_gather_us / q;

    const double avg_merge_us =
        q == 0.0 ? 0.0 : s.total_merge_us / q;

    const double avg_total_us =
        q == 0.0 ? 0.0 : s.total_query_us / q;

    const double comm_ratio =
        avg_total_us == 0.0
        ? 0.0
        : (avg_bcast_us + avg_gather_us) / avg_total_us;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "method,query_count,avg_candidates,"
              << "avg_bcast_us,avg_local_us,avg_gather_us,"
              << "avg_merge_us,avg_total_us,comm_ratio\n";

    std::cout << method_name << ","
              << static_cast<unsigned long long>(s.query_count) << ","
              << avg_candidates << ","
              << avg_bcast_us << ","
              << avg_local_us << ","
              << avg_gather_us << ","
              << avg_merge_us << ","
              << avg_total_us << ","
              << comm_ratio << "\n";

    std::cout.flush();
}

// ================================================================
// query 广播 
// ================================================================

static inline void ivfpq_mpi_pthread_bcast_query_optimized(
    const IVFPQIndex& local_index,
    const float* query,
    std::vector<float>& query_buffer,
    MPI_Comm comm,
    int root
) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    assert(local_index.vecdim > 0);
    assert(local_index.vecdim <=
           static_cast<size_t>(std::numeric_limits<int>::max()));

    query_buffer.resize(local_index.vecdim);

    if (rank == root) {
        assert(query != nullptr);
        std::memcpy(
            query_buffer.data(),
            query,
            local_index.vecdim * sizeof(float)
        );
    }

    MPI_Bcast(
        query_buffer.data(),
        static_cast<int>(local_index.vecdim),
        MPI_FLOAT,
        root,
        comm
    );
}

static inline double ivfpq_mpi_pthread_bcast_query_profile(
    const IVFPQIndex& local_index,
    const float* query,
    std::vector<float>& query_buffer,
    MPI_Comm comm,
    int root
) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    assert(local_index.vecdim > 0);
    assert(local_index.vecdim <=
           static_cast<size_t>(std::numeric_limits<int>::max()));

    query_buffer.resize(local_index.vecdim);

    if (rank == root) {
        assert(query != nullptr);
        std::memcpy(
            query_buffer.data(),
            query,
            local_index.vecdim * sizeof(float)
        );
    }

    const double t0 = MPI_Wtime();

    MPI_Bcast(
        query_buffer.data(),
        static_cast<int>(local_index.vecdim),
        MPI_FLOAT,
        root,
        comm
    );

    const double t1 = MPI_Wtime();

    return (t1 - t0) * 1000.0 * 1000.0;
}

enum IVFPQMPIThreadJobType {
    IVFPQ_MPI_THREAD_JOB_CLUSTER_STATIC = 0,
    IVFPQ_MPI_THREAD_JOB_CLUSTER_DYNAMIC = 1,
    IVFPQ_MPI_THREAD_JOB_RERANK_STATIC = 2
};

struct IVFPQMPIThreadPool;

struct IVFPQMPIThreadWorkerArg {
    IVFPQMPIThreadPool* pool = nullptr;
    int tid = 0;
};

struct IVFPQMPIThreadPool {
    int thread_num = 1;

    std::vector<pthread_t> workers;
    std::vector<IVFPQMPIThreadWorkerArg> worker_args;

    pthread_mutex_t mutex;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;

    unsigned long long generation = 0;
    int active_workers = 0;
    bool stop = false;

    IVFPQMPIThreadJobType job_type =
        IVFPQ_MPI_THREAD_JOB_CLUSTER_STATIC;

    const IVFPQIndex* index = nullptr;
    const float* query = nullptr;

    const std::vector<uint32_t>* probe_lists = nullptr;
    const std::vector<float>* lut = nullptr;
    const std::vector<std::pair<float, uint32_t>>* candidates = nullptr;

    size_t local_p = 0;
    size_t k = 0;

    std::atomic<size_t> next_task;
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps;
};

// ================================================================
// top-k merge / heap helper
// ================================================================

static inline std::vector<std::pair<float, uint32_t>>
ivfpq_mpi_pthread_heap_to_vector(
    std::priority_queue<std::pair<float, uint32_t>> heap
) {
    std::vector<std::pair<float, uint32_t>> result;
    result.reserve(heap.size());

    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }

    return result;
}

// ================================================================
// rank 内线程负载均衡统计
// ================================================================

struct IVFPQThreadBalanceStats {
    unsigned long long query_count = 0;
    unsigned long long total_cluster_count = 0;
    unsigned long long total_scan_count = 0;

    double total_work_us = 0.0;
    double max_work_us = 0.0;
    double min_work_us = std::numeric_limits<double>::max();
};

static std::vector<IVFPQThreadBalanceStats> g_ivfpq_thread_balance_stats;

static inline void ivfpq_thread_balance_reset(int thread_num) {
    g_ivfpq_thread_balance_stats.clear();
    g_ivfpq_thread_balance_stats.resize(thread_num);
}

static inline void ivfpq_thread_balance_update(
    int tid,
    unsigned long long cluster_count,
    unsigned long long scan_count,
    double work_us
) {
    if (tid < 0 || tid >= static_cast<int>(g_ivfpq_thread_balance_stats.size())) {
        return;
    }

    IVFPQThreadBalanceStats& s = g_ivfpq_thread_balance_stats[tid];

    s.query_count += 1;
    s.total_cluster_count += cluster_count;
    s.total_scan_count += scan_count;
    s.total_work_us += work_us;

    if (work_us > s.max_work_us) {
        s.max_work_us = work_us;
    }

    if (work_us < s.min_work_us) {
        s.min_work_us = work_us;
    }
}

// ================================================================
// worker 实际执行逻辑
// ================================================================

static inline void ivfpq_mpi_pthread_pool_process_tid(
    IVFPQMPIThreadPool* pool,
    int tid
) {
    assert(pool != nullptr);
    assert(pool->index != nullptr);

    const IVFPQIndex& index = *(pool->index);

    std::priority_queue<std::pair<float, uint32_t>>& local_heap =
        pool->local_heaps[static_cast<size_t>(tid)];

    if (pool->job_type == IVFPQ_MPI_THREAD_JOB_CLUSTER_STATIC) {
        assert(pool->probe_lists != nullptr);
        assert(pool->lut != nullptr);

        const std::vector<uint32_t>& probe_lists = *(pool->probe_lists);
        const std::vector<float>& lut = *(pool->lut);

        for (size_t pi = static_cast<size_t>(tid);
             pi < probe_lists.size();
             pi += static_cast<size_t>(pool->thread_num)) {

            const uint32_t cid = probe_lists[pi];

            const std::vector<uint32_t>& ids =
                index.inverted_lists[cid];

            const std::vector<uint8_t>& codes_in_list =
                index.inverted_codes[cid];

            const uint8_t* code_data = codes_in_list.data();

            for (size_t pos = 0; pos < ids.size(); ++pos) {
                const uint8_t* code =
                    code_data + pos * index.M;

                const float dis =
                    ivfpq_adc_distance_code(index, code, lut);

                ivfpq_push_topk(
                    local_heap,
                    dis,
                    ids[pos],
                    pool->local_p
                );
            }
        }

        return;
    }
    // if (pool->job_type == IVFPQ_MPI_THREAD_JOB_CLUSTER_STATIC) {
    //     assert(pool->probe_lists != nullptr);
    //     assert(pool->lut != nullptr);

    //     const std::vector<uint32_t>& probe_lists = *(pool->probe_lists);
    //     const std::vector<float>& lut = *(pool->lut);

    //     unsigned long long local_cluster_count = 0;
    //     unsigned long long local_scan_count = 0;

    //     const auto time_begin =
    //         std::chrono::high_resolution_clock::now();

    //     for (size_t pi = static_cast<size_t>(tid);
    //         pi < probe_lists.size();
    //         pi += static_cast<size_t>(pool->thread_num)) {

    //         const uint32_t cid = probe_lists[pi];

    //         const std::vector<uint32_t>& ids =
    //             index.inverted_lists[cid];

    //         const std::vector<uint8_t>& codes_in_list =
    //             index.inverted_codes[cid];

    //         const uint8_t* code_data = codes_in_list.data();

    //         local_cluster_count += 1;
    //         local_scan_count += static_cast<unsigned long long>(ids.size());

    //         for (size_t pos = 0; pos < ids.size(); ++pos) {
    //             const uint8_t* code =
    //                 code_data + pos * index.M;

    //             const float dis =
    //                 ivfpq_adc_distance_code(index, code, lut);

    //             ivfpq_push_topk(
    //                 local_heap,
    //                 dis,
    //                 ids[pos],
    //                 pool->local_p
    //             );
    //         }
    //     }

    //     const auto time_end =
    //         std::chrono::high_resolution_clock::now();

    //     const double work_us =
    //         std::chrono::duration<double, std::micro>(
    //             time_end - time_begin
    //         ).count();

    //     ivfpq_thread_balance_update(
    //         tid,
    //         local_cluster_count,
    //         local_scan_count,
    //         work_us
    //     );

    //     return;
    // }

    if (pool->job_type == IVFPQ_MPI_THREAD_JOB_CLUSTER_DYNAMIC) {
        assert(pool->probe_lists != nullptr);
        assert(pool->lut != nullptr);

        const std::vector<uint32_t>& probe_lists = *(pool->probe_lists);
        const std::vector<float>& lut = *(pool->lut);

        while (true) {
            const size_t task_id =
                pool->next_task.fetch_add(
                    1,
                    std::memory_order_relaxed
                );

            if (task_id >= probe_lists.size()) {
                break;
            }

            const uint32_t cid = probe_lists[task_id];

            const std::vector<uint32_t>& ids =
                index.inverted_lists[cid];

            const std::vector<uint8_t>& codes_in_list =
                index.inverted_codes[cid];

            const uint8_t* code_data = codes_in_list.data();

            for (size_t pos = 0; pos < ids.size(); ++pos) {
                const uint8_t* code =
                    code_data + pos * index.M;

                const float dis =
                    ivfpq_adc_distance_code(index, code, lut);

                ivfpq_push_topk(
                    local_heap,
                    dis,
                    ids[pos],
                    pool->local_p
                );
            }
        }

        return;
    }
    // if (pool->job_type == IVFPQ_MPI_THREAD_JOB_CLUSTER_DYNAMIC) {
    //     assert(pool->probe_lists != nullptr);
    //     assert(pool->lut != nullptr);

    //     const std::vector<uint32_t>& probe_lists = *(pool->probe_lists);
    //     const std::vector<float>& lut = *(pool->lut);

    //     unsigned long long local_cluster_count = 0;
    //     unsigned long long local_scan_count = 0;

    //     const auto time_begin =
    //         std::chrono::high_resolution_clock::now();

    //     while (true) {
    //         const size_t task_id =
    //             pool->next_task.fetch_add(
    //                 1,
    //                 std::memory_order_relaxed
    //             );

    //         if (task_id >= probe_lists.size()) {
    //             break;
    //         }

    //         const uint32_t cid = probe_lists[task_id];

    //         const std::vector<uint32_t>& ids =
    //             index.inverted_lists[cid];

    //         const std::vector<uint8_t>& codes_in_list =
    //             index.inverted_codes[cid];

    //         const uint8_t* code_data = codes_in_list.data();

    //         local_cluster_count += 1;
    //         local_scan_count += static_cast<unsigned long long>(ids.size());

    //         for (size_t pos = 0; pos < ids.size(); ++pos) {
    //             const uint8_t* code =
    //                 code_data + pos * index.M;

    //             const float dis =
    //                 ivfpq_adc_distance_code(index, code, lut);

    //             ivfpq_push_topk(
    //                 local_heap,
    //                 dis,
    //                 ids[pos],
    //                 pool->local_p
    //             );
    //         }
    //     }

    //     const auto time_end =
    //         std::chrono::high_resolution_clock::now();

    //     const double work_us =
    //         std::chrono::duration<double, std::micro>(
    //             time_end - time_begin
    //         ).count();

    //     ivfpq_thread_balance_update(
    //         tid,
    //         local_cluster_count,
    //         local_scan_count,
    //         work_us
    //     );

    //     return;
    // }

    // if (pool->job_type == IVFPQ_MPI_THREAD_JOB_RERANK_STATIC) {
    //     assert(pool->query != nullptr);
    //     assert(pool->candidates != nullptr);

    //     const std::vector<std::pair<float, uint32_t>>& candidates =
    //         *(pool->candidates);

    //     const size_t total = candidates.size();

    //     const size_t begin =
    //         total * static_cast<size_t>(tid)
    //         / static_cast<size_t>(pool->thread_num);

    //     const size_t end =
    //         total * static_cast<size_t>(tid + 1)
    //         / static_cast<size_t>(pool->thread_num);

    //     for (size_t i = begin; i < end; ++i) {
    //         const uint32_t local_id = candidates[i].second;

    //         if (static_cast<size_t>(local_id) >= index.base_number) {
    //             continue;
    //         }

    //         const float* x =
    //             index.base + static_cast<size_t>(local_id) * index.vecdim;

    //         const float dis =
    //             1.0f - InnerProductSIMD(pool->query, x, index.vecdim);

    //         ivfpq_push_topk(
    //             local_heap,
    //             dis,
    //             local_id,
    //             pool->k
    //         );
    //     }

    //     return;
    // }
}

// ================================================================
// 持久化 worker 主循环
// ================================================================

static void* ivfpq_mpi_pthread_pool_worker(void* arg) {
    IVFPQMPIThreadWorkerArg* worker_arg =
        static_cast<IVFPQMPIThreadWorkerArg*>(arg);

    IVFPQMPIThreadPool* pool = worker_arg->pool;
    const int tid = worker_arg->tid;

    unsigned long long seen_generation = 0;

    while (true) {
        pthread_mutex_lock(&pool->mutex);

        while (seen_generation == pool->generation && !pool->stop) {
            pthread_cond_wait(&pool->start_cond, &pool->mutex);
        }

        if (pool->stop) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        seen_generation = pool->generation;

        pthread_mutex_unlock(&pool->mutex);

        ivfpq_mpi_pthread_pool_process_tid(pool, tid);

        pthread_mutex_lock(&pool->mutex);

        pool->active_workers--;

        if (pool->active_workers == 0) {
            pthread_cond_signal(&pool->done_cond);
        }

        pthread_mutex_unlock(&pool->mutex);
    }

    return nullptr;
}

// ================================================================
// 创建 / 获取 thread pool
//
// 注意：按 thread_num 维护多个静态 pool。
// 一般实验中 thread_num 固定，所以只会创建一个 pool。
// ================================================================

static inline IVFPQMPIThreadPool* ivfpq_mpi_pthread_get_pool(int thread_num) {
    assert(thread_num > 0);

    static std::vector<IVFPQMPIThreadPool*> pools(65, nullptr);

    if (thread_num >= static_cast<int>(pools.size())) {
        pools.resize(static_cast<size_t>(thread_num + 1), nullptr);
    }

    IVFPQMPIThreadPool*& pool =
        pools[static_cast<size_t>(thread_num)];

    if (pool != nullptr) {
        return pool;
    }

    pool = new IVFPQMPIThreadPool();
    pool->thread_num = thread_num;
    pool->next_task.store(0, std::memory_order_relaxed);

    pthread_mutex_init(&pool->mutex, nullptr);
    pthread_cond_init(&pool->start_cond, nullptr);
    pthread_cond_init(&pool->done_cond, nullptr);

    if (thread_num > 1) {
        pool->workers.resize(static_cast<size_t>(thread_num - 1));
        pool->worker_args.resize(static_cast<size_t>(thread_num - 1));

        for (int tid = 1; tid < thread_num; ++tid) {
            IVFPQMPIThreadWorkerArg& arg =
                pool->worker_args[static_cast<size_t>(tid - 1)];

            arg.pool = pool;
            arg.tid = tid;

            const int ret = pthread_create(
                &pool->workers[static_cast<size_t>(tid - 1)],
                nullptr,
                ivfpq_mpi_pthread_pool_worker,
                &arg
            );

            assert(ret == 0);
        }
    }

    return pool;
}

static inline void ivfpq_mpi_pthread_run_pool_job(
    IVFPQMPIThreadPool* pool,
    IVFPQMPIThreadJobType job_type,
    const IVFPQIndex& index,
    const float* query,
    const std::vector<uint32_t>* probe_lists,
    const std::vector<float>* lut,
    const std::vector<std::pair<float, uint32_t>>* candidates,
    size_t local_p,
    size_t k
) {
    assert(pool != nullptr);
    assert(pool->thread_num > 0);

    pthread_mutex_lock(&pool->mutex);

    pool->job_type = job_type;
    pool->index = &index;
    pool->query = query;
    pool->probe_lists = probe_lists;
    pool->lut = lut;
    pool->candidates = candidates;
    pool->local_p = local_p;
    pool->k = k;

    pool->next_task.store(0, std::memory_order_relaxed);

    pool->local_heaps.clear();
    pool->local_heaps.resize(static_cast<size_t>(pool->thread_num));

    pool->active_workers = pool->thread_num - 1;
    pool->generation++;

    pthread_cond_broadcast(&pool->start_cond);
    pthread_mutex_unlock(&pool->mutex);

    // 主线程参与计算 tid=0，避免只等待 worker。
    ivfpq_mpi_pthread_pool_process_tid(pool, 0);

    pthread_mutex_lock(&pool->mutex);

    while (pool->active_workers > 0) {
        pthread_cond_wait(&pool->done_cond, &pool->mutex);
    }

    pthread_mutex_unlock(&pool->mutex);
}

// ================================================================
// 合并 thread pool 中各线程 local_heap
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_pthread_merge_thread_heaps(
    IVFPQMPIThreadPool* pool,
    size_t keep_k
) {
    std::priority_queue<std::pair<float, uint32_t>> result;

    for (int tid = 0; tid < pool->thread_num; ++tid) {
        std::priority_queue<std::pair<float, uint32_t>>& heap =
            pool->local_heaps[static_cast<size_t>(tid)];

        while (!heap.empty()) {
            const auto item = heap.top();
            heap.pop();

            ivfpq_push_topk(
                result,
                item.first,
                item.second,
                keep_k
            );
        }
    }

    return result;
}

// ================================================================
// 本地版本一：rank 内 Pthread 簇级静态划分
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_pthread_refine_cluster_pool_local(
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
    assert(top_p > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }

    if (thread_num <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(index, query, nprobe, probe_lists);

    if (probe_lists.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    int T = std::max(1, thread_num);
    T = std::min(T, static_cast<int>(probe_lists.size()));

    if (T <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    IVFPQMPIThreadPool* pool = ivfpq_mpi_pthread_get_pool(T);

    ivfpq_mpi_pthread_run_pool_job(
        pool,
        IVFPQ_MPI_THREAD_JOB_CLUSTER_STATIC,
        index,
        query,
        &probe_lists,
        &lut,
        nullptr,
        top_p,
        k
    );

    std::priority_queue<std::pair<float, uint32_t>> candidates =
        ivfpq_mpi_pthread_merge_thread_heaps(pool, top_p);

    return ivfpq_rerank_flat(index, query, candidates, k);
}

// ================================================================
// 本地版本二：rank 内 Pthread 簇级动态划分
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_pthread_refine_dynamic_pool_local(
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
    assert(top_p > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }

    if (thread_num <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(index, query, nprobe, probe_lists);

    if (probe_lists.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    int T = std::max(1, thread_num);
    T = std::min(T, static_cast<int>(probe_lists.size()));

    if (T <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    IVFPQMPIThreadPool* pool = ivfpq_mpi_pthread_get_pool(T);

    ivfpq_mpi_pthread_run_pool_job(
        pool,
        IVFPQ_MPI_THREAD_JOB_CLUSTER_DYNAMIC,
        index,
        query,
        &probe_lists,
        &lut,
        nullptr,
        top_p,
        k
    );

    std::priority_queue<std::pair<float, uint32_t>> candidates =
        ivfpq_mpi_pthread_merge_thread_heaps(pool, top_p);

    return ivfpq_rerank_flat(index, query, candidates, k);
}

// ================================================================
// 本地 ADC 候选生成：用于 rerank 并行版本
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_pthread_adc_candidates_serial_reuse(
    const IVFPQIndex& index,
    const float* query,
    size_t top_p,
    size_t nprobe
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(top_p > 0);
    assert(nprobe > 0);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(index, query, nprobe, probe_lists);

    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (uint32_t cid : probe_lists) {
        const std::vector<uint32_t>& ids =
            index.inverted_lists[cid];

        const std::vector<uint8_t>& codes_in_list =
            index.inverted_codes[cid];

        const uint8_t* code_data = codes_in_list.data();

        for (size_t pos = 0; pos < ids.size(); ++pos) {
            const uint8_t* code =
                code_data + pos * index.M;

            const float dis =
                ivfpq_adc_distance_code(index, code, lut);

            ivfpq_push_topk(
                candidates,
                dis,
                ids[pos],
                top_p
            );
        }
    }

    return candidates;
}

// ================================================================
// 本地版本三：rank 内 Pthread rerank 阶段并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_pthread_rerank_static_pool_local(
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
    assert(top_p > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }

    std::priority_queue<std::pair<float, uint32_t>> adc_candidates =
        ivfpq_mpi_pthread_adc_candidates_serial_reuse(
            index,
            query,
            top_p,
            nprobe
        );

    if (adc_candidates.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

    if (thread_num <= 1) {
        return ivfpq_rerank_flat(index, query, adc_candidates, k);
    }

    std::vector<std::pair<float, uint32_t>> candidates =
        ivfpq_mpi_pthread_heap_to_vector(adc_candidates);

    int T = std::max(1, thread_num);
    T = std::min(T, static_cast<int>(candidates.size()));

    if (T <= 1) {
        std::priority_queue<std::pair<float, uint32_t>> heap;
        for (const auto& item : candidates) {
            heap.push(item);
        }
        return ivfpq_rerank_flat(index, query, heap, k);
    }

    IVFPQMPIThreadPool* pool = ivfpq_mpi_pthread_get_pool(T);

    ivfpq_mpi_pthread_run_pool_job(
        pool,
        IVFPQ_MPI_THREAD_JOB_RERANK_STATIC,
        index,
        query,
        nullptr,
        nullptr,
        &candidates,
        top_p,
        k
    );

    return ivfpq_mpi_pthread_merge_thread_heaps(pool, k);
}

// ================================================================
// MPI gather：打包 dist + id，一次 Gather 完成
// ================================================================

static inline void ivfpq_mpi_pthread_pack_local_result_packed(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    size_t local_start,
    size_t k,
    std::vector<IVFPQMPIPackedItem>& send_items
) {
    const uint32_t invalid_id =
        std::numeric_limits<uint32_t>::max();

    const float invalid_dist =
        std::numeric_limits<float>::max();

    send_items.resize(k);

    for (size_t i = 0; i < k; ++i) {
        send_items[i].dist = invalid_dist;
        send_items[i].id = invalid_id;
    }

    size_t pos = 0;

    while (!local_result.empty() && pos < k) {
        const auto item = local_result.top();
        local_result.pop();

        const size_t gid64 =
            local_start + static_cast<size_t>(item.second);

        if (gid64 > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }

        send_items[pos].dist = item.first;
        send_items[pos].id = static_cast<uint32_t>(gid64);

        ++pos;
    }
}

static inline void ivfpq_mpi_pthread_pack_local_result_split(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    size_t local_start,
    size_t k,
    std::vector<float>& send_dist,
    std::vector<uint32_t>& send_id
) {
    const uint32_t invalid_id =
        std::numeric_limits<uint32_t>::max();

    const float invalid_dist =
        std::numeric_limits<float>::max();

    send_dist.assign(k, invalid_dist);
    send_id.assign(k, invalid_id);

    size_t pos = 0;

    while (!local_result.empty() && pos < k) {
        const auto item = local_result.top();
        local_result.pop();

        const size_t gid64 =
            local_start + static_cast<size_t>(item.second);

        if (gid64 > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }

        send_dist[pos] = item.first;
        send_id[pos] = static_cast<uint32_t>(gid64);

        ++pos;
    }
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_pthread_gather_global_result_packed(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    const IVFPQMPIShardInfo& shard_info,
    size_t k,
    MPI_Comm comm,
    int root
) {
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<IVFPQMPIPackedItem> send_items;

    ivfpq_mpi_pthread_pack_local_result_packed(
        local_result,
        shard_info.local_start,
        k,
        send_items
    );

    std::vector<IVFPQMPIPackedItem> recv_items;

    if (rank == root) {
        recv_items.resize(static_cast<size_t>(size) * k);
    }

    const size_t byte_count_per_rank =
        k * sizeof(IVFPQMPIPackedItem);

    assert(byte_count_per_rank <=
           static_cast<size_t>(std::numeric_limits<int>::max()));

    const int mpi_byte_count =
        static_cast<int>(byte_count_per_rank);

    MPI_Gather(
        send_items.data(),
        mpi_byte_count,
        MPI_BYTE,
        rank == root ? recv_items.data() : nullptr,
        mpi_byte_count,
        MPI_BYTE,
        root,
        comm
    );

    std::priority_queue<std::pair<float, uint32_t>> global_result;

    if (rank != root) {
        return global_result;
    }

    const uint32_t invalid_id =
        std::numeric_limits<uint32_t>::max();

    for (const IVFPQMPIPackedItem& item : recv_items) {
        if (item.id == invalid_id) {
            continue;
        }

        if (!std::isfinite(item.dist)) {
            continue;
        }

        ivfpq_push_topk(
            global_result,
            item.dist,
            item.id,
            k
        );
    }

    return global_result;
}

// ================================================================
// two-gather profile
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_pthread_gather_global_result_split_profile(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    const IVFPQMPIShardInfo& shard_info,
    size_t k,
    double bcast_us,
    double local_us,
    double total_begin_time,
    MPI_Comm comm,
    int root
) {
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<float> send_dist;
    std::vector<uint32_t> send_id;

    ivfpq_mpi_pthread_pack_local_result_split(
        local_result,
        shard_info.local_start,
        k,
        send_dist,
        send_id
    );

    std::vector<float> recv_dist;
    std::vector<uint32_t> recv_id;

    if (rank == root) {
        recv_dist.resize(static_cast<size_t>(size) * k);
        recv_id.resize(static_cast<size_t>(size) * k);
    }

    const double t_gather_begin = MPI_Wtime();

    MPI_Gather(
        send_dist.data(),
        static_cast<int>(k),
        MPI_FLOAT,
        rank == root ? recv_dist.data() : nullptr,
        static_cast<int>(k),
        MPI_FLOAT,
        root,
        comm
    );

    MPI_Gather(
        send_id.data(),
        static_cast<int>(k),
        MPI_UINT32_T,
        rank == root ? recv_id.data() : nullptr,
        static_cast<int>(k),
        MPI_UINT32_T,
        root,
        comm
    );

    const double t_gather_end = MPI_Wtime();

    std::priority_queue<std::pair<float, uint32_t>> global_result;

    if (rank != root) {
        return global_result;
    }

    const double t_merge_begin = MPI_Wtime();

    const uint32_t invalid_id =
        std::numeric_limits<uint32_t>::max();

    for (size_t i = 0; i < recv_id.size(); ++i) {
        if (recv_id[i] == invalid_id) {
            continue;
        }

        if (!std::isfinite(recv_dist[i])) {
            continue;
        }

        ivfpq_push_topk(
            global_result,
            recv_dist[i],
            recv_id[i],
            k
        );
    }

    const double t_merge_end = MPI_Wtime();

    const double gather_us =
        (t_gather_end - t_gather_begin) * 1000.0 * 1000.0;

    const double merge_us =
        (t_merge_end - t_merge_begin) * 1000.0 * 1000.0;

    const double total_us =
        (t_merge_end - total_begin_time) * 1000.0 * 1000.0;

    ivfpq_mpi_comm_profile_update(
        static_cast<size_t>(size) * k,
        bcast_us,
        local_us,
        gather_us,
        merge_us,
        total_us
    );

    return global_result;
}

// ================================================================
// profile 版本：一次 MPI_Gather + root merge
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_pthread_gather_global_result_packed_profile(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    const IVFPQMPIShardInfo& shard_info,
    size_t k,
    MPI_Comm comm,
    int root
) {
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<IVFPQMPIPackedItem> send_items;

    ivfpq_mpi_pthread_pack_local_result_packed(
        local_result,
        shard_info.local_start,
        k,
        send_items
    );

    std::vector<IVFPQMPIPackedItem> recv_items;

    if (rank == root) {
        recv_items.resize(static_cast<size_t>(size) * k);
    }

    const size_t byte_count_per_rank =
        k * sizeof(IVFPQMPIPackedItem);

    assert(byte_count_per_rank <=
           static_cast<size_t>(std::numeric_limits<int>::max()));

    const int mpi_byte_count =
        static_cast<int>(byte_count_per_rank);

    const double t_gather_begin = MPI_Wtime();

    MPI_Gather(
        send_items.data(),
        mpi_byte_count,
        MPI_BYTE,
        rank == root ? recv_items.data() : nullptr,
        mpi_byte_count,
        MPI_BYTE,
        root,
        comm
    );

    const double t_gather_end = MPI_Wtime();

    std::priority_queue<std::pair<float, uint32_t>> global_result;

    if (rank != root) {
        return global_result;
    }

    const double t_merge_begin = MPI_Wtime();

    const uint32_t invalid_id =
        std::numeric_limits<uint32_t>::max();

    for (const IVFPQMPIPackedItem& item : recv_items) {
        if (item.id == invalid_id) {
            continue;
        }

        if (!std::isfinite(item.dist)) {
            continue;
        }

        ivfpq_push_topk(
            global_result,
            item.dist,
            item.id,
            k
        );
    }

    const double t_merge_end = MPI_Wtime();

    const double gather_us =
        (t_gather_end - t_gather_begin) * 1000.0 * 1000.0;

    const double merge_us =
        (t_merge_end - t_merge_begin) * 1000.0 * 1000.0;

    ivfpq_mpi_merge_profile_update(
        static_cast<size_t>(size) * k,
        gather_us,
        merge_us
    );

    return global_result;
}

// ================================================================
// single-gather profile：Item{dist,id} 一次 MPI_Gather
// 用于通信优化版本
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_pthread_gather_global_result_packed_comm_profile(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    const IVFPQMPIShardInfo& shard_info,
    size_t k,
    double bcast_us,
    double local_us,
    double total_begin_time,
    MPI_Comm comm,
    int root
) {
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<IVFPQMPIPackedItem> send_items;

    ivfpq_mpi_pthread_pack_local_result_packed(
        local_result,
        shard_info.local_start,
        k,
        send_items
    );

    std::vector<IVFPQMPIPackedItem> recv_items;

    if (rank == root) {
        recv_items.resize(static_cast<size_t>(size) * k);
    }

    const size_t byte_count_per_rank =
        k * sizeof(IVFPQMPIPackedItem);

    assert(byte_count_per_rank <=
           static_cast<size_t>(std::numeric_limits<int>::max()));

    const int mpi_byte_count =
        static_cast<int>(byte_count_per_rank);

    const double t_gather_begin = MPI_Wtime();

    MPI_Gather(
        send_items.data(),
        mpi_byte_count,
        MPI_BYTE,
        rank == root ? recv_items.data() : nullptr,
        mpi_byte_count,
        MPI_BYTE,
        root,
        comm
    );

    const double t_gather_end = MPI_Wtime();

    std::priority_queue<std::pair<float, uint32_t>> global_result;

    if (rank != root) {
        return global_result;
    }

    const double t_merge_begin = MPI_Wtime();

    const uint32_t invalid_id =
        std::numeric_limits<uint32_t>::max();

    for (const IVFPQMPIPackedItem& item : recv_items) {
        if (item.id == invalid_id) {
            continue;
        }

        if (!std::isfinite(item.dist)) {
            continue;
        }

        ivfpq_push_topk(
            global_result,
            item.dist,
            item.id,
            k
        );
    }

    const double t_merge_end = MPI_Wtime();

    const double gather_us =
        (t_gather_end - t_gather_begin) * 1000.0 * 1000.0;

    const double merge_us =
        (t_merge_end - t_merge_begin) * 1000.0 * 1000.0;

    const double total_us =
        (t_merge_end - total_begin_time) * 1000.0 * 1000.0;

    ivfpq_mpi_comm_profile_update(
        static_cast<size_t>(size) * k,
        bcast_us,
        local_us,
        gather_us,
        merge_us,
        total_us
    );

    return global_result;
}

// ================================================================
// 对外接口一：MPI + Pthread 簇级静态并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_pthread_cluster_static(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(local_index.vecdim > 0);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    std::vector<float> query_buffer;

    ivfpq_mpi_pthread_bcast_query_optimized(
        local_index,
        query,
        query_buffer,
        comm,
        root
    );

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_pthread_refine_cluster_pool_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    return ivfpq_mpi_pthread_gather_global_result_packed(
        local_result,
        shard_info,
        k,
        comm,
        root
    );
}

// ================================================================
// 对外接口二：MPI + Pthread 簇级动态并行
//
// 推荐优先测这个版本。
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_pthread_cluster_dynamic(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(local_index.vecdim > 0);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    std::vector<float> query_buffer;

    ivfpq_mpi_pthread_bcast_query_optimized(
        local_index,
        query,
        query_buffer,
        comm,
        root
    );

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_pthread_refine_dynamic_pool_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    return ivfpq_mpi_pthread_gather_global_result_packed(
        local_result,
        shard_info,
        k,
        comm,
        root
    );
}

// ================================================================
// 对外接口三：MPI + Pthread rerank 阶段并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_pthread_rerank_static(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(local_index.vecdim > 0);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    std::vector<float> query_buffer;

    ivfpq_mpi_pthread_bcast_query_optimized(
        local_index,
        query,
        query_buffer,
        comm,
        root
    );

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_pthread_rerank_static_pool_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    return ivfpq_mpi_pthread_gather_global_result_packed(
        local_result,
        shard_info,
        k,
        comm,
        root
    );
}

static inline void ivfpq_thread_balance_dump(
    int thread_num,
    const std::string& output_csv_path = "",
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int fields = 9;

    std::vector<double> local_data(
        static_cast<size_t>(thread_num) * fields,
        0.0
    );

    // 计算当前 rank 内线程不均衡系数
    double sum_avg_work = 0.0;
    double max_avg_work = 0.0;

    for (int t = 0; t < thread_num; ++t) {
        const IVFPQThreadBalanceStats& s =
            g_ivfpq_thread_balance_stats[t];

        double avg_work =
            s.query_count == 0
            ? 0.0
            : s.total_work_us / static_cast<double>(s.query_count);

        sum_avg_work += avg_work;

        if (avg_work > max_avg_work) {
            max_avg_work = avg_work;
        }
    }

    double mean_avg_work =
        thread_num == 0
        ? 0.0
        : sum_avg_work / static_cast<double>(thread_num);

    double imbalance =
        mean_avg_work == 0.0
        ? 0.0
        : max_avg_work / mean_avg_work;

    for (int t = 0; t < thread_num; ++t) {
        const IVFPQThreadBalanceStats& s =
            g_ivfpq_thread_balance_stats[t];

        double query_count =
            static_cast<double>(s.query_count);

        double avg_cluster =
            s.query_count == 0
            ? 0.0
            : static_cast<double>(s.total_cluster_count) / query_count;

        double avg_scan =
            s.query_count == 0
            ? 0.0
            : static_cast<double>(s.total_scan_count) / query_count;

        double avg_work =
            s.query_count == 0
            ? 0.0
            : s.total_work_us / query_count;

        double min_work =
            s.query_count == 0
            ? 0.0
            : s.min_work_us;

        const size_t base =
            static_cast<size_t>(t) * fields;

        local_data[base + 0] = static_cast<double>(rank);
        local_data[base + 1] = static_cast<double>(t);
        local_data[base + 2] = query_count;
        local_data[base + 3] = avg_cluster;
        local_data[base + 4] = avg_scan;
        local_data[base + 5] = avg_work;
        local_data[base + 6] = s.max_work_us;
        local_data[base + 7] = min_work;
        local_data[base + 8] = imbalance;
    }

    std::vector<double> all_data;

    if (rank == root) {
        all_data.resize(
            static_cast<size_t>(size) *
            static_cast<size_t>(thread_num) *
            fields
        );
    }

    MPI_Gather(
        local_data.data(),
        thread_num * fields,
        MPI_DOUBLE,
        rank == root ? all_data.data() : nullptr,
        thread_num * fields,
        MPI_DOUBLE,
        root,
        comm
    );

    if (rank != root) {
        return;
    }

    std::ofstream fout;
    std::ostream* out = &std::cout;

    if (!output_csv_path.empty()) {
        fout.open(output_csv_path.c_str());

        if (fout.is_open()) {
            out = &fout;
        }
    }

    (*out) << std::fixed << std::setprecision(6);

    (*out) << "rank,thread,query_count,"
           << "avg_cluster_count,avg_scan_count,"
           << "avg_work_us,max_work_us,min_work_us,"
           << "thread_imbalance\n";

    for (int r = 0; r < size; ++r) {
        for (int t = 0; t < thread_num; ++t) {
            const size_t base =
                (static_cast<size_t>(r) * thread_num + t) *
                fields;

            (*out) << static_cast<int>(all_data[base + 0]) << ","
                   << static_cast<int>(all_data[base + 1]) << ","
                   << static_cast<unsigned long long>(all_data[base + 2]) << ","
                   << all_data[base + 3] << ","
                   << all_data[base + 4] << ","
                   << all_data[base + 5] << ","
                   << all_data[base + 6] << ","
                   << all_data[base + 7] << ","
                   << all_data[base + 8] << "\n";
        }
    }

    out->flush();

    if (fout.is_open()) {
        fout.close();
    }
}

// ================================================================
// 对外接口：MPI + Pthread dynamic cluster + merge profile
// 用于统计 root 端 merge 开销
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_pthread_cluster_dynamic_merge_profile(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(local_index.vecdim > 0);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    std::vector<float> query_buffer;

    ivfpq_mpi_pthread_bcast_query_optimized(
        local_index,
        query,
        query_buffer,
        comm,
        root
    );

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_pthread_refine_dynamic_pool_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    return ivfpq_mpi_pthread_gather_global_result_packed_profile(
        local_result,
        shard_info,
        k,
        comm,
        root
    );
}

// ================================================================
// 通信 profile 接口一：two Gather baseline
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_pthread_cluster_dynamic_two_gather_comm_profile(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(local_index.vecdim > 0);

    const double t_total_begin = MPI_Wtime();

    std::vector<float> query_buffer;

    const double bcast_us =
        ivfpq_mpi_pthread_bcast_query_profile(
            local_index,
            query,
            query_buffer,
            comm,
            root
        );

    const double t_local_begin = MPI_Wtime();

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_pthread_refine_dynamic_pool_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    const double t_local_end = MPI_Wtime();

    const double local_us =
        (t_local_end - t_local_begin) * 1000.0 * 1000.0;

    return ivfpq_mpi_pthread_gather_global_result_split_profile(
        local_result,
        shard_info,
        k,
        bcast_us,
        local_us,
        t_total_begin,
        comm,
        root
    );
}

// ================================================================
// 通信 profile 接口二：single Gather optimized
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_pthread_cluster_dynamic_single_gather_comm_profile(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(local_index.vecdim > 0);

    const double t_total_begin = MPI_Wtime();

    std::vector<float> query_buffer;

    const double bcast_us =
        ivfpq_mpi_pthread_bcast_query_profile(
            local_index,
            query,
            query_buffer,
            comm,
            root
        );

    const double t_local_begin = MPI_Wtime();

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_pthread_refine_dynamic_pool_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    const double t_local_end = MPI_Wtime();

    const double local_us =
        (t_local_end - t_local_begin) * 1000.0 * 1000.0;

    return ivfpq_mpi_pthread_gather_global_result_packed_comm_profile(
        local_result,
        shard_info,
        k,
        bcast_us,
        local_us,
        t_total_begin,
        comm,
        root
    );
}