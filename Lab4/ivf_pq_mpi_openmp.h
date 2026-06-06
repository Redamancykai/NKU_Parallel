#pragma once

#include <mpi.h>
#include <omp.h>

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

#include "ivf_pq_mpi.h"

// ================================================================
// IVF-PQ-MPI-OpenMP
// ================================================================

struct IVFPQMPIOMPItem {
    float dist;
    uint32_t id;
};

// ================================================================
// query 广播
// ================================================================

static inline void ivfpq_mpi_openmp_bcast_query(
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

// ================================================================
// 轻量线程局部 top-k
// ================================================================

struct IVFPQMPIOMPFastTopK {
    size_t limit = 0;
    std::vector<std::pair<float, uint32_t>> heap;

    void reset(size_t keep) {
        limit = keep;
        heap.clear();

        if (heap.capacity() < keep + 1) {
            heap.reserve(keep + 1);
        }
    }

    inline void push(float dist, uint32_t id) {
        if (limit == 0) {
            return;
        }

        std::pair<float, uint32_t> item(dist, id);

        if (heap.size() < limit) {
            heap.push_back(item);
            std::push_heap(heap.begin(), heap.end());
            return;
        }

        // heap.front() 是当前 top-k 中距离最大的元素，即最差候选。
        if (dist < heap.front().first) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = item;
            std::push_heap(heap.begin(), heap.end());
        }
    }
};

struct alignas(64) IVFPQMPIOMPThreadLocal {
    IVFPQMPIOMPFastTopK topk;
    char padding[64];
};

static inline void ivfpq_mpi_openmp_runtime_init() {
    static bool initialized = false;

    if (!initialized) {
        omp_set_dynamic(0);
        omp_set_nested(0);
        omp_set_max_active_levels(1);
        initialized = true;
    }
}

// ================================================================
// heap -> vector
// ================================================================

static inline std::vector<std::pair<float, uint32_t>>
ivfpq_mpi_openmp_heap_to_vector(
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
// 扫描一个倒排表
// ================================================================

static inline void ivfpq_mpi_openmp_scan_one_list(
    const IVFPQIndex& index,
    uint32_t cid,
    const std::vector<float>& lut,
    IVFPQMPIOMPFastTopK& local_topk
) {
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

        local_topk.push(dis, ids[pos]);
    }
}

// ================================================================
// 合并线程局部 top-k
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_openmp_merge_thread_topks(
    std::vector<IVFPQMPIOMPThreadLocal>& locals,
    size_t keep_k
) {
    std::priority_queue<std::pair<float, uint32_t>> result;

    for (size_t tid = 0; tid < locals.size(); ++tid) {
        const std::vector<std::pair<float, uint32_t>>& heap =
            locals[tid].topk.heap;

        for (size_t i = 0; i < heap.size(); ++i) {
            ivfpq_push_topk(
                result,
                heap[i].first,
                heap[i].second,
                keep_k
            );
        }
    }

    return result;
}

// ================================================================
// 本地版本一：rank 内 OpenMP 簇级 static 并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_openmp_cluster_static_local(
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
    assert(thread_num > 0);

    if (top_p < k) {
        top_p = k;
    }

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (thread_num <= 1) {
        return ivfpq_search_simd(
            index,
            query,
            k,
            top_p,
            nprobe
        );
    }

    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(
        index,
        query,
        nprobe,
        probe_lists
    );

    if (probe_lists.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

    std::vector<float> lut;
    ivfpq_build_lut(
        index,
        query,
        lut
    );

    int T = std::max(1, thread_num);
    T = std::min(T, static_cast<int>(probe_lists.size()));

    ivfpq_mpi_openmp_runtime_init();

    std::vector<IVFPQMPIOMPThreadLocal> locals(
        static_cast<size_t>(T)
    );

    const size_t probe_count = probe_lists.size();

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();

        IVFPQMPIOMPFastTopK& local_topk =
            locals[static_cast<size_t>(tid)].topk;

        local_topk.reset(top_p);

        for (size_t pi = static_cast<size_t>(tid);
             pi < probe_count;
             pi += static_cast<size_t>(T)) {

            const uint32_t cid = probe_lists[pi];

            ivfpq_mpi_openmp_scan_one_list(
                index,
                cid,
                lut,
                local_topk
            );
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> candidates =
        ivfpq_mpi_openmp_merge_thread_topks(
            locals,
            top_p
        );

    return ivfpq_rerank_flat(
        index,
        query,
        candidates,
        k
    );
}

// ================================================================
// 本地版本二：rank 内 OpenMP 簇级 dynamic 并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_openmp_cluster_dynamic_local(
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
    assert(thread_num > 0);

    if (top_p < k) {
        top_p = k;
    }

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (thread_num <= 1) {
        return ivfpq_search_simd(
            index,
            query,
            k,
            top_p,
            nprobe
        );
    }

    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(
        index,
        query,
        nprobe,
        probe_lists
    );

    if (probe_lists.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

    std::vector<float> lut;
    ivfpq_build_lut(
        index,
        query,
        lut
    );

    int T = std::max(1, thread_num);
    T = std::min(T, static_cast<int>(probe_lists.size()));

    ivfpq_mpi_openmp_runtime_init();

    std::vector<IVFPQMPIOMPThreadLocal> locals(
        static_cast<size_t>(T)
    );

    const size_t probe_count = probe_lists.size();

    std::atomic<size_t> next_task;
    next_task.store(0, std::memory_order_relaxed);

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();

        IVFPQMPIOMPFastTopK& local_topk =
            locals[static_cast<size_t>(tid)].topk;

        local_topk.reset(top_p);

        while (true) {
            const size_t pi =
                next_task.fetch_add(
                    1,
                    std::memory_order_relaxed
                );

            if (pi >= probe_count) {
                break;
            }

            const uint32_t cid = probe_lists[pi];

            ivfpq_mpi_openmp_scan_one_list(
                index,
                cid,
                lut,
                local_topk
            );
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> candidates =
        ivfpq_mpi_openmp_merge_thread_topks(
            locals,
            top_p
        );

    return ivfpq_rerank_flat(
        index,
        query,
        candidates,
        k
    );
}

// ================================================================
// 串行 ADC 候选生成：供 rerank 并行版本使用
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_openmp_adc_candidates_serial(
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
    ivfpq_select_probe_lists(
        index,
        query,
        nprobe,
        probe_lists
    );

    std::vector<float> lut;
    ivfpq_build_lut(
        index,
        query,
        lut
    );

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
// 本地版本三：rank 内 OpenMP rerank 阶段并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_openmp_rerank_static_local(
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
    assert(thread_num > 0);

    if (top_p < k) {
        top_p = k;
    }

    std::priority_queue<std::pair<float, uint32_t>> adc_candidates =
        ivfpq_mpi_openmp_adc_candidates_serial(
            index,
            query,
            top_p,
            nprobe
        );

    if (adc_candidates.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

    if (thread_num <= 1) {
        return ivfpq_rerank_flat(
            index,
            query,
            adc_candidates,
            k
        );
    }

    std::vector<std::pair<float, uint32_t>> candidates =
        ivfpq_mpi_openmp_heap_to_vector(adc_candidates);

    int T = std::max(1, thread_num);
    T = std::min(T, static_cast<int>(candidates.size()));

    ivfpq_mpi_openmp_runtime_init();

    std::vector<IVFPQMPIOMPThreadLocal> locals(
        static_cast<size_t>(T)
    );

    const size_t candidate_count = candidates.size();

    #pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();

        IVFPQMPIOMPFastTopK& local_topk =
            locals[static_cast<size_t>(tid)].topk;

        local_topk.reset(k);

        const size_t begin =
            candidate_count * static_cast<size_t>(tid)
            / static_cast<size_t>(T);

        const size_t end =
            candidate_count * static_cast<size_t>(tid + 1)
            / static_cast<size_t>(T);

        for (size_t i = begin; i < end; ++i) {
            const uint32_t local_id = candidates[i].second;

            if (static_cast<size_t>(local_id) >= index.base_number) {
                continue;
            }

            const float* x =
                index.base + static_cast<size_t>(local_id) * index.vecdim;

            const float dis =
                1.0f - InnerProductSIMD(
                    query,
                    x,
                    index.vecdim
                );

            local_topk.push(dis, local_id);
        }
    }

    return ivfpq_mpi_openmp_merge_thread_topks(
        locals,
        k
    );
}

// ================================================================
// local result 打包：local id -> global id
// ================================================================

static inline void ivfpq_mpi_openmp_pack_local_result(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    size_t local_start,
    size_t k,
    std::vector<IVFPQMPIOMPItem>& send_items
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

// ================================================================
// packed MPI_Gather
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_mpi_openmp_gather_global_result(
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

    std::vector<IVFPQMPIOMPItem> send_items;

    ivfpq_mpi_openmp_pack_local_result(
        local_result,
        shard_info.local_start,
        k,
        send_items
    );

    std::vector<IVFPQMPIOMPItem> recv_items;

    if (rank == root) {
        recv_items.resize(static_cast<size_t>(size) * k);
    }

    const size_t byte_count_per_rank =
        k * sizeof(IVFPQMPIOMPItem);

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

    for (size_t i = 0; i < recv_items.size(); ++i) {
        const IVFPQMPIOMPItem& item = recv_items[i];

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
// 对外接口一：MPI + OpenMP 簇级 static 并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_openmp_cluster_static(
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

    ivfpq_mpi_openmp_bcast_query(
        local_index,
        query,
        query_buffer,
        comm,
        root
    );

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_openmp_cluster_static_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    return ivfpq_mpi_openmp_gather_global_result(
        local_result,
        shard_info,
        k,
        comm,
        root
    );
}

// ================================================================
// 对外接口二：MPI + OpenMP 簇级 dynamic 并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_openmp_cluster_dynamic(
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

    ivfpq_mpi_openmp_bcast_query(
        local_index,
        query,
        query_buffer,
        comm,
        root
    );

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_openmp_cluster_dynamic_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    return ivfpq_mpi_openmp_gather_global_result(
        local_result,
        shard_info,
        k,
        comm,
        root
    );
}

// ================================================================
// 对外接口三：MPI + OpenMP rerank 阶段并行
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_openmp_rerank_static(
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

    ivfpq_mpi_openmp_bcast_query(
        local_index,
        query,
        query_buffer,
        comm,
        root
    );

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_openmp_rerank_static_local(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            thread_num
        );

    return ivfpq_mpi_openmp_gather_global_result(
        local_result,
        shard_info,
        k,
        comm,
        root
    );
}