#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iomanip>

#include "ivf_pq_simd.h"

// ================================================================
// IVF-PQ-SIMD + MPI baseline
// ================================================================

struct IVFPQMPIShardInfo {
    size_t global_base_number = 0;
    size_t local_start = 0;
    size_t local_count = 0;
    int rank = 0;
    int size = 1;
};

static inline void ivfpq_mpi_get_rank_range(
    size_t total,
    int rank,
    int size,
    size_t& start,
    size_t& count
) {
    assert(size > 0);
    assert(rank >= 0 && rank < size);

    const size_t p = static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);

    const size_t base_count = total / p;
    const size_t rem = total % p;

    count = base_count + (r < rem ? 1 : 0);
    start = r * base_count + std::min(r, rem);
}

static inline std::string ivfpq_mpi_default_index_path(
    const std::string& prefix,
    size_t global_base_number,
    size_t vecdim,
    size_t local_start,
    size_t local_count,
    size_t nlist,
    size_t M,
    size_t Ks,
    size_t pq_train_iters,
    size_t ivf_train_iters,
    int rank,
    int size
) {
    return prefix +
           "_rank_" + std::to_string(rank) +
           "_of_" + std::to_string(size) +
           "_global_" + std::to_string(global_base_number) +
           "_dim_" + std::to_string(vecdim) +
           "_start_" + std::to_string(local_start) +
           "_count_" + std::to_string(local_count) +
           "_nlist_" + std::to_string(nlist) +
           "_M_" + std::to_string(M) +
           "_Ks_" + std::to_string(Ks) +
           "_pqiter_" + std::to_string(pq_train_iters) +
           "_ivfiter_" + std::to_string(ivf_train_iters) +
           ".bin";
}

static inline IVFPQMPIShardInfo ivfpq_mpi_build_local_index(
    IVFPQIndex& local_index,
    const float* global_base,
    size_t global_base_number,
    size_t vecdim,
    size_t nlist,
    size_t M = 16,
    size_t Ks = 256,
    size_t pq_train_iters = 10,
    size_t ivf_train_iters = 10,
    MPI_Comm comm = MPI_COMM_WORLD,
    const std::string& index_prefix = "files/ivfpq_mpi_index"
) {
    assert(global_base != nullptr);
    assert(global_base_number > 0);
    assert(vecdim > 0);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t local_start = 0;
    size_t local_count = 0;
    ivfpq_mpi_get_rank_range(global_base_number, rank, size,
                             local_start, local_count);

    if (local_count == 0) {
        if (rank == 0) {
            std::cerr << "[IVF-PQ-MPI] Error: MPI process number is larger than base_number."
                      << std::endl;
        }
        MPI_Abort(comm, 1);
    }

    size_t local_nlist = nlist;
    if (local_nlist > local_count) {
        local_nlist = local_count;
        std::cerr << "[IVF-PQ-MPI] rank " << rank
                  << ": nlist is larger than local_count, use local_nlist = "
                  << local_nlist << std::endl;
    }

    const float* local_base = global_base + local_start * vecdim;

    ivfpq_ensure_files_dir();
    const std::string index_path = ivfpq_mpi_default_index_path(
        index_prefix,
        global_base_number,
        vecdim,
        local_start,
        local_count,
        local_nlist,
        M,
        Ks,
        pq_train_iters,
        ivf_train_iters,
        rank,
        size
    );

    if (rank == 0) {
        std::cerr << "[IVF-PQ-MPI] Build/load local IVF-PQ indexes, MPI size = "
                  << size << std::endl;
    }

    std::cerr << "[IVF-PQ-MPI] rank " << rank
              << ": global range [" << local_start << ", "
              << (local_start + local_count) << "), local_count = "
              << local_count << ", index_path = " << index_path << std::endl;

    ivfpq_build(
        local_index,
        local_base,
        local_count,
        vecdim,
        local_nlist,
        M,
        Ks,
        pq_train_iters,
        ivf_train_iters,
        index_path
    );

    MPI_Barrier(comm);

    IVFPQMPIShardInfo info;
    info.global_base_number = global_base_number;
    info.local_start = local_start;
    info.local_count = local_count;
    info.rank = rank;
    info.size = size;
    return info;
}

static inline void ivfpq_mpi_pack_local_result(
    std::priority_queue<std::pair<float, uint32_t>> local_result,
    size_t local_start,
    size_t k,
    std::vector<float>& send_dist,
    std::vector<uint32_t>& send_id
) {
    send_dist.assign(k, std::numeric_limits<float>::max());
    send_id.assign(k, std::numeric_limits<uint32_t>::max());

    size_t pos = 0;
    while (!local_result.empty() && pos < k) {
        const auto item = local_result.top();
        local_result.pop();

        const size_t gid64 = local_start + static_cast<size_t>(item.second);
        if (gid64 > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            continue;
        }

        send_dist[pos] = item.first;
        send_id[pos] = static_cast<uint32_t>(gid64);
        ++pos;
    }
}

// ================================================================
// MPI rank 间负载均衡统计结构
// ================================================================

struct IVFPQMPIRankBalanceStats {
    unsigned long long query_count = 0;
    unsigned long long total_scan_count = 0;

    double total_local_search_us = 0.0;
    double max_local_search_us = 0.0;
    double min_local_search_us = std::numeric_limits<double>::max();

    // 当前 rank 等待最慢 rank 的时间：
    // wait_r = max_rank_search_time - local_search_time
    double total_wait_us = 0.0;
};

static IVFPQMPIRankBalanceStats g_ivfpq_mpi_rank_balance_stats;

static inline void ivfpq_mpi_reset_rank_balance_stats() {
    g_ivfpq_mpi_rank_balance_stats.query_count = 0;
    g_ivfpq_mpi_rank_balance_stats.total_scan_count = 0;
    g_ivfpq_mpi_rank_balance_stats.total_local_search_us = 0.0;
    g_ivfpq_mpi_rank_balance_stats.max_local_search_us = 0.0;
    g_ivfpq_mpi_rank_balance_stats.min_local_search_us =
        std::numeric_limits<double>::max();
    g_ivfpq_mpi_rank_balance_stats.total_wait_us = 0.0;
}

static inline void ivfpq_mpi_update_rank_balance_stats(
    unsigned long long scan_count,
    double local_search_us,
    double max_rank_search_us
) {
    IVFPQMPIRankBalanceStats& s = g_ivfpq_mpi_rank_balance_stats;

    s.query_count += 1;
    s.total_scan_count += scan_count;
    s.total_local_search_us += local_search_us;

    if (local_search_us > s.max_local_search_us) {
        s.max_local_search_us = local_search_us;
    }

    if (local_search_us < s.min_local_search_us) {
        s.min_local_search_us = local_search_us;
    }

    s.total_wait_us += (max_rank_search_us - local_search_us);
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_local_profile_scan_count(
    const IVFPQIndex& index,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    unsigned long long& scan_count
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);

    if (top_p < k) {
        top_p = k;
    }

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

    scan_count = 0;
    for (uint32_t cid : probe_lists) {
        scan_count += static_cast<unsigned long long>(
            index.inverted_lists[cid].size()
        );
    }

    if (probe_lists.empty()) {
        return std::priority_queue<std::pair<float, uint32_t>>();
    }

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

    return ivfpq_rerank_flat(
        index,
        query,
        candidates,
        k
    );
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(k > 0);
    assert(local_index.vecdim > 0);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<float> query_buffer(local_index.vecdim, 0.0f);
    if (rank == root) {
        assert(query != nullptr);
        std::memcpy(query_buffer.data(), query,
                    local_index.vecdim * sizeof(float));
    }

    MPI_Bcast(query_buffer.data(),
              static_cast<int>(local_index.vecdim),
              MPI_FLOAT,
              root,
              comm);

    // 本地 IVF-PQ-SIMD 搜索。这里直接复用原来的单进程函数。
    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe
        );

    std::vector<float> send_dist;
    std::vector<uint32_t> send_id;
    ivfpq_mpi_pack_local_result(local_result,
                                shard_info.local_start,
                                k,
                                send_dist,
                                send_id);

    std::vector<float> recv_dist;
    std::vector<uint32_t> recv_id;
    if (rank == root) {
        recv_dist.resize(static_cast<size_t>(size) * k);
        recv_id.resize(static_cast<size_t>(size) * k);
    }

    MPI_Gather(send_dist.data(),
               static_cast<int>(k),
               MPI_FLOAT,
               rank == root ? recv_dist.data() : nullptr,
               static_cast<int>(k),
               MPI_FLOAT,
               root,
               comm);

    MPI_Gather(send_id.data(),
               static_cast<int>(k),
               MPI_UINT32_T,
               rank == root ? recv_id.data() : nullptr,
               static_cast<int>(k),
               MPI_UINT32_T,
               root,
               comm);

    std::priority_queue<std::pair<float, uint32_t>> global_result;
    if (rank != root) {
        return global_result;
    }

    const uint32_t invalid_id = std::numeric_limits<uint32_t>::max();
    for (size_t i = 0; i < recv_id.size(); ++i) {
        if (recv_id[i] == invalid_id) {
            continue;
        }
        if (!std::isfinite(recv_dist[i])) {
            continue;
        }
        ivfpq_push_topk(global_result, recv_dist[i], recv_id[i], k);
    }

    return global_result;
}

// ================================================================
// IVF-PQ-SIMD + MPI rank 负载均衡 profile 版本
// ================================================================

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_mpi_rank_balance_profile(
    const IVFPQIndex& local_index,
    const IVFPQMPIShardInfo& shard_info,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    assert(local_index.trained);
    assert(k > 0);
    assert(local_index.vecdim > 0);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::vector<float> query_buffer(local_index.vecdim, 0.0f);

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

    unsigned long long scan_count = 0;

    const double t0 = MPI_Wtime();

    std::priority_queue<std::pair<float, uint32_t>> local_result =
        ivfpq_search_simd_local_profile_scan_count(
            local_index,
            query_buffer.data(),
            k,
            top_p,
            nprobe,
            scan_count
        );

    const double t1 = MPI_Wtime();

    const double local_search_us = (t1 - t0) * 1000.0 * 1000.0;

    double max_rank_search_us = 0.0;
    MPI_Allreduce(
        &local_search_us,
        &max_rank_search_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        comm
    );

    ivfpq_mpi_update_rank_balance_stats(
        scan_count,
        local_search_us,
        max_rank_search_us
    );

    std::vector<float> send_dist;
    std::vector<uint32_t> send_id;

    ivfpq_mpi_pack_local_result(
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

    std::priority_queue<std::pair<float, uint32_t>> global_result;

    if (rank != root) {
        return global_result;
    }

    const uint32_t invalid_id = std::numeric_limits<uint32_t>::max();

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

    return global_result;
}

// ================================================================
// 输出 MPI rank 间负载均衡统计结果
// ================================================================

static inline void ivfpq_mpi_dump_rank_balance_stats(
    const IVFPQMPIShardInfo& shard_info,
    const std::string& output_csv_path = "",
    MPI_Comm comm = MPI_COMM_WORLD,
    int root = 0
) {
    int rank = 0;
    int size = 1;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const IVFPQMPIRankBalanceStats& s =
        g_ivfpq_mpi_rank_balance_stats;

    const double query_count =
        static_cast<double>(s.query_count);

    const double avg_scan_count =
        s.query_count == 0
        ? 0.0
        : static_cast<double>(s.total_scan_count) / query_count;

    const double avg_local_search_us =
        s.query_count == 0
        ? 0.0
        : s.total_local_search_us / query_count;

    const double avg_wait_us =
        s.query_count == 0
        ? 0.0
        : s.total_wait_us / query_count;

    const double min_local_search_us =
        s.query_count == 0
        ? 0.0
        : s.min_local_search_us;

    double local_data[8];

    local_data[0] = static_cast<double>(rank);
    local_data[1] = static_cast<double>(shard_info.local_count);
    local_data[2] = static_cast<double>(s.query_count);
    local_data[3] = avg_scan_count;
    local_data[4] = avg_local_search_us;
    local_data[5] = s.max_local_search_us;
    local_data[6] = min_local_search_us;
    local_data[7] = avg_wait_us;

    std::vector<double> all_data;

    if (rank == root) {
        all_data.resize(static_cast<size_t>(size) * 8);
    }

    MPI_Gather(
        local_data,
        8,
        MPI_DOUBLE,
        rank == root ? all_data.data() : nullptr,
        8,
        MPI_DOUBLE,
        root,
        comm
    );

    if (rank != root) {
        return;
    }

    double sum_avg_search = 0.0;
    double max_avg_search = 0.0;

    for (int r = 0; r < size; ++r) {
        const double avg_search =
            all_data[static_cast<size_t>(r) * 8 + 4];

        sum_avg_search += avg_search;

        if (avg_search > max_avg_search) {
            max_avg_search = avg_search;
        }
    }

    const double mean_avg_search =
        size == 0 ? 0.0 : sum_avg_search / static_cast<double>(size);

    const double imbalance =
        mean_avg_search == 0.0
        ? 0.0
        : max_avg_search / mean_avg_search;

    std::ofstream fout;
    std::ostream* out = &std::cerr;

    if (!output_csv_path.empty()) {
        fout.open(output_csv_path.c_str());
        if (fout.is_open()) {
            out = &fout;
        }
    }

    (*out) << std::fixed << std::setprecision(6);

    (*out) << "rank,local_count,query_count,"
           << "avg_scan_count,avg_local_search_us,"
           << "max_local_search_us,min_local_search_us,"
           << "avg_wait_us,imbalance\n";

    for (int r = 0; r < size; ++r) {
        const size_t base = static_cast<size_t>(r) * 8;

        (*out) << static_cast<int>(all_data[base + 0]) << ","
               << static_cast<unsigned long long>(all_data[base + 1]) << ","
               << static_cast<unsigned long long>(all_data[base + 2]) << ","
               << all_data[base + 3] << ","
               << all_data[base + 4] << ","
               << all_data[base + 5] << ","
               << all_data[base + 6] << ","
               << all_data[base + 7] << ","
               << imbalance << "\n";
    }

    (*out) << "summary,,,,,,,,"
           << "imbalance=" << imbalance << "\n";

    if (fout.is_open()) {
        fout.close();
    }
}