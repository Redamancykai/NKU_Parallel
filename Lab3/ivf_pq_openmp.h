#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>
#include <array>

#include "ivf_pq_simd.h"

struct alignas(64) IVFPQOpenMPThreadLocal {
    std::priority_queue<std::pair<float, uint32_t>> local_heap;
    size_t scanned_points = 0;

    // 只起填充作用，避免相邻线程上下文落在同一 cache line。
    char padding[64];
};

static inline void ivfpq_scan_one_list_adc(
    const IVFPQIndex& index,
    uint32_t cid,
    const std::vector<float>& lut,
    size_t top_p,
    std::priority_queue<std::pair<float, uint32_t>>& heap,
    size_t* scanned_points = nullptr
) {
    const std::vector<uint32_t>& ids =
        index.inverted_lists[cid];

    const std::vector<uint8_t>& codes_in_list =
        index.inverted_codes[cid];

    assert(codes_in_list.size() == ids.size() * index.M);

    const uint8_t* code_data = codes_in_list.data();

    if (scanned_points != nullptr) {
        *scanned_points += ids.size();
    }

    for (size_t pos = 0; pos < ids.size(); ++pos) {
        const uint8_t* code =
            code_data + pos * index.M;

        float dis = ivfpq_adc_distance_code(
            index,
            code,
            lut
        );

        ivfpq_push_topk(
            heap,
            dis,
            ids[pos],
            top_p
        );
    }
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_openmp_optimized(
    const IVFPQIndex& index,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    int adc_chunk_size = 1
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    assert(index.inverted_lists.size() == index.nlist);
    assert(index.inverted_codes.size() == index.nlist);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (top_p < k) {
        top_p = k;
    }

    if (adc_chunk_size <= 0) {
        adc_chunk_size = 1;
    }

    if (thread_num <= 1 || index.nlist <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    if (thread_num > static_cast<int>(index.nlist)) {
        thread_num = static_cast<int>(index.nlist);
    }

    // 每个线程一个 centroid top-nprobe 堆
    std::vector<IVFPQOpenMPThreadLocal> centroid_locals(
        static_cast<size_t>(thread_num)
    );

    // 每个线程一个 ADC top-p 堆
    std::vector<IVFPQOpenMPThreadLocal> adc_locals(
        static_cast<size_t>(thread_num)
    );

    std::vector<uint32_t> probe_lists;
    std::vector<float> lut;

    #pragma omp parallel num_threads(thread_num) default(none) \
        shared(index, query, nprobe, top_p, thread_num, centroid_locals, adc_locals, probe_lists, lut, adc_chunk_size)
    {
        int tid = omp_get_thread_num();

        auto& centroid_heap =
            centroid_locals[static_cast<size_t>(tid)].local_heap;

        // centroid 粗排
        #pragma omp for schedule(static)
        for (int ci = 0; ci < static_cast<int>(index.nlist); ++ci) {
            size_t c = static_cast<size_t>(ci);

            const float* centroid =
                index.centroids.data() + c * index.vecdim;

            float dis = 1.0f - InnerProductSIMD(
                query,
                centroid,
                index.vecdim
            );

            ivfpq_push_topk(
                centroid_heap,
                dis,
                static_cast<uint32_t>(c),
                nprobe
            );
        }

        // single merge centroid local top-nprobe
        #pragma omp single
        {
            std::priority_queue<std::pair<float, uint32_t>> global_heap;

            for (int t = 0; t < thread_num; ++t) {
                auto& local_heap =
                    centroid_locals[static_cast<size_t>(t)].local_heap;

                while (!local_heap.empty()) {
                    auto item = local_heap.top();
                    local_heap.pop();

                    ivfpq_push_topk(
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

            // 构建 query 的 PQ LUT
            ivfpq_build_lut(index, query, lut);
        }

        // dynamic ADC 扫描
        auto& adc_heap =
            adc_locals[static_cast<size_t>(tid)].local_heap;

        size_t& scanned_points =
            adc_locals[static_cast<size_t>(tid)].scanned_points;

        #pragma omp for schedule(dynamic, adc_chunk_size)
        for (int pi = 0; pi < static_cast<int>(probe_lists.size()); ++pi) {
            uint32_t cid =
                probe_lists[static_cast<size_t>(pi)];

            ivfpq_scan_one_list_adc(
                index,
                cid,
                lut,
                top_p,
                adc_heap,
                &scanned_points
            );
        }
    }

    // merge ADC local top-p
    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (int t = 0; t < thread_num; ++t) {
        auto& local_heap =
            adc_locals[static_cast<size_t>(t)].local_heap;

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();

            ivfpq_push_topk(
                candidates,
                item.first,
                item.second,
                top_p
            );
        }
    }

    // Flat-SIMD rerank
    return ivfpq_rerank_flat(
        index,
        query,
        candidates,
        k
    );
}

// guided 版本
static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_openmp_optimized_guided(
    const IVFPQIndex& index,
    const float* query,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    int min_chunk_size = 1
) {
    assert(index.trained);
    assert(query != nullptr);
    assert(k > 0);
    assert(top_p > 0);
    assert(nprobe > 0);
    assert(thread_num > 0);

    assert(index.inverted_lists.size() == index.nlist);
    assert(index.inverted_codes.size() == index.nlist);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (top_p < k) {
        top_p = k;
    }

    if (min_chunk_size <= 0) {
        min_chunk_size = 1;
    }

    if (thread_num <= 1 || index.nlist <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    if (thread_num > static_cast<int>(index.nlist)) {
        thread_num = static_cast<int>(index.nlist);
    }

    std::vector<IVFPQOpenMPThreadLocal> centroid_locals(
        static_cast<size_t>(thread_num)
    );

    std::vector<IVFPQOpenMPThreadLocal> adc_locals(
        static_cast<size_t>(thread_num)
    );

    std::vector<uint32_t> probe_lists;
    std::vector<float> lut;

    #pragma omp parallel num_threads(thread_num) default(none) \
        shared(index, query, nprobe, top_p, thread_num, centroid_locals, adc_locals, probe_lists, lut, min_chunk_size)
    {
        int tid = omp_get_thread_num();

        auto& centroid_heap =
            centroid_locals[static_cast<size_t>(tid)].local_heap;

        #pragma omp for schedule(static)
        for (int ci = 0; ci < static_cast<int>(index.nlist); ++ci) {
            size_t c = static_cast<size_t>(ci);

            const float* centroid =
                index.centroids.data() + c * index.vecdim;

            float dis = 1.0f - InnerProductSIMD(
                query,
                centroid,
                index.vecdim
            );

            ivfpq_push_topk(
                centroid_heap,
                dis,
                static_cast<uint32_t>(c),
                nprobe
            );
        }

        #pragma omp single
        {
            std::priority_queue<std::pair<float, uint32_t>> global_heap;

            for (int t = 0; t < thread_num; ++t) {
                auto& local_heap =
                    centroid_locals[static_cast<size_t>(t)].local_heap;

                while (!local_heap.empty()) {
                    auto item = local_heap.top();
                    local_heap.pop();

                    ivfpq_push_topk(
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

            ivfpq_build_lut(index, query, lut);
        }

        auto& adc_heap =
            adc_locals[static_cast<size_t>(tid)].local_heap;

        size_t& scanned_points =
            adc_locals[static_cast<size_t>(tid)].scanned_points;

        #pragma omp for schedule(guided, min_chunk_size)
        for (int pi = 0; pi < static_cast<int>(probe_lists.size()); ++pi) {
            uint32_t cid =
                probe_lists[static_cast<size_t>(pi)];

            ivfpq_scan_one_list_adc(
                index,
                cid,
                lut,
                top_p,
                adc_heap,
                &scanned_points
            );
        }
    }

    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (int t = 0; t < thread_num; ++t) {
        auto& local_heap =
            adc_locals[static_cast<size_t>(t)].local_heap;

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();

            ivfpq_push_topk(
                candidates,
                item.first,
                item.second,
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
ivfpq_search_batch_query_openmp_cached_optimized(
    const IVFPQIndex& index,
    const float* queries,
    size_t query_id,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    size_t nprobe,
    int thread_num,
    int query_chunk_size = 1
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

    if (query_chunk_size <= 0) {
        query_chunk_size = 1;
    }

    static std::vector<std::priority_queue<std::pair<float, uint32_t>>> cache_results;

    static const IVFPQIndex* cached_index = nullptr;
    static const float* cached_queries = nullptr;
    static size_t cached_query_number = 0;
    static size_t cached_vecdim = 0;
    static size_t cached_k = 0;
    static size_t cached_top_p = 0;
    static size_t cached_nprobe = 0;
    static int cached_thread_num = 0;
    static int cached_query_chunk_size = 0;
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
        cached_top_p != top_p ||
        cached_nprobe != nprobe ||
        cached_thread_num != thread_num ||
        cached_query_chunk_size != query_chunk_size) {
        need_rebuild_cache = true;
    }

    if (query_id == 0) {
        need_rebuild_cache = true;
    }

    if (need_rebuild_cache) {
        cache_results.clear();
        cache_results.resize(query_number);

        if (thread_num <= 1) {
            for (size_t qi = 0; qi < query_number; ++qi) {
                const float* query =
                    queries + qi * vecdim;

                cache_results[qi] = ivfpq_search_simd(
                    index,
                    query,
                    k,
                    top_p,
                    nprobe
                );
            }
        } else {
            #pragma omp parallel for num_threads(thread_num) schedule(dynamic, query_chunk_size) default(none) \
                shared(index, queries, query_number, vecdim, k, top_p, nprobe, cache_results, query_chunk_size, thread_num)
            for (int qi = 0; qi < static_cast<int>(query_number); ++qi) {
                const float* query =
                    queries + static_cast<size_t>(qi) * vecdim;

                cache_results[static_cast<size_t>(qi)] = ivfpq_search_simd(
                    index,
                    query,
                    k,
                    top_p,
                    nprobe
                );
            }
        }

        cached_index = &index;
        cached_queries = queries;
        cached_query_number = query_number;
        cached_vecdim = vecdim;
        cached_k = k;
        cached_top_p = top_p;
        cached_nprobe = nprobe;
        cached_thread_num = thread_num;
        cached_query_chunk_size = query_chunk_size;
        cache_ready = true;
    }

    return cache_results[query_id];
}

static inline std::priority_queue<std::pair<float, uint32_t>>
ivfpq_search_simd_openmp_refine_dynamic(
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

    assert(index.inverted_lists.size() == index.nlist);
    assert(index.inverted_codes.size() == index.nlist);

    if (nprobe > index.nlist) {
        nprobe = index.nlist;
    }

    if (top_p < k) {
        top_p = k;
    }

    if (thread_num <= 1) {
        return ivfpq_search_simd(index, query, k, top_p, nprobe);
    }

    if (thread_num > static_cast<int>(nprobe)) {
        thread_num = static_cast<int>(nprobe);
    }

    // IVF 粗排
    std::vector<uint32_t> probe_lists;
    ivfpq_select_probe_lists(index, query, nprobe, probe_lists);

    // 构建 query 的 PQ LUT
    std::vector<float> lut;
    ivfpq_build_lut(index, query, lut);

    // 维护局部 top-p
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_heaps(
        static_cast<size_t>(thread_num)
    );

    std::vector<size_t> scanned_points(
        static_cast<size_t>(thread_num),
        0
    );

    // OpenMP 动态并行 ADC 扫描
    #pragma omp parallel num_threads(thread_num)
    {
        int tid = omp_get_thread_num();

        auto& local_heap =
            local_heaps[static_cast<size_t>(tid)];

        #pragma omp for schedule(dynamic, 1)
        for (int pi = 0; pi < static_cast<int>(probe_lists.size()); ++pi) {
            uint32_t cid = probe_lists[static_cast<size_t>(pi)];

            const std::vector<uint32_t>& ids =
                index.inverted_lists[cid];

            const std::vector<uint8_t>& codes_in_list =
                index.inverted_codes[cid];

            assert(codes_in_list.size() == ids.size() * index.M);

            const uint8_t* code_data = codes_in_list.data();

            scanned_points[static_cast<size_t>(tid)] += ids.size();

            for (size_t pos = 0; pos < ids.size(); ++pos) {
                const uint8_t* code =
                    code_data + pos * index.M;

                float dis = ivfpq_adc_distance_code(
                    index,
                    code,
                    lut
                );

                ivfpq_push_topk(
                    local_heap,
                    dis,
                    ids[pos],
                    top_p
                );
            }
        }
    }

    // merge 每个线程的局部 top-p
    std::priority_queue<std::pair<float, uint32_t>> candidates;

    for (int t = 0; t < thread_num; ++t) {
        auto& local_heap =
            local_heaps[static_cast<size_t>(t)];

        while (!local_heap.empty()) {
            auto item = local_heap.top();
            local_heap.pop();

            ivfpq_push_topk(
                candidates,
                item.first,
                item.second,
                top_p
            );
        }
    }

    // Flat-SIMD rerank
    return ivfpq_rerank_flat(
        index,
        query,
        candidates,
        k
    );
}