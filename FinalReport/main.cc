#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cmath>
#include <memory>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
#include "ann_common.cuh"
#include "ann_kernels.cuh"
#include "ann_index.cuh"
#include "ann_search.cuh"
#include "ann_cpu.cuh"
#include "ann_hybrid.cuh"
#include "ann_metrics.cuh"

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = ParseOptions(argc, argv);

        size_t query_number = 0;
        size_t base_number = 0;
        size_t gt_query_number = 0;
        size_t gt_dim = 0;
        size_t query_dim = 0;
        size_t base_dim = 0;

        std::string query_path = JoinPath(options.data_path, "DEEP100K.query.fbin");
        std::string gt_path = JoinPath(options.data_path, "DEEP100K.gt.query.100k.top100.bin");
        std::string base_path = JoinPath(options.data_path, "DEEP100K.base.100k.fbin");
        bool uses_ivf = options.index == "ivf" || options.index == "ivf-group" ||
                        options.index == "ivf-fused" || options.index == "ivf-union" ||
                        options.index == "ivfpq";
        if (options.hybrid && uses_ivf && options.index != "ivf") {
            throw std::runtime_error("--hybrid IVF currently uses --index ivf");
        }
        if (options.hybrid && !uses_ivf && options.mode == "int8") {
            throw std::runtime_error("--hybrid currently supports baseline, topk, and fp16 modes");
        }

        std::unique_ptr<float[]> queries(LoadData<float>(query_path, query_number, query_dim));
        std::unique_ptr<int[]> gt(LoadData<int>(gt_path, gt_query_number, gt_dim));
        std::unique_ptr<float[]> base;
        if (uses_ivf) {
            LoadDataHeader(base_path, base_number, base_dim);
        } else {
            base.reset(LoadData<float>(base_path, base_number, base_dim));
        }

        if (query_dim != base_dim) {
            throw std::runtime_error("query dimension does not match base dimension");
        }
        size_t search_dim = options.dims == 0 ? query_dim : options.dims;
        if (search_dim == 0 || search_dim > query_dim) {
            throw std::runtime_error("--dims must be in [1, vector dimension], or 0 for full dimension");
        }
        if (gt_query_number < query_number) {
            throw std::runtime_error("ground-truth query count is smaller than query count");
        }
        if (gt_dim < options.k) {
            throw std::runtime_error("ground-truth top-k is smaller than requested k");
        }
        if (base_number < options.k) {
            throw std::runtime_error("base vector count is smaller than requested k");
        }

        query_number = std::min(query_number, options.query_count);
        if (query_number == 0) {
            throw std::runtime_error("query count must be positive");
        }
        std::cerr << "run gpu flat GEMM mode=" << options.mode
                  << " index=" << options.index
                  << " hybrid=" << options.hybrid
                  << " cpu_ratio=" << options.cpu_ratio
                  << ": queries=" << query_number
                  << " base=" << base_number
                  << " dim=" << query_dim
                  << " search_dim=" << search_dim
                  << " batch=" << options.batch_size
                  << " group_strategy=" << options.group_strategy
                  << " rerank=" << options.rerank
                  << " k=" << options.k << "\n";

        double elapsed_ms = 0.0;
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> results;
        float int8_scale = 0.0f;
        HybridSearchStats hybrid_stats;
        IVFHybridSearchStats ivf_hybrid_stats;
        bool used_ivf_hybrid = false;
        if (uses_ivf) {
            size_t nprobe = std::min(options.nprobe, options.nlist);
            IVFCPUIndex ivf_index;
            std::string cache_path = IVFCachePath(base_number,
                                                  query_dim,
                                                  options.nlist,
                                                  options.train_iters);
            bool loaded_full_index = LoadIVFIndex(cache_path, ivf_index) &&
                                     ivf_index.nlist == options.nlist &&
                                     ivf_index.vecdim == query_dim &&
                                     ivf_index.base_number == base_number &&
                                     ivf_index.full_base.size() == base_number * query_dim;
            if (loaded_full_index) {
                std::cerr << "[IVF] loaded cached full index " << cache_path << "\n";
            } else {
                size_t loaded_base_number = 0;
                size_t loaded_base_dim = 0;
                base.reset(LoadData<float>(base_path, loaded_base_number, loaded_base_dim));
                if (loaded_base_number != base_number || loaded_base_dim != base_dim) {
                    throw std::runtime_error("base header changed while loading base payload");
                }
                ivf_index = BuildIVFIndex(base.get(),
                                          base_number,
                                          query_dim,
                                          options.nlist,
                                          options.train_iters);
            }
            std::vector<uint32_t> probe_lists;
            auto probe_start = std::chrono::high_resolution_clock::now();
            SelectProbeLists(ivf_index,
                             queries.get(),
                             query_number,
                             query_dim,
                             nprobe,
                             probe_lists);
            auto probe_end = std::chrono::high_resolution_clock::now();
            double probe_ms = std::chrono::duration<double, std::milli>(
                probe_end - probe_start).count();
            std::cerr << "[IVF] selected probe lists in " << probe_ms << " ms\n";
            ivf_hybrid_stats.probe_ms = probe_ms;
            std::vector<uint32_t> query_order = BuildQueryOrder(probe_lists,
                                                                query_number,
                                                                nprobe,
                                                                options.group_strategy);
            const float* original_base =
                !ivf_index.full_base.empty() ? ivf_index.full_base.data() : base.get();
            if (options.hybrid) {
                ivf_index.full_base.clear();
                ivf_index.full_base.shrink_to_fit();
                results = CudaIVFFusedSearchBatch(ivf_index,
                                                  queries.get(),
                                                  probe_lists,
                                                  query_order,
                                                  query_number,
                                                  options.k,
                                                  options.batch_size,
                                                  nprobe,
                                                  elapsed_ms,
                                                  &ivf_hybrid_stats);
                elapsed_ms = ivf_hybrid_stats.end_to_end_ms;
                used_ivf_hybrid = true;
            } else if (options.index == "ivfpq") {
                auto pq_index = BuildIVFPQIndex(ivf_index,
                                                options.pq_m,
                                                options.pq_ks,
                                                options.train_iters);
                results = CudaIVFPQSearchBatch(pq_index,
                                               original_base,
                                               queries.get(),
                                               probe_lists,
                                               query_number,
                                               options.k,
                                               options.batch_size,
                                               nprobe,
                                               query_order,
                                               options.rerank,
                                               elapsed_ms);
            } else if (options.index == "ivf-fused") {
                ivf_index.full_base.clear();
                ivf_index.full_base.shrink_to_fit();
                results = CudaIVFFusedSearchBatch(ivf_index,
                                                  queries.get(),
                                                  probe_lists,
                                                  query_order,
                                                  query_number,
                                                  options.k,
                                                  options.batch_size,
                                                  nprobe,
                                                  elapsed_ms);
            } else if (options.index == "ivf-union") {
                ivf_index.full_base.clear();
                ivf_index.full_base.shrink_to_fit();
                results = CudaIVFUnionSearchBatch(ivf_index,
                                                  queries.get(),
                                                  probe_lists,
                                                  query_order,
                                                  query_number,
                                                  options.k,
                                                  options.batch_size,
                                                  nprobe,
                                                  elapsed_ms);
            } else {
                ivf_index.full_base.clear();
                ivf_index.full_base.shrink_to_fit();
                results = CudaIVFSearchBatch(ivf_index,
                                             queries.get(),
                                             probe_lists,
                                             query_number,
                                             options.k,
                                             options.batch_size,
                                             nprobe,
                                             query_order,
                                             options.index == "ivf-group",
                                             elapsed_ms);
            }
        } else if (options.mode == "int8") {
            auto quantized = QuantizeInt8(base.get(),
                                          queries.get(),
                                          base_number,
                                          query_number,
                                          query_dim,
                                          search_dim);
            int8_scale = quantized.scale;
            results = CudaFlatSearchBatchInt8(quantized,
                                              base_number,
                                              query_number,
                                              options.k,
                                              options.batch_size,
                                              elapsed_ms);
        } else if (options.hybrid) {
            results = HybridFlatSearchBatch(base.get(),
                                            queries.get(),
                                            base_number,
                                            query_number,
                                            query_dim,
                                            search_dim,
                                            options.k,
                                            options.batch_size,
                                            options.mode,
                                            options.rerank,
                                            options.cpu_ratio,
                                            hybrid_stats);
            elapsed_ms = hybrid_stats.wall_ms;
        } else {
            results = CudaFlatSearchBatch(base.get(),
                                          queries.get(),
                                          base_number,
                                          query_number,
                                          query_dim,
                                          search_dim,
                                          options.k,
                                          options.batch_size,
                                          options.mode,
                                          options.rerank,
                                          elapsed_ms);
        }

        double recall_sum = 0.0;
        for (size_t i = 0; i < query_number; ++i) {
            recall_sum += ComputeRecallAtK(results[i], gt.get(), gt_dim, i, options.k);
        }

        double avg_recall = recall_sum / static_cast<double>(query_number);
        double avg_latency_us = elapsed_ms * 1000.0 / static_cast<double>(query_number);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "index: " << options.index << "\n";
        std::cout << "mode: " << options.mode << "\n";
        std::cout << "dims: " << search_dim << "\n";
        std::cout << "group strategy: " << options.group_strategy << "\n";
        std::cout << "rerank: " << options.rerank << "\n";
        std::cout << "hybrid: " << options.hybrid << "\n";
        if (options.mode == "int8") {
            std::cout << "int8 scale: " << int8_scale << "\n";
        }
        if (used_ivf_hybrid) {
            std::cout << "hybrid strategy: cpu-probe-gpu-resident-ivf\n";
            std::cout << "ivf probe time (ms): " << ivf_hybrid_stats.probe_ms << "\n";
            std::cout << "ivf resident upload time (ms): " << ivf_hybrid_stats.resident_upload_ms << "\n";
            std::cout << "ivf resident bytes: " << ivf_hybrid_stats.resident_bytes << "\n";
            std::cout << "ivf query batches: " << ivf_hybrid_stats.query_batches << "\n";
            std::cout << "ivf query/probe upload time (ms): " << ivf_hybrid_stats.query_probe_upload_ms << "\n";
            std::cout << "ivf gpu batch scan time (ms): " << ivf_hybrid_stats.gpu_batch_ms << "\n";
            std::cout << "ivf end-to-end time (ms): " << ivf_hybrid_stats.end_to_end_ms << "\n";
        } else if (options.hybrid) {
            std::cout << "hybrid cpu ratio: " << options.cpu_ratio << "\n";
            std::cout << "hybrid gpu queries: " << hybrid_stats.gpu_queries << "\n";
            std::cout << "hybrid cpu queries: " << hybrid_stats.cpu_queries << "\n";
            std::cout << "hybrid gpu time (ms): " << hybrid_stats.gpu_ms << "\n";
            std::cout << "hybrid cpu time (ms): " << hybrid_stats.cpu_ms << "\n";
            std::cout << "hybrid wall time (ms): " << hybrid_stats.wall_ms << "\n";
        }
        std::cout << "average recall: " << avg_recall << "\n";
        std::cout << "average latency (us): " << avg_latency_us << "\n";
        std::cout << "total search time (ms): " << elapsed_ms << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        PrintUsage(argv[0]);
        return 1;
    }

    return 0;
}
