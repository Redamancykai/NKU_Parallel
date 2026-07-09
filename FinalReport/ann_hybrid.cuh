struct HybridSearchStats {
    size_t gpu_queries = 0;
    size_t cpu_queries = 0;
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;
    double wall_ms = 0.0;
};

std::vector<std::priority_queue<std::pair<float, uint32_t>>> HybridFlatSearchBatch(
    const float* base,
    const float* queries,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t search_dim,
    size_t k,
    size_t batch_size,
    const std::string& mode,
    size_t rerank_count,
    double cpu_ratio,
    HybridSearchStats& stats) {
    size_t cpu_queries = static_cast<size_t>(
        std::llround(static_cast<double>(query_number) * cpu_ratio));
    cpu_queries = std::min(cpu_queries, query_number);
    if (cpu_ratio > 0.0 && cpu_queries == 0 && query_number > 0) {
        cpu_queries = 1;
    }
    size_t gpu_queries = query_number - cpu_queries;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> gpu_results;
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> cpu_results;
    std::exception_ptr worker_error = nullptr;

    auto wall_start = std::chrono::high_resolution_clock::now();
    std::thread cpu_worker([&]() {
        try {
            if (cpu_queries > 0) {
                cpu_results = CpuFlatSearchBatch(base,
                                                 queries + gpu_queries * vecdim,
                                                 base_number,
                                                 cpu_queries,
                                                 vecdim,
                                                 search_dim,
                                                 k,
                                                 rerank_count,
                                                 stats.cpu_ms);
            }
        } catch (...) {
            worker_error = std::current_exception();
        }
    });

    try {
        if (gpu_queries > 0) {
            gpu_results = CudaFlatSearchBatch(base,
                                              queries,
                                              base_number,
                                              gpu_queries,
                                              vecdim,
                                              search_dim,
                                              k,
                                              batch_size,
                                              mode,
                                              rerank_count,
                                              stats.gpu_ms);
        }
    } catch (...) {
        cpu_worker.join();
        throw;
    }

    cpu_worker.join();
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }
    auto wall_end = std::chrono::high_resolution_clock::now();

    for (size_t q = 0; q < gpu_queries; ++q) {
        results[q] = std::move(gpu_results[q]);
    }
    for (size_t q = 0; q < cpu_queries; ++q) {
        results[gpu_queries + q] = std::move(cpu_results[q]);
    }

    stats.gpu_queries = gpu_queries;
    stats.cpu_queries = cpu_queries;
    stats.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    return results;
}
