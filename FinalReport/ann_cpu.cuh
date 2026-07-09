std::priority_queue<std::pair<float, uint32_t>> CpuFlatSearchOne(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t search_dim,
    size_t k,
    size_t rerank_count) {
    size_t candidate_count = rerank_count == 0 ? k : rerank_count;
    bool do_rerank = rerank_count > k;
    std::priority_queue<std::pair<float, uint32_t>> heap;
    for (size_t base_id = 0; base_id < base_number; ++base_id) {
        const float* base_vec = base + base_id * vecdim;
        float dot = 0.0f;
        for (size_t d = 0; d < search_dim; ++d) {
            dot += base_vec[d] * query[d];
        }
        float distance = 1.0f - dot;
        if (heap.size() < candidate_count) {
            heap.push({distance, static_cast<uint32_t>(base_id)});
        } else if (distance < heap.top().first) {
            heap.pop();
            heap.push({distance, static_cast<uint32_t>(base_id)});
        }
    }
    if (do_rerank) {
        std::priority_queue<std::pair<float, uint32_t>> reranked;
        while (!heap.empty()) {
            uint32_t base_id = heap.top().second;
            heap.pop();

            const float* base_vec = base + static_cast<size_t>(base_id) * vecdim;
            float dot = 0.0f;
            for (size_t d = 0; d < vecdim; ++d) {
                dot += base_vec[d] * query[d];
            }
            float distance = 1.0f - dot;
            if (reranked.size() < k) {
                reranked.push({distance, base_id});
            } else if (distance < reranked.top().first) {
                reranked.pop();
                reranked.push({distance, base_id});
            }
        }
        return reranked;
    }
    return heap;
}

std::priority_queue<std::pair<float, uint32_t>> CpuFlatSearchOne(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t search_dim,
    size_t k) {
    return CpuFlatSearchOne(base, query, base_number, vecdim, search_dim, k, 0);
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> CpuFlatSearchBatch(
    const float* base,
    const float* queries,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t search_dim,
    size_t k,
    size_t rerank_count,
    double& elapsed_ms) {
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t q = 0; q < query_number; ++q) {
        results[q] = CpuFlatSearchOne(base,
                                      queries + q * vecdim,
                                      base_number,
                                      vecdim,
                                      search_dim,
                                      k,
                                      rerank_count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return results;
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> CpuFlatSearchBatch(
    const float* base,
    const float* queries,
    size_t base_number,
    size_t query_number,
    size_t vecdim,
    size_t search_dim,
    size_t k,
    double& elapsed_ms) {
    return CpuFlatSearchBatch(base,
                              queries,
                              base_number,
                              query_number,
                              vecdim,
                              search_dim,
                              k,
                              0,
                              elapsed_ms);
}
