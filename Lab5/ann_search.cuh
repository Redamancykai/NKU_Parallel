std::vector<std::priority_queue<std::pair<float, uint32_t>>> CudaFlatSearchBatch(
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
    double& elapsed_ms) {
    float* d_base = nullptr;
    float* d_queries = nullptr;
    float* d_distances = nullptr;
    float* d_top_distances = nullptr;
    float* d_final_distances = nullptr;
    __half* d_base_half = nullptr;
    __half* d_queries_half = nullptr;
    uint32_t* d_top_ids = nullptr;
    uint32_t* d_final_ids = nullptr;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    std::vector<float> top_distances(batch_size * k);
    std::vector<uint32_t> top_ids(batch_size * k);
    size_t candidate_count = rerank_count == 0 ? k : rerank_count;
    bool do_rerank = rerank_count > k;

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base),
                              base_number * vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries),
                              batch_size * vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distances),
                              batch_size * base_number * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_distances),
                              batch_size * candidate_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_ids),
                              batch_size * candidate_count * sizeof(uint32_t)));
        if (do_rerank) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_final_distances),
                                  batch_size * k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_final_ids),
                                  batch_size * k * sizeof(uint32_t)));
        }
        if (mode == "fp16") {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base_half),
                                  base_number * vecdim * sizeof(__half)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries_half),
                                  batch_size * vecdim * sizeof(__half)));
        }

        CUDA_CHECK(cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float),
                              cudaMemcpyHostToDevice));
        if (mode == "fp16") {
            size_t base_values = base_number * vecdim;
            ConvertFloatToHalfKernel<<<static_cast<unsigned int>((base_values + 255) / 256), 256>>>(
                d_base, d_base_half, base_values);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t first = 0; first < query_number; first += batch_size) {
            size_t current_batch = std::min(batch_size, query_number - first);
            CUDA_CHECK(cudaMemcpy(d_queries,
                                  queries + first * vecdim,
                                  current_batch * vecdim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            if (mode == "fp16") {
                size_t query_values = current_batch * vecdim;
                ConvertFloatToHalfKernel<<<static_cast<unsigned int>((query_values + 255) / 256), 256>>>(
                    d_queries, d_queries_half, query_values);
                CUDA_CHECK(cudaGetLastError());
            }

            dim3 block(kTile, kTile);
            dim3 grid((current_batch + kTile - 1) / kTile,
                      (base_number + kTile - 1) / kTile);
            if (mode == "fp16") {
                InnerProductDistanceGemmHalfKernel<<<grid, block>>>(
                    d_base_half,
                    d_queries_half,
                    d_distances,
                    base_number,
                    search_dim,
                    vecdim,
                    vecdim,
                    current_batch);
            } else {
                InnerProductDistanceGemmKernel<<<grid, block>>>(
                    d_base,
                    d_queries,
                    d_distances,
                    base_number,
                    search_dim,
                    vecdim,
                    vecdim,
                    current_batch);
            }
            CUDA_CHECK(cudaGetLastError());

            if (mode == "baseline") {
                TopKKernel<<<static_cast<unsigned int>(current_batch), 1>>>(
                    d_distances,
                    d_top_ids,
                    d_top_distances,
                    base_number,
                    current_batch,
                    candidate_count);
            } else {
                int topk_threads = candidate_count >= 64 ? 64 : kTopKThreads;
                size_t shared_bytes = static_cast<size_t>(topk_threads) *
                                      candidate_count *
                                      (sizeof(float) + sizeof(uint32_t));
                TopKParallelKernel<<<static_cast<unsigned int>(current_batch),
                                     topk_threads,
                                     shared_bytes>>>(
                    d_distances,
                    d_top_ids,
                    d_top_distances,
                    base_number,
                    current_batch,
                    candidate_count);
            }
            CUDA_CHECK(cudaGetLastError());

            const float* copy_distances = d_top_distances;
            const uint32_t* copy_ids = d_top_ids;
            if (do_rerank) {
                size_t rerank_shared_bytes = candidate_count * (sizeof(float) + sizeof(uint32_t));
                RerankCandidatesKernel<<<static_cast<unsigned int>(current_batch),
                                         static_cast<unsigned int>(candidate_count),
                                         rerank_shared_bytes>>>(
                    d_base,
                    d_queries,
                    d_top_ids,
                    d_final_ids,
                    d_final_distances,
                    vecdim,
                    candidate_count,
                    current_batch,
                    k);
                CUDA_CHECK(cudaGetLastError());
                copy_distances = d_final_distances;
                copy_ids = d_final_ids;
            }

            CUDA_CHECK(cudaMemcpy(top_distances.data(), copy_distances,
                                  current_batch * k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(top_ids.data(), copy_ids,
                                  current_batch * k * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            for (size_t q = 0; q < current_batch; ++q) {
                auto& queue = results[first + q];
                for (size_t j = 0; j < k; ++j) {
                    queue.push({top_distances[q * k + j], top_ids[q * k + j]});
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } catch (...) {
        cudaFree(d_base);
        cudaFree(d_queries);
        cudaFree(d_distances);
        cudaFree(d_top_distances);
        cudaFree(d_final_distances);
        cudaFree(d_base_half);
        cudaFree(d_queries_half);
        cudaFree(d_top_ids);
        cudaFree(d_final_ids);
        throw;
    }

    cudaFree(d_base);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_top_distances);
    cudaFree(d_final_distances);
    cudaFree(d_base_half);
    cudaFree(d_queries_half);
    cudaFree(d_top_ids);
    cudaFree(d_final_ids);
    return results;
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> CudaFlatSearchBatchInt8(
    const QuantizedInt8Data& quantized,
    size_t base_number,
    size_t query_number,
    size_t k,
    size_t batch_size,
    double& elapsed_ms) {
    int32_t* d_base = nullptr;
    int32_t* d_queries = nullptr;
    float* d_distances = nullptr;
    float* d_top_distances = nullptr;
    uint32_t* d_top_ids = nullptr;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    std::vector<float> top_distances(batch_size * k);
    std::vector<uint32_t> top_ids(batch_size * k);

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base),
                              quantized.base.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries),
                              batch_size * quantized.packed_words * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distances),
                              batch_size * base_number * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_distances),
                              batch_size * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_ids),
                              batch_size * k * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(d_base,
                              quantized.base.data(),
                              quantized.base.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t first = 0; first < query_number; first += batch_size) {
            size_t current_batch = std::min(batch_size, query_number - first);
            CUDA_CHECK(cudaMemcpy(d_queries,
                                  quantized.queries.data() + first * quantized.packed_words,
                                  current_batch * quantized.packed_words * sizeof(int32_t),
                                  cudaMemcpyHostToDevice));

            dim3 block(kTile, kTile);
            dim3 grid((current_batch + kTile - 1) / kTile,
                      (base_number + kTile - 1) / kTile);
            Int8DistanceKernel<<<grid, block>>>(
                d_base, d_queries, d_distances, base_number, quantized.packed_words, current_batch);
            CUDA_CHECK(cudaGetLastError());

            size_t shared_bytes = kTopKThreads * k * (sizeof(float) + sizeof(uint32_t));
            TopKParallelKernel<<<static_cast<unsigned int>(current_batch),
                                 kTopKThreads,
                                 shared_bytes>>>(
                d_distances, d_top_ids, d_top_distances, base_number, current_batch, k);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(top_distances.data(), d_top_distances,
                                  current_batch * k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(top_ids.data(), d_top_ids,
                                  current_batch * k * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            for (size_t q = 0; q < current_batch; ++q) {
                auto& queue = results[first + q];
                for (size_t j = 0; j < k; ++j) {
                    queue.push({top_distances[q * k + j], top_ids[q * k + j]});
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } catch (...) {
        cudaFree(d_base);
        cudaFree(d_queries);
        cudaFree(d_distances);
        cudaFree(d_top_distances);
        cudaFree(d_top_ids);
        throw;
    }

    cudaFree(d_base);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_top_distances);
    cudaFree(d_top_ids);
    return results;
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> CudaIVFSearchBatch(
    const IVFCPUIndex& index,
    const float* queries,
    const std::vector<uint32_t>& all_probe_lists,
    size_t query_number,
    size_t k,
    size_t batch_size,
    size_t nprobe,
    const std::vector<uint32_t>& query_order,
    bool grouped,
    double& elapsed_ms) {
    float* d_vectors = nullptr;
    float* d_queries = nullptr;
    float* d_distances = nullptr;
    float* d_top_distances = nullptr;
    uint32_t* d_ids = nullptr;
    uint32_t* d_probe_lists = nullptr;
    uint32_t* d_query_ids = nullptr;
    uint32_t* d_top_ids = nullptr;

    size_t max_list_size = 0;
    for (uint32_t s : index.sizes) {
        max_list_size = std::max(max_list_size, static_cast<size_t>(s));
    }

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    std::vector<float> top_distances(batch_size * k);
    std::vector<uint32_t> top_ids(batch_size * k);
    std::vector<float> batch_queries(batch_size * index.vecdim);
    std::vector<uint32_t> batch_probe_lists(batch_size * nprobe);

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vectors),
                              index.vectors.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ids),
                              index.ids.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries),
                              batch_size * index.vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distances),
                              max_list_size * batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_distances),
                              batch_size * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_ids),
                              batch_size * k * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probe_lists),
                              batch_size * nprobe * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query_ids),
                              batch_size * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(d_vectors,
                              index.vectors.data(),
                              index.vectors.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids,
                              index.ids.data(),
                              index.ids.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t first = 0; first < query_number; first += batch_size) {
            size_t current_batch = std::min(batch_size, query_number - first);
            for (size_t q = 0; q < current_batch; ++q) {
                uint32_t original_q = query_order[first + q];
                std::copy(queries + static_cast<size_t>(original_q) * index.vecdim,
                          queries + (static_cast<size_t>(original_q) + 1) * index.vecdim,
                          batch_queries.begin() + q * index.vecdim);
                std::copy(all_probe_lists.begin() + static_cast<size_t>(original_q) * nprobe,
                          all_probe_lists.begin() + (static_cast<size_t>(original_q) + 1) * nprobe,
                          batch_probe_lists.begin() + q * nprobe);
            }
            CUDA_CHECK(cudaMemcpy(d_queries,
                                  batch_queries.data(),
                                  current_batch * index.vecdim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_probe_lists,
                                  batch_probe_lists.data(),
                                  current_batch * nprobe * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            InitTopKKernel<<<static_cast<unsigned int>((current_batch * k + 255) / 256), 256>>>(
                d_top_distances, d_top_ids, current_batch, k);
            CUDA_CHECK(cudaGetLastError());

            std::vector<uint32_t> union_clusters;
            union_clusters.reserve(current_batch * nprobe);
            std::vector<std::vector<uint32_t>> cluster_queries;
            if (grouped) {
                cluster_queries.assign(index.nlist, {});
            }

            std::vector<uint8_t> seen(index.nlist, 0);
            for (size_t q = 0; q < current_batch; ++q) {
                for (size_t p = 0; p < nprobe; ++p) {
                    uint32_t c = batch_probe_lists[q * nprobe + p];
                    if (!seen[c]) {
                        seen[c] = 1;
                        union_clusters.push_back(c);
                    }
                    if (grouped) {
                        cluster_queries[c].push_back(static_cast<uint32_t>(q));
                    }
                }
            }

            for (uint32_t c : union_clusters) {
                size_t list_size = index.sizes[c];
                if (list_size == 0) {
                    continue;
                }
                const float* vectors_ptr =
                    d_vectors + static_cast<size_t>(index.offsets[c]) * index.vecdim;
                const uint32_t* ids_ptr = d_ids + index.offsets[c];

                if (!grouped) {
                    dim3 block(kTile, kTile);
                    dim3 grid((current_batch + kTile - 1) / kTile,
                              (list_size + kTile - 1) / kTile);
                    ClusterDistanceAllQueriesKernel<<<grid, block>>>(
                        vectors_ptr,
                        d_queries,
                        d_distances,
                        list_size,
                        index.vecdim,
                        current_batch);
                    CUDA_CHECK(cudaGetLastError());
                    UpdateTopKAllQueriesKernel<<<static_cast<unsigned int>(current_batch), 1>>>(
                        d_distances,
                        ids_ptr,
                        d_probe_lists,
                        d_top_distances,
                        d_top_ids,
                        list_size,
                        current_batch,
                        nprobe,
                        c,
                        k);
                } else {
                    const std::vector<uint32_t>& qids = cluster_queries[c];
                    size_t group_size = qids.size();
                    if (group_size == 0) {
                        continue;
                    }
                    CUDA_CHECK(cudaMemcpy(d_query_ids,
                                          qids.data(),
                                          group_size * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice));
                    dim3 block(kTile, kTile);
                    dim3 grid((group_size + kTile - 1) / kTile,
                              (list_size + kTile - 1) / kTile);
                    ClusterDistanceSelectedQueriesKernel<<<grid, block>>>(
                        vectors_ptr,
                        d_queries,
                        d_query_ids,
                        d_distances,
                        list_size,
                        index.vecdim,
                        group_size);
                    CUDA_CHECK(cudaGetLastError());
                    UpdateTopKSelectedQueriesKernel<<<static_cast<unsigned int>(group_size), 1>>>(
                        d_distances,
                        ids_ptr,
                        d_query_ids,
                        d_top_distances,
                        d_top_ids,
                        list_size,
                        group_size,
                        k);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            CUDA_CHECK(cudaMemcpy(top_distances.data(),
                                  d_top_distances,
                                  current_batch * k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(top_ids.data(),
                                  d_top_ids,
                                  current_batch * k * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            for (size_t q = 0; q < current_batch; ++q) {
                auto& heap = results[query_order[first + q]];
                for (size_t j = 0; j < k; ++j) {
                    heap.push({top_distances[q * k + j], top_ids[q * k + j]});
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } catch (...) {
        cudaFree(d_vectors);
        cudaFree(d_queries);
        cudaFree(d_distances);
        cudaFree(d_top_distances);
        cudaFree(d_ids);
        cudaFree(d_probe_lists);
        cudaFree(d_query_ids);
        cudaFree(d_top_ids);
        throw;
    }

    cudaFree(d_vectors);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_top_distances);
    cudaFree(d_ids);
    cudaFree(d_probe_lists);
    cudaFree(d_query_ids);
    cudaFree(d_top_ids);
    return results;
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> CudaIVFFusedSearchBatch(
    const IVFCPUIndex& index,
    const float* queries,
    const std::vector<uint32_t>& all_probe_lists,
    const std::vector<uint32_t>& query_order,
    size_t query_number,
    size_t k,
    size_t batch_size,
    size_t nprobe,
    double& elapsed_ms) {
    float* d_vectors = nullptr;
    float* d_queries = nullptr;
    float* d_top_distances = nullptr;
    uint32_t* d_ids = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_sizes = nullptr;
    uint32_t* d_probe_lists = nullptr;
    uint32_t* d_top_ids = nullptr;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    std::vector<float> top_distances(batch_size * k);
    std::vector<uint32_t> top_ids(batch_size * k);
    std::vector<float> batch_queries(batch_size * index.vecdim);
    std::vector<uint32_t> batch_probe_lists(batch_size * nprobe);

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vectors),
                              index.vectors.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ids),
                              index.ids.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_offsets),
                              index.offsets.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sizes),
                              index.sizes.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries),
                              batch_size * index.vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probe_lists),
                              batch_size * nprobe * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_distances),
                              batch_size * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_ids),
                              batch_size * k * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(d_vectors,
                              index.vectors.data(),
                              index.vectors.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids,
                              index.ids.data(),
                              index.ids.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets,
                              index.offsets.data(),
                              index.offsets.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sizes,
                              index.sizes.data(),
                              index.sizes.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t first = 0; first < query_number; first += batch_size) {
            size_t current_batch = std::min(batch_size, query_number - first);
            for (size_t q = 0; q < current_batch; ++q) {
                uint32_t original_q = query_order[first + q];
                std::copy(queries + static_cast<size_t>(original_q) * index.vecdim,
                          queries + (static_cast<size_t>(original_q) + 1) * index.vecdim,
                          batch_queries.begin() + q * index.vecdim);
                std::copy(all_probe_lists.begin() + static_cast<size_t>(original_q) * nprobe,
                          all_probe_lists.begin() + (static_cast<size_t>(original_q) + 1) * nprobe,
                          batch_probe_lists.begin() + q * nprobe);
            }

            CUDA_CHECK(cudaMemcpy(d_queries,
                                  batch_queries.data(),
                                  current_batch * index.vecdim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_probe_lists,
                                  batch_probe_lists.data(),
                                  current_batch * nprobe * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));

            size_t shared_bytes = kTopKThreads * k * (sizeof(float) + sizeof(uint32_t));
            FusedIVFTopKKernel<<<static_cast<unsigned int>(current_batch),
                                 kTopKThreads,
                                 shared_bytes>>>(
                d_vectors,
                d_ids,
                d_offsets,
                d_sizes,
                d_queries,
                d_probe_lists,
                d_top_distances,
                d_top_ids,
                index.vecdim,
                nprobe,
                current_batch,
                k);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(top_distances.data(),
                                  d_top_distances,
                                  current_batch * k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(top_ids.data(),
                                  d_top_ids,
                                  current_batch * k * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            for (size_t q = 0; q < current_batch; ++q) {
                auto& heap = results[query_order[first + q]];
                for (size_t j = 0; j < k; ++j) {
                    heap.push({top_distances[q * k + j], top_ids[q * k + j]});
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } catch (...) {
        cudaFree(d_vectors);
        cudaFree(d_queries);
        cudaFree(d_top_distances);
        cudaFree(d_ids);
        cudaFree(d_offsets);
        cudaFree(d_sizes);
        cudaFree(d_probe_lists);
        cudaFree(d_top_ids);
        throw;
    }

    cudaFree(d_vectors);
    cudaFree(d_queries);
    cudaFree(d_top_distances);
    cudaFree(d_ids);
    cudaFree(d_offsets);
    cudaFree(d_sizes);
    cudaFree(d_probe_lists);
    cudaFree(d_top_ids);
    return results;
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> CudaIVFUnionSearchBatch(
    const IVFCPUIndex& index,
    const float* queries,
    const std::vector<uint32_t>& all_probe_lists,
    const std::vector<uint32_t>& query_order,
    size_t query_number,
    size_t k,
    size_t batch_size,
    size_t nprobe,
    double& elapsed_ms) {
    float* d_vectors = nullptr;
    float* d_queries = nullptr;
    float* d_distances = nullptr;
    float* d_top_distances = nullptr;
    uint32_t* d_probe_lists = nullptr;
    uint32_t* d_candidate_positions = nullptr;
    uint32_t* d_candidate_ids = nullptr;
    uint32_t* d_candidate_clusters = nullptr;
    uint32_t* d_top_ids = nullptr;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    std::vector<float> top_distances(batch_size * k);
    std::vector<uint32_t> top_ids(batch_size * k);
    std::vector<float> batch_queries(batch_size * index.vecdim);
    std::vector<uint32_t> batch_probe_lists(batch_size * nprobe);
    std::vector<uint32_t> candidate_positions;
    std::vector<uint32_t> candidate_ids;
    std::vector<uint32_t> candidate_clusters;
    candidate_positions.reserve(index.base_number);
    candidate_ids.reserve(index.base_number);
    candidate_clusters.reserve(index.base_number);

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vectors),
                              index.vectors.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries),
                              batch_size * index.vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probe_lists),
                              batch_size * nprobe * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidate_positions),
                              index.base_number * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidate_ids),
                              index.base_number * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidate_clusters),
                              index.base_number * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_distances),
                              index.base_number * batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_distances),
                              batch_size * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_ids),
                              batch_size * k * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(d_vectors,
                              index.vectors.data(),
                              index.vectors.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t first = 0; first < query_number; first += batch_size) {
            size_t current_batch = std::min(batch_size, query_number - first);
            for (size_t q = 0; q < current_batch; ++q) {
                uint32_t original_q = query_order[first + q];
                std::copy(queries + static_cast<size_t>(original_q) * index.vecdim,
                          queries + (static_cast<size_t>(original_q) + 1) * index.vecdim,
                          batch_queries.begin() + q * index.vecdim);
                std::copy(all_probe_lists.begin() + static_cast<size_t>(original_q) * nprobe,
                          all_probe_lists.begin() + (static_cast<size_t>(original_q) + 1) * nprobe,
                          batch_probe_lists.begin() + q * nprobe);
            }

            std::vector<uint8_t> seen(index.nlist, 0);
            std::vector<uint32_t> union_clusters;
            union_clusters.reserve(current_batch * nprobe);
            for (size_t q = 0; q < current_batch; ++q) {
                for (size_t p = 0; p < nprobe; ++p) {
                    uint32_t c = batch_probe_lists[q * nprobe + p];
                    if (!seen[c]) {
                        seen[c] = 1;
                        union_clusters.push_back(c);
                    }
                }
            }

            candidate_positions.clear();
            candidate_ids.clear();
            candidate_clusters.clear();
            for (uint32_t c : union_clusters) {
                uint32_t offset = index.offsets[c];
                uint32_t list_size = index.sizes[c];
                for (uint32_t i = 0; i < list_size; ++i) {
                    uint32_t pos = offset + i;
                    candidate_positions.push_back(pos);
                    candidate_ids.push_back(index.ids[pos]);
                    candidate_clusters.push_back(c);
                }
            }
            size_t candidate_count = candidate_positions.size();
            if (candidate_count == 0) {
                throw std::runtime_error("empty IVF candidate union");
            }

            CUDA_CHECK(cudaMemcpy(d_queries,
                                  batch_queries.data(),
                                  current_batch * index.vecdim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_probe_lists,
                                  batch_probe_lists.data(),
                                  current_batch * nprobe * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_candidate_positions,
                                  candidate_positions.data(),
                                  candidate_count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_candidate_ids,
                                  candidate_ids.data(),
                                  candidate_count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_candidate_clusters,
                                  candidate_clusters.data(),
                                  candidate_count * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));

            dim3 block(kTile, kTile);
            dim3 grid((current_batch + kTile - 1) / kTile,
                      (candidate_count + kTile - 1) / kTile);
            UnionDistanceAllQueriesKernel<<<grid, block>>>(
                d_vectors,
                d_candidate_positions,
                d_queries,
                d_distances,
                candidate_count,
                index.vecdim,
                current_batch);
            CUDA_CHECK(cudaGetLastError());

            size_t topk_shared = kTopKThreads * k * (sizeof(float) + sizeof(uint32_t));
            TopKUnionFilteredKernel<<<static_cast<unsigned int>(current_batch),
                                      kTopKThreads,
                                      topk_shared>>>(
                d_distances,
                d_candidate_ids,
                d_candidate_clusters,
                d_probe_lists,
                d_top_ids,
                d_top_distances,
                candidate_count,
                current_batch,
                nprobe,
                k);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(top_distances.data(),
                                  d_top_distances,
                                  current_batch * k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(top_ids.data(),
                                  d_top_ids,
                                  current_batch * k * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            for (size_t q = 0; q < current_batch; ++q) {
                auto& heap = results[query_order[first + q]];
                for (size_t j = 0; j < k; ++j) {
                    heap.push({top_distances[q * k + j], top_ids[q * k + j]});
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } catch (...) {
        cudaFree(d_vectors);
        cudaFree(d_queries);
        cudaFree(d_distances);
        cudaFree(d_top_distances);
        cudaFree(d_probe_lists);
        cudaFree(d_candidate_positions);
        cudaFree(d_candidate_ids);
        cudaFree(d_candidate_clusters);
        cudaFree(d_top_ids);
        throw;
    }

    cudaFree(d_vectors);
    cudaFree(d_queries);
    cudaFree(d_distances);
    cudaFree(d_top_distances);
    cudaFree(d_probe_lists);
    cudaFree(d_candidate_positions);
    cudaFree(d_candidate_ids);
    cudaFree(d_candidate_clusters);
    cudaFree(d_top_ids);
    return results;
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> CudaIVFPQSearchBatch(
    const IVFPQCPUIndex& pq,
    const float* base,
    const float* queries,
    const std::vector<uint32_t>& all_probe_lists,
    size_t query_number,
    size_t k,
    size_t batch_size,
    size_t nprobe,
    const std::vector<uint32_t>& query_order,
    size_t rerank_count,
    double& elapsed_ms) {
    const IVFCPUIndex& ivf = *pq.ivf;
    float* d_base = nullptr;
    uint8_t* d_codes = nullptr;
    float* d_codebooks = nullptr;
    float* d_centroids = nullptr;
    float* d_queries = nullptr;
    float* d_lut = nullptr;
    float* d_top_distances = nullptr;
    float* d_final_distances = nullptr;
    uint32_t* d_ids = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_sizes = nullptr;
    uint32_t* d_probe_lists = nullptr;
    uint32_t* d_top_ids = nullptr;
    uint32_t* d_final_ids = nullptr;

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_number);
    std::vector<float> top_distances(batch_size * k);
    std::vector<uint32_t> top_ids(batch_size * k);
    std::vector<float> batch_queries(batch_size * ivf.vecdim);
    std::vector<uint32_t> batch_probe_lists(batch_size * nprobe);
    size_t candidate_count = rerank_count > k ? rerank_count : k;
    bool do_rerank = candidate_count > k;

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base),
                              ivf.base_number * ivf.vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_codes),
                              pq.codes.size() * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_codebooks),
                              pq.codebooks.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centroids),
                              ivf.centroids.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ids),
                              ivf.ids.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_offsets),
                              ivf.offsets.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sizes),
                              ivf.sizes.size() * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_queries),
                              batch_size * ivf.vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_probe_lists),
                              batch_size * nprobe * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lut),
                              batch_size * pq.M * pq.Ks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_distances),
                              batch_size * candidate_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_top_ids),
                              batch_size * candidate_count * sizeof(uint32_t)));
        if (do_rerank) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_final_distances),
                                  batch_size * k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_final_ids),
                                  batch_size * k * sizeof(uint32_t)));
        }
        CUDA_CHECK(cudaMemcpy(d_base,
                              base,
                              ivf.base_number * ivf.vecdim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_codes, pq.codes.data(), pq.codes.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_codebooks,
                              pq.codebooks.data(),
                              pq.codebooks.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_centroids,
                              ivf.centroids.data(),
                              ivf.centroids.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids,
                              ivf.ids.data(),
                              ivf.ids.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets,
                              ivf.offsets.data(),
                              ivf.offsets.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sizes,
                              ivf.sizes.data(),
                              ivf.sizes.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t first = 0; first < query_number; first += batch_size) {
            size_t current_batch = std::min(batch_size, query_number - first);
            for (size_t q = 0; q < current_batch; ++q) {
                uint32_t original_q = query_order[first + q];
                std::copy(queries + static_cast<size_t>(original_q) * ivf.vecdim,
                          queries + (static_cast<size_t>(original_q) + 1) * ivf.vecdim,
                          batch_queries.begin() + q * ivf.vecdim);
                std::copy(all_probe_lists.begin() + static_cast<size_t>(original_q) * nprobe,
                          all_probe_lists.begin() + (static_cast<size_t>(original_q) + 1) * nprobe,
                          batch_probe_lists.begin() + q * nprobe);
            }
            CUDA_CHECK(cudaMemcpy(d_queries,
                                  batch_queries.data(),
                                  current_batch * ivf.vecdim * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_probe_lists,
                                  batch_probe_lists.data(),
                                  current_batch * nprobe * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice));

            size_t lut_total = current_batch * pq.M * pq.Ks;
            BuildPQLUTKernel<<<static_cast<unsigned int>((lut_total + 255) / 256), 256>>>(
                d_queries, d_codebooks, d_lut, current_batch, ivf.vecdim, pq.M, pq.Ks, pq.dsub);
            CUDA_CHECK(cudaGetLastError());

            size_t fused_shared_bytes =
                (ivf.vecdim + 1) * sizeof(float) +
                kPQTopKThreads * kPQLocalTopK * (sizeof(float) + sizeof(uint32_t));
            FusedIVFPQTopKKernel<<<static_cast<unsigned int>(current_batch),
                                    kPQTopKThreads,
                                    fused_shared_bytes>>>(
                d_codes,
                d_ids,
                d_offsets,
                d_sizes,
                d_centroids,
                d_queries,
                d_probe_lists,
                d_lut,
                d_top_distances,
                d_top_ids,
                ivf.vecdim,
                nprobe,
                current_batch,
                pq.M,
                pq.Ks,
                candidate_count);
            CUDA_CHECK(cudaGetLastError());

            const float* copy_distances = d_top_distances;
            const uint32_t* copy_ids = d_top_ids;
            if (do_rerank) {
                size_t rerank_shared_bytes = candidate_count * (sizeof(float) + sizeof(uint32_t));
                RerankCandidatesKernel<<<static_cast<unsigned int>(current_batch),
                                         static_cast<unsigned int>(candidate_count),
                                         rerank_shared_bytes>>>(
                    d_base,
                    d_queries,
                    d_top_ids,
                    d_final_ids,
                    d_final_distances,
                    ivf.vecdim,
                    candidate_count,
                    current_batch,
                    k);
                CUDA_CHECK(cudaGetLastError());
                copy_distances = d_final_distances;
                copy_ids = d_final_ids;
            }

            CUDA_CHECK(cudaMemcpy(top_distances.data(),
                                  copy_distances,
                                  current_batch * k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(top_ids.data(),
                                  copy_ids,
                                  current_batch * k * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

            for (size_t q = 0; q < current_batch; ++q) {
                auto& heap = results[query_order[first + q]];
                for (size_t j = 0; j < k; ++j) {
                    heap.push({top_distances[q * k + j], top_ids[q * k + j]});
                }
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    } catch (...) {
        cudaFree(d_base);
        cudaFree(d_codes);
        cudaFree(d_codebooks);
        cudaFree(d_centroids);
        cudaFree(d_queries);
        cudaFree(d_lut);
        cudaFree(d_top_distances);
        cudaFree(d_final_distances);
        cudaFree(d_ids);
        cudaFree(d_offsets);
        cudaFree(d_sizes);
        cudaFree(d_probe_lists);
        cudaFree(d_top_ids);
        cudaFree(d_final_ids);
        throw;
    }

    cudaFree(d_base);
    cudaFree(d_codes);
    cudaFree(d_codebooks);
    cudaFree(d_centroids);
    cudaFree(d_queries);
    cudaFree(d_lut);
    cudaFree(d_top_distances);
    cudaFree(d_final_distances);
    cudaFree(d_ids);
    cudaFree(d_offsets);
    cudaFree(d_sizes);
    cudaFree(d_probe_lists);
    cudaFree(d_top_ids);
    cudaFree(d_final_ids);
    return results;
}
