__device__ inline void InsertTopK(float candidate,
                                  uint32_t candidate_id,
                                  float* best_dist,
                                  uint32_t* best_id,
                                  size_t k) {
    int worst_slot = 0;
    float worst_dist = best_dist[0];
    for (size_t j = 1; j < k; ++j) {
        if (best_dist[j] > worst_dist ||
            (best_dist[j] == worst_dist && best_id[j] > best_id[worst_slot])) {
            worst_dist = best_dist[j];
            worst_slot = static_cast<int>(j);
        }
    }

    if (candidate < worst_dist ||
        (candidate == worst_dist && candidate_id < best_id[worst_slot])) {
        best_dist[worst_slot] = candidate;
        best_id[worst_slot] = candidate_id;
    }
}

__global__ void InnerProductDistanceGemmKernel(const float* __restrict__ base,
                                               const float* __restrict__ queries,
                                               float* __restrict__ distances,
                                               size_t base_number,
                                               size_t vecdim,
                                               size_t base_stride,
                                               size_t query_stride,
                                               size_t batch_count) {
    __shared__ float base_tile[kTile][kTile];
    __shared__ float query_tile[kTile][kTile];

    size_t base_row = static_cast<size_t>(blockIdx.y) * kTile + threadIdx.y;
    size_t query_col = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    float dot = 0.0f;

    for (size_t tile = 0; tile < vecdim; tile += kTile) {
        size_t base_dim = tile + threadIdx.x;
        size_t query_dim = tile + threadIdx.y;

        base_tile[threadIdx.y][threadIdx.x] =
            (base_row < base_number && base_dim < vecdim)
                ? base[base_row * base_stride + base_dim]
                : 0.0f;

        query_tile[threadIdx.y][threadIdx.x] =
            (query_col < batch_count && query_dim < vecdim)
                ? queries[query_col * query_stride + query_dim]
                : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            dot += base_tile[threadIdx.y][k] * query_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (base_row < base_number && query_col < batch_count) {
        distances[query_col * base_number + base_row] = 1.0f - dot;
    }
}

__global__ void InnerProductDistanceGemmHalfKernel(const __half* __restrict__ base,
                                                   const __half* __restrict__ queries,
                                                   float* __restrict__ distances,
                                                   size_t base_number,
                                                   size_t vecdim,
                                                   size_t base_stride,
                                                   size_t query_stride,
                                                   size_t batch_count) {
    __shared__ __half base_tile[kTile][kTile];
    __shared__ __half query_tile[kTile][kTile];

    size_t base_row = static_cast<size_t>(blockIdx.y) * kTile + threadIdx.y;
    size_t query_col = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    float dot = 0.0f;

    for (size_t tile = 0; tile < vecdim; tile += kTile) {
        size_t base_dim = tile + threadIdx.x;
        size_t query_dim = tile + threadIdx.y;

        base_tile[threadIdx.y][threadIdx.x] =
            (base_row < base_number && base_dim < vecdim)
                ? base[base_row * base_stride + base_dim]
                : __float2half(0.0f);

        query_tile[threadIdx.y][threadIdx.x] =
            (query_col < batch_count && query_dim < vecdim)
                ? queries[query_col * query_stride + query_dim]
                : __float2half(0.0f);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            dot += __half2float(base_tile[threadIdx.y][k]) *
                   __half2float(query_tile[k][threadIdx.x]);
        }

        __syncthreads();
    }

    if (base_row < base_number && query_col < batch_count) {
        distances[query_col * base_number + base_row] = 1.0f - dot;
    }
}

__global__ void AssignNearestCentroidKernel(const float* __restrict__ vectors,
                                            const float* __restrict__ centroids,
                                            uint32_t* __restrict__ assignments,
                                            size_t vector_count,
                                            size_t nlist,
                                            size_t vecdim) {
    size_t vector_id = blockIdx.x;
    if (vector_id >= vector_count) {
        return;
    }

    extern __shared__ unsigned char shared[];
    float* shared_dot = reinterpret_cast<float*>(shared);
    uint32_t* shared_id = reinterpret_cast<uint32_t*>(shared_dot + blockDim.x);

    const float* x = vectors + vector_id * vecdim;
    float best_dot = -kDeviceInf;
    uint32_t best_id = 0;
    for (size_t c = threadIdx.x; c < nlist; c += blockDim.x) {
        const float* centroid = centroids + c * vecdim;
        float dot = 0.0f;
        for (size_t d = 0; d < vecdim; ++d) {
            dot += x[d] * centroid[d];
        }
        if (dot > best_dot) {
            best_dot = dot;
            best_id = static_cast<uint32_t>(c);
        }
    }

    shared_dot[threadIdx.x] = best_dot;
    shared_id[threadIdx.x] = best_id;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other_dot = shared_dot[threadIdx.x + stride];
            uint32_t other_id = shared_id[threadIdx.x + stride];
            if (other_dot > shared_dot[threadIdx.x] ||
                (other_dot == shared_dot[threadIdx.x] && other_id < shared_id[threadIdx.x])) {
                shared_dot[threadIdx.x] = other_dot;
                shared_id[threadIdx.x] = other_id;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        assignments[vector_id] = shared_id[0];
    }
}

__global__ void ConvertFloatToHalfKernel(const float* __restrict__ input,
                                         __half* __restrict__ output,
                                         size_t count) {
    size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        output[id] = __float2half(input[id]);
    }
}

__global__ void Int8DistanceKernel(const int32_t* __restrict__ base,
                                   const int32_t* __restrict__ queries,
                                   float* __restrict__ distances,
                                   size_t base_number,
                                   size_t packed_words,
                                   size_t batch_count) {
    size_t base_row = static_cast<size_t>(blockIdx.y) * kTile + threadIdx.y;
    size_t query_col = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    if (base_row >= base_number || query_col >= batch_count) {
        return;
    }

    int acc = 0;
    const int32_t* base_vec = base + base_row * packed_words;
    const int32_t* query_vec = queries + query_col * packed_words;
    for (size_t p = 0; p < packed_words; ++p) {
        acc = __dp4a(base_vec[p], query_vec[p], acc);
    }
    distances[query_col * base_number + base_row] = -static_cast<float>(acc);
}

__global__ void InitTopKKernel(float* __restrict__ top_distances,
                               uint32_t* __restrict__ top_ids,
                               size_t batch_count,
                               size_t k) {
    size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = batch_count * k;
    if (id < total) {
        top_distances[id] = kDeviceInf;
        top_ids[id] = 0xffffffffu;
    }
}

__global__ void ClusterDistanceAllQueriesKernel(const float* __restrict__ vectors,
                                                const float* __restrict__ queries,
                                                float* __restrict__ distances,
                                                size_t list_size,
                                                size_t vecdim,
                                                size_t batch_count) {
    __shared__ float vec_tile[kTile][kTile];
    __shared__ float query_tile[kTile][kTile];

    size_t row = static_cast<size_t>(blockIdx.y) * kTile + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    float dot = 0.0f;

    for (size_t tile = 0; tile < vecdim; tile += kTile) {
        size_t vd = tile + threadIdx.x;
        size_t qd = tile + threadIdx.y;
        vec_tile[threadIdx.y][threadIdx.x] =
            (row < list_size && vd < vecdim) ? vectors[row * vecdim + vd] : 0.0f;
        query_tile[threadIdx.y][threadIdx.x] =
            (col < batch_count && qd < vecdim) ? queries[col * vecdim + qd] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            dot += vec_tile[threadIdx.y][k] * query_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < list_size && col < batch_count) {
        distances[col * list_size + row] = 1.0f - dot;
    }
}

__global__ void UnionDistanceAllQueriesKernel(const float* __restrict__ vectors,
                                              const uint32_t* __restrict__ candidate_positions,
                                              const float* __restrict__ queries,
                                              float* __restrict__ distances,
                                              size_t candidate_count,
                                              size_t vecdim,
                                              size_t batch_count) {
    __shared__ float vec_tile[kTile][kTile];
    __shared__ float query_tile[kTile][kTile];

    size_t row = static_cast<size_t>(blockIdx.y) * kTile + threadIdx.y;
    size_t col = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    float dot = 0.0f;

    uint32_t vector_pos = 0;
    if (row < candidate_count) {
        vector_pos = candidate_positions[row];
    }

    for (size_t tile = 0; tile < vecdim; tile += kTile) {
        size_t vd = tile + threadIdx.x;
        size_t qd = tile + threadIdx.y;
        vec_tile[threadIdx.y][threadIdx.x] =
            (row < candidate_count && vd < vecdim)
                ? vectors[static_cast<size_t>(vector_pos) * vecdim + vd]
                : 0.0f;
        query_tile[threadIdx.y][threadIdx.x] =
            (col < batch_count && qd < vecdim) ? queries[col * vecdim + qd] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            dot += vec_tile[threadIdx.y][k] * query_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < candidate_count && col < batch_count) {
        distances[col * candidate_count + row] = 1.0f - dot;
    }
}

__global__ void ClusterDistanceSelectedQueriesKernel(const float* __restrict__ vectors,
                                                     const float* __restrict__ queries,
                                                     const uint32_t* __restrict__ query_ids,
                                                     float* __restrict__ distances,
                                                     size_t list_size,
                                                     size_t vecdim,
                                                     size_t group_size) {
    __shared__ float vec_tile[kTile][kTile];
    __shared__ float query_tile[kTile][kTile];

    size_t row = static_cast<size_t>(blockIdx.y) * kTile + threadIdx.y;
    size_t group_col = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    float dot = 0.0f;

    uint32_t query_id = 0;
    if (group_col < group_size) {
        query_id = query_ids[group_col];
    }

    for (size_t tile = 0; tile < vecdim; tile += kTile) {
        size_t vd = tile + threadIdx.x;
        size_t qd = tile + threadIdx.y;
        vec_tile[threadIdx.y][threadIdx.x] =
            (row < list_size && vd < vecdim) ? vectors[row * vecdim + vd] : 0.0f;
        query_tile[threadIdx.y][threadIdx.x] =
            (group_col < group_size && qd < vecdim)
                ? queries[static_cast<size_t>(query_id) * vecdim + qd]
                : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            dot += vec_tile[threadIdx.y][k] * query_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < list_size && group_col < group_size) {
        distances[group_col * list_size + row] = 1.0f - dot;
    }
}

__device__ bool QueryProbesCluster(const uint32_t* probe_lists,
                                   size_t nprobe,
                                   size_t query_id,
                                   uint32_t cluster_id);

__global__ void TopKUnionFilteredKernel(const float* __restrict__ distances,
                                        const uint32_t* __restrict__ candidate_ids,
                                        const uint32_t* __restrict__ candidate_clusters,
                                        const uint32_t* __restrict__ probe_lists,
                                        uint32_t* __restrict__ out_ids,
                                        float* __restrict__ out_distances,
                                        size_t candidate_count,
                                        size_t batch_count,
                                        size_t nprobe,
                                        size_t k) {
    size_t query_id = blockIdx.x;
    if (query_id >= batch_count) {
        return;
    }

    extern __shared__ unsigned char shared[];
    float* shared_dist = reinterpret_cast<float*>(shared);
    uint32_t* shared_id = reinterpret_cast<uint32_t*>(shared_dist + blockDim.x * k);

    float local_dist[kMaxResultK];
    uint32_t local_id[kMaxResultK];
    for (int i = 0; i < kMaxResultK; ++i) {
        local_dist[i] = kDeviceInf;
        local_id[i] = 0xffffffffu;
    }

    const float* query_distances = distances + query_id * candidate_count;
    for (size_t cand = threadIdx.x; cand < candidate_count; cand += blockDim.x) {
        uint32_t cluster = candidate_clusters[cand];
        if (QueryProbesCluster(probe_lists, nprobe, query_id, cluster)) {
            InsertTopK(query_distances[cand], candidate_ids[cand], local_dist, local_id, k);
        }
    }

    size_t offset = static_cast<size_t>(threadIdx.x) * k;
    for (size_t j = 0; j < k; ++j) {
        shared_dist[offset + j] = local_dist[j];
        shared_id[offset + j] = local_id[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_dist[kMaxResultK];
        uint32_t best_id[kMaxResultK];
        for (int i = 0; i < kMaxResultK; ++i) {
            best_dist[i] = kDeviceInf;
            best_id[i] = 0xffffffffu;
        }
        for (size_t t = 0; t < blockDim.x; ++t) {
            size_t base = t * k;
            for (size_t j = 0; j < k; ++j) {
                InsertTopK(shared_dist[base + j], shared_id[base + j], best_dist, best_id, k);
            }
        }
        for (size_t j = 0; j < k; ++j) {
            out_distances[query_id * k + j] = best_dist[j];
            out_ids[query_id * k + j] = best_id[j];
        }
    }
}

__device__ bool QueryProbesCluster(const uint32_t* probe_lists,
                                   size_t nprobe,
                                   size_t query_id,
                                   uint32_t cluster_id) {
    const uint32_t* probes = probe_lists + query_id * nprobe;
    for (size_t p = 0; p < nprobe; ++p) {
        if (probes[p] == cluster_id) {
            return true;
        }
    }
    return false;
}

__global__ void UpdateTopKAllQueriesKernel(const float* __restrict__ distances,
                                           const uint32_t* __restrict__ ids,
                                           const uint32_t* __restrict__ probe_lists,
                                           float* __restrict__ top_distances,
                                           uint32_t* __restrict__ top_ids,
                                           size_t list_size,
                                           size_t batch_count,
                                           size_t nprobe,
                                           uint32_t cluster_id,
                                           size_t k) {
    size_t query_id = blockIdx.x;
    if (query_id >= batch_count ||
        !QueryProbesCluster(probe_lists, nprobe, query_id, cluster_id)) {
        return;
    }

    float best_dist[kMaxTopK];
    uint32_t best_id[kMaxTopK];
    for (int i = 0; i < kMaxTopK; ++i) {
        best_dist[i] = top_distances[query_id * k + i];
        best_id[i] = top_ids[query_id * k + i];
    }

    const float* query_dist = distances + query_id * list_size;
    for (size_t i = 0; i < list_size; ++i) {
        InsertTopK(query_dist[i], ids[i], best_dist, best_id, k);
    }

    for (size_t i = 0; i < k; ++i) {
        top_distances[query_id * k + i] = best_dist[i];
        top_ids[query_id * k + i] = best_id[i];
    }
}

__global__ void UpdateTopKSelectedQueriesKernel(const float* __restrict__ distances,
                                                const uint32_t* __restrict__ ids,
                                                const uint32_t* __restrict__ query_ids,
                                                float* __restrict__ top_distances,
                                                uint32_t* __restrict__ top_ids,
                                                size_t list_size,
                                                size_t group_size,
                                                size_t k) {
    size_t group_query = blockIdx.x;
    if (group_query >= group_size) {
        return;
    }
    uint32_t query_id = query_ids[group_query];

    float best_dist[kMaxTopK];
    uint32_t best_id[kMaxTopK];
    for (int i = 0; i < kMaxTopK; ++i) {
        best_dist[i] = top_distances[static_cast<size_t>(query_id) * k + i];
        best_id[i] = top_ids[static_cast<size_t>(query_id) * k + i];
    }

    const float* query_dist = distances + group_query * list_size;
    for (size_t i = 0; i < list_size; ++i) {
        InsertTopK(query_dist[i], ids[i], best_dist, best_id, k);
    }

    for (size_t i = 0; i < k; ++i) {
        top_distances[static_cast<size_t>(query_id) * k + i] = best_dist[i];
        top_ids[static_cast<size_t>(query_id) * k + i] = best_id[i];
    }
}

__global__ void FusedIVFTopKKernel(const float* __restrict__ vectors,
                                   const uint32_t* __restrict__ ids,
                                   const uint32_t* __restrict__ offsets,
                                   const uint32_t* __restrict__ sizes,
                                   const float* __restrict__ queries,
                                   const uint32_t* __restrict__ probe_lists,
                                   float* __restrict__ out_distances,
                                   uint32_t* __restrict__ out_ids,
                                   size_t vecdim,
                                   size_t nprobe,
                                   size_t batch_count,
                                   size_t k) {
    size_t query_id = blockIdx.x;
    if (query_id >= batch_count) {
        return;
    }

    float local_dist[kMaxResultK];
    uint32_t local_id[kMaxResultK];
    for (int i = 0; i < kMaxResultK; ++i) {
        local_dist[i] = kDeviceInf;
        local_id[i] = 0xffffffffu;
    }

    const float* query = queries + query_id * vecdim;
    const uint32_t* probes = probe_lists + query_id * nprobe;
    for (size_t p = 0; p < nprobe; ++p) {
        uint32_t cluster = probes[p];
        uint32_t list_offset = offsets[cluster];
        uint32_t list_size = sizes[cluster];
        for (uint32_t i = threadIdx.x; i < list_size; i += blockDim.x) {
            uint32_t vector_pos = list_offset + i;
            const float* vector = vectors + static_cast<size_t>(vector_pos) * vecdim;
            float dot = 0.0f;
            for (size_t d = 0; d < vecdim; ++d) {
                dot += vector[d] * query[d];
            }
            InsertTopK(1.0f - dot, ids[vector_pos], local_dist, local_id, k);
        }
    }

    extern __shared__ unsigned char shared[];
    float* shared_dist = reinterpret_cast<float*>(shared);
    uint32_t* shared_id = reinterpret_cast<uint32_t*>(shared_dist + blockDim.x * k);
    size_t offset = static_cast<size_t>(threadIdx.x) * k;
    for (size_t j = 0; j < k; ++j) {
        shared_dist[offset + j] = local_dist[j];
        shared_id[offset + j] = local_id[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_dist[kMaxResultK];
        uint32_t best_id[kMaxResultK];
        for (int i = 0; i < kMaxResultK; ++i) {
            best_dist[i] = kDeviceInf;
            best_id[i] = 0xffffffffu;
        }
        for (size_t t = 0; t < blockDim.x; ++t) {
            size_t base = t * k;
            for (size_t j = 0; j < k; ++j) {
                InsertTopK(shared_dist[base + j], shared_id[base + j], best_dist, best_id, k);
            }
        }
        for (size_t j = 0; j < k; ++j) {
            out_distances[query_id * k + j] = best_dist[j];
            out_ids[query_id * k + j] = best_id[j];
        }
    }
}

__global__ void BuildPQLUTKernel(const float* __restrict__ queries,
                                 const float* __restrict__ codebooks,
                                 float* __restrict__ lut,
                                 size_t batch_count,
                                 size_t vecdim,
                                 size_t M,
                                 size_t Ks,
                                 size_t dsub) {
    size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = batch_count * M * Ks;
    if (id >= total) {
        return;
    }
    size_t c = id % Ks;
    size_t m = (id / Ks) % M;
    size_t q = id / (M * Ks);

    const float* query = queries + q * vecdim + m * dsub;
    const float* center = codebooks + (m * Ks + c) * dsub;
    float dot = 0.0f;
    for (size_t d = 0; d < dsub; ++d) {
        dot += query[d] * center[d];
    }
    lut[id] = dot;
}

__global__ void FusedIVFPQTopKKernel(const uint8_t* __restrict__ codes,
                                     const uint32_t* __restrict__ ids,
                                     const uint32_t* __restrict__ offsets,
                                     const uint32_t* __restrict__ sizes,
                                     const float* __restrict__ centroids,
                                     const float* __restrict__ queries,
                                     const uint32_t* __restrict__ probe_lists,
                                     const float* __restrict__ lut,
                                     float* __restrict__ out_distances,
                                     uint32_t* __restrict__ out_ids,
                                     size_t vecdim,
                                     size_t nprobe,
                                     size_t batch_count,
                                     size_t M,
                                     size_t Ks,
                                     size_t candidate_count) {
    size_t query_id = blockIdx.x;
    if (query_id >= batch_count) {
        return;
    }

    extern __shared__ unsigned char shared[];
    float* query_shared = reinterpret_cast<float*>(shared);
    for (size_t d = threadIdx.x; d < vecdim; d += blockDim.x) {
        query_shared[d] = queries[query_id * vecdim + d];
    }
    __syncthreads();

    float local_dist[kPQLocalTopK];
    uint32_t local_id[kPQLocalTopK];
    for (int i = 0; i < kPQLocalTopK; ++i) {
        local_dist[i] = kDeviceInf;
        local_id[i] = 0xffffffffu;
    }

    const uint32_t* probes = probe_lists + query_id * nprobe;
    const float* query_lut = lut + query_id * M * Ks;
    float* cluster_score_shared = query_shared + vecdim;
    for (size_t p = 0; p < nprobe; ++p) {
        uint32_t cluster = probes[p];
        if (threadIdx.x == 0) {
            const float* centroid = centroids + static_cast<size_t>(cluster) * vecdim;
            float cluster_score = 0.0f;
            for (size_t d = 0; d < vecdim; ++d) {
                cluster_score += query_shared[d] * centroid[d];
            }
            cluster_score_shared[0] = cluster_score;
        }
        __syncthreads();

        uint32_t list_offset = offsets[cluster];
        uint32_t list_size = sizes[cluster];
        for (uint32_t i = threadIdx.x; i < list_size; i += blockDim.x) {
            uint32_t vector_pos = list_offset + i;
            const uint8_t* code = codes + static_cast<size_t>(vector_pos) * M;
            float score = cluster_score_shared[0];
            for (size_t m = 0; m < M; ++m) {
                score += query_lut[m * Ks + code[m]];
            }
            InsertTopK(1.0f - score, ids[vector_pos], local_dist, local_id, kPQLocalTopK);
        }
        __syncthreads();
    }

    float* shared_dist = reinterpret_cast<float*>(cluster_score_shared + 1);
    uint32_t* shared_id =
        reinterpret_cast<uint32_t*>(shared_dist + blockDim.x * kPQLocalTopK);
    size_t offset = static_cast<size_t>(threadIdx.x) * kPQLocalTopK;
    for (size_t j = 0; j < kPQLocalTopK; ++j) {
        shared_dist[offset + j] = local_dist[j];
        shared_id[offset + j] = local_id[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_dist[kMaxTopK];
        uint32_t best_id[kMaxTopK];
        for (int i = 0; i < kMaxTopK; ++i) {
            best_dist[i] = kDeviceInf;
            best_id[i] = 0xffffffffu;
        }
        for (size_t t = 0; t < blockDim.x; ++t) {
            size_t base = t * kPQLocalTopK;
            for (size_t j = 0; j < kPQLocalTopK; ++j) {
                InsertTopK(shared_dist[base + j],
                           shared_id[base + j],
                           best_dist,
                           best_id,
                           candidate_count);
            }
        }
        for (size_t j = 0; j < candidate_count; ++j) {
            out_distances[query_id * candidate_count + j] = best_dist[j];
            out_ids[query_id * candidate_count + j] = best_id[j];
        }
    }
}

__global__ void TopKKernel(const float* __restrict__ distances,
                           uint32_t* __restrict__ out_ids,
                           float* __restrict__ out_distances,
                           size_t base_number,
                           size_t batch_count,
                           size_t k) {
    size_t query_id = blockIdx.x;
    if (query_id >= batch_count) {
        return;
    }

    float best_dist[kMaxTopK];
    uint32_t best_id[kMaxTopK];

    for (int i = 0; i < kMaxTopK; ++i) {
        best_dist[i] = kDeviceInf;
        best_id[i] = 0xffffffffu;
    }

    const float* query_distances = distances + query_id * base_number;
    for (size_t base_id = 0; base_id < base_number; ++base_id) {
        float candidate = query_distances[base_id];

        InsertTopK(candidate, static_cast<uint32_t>(base_id), best_dist, best_id, k);
    }

    for (size_t j = 0; j < k; ++j) {
        out_distances[query_id * k + j] = best_dist[j];
        out_ids[query_id * k + j] = best_id[j];
    }
}

__global__ void TopKParallelKernel(const float* __restrict__ distances,
                                   uint32_t* __restrict__ out_ids,
                                   float* __restrict__ out_distances,
                                   size_t base_number,
                                   size_t batch_count,
                                   size_t k) {
    size_t query_id = blockIdx.x;
    if (query_id >= batch_count) {
        return;
    }

    extern __shared__ unsigned char shared[];
    float* shared_dist = reinterpret_cast<float*>(shared);
    uint32_t* shared_id = reinterpret_cast<uint32_t*>(shared_dist + blockDim.x * k);

    float local_dist[kMaxTopK];
    uint32_t local_id[kMaxTopK];
    for (int i = 0; i < kMaxTopK; ++i) {
        local_dist[i] = kDeviceInf;
        local_id[i] = 0xffffffffu;
    }

    const float* query_distances = distances + query_id * base_number;
    for (size_t base_id = threadIdx.x; base_id < base_number; base_id += blockDim.x) {
        InsertTopK(query_distances[base_id],
                   static_cast<uint32_t>(base_id),
                   local_dist,
                   local_id,
                   k);
    }

    size_t offset = static_cast<size_t>(threadIdx.x) * k;
    for (size_t j = 0; j < k; ++j) {
        shared_dist[offset + j] = local_dist[j];
        shared_id[offset + j] = local_id[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_dist[kMaxTopK];
        uint32_t best_id[kMaxTopK];
        for (int i = 0; i < kMaxTopK; ++i) {
            best_dist[i] = kDeviceInf;
            best_id[i] = 0xffffffffu;
        }

        for (size_t t = 0; t < blockDim.x; ++t) {
            for (size_t j = 0; j < k; ++j) {
                size_t idx = t * k + j;
                InsertTopK(shared_dist[idx], shared_id[idx], best_dist, best_id, k);
            }
        }

        for (size_t j = 0; j < k; ++j) {
            out_distances[query_id * k + j] = best_dist[j];
            out_ids[query_id * k + j] = best_id[j];
        }
    }
}

__global__ void RerankCandidatesKernel(const float* __restrict__ base,
                                       const float* __restrict__ queries,
                                       const uint32_t* __restrict__ candidate_ids,
                                       uint32_t* __restrict__ out_ids,
                                       float* __restrict__ out_distances,
                                       size_t vecdim,
                                       size_t candidate_count,
                                       size_t batch_count,
                                       size_t k) {
    size_t query_id = blockIdx.x;
    if (query_id >= batch_count) {
        return;
    }

    extern __shared__ unsigned char shared[];
    float* candidate_dist = reinterpret_cast<float*>(shared);
    uint32_t* candidate_id_shared = reinterpret_cast<uint32_t*>(candidate_dist + candidate_count);

    if (threadIdx.x < candidate_count) {
        uint32_t base_id = candidate_ids[query_id * candidate_count + threadIdx.x];
        const float* query = queries + query_id * vecdim;
        const float* base_vec = base + static_cast<size_t>(base_id) * vecdim;
        float dot = 0.0f;
        for (size_t d = 0; d < vecdim; ++d) {
            dot += base_vec[d] * query[d];
        }
        candidate_dist[threadIdx.x] = 1.0f - dot;
        candidate_id_shared[threadIdx.x] = base_id;
    }
    __syncthreads();

    float best_dist[kMaxTopK];
    uint32_t best_id[kMaxTopK];
    if (threadIdx.x == 0) {
        for (int i = 0; i < kMaxTopK; ++i) {
            best_dist[i] = kDeviceInf;
            best_id[i] = 0xffffffffu;
        }

        for (size_t c = 0; c < candidate_count; ++c) {
            InsertTopK(candidate_dist[c], candidate_id_shared[c], best_dist, best_id, k);
        }

        for (size_t j = 0; j < k; ++j) {
            out_distances[query_id * k + j] = best_dist[j];
            out_ids[query_id * k + j] = best_id[j];
        }
    }
}
