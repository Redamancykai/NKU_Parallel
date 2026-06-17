struct QuantizedInt8Data {
    std::vector<int32_t> base;
    std::vector<int32_t> queries;
    size_t packed_words = 0;
    float scale = 1.0f;
};

int8_t QuantizeI8(float value, float scale) {
    int q = static_cast<int>(std::lrint(value * scale));
    q = std::max(-127, std::min(127, q));
    return static_cast<int8_t>(q);
}

uint32_t Pack4I8(const int8_t* values) {
    uint32_t word = 0;
    for (int i = 0; i < 4; ++i) {
        word |= static_cast<uint32_t>(static_cast<uint8_t>(values[i])) << (8 * i);
    }
    return word;
}

QuantizedInt8Data QuantizeInt8(const float* base,
                               const float* queries,
                               size_t base_number,
                               size_t query_number,
                               size_t vecdim,
                               size_t search_dim) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < base_number; ++i) {
        for (size_t d = 0; d < search_dim; ++d) {
            max_abs = std::max(max_abs, std::fabs(base[i * vecdim + d]));
        }
    }
    for (size_t i = 0; i < query_number; ++i) {
        for (size_t d = 0; d < search_dim; ++d) {
            max_abs = std::max(max_abs, std::fabs(queries[i * vecdim + d]));
        }
    }

    QuantizedInt8Data out;
    out.scale = max_abs == 0.0f ? 1.0f : 127.0f / max_abs;
    out.packed_words = (search_dim + 3) / 4;
    out.base.resize(base_number * out.packed_words);
    out.queries.resize(query_number * out.packed_words);

    auto pack_vectors = [&](const float* input,
                            size_t count,
                            std::vector<int32_t>& output) {
        int8_t tmp[4];
        for (size_t i = 0; i < count; ++i) {
            for (size_t p = 0; p < out.packed_words; ++p) {
                for (int lane = 0; lane < 4; ++lane) {
                    size_t d = p * 4 + lane;
                    tmp[lane] = d < search_dim
                                    ? QuantizeI8(input[i * vecdim + d], out.scale)
                                    : 0;
                }
                output[i * out.packed_words + p] =
                    static_cast<int32_t>(Pack4I8(tmp));
            }
        }
    };

    pack_vectors(base, base_number, out.base);
    pack_vectors(queries, query_number, out.queries);
    return out;
}

struct IVFCPUIndex {
    size_t nlist = 0;
    size_t vecdim = 0;
    size_t base_number = 0;
    std::vector<float> centroids;
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> ids;
    std::vector<float> vectors;
    std::vector<float> full_base;
};

float DotProduct(const float* a, const float* b, size_t dim) {
    float dot = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        dot += a[d] * b[d];
    }
    return dot;
}

void AssignNearestCentroidsGPU(const float* base,
                               size_t base_number,
                               size_t vecdim,
                               const std::vector<float>& centroids,
                               size_t nlist,
                               std::vector<uint32_t>& assignments) {
    float* d_base = nullptr;
    float* d_centroids = nullptr;
    uint32_t* d_assignments = nullptr;
    assignments.assign(base_number, 0);

    try {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_base),
                              base_number * vecdim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centroids),
                              centroids.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_assignments),
                              base_number * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_base,
                              base,
                              base_number * vecdim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_centroids,
                              centroids.data(),
                              centroids.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        unsigned int threads = 128;
        size_t shared_bytes = threads * (sizeof(float) + sizeof(uint32_t));
        AssignNearestCentroidKernel<<<static_cast<unsigned int>(base_number),
                                      threads,
                                      shared_bytes>>>(
            d_base, d_centroids, d_assignments, base_number, nlist, vecdim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(assignments.data(),
                              d_assignments,
                              base_number * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
        cudaFree(d_base);
        cudaFree(d_centroids);
        cudaFree(d_assignments);
        throw;
    }

    cudaFree(d_base);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
}

std::string IVFCachePath(size_t base_number, size_t vecdim, size_t nlist, size_t iters) {
    return "ivf_gpu_full_index_base_" + std::to_string(base_number) +
           "_dim_" + std::to_string(vecdim) +
           "_nlist_" + std::to_string(nlist) +
           "_iters_" + std::to_string(iters) + ".bin";
}

bool LoadIVFIndex(const std::string& path, IVFCPUIndex& index) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.read(reinterpret_cast<char*>(&index.nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&index.vecdim), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&index.base_number), sizeof(size_t));
    size_t centroid_count = 0;
    size_t list_count = 0;
    size_t id_count = 0;
    size_t vector_count = 0;
    size_t full_base_count = 0;
    in.read(reinterpret_cast<char*>(&centroid_count), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&list_count), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&id_count), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&vector_count), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&full_base_count), sizeof(size_t));
    if (!in || list_count != index.nlist) {
        return false;
    }
    index.centroids.resize(centroid_count);
    index.offsets.resize(list_count);
    index.sizes.resize(list_count);
    index.ids.resize(id_count);
    index.vectors.resize(vector_count);
    index.full_base.resize(full_base_count);
    in.read(reinterpret_cast<char*>(index.centroids.data()), centroid_count * sizeof(float));
    in.read(reinterpret_cast<char*>(index.offsets.data()), list_count * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(index.sizes.data()), list_count * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(index.ids.data()), id_count * sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(index.vectors.data()), vector_count * sizeof(float));
    in.read(reinterpret_cast<char*>(index.full_base.data()), full_base_count * sizeof(float));
    return static_cast<bool>(in);
}

void SaveIVFIndex(const std::string& path, const IVFCPUIndex& index) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    size_t centroid_count = index.centroids.size();
    size_t list_count = index.offsets.size();
    size_t id_count = index.ids.size();
    size_t vector_count = index.vectors.size();
    size_t full_base_count = index.full_base.size();
    out.write(reinterpret_cast<const char*>(&index.nlist), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&index.vecdim), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&index.base_number), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&centroid_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&list_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&id_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&vector_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&full_base_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(index.centroids.data()), centroid_count * sizeof(float));
    out.write(reinterpret_cast<const char*>(index.offsets.data()), list_count * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(index.sizes.data()), list_count * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(index.ids.data()), id_count * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(index.vectors.data()), vector_count * sizeof(float));
    out.write(reinterpret_cast<const char*>(index.full_base.data()), full_base_count * sizeof(float));
}

IVFCPUIndex BuildIVFIndex(const float* base,
                          size_t base_number,
                          size_t vecdim,
                          size_t nlist,
                          size_t train_iters) {
    IVFCPUIndex index;
    std::string cache_path = IVFCachePath(base_number, vecdim, nlist, train_iters);
    if (LoadIVFIndex(cache_path, index) &&
        index.nlist == nlist &&
        index.vecdim == vecdim &&
        index.base_number == base_number &&
        index.full_base.size() == base_number * vecdim) {
        std::cerr << "[IVF] loaded cached index " << cache_path << "\n";
        return index;
    }

    std::cerr << "[IVF] building index nlist=" << nlist
              << " iters=" << train_iters << "\n";
    index.nlist = nlist;
    index.vecdim = vecdim;
    index.base_number = base_number;
    index.full_base.assign(base, base + base_number * vecdim);
    index.centroids.assign(nlist * vecdim, 0.0f);

    for (size_t c = 0; c < nlist; ++c) {
        size_t id = c * base_number / nlist;
        std::copy(base + id * vecdim, base + (id + 1) * vecdim,
                  index.centroids.begin() + c * vecdim);
    }

    std::vector<uint32_t> assign(base_number, 0);
    std::vector<float> sums(nlist * vecdim, 0.0f);
    std::vector<uint32_t> counts(nlist, 0);

    for (size_t iter = 0; iter < train_iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        AssignNearestCentroidsGPU(base, base_number, vecdim, index.centroids, nlist, assign);
        for (size_t i = 0; i < base_number; ++i) {
            uint32_t c = assign[i];
            counts[c]++;
            float* sum = sums.data() + static_cast<size_t>(c) * vecdim;
            const float* x = base + i * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                sum[d] += x[d];
            }
        }

        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }
            float inv = 1.0f / static_cast<float>(counts[c]);
            float* centroid = index.centroids.data() + c * vecdim;
            float* sum = sums.data() + c * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                centroid[d] = sum[d] * inv;
            }
        }
        std::cerr << "[IVF] train iter " << (iter + 1) << "/" << train_iters << "\n";
    }

    std::fill(counts.begin(), counts.end(), 0);
    AssignNearestCentroidsGPU(base, base_number, vecdim, index.centroids, nlist, assign);
    for (size_t i = 0; i < base_number; ++i) {
        uint32_t c = assign[i];
        counts[c]++;
    }

    index.offsets.assign(nlist, 0);
    index.sizes.assign(nlist, 0);
    uint32_t offset = 0;
    for (size_t c = 0; c < nlist; ++c) {
        index.offsets[c] = offset;
        index.sizes[c] = counts[c];
        offset += counts[c];
    }

    index.ids.assign(base_number, 0);
    index.vectors.assign(base_number * vecdim, 0.0f);
    std::vector<uint32_t> cursor = index.offsets;
    for (size_t i = 0; i < base_number; ++i) {
        uint32_t c = assign[i];
        uint32_t pos = cursor[c]++;
        index.ids[pos] = static_cast<uint32_t>(i);
        std::copy(base + i * vecdim, base + (i + 1) * vecdim,
                  index.vectors.begin() + static_cast<size_t>(pos) * vecdim);
    }

    SaveIVFIndex(cache_path, index);
    std::cerr << "[IVF] saved index " << cache_path << "\n";
    return index;
}

void SelectProbeLists(const IVFCPUIndex& index,
                      const float* queries,
                      size_t query_number,
                      size_t vecdim,
                      size_t nprobe,
                      std::vector<uint32_t>& probe_lists) {
    nprobe = std::min(nprobe, index.nlist);
    probe_lists.assign(query_number * nprobe, 0);
    std::vector<std::pair<float, uint32_t>> scores(index.nlist);
    for (size_t q = 0; q < query_number; ++q) {
        const float* query = queries + q * vecdim;
        for (size_t c = 0; c < index.nlist; ++c) {
            float dot = DotProduct(query, index.centroids.data() + c * vecdim, vecdim);
            scores[c] = {-dot, static_cast<uint32_t>(c)};
        }
        if (nprobe < scores.size()) {
            std::nth_element(scores.begin(), scores.begin() + nprobe, scores.end());
        }
        std::sort(scores.begin(), scores.begin() + nprobe);
        for (size_t p = 0; p < nprobe; ++p) {
            probe_lists[q * nprobe + p] = scores[p].second;
        }
    }
}

uint64_t PackClusterKey(const std::vector<uint32_t>& clusters, size_t begin) {
    uint64_t key = 0;
    for (size_t i = 0; i < 4; ++i) {
        size_t pos = begin + i;
        uint64_t value = pos < clusters.size()
                             ? static_cast<uint64_t>(clusters[pos])
                             : 0xffffull;
        key = (key << 16) | (value & 0xffffull);
    }
    return key;
}

std::vector<uint32_t> BuildQueryOrder(const std::vector<uint32_t>& probe_lists,
                                      size_t query_number,
                                      size_t nprobe,
                                      const std::string& strategy) {
    std::vector<uint32_t> order(query_number);
    for (size_t q = 0; q < query_number; ++q) {
        order[q] = static_cast<uint32_t>(q);
    }
    if (strategy == "none" || query_number == 0) {
        return order;
    }

    struct QueryKey {
        uint64_t key0 = 0;
        uint64_t key1 = 0;
        uint32_t query = 0;
    };
    std::vector<QueryKey> keys(query_number);
    for (size_t q = 0; q < query_number; ++q) {
        const uint32_t* probes = probe_lists.data() + q * nprobe;
        if (strategy == "primary") {
            uint32_t primary = nprobe > 0 ? probes[0] : 0;
            uint32_t secondary = nprobe > 1 ? probes[1] : 0xffffffffu;
            keys[q] = {primary, secondary, static_cast<uint32_t>(q)};
        } else {
            std::vector<uint32_t> sorted(probes, probes + nprobe);
            std::sort(sorted.begin(), sorted.end());
            keys[q] = {PackClusterKey(sorted, 0), PackClusterKey(sorted, 4),
                       static_cast<uint32_t>(q)};
        }
    }
    std::stable_sort(keys.begin(), keys.end(), [](const QueryKey& a, const QueryKey& b) {
        if (a.key0 != b.key0) {
            return a.key0 < b.key0;
        }
        if (a.key1 != b.key1) {
            return a.key1 < b.key1;
        }
        return a.query < b.query;
    });
    for (size_t i = 0; i < query_number; ++i) {
        order[i] = keys[i].query;
    }
    return order;
}

struct IVFPQCPUIndex {
    const IVFCPUIndex* ivf = nullptr;
    size_t M = 0;
    size_t Ks = 0;
    size_t dsub = 0;
    std::vector<float> codebooks;
    std::vector<uint8_t> codes;
};

std::string IVFPQCachePath(size_t base_number,
                           size_t vecdim,
                           size_t nlist,
                           size_t ivf_iters,
                           size_t M,
                           size_t Ks,
                           size_t pq_iters) {
    return "ivfpq_gpu_index_base_" + std::to_string(base_number) +
           "_dim_" + std::to_string(vecdim) +
           "_nlist_" + std::to_string(nlist) +
           "_ivfiters_" + std::to_string(ivf_iters) +
           "_M_" + std::to_string(M) +
           "_Ks_" + std::to_string(Ks) +
           "_pqiters_" + std::to_string(pq_iters) + ".bin";
}

bool LoadIVFPQIndex(const std::string& path, const IVFCPUIndex& ivf, IVFPQCPUIndex& pq) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    size_t base_number = 0;
    size_t vecdim = 0;
    size_t nlist = 0;
    size_t codebook_count = 0;
    size_t code_count = 0;
    in.read(reinterpret_cast<char*>(&base_number), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&vecdim), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&nlist), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&pq.M), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&pq.Ks), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&pq.dsub), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&codebook_count), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&code_count), sizeof(size_t));
    if (!in || base_number != ivf.base_number || vecdim != ivf.vecdim || nlist != ivf.nlist) {
        return false;
    }
    pq.ivf = &ivf;
    pq.codebooks.resize(codebook_count);
    pq.codes.resize(code_count);
    in.read(reinterpret_cast<char*>(pq.codebooks.data()), codebook_count * sizeof(float));
    in.read(reinterpret_cast<char*>(pq.codes.data()), code_count * sizeof(uint8_t));
    return static_cast<bool>(in);
}

void SaveIVFPQIndex(const std::string& path, const IVFPQCPUIndex& pq) {
    const IVFCPUIndex& ivf = *pq.ivf;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    size_t codebook_count = pq.codebooks.size();
    size_t code_count = pq.codes.size();
    out.write(reinterpret_cast<const char*>(&ivf.base_number), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&ivf.vecdim), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&ivf.nlist), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&pq.M), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&pq.Ks), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&pq.dsub), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&codebook_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&code_count), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(pq.codebooks.data()), codebook_count * sizeof(float));
    out.write(reinterpret_cast<const char*>(pq.codes.data()), code_count * sizeof(uint8_t));
}

float L2Subspace(const float* x, const float* c, size_t dsub) {
    float dist = 0.0f;
    for (size_t d = 0; d < dsub; ++d) {
        float diff = x[d] - c[d];
        dist += diff * diff;
    }
    return dist;
}

uint8_t NearestPQCentroid(const float* x,
                          const std::vector<float>& codebooks,
                          size_t M,
                          size_t Ks,
                          size_t dsub,
                          size_t m) {
    const float* book = codebooks.data() + m * Ks * dsub;
    uint8_t best = 0;
    float best_dist = L2Subspace(x, book, dsub);
    for (size_t c = 1; c < Ks; ++c) {
        float dist = L2Subspace(x, book + c * dsub, dsub);
        if (dist < best_dist) {
            best_dist = dist;
            best = static_cast<uint8_t>(c);
        }
    }
    return best;
}

IVFPQCPUIndex BuildIVFPQIndex(const IVFCPUIndex& ivf,
                              size_t M,
                              size_t Ks,
                              size_t train_iters) {
    if (ivf.vecdim % M != 0) {
        throw std::runtime_error("vecdim must be divisible by --pq-m");
    }

    std::string cache_path = IVFPQCachePath(ivf.base_number,
                                            ivf.vecdim,
                                            ivf.nlist,
                                            train_iters,
                                            M,
                                            Ks,
                                            train_iters);
    IVFPQCPUIndex cached;
    if (LoadIVFPQIndex(cache_path, ivf, cached) &&
        cached.M == M &&
        cached.Ks == Ks &&
        cached.dsub == ivf.vecdim / M &&
        cached.codebooks.size() == M * Ks * (ivf.vecdim / M) &&
        cached.codes.size() == ivf.base_number * M) {
        std::cerr << "[IVFPQ] loaded cached index " << cache_path << "\n";
        return cached;
    }

    IVFPQCPUIndex pq;
    pq.ivf = &ivf;
    pq.M = M;
    pq.Ks = Ks;
    pq.dsub = ivf.vecdim / M;
    pq.codebooks.assign(M * Ks * pq.dsub, 0.0f);
    pq.codes.assign(ivf.base_number * M, 0);

    std::vector<uint32_t> vector_centroids(ivf.base_number, 0);
    for (size_t c = 0; c < ivf.nlist; ++c) {
        size_t offset = ivf.offsets[c];
        size_t size = ivf.sizes[c];
        for (size_t i = 0; i < size; ++i) {
            vector_centroids[offset + i] = static_cast<uint32_t>(c);
        }
    }
    std::vector<float> residuals(ivf.base_number * ivf.vecdim, 0.0f);
    for (size_t i = 0; i < ivf.base_number; ++i) {
        const float* x = ivf.vectors.data() + i * ivf.vecdim;
        const float* centroid =
            ivf.centroids.data() + static_cast<size_t>(vector_centroids[i]) * ivf.vecdim;
        float* residual = residuals.data() + i * ivf.vecdim;
        for (size_t d = 0; d < ivf.vecdim; ++d) {
            residual[d] = x[d] - centroid[d];
        }
    }

    std::vector<uint8_t> assign(ivf.base_number, 0);
    std::vector<float> sums(Ks * pq.dsub, 0.0f);
    std::vector<uint32_t> counts(Ks, 0);

    std::cerr << "[IVFPQ] training PQ M=" << M << " Ks=" << Ks
              << " iters=" << train_iters << "\n";
    for (size_t m = 0; m < M; ++m) {
        for (size_t c = 0; c < Ks; ++c) {
            size_t id = c * ivf.base_number / Ks;
            const float* src = residuals.data() + id * ivf.vecdim + m * pq.dsub;
            std::copy(src,
                      src + pq.dsub,
                      pq.codebooks.begin() + (m * Ks + c) * pq.dsub);
        }

        for (size_t iter = 0; iter < train_iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);
            for (size_t i = 0; i < ivf.base_number; ++i) {
                const float* x = residuals.data() + i * ivf.vecdim + m * pq.dsub;
                uint8_t cid = NearestPQCentroid(x, pq.codebooks, M, Ks, pq.dsub, m);
                assign[i] = cid;
                counts[cid]++;
                float* sum = sums.data() + static_cast<size_t>(cid) * pq.dsub;
                for (size_t d = 0; d < pq.dsub; ++d) {
                    sum[d] += x[d];
                }
            }
            for (size_t c = 0; c < Ks; ++c) {
                if (counts[c] == 0) {
                    continue;
                }
                float inv = 1.0f / static_cast<float>(counts[c]);
                float* dst = pq.codebooks.data() + (m * Ks + c) * pq.dsub;
                const float* sum = sums.data() + c * pq.dsub;
                for (size_t d = 0; d < pq.dsub; ++d) {
                    dst[d] = sum[d] * inv;
                }
            }
        }

        for (size_t i = 0; i < ivf.base_number; ++i) {
            const float* x = residuals.data() + i * ivf.vecdim + m * pq.dsub;
            pq.codes[i * M + m] = NearestPQCentroid(x, pq.codebooks, M, Ks, pq.dsub, m);
        }
        std::cerr << "[IVFPQ] trained subspace " << (m + 1) << "/" << M << "\n";
    }
    SaveIVFPQIndex(cache_path, pq);
    std::cerr << "[IVFPQ] saved index " << cache_path << "\n";
    return pq;
}
