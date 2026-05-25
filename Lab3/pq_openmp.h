#pragma once

#include <arm_neon.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "flat_simd.h"
#include "pq_simd.h"

class PQOpenMPIndex {
public:
    PQOpenMPIndex(
        size_t M_ = 4,
        size_t Ks_ = 256,
        size_t train_iters_ = 10,
        int num_threads_ = 8
    )
        : M(M_),
          Ks(Ks_),
          train_iters(train_iters_),
          n(0),
          dim(0),
          dsub(0),
          base(nullptr),
          trained(false),
          num_threads(std::max(1, num_threads_)) {}

    // 构建 PQ 索引
    void build_pq_index(const float* base_data, size_t base_number, size_t vecdim) {
        assert(base_data != nullptr);
        assert(base_number > 0);
        assert(vecdim % M == 0);
        assert(Ks <= 256);

        base = base_data;
        n = base_number;
        dim = vecdim;
        dsub = dim / M;

        codebooks.assign(M * Ks * dsub, 0.0f);
        codes.assign(n * M, 0);

        train_codebooks();
        encode_base();

        trained = true;
    }

    // OpenMP 并行 LUT 构建

    void build_lut_openmp(const float* query, std::vector<float>& lut) const {
        assert(trained);
        assert(query != nullptr);

        if (lut.size() != M * Ks) {
            lut.resize(M * Ks);
        }

        const size_t centroid_block = 4;
        const size_t blocks_per_subspace =
            (Ks + centroid_block - 1) / centroid_block;
        const size_t total_blocks = M * blocks_per_subspace;

        int actual_threads = num_threads;
        if (actual_threads <= 0) {
            actual_threads = 1;
        }
        if (static_cast<size_t>(actual_threads) > total_blocks) {
            actual_threads = static_cast<int>(total_blocks);
        }

        // OpenMP 类中心集合并行
        // 每次迭代处理一个 4-centroid block
#pragma omp parallel for num_threads(actual_threads) schedule(static)
        for (long long task = 0; task < static_cast<long long>(total_blocks); ++task) {
            size_t m = static_cast<size_t>(task) / blocks_per_subspace;
            size_t block_id = static_cast<size_t>(task) % blocks_per_subspace;
            size_t c_start = block_id * centroid_block;

            compute_lut_centroid_block(query, lut.data(), m, c_start);
        }
    }

    void build_lut(const float* query, std::vector<float>& lut) const {
        build_lut_openmp(query, lut);
    }

    // ADC 查表累加
    float adc_distance(size_t base_id, const std::vector<float>& lut) const {
        assert(trained);
        assert(base_id < n);

        const uint8_t* code = codes.data() + base_id * M;
        float dis = 0.0f;

        for (size_t m = 0; m < M; ++m) {
            uint8_t cid = code[m];
            dis += lut[m * Ks + cid];
        }

        return dis;
    }

    std::priority_queue<std::pair<float, uint32_t>>
    search_adc(const float* query, size_t top_p) const {
        assert(trained);
        assert(query != nullptr);
        assert(top_p > 0);

        if (top_p > n) {
            top_p = n;
        }

        std::vector<float> lut;

        // LUT 阶段使用 OpenMP 并行
        build_lut_openmp(query, lut);

        std::priority_queue<std::pair<float, uint32_t>> heap;

        size_t i = 0;

        for (; i + 4 <= n; i += 4) {
            const uint8_t* code0 = codes.data() + (i + 0) * M;
            const uint8_t* code1 = codes.data() + (i + 1) * M;
            const uint8_t* code2 = codes.data() + (i + 2) * M;
            const uint8_t* code3 = codes.data() + (i + 3) * M;

            float dis0 = 0.0f;
            float dis1 = 0.0f;
            float dis2 = 0.0f;
            float dis3 = 0.0f;

            for (size_t m = 0; m < M; ++m) {
                const float* lut_m = lut.data() + m * Ks;

                dis0 += lut_m[code0[m]];
                dis1 += lut_m[code1[m]];
                dis2 += lut_m[code2[m]];
                dis3 += lut_m[code3[m]];
            }

            push_topk(heap, dis0, static_cast<uint32_t>(i + 0), top_p);
            push_topk(heap, dis1, static_cast<uint32_t>(i + 1), top_p);
            push_topk(heap, dis2, static_cast<uint32_t>(i + 2), top_p);
            push_topk(heap, dis3, static_cast<uint32_t>(i + 3), top_p);
        }

        for (; i < n; ++i) {
            const uint8_t* code = codes.data() + i * M;
            float dis = 0.0f;

            for (size_t m = 0; m < M; ++m) {
                uint8_t cid = code[m];
                dis += lut[m * Ks + cid];
            }

            push_topk(heap, dis, static_cast<uint32_t>(i), top_p);
        }

        return heap;
    }

    // Flat-SIMD 精排
    std::priority_queue<std::pair<float, uint32_t>>
    rerank_flat(
        const float* query,
        std::priority_queue<std::pair<float, uint32_t>> candidates,
        size_t k
    ) const {
        assert(trained);
        assert(query != nullptr);
        assert(k > 0);

        std::priority_queue<std::pair<float, uint32_t>> result;

        while (!candidates.empty()) {
            uint32_t id = candidates.top().second;
            candidates.pop();

            const float* x = base + static_cast<size_t>(id) * dim;

            float dot = InnerProductSIMD(x, query, dim);
            float dis = 1.0f - dot;

            push_topk(result, dis, id, k);
        }

        return result;
    }

    std::priority_queue<std::pair<float, uint32_t>>
    pq_search_openmp(const float* query, size_t k, size_t top_p) const {
        assert(trained);
        assert(query != nullptr);
        assert(k > 0);

        if (top_p < k) {
            top_p = k;
        }

        std::priority_queue<std::pair<float, uint32_t>> candidates =
            search_adc(query, top_p);

        return rerank_flat(query, candidates, k);
    }

private:
    size_t M;
    size_t Ks;
    size_t train_iters;

    size_t n;
    size_t dim;
    size_t dsub;

    const float* base;
    bool trained;

    int num_threads;

    std::vector<float> codebooks;
    std::vector<uint8_t> codes;

private:
    static inline void push_topk(
        std::priority_queue<std::pair<float, uint32_t>>& heap,
        float dis,
        uint32_t id,
        size_t k
    ) {
        if (heap.size() < k) {
            heap.emplace(dis, id);
        } else if (dis < heap.top().first) {
            heap.pop();
            heap.emplace(dis, id);
        }
    }

    const float* subvec_ptr(size_t vec_id, size_t m) const {
        return base + vec_id * dim + m * dsub;
    }

    float* centroid_ptr(size_t m, size_t c) {
        return codebooks.data() + (m * Ks + c) * dsub;
    }

    const float* centroid_ptr(size_t m, size_t c) const {
        return codebooks.data() + (m * Ks + c) * dsub;
    }

    void init_centroids(size_t m) {
        for (size_t c = 0; c < Ks; ++c) {
            size_t id = (c * n) / Ks;
            if (id >= n) {
                id = n - 1;
            }

            std::memcpy(
                centroid_ptr(m, c),
                subvec_ptr(id, m),
                dsub * sizeof(float)
            );
        }
    }

    uint8_t nearest_centroid(const float* x, size_t m) const {
        float nearest_dis = std::numeric_limits<float>::max();
        uint8_t nearest_id = 0;

        for (size_t c = 0; c < Ks; ++c) {
            float dis = l2_distance_simd(x, centroid_ptr(m, c), dsub);

            if (dis < nearest_dis) {
                nearest_dis = dis;
                nearest_id = static_cast<uint8_t>(c);
            }
        }

        return nearest_id;
    }

    void train_codebooks() {
        std::vector<uint8_t> assign(n, 0);
        std::vector<float> sums(Ks * dsub, 0.0f);
        std::vector<uint32_t> counts(Ks, 0);

        for (size_t m = 0; m < M; ++m) {
            init_centroids(m);

            for (size_t it = 0; it < train_iters; ++it) {
                std::fill(sums.begin(), sums.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                for (size_t i = 0; i < n; ++i) {
                    const float* x = subvec_ptr(i, m);
                    uint8_t cid = nearest_centroid(x, m);

                    assign[i] = cid;
                    counts[cid]++;

                    float* sum = sums.data() + static_cast<size_t>(cid) * dsub;

                    for (size_t j = 0; j < dsub; ++j) {
                        sum[j] += x[j];
                    }
                }

                for (size_t c = 0; c < Ks; ++c) {
                    float* center = centroid_ptr(m, c);

                    if (counts[c] == 0) {
                        size_t id = (c * 9973 + it * 7919 + m * 104729) % n;

                        std::memcpy(
                            center,
                            subvec_ptr(id, m),
                            dsub * sizeof(float)
                        );

                        continue;
                    }

                    float inv = 1.0f / static_cast<float>(counts[c]);
                    const float* sum = sums.data() + c * dsub;

                    for (size_t j = 0; j < dsub; ++j) {
                        center[j] = sum[j] * inv;
                    }
                }
            }
        }
    }

    void encode_base() {
        for (size_t i = 0; i < n; ++i) {
            for (size_t m = 0; m < M; ++m) {
                const float* x = subvec_ptr(i, m);
                codes[i * M + m] = nearest_centroid(x, m);
            }
        }
    }

    // LUT block 计算核心
    void compute_lut_centroid_block(
        const float* query,
        float* lut,
        size_t m,
        size_t c_start
    ) const {
        const float* qsub = query + m * dsub;
        float* lut_m = lut + m * Ks;

        // 不足 4 个 centroid 的尾部
        if (c_start + 4 > Ks) {
            for (size_t c = c_start; c < Ks; ++c) {
                const float* center = centroid_ptr(m, c);
                lut_m[c] = l2_distance_simd(qsub, center, dsub);
            }
            return;
        }

        const float* center0 = centroid_ptr(m, c_start + 0);
        const float* center1 = centroid_ptr(m, c_start + 1);
        const float* center2 = centroid_ptr(m, c_start + 2);
        const float* center3 = centroid_ptr(m, c_start + 3);

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        size_t j = 0;

        for (; j + 4 <= dsub; j += 4) {
            float32x4_t qv = vld1q_f32(qsub + j);

            float32x4_t c0 = vld1q_f32(center0 + j);
            float32x4_t c1 = vld1q_f32(center1 + j);
            float32x4_t c2 = vld1q_f32(center2 + j);
            float32x4_t c3 = vld1q_f32(center3 + j);

            float32x4_t d0 = vsubq_f32(qv, c0);
            float32x4_t d1 = vsubq_f32(qv, c1);
            float32x4_t d2 = vsubq_f32(qv, c2);
            float32x4_t d3 = vsubq_f32(qv, c3);

            acc0 = vfmaq_f32(acc0, d0, d0);
            acc1 = vfmaq_f32(acc1, d1, d1);
            acc2 = vfmaq_f32(acc2, d2, d2);
            acc3 = vfmaq_f32(acc3, d3, d3);
        }

        float dis0 = vaddvq_f32(acc0);
        float dis1 = vaddvq_f32(acc1);
        float dis2 = vaddvq_f32(acc2);
        float dis3 = vaddvq_f32(acc3);

        for (; j < dsub; ++j) {
            float q = qsub[j];

            float t0 = q - center0[j];
            float t1 = q - center1[j];
            float t2 = q - center2[j];
            float t3 = q - center3[j];

            dis0 += t0 * t0;
            dis1 += t1 * t1;
            dis2 += t2 * t2;
            dis3 += t3 * t3;
        }

        lut_m[c_start + 0] = dis0;
        lut_m[c_start + 1] = dis1;
        lut_m[c_start + 2] = dis2;
        lut_m[c_start + 3] = dis3;
    }
};