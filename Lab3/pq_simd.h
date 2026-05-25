# pragma once
#include <arm_neon.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>
#include <queue>

// 用于聚类的 L2 Distance，使用 SIMD 指令集优化
static inline float l2_distance_simd(const float *a, const float *b, size_t dim) {
    size_t i = 0;

    // 采用二路累加器加速运算
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);

    // 8 路循环展开，一次处理 8 个 float
    for (; i + 8 <= dim; i += 8) {
        // 加载 a，b 的前 4 个 float，使用 vsubq_f32 并行计算差值
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t d0 = vsubq_f32(a0, b0);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t d1 = vsubq_f32(a1, b1);

        // fma 指令并行计算平方和
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }

    float32x4_t acc = vaddq_f32(acc0, acc1);

    float dis = vaddvq_f32(acc);

    for (; i < dim; ++i) {
        float t = a[i] - b[i];
        dis += t * t;
    }

    return dis;
}

class PQIndex{
public:
    PQIndex(size_t M_ = 4, size_t Ks_ = 256, size_t train_iters_ = 10)
        : M(M_), Ks(Ks_), train_iters(train_iters_),
          n(0), dim(0), dsub(0), base(nullptr), trained(false) {}

    // 构建 PQ 索引
    void build_pq_index(const float *base_data, size_t base_number, size_t vecdim) {
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

    // 手工展开 4 个 centroid 构建 LUT 索引 
    void build_lut(const float *query, std::vector<float> &lut) const {
        assert(trained);
        assert(query != nullptr);

        lut.assign(M * Ks, 0.0f);

        // 遍历每个子空间 m
        for (size_t m = 0; m < M; ++m) {
            const float *qsub = query + m * dsub;
            float *lut_m = lut.data() + m * Ks;

            size_t c = 0;

            // 一次计算 4 个 centroid 的距离
            for (; c + 4 <= Ks; c += 4) {
                const float *center0 = centroid_ptr(m, c + 0);
                const float *center1 = centroid_ptr(m, c + 1);
                const float *center2 = centroid_ptr(m, c + 2);
                const float *center3 = centroid_ptr(m, c + 3);

                // 为 4 个聚类中心分别初始化累加器，并行计算 4 个 float 的平方和
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);

                size_t j = 0;

                // 遍历子向量内部维度，四路展开
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

                // 对每个聚类中心的累加器横向求和，得到最终 L2 距离平方
                float dis0 = vaddvq_f32(acc0);
                float dis1 = vaddvq_f32(acc1);
                float dis2 = vaddvq_f32(acc2);
                float dis3 = vaddvq_f32(acc3);

                // 处理 dsub 不是 4 的倍数的尾部
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

                lut_m[c + 0] = dis0;
                lut_m[c + 1] = dis1;
                lut_m[c + 2] = dis2;
                lut_m[c + 3] = dis3;
            }

            // 处理剩余不足 4 个 centroid
            for (; c < Ks; ++c) {
                const float *center = centroid_ptr(m, c);
                lut_m[c] = l2_distance_simd(qsub, center, dsub);
            }
        }
    }

    // 计算 ADC 距离
    float adc_distance(size_t base_id, const std::vector<float> &lut) const {
        const uint8_t *code = codes.data() + base_id * M;
        float dis = 0.0f;

        for (size_t m = 0; m < M; ++m) {
            uint8_t cid = code[m];
            dis += lut[m * Ks + cid];
        }

        return dis;
    }

    // ADC 粗排
    std::priority_queue<std::pair<float, uint32_t>> search_adc(const float *query, size_t top_p) const {
        assert(trained);
        assert(query != nullptr);

        if (top_p > n) {
            top_p = n;
        }

        std::vector<float> lut;
        build_lut(query, lut);
        std::priority_queue<std::pair<float, uint32_t>> heap;

        size_t i = 0;
        // 一次处理 4 个 base 向量
        for (; i + 4 <= n; i += 4) {
            // 获取连续 4 个库向量的 PQ 编码指针
            const uint8_t *code0 = codes.data() + (i + 0) * M;
            const uint8_t *code1 = codes.data() + (i + 1) * M;
            const uint8_t *code2 = codes.data() + (i + 2) * M;
            const uint8_t *code3 = codes.data() + (i + 3) * M;

            // 初始化 4 个向量的 PQ 距离
            float dis0 = 0.0f;
            float dis1 = 0.0f;
            float dis2 = 0.0f;
            float dis3 = 0.0f;

            // 遍历所有 M 个 PQ 子空间，累加每个子空间的查表距离
            for (size_t m = 0; m < M; ++m) {
                const float *lut_m = lut.data() + m * Ks;
                dis0 += lut_m[code0[m]];
                dis1 += lut_m[code1[m]];
                dis2 += lut_m[code2[m]];
                dis3 += lut_m[code3[m]];
            }

            // 生成 4 个向量的原始 ID
            uint32_t id0 = static_cast<uint32_t>(i + 0);
            uint32_t id1 = static_cast<uint32_t>(i + 1);
            uint32_t id2 = static_cast<uint32_t>(i + 2);
            uint32_t id3 = static_cast<uint32_t>(i + 3);

            // 维护最大堆
            if (heap.size() < top_p) {
                heap.push(std::make_pair(dis0, id0));
            } 
            else if (dis0 < heap.top().first) {
                heap.pop();
                heap.push(std::make_pair(dis0, id0));
            }

            if (heap.size() < top_p) {
                heap.push(std::make_pair(dis1, id1));
            } 
            else if (dis1 < heap.top().first) {
                heap.pop();
                heap.push(std::make_pair(dis1, id1));
            }
            
            if (heap.size() < top_p) {
                heap.push(std::make_pair(dis2, id2));
            } 
            else if (dis2 < heap.top().first) {
                heap.pop();
                heap.push(std::make_pair(dis2, id2));
            }

            if (heap.size() < top_p) {
                heap.push(std::make_pair(dis3, id3));
            } 
            else if (dis3 < heap.top().first) {
                heap.pop();
                heap.push(std::make_pair(dis3, id3));
            }
        }

        // 处理剩余不足 4 个的 base 向量
        for (; i < n; ++i) {
            const uint8_t *code = codes.data() + i * M;
            float dis = 0.0f;

            for (size_t m = 0; m < M; ++m) {
                uint8_t cid = code[m];
                dis += lut[m * Ks + cid];
            }

            uint32_t id = static_cast<uint32_t>(i);
            
            if (heap.size() < top_p) {
                heap.push(std::make_pair(dis, id));
            } 
            else if (dis < heap.top().first) {
                heap.pop();
                heap.push(std::make_pair(dis, id));
            }
        }

        return heap;
    }

    // ADC 精排
    std::priority_queue<std::pair<float, uint32_t>>
    rerank_flat(
        const float *query,
        std::priority_queue<std::pair<float, uint32_t>> candidates,
        size_t k
    ) const {
        assert(trained);
        assert(query != nullptr);

        std::priority_queue<std::pair<float, uint32_t>> result;

        while (!candidates.empty()) {
            uint32_t id = candidates.top().second;
            candidates.pop();

            const float *x = base + static_cast<size_t>(id) * dim;

            float dis = InnerProductSIMDNeon_Unroll32(x, query, dim);

            if (result.size() < k) {
                result.push(std::make_pair(dis, id));

            }
            else if (dis < result.top().first) {
                result.pop();
                result.push(std::make_pair(dis, id));
            }
        }

        return result;
    }

    std::priority_queue<std::pair<float, uint32_t>>
    pq_search(const float *query, size_t k, size_t top_p) const {
        assert(trained);

        if (top_p < k) {
            top_p = k;
        }

        std::priority_queue<std::pair<float, uint32_t>> candidates = search_adc(query, top_p);
        return rerank_flat(query, candidates, k);
    }

private:
    size_t M;            // 子空间数量，例如 4
    size_t Ks;           // 每个子空间的聚类中心数量，例如 256
    size_t train_iters;  // KMeans 迭代次数
    size_t n;            // base 向量数量
    size_t dim;          // 原始向量维度，例如 96
    size_t dsub;         // 每个子空间维度，例如 24
    const float *base;   // 原始 base 数据，不复制，只保存指针
    bool trained;

    // codebooks[m][c][j]
    // 第 m 个子空间，第 c 个中心，第 j 维
    std::vector<float> codebooks;

    // codes[i][m]
    // 第 i 条 base 向量在第 m 个子空间上的中心编号
    std::vector<uint8_t> codes;

    // 第 vec_id 条向量的第 m 个子空间的起始地址
    const float* subvec_ptr(size_t vec_id, size_t m) const {
    return base + vec_id * dim + m * dsub;
    }

    
    // 第 m 个子空间第 c 个中心的起始地址
    float* centroid_ptr(size_t m, size_t c) {
        return codebooks.data() + (m * Ks + c) * dsub;
    }

    const float* centroid_ptr(size_t m, size_t c) const {
        return codebooks.data() + (m * Ks + c) * dsub;
    }

    // 初始化第 m 个聚类的中心
    void init_centroids(size_t m) {
        for(size_t c = 0; c < Ks; ++c) {
            size_t id = (c * n) / Ks;
            if(id >= n) {
                id = n - 1;
            }

            std::memcpy(centroid_ptr(m, c), subvec_ptr(id, m), dsub*sizeof(float));
        }
    }

    // 寻找最近聚类中心
    uint8_t nearest_centroid(const float *x, size_t m) {
        float nearest_dis = std::numeric_limits<float>::max();
        uint8_t nearest_id = 0;

        for(size_t c = 0; c < Ks; ++c) {
            float dis = l2_distance_simd(x, centroid_ptr(m, c), dsub);

            if(dis < nearest_dis) {
                nearest_dis = dis;
                nearest_id = static_cast<uint8_t>(c);
            }
        }

        return nearest_id;
    }

    // 对每个子空间的 N 个子向量进行聚类得到 256 个聚类中心
    void train_codebooks() {
        std::vector<uint8_t> assign(n, 0); // 记录每个子向量的归属
        std::vector<float> sums(Ks * dsub, 0.0f); // 记录每个聚类的和
        std::vector<uint32_t> counts(Ks, 0); // 记录每个聚类的子向量数量

        for (size_t m = 0; m < M; ++m) {
            // 初始化第 m 个子空间的中心
            init_centroids(m);

            for (size_t it = 0; it < train_iters; ++it) {
                std::fill(sums.begin(), sums.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                // 每个子向量分配到最近中心
                for (size_t i = 0; i < n; ++i) {
                    const float *x = subvec_ptr(i, m);
                    uint8_t cid = nearest_centroid(x, m);

                    assign[i] = cid;
                    counts[cid]++;

                    float *sum = sums.data() + static_cast<size_t>(cid) * dsub;
                    for (size_t j = 0; j < dsub; ++j) {
                        sum[j] += x[j];
                    }
                }

                // 更新聚类中心
                for (size_t c = 0; c < Ks; ++c) {
                    float *center = centroid_ptr(m, c);

                    if (counts[c] == 0) {
                        // 空簇处理：重新选一个 base 子向量作为中心
                        size_t id = (c * 9973 + it * 7919 + m * 104729) % n;
                        std::memcpy(center, subvec_ptr(id, m), dsub * sizeof(float));
                        continue;
                    }

                    const float inv = 1.0f / static_cast<float>(counts[c]);
                    const float *sum = sums.data() + c * dsub;

                    for (size_t j = 0; j < dsub; ++j) {
                        center[j] = sum[j] * inv;
                    }
                }
            }
        }
    }

    // PQ 编码
    void encode_base() {
        for (size_t i = 0; i < n; ++i) {
            for (size_t m = 0; m < M; ++m) {
                const float *x = subvec_ptr(i, m);
                codes[i * M + m] = nearest_centroid(x, m);
            }
        }
    }
};