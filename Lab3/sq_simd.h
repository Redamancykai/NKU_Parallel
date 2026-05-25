#pragma once
#include <arm_neon.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

// 量化原始 float 数据为 uint8_t
static inline uint8_t quantize_float_to_u8(float x, float min_val, float inv_step) {
    int q = static_cast<int>(std::round((x - min_val) * inv_step));
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return static_cast<uint8_t>(q);
}

// 计算全局最值
static inline void compute_min_max(const float* data, size_t total, float& min_val, float& max_val) {
    min_val = std::numeric_limits<float>::max();
    max_val = -std::numeric_limits<float>::max();

    for (size_t i = 0; i < total; ++i) {
        float x = data[i];
        if (x < min_val) min_val = x;
        if (x > max_val) max_val = x;
    }

    if (max_val <= min_val) {
        max_val = min_val + 1e-6f;
    }
}

// 量化原始 base 数据为 uint8_t
static inline uint8_t* quantize_base_u8(const float* base,
    size_t base_number,
    size_t vecdim,
    float& min_val,
    float& max_val,
    float& step,
    float& inv_step,
    std::vector<uint32_t>& base_qsum, // 向量元素之和
    std::vector<float>& base_bias // 每个向量的偏置，加速内积计算
) {
    assert(base != nullptr);

    const size_t total = base_number * vecdim;

    compute_min_max(base, total, min_val, max_val);

    step = (max_val - min_val) / 255.0f;
    inv_step = 255.0f / (max_val - min_val);

    uint8_t* base_q = new uint8_t[total];

    base_qsum.assign(base_number, 0);
    base_bias.assign(base_number, 0.0f);

    for (size_t i = 0; i < base_number; ++i) {
        const float* xb = base + i * vecdim;
        uint8_t* xq = base_q + i * vecdim;

        uint32_t s = 0;

        for (size_t j = 0; j < vecdim; ++j) {
            uint8_t q = quantize_float_to_u8(xb[j], min_val, inv_step);
            xq[j] = q;
            s += q;
        }

        base_qsum[i] = s;
    }

    const float min_step = min_val * step;

    for (size_t i = 0; i < base_number; ++i) {
        base_bias[i] = min_step * static_cast<float>(base_qsum[i]);
    }

    return base_q;
}


// query 量化
static inline void quantize_query_u8(
    const float* query,
    size_t vecdim,
    float min_val,
    float inv_step,
    uint8_t* query_q
) {
    assert(query != nullptr);
    assert(query_q != nullptr);

    for (size_t j = 0; j < vecdim; ++j) {
        query_q[j] = quantize_float_to_u8(query[j], min_val, inv_step);
    }
}

static inline uint32_t dot_u8_neon_unroll32(const uint8_t* a, const uint8_t* b, size_t dim) {
    assert(a != nullptr);
    assert(b != nullptr);
    assert(dim % 16 == 0);

    // 采用四路累加器加速运算，一次计算 16 个 uint8
    uint32x4_t acc0 = vdupq_n_u32(0);
    uint32x4_t acc1 = vdupq_n_u32(0);
    uint32x4_t acc2 = vdupq_n_u32(0);
    uint32x4_t acc3 = vdupq_n_u32(0);

    size_t i = 0;

    // 32 路循环展开，每次处理 32 个 uint8
    for (; i + 31 < dim; i += 32) {
        // 使用 vld1q_u8 加载 uint8
        uint8x16_t a0 = vld1q_u8(a + i);
        uint8x16_t b0 = vld1q_u8(b + i);

        uint8x16_t a1 = vld1q_u8(a + i + 16);
        uint8x16_t b1 = vld1q_u8(b + i + 16);

        // 防止溢出拆分成高 64 位和低 64 位分别计算
        uint16x8_t p0_low  = vmull_u8(vget_low_u8(a0),  vget_low_u8(b0));
        uint16x8_t p0_high = vmull_u8(vget_high_u8(a0), vget_high_u8(b0));

        uint16x8_t p1_low  = vmull_u8(vget_low_u8(a1),  vget_low_u8(b1));
        uint16x8_t p1_high = vmull_u8(vget_high_u8(a1), vget_high_u8(b1));

        // 合并累加到 32 位累加器
        acc0 = vpadalq_u16(acc0, p0_low);
        acc1 = vpadalq_u16(acc1, p0_high);
        acc2 = vpadalq_u16(acc2, p1_low);
        acc3 = vpadalq_u16(acc3, p1_high);
    }

    // 处理剩余维度
    for (; i + 15 < dim; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);

        uint16x8_t p_low  = vmull_u8(vget_low_u8(va),  vget_low_u8(vb));
        uint16x8_t p_high = vmull_u8(vget_high_u8(va), vget_high_u8(vb));

        acc0 = vpadalq_u16(acc0, p_low);
        acc1 = vpadalq_u16(acc1, p_high);
    }

    uint32x4_t acc = vaddq_u32(vaddq_u32(acc0, acc1), vaddq_u32(acc2, acc3));

    uint32_t tmp[4];
    vst1q_u32(tmp, acc);

    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}


// SQ 搜索
static inline std::priority_queue<std::pair<float, uint32_t>> sq_search(
    const float* base,
    const uint8_t* base_q,
    const float* query,
    const std::vector<uint32_t>& base_qsum,
    const std::vector<float>& base_bias,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t top_p,
    float min_val,
    float step,
    float inv_step
) {
    assert(base != nullptr);
    assert(base_q != nullptr);
    assert(query != nullptr);
    assert(vecdim % 16 == 0);
    assert(vecdim % 8 == 0);
    assert(base_qsum.size() >= base_number);
    assert(base_bias.size() >= base_number);

    if (top_p < k) {
        top_p = k;
    }

    if (top_p > base_number) {
        top_p = base_number;
    }

    // 1. 量化 query
    std::vector<uint8_t> query_q(vecdim);

    quantize_query_u8(query, vecdim, min_val, inv_step, query_q.data());

    const float step2 = step * step;

    using Candidate = std::pair<float, uint32_t>;

    // 维护最小堆
    struct MinScoreCmp {
        bool operator()(const Candidate& a, const Candidate& b) const {
            return a.first > b.first;
        }
    };

    std::priority_queue<Candidate, std::vector<Candidate>, MinScoreCmp> coarse_pq;

    // 2. SQ 粗排
    for (size_t i = 0; i < base_number; ++i) {
        // 获取第 i 个向量的量化数据
        const uint8_t* bq = base_q + i * vecdim;

        // 计算量化向量点积
        uint32_t qdot = dot_u8_neon_unroll32(bq, query_q.data(), vecdim);

        // 利用反量化公式计算 score
        float score = base_bias[i] + step2 * static_cast<float>(qdot);

        // 当前向量编号
        uint32_t id = static_cast<uint32_t>(i);

        // 最小堆维护
        if (coarse_pq.size() < top_p) {
            coarse_pq.push(std::make_pair(score, id));
        } 
        else if (score > coarse_pq.top().first) {
            coarse_pq.pop();
            coarse_pq.push(std::make_pair(score, id));
        }
    }

    // 3. float 精排
    // result 是最大堆，堆顶保存当前 Top-k 中距离最大的元素。
    std::priority_queue<std::pair<float, uint32_t>> result;

    while (!coarse_pq.empty()) {
        uint32_t id = coarse_pq.top().second;
        coarse_pq.pop();

        const float* xb = base + static_cast<size_t>(id) * vecdim;

        float exact_dist = InnerProductSIMDNeon_Unroll32(xb, query, vecdim);

        if (result.size() < k) {
            result.push(std::make_pair(exact_dist, id));
        } 
        else if (exact_dist < result.top().first) {
            result.pop();
            result.push(std::make_pair(exact_dist, id));
        }
    }

    return result;
}