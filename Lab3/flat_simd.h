#pragma once
#include <queue>
#include <arm_neon.h>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <utility>

struct simd8float32 {
    float32x4x2_t data;
    simd8float32() = default;

    explicit simd8float32(float x) {
        data.val[0] = vdupq_n_f32(x);
        data.val[1] = vdupq_n_f32(x);
    }

    explicit simd8float32(const float* x) {
        data.val[0] = vld1q_f32(x);
        data.val[1] = vld1q_f32(x + 4);
    }

    simd8float32 operator*(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    simd8float32 operator+(const simd8float32& other) const {
        simd8float32 result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    simd8float32& operator+=(const simd8float32& other) {
        data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return *this;
    }

    void fmadd(const simd8float32& a, const simd8float32& b) {
        data.val[0] = vmlaq_f32(data.val[0], a.data.val[0], b.data.val[0]);
        data.val[1] = vmlaq_f32(data.val[1], a.data.val[1], b.data.val[1]);
    }

    void storeu(float* out) const {
        vst1q_f32(out, data.val[0]);
        vst1q_f32(out + 4, data.val[1]);
    }
};

/*float InnerProductSIMDNeon(float* b1, float* b2, size_t vecdim) {
    assert(vecdim % 8 == 0); //假设维度能被8整除

    simd8float32 sum(0.0); //8xfloat32全部初始化为0
    for(int i = 0; i < vecdim; i += 8) {
        simd8float32 s1(b1 + i), s2(b2 + i);
        simd8float32 m = s1 * s2;
        sum += m;
    }

    float tmp[8];
    sum.storeu(tmp);
    float dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return 1 - dis;
}*/

/*float InnerProductSIMDNeon_FMA(const float* b1, const float* b2, size_t vecdim) {
    assert(vecdim % 8 == 0);

    simd8float32 sum(0.0);
    for(size_t i = 0; i < vecdim; i += 8) {
        simd8float32 s1(b1 + i), s2(b2 + i);
        sum.fmadd(s1, s2); // 融合乘加运算
    }

    float dis = vaddvq_f32(sum.data.val[0]) + vaddvq_f32(sum.data.val[1]); // 使用 NEON 向量求和指令，速度更快
    return 1 - dis;
}*/

float inline InnerProductSIMDNeon_Unroll32(const float* b1, const float* b2, size_t vecdim) {
    assert(vecdim % 32 == 0);

    // 定义 4 个 128 位累加器进行 4 路累加
    simd8float32 sum0(0.0);
    simd8float32 sum1(0.0);
    simd8float32 sum2(0.0);
    simd8float32 sum3(0.0);

    // 32 位循环展开，每次处理 32 个 float
    for(size_t i = 0; i < vecdim; i += 32) {
        // 加载 b1，b2 的前 8 个 float，使用 fma 乘加指令累加到 sum0
        simd8float32 r0(b1 + i);
        simd8float32 s0(b2 + i);
        sum0.fmadd(r0, s0);

        simd8float32 r1(b1 + i + 8);
        simd8float32 s1(b2 + i + 8);
        sum1.fmadd(r1, s1);

        simd8float32 r2(b1 + i + 16);
        simd8float32 s2(b2 + i + 16);
        sum2.fmadd(r2, s2);

        simd8float32 r3(b1 + i + 24);
        simd8float32 s3(b2 + i + 24);
        sum3.fmadd(r3, s3);
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;

    float dot = vaddvq_f32(sum0.data.val[0]) + vaddvq_f32(sum0.data.val[1]);
    return 1 - dot;
}

std::priority_queue<std::pair<float, uint32_t> > flat_search_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < base_number; ++i) {
        // DEEP100K数据集使用ip距离
        float dis = InnerProductSIMDNeon_Unroll32(base + i * vecdim, query, vecdim);

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}

float inline InnerProductSIMD(const float* b1, const float* b2, size_t vecdim) {
    assert(vecdim % 32 == 0);

    // 定义 4 个 128 位累加器进行 4 路累加
    simd8float32 sum0(0.0);
    simd8float32 sum1(0.0);
    simd8float32 sum2(0.0);
    simd8float32 sum3(0.0);

    // 32 位循环展开，每次处理 32 个 float
    for(size_t i = 0; i < vecdim; i += 32) {
        // 加载 b1，b2 的前 8 个 float，使用 fma 乘加指令累加到 sum0
        simd8float32 r0(b1 + i);
        simd8float32 s0(b2 + i);
        sum0.fmadd(r0, s0);

        simd8float32 r1(b1 + i + 8);
        simd8float32 s1(b2 + i + 8);
        sum1.fmadd(r1, s1);

        simd8float32 r2(b1 + i + 16);
        simd8float32 s2(b2 + i + 16);
        sum2.fmadd(r2, s2);

        simd8float32 r3(b1 + i + 24);
        simd8float32 s3(b2 + i + 24);
        sum3.fmadd(r3, s3);
    }

    sum0 += sum1;
    sum2 += sum3;
    sum0 += sum2;

    float dot = vaddvq_f32(sum0.data.val[0]) + vaddvq_f32(sum0.data.val[1]);
    return dot;
}