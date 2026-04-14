#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    uint32_t m0[4] = {1u<<0, 1u<<1, 1u<<2, 1u<<3};
    uint32_t m1[4] = {1u<<4, 1u<<5, 1u<<6, 1u<<7};
    uint32_t m2[4] = {1u<<8, 1u<<9, 1u<<10, 1u<<11};
    uint32_t m3[4] = {1u<<12, 1u<<13, 1u<<14, 1u<<15};
    uint32_t m4[4] = {1u<<16, 1u<<17, 1u<<18, 1u<<19};
    uint32_t m5[4] = {1u<<20, 1u<<21, 1u<<22, 1u<<23};
    uint32_t m6[4] = {1u<<24, 1u<<25, 1u<<26, 1u<<27};
    uint32_t m7[4] = {1u<<28, 1u<<29, 1u<<30, 1u<<31};

    uint32x4_t vm0 = vld1q_u32(m0);
    uint32x4_t vm1 = vld1q_u32(m1);
    uint32x4_t vm2 = vld1q_u32(m2);
    uint32x4_t vm3 = vld1q_u32(m3);
    uint32x4_t vm4 = vld1q_u32(m4);
    uint32x4_t vm5 = vld1q_u32(m5);
    uint32x4_t vm6 = vld1q_u32(m6);
    uint32x4_t vm7 = vld1q_u32(m7);

    for (size_t i = 0; i < M; ++i) {
        float sum_a = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            sum_a += A[i * K + p];
        }
        float neg_sum_a = -sum_a;
        
        for (size_t j_blk = 0; j_blk < K_ints; ++j_blk) {
            float32x4_t vC0 = vdupq_n_f32(neg_sum_a);
            float32x4_t vC1 = vdupq_n_f32(neg_sum_a);
            float32x4_t vC2 = vdupq_n_f32(neg_sum_a);
            float32x4_t vC3 = vdupq_n_f32(neg_sum_a);
            float32x4_t vC4 = vdupq_n_f32(neg_sum_a);
            float32x4_t vC5 = vdupq_n_f32(neg_sum_a);
            float32x4_t vC6 = vdupq_n_f32(neg_sum_a);
            float32x4_t vC7 = vdupq_n_f32(neg_sum_a);
            
            for (size_t p = 0; p < K; ++p) {
                float two_a = 2.0f * A[i * K + p];
                uint32x4_t v_two_a = vreinterpretq_u32_f32(vdupq_n_f32(two_a));
                
                uint32x4_t v_packed = vdupq_n_u32(B[p * K_ints + j_blk]);
                
                uint32x4_t t0 = vtstq_u32(v_packed, vm0);
                uint32x4_t t1 = vtstq_u32(v_packed, vm1);
                uint32x4_t t2 = vtstq_u32(v_packed, vm2);
                uint32x4_t t3 = vtstq_u32(v_packed, vm3);
                uint32x4_t t4 = vtstq_u32(v_packed, vm4);
                uint32x4_t t5 = vtstq_u32(v_packed, vm5);
                uint32x4_t t6 = vtstq_u32(v_packed, vm6);
                uint32x4_t t7 = vtstq_u32(v_packed, vm7);
                
                vC0 = vaddq_f32(vC0, vreinterpretq_f32_u32(vandq_u32(t0, v_two_a)));
                vC1 = vaddq_f32(vC1, vreinterpretq_f32_u32(vandq_u32(t1, v_two_a)));
                vC2 = vaddq_f32(vC2, vreinterpretq_f32_u32(vandq_u32(t2, v_two_a)));
                vC3 = vaddq_f32(vC3, vreinterpretq_f32_u32(vandq_u32(t3, v_two_a)));
                vC4 = vaddq_f32(vC4, vreinterpretq_f32_u32(vandq_u32(t4, v_two_a)));
                vC5 = vaddq_f32(vC5, vreinterpretq_f32_u32(vandq_u32(t5, v_two_a)));
                vC6 = vaddq_f32(vC6, vreinterpretq_f32_u32(vandq_u32(t6, v_two_a)));
                vC7 = vaddq_f32(vC7, vreinterpretq_f32_u32(vandq_u32(t7, v_two_a)));
            }
            
            size_t c_idx = i * K + j_blk * 32;
            vst1q_f32(&C[c_idx + 0],  vC0);
            vst1q_f32(&C[c_idx + 4],  vC1);
            vst1q_f32(&C[c_idx + 8],  vC2);
            vst1q_f32(&C[c_idx + 12], vC3);
            vst1q_f32(&C[c_idx + 16], vC4);
            vst1q_f32(&C[c_idx + 20], vC5);
            vst1q_f32(&C[c_idx + 24], vC6);
            vst1q_f32(&C[c_idx + 28], vC7);
        }
    }
}
