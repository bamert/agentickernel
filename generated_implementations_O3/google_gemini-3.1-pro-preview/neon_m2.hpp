#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    const uint32x4_t bit_mask[8] = {
        vmakeq_u32(1<<0, 1<<1, 1<<2, 1<<3),
        vmakeq_u32(1<<4, 1<<5, 1<<6, 1<<7),
        vmakeq_u32(1<<8, 1<<9, 1<<10, 1<<11),
        vmakeq_u32(1<<12, 1<<13, 1<<14, 1<<15),
        vmakeq_u32(1<<16, 1<<17, 1<<18, 1<<19),
        vmakeq_u32(1<<20, 1<<21, 1<<22, 1<<23),
        vmakeq_u32(1<<24, 1<<25, 1<<26, 1<<27),
        vmakeq_u32(1<<28, 1<<29, 1<<30, 1u<<31)
    };

    for (size_t i = 0; i < M; i += 2) {
        float row_sum[2] = {0.0f, 0.0f};
        for (size_t p = 0; p < K; ++p) {
            row_sum[0] += A[i * K + p];
            if (i + 1 < M) row_sum[1] += A[(i + 1) * K + p];
        }

        for (size_t j_word = 0; j_word < K_ints; ++j_word) {
            float32x4_t sum0[8];
            float32x4_t sum1[8];
            for (int k = 0; k < 8; ++k) {
                sum0[k] = vdupq_n_f32(0.0f);
                sum1[k] = vdupq_n_f32(0.0f);
            }
            
            for (size_t p = 0; p < K; ++p) {
                uint32_t b_val = B[p * K_ints + j_word];
                uint32x4_t b_vec = vdupq_n_u32(b_val);
                
                uint32x4_t a_vec_u0 = vreinterpretq_u32_f32(vdupq_n_f32(A[i * K + p]));
                uint32x4_t a_vec_u1 = i + 1 < M ? vreinterpretq_u32_f32(vdupq_n_f32(A[(i + 1) * K + p])) : vdupq_n_u32(0);
                
                // loop unrolled for performance
                uint32x4_t m0 = vtstq_u32(b_vec, bit_mask[0]);
                sum0[0] = vaddq_f32(sum0[0], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m0)));
                sum1[0] = vaddq_f32(sum1[0], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m0)));

                uint32x4_t m1 = vtstq_u32(b_vec, bit_mask[1]);
                sum0[1] = vaddq_f32(sum0[1], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m1)));
                sum1[1] = vaddq_f32(sum1[1], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m1)));

                uint32x4_t m2 = vtstq_u32(b_vec, bit_mask[2]);
                sum0[2] = vaddq_f32(sum0[2], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m2)));
                sum1[2] = vaddq_f32(sum1[2], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m2)));

                uint32x4_t m3 = vtstq_u32(b_vec, bit_mask[3]);
                sum0[3] = vaddq_f32(sum0[3], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m3)));
                sum1[3] = vaddq_f32(sum1[3], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m3)));

                uint32x4_t m4 = vtstq_u32(b_vec, bit_mask[4]);
                sum0[4] = vaddq_f32(sum0[4], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m4)));
                sum1[4] = vaddq_f32(sum1[4], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m4)));

                uint32x4_t m5 = vtstq_u32(b_vec, bit_mask[5]);
                sum0[5] = vaddq_f32(sum0[5], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m5)));
                sum1[5] = vaddq_f32(sum1[5], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m5)));

                uint32x4_t m6 = vtstq_u32(b_vec, bit_mask[6]);
                sum0[6] = vaddq_f32(sum0[6], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m6)));
                sum1[6] = vaddq_f32(sum1[6], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m6)));

                uint32x4_t m7 = vtstq_u32(b_vec, bit_mask[7]);
                sum0[7] = vaddq_f32(sum0[7], vreinterpretq_f32_u32(vandq_u32(a_vec_u0, m7)));
                sum1[7] = vaddq_f32(sum1[7], vreinterpretq_f32_u32(vandq_u32(a_vec_u1, m7)));
            }
            
            float32x4_t two = vdupq_n_f32(2.0f);
            float32x4_t rs0 = vdupq_n_f32(row_sum[0]);
            float32x4_t rs1 = vdupq_n_f32(row_sum[1]);
            
            for (int k = 0; k < 8; ++k) {
                float32x4_t c0 = vmlsq_f32(vaddq_f32(sum0[k], sum0[k]), rs0, vdupq_n_f32(1.0f)); // sum0 * 2 - rs
                float32x4_t c1 = vmlsq_f32(vaddq_f32(sum1[k], sum1[k]), rs1, vdupq_n_f32(1.0f));
                
                // Better: c0 = fms(rs0, sum0, 2)? No, just sum0*2 - rs0.  
                // vaddq(sum0, sum0) is sum0*2. Then vsubq(sum0*2, rs0)
                float32x4_t final0 = vsubq_f32(vaddq_f32(sum0[k], sum0[k]), rs0);
                float32x4_t final1 = vsubq_f32(vaddq_f32(sum1[k], sum1[k]), rs1);

                vst1q_f32(&C[i * K + j_word * 32 + k * 4], final0);
                if (i + 1 < M) {
                    vst1q_f32(&C[(i + 1) * K + j_word * 32 + k * 4], final1);
                }
            }
        }
    }
}
