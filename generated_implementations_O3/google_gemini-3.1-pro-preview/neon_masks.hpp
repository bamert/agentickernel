#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    uint32_t mask_arr[32];
    for (int i = 0; i < 32; ++i) {
        mask_arr[i] = 1u << i;
    }
    
    uint32x4_t masks[8];
    for (int k = 0; k < 8; ++k) {
        masks[k] = vld1q_u32(&mask_arr[k * 4]);
    }

    size_t i = 0;
    for (; i + 1 < M; i += 2) {
        float row_sum[2] = {0.0f, 0.0f};
        for (size_t p = 0; p < K; ++p) {
            row_sum[0] += A[i * K + p];
            row_sum[1] += A[(i + 1) * K + p];
        }

        for (size_t j_word = 0; j_word < K_ints; ++j_word) {
            float32x4_t sum0[8];
            float32x4_t sum1[8];
            for (int k = 0; k < 8; ++k) {
                sum0[k] = vdupq_n_f32(0.0f);
                sum1[k] = vdupq_n_f32(0.0f);
            }
            
            for (size_t p = 0; p < K; ++p) {
                uint32x4_t b_vec = vdupq_n_u32(B[p * K_ints + j_word]);
                
                uint32x4_t a0_u = vreinterpretq_u32_f32(vdupq_n_f32(A[i * K + p]));
                uint32x4_t a1_u = vreinterpretq_u32_f32(vdupq_n_f32(A[(i + 1) * K + p]));
                
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32x4_t m = vtstq_u32(b_vec, masks[k]);
                    sum0[k] = vaddq_f32(sum0[k], vreinterpretq_f32_u32(vandq_u32(a0_u, m)));
                    sum1[k] = vaddq_f32(sum1[k], vreinterpretq_f32_u32(vandq_u32(a1_u, m)));
                }
            }
            
            float32x4_t rs0 = vdupq_n_f32(row_sum[0]);
            float32x4_t rs1 = vdupq_n_f32(row_sum[1]);
            
            for (int k = 0; k < 8; ++k) {
                float32x4_t final0 = vsubq_f32(vaddq_f32(sum0[k], sum0[k]), rs0);
                float32x4_t final1 = vsubq_f32(vaddq_f32(sum1[k], sum1[k]), rs1);
                
                vst1q_f32(&C[i * K + j_word * 32 + k * 4], final0);
                vst1q_f32(&C[(i + 1) * K + j_word * 32 + k * 4], final1);
            }
        }
    }

    for (; i < M; ++i) {
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += A[i * K + p];
        }

        for (size_t j_word = 0; j_word < K_ints; ++j_word) {
            float32x4_t sum[8];
            for (int k = 0; k < 8; ++k) {
                sum[k] = vdupq_n_f32(0.0f);
            }
            
            for (size_t p = 0; p < K; ++p) {
                uint32x4_t b_vec = vdupq_n_u32(B[p * K_ints + j_word]);
                uint32x4_t a_u = vreinterpretq_u32_f32(vdupq_n_f32(A[i * K + p]));
                
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32x4_t m = vtstq_u32(b_vec, masks[k]);
                    sum[k] = vaddq_f32(sum[k], vreinterpretq_f32_u32(vandq_u32(a_u, m)));
                }
            }
            
            float32x4_t rs = vdupq_n_f32(row_sum);
            
            for (int k = 0; k < 8; ++k) {
                float32x4_t final_val = vsubq_f32(vaddq_f32(sum[k], sum[k]), rs);
                vst1q_f32(&C[i * K + j_word * 32 + k * 4], final_val);
            }
        }
    }
}
