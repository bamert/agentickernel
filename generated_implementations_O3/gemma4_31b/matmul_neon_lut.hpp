#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    // Precompute sign lookup table for 4-bit patterns
    float32x4_t sign_lut[16];
    for (int i = 0; i < 16; ++i) {
        float s[4];
        for (int b = 0; b < 4; ++b) {
            s[b] = ((i >> b) & 1) ? 1.0f : -1.0f;
        }
        sign_lut[i] = vld1q_f32(s);
    }

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f);
            float32x4_t acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f);
            float32x4_t acc7 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                const float a_val = rowA[p];
                const float32x4_t v_a = vdupq_n_f32(a_val);
                const uint32_t packed = B[p * K_ints + j_int];

                // Extract 4 bits at a time and use the LUT
                acc0 = vmlaq_f32(acc0, sign_lut[(packed >> 0) & 0xF], v_a);
                acc1 = vmlaq_f32(acc1, sign_lut[(packed >> 4) & 0xF], v_a);
                acc2 = vmlaq_f32(acc2, sign_lut[(packed >> 8) & 0xF], v_a);
                acc3 = vmlaq_f32(acc3, sign_lut[(packed >> 12) & 0xF], v_a);
                acc4 = vmlaq_f32(acc4, sign_lut[(packed >> 16) & 0xF], v_a);
                acc5 = vmlaq_f32(acc5, sign_lut[(packed >> 20) & 0xF], v_a);
                acc6 = vmlaq_f32(acc6, sign_lut[(packed >> 24) & 0xF], v_a);
                acc7 = vmlaq_f32(acc7, sign_lut[(packed >> 28) & 0xF], v_a);
            }

            float* out_ptr = &rowC[j_int * 32];
            vst1q_f32(out_ptr, acc0);
            vst1q_f32(out_ptr + 4, acc1);
            vst1q_f32(out_ptr + 8, acc2);
            vst1q_f32(out_ptr + 12, acc3);
            vst1q_f32(out_ptr + 16, acc4);
            vst1q_f32(out_ptr + 20, acc5);
            vst1q_f32(out_ptr + 24, acc6);
            vst1q_f32(out_ptr + 28, acc7);
        }
    }
}
