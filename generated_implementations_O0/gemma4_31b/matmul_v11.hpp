#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Optimized Matrix Multiplication with Packed Binary Matrix B
// This version refines v8 by improving memory access patterns,
// removing redundant index calculations, and streamlining the inner loop.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    
    // Precompute sign table for 8-bit patterns.
    // Each entry in sign_table stores signs for 4 bits.
    // Total size: 256 * 2 * 16 bytes = 8 KB.
    float32x4_t sign_table[512];
    for (int m = 0; m < 256; ++m) {
        float signs_low[4], signs_high[4];
        for (int b = 0; b < 4; ++b) {
            signs_low[b] = (m & (1 << b)) ? 1.0f : -1.0f;
            signs_high[b] = (m & (1 << (b + 4))) ? 1.0f : -1.0f;
        }
        sign_table[2 * m] = vld1q_f32(signs_low);
        sign_table[2 * m + 1] = vld1q_f32(signs_high);
    }

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = A + i * K;
        float* row_C = C + i * K;

        // Use a local accumulator buffer to keep data in L1.
        // K is a multiple of 32, and max K is 3072.
        float32x4_t accs[768]; 
        for (size_t v = 0; v < K / 4; ++v) {
            accs[v] = vdupq_n_f32(0.0f);
        }

        for (size_t p = 0; p < K; ++p) {
            const float a_val = row_A[p];
            if (a_val == 0.0f) continue;
            
            const float32x4_t v_a = vdupq_n_f32(a_val);
            const uint32_t* row_B_packed = B + p * K_ints;

            // Inner loop processes each 32-bit integer of B.
            // We use a pointer to accs to avoid redundant index calculations.
            float32x4_t* p_acc = accs;

            for (size_t b = 0; b < K_ints; ++b) {
                const uint32_t packed = row_B_packed[b];
                
                const uint8_t b0 = (uint8_t)(packed & 0xFF);
                const uint8_t b1 = (uint8_t)((packed >> 8) & 0xFF);
                const uint8_t b2 = (uint8_t)((packed >> 16) & 0xFF);
                const uint8_t b3 = (uint8_t)((packed >> 24) & 0xFF);

                // Each byte lookup results in two 4-float sign vectors.
                // We perform 8 multiply-accumulate operations per integer.
                p_acc[0] = vmlaq_f32(p_acc[0], v_a, sign_table[2 * b0]);
                p_acc[1] = vmlaq_f32(p_acc[1], v_a, sign_table[2 * b0 + 1]);
                p_acc[2] = vmlaq_f32(p_acc[2], v_a, sign_table[2 * b1]);
                p_acc[3] = vmlaq_f32(p_acc[3], v_a, sign_table[2 * b1 + 1]);
                p_acc[4] = vmlaq_f32(p_acc[4], v_a, sign_table[2 * b2]);
                p_acc[5] = vmlaq_f32(p_acc[5], v_a, sign_table[2 * b2 + 1]);
                p_acc[6] = vmlaq_f32(p_acc[6], v_a, sign_table[2 * b3]);
                p_acc[7] = vmlaq_f32(p_acc[7], v_a, sign_table[2 * b3 + 1]);
                
                p_acc += 8;
            }
        }

        // Final store from accumulators to output matrix C.
        for (size_t v = 0; v < K / 4; ++v) {
            vst1q_f32(row_C + v * 4, accs[v]);
        }
    }
}
