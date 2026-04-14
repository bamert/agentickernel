#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v7:
// Focus on loop unrolling and reducing redundant loads between chunks.
// The main loop is the matrix multiplication (M x K x bits).
// We'll process each Row of A, and iterate over Rows of B.
// Using explicit NEON instructions to avoid memory loads for every bit set.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a = rowA[p];
            float32x4_t v_a = vdupq_n_f32(a);
            float32x4_t v_neg_a = vnegq_f32(v_a);
            
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                // Process 32 elements. Each 8 chunks of 4.
                for (size_t b = 0; b < 8; ++b) {
                    uint32_t bits = (packed >> (b * 4)) & 0xF;
                    
                    // Directly construct float vector based on bits
                    float f[4];
                    f[0] = (bits & 1) ? a : -a;
                    f[1] = (bits & 2) ? a : -a;
                    f[2] = (bits & 4) ? a : -a;
                    f[3] = (bits & 8) ? a : -a;
                    
                    float32x4_t v_signs = vld1q_f32(f);
                    float32x4_t v_res = vld1q_f32(&rowC[j_int * 32 + b * 4]);
                    vst1q_f32(&rowC[j_int * 32 + b * 4], vaddq_f32(v_res, v_signs));
                }
            }
        }
    }
}
