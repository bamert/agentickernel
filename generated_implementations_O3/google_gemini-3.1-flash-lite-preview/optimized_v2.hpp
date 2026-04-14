#pragma once
#include <cstdint>
#include <cstddef>

// Optimized implementation using NEON intrinsics.
// Instead of iterating bit by bit, we can process 4 or 8 floats at once.
// Matrix B is packed (1 bit = +1, 0 bit = -1).
// This implies -1 = -1.0, 1 = 1.0. 
// A single integer from B contains 32 bits.
// We can expand these bits to float vectors: 1 -> +1.0, 0 -> -1.0.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        // Initialize output row to 0
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = rowA[p];
            float32x4_t a_vec = vdupq_n_f32(a_val);
            float32x4_t neg_a_vec = vnegq_f32(a_vec);
            
            const uint32_t* rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                // Process 32 bits in groups of 4 using NEON
                for (size_t b = 0; b < 8; ++b) {
                    uint32_t bits = (packed >> (b * 4)) & 0xF;
                    
                    float elements[4];
                    for (int k = 0; k < 4; ++k) {
                        elements[k] = ((bits >> k) & 1) ? a_val : -a_val;
                    }
                    float32x4_t v_signs = vld1q_f32(elements);
                    
                    float32x4_t res = vld1q_f32(&rowC[j_int * 32 + b * 4]);
                    vst1q_f32(&rowC[j_int * 32 + b * 4], vaddq_f32(res, v_signs));
                }
            }
        }
    }
}
