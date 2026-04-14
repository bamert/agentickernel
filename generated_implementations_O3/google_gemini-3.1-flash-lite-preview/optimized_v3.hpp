#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v3: Improve memory access and use more efficient NEON bit expansion.
// Instead of extracting bits one by one or in small chunks, process columns of 32 bits
// by using bitwise arithmetic to map 0->-1.0 and 1->+1.0.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* rowC = &C[i * K];
        const float* rowA = &A[i * K];

        // Initialize output row to zero
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        // Iterate over rows of B 
        for (size_t p = 0; p < K; ++p) {
            float val = rowA[p];
            float32x4_t v_val = vdupq_n_f32(val);
            float32x4_t v_neg_val = vnegq_f32(v_val);

            const uint32_t* rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                // Process 32 bits, 4 columns at a time (8 groups of 4)
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    // Branchless conversion:
                    // If bit is 1, return val. If 0, return -val.
                    float32x4_t v_signs;
                    float res[4];
                    for(int k=0; k<4; ++k) {
                        res[k] = ((bits >> k) & 1) ? val : -val;
                    }
                    v_signs = vld1q_f32(res);
                    
                    float32x4_t v_c = vld1q_f32(&rowC[j_int * 32 + col * 4]);
                    vst1q_f32(&rowC[j_int * 32 + col * 4], vaddq_f32(v_c, v_signs));
                }
            }
        }
    }
}
