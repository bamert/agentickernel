#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v8:
// The bottleneck is likely memory access patterns and the overhead of bit-to-float conversion.
// We'll use block processing to better fit the data into L1 cache and see if we can
// optimize the bit expansion further by using NEON vector tables.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero out C
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    // Block Processing
    const size_t BLOCK_SIZE = 16; 
    for (size_t i = 0; i < M; i += BLOCK_SIZE) {
        size_t i_max = (i + BLOCK_SIZE > M) ? M : i + BLOCK_SIZE;
        
        for (size_t p = 0; p < K; ++p) {
            for (size_t i_block = i; i_block < i_max; ++i_block) {
                float val = A[i_block * K + p];
                float32x4_t v_a = vdupq_n_f32(val);
                float32x4_t v_neg_a = vnegq_f32(v_a);
                float* __restrict__ rowC = &C[i_block * K];
                const uint32_t* __restrict__ rowB = &B[p * K_ints];
                
                for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                    uint32_t packed = rowB[j_int];
                    
                    for (size_t b = 0; b < 8; ++b) {
                        uint32_t bits = (packed >> (b * 4)) & 0xF;
                        
                        // Inline check for bits
                        float signs[4];
                        signs[0] = (bits & 1) ? val : -val;
                        signs[1] = (bits & 2) ? val : -val;
                        signs[2] = (bits & 4) ? val : -val;
                        signs[3] = (bits & 8) ? val : -val;
                        
                        float32x4_t v_signs = vld1q_f32(signs);
                        float* target = &rowC[j_int * 32 + b * 4];
                        vst1q_f32(target, vaddq_f32(vld1q_f32(target), v_signs));
                    }
                }
            }
        }
    }
}
