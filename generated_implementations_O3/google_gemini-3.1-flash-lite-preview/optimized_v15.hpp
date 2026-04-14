#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v15:
// The current best (v3) uses a loop that the compiler likely hoists well.
// v15: Let's try to unroll by a factor of 2 or 4 the K_ints loop inside the p loop.
// This might help with instruction-level parallelism and hiding NEON latency.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            size_t j_int = 0;
            for (; j_int + 1 < K_ints; j_int += 2) {
                // Manually unroll by 2 inner blocks
                uint32_t packed1 = rowB[j_int];
                uint32_t packed2 = rowB[j_int+1];
                
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t b1 = (packed1 >> (col * 4)) & 0xF;
                    uint32_t b2 = (packed2 >> (col * 4)) & 0xF;
                    
                    float f1[4], f2[4];
                    for(int k=0; k<4; ++k) {
                        f1[k] = (b1 & (1 << k)) ? val : n_val;
                        f2[k] = (b2 & (1 << k)) ? val : n_val;
                    }
                    
                    float* target1 = &rowC[j_int * 32 + col * 4];
                    float* target2 = &rowC[(j_int + 1) * 32 + col * 4];
                    
                    vst1q_f32(target1, vaddq_f32(vld1q_f32(target1), vld1q_f32(f1)));
                    vst1q_f32(target2, vaddq_f32(vld1q_f32(target2), vld1q_f32(f2)));
                }
            }
            // Handle last chunk if odd number
            for (; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t bits = (packed >> (col * 4)) & 0xF;
                    float f[4];
                    for(int k=0; k<4; ++k) f[k] = (bits & (1 << k)) ? val : n_val;
                    float* target = &rowC[j_int * 32 + col * 4];
                    vst1q_f32(target, vaddq_f32(vld1q_f32(target), vld1q_f32(f)));
                }
            }
        }
    }
}
