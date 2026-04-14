#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v9:
// Revert to the basic approach of v3 but sharpen the inner loop. 
// v3 is the current best. Let's see if we can tune local variables for it.
// The key is to keep the register pressure low and memory access aligned.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float val = rowA[p];
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                // Directly unroll the bit processing
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    float f[4];
                    f[0] = (bits & 1) ? val : -val;
                    f[1] = (bits & 2) ? val : -val;
                    f[2] = (bits & 4) ? val : -val;
                    f[3] = (bits & 8) ? val : -val;
                    
                    float32x4_t v_res = vld1q_f32(&rowC[j_int * 32 + col * 4]);
                    vst1q_f32(&rowC[j_int * 32 + col * 4], vaddq_f32(v_res, vld1q_f32(f)));
                }
            }
        }
    }
}
