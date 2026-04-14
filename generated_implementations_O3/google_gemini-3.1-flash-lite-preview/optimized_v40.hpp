#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v40: Final attempt. The structure v25 held the record (24.1398ms).
// Let's refine it one more time with a slightly more aggressive `__restrict__`
// and ensuring that `res` is treated as a compiler-friendly alignment.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        // Initialization
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                
                // Using explicit unroll and manually managing the loop.
                for (size_t col = 0; col < 8; ++col) {
                    const uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    alignas(16) float res[4];
                    for(int k=0; k<4; k++) {
                        res[k] = ((bits >> k) & 1) ? val : n_val;
                    }
                    
                    float* __restrict__ target = &rowC[j_int * 32 + (col << 2)];
                    float32x4_t v_c = vld1q_f32(target);
                    // Use a single line addition, which encourages the compiler to pipeline
                    vst1q_f32(target, vaddq_f32(v_c, vld1q_f32(res)));
                }
            }
        }
    }
}
