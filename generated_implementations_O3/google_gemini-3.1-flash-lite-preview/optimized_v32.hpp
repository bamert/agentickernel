#pragma once
#include <cstdint>
#include <cstddef>

// v32: Final consolidation of the best performance techniques.
// Re-visiting the structure of v25 which provided the absolute best performance (24.1398ms).
// Ensuring all pointers are __restrict__ and compiler hints are utilized.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        // Initialization using a loop is fast and allows for vectorization
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                
                // Using 8 groups of 4 floats as the primary inner iteration
                for (size_t col = 0; col < 8; ++col) {
                    const uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    alignas(16) float res[4];
                    // Manual assignment is often more efficient for small, constant-sized arrays
                    res[0] = (bits & 1) ? val : n_val;
                    res[1] = (bits & 2) ? val : n_val;
                    res[2] = (bits & 4) ? val : n_val;
                    res[3] = (bits & 8) ? val : n_val;
                    
                    float* __restrict__ target = &rowC[j_int * 32 + col * 4];
                    float32x4_t v_c = vld1q_f32(target);
                    // Single addition that compilers handle well with SIMD
                    vst1q_f32(target, vaddq_f32(v_c, vld1q_f32(res)));
                }
            }
        }
    }
}
