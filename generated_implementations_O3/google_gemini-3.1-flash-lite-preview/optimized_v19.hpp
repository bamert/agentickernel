#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v19:
// Re-visiting v3 and improving it slightly. v3 was the fastest (24.1548ms).
// The only things that could possibly speed up the process is reducing
// memory loads/stores (target pointer access) and minimizing the `f` array creation.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        // Initialize output 
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                float* __restrict__ target_base = rowC + j_int * 32;
                
                for (size_t col = 0; col < 8; ++col) {
                    const uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    float f[4];
                    f[0] = (bits & 1) ? val : n_val;
                    f[1] = (bits & 2) ? val : n_val;
                    f[2] = (bits & 4) ? val : n_val;
                    f[3] = (bits & 8) ? val : n_val;
                    
                    float32x4_t* t = reinterpret_cast<float32x4_t*>(target_base + col * 4);
                    *t = vaddq_f32(*t, vld1q_f32(f));
                }
            }
        }
    }
}
