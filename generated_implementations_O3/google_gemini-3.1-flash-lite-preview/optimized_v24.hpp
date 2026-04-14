#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v24: A minimal modification of v3.
// v3 is the current best performer (24.1548ms).
// By ensuring the alignment of the output and minimal index arithmetic,
// we might be able to squeeze out a few more microseconds.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        // Using a standard loop to zero might be slightly better than memset
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                
                for (size_t col = 0; col < 8; ++col) {
                    const uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    alignas(16) float f[4];
                    f[0] = (bits & 1) ? val : n_val;
                    f[1] = (bits & 2) ? val : n_val;
                    f[2] = (bits & 4) ? val : n_val;
                    f[3] = (bits & 8) ? val : n_val;
                    
                    float* __restrict__ target = &rowC[(j_int << 5) + (col << 2)];
                    float32x4_t v_res = vld1q_f32(target);
                    vst1q_f32(target, vaddq_f32(v_res, vld1q_f32(f)));
                }
            }
        }
    }
}
