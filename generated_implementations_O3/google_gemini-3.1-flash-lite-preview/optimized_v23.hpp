#pragma once
#include <cstdint>
#include <cstddef>

// Final attempt to improve on v3 (24.1548ms).
// A potential performance boost could be achieved by using pointer increments 
// instead of recomputing indices `rowC[j_int * 32 + col * 4]`.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float val = rowA[p];
            const float n_val = -val;
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            float* __restrict__ target_ptr = rowC;
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                
                for (size_t col = 0; col < 8; ++col) {
                    const uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    float f[4];
                    f[0] = (bits & 1) ? val : n_val;
                    f[1] = (bits & 2) ? val : n_val;
                    f[2] = (bits & 4) ? val : n_val;
                    f[3] = (bits & 8) ? val : n_val;
                    
                    float32x4_t v_c = vld1q_f32(target_ptr);
                    vst1q_f32(target_ptr, vaddq_f32(v_c, vld1q_f32(f)));
                    target_ptr += 4;
                }
            }
        }
    }
}
