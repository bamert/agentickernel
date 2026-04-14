#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v5: Use bitwise tricks to map 0/1 bits to float values directly.
// Bit 1 -> 1.0f (0x3F800000), Bit 0 -> -1.0f (0xBF800000)
// We can use vector table lookups (vtbl/vld1_lane) or simple AND/OR operations
// to construct the vector.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a = rowA[p];
            float32x4_t v_a = vdupq_n_f32(a);
            float32x4_t v_neg_a = vnegq_f32(v_a);
            
            const uint32_t* rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                for (size_t b = 0; b < 8; ++b) {
                    uint32_t bits = (packed >> (b * 4)) & 0xF;
                    
                    float f[4];
                    f[0] = (bits & 1) ? a : -a;
                    f[1] = (bits & 2) ? a : -a;
                    f[2] = (bits & 4) ? a : -a;
                    f[3] = (bits & 8) ? a : -a;
                    
                    float32x4_t v_f = vld1q_f32(f);
                    float32x4_t v_c = vld1q_f32(&rowC[j_int * 32 + b * 4]);
                    vst1q_f32(&rowC[j_int * 32 + b * 4], vaddq_f32(v_c, v_f));
                }
            }
        }
    }
}
