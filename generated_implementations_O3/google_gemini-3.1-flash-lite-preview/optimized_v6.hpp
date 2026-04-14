#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v6:
// The bottleneck in previous versions is the bit-to-sign conversion.
// We can use a bit-manipulation trick:
// (bit == 1) -> sign bit 0, (bit == 0) -> sign bit 1.
// By shifting, mask, and using NEON bitwise ops, we can avoid standard branching.
// A float is 0x3F800000 (+1.0) and 0xBF800000 (-1.0).
// Notice the sign bit is at bit 31.
// If we have a bit 'b' (0 or 1), the float is:
// 0x3F800000 ^ ((b == 0) << 31)

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* rowC = &C[i * K];
        const float* rowA = &A[i * K];

        // Initialize output row to zero
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        // Process in 32-bit chunks along K
        for (size_t p = 0; p < K; ++p) {
            float a = rowA[p];
            float32x4_t v_a = vdupq_n_f32(a);
            float32x4_t v_neg_a = vnegq_f32(v_a);
            
            const uint32_t* rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                for (size_t b = 0; b < 8; ++b) {
                    uint32_t bits = (packed >> (b * 4)) & 0xF;
                    
                    // We need a vector result for 4 elements based on 4 bits
                    alignas(16) float f[4];
                    f[0] = (bits & 1) ? a : -a;
                    f[1] = (bits & 2) ? a : -a;
                    f[2] = (bits & 4) ? a : -a;
                    f[3] = (bits & 8) ? a : -a;
                    
                    float32x4_t v_f = vld1q_f32(f);
                    float* target = &rowC[j_int * 32 + b * 4];
                    vst1q_f32(target, vaddq_f32(vld1q_f32(target), v_f));
                }
            }
        }
    }
}
