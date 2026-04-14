#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v13:
// The current best is v3 (24.1548ms).
// Can we further improve by reducing the floating-point construction of the array?
// Using NEON instructions to perform the sign-flip based on the bits.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        // Clear rowC
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float val = rowA[p];
            float32x4_t v_val = vdupq_n_f32(val);
            float32x4_t v_neg_val = vnegq_f32(v_val);
            
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    // Construct 4-vector signs.
                    // This is still the bottleneck, but let's try to unroll/inline it even more.
                    float32x4_t v_signs;
                    
                    // We can use bitwise operations to set signs.
                    // But we don't have bit-to-float conversions, so we'll stick to a compact array.
                    float f[4];
                    f[0] = (bits & 1) ? val : -val;
                    f[1] = (bits & 2) ? val : -val;
                    f[2] = (bits & 4) ? val : -val;
                    f[3] = (bits & 8) ? val : -val;
                    
                    float32x4_t v_f = vld1q_f32(f);
                    float* ptr = &rowC[j_int * 32 + col * 4];
                    vst1q_f32(ptr, vaddq_f32(vld1q_f32(ptr), v_f));
                }
            }
        }
    }
}
