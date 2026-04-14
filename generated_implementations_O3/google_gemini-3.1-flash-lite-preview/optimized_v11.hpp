#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v11:
// The current best is v3 (24.1548ms).
// Let's try to improve on it by using vector shifting/masking to eliminate the 
// branchy construction of signs, potentially reducing instruction count.
// We can use the bit pattern (1.0f vs -1.0f) directly.
// +1.0f is 0x3F800000, -1.0f is 0xBF800000.
// If the bit is '1', we want 0x3F800000. If '0', we want 0xBF800000.
// This is exactly: 0x3F800000 ^ (bit == 0 ? 0x80000000 : 0)

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    const uint32_t POS_ONE_HEX = 0x3F800000;
    const uint32_t NEG_ONE_HEX = 0xBF800000;
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* rowC = &C[i * K];
        const float* rowA = &A[i * K];

        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float val = rowA[p];
            // If val is positive, +1 = val, -1 = -val.
            // If val is negative, +1 = val, -1 = -val.
            // This is just scaling the signed values.
            
            float abs_val = (val >= 0) ? val : -val;
            float neg_abs_val = -abs_val;
            
            const uint32_t* rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    float f[4];
                    f[0] = (bits & 1) ? abs_val : neg_abs_val;
                    f[1] = (bits & 2) ? abs_val : neg_abs_val;
                    f[2] = (bits & 4) ? abs_val : neg_abs_val;
                    f[3] = (bits & 8) ? abs_val : neg_abs_val;
                    
                    // Add val sign adjustment if original val was negative
                    if (val < 0) {
                        for(int k=0; k<4; ++k) f[k] = -f[k];
                    }
                    
                    float32x4_t v_f = vld1q_f32(f);
                    float* target = &rowC[j_int * 32 + col * 4];
                    vst1q_f32(target, vaddq_f32(vld1q_f32(target), v_f));
                }
            }
        }
    }
}
