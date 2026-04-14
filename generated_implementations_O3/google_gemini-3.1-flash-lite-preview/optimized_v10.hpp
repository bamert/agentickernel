#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v10:
// To beat 24.15ms, we need to reduce the overhead in the inner loop.
// The current bottleneck is calculating the 4 float values based on bits.
// We can use NEON bitwise clear/set/vector-select to build the float vector of +/-val.
// (1.0f in hex is 0x3F800000, -1.0f is 0xBF800000)
// This lets us construct the result without loading from memory.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ rowC = &C[i * K];
        const float* __restrict__ rowA = &A[i * K];

        // Initialize output
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float val = rowA[p];
            float32x4_t v_a = vdupq_n_f32(val);
            float32x4_t v_neg_a = vnegq_f32(v_a);
            
            const uint32_t* __restrict__ rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                for (size_t col = 0; col < 8; ++col) {
                    uint32_t bits = (packed >> (col * 4)) & 0xF;
                    
                    // Construct 4-vector directly
                    float res[4];
                    res[0] = (bits & 1) ? val : -val;
                    res[1] = (bits & 2) ? val : -val;
                    res[2] = (bits & 4) ? val : -val;
                    res[3] = (bits & 8) ? val : -val;

                    float32x4_t v_signs = vld1q_f32(res);
                    float* ptr = &rowC[j_int * 32 + col * 4];
                    vst1q_f32(ptr, vaddq_f32(vld1q_f32(ptr), v_signs));
                }
            }
        }
    }
}
