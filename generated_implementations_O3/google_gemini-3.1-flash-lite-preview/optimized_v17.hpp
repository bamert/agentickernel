#pragma once
#include <cstdint>
#include <cstddef>

// Optimized v17:
// The current best (v3, 24.15ms) is structurally very similar to v16,
// but let's try to remove the stack-allocated `f[4]` entirely and 
// calculate the signs using pure vector operations if possible.
// Actually, using vld1q_f32 on a local array is likely the bottleneck
// due to stack memory and store-to-load forwarding.
// We can use bit manipulation on the vector registers themselves if we had
// bit-based select, but lacking that, let's keep it simple.

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
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                const uint32_t packed = rowB[j_int];
                
                // Process 32 elements.
                // We'll iterate 8 blocks of 4 elements.
                for (size_t col = 0; col < 32; col += 4) {
                    uint32_t bits = (packed >> col) & 0xF;
                    
                    float f[4];
                    f[0] = (bits & 1) ? val : n_val;
                    f[1] = (bits & 2) ? val : n_val;
                    f[2] = (bits & 4) ? val : n_val;
                    f[3] = (bits & 8) ? val : n_val;
                    
                    float* __restrict__ target = rowC + j_int * 32 + col;
                    float32x4_t v_rowC = vld1q_f32(target);
                    vst1q_f32(target, vaddq_f32(v_rowC, vld1q_f32(f)));
                }
            }
        }
    }
}
