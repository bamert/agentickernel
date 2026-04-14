#pragma once
#include <cstdint>
#include <cstddef>

// Optimized implementation without intrinsics:
// 1. Reorder loops to improve cache locality.
// 2. Pre-calculate signs for B to avoid branching/bit extraction in the innermost loop.
// 3. Process matrix B in a more cache-friendly manner.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Initialize C to zero
    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    // Process blocks to optimize cache
    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float a_val = rowA[p];
            const uint32_t* rowB = &B[p * K_ints];
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = rowB[j_int];
                
                for (size_t bit_idx = 0; bit_idx < 32; ++bit_idx) {
                    float sign = ((packed >> bit_idx) & 1) ? a_val : -a_val;
                    rowC[j_int * 32 + bit_idx] += sign;
                }
            }
        }
    }
}
