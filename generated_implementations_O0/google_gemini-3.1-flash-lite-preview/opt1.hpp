#pragma once
#include <cstdint>
#include <cstddef>

// Optimization 1: Loop reordering and pre-calculation of signs for cache efficiency.
// Transposing the logic to process chunks of B and accumulating into C.
// The inner loop over 'p' is now the outermost inside the 'i' loop, 
// allowing better reuse of A values.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Initialize C to zero
    for (size_t i = 0; i < M * K; ++i) {
        C[i] = 0.0f;
    }

    // Process A[i][p] * B[p][j]
    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            const uint32_t* row_B_packed = &B[p * K_ints];

            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = row_B_packed[j_int];
                for (size_t bit = 0; bit < 32; ++bit) {
                    float sign = ((packed >> bit) & 1) ? 1.0f : -1.0f;
                    row_C[j_int * 32 + bit] += val_a * sign;
                }
            }
        }
    }
}
