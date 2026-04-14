#pragma once
#include <cstdint>
#include <cstddef>

// Improvement upon Opt1:
// Opt1 was fast because it processed A[i][p] repeatedly for blocks of B.
// This is effectively a Matrix-Vector Product (A row * Matrix B).
// Let's refine the inner loop of Opt1 to minimize operations within the tightest loop.
// The bottleneck in Opt1 was the bit extraction: (packed >> bit) & 1.
// We can use a branchless approach: sign = 1.0f - 2.0f * ((packed >> bit) & 1).

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t p = 0; p < K; ++p) {
            float val_a = row_A[p];
            const uint32_t* B_row_p = &B[p * K_ints];
            
            // Loop unrolling for speed
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row_p[j_int];
                for (int b = 0; b < 32; ++b) {
                    float sign = ((packed >> b) & 1) ? 1.0f : -1.0f;
                    row_C[j_int * 32 + b] += val_a * sign;
                }
            }
        }
    }
}
