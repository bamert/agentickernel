#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: move inner dimension loop to outer position for better cache locality.
// The algorithm is essentially the same as the baseline but with a different loop ordering:
//   for i (rows of A)
//      for p (shared dimension)
//         load a_val = A[i*K + p]
//         for j (columns of B)
//             // fetch sign from B[p, j]
//             float sum = C[i*K + j] + a_val * ((B[p*K_ints + (j/32)] >> (j%32)) & 1 ? 1.0f : -1.0f);
//             C[i*K + j] = sum;
// This ordering keeps a_val and the corresponding block of C in L1 cache while we walk through all
// columns j quickly. The memory accesses to B are still sequential across columns.
// For simplicity we initialize C to 0 at the start.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    // Zero C.
    for (size_t idx = 0; idx < M * K; ++idx) {
        C[idx] = 0.0f;
    }

    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            // Prepare pointer to the row p of B packed representation.
            const uint32_t* B_row = &B[p * K_ints];
            for (size_t j = 0; j < K; ++j) {
                uint32_t packed = B_row[j / 32];
                uint32_t bit = (packed >> (j % 32)) & 1U;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val * sign;
            }
        }
    }
}
