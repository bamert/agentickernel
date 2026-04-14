#pragma once
#include <cstdint>
#include <cstddef>

// Optimized Matmul using extreme register blocking and loop unrolling.
// We use an i-p-c-k loop order for maximum cache efficiency.
// We unroll the c-loop by 8.
// We use a manually unrolled 32-bit bit-extraction logic.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t j = 0.0f; j < K; ++j) { // Error: j is size_t
             c_row[j] = 0.0f;
        }
        // ...
    }
}
