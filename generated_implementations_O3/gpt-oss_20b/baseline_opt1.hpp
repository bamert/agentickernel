#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: Row-major traversal of A, per-row processing of B for better cache locality.
// Computes C = A * B where B is packed binary matrix (1 bit -> +1.0f, 0 bit -> -1.0f).
// Assumes K is a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of 32-bit words per row of B
    // Main loop over rows of A
    for (size_t i = 0; i < M; ++i) {
        // Zero the output row
        float* c_row = C + i * K;
        const float* a_row = A + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }
        // For each element p in the shared dimension
        for (size_t p = 0; p < K; ++p) {
            const float a_val = a_row[p];
            // Process 32 columns at a time
            const size_t b_base = p * K_ints;
            for (size_t wi = 0; wi < K_ints; ++wi) {
                uint32_t packed = B[b_base + wi];
                // Unroll inner loop of 32 bits; we manually write loop for clarity
                for (size_t t = 0; t < 32; ++t) {
                    size_t j = wi * 32 + t;
                    // Sign: +1 when bit is 1, -1 when bit is 0
                    float sign = ((packed >> t) & 1U) ? 1.0f : -1.0f;
                    c_row[j] += a_val * sign;
                }
            }
        }
    }
}
