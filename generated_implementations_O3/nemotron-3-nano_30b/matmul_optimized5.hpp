#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: accumulate per column in a temporary buffer then store.
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K >> 5; // K / 32

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        // Temporary accumulator for this row of C (size K)
        float accum[K]; // Variable‑length array, OK because K is known at runtime
        for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;

            // Process each 32‑bit chunk of the packed row
            #pragma unroll 32
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                // Process each of the 32 bits
                #pragma unroll
                for (int t = 0; t < 32; ++t) {
                    float sign = ((word >> t) & 1u) ? 1.0f : -1.0f;
                    size_t col_idx = chunk * 32 + t;
                    accum[col_idx] += a_val * sign;
                }
            }
        }

        // Store the results back to global memory
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = accum[j];
        }
    }
}