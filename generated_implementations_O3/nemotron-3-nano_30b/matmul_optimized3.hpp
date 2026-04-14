#pragma once
#include <cstdint>
#include <cstddef>

// Optimized matmul: process whole 32‑bit words at once, accumulate per column.
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K >> 5;                     // K / 32

    for (size_t i = 0; i < M; ++i) {           // Each row of A (and C)
        const float* A_row = A + i * K;         // Pointer to row i of A
        float*       C_row = C + i * K;         // Pointer to row i of C

        // Process each row of packed B
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];             // Load scalar from A
            const uint32_t* B_row = B + p * K_ints; // Packed bits for row p

            // Process 32‑column chunks
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t packed = B_row[chunk];

                // Unroll the 32‑bit loop – one iteration per column in the chunk
                #pragma unroll 32
                for (int t = 0; t < 32; ++t) {
                    // Extract bit t and convert to +1.0f or -1.0f
                    float sign = ((packed >> t) & 1u) ? 1.0f : -1.0f;
                    size_t col_idx = chunk * 32 + t;   // Global column index
                    C_row[col_idx] += a_val * sign;   // Accumulate contribution
                }
            }
        }
    }
}