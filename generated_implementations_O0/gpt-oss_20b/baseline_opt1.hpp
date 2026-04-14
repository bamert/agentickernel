#pragma once

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Optimized matrix multiplication.
 * Matrix C = Matrix A * Matrix B (Naïve Textbook Method).
 * A: Float matrix (M rows, K cols).
 * B: Packed binary matrix (K rows, K cols). 1 bit  = +1.0f, 0 bit = -1.0f.
 * C: Output float matrix (M rows, K cols).
 * K is guaranteed to be a multiple of 32.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;          // Number of 32‑bit words per row of B
    const size_t RowA = K;
    const size_t RowC = K;

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * RowA;
        float* Ci   = C + i * RowC;

        // Initialize this row of the result matrix.
        for (size_t j = 0; j < K; ++j) {
            Ci[j] = 0.0f;
        }

        // Accumulate for each element of the row of A.
        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints; // Start of packed B row

            for (size_t w = 0; w < K_ints; ++w) {
                uint32_t word = B_row[w];

                for (size_t b = 0; b < 32; ++b) {
                    const size_t j = w * 32 + b; // Column index in C
                    // Convert the bit to a sign (+1 or -1).
                    const float bit = static_cast<float>((word >> b) & 1u);
                    const float sign = 2.0f * bit - 1.0f; // 1 -> +1, 0 -> -1
                    Ci[j] += a_val * sign;
                }
            }
        }
    }
}
