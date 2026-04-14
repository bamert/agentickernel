#pragma once

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Further optimized matrix multiplication.
 * Matrix C = Matrix A * Matrix B (Naïve Textbook Method).
 * A: Float matrix (M rows, K cols).
 * B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
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

        // Initialize this row of the output.
        for (size_t j = 0; j < K; ++j) {
            Ci[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints; // Start of packed B row

            for (size_t w = 0; w < K_ints; ++w) {
                uint32_t word = B_row[w];

                // Unroll 16 bits per inner iteration to reduce loop overhead.
                // We'll compute two chunks of 8 bits each.
                for (size_t b = 0; b < 32; b += 16) {
                    // Process bits b..b+7
                    for (size_t shift = 0; shift < 8; ++shift) {
                        size_t col = w * 32 + b + shift;
                        float sign = 1.0f - 2.0f * static_cast<float>((word >> shift) & 1u);
                        Ci[col] += a_val * sign;
                    }
                    // Process bits b+8..b+15
                    uint32_t word_shifted = word >> 8;
                    for (size_t shift = 0; shift < 8; ++shift) {
                        size_t col = w * 32 + b + 8 + shift;
                        float sign = 1.0f - 2.0f * static_cast<float>((word_shifted >> shift) & 1u);
                        Ci[col] += a_val * sign;
                    }
                }
            }
        }
    }
}
