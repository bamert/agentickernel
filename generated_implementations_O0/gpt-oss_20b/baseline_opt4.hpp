#pragma once

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Optimized matrix multiplication with branchless per‑bit accumulation.
 * Matrix C = Matrix A * Matrix B (Naïve Textbook Method).
 * A: Float matrix (M rows, K cols).
 * B: Packed binary matrix (K rows, K cols). 1 bit  = +1.0f, 0 bit = -1.0f.
 * C: Output float matrix (M rows, K cols).
 * K is guaranteed to be a multiple of 32.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;          // Number of 32‑bit words per row of B

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float* Ci   = C + i * K;

        // Zero this row of the output matrix.
        for (size_t j = 0; j < K; ++j) {
            Ci[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t base = w * 32;   // Column offset for this word

                for (size_t b = 0; b < 32; ++b) {
                    const size_t col = base + b;
                    // Extract the bit and add with the appropriate sign.
                    const float sign = (static_cast<float>((word >> b) & 1u)) * 2.0f - 1.0f;
                    Ci[col] += a_val * sign;
                }
            }
        }
    }
}
