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

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float* Ci   = C + i * K;

        for (size_t j = 0; j < K; ++j) Ci[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t base = w * 32;

                for (size_t b = 0; b < 32; b += 4) {
                    const uint32_t mask = (word >> b) & 0xFu;
                    const float sign0 = 1.0f - 2.0f * static_cast<float>(mask & 1u);
                    const float sign1 = 1.0f - 2.0f * static_cast<float>((mask >> 1) & 1u);
                    const float sign2 = 1.0f - 2.0f * static_cast<float>((mask >> 2) & 1u);
                    const float sign3 = 1.0f - 2.0f * static_cast<float>((mask >> 3) & 1u);

                    Ci[base + b + 0] += a_val * sign0;
                    Ci[base + b + 1] += a_val * sign1;
                    Ci[base + b + 2] += a_val * sign2;
                    Ci[base + b + 3] += a_val * sign3;
                }
            }
        }
    }
}
