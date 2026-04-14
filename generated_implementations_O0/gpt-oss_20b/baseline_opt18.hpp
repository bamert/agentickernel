#pragma once

using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Matrix multiplication – compute sign on the fly instead of lookup table.
 * Each 32‑bit word is processed 32 times: shift, mask, compute sign.
 * No extra memory traffic, but more arithmetic per element.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float* Ci = C + i * K;

        /* Zero output row */
        for (size_t j = 0; j < K; ++j) Ci[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t base = w * 32;

                for (size_t b = 0; b < 32; ++b) {
                    const size_t col = base + b;
                    float sign = 1.0f - 2.0f * static_cast<float>((word >> b) & 1u);
                    Ci[col] += a_val * sign;
                }
            }
        }
    }
}
