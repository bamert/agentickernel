#pragma once

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Matrix multiplication – tabulated sign lookup version.
 * Uses a per‑call lookup table of +1/-1 signs for 8‑bit chunks.
 * Should improve performance over per‑bit operations.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t RowC = K;

    // Pre‑compute a lookup table of signs for 8‑bit values.
    // table[byte_val * 8 + bit] = +1.0f or -1.0f
    float signs_table[256 * 8];
    for (int byte_val = 0; byte_val < 256; ++byte_val) {
        for (int bit = 0; bit < 8; ++bit) {
            signs_table[(byte_val << 3) | bit] = (byte_val & (1 << bit)) ? 1.0f : -1.0f;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        const float* Ai = A + i * K;
        float* Ci   = C + i * RowC;

        // Zero the output row
        for (size_t j = 0; j < K; ++j) Ci[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t base = w * 32;

                // Process 4 bytes of the 32‑bit word
                for (int byte_offset = 0; byte_offset < 4; ++byte_offset) {
                    uint8_t byte_val = static_cast<uint8_t>(word >> (byte_offset * 8));
                    const float* sign_ptr = &signs_table[(byte_val << 3)];
                    size_t col_start = base + byte_offset * 8;

                    // Unroll 8 columns
                    Ci[col_start + 0] += a_val * sign_ptr[0];
                    Ci[col_start + 1] += a_val * sign_ptr[1];
                    Ci[col_start + 2] += a_val * sign_ptr[2];
                    Ci[col_start + 3] += a_val * sign_ptr[3];
                    Ci[col_start + 4] += a_val * sign_ptr[4];
                    Ci[col_start + 5] += a_val * sign_ptr[5];
                    Ci[col_start + 6] += a_val * sign_ptr[6];
                    Ci[col_start + 7] += a_val * sign_ptr[7];
                }
            }
        }
    }
}
