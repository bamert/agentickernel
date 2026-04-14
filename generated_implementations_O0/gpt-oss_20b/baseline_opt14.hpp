#pragma once

// Basic type definitions for compilation without external headers
using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Matrix multiplication – highly‑tuned scalar implementation using a
 * lookup table of +/-1.0f for each 8‑bit value and several small
 * optimisations:
 *   • Use the *restrict* keyword to allow compilers to optimise aliasing.
 *   • Compute K_ints via right shift.
 *   • Zero the output row with a small loop that writes 4 floats at once.
 *   • Unroll the byte loop to avoid an inner loop counter.
 */
void matmul(const float* restrict A, const uint32_t* restrict B, float* restrict C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;          // K / 32
    const size_t RowC   = K;

    /* Sign lookup table: 256 * 8 floats (+1/-1) */
    float signs_tbl[256 * 8];
    for (int byte_val = 0; byte_val < 256; ++byte_val) {
        for (int bit = 0; bit < 8; ++bit) {
            signs_tbl[(byte_val << 3) | bit] = (byte_val & (1 << bit)) ? 1.0f : -1.0f;
        }
    }

    for (size_t i = 0; i < M; ++i) {
        const float* restrict Ai = A + i * K;
        float* restrict Ci       = C + i * RowC;

        /* Zero the output row – write 4 floats at a time */
        for (size_t j = 0; j < K; j += 4) {
            Ci[j] = 0.0f; Ci[j+1] = 0.0f; Ci[j+2] = 0.0f; Ci[j+3] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            const float a_val   = Ai[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w) {
                const uint32_t word = B_row[w];
                const size_t base   = w * 32;

                /* Unrolled 4‑byte processing */
                {
                    uint8_t b0 = static_cast<uint8_t>(word >> 0 );
                    const float* s0 = &signs_tbl[(b0 << 3)];
                    Ci[base + 0] += a_val * s0[0];
                    Ci[base + 1] += a_val * s0[1];
                    Ci[base + 2] += a_val * s0[2];
                    Ci[base + 3] += a_val * s0[3];
                    Ci[base + 4] += a_val * s0[4];
                    Ci[base + 5] += a_val * s0[5];
                    Ci[base + 6] += a_val * s0[6];
                    Ci[base + 7] += a_val * s0[7];
                }
                {
                    uint8_t b1 = static_cast<uint8_t>(word >> 8 );
                    const float* s1 = &signs_tbl[(b1 << 3)];
                    Ci[base + 8]  += a_val * s1[0];
                    Ci[base + 9]  += a_val * s1[1];
                    Ci[base +10]  += a_val * s1[2];
                    Ci[base +11]  += a_val * s1[3];
                    Ci[base +12]  += a_val * s1[4];
                    Ci[base +13]  += a_val * s1[5];
                    Ci[base +14]  += a_val * s1[6];
                    Ci[base +15]  += a_val * s1[7];
                }
                {
                    uint8_t b2 = static_cast<uint8_t>(word >> 16 );
                    const float* s2 = &signs_tbl[(b2 << 3)];
                    Ci[base +16]  += a_val * s2[0];
                    Ci[base +17]  += a_val * s2[1];
                    Ci[base +18]  += a_val * s2[2];
                    Ci[base +19]  += a_val * s2[3];
                    Ci[base +20]  += a_val * s2[4];
                    Ci[base +21]  += a_val * s2[5];
                    Ci[base +22]  += a_val * s2[6];
                    Ci[base +23]  += a_val * s2[7];
                }
                {
                    uint8_t b3 = static_cast<uint8_t>(word >> 24 );
                    const float* s3 = &signs_tbl[(b3 << 3)];
                    Ci[base +24]  += a_val * s3[0];
                    Ci[base +25]  += a_val * s3[1];
                    Ci[base +26]  += a_val * s3[2];
                    Ci[base +27]  += a_val * s3[3];
                    Ci[base +28]  += a_val * s3[4];
                    Ci[base +29]  += a_val * s3[5];
                    Ci[base +30]  += a_val * s3[6];
                    Ci[base +31]  += a_val * s3[7];
                }
            }
        }
    }
}
