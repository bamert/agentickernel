#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time table: for each byte value store 8 floats (+1.0f or -1.0f).
constexpr std::array<std::array<float, 8>, 256> sign_table = [](){
    std::array<std::array<float, 8>, 256> tbl{};
    for (size_t b = 0; b < 256; ++b) {
        for (size_t bit = 0; bit < 8; ++bit) {
            tbl[b][bit] = ((b >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Matrix multiply processing two rows of A (and C) together, unrolling the inner K loop by 8.
// A: M×K float (row‑major)
// B: K×K binary packed (1 → +1.0f, 0 → -1.0f) stored as K/32 uint32_t per row
// C: M×K float (row‑major)
// K is a multiple of 32 (and therefore of 8).
void matmul(const float* __restrict A,
            const uint32_t* __restrict B,
            float* __restrict C,
            size_t M,
            size_t K)
{
    const size_t K_ints = K / 32; // words per row of B

    size_t i = 0;
    for (; i + 1 < M; i += 2) {
        const float* a_row0 = A + i * K;
        const float* a_row1 = A + (i + 1) * K;
        float* c_row0 = C + i * K;
        float* c_row1 = C + (i + 1) * K;
        for (size_t block = 0; block < K_ints; ++block) {
            float acc0[32] = {};
            float acc1[32] = {};
            const size_t base = block * 32;
            const uint32_t* b_ptr = B + block; // start of this 32‑column block in row 0
            // Unroll by 8 elements of A.
            for (size_t p = 0; p + 7 < K; p += 8) {
                // Prefetch next B words and A values to help cache.
                __builtin_prefetch(b_ptr + 32);
                __builtin_prefetch(a_row0 + p + 16);
                __builtin_prefetch(a_row1 + p + 16);
                // ---- element p ----
                float a0 = a_row0[p];
                float a1 = a_row1[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0[c];
                    acc1[c]      += a1 * tbl0[c];
                    acc0[8 + c]  += a0 * tbl1[c];
                    acc1[8 + c]  += a1 * tbl1[c];
                    acc0[16 + c] += a0 * tbl2[c];
                    acc1[16 + c] += a1 * tbl2[c];
                    acc0[24 + c] += a0 * tbl3[c];
                    acc1[24 + c] += a1 * tbl3[c];
                }
                // ---- element p+1 ----
                a0 = a_row0[p + 1];
                a1 = a_row1[p + 1];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_1 = sign_table[b0];
                const auto& tbl1_1 = sign_table[b1];
                const auto& tbl2_1 = sign_table[b2];
                const auto& tbl3_1 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0_1[c];
                    acc1[c]      += a1 * tbl0_1[c];
                    acc0[8 + c]  += a0 * tbl1_1[c];
                    acc1[8 + c]  += a1 * tbl1_1[c];
                    acc0[16 + c] += a0 * tbl2_1[c];
                    acc1[16 + c] += a1 * tbl2_1[c];
                    acc0[24 + c] += a0 * tbl3_1[c];
                    acc1[24 + c] += a1 * tbl3_1[c];
                }
                // ---- element p+2 ----
                a0 = a_row0[p + 2];
                a1 = a_row1[p + 2];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_2 = sign_table[b0];
                const auto& tbl1_2 = sign_table[b1];
                const auto& tbl2_2 = sign_table[b2];
                const auto& tbl3_2 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0_2[c];
                    acc1[c]      += a1 * tbl0_2[c];
                    acc0[8 + c]  += a0 * tbl1_2[c];
                    acc1[8 + c]  += a1 * tbl1_2[c];
                    acc0[16 + c] += a0 * tbl2_2[c];
                    acc1[16 + c] += a1 * tbl2_2[c];
                    acc0[24 + c] += a0 * tbl3_2[c];
                    acc1[24 + c] += a1 * tbl3_2[c];
                }
                // ---- element p+3 ----
                a0 = a_row0[p + 3];
                a1 = a_row1[p + 3];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_3 = sign_table[b0];
                const auto& tbl1_3 = sign_table[b1];
                const auto& tbl2_3 = sign_table[b2];
                const auto& tbl3_3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0_3[c];
                    acc1[c]      += a1 * tbl0_3[c];
                    acc0[8 + c]  += a0 * tbl1_3[c];
                    acc1[8 + c]  += a1 * tbl1_3[c];
                    acc0[16 + c] += a0 * tbl2_3[c];
                    acc1[16 + c] += a1 * tbl2_3[c];
                    acc0[24 + c] += a0 * tbl3_3[c];
                    acc1[24 + c] += a1 * tbl3_3[c];
                }
                // ---- element p+4 ----
                a0 = a_row0[p + 4];
                a1 = a_row1[p + 4];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_4 = sign_table[b0];
                const auto& tbl1_4 = sign_table[b1];
                const auto& tbl2_4 = sign_table[b2];
                const auto& tbl3_4 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0_4[c];
                    acc1[c]      += a1 * tbl0_4[c];
                    acc0[8 + c]  += a0 * tbl1_4[c];
                    acc1[8 + c]  += a1 * tbl1_4[c];
                    acc0[16 + c] += a0 * tbl2_4[c];
                    acc1[16 + c] += a1 * tbl2_4[c];
                    acc0[24 + c] += a0 * tbl3_4[c];
                    acc1[24 + c] += a1 * tbl3_4[c];
                }
                // ---- element p+5 ----
                a0 = a_row0[p + 5];
                a1 = a_row1[p + 5];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_5 = sign_table[b0];
                const auto& tbl1_5 = sign_table[b1];
                const auto& tbl2_5 = sign_table[b2];
                const auto& tbl3_5 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0_5[c];
                    acc1[c]      += a1 * tbl0_5[c];
                    acc0[8 + c]  += a0 * tbl1_5[c];
                    acc1[8 + c]  += a1 * tbl1_5[c];
                    acc0[16 + c] += a0 * tbl2_5[c];
                    acc1[16 + c] += a1 * tbl2_5[c];
                    acc0[24 + c] += a0 * tbl3_5[c];
                    acc1[24 + c] += a1 * tbl3_5[c];
                }
                // ---- element p+6 ----
                a0 = a_row0[p + 6];
                a1 = a_row1[p + 6];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_6 = sign_table[b0];
                const auto& tbl1_6 = sign_table[b1];
                const auto& tbl2_6 = sign_table[b2];
                const auto& tbl3_6 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0_6[c];
                    acc1[c]      += a1 * tbl0_6[c];
                    acc0[8 + c]  += a0 * tbl1_6[c];
                    acc1[8 + c]  += a1 * tbl1_6[c];
                    acc0[16 + c] += a0 * tbl2_6[c];
                    acc1[16 + c] += a1 * tbl2_6[c];
                    acc0[24 + c] += a0 * tbl3_6[c];
                    acc1[24 + c] += a1 * tbl3_6[c];
                }
                // ---- element p+7 ----
                a0 = a_row0[p + 7];
                a1 = a_row1[p + 7];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_7 = sign_table[b0];
                const auto& tbl1_7 = sign_table[b1];
                const auto& tbl2_7 = sign_table[b2];
                const auto& tbl3_7 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0_7[c];
                    acc1[c]      += a1 * tbl0_7[c];
                    acc0[8 + c]  += a0 * tbl1_7[c];
                    acc1[8 + c]  += a1 * tbl1_7[c];
                    acc0[16 + c] += a0 * tbl2_7[c];
                    acc1[16 + c] += a1 * tbl2_7[c];
                    acc0[24 + c] += a0 * tbl3_7[c];
                    acc1[24 + c] += a1 * tbl3_7[c];
                }
            }
            // Process any leftover K elements (should not happen for multiples of 32).
            for (size_t p = K - (K % 8); p < K; ++p) {
                float a0 = a_row0[p];
                float a1 = a_row1[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * tbl0[c];
                    acc1[c]      += a1 * tbl0[c];
                    acc0[8 + c]  += a0 * tbl1[c];
                    acc1[8 + c]  += a1 * tbl1[c];
                    acc0[16 + c] += a0 * tbl2[c];
                    acc1[16 + c] += a1 * tbl2[c];
                    acc0[24 + c] += a0 * tbl3[c];
                    acc1[24 + c] += a1 * tbl3[c];
                }
            }
            // Store results for the two rows.
            for (int t = 0; t < 32; ++t) {
                c_row0[base + t] = acc0[t];
                c_row1[base + t] = acc1[t];
            }
        }
    }
    // Handle possible final row when M is odd.
    if (i < M) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block) {
            float acc[32] = {};
            const size_t base = block * 32;
            const uint32_t* b_ptr = B + block;
            for (size_t p = 0; p + 7 < K; p += 8) {
                __builtin_prefetch(b_ptr + 32);
                __builtin_prefetch(a_row + p + 16);
                for (size_t off = 0; off < 8; ++off) {
                    float a_val = a_row[p + off];
                    uint32_t packed = *b_ptr; b_ptr += K_ints;
                    uint8_t b0 = packed & 0xFFu;
                    uint8_t b1 = (packed >> 8) & 0xFFu;
                    uint8_t b2 = (packed >> 16) & 0xFFu;
                    uint8_t b3 = (packed >> 24) & 0xFFu;
                    const auto& tbl0 = sign_table[b0];
                    const auto& tbl1 = sign_table[b1];
                    const auto& tbl2 = sign_table[b2];
                    const auto& tbl3 = sign_table[b3];
                    for (int c = 0; c < 8; ++c) {
                        acc[c]      += a_val * tbl0[c];
                        acc[8 + c]  += a_val * tbl1[c];
                        acc[16 + c] += a_val * tbl2[c];
                        acc[24 + c] += a_val * tbl3[c];
                    }
                }
            }
            for (size_t p = K - (K % 8); p < K; ++p) {
                float a_val = a_row[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc[c]      += a_val * tbl0[c];
                    acc[8 + c]  += a_val * tbl1[c];
                    acc[16 + c] += a_val * tbl2[c];
                    acc[24 + c] += a_val * tbl3[c];
                }
            }
            for (int t = 0; t < 32; ++t) {
                c_row[base + t] = acc[t];
            }
        }
    }
}
