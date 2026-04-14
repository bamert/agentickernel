#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time sign table: 8 floats (+1.0f or -1.0f) per possible byte value.
constexpr std::array<std::array<float, 8>, 256> sign_table = [](){
    std::array<std::array<float, 8>, 256> tbl{};
    for (size_t b = 0; b < 256; ++b) {
        for (size_t bit = 0; bit < 8; ++bit) {
            tbl[b][bit] = ((b >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Matrix multiply processing two rows of A (and thus C) together to reuse B accesses.
// A: M×K float (row‑major)
// B: K×K binary packed (1 → +1.0f, 0 → -1.0f) stored as K/32 uint32_t per row
// C: M×K float (row‑major)
// K is a multiple of 32.
void matmul(const float* __restrict A,
            const uint32_t* __restrict B,
            float* __restrict C,
            size_t M,
            size_t K)
{
    const size_t K_ints = K / 32; // number of 32‑bit words per row of B

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
            const uint32_t* b_ptr = B + block; // points to B[0][block]
            // Unroll inner K loop by 4 (K is multiple of 32 ⇒ also multiple of 4).
            for (size_t p = 0; p + 3 < K; p += 4) {
                // ---- element p ----
                float a0 = a_row0[p];
                float a1 = a_row1[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8)  & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                // accumulate for both rows
                for (int c = 0; c < 8; ++c) {
                    acc0[c]     += a0 * tbl0[c];
                    acc1[c]     += a1 * tbl0[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[8 + c]  += a0 * tbl1[c];
                    acc1[8 + c]  += a1 * tbl1[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[16 + c] += a0 * tbl2[c];
                    acc1[16 + c] += a1 * tbl2[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[24 + c] += a0 * tbl3[c];
                    acc1[24 + c] += a1 * tbl3[c];
                }
                // ---- element p+1 ----
                a0 = a_row0[p + 1];
                a1 = a_row1[p + 1];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu;
                b1 = (packed >> 8)  & 0xFFu;
                b2 = (packed >> 16) & 0xFFu;
                b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_1 = sign_table[b0];
                const auto& tbl1_1 = sign_table[b1];
                const auto& tbl2_1 = sign_table[b2];
                const auto& tbl3_1 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]     += a0 * tbl0_1[c];
                    acc1[c]     += a1 * tbl0_1[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[8 + c]  += a0 * tbl1_1[c];
                    acc1[8 + c]  += a1 * tbl1_1[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[16 + c] += a0 * tbl2_1[c];
                    acc1[16 + c] += a1 * tbl2_1[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[24 + c] += a0 * tbl3_1[c];
                    acc1[24 + c] += a1 * tbl3_1[c];
                }
                // ---- element p+2 ----
                a0 = a_row0[p + 2];
                a1 = a_row1[p + 2];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu;
                b1 = (packed >> 8)  & 0xFFu;
                b2 = (packed >> 16) & 0xFFu;
                b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_2 = sign_table[b0];
                const auto& tbl1_2 = sign_table[b1];
                const auto& tbl2_2 = sign_table[b2];
                const auto& tbl3_2 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]     += a0 * tbl0_2[c];
                    acc1[c]     += a1 * tbl0_2[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[8 + c]  += a0 * tbl1_2[c];
                    acc1[8 + c]  += a1 * tbl1_2[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[16 + c] += a0 * tbl2_2[c];
                    acc1[16 + c] += a1 * tbl2_2[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[24 + c] += a0 * tbl3_2[c];
                    acc1[24 + c] += a1 * tbl3_2[c];
                }
                // ---- element p+3 ----
                a0 = a_row0[p + 3];
                a1 = a_row1[p + 3];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu;
                b1 = (packed >> 8)  & 0xFFu;
                b2 = (packed >> 16) & 0xFFu;
                b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_3 = sign_table[b0];
                const auto& tbl1_3 = sign_table[b1];
                const auto& tbl2_3 = sign_table[b2];
                const auto& tbl3_3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]     += a0 * tbl0_3[c];
                    acc1[c]     += a1 * tbl0_3[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[8 + c]  += a0 * tbl1_3[c];
                    acc1[8 + c]  += a1 * tbl1_3[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[16 + c] += a0 * tbl2_3[c];
                    acc1[16 + c] += a1 * tbl2_3[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[24 + c] += a0 * tbl3_3[c];
                    acc1[24 + c] += a1 * tbl3_3[c];
                }
            }
            // Process any leftover K elements (should not happen for multiples of 32).
            for (size_t p = K - (K % 4); p < K; ++p) {
                float a0 = a_row0[p];
                float a1 = a_row1[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8)  & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]     += a0 * tbl0[c];
                    acc1[c]     += a1 * tbl0[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[8 + c]  += a0 * tbl1[c];
                    acc1[8 + c]  += a1 * tbl1[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[16 + c] += a0 * tbl2[c];
                    acc1[16 + c] += a1 * tbl2[c];
                }
                for (int c = 0; c < 8; ++c) {
                    acc0[24 + c] += a0 * tbl3[c];
                    acc1[24 + c] += a1 * tbl3[c];
                }
            }
            // Store results for both rows.
            for (int t = 0; t < 32; ++t) {
                c_row0[base + t] = acc0[t];
                c_row1[base + t] = acc1[t];
            }
        }
    }
    // Handle possible last row if M is odd.
    if (i < M) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block) {
            float acc[32] = {};
            const size_t base = block * 32;
            const uint32_t* b_ptr = B + block;
            for (size_t p = 0; p + 3 < K; p += 4) {
                // element p
                float a0 = a_row[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8)  & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) acc[c]     += a0 * tbl0[c];
                for (int c = 0; c < 8; ++c) acc[8 + c]  += a0 * tbl1[c];
                for (int c = 0; c < 8; ++c) acc[16 + c] += a0 * tbl2[c];
                for (int c = 0; c < 8; ++c) acc[24 + c] += a0 * tbl3[c];
                // element p+1
                a0 = a_row[p + 1];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_1 = sign_table[b0];
                const auto& tbl1_1 = sign_table[b1];
                const auto& tbl2_1 = sign_table[b2];
                const auto& tbl3_1 = sign_table[b3];
                for (int c = 0; c < 8; ++c) acc[c]     += a0 * tbl0_1[c];
                for (int c = 0; c < 8; ++c) acc[8 + c]  += a0 * tbl1_1[c];
                for (int c = 0; c < 8; ++c) acc[16 + c] += a0 * tbl2_1[c];
                for (int c = 0; c < 8; ++c) acc[24 + c] += a0 * tbl3_1[c];
                // element p+2
                a0 = a_row[p + 2];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_2 = sign_table[b0];
                const auto& tbl1_2 = sign_table[b1];
                const auto& tbl2_2 = sign_table[b2];
                const auto& tbl3_2 = sign_table[b3];
                for (int c = 0; c < 8; ++c) acc[c]     += a0 * tbl0_2[c];
                for (int c = 0; c < 8; ++c) acc[8 + c]  += a0 * tbl1_2[c];
                for (int c = 0; c < 8; ++c) acc[16 + c] += a0 * tbl2_2[c];
                for (int c = 0; c < 8; ++c) acc[24 + c] += a0 * tbl3_2[c];
                // element p+3
                a0 = a_row[p + 3];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0_3 = sign_table[b0];
                const auto& tbl1_3 = sign_table[b1];
                const auto& tbl2_3 = sign_table[b2];
                const auto& tbl3_3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) acc[c]     += a0 * tbl0_3[c];
                for (int c = 0; c < 8; ++c) acc[8 + c]  += a0 * tbl1_3[c];
                for (int c = 0; c < 8; ++c) acc[16 + c] += a0 * tbl2_3[c];
                for (int c = 0; c < 8; ++c) acc[24 + c] += a0 * tbl3_3[c];
            }
            // remaining K elements (unlikely)
            for (size_t p = K - (K % 4); p < K; ++p) {
                float a0 = a_row[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) acc[c]     += a0 * tbl0[c];
                for (int c = 0; c < 8; ++c) acc[8 + c]  += a0 * tbl1[c];
                for (int c = 0; c < 8; ++c) acc[16 + c] += a0 * tbl2[c];
                for (int c = 0; c < 8; ++c) acc[24 + c] += a0 * tbl3[c];
            }
            for (int t = 0; t < 32; ++t) {
                c_row[base + t] = acc[t];
            }
        }
    }
}
