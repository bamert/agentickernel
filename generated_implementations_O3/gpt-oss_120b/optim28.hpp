#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time sign table: 8 floats (+1.0f or -1.0f) for every possible byte value.
constexpr std::array<std::array<float, 8>, 256> sign_table = [](){
    std::array<std::array<float, 8>, 256> tbl{};
    for (size_t b = 0; b < 256; ++b) {
        for (size_t bit = 0; bit < 8; ++bit) {
            tbl[b][bit] = ((b >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Matrix multiply with inner‑K loop unrolled by 4 and pointer‑incremented access to B.
// K is guaranteed to be a multiple of 32 (hence also of 4).
void matmul(const float* __restrict A,
            const uint32_t* __restrict B,
            float* __restrict C,
            size_t M,
            size_t K)
{
    const size_t K_ints = K / 32; // 32‑bit words per row of B

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block) {
            float acc[32] = {};
            const size_t base = block * 32;
            const uint32_t* b_ptr = B + block; // points to B[0][block]
            // Unroll by 4 elements of A (K is multiple of 4).
            for (size_t p = 0; p < K; p += 4) {
                // ---- element p ----
                float a0 = a_row[p];
                uint32_t packed0 = *b_ptr; b_ptr += K_ints;
                uint8_t b0_0 = packed0 & 0xFFu;
                uint8_t b0_1 = (packed0 >> 8)  & 0xFFu;
                uint8_t b0_2 = (packed0 >> 16) & 0xFFu;
                uint8_t b0_3 = (packed0 >> 24) & 0xFFu;
                const auto& tbl0_0 = sign_table[b0_0];
                const auto& tbl0_1 = sign_table[b0_1];
                const auto& tbl0_2 = sign_table[b0_2];
                const auto& tbl0_3 = sign_table[b0_3];
                acc[0]  += a0 * tbl0_0[0];  acc[1]  += a0 * tbl0_0[1];
                acc[2]  += a0 * tbl0_0[2];  acc[3]  += a0 * tbl0_0[3];
                acc[4]  += a0 * tbl0_0[4];  acc[5]  += a0 * tbl0_0[5];
                acc[6]  += a0 * tbl0_0[6];  acc[7]  += a0 * tbl0_0[7];
                acc[8]  += a0 * tbl0_1[0];  acc[9]  += a0 * tbl0_1[1];
                acc[10] += a0 * tbl0_1[2];  acc[11] += a0 * tbl0_1[3];
                acc[12] += a0 * tbl0_1[4];  acc[13] += a0 * tbl0_1[5];
                acc[14] += a0 * tbl0_1[6];  acc[15] += a0 * tbl0_1[7];
                acc[16] += a0 * tbl0_2[0];  acc[17] += a0 * tbl0_2[1];
                acc[18] += a0 * tbl0_2[2];  acc[19] += a0 * tbl0_2[3];
                acc[20] += a0 * tbl0_2[4];  acc[21] += a0 * tbl0_2[5];
                acc[22] += a0 * tbl0_2[6];  acc[23] += a0 * tbl0_2[7];
                acc[24] += a0 * tbl0_3[0];  acc[25] += a0 * tbl0_3[1];
                acc[26] += a0 * tbl0_3[2];  acc[27] += a0 * tbl0_3[3];
                acc[28] += a0 * tbl0_3[4];  acc[29] += a0 * tbl0_3[5];
                acc[30] += a0 * tbl0_3[6];  acc[31] += a0 * tbl0_3[7];
                // ---- element p+1 ----
                float a1 = a_row[p + 1];
                uint32_t packed1 = *b_ptr; b_ptr += K_ints;
                uint8_t b1_0 = packed1 & 0xFFu;
                uint8_t b1_1 = (packed1 >> 8)  & 0xFFu;
                uint8_t b1_2 = (packed1 >> 16) & 0xFFu;
                uint8_t b1_3 = (packed1 >> 24) & 0xFFu;
                const auto& tbl1_0 = sign_table[b1_0];
                const auto& tbl1_1 = sign_table[b1_1];
                const auto& tbl1_2 = sign_table[b1_2];
                const auto& tbl1_3 = sign_table[b1_3];
                acc[0]  += a1 * tbl1_0[0];  acc[1]  += a1 * tbl1_0[1];
                acc[2]  += a1 * tbl1_0[2];  acc[3]  += a1 * tbl1_0[3];
                acc[4]  += a1 * tbl1_0[4];  acc[5]  += a1 * tbl1_0[5];
                acc[6]  += a1 * tbl1_0[6];  acc[7]  += a1 * tbl1_0[7];
                acc[8]  += a1 * tbl1_1[0];  acc[9]  += a1 * tbl1_1[1];
                acc[10] += a1 * tbl1_1[2];  acc[11] += a1 * tbl1_1[3];
                acc[12] += a1 * tbl1_1[4];  acc[13] += a1 * tbl1_1[5];
                acc[14] += a1 * tbl1_1[6];  acc[15] += a1 * tbl1_1[7];
                acc[16] += a1 * tbl1_2[0];  acc[17] += a1 * tbl1_2[1];
                acc[18] += a1 * tbl1_2[2];  acc[19] += a1 * tbl1_2[3];
                acc[20] += a1 * tbl1_2[4];  acc[21] += a1 * tbl1_2[5];
                acc[22] += a1 * tbl1_2[6];  acc[23] += a1 * tbl1_2[7];
                acc[24] += a1 * tbl1_3[0];  acc[25] += a1 * tbl1_3[1];
                acc[26] += a1 * tbl1_3[2];  acc[27] += a1 * tbl1_3[3];
                acc[28] += a1 * tbl1_3[4];  acc[29] += a1 * tbl1_3[5];
                acc[30] += a1 * tbl1_3[6];  acc[31] += a1 * tbl1_3[7];
                // ---- element p+2 ----
                float a2 = a_row[p + 2];
                uint32_t packed2 = *b_ptr; b_ptr += K_ints;
                uint8_t b2_0 = packed2 & 0xFFu;
                uint8_t b2_1 = (packed2 >> 8)  & 0xFFu;
                uint8_t b2_2 = (packed2 >> 16) & 0xFFu;
                uint8_t b2_3 = (packed2 >> 24) & 0xFFu;
                const auto& tbl2_0 = sign_table[b2_0];
                const auto& tbl2_1 = sign_table[b2_1];
                const auto& tbl2_2 = sign_table[b2_2];
                const auto& tbl2_3 = sign_table[b2_3];
                acc[0]  += a2 * tbl2_0[0];  acc[1]  += a2 * tbl2_0[1];
                acc[2]  += a2 * tbl2_0[2];  acc[3]  += a2 * tbl2_0[3];
                acc[4]  += a2 * tbl2_0[4];  acc[5]  += a2 * tbl2_0[5];
                acc[6]  += a2 * tbl2_0[6];  acc[7]  += a2 * tbl2_0[7];
                acc[8]  += a2 * tbl2_1[0];  acc[9]  += a2 * tbl2_1[1];
                acc[10] += a2 * tbl2_1[2];  acc[11] += a2 * tbl2_1[3];
                acc[12] += a2 * tbl2_1[4];  acc[13] += a2 * tbl2_1[5];
                acc[14] += a2 * tbl2_1[6];  acc[15] += a2 * tbl2_1[7];
                acc[16] += a2 * tbl2_2[0];  acc[17] += a2 * tbl2_2[1];
                acc[18] += a2 * tbl2_2[2];  acc[19] += a2 * tbl2_2[3];
                acc[20] += a2 * tbl2_2[4];  acc[21] += a2 * tbl2_2[5];
                acc[22] += a2 * tbl2_2[6];  acc[23] += a2 * tbl2_2[7];
                acc[24] += a2 * tbl2_3[0];  acc[25] += a2 * tbl2_3[1];
                acc[26] += a2 * tbl2_3[2];  acc[27] += a2 * tbl2_3[3];
                acc[28] += a2 * tbl2_3[4];  acc[29] += a2 * tbl2_3[5];
                acc[30] += a2 * tbl2_3[6];  acc[31] += a2 * tbl2_3[7];
                // ---- element p+3 ----
                float a3 = a_row[p + 3];
                uint32_t packed3 = *b_ptr; b_ptr += K_ints;
                uint8_t b3_0 = packed3 & 0xFFu;
                uint8_t b3_1 = (packed3 >> 8)  & 0xFFu;
                uint8_t b3_2 = (packed3 >> 16) & 0xFFu;
                uint8_t b3_3 = (packed3 >> 24) & 0xFFu;
                const auto& tbl3_0 = sign_table[b3_0];
                const auto& tbl3_1 = sign_table[b3_1];
                const auto& tbl3_2 = sign_table[b3_2];
                const auto& tbl3_3 = sign_table[b3_3];
                acc[0]  += a3 * tbl3_0[0];  acc[1]  += a3 * tbl3_0[1];
                acc[2]  += a3 * tbl3_0[2];  acc[3]  += a3 * tbl3_0[3];
                acc[4]  += a3 * tbl3_0[4];  acc[5]  += a3 * tbl3_0[5];
                acc[6]  += a3 * tbl3_0[6];  acc[7]  += a3 * tbl3_0[7];
                acc[8]  += a3 * tbl3_1[0];  acc[9]  += a3 * tbl3_1[1];
                acc[10] += a3 * tbl3_1[2];  acc[11] += a3 * tbl3_1[3];
                acc[12] += a3 * tbl3_1[4];  acc[13] += a3 * tbl3_1[5];
                acc[14] += a3 * tbl3_1[6];  acc[15] += a3 * tbl3_1[7];
                acc[16] += a3 * tbl3_2[0];  acc[17] += a3 * tbl3_2[1];
                acc[18] += a3 * tbl3_2[2];  acc[19] += a3 * tbl3_2[3];
                acc[20] += a3 * tbl3_2[4];  acc[21] += a3 * tbl3_2[5];
                acc[22] += a3 * tbl3_2[6];  acc[23] += a3 * tbl3_2[7];
                acc[24] += a3 * tbl3_3[0];  acc[25] += a3 * tbl3_3[1];
                acc[26] += a3 * tbl3_3[2];  acc[27] += a3 * tbl3_3[3];
                acc[28] += a3 * tbl3_3[4];  acc[29] += a3 * tbl3_3[5];
                acc[30] += a3 * tbl3_3[6];  acc[31] += a3 * tbl3_3[7];
            }
            for (int t = 0; t < 32; ++t) {
                c_row[base + t] = acc[t];
            }
        }
    }
}
