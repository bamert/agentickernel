#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time sign table: for each byte value store 8 floats (+1.0f or -1.0f).
constexpr std::array<std::array<float, 8>, 256> sign_table = [](){
    std::array<std::array<float, 8>, 256> tbl{};
    for (size_t byte = 0; byte < 256; ++byte) {
        for (size_t bit = 0; bit < 8; ++bit) {
            tbl[byte][bit] = ((byte >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Matrix multiply with unrolled inner K loop (step of 2).
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of 32‑bit words per row of B
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block) {
            float acc[32] = {};
            const size_t base = block * 32;
            size_t p = 0;
            // process two elements per iteration
            for (; p + 1 < K; p += 2) {
                // first element
                float a0 = a_row[p];
                uint32_t packed0 = B[p * K_ints + block];
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
                // second element
                float a1 = a_row[p + 1];
                uint32_t packed1 = B[(p + 1) * K_ints + block];
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
            }
            // handle possible leftover when K is odd (not the case for multiples of 32, but kept for safety)
            for (; p < K; ++p) {
                float a_val = a_row[p];
                uint32_t packed = B[p * K_ints + block];
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8)  & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                acc[0]  += a_val * tbl0[0];  acc[1]  += a_val * tbl0[1];
                acc[2]  += a_val * tbl0[2];  acc[3]  += a_val * tbl0[3];
                acc[4]  += a_val * tbl0[4];  acc[5]  += a_val * tbl0[5];
                acc[6]  += a_val * tbl0[6];  acc[7]  += a_val * tbl0[7];
                acc[8]  += a_val * tbl1[0];  acc[9]  += a_val * tbl1[1];
                acc[10] += a_val * tbl1[2];  acc[11] += a_val * tbl1[3];
                acc[12] += a_val * tbl1[4];  acc[13] += a_val * tbl1[5];
                acc[14] += a_val * tbl1[6];  acc[15] += a_val * tbl1[7];
                acc[16] += a_val * tbl2[0];  acc[17] += a_val * tbl2[1];
                acc[18] += a_val * tbl2[2];  acc[19] += a_val * tbl2[3];
                acc[20] += a_val * tbl2[4];  acc[21] += a_val * tbl2[5];
                acc[22] += a_val * tbl2[6];  acc[23] += a_val * tbl2[7];
                acc[24] += a_val * tbl3[0];  acc[25] += a_val * tbl3[1];
                acc[26] += a_val * tbl3[2];  acc[27] += a_val * tbl3[3];
                acc[28] += a_val * tbl3[4];  acc[29] += a_val * tbl3[5];
                acc[30] += a_val * tbl3[6];  acc[31] += a_val * tbl3[7];
            }
            // store the block result
            for (int t = 0; t < 32; ++t) {
                c_row[base + t] = acc[t];
            }
        }
    }
}
