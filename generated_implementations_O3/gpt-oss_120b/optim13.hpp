#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Sign table: for each byte value provide 8 floats (+1 / -1).
constexpr std::array<std::array<float,8>,256> sign_table = [](){
    std::array<std::array<float,8>,256> tbl{};
    for (size_t byte=0; byte<256; ++byte){
        for (size_t bit=0; bit<8; ++bit){
            tbl[byte][bit] = ((byte>>bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Process all rows together to share reads of B.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K / 32; // 32‑bit words per row of B
    // For each block of 32 columns.
    for (size_t block=0; block<K_ints; ++block){
        // allocate accumulator for every output row in this block
        float acc[32][32] = {}; // acc[row][col]
        const size_t base = block*32;
        // iterate over K dimension
        for (size_t p=0; p<K; ++p){
            uint32_t packed = B[p*K_ints + block];
            // split into four bytes
            uint8_t b0 = packed & 0xFFu;
            uint8_t b1 = (packed>>8) & 0xFFu;
            uint8_t b2 = (packed>>16) & 0xFFu;
            uint8_t b3 = (packed>>24) & 0xFFu;
            const auto& tbl0 = sign_table[b0];
            const auto& tbl1 = sign_table[b1];
            const auto& tbl2 = sign_table[b2];
            const auto& tbl3 = sign_table[b3];
            // for each row, accumulate
            for (size_t i=0; i<M; ++i){
                float a_val = A[i*K + p];
                // unrolled 32 columns
                acc[i][0]  += a_val * tbl0[0];  acc[i][1]  += a_val * tbl0[1];
                acc[i][2]  += a_val * tbl0[2];  acc[i][3]  += a_val * tbl0[3];
                acc[i][4]  += a_val * tbl0[4];  acc[i][5]  += a_val * tbl0[5];
                acc[i][6]  += a_val * tbl0[6];  acc[i][7]  += a_val * tbl0[7];
                acc[i][8]  += a_val * tbl1[0];  acc[i][9]  += a_val * tbl1[1];
                acc[i][10] += a_val * tbl1[2];  acc[i][11] += a_val * tbl1[3];
                acc[i][12] += a_val * tbl1[4];  acc[i][13] += a_val * tbl1[5];
                acc[i][14] += a_val * tbl1[6];  acc[i][15] += a_val * tbl1[7];
                acc[i][16] += a_val * tbl2[0];  acc[i][17] += a_val * tbl2[1];
                acc[i][18] += a_val * tbl2[2];  acc[i][19] += a_val * tbl2[3];
                acc[i][20] += a_val * tbl2[4];  acc[i][21] += a_val * tbl2[5];
                acc[i][22] += a_val * tbl2[6];  acc[i][23] += a_val * tbl2[7];
                acc[i][24] += a_val * tbl3[0];  acc[i][25] += a_val * tbl3[1];
                acc[i][26] += a_val * tbl3[2];  acc[i][27] += a_val * tbl3[3];
                acc[i][28] += a_val * tbl3[4];  acc[i][29] += a_val * tbl3[5];
                acc[i][30] += a_val * tbl3[6];  acc[i][31] += a_val * tbl3[7];
            }
        }
        // write results back to C
        for (size_t i=0; i<M; ++i){
            float* c_row = C + i*K + base;
            for (int t=0; t<32; ++t){
                c_row[t] = acc[i][t];
            }
        }
    }
}
