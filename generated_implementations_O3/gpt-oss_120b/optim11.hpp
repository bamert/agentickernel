#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time generated sign table: 8 floats per byte value (+1 / -1).
constexpr std::array<std::array<float,8>,256> sign_table = [](){
    std::array<std::array<float,8>,256> tbl{};
    for (size_t byte=0; byte<256; ++byte){
        for (size_t bit=0; bit<8; ++bit){
            tbl[byte][bit] = ((byte>>bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Process two rows of A (and thus C) at a time to amortise the load of B.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K / 32; // number of 32‑bit blocks per row of B
    size_t i = 0;
    for (; i+1 < M; i += 2){
        const float* a_row0 = A + i*K;
        const float* a_row1 = A + (i+1)*K;
        float* c_row0 = C + i*K;
        float* c_row1 = C + (i+1)*K;
        for (size_t block = 0; block < K_ints; ++block){
            float acc0[32] = {};
            float acc1[32] = {};
            const size_t base = block*32;
            for (size_t p = 0; p < K; ++p){
                float a0 = a_row0[p];
                float a1 = a_row1[p];
                uint32_t packed = B[p*K_ints + block];
                // split packed into four bytes
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed>>8) & 0xFFu;
                uint8_t b2 = (packed>>16) & 0xFFu;
                uint8_t b3 = (packed>>24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                // unrolled accumulation for both rows
                acc0[0]  += a0 * tbl0[0];  acc1[0]  += a1 * tbl0[0];
                acc0[1]  += a0 * tbl0[1];  acc1[1]  += a1 * tbl0[1];
                acc0[2]  += a0 * tbl0[2];  acc1[2]  += a1 * tbl0[2];
                acc0[3]  += a0 * tbl0[3];  acc1[3]  += a1 * tbl0[3];
                acc0[4]  += a0 * tbl0[4];  acc1[4]  += a1 * tbl0[4];
                acc0[5]  += a0 * tbl0[5];  acc1[5]  += a1 * tbl0[5];
                acc0[6]  += a0 * tbl0[6];  acc1[6]  += a1 * tbl0[6];
                acc0[7]  += a0 * tbl0[7];  acc1[7]  += a1 * tbl0[7];
                acc0[8]  += a0 * tbl1[0];  acc1[8]  += a1 * tbl1[0];
                acc0[9]  += a0 * tbl1[1];  acc1[9]  += a1 * tbl1[1];
                acc0[10] += a0 * tbl1[2];  acc1[10] += a1 * tbl1[2];
                acc0[11] += a0 * tbl1[3];  acc1[11] += a1 * tbl1[3];
                acc0[12] += a0 * tbl1[4];  acc1[12] += a1 * tbl1[4];
                acc0[13] += a0 * tbl1[5];  acc1[13] += a1 * tbl1[5];
                acc0[14] += a0 * tbl1[6];  acc1[14] += a1 * tbl1[6];
                acc0[15] += a0 * tbl1[7];  acc1[15] += a1 * tbl1[7];
                acc0[16] += a0 * tbl2[0];  acc1[16] += a1 * tbl2[0];
                acc0[17] += a0 * tbl2[1];  acc1[17] += a1 * tbl2[1];
                acc0[18] += a0 * tbl2[2];  acc1[18] += a1 * tbl2[2];
                acc0[19] += a0 * tbl2[3];  acc1[19] += a1 * tbl2[3];
                acc0[20] += a0 * tbl2[4];  acc1[20] += a1 * tbl2[4];
                acc0[21] += a0 * tbl2[5];  acc1[21] += a1 * tbl2[5];
                acc0[22] += a0 * tbl2[6];  acc1[22] += a1 * tbl2[6];
                acc0[23] += a0 * tbl2[7];  acc1[23] += a1 * tbl2[7];
                acc0[24] += a0 * tbl3[0];  acc1[24] += a1 * tbl3[0];
                acc0[25] += a0 * tbl3[1];  acc1[25] += a1 * tbl3[1];
                acc0[26] += a0 * tbl3[2];  acc1[26] += a1 * tbl3[2];
                acc0[27] += a0 * tbl3[3];  acc1[27] += a1 * tbl3[3];
                acc0[28] += a0 * tbl3[4];  acc1[28] += a1 * tbl3[4];
                acc0[29] += a0 * tbl3[5];  acc1[29] += a1 * tbl3[5];
                acc0[30] += a0 * tbl3[6];  acc1[30] += a1 * tbl3[6];
                acc0[31] += a0 * tbl3[7];  acc1[31] += a1 * tbl3[7];
            }
            // store back
            for (int t=0; t<32; ++t){
                c_row0[base + t] = acc0[t];
                c_row1[base + t] = acc1[t];
            }
        }
    }
    // handle possible last row if M is odd
    if (i < M){
        const float* a_row = A + i*K;
        float* c_row = C + i*K;
        for (size_t block = 0; block < K_ints; ++block){
            float acc[32] = {};
            const size_t base = block*32;
            for (size_t p = 0; p < K; ++p){
                float a_val = a_row[p];
                uint32_t packed = B[p*K_ints + block];
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed>>8) & 0xFFu;
                uint8_t b2 = (packed>>16) & 0xFFu;
                uint8_t b3 = (packed>>24) & 0xFFu;
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
            for (int t=0; t<32; ++t){
                c_row[base + t] = acc[t];
            }
        }
    }
}
