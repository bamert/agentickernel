#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Sign table: for each byte value store 8 floats (+1 / -1).
constexpr std::array<std::array<float,8>,256> sign_table = [](){
    std::array<std::array<float,8>,256> tbl{};
    for (size_t byte=0; byte<256; ++byte){
        for (size_t bit=0; bit<8; ++bit){
            tbl[byte][bit] = ((byte>>bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Unroll inner K loop by 8 to further reduce overhead.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K / 32;
    for (size_t i=0; i<M; ++i){
        const float* a_row = A + i*K;
        float* c_row = C + i*K;
        for (size_t block=0; block<K_ints; ++block){
            float acc[32] = {};
            const size_t base = block*32;
            size_t p = 0;
            // Process 8 elements per iteration.
            for (; p+7<K; p+=8){
                // Unrolled macro for each of the 8 positions.
                // To keep code readable we expand manually.
                // ---- idx 0 ----
                float a0 = a_row[p];
                uint32_t pk0 = B[p*K_ints + block];
                uint8_t b0_0 = pk0 & 0xFFu; uint8_t b0_1 = (pk0>>8) & 0xFFu; uint8_t b0_2 = (pk0>>16) & 0xFFu; uint8_t b0_3 = (pk0>>24) & 0xFFu;
                const auto& tbl0_0 = sign_table[b0_0]; const auto& tbl0_1 = sign_table[b0_1]; const auto& tbl0_2 = sign_table[b0_2]; const auto& tbl0_3 = sign_table[b0_3];
                // ---- idx 1 ----
                float a1 = a_row[p+1];
                uint32_t pk1 = B[(p+1)*K_ints + block];
                uint8_t b1_0 = pk1 & 0xFFu; uint8_t b1_1 = (pk1>>8) & 0xFFu; uint8_t b1_2 = (pk1>>16) & 0xFFu; uint8_t b1_3 = (pk1>>24) & 0xFFu;
                const auto& tbl1_0 = sign_table[b1_0]; const auto& tbl1_1 = sign_table[b1_1]; const auto& tbl1_2 = sign_table[b1_2]; const auto& tbl1_3 = sign_table[b1_3];
                // ---- idx 2 ----
                float a2 = a_row[p+2];
                uint32_t pk2 = B[(p+2)*K_ints + block];
                uint8_t b2_0 = pk2 & 0xFFu; uint8_t b2_1 = (pk2>>8) & 0xFFu; uint8_t b2_2 = (pk2>>16) & 0xFFu; uint8_t b2_3 = (pk2>>24) & 0xFFu;
                const auto& tbl2_0 = sign_table[b2_0]; const auto& tbl2_1 = sign_table[b2_1]; const auto& tbl2_2 = sign_table[b2_2]; const auto& tbl2_3 = sign_table[b2_3];
                // ---- idx 3 ----
                float a3 = a_row[p+3];
                uint32_t pk3 = B[(p+3)*K_ints + block];
                uint8_t b3_0 = pk3 & 0xFFu; uint8_t b3_1 = (pk3>>8) & 0xFFu; uint8_t b3_2 = (pk3>>16) & 0xFFu; uint8_t b3_3 = (pk3>>24) & 0xFFu;
                const auto& tbl3_0 = sign_table[b3_0]; const auto& tbl3_1 = sign_table[b3_1]; const auto& tbl3_2 = sign_table[b3_2]; const auto& tbl3_3 = sign_table[b3_3];
                // ---- idx 4 ----
                float a4 = a_row[p+4];
                uint32_t pk4 = B[(p+4)*K_ints + block];
                uint8_t b4_0 = pk4 & 0xFFu; uint8_t b4_1 = (pk4>>8) & 0xFFu; uint8_t b4_2 = (pk4>>16) & 0xFFu; uint8_t b4_3 = (pk4>>24) & 0xFFu;
                const auto& tbl4_0 = sign_table[b4_0]; const auto& tbl4_1 = sign_table[b4_1]; const auto& tbl4_2 = sign_table[b4_2]; const auto& tbl4_3 = sign_table[b4_3];
                // ---- idx 5 ----
                float a5 = a_row[p+5];
                uint32_t pk5 = B[(p+5)*K_ints + block];
                uint8_t b5_0 = pk5 & 0xFFu; uint8_t b5_1 = (pk5>>8) & 0xFFu; uint8_t b5_2 = (pk5>>16) & 0xFFu; uint8_t b5_3 = (pk5>>24) & 0xFFu;
                const auto& tbl5_0 = sign_table[b5_0]; const auto& tbl5_1 = sign_table[b5_1]; const auto& tbl5_2 = sign_table[b5_2]; const auto& tbl5_3 = sign_table[b5_3];
                // ---- idx 6 ----
                float a6 = a_row[p+6];
                uint32_t pk6 = B[(p+6)*K_ints + block];
                uint8_t b6_0 = pk6 & 0xFFu; uint8_t b6_1 = (pk6>>8) & 0xFFu; uint8_t b6_2 = (pk6>>16) & 0xFFu; uint8_t b6_3 = (pk6>>24) & 0xFFu;
                const auto& tbl6_0 = sign_table[b6_0]; const auto& tbl6_1 = sign_table[b6_1]; const auto& tbl6_2 = sign_table[b6_2]; const auto& tbl6_3 = sign_table[b6_3];
                // ---- idx 7 ----
                float a7 = a_row[p+7];
                uint32_t pk7 = B[(p+7)*K_ints + block];
                uint8_t b7_0 = pk7 & 0xFFu; uint8_t b7_1 = (pk7>>8) & 0xFFu; uint8_t b7_2 = (pk7>>16) & 0xFFu; uint8_t b7_3 = (pk7>>24) & 0xFFu;
                const auto& tbl7_0 = sign_table[b7_0]; const auto& tbl7_1 = sign_table[b7_1]; const auto& tbl7_2 = sign_table[b7_2]; const auto& tbl7_3 = sign_table[b7_3];
                // Accumulate all 8 rows (each contributes to the same 32 columns).
                // idx0
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
                // idx1
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
                // idx2
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
                // idx3
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
                // idx4
                acc[0]  += a4 * tbl4_0[0];  acc[1]  += a4 * tbl4_0[1];
                acc[2]  += a4 * tbl4_0[2];  acc[3]  += a4 * tbl4_0[3];
                acc[4]  += a4 * tbl4_0[4];  acc[5]  += a4 * tbl4_0[5];
                acc[6]  += a4 * tbl4_0[6];  acc[7]  += a4 * tbl4_0[7];
                acc[8]  += a4 * tbl4_1[0];  acc[9]  += a4 * tbl4_1[1];
                acc[10] += a4 * tbl4_1[2];  acc[11] += a4 * tbl4_1[3];
                acc[12] += a4 * tbl4_1[4];  acc[13] += a4 * tbl4_1[5];
                acc[14] += a4 * tbl4_1[6];  acc[15] += a4 * tbl4_1[7];
                acc[16] += a4 * tbl4_2[0];  acc[17] += a4 * tbl4_2[1];
                acc[18] += a4 * tbl4_2[2];  acc[19] += a4 * tbl4_2[3];
                acc[20] += a4 * tbl4_2[4];  acc[21] += a4 * tbl4_2[5];
                acc[22] += a4 * tbl4_2[6];  acc[23] += a4 * tbl4_2[7];
                acc[24] += a4 * tbl4_3[0];  acc[25] += a4 * tbl4_3[1];
                acc[26] += a4 * tbl4_3[2];  acc[27] += a4 * tbl4_3[3];
                acc[28] += a4 * tbl4_3[4];  acc[29] += a4 * tbl4_3[5];
                acc[30] += a4 * tbl4_3[6];  acc[31] += a4 * tbl4_3[7];
                // idx5
                acc[0]  += a5 * tbl5_0[0];  acc[1]  += a5 * tbl5_0[1];
                acc[2]  += a5 * tbl5_0[2];  acc[3]  += a5 * tbl5_0[3];
                acc[4]  += a5 * tbl5_0[4];  acc[5]  += a5 * tbl5_0[5];
                acc[6]  += a5 * tbl5_0[6];  acc[7]  += a5 * tbl5_0[7];
                acc[8]  += a5 * tbl5_1[0];  acc[9]  += a5 * tbl5_1[1];
                acc[10] += a5 * tbl5_1[2];  acc[11] += a5 * tbl5_1[3];
                acc[12] += a5 * tbl5_1[4];  acc[13] += a5 * tbl5_1[5];
                acc[14] += a5 * tbl5_1[6];  acc[15] += a5 * tbl5_1[7];
                acc[16] += a5 * tbl5_2[0];  acc[17] += a5 * tbl5_2[1];
                acc[18] += a5 * tbl5_2[2];  acc[19] += a5 * tbl5_2[3];
                acc[20] += a5 * tbl5_2[4];  acc[21] += a5 * tbl5_2[5];
                acc[22] += a5 * tbl5_2[6];  acc[23] += a5 * tbl5_2[7];
                acc[24] += a5 * tbl5_3[0];  acc[25] += a5 * tbl5_3[1];
                acc[26] += a5 * tbl5_3[2];  acc[27] += a5 * tbl5_3[3];
                acc[28] += a5 * tbl5_3[4];  acc[29] += a5 * tbl5_3[5];
                acc[30] += a5 * tbl5_3[6];  acc[31] += a5 * tbl5_3[7];
                // idx6
                acc[0]  += a6 * tbl6_0[0];  acc[1]  += a6 * tbl6_0[1];
                acc[2]  += a6 * tbl6_0[2];  acc[3]  += a6 * tbl6_0[3];
                acc[4]  += a6 * tbl6_0[4];  acc[5]  += a6 * tbl6_0[5];
                acc[6]  += a6 * tbl6_0[6];  acc[7]  += a6 * tbl6_0[7];
                acc[8]  += a6 * tbl6_1[0];  acc[9]  += a6 * tbl6_1[1];
                acc[10] += a6 * tbl6_1[2];  acc[11] += a6 * tbl6_1[3];
                acc[12] += a6 * tbl6_1[4];  acc[13] += a6 * tbl6_1[5];
                acc[14] += a6 * tbl6_1[6];  acc[15] += a6 * tbl6_1[7];
                acc[16] += a6 * tbl6_2[0];  acc[17] += a6 * tbl6_2[1];
                acc[18] += a6 * tbl6_2[2];  acc[19] += a6 * tbl6_2[3];
                acc[20] += a6 * tbl6_2[4];  acc[21] += a6 * tbl6_2[5];
                acc[22] += a6 * tbl6_2[6];  acc[23] += a6 * tbl6_2[7];
                acc[24] += a6 * tbl6_3[0];  acc[25] += a6 * tbl6_3[1];
                acc[26] += a6 * tbl6_3[2];  acc[27] += a6 * tbl6_3[3];
                acc[28] += a6 * tbl6_3[4];  acc[29] += a6 * tbl6_3[5];
                acc[30] += a6 * tbl6_3[6];  acc[31] += a6 * tbl6_3[7];
                // idx7
                acc[0]  += a7 * tbl7_0[0];  acc[1]  += a7 * tbl7_0[1];
                acc[2]  += a7 * tbl7_0[2];  acc[3]  += a7 * tbl7_0[3];
                acc[4]  += a7 * tbl7_0[4];  acc[5]  += a7 * tbl7_0[5];
                acc[6]  += a7 * tbl7_0[6];  acc[7]  += a7 * tbl7_0[7];
                acc[8]  += a7 * tbl7_1[0];  acc[9]  += a7 * tbl7_1[1];
                acc[10] += a7 * tbl7_1[2];  acc[11] += a7 * tbl7_1[3];
                acc[12] += a7 * tbl7_1[4];  acc[13] += a7 * tbl7_1[5];
                acc[14] += a7 * tbl7_1[6];  acc[15] += a7 * tbl7_1[7];
                acc[16] += a7 * tbl7_2[0];  acc[17] += a7 * tbl7_2[1];
                acc[18] += a7 * tbl7_2[2];  acc[19] += a7 * tbl7_2[3];
                acc[20] += a7 * tbl7_2[4];  acc[21] += a7 * tbl7_2[5];
                acc[22] += a7 * tbl7_2[6];  acc[23] += a7 * tbl7_2[7];
                acc[24] += a7 * tbl7_3[0];  acc[25] += a7 * tbl7_3[1];
                acc[26] += a7 * tbl7_3[2];  acc[27] += a7 * tbl7_3[3];
                acc[28] += a7 * tbl7_3[4];  acc[29] += a7 * tbl7_3[5];
                acc[30] += a7 * tbl7_3[6];  acc[31] += a7 * tbl7_3[7];
            }
            // handle any leftover (K is multiple of 32, so none, but keep safety)
            for (; p<K; ++p){
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
