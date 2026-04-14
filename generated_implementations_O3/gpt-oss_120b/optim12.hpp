#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time sign table: 8 floats (+1/-1) for each possible byte value.
constexpr std::array<std::array<float,8>,256> sign_table = [](){
    std::array<std::array<float,8>,256> tbl{};
    for (size_t byte=0; byte<256; ++byte){
        for (size_t bit=0; bit<8; ++bit){
            tbl[byte][bit] = ((byte>>bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Unrolled inner loop over p (step of 2) to reduce loop overhead.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K / 32; // 32‑bit words per B row
    for (size_t i=0; i<M; ++i){
        const float* a_row = A + i*K;
        float* c_row = C + i*K;
        for (size_t block=0; block<K_ints; ++block){
            float acc[32] = {};
            const size_t base = block*32;
            size_t p = 0;
            for (; p+1 < K; p+=2){
                // first element
                float a0 = a_row[p];
                uint32_t packed0 = B[p*K_ints + block];
                uint8_t b00 = packed0 & 0xFFu;
                uint8_t b01 = (packed0>>8) & 0xFFu;
                uint8_t b02 = (packed0>>16) & 0xFFu;
                uint8_t b03 = (packed0>>24) & 0xFFu;
                const auto& tbl00 = sign_table[b00];
                const auto& tbl01 = sign_table[b01];
                const auto& tbl02 = sign_table[b02];
                const auto& tbl03 = sign_table[b03];
                acc[0]  += a0 * tbl00[0];  acc[1]  += a0 * tbl00[1];
                acc[2]  += a0 * tbl00[2];  acc[3]  += a0 * tbl00[3];
                acc[4]  += a0 * tbl00[4];  acc[5]  += a0 * tbl00[5];
                acc[6]  += a0 * tbl00[6];  acc[7]  += a0 * tbl00[7];
                acc[8]  += a0 * tbl01[0];  acc[9]  += a0 * tbl01[1];
                acc[10] += a0 * tbl01[2];  acc[11] += a0 * tbl01[3];
                acc[12] += a0 * tbl01[4];  acc[13] += a0 * tbl01[5];
                acc[14] += a0 * tbl01[6];  acc[15] += a0 * tbl01[7];
                acc[16] += a0 * tbl02[0];  acc[17] += a0 * tbl02[1];
                acc[18] += a0 * tbl02[2];  acc[19] += a0 * tbl02[3];
                acc[20] += a0 * tbl02[4];  acc[21] += a0 * tbl02[5];
                acc[22] += a0 * tbl02[6];  acc[23] += a0 * tbl02[7];
                acc[24] += a0 * tbl03[0];  acc[25] += a0 * tbl03[1];
                acc[26] += a0 * tbl03[2];  acc[27] += a0 * tbl03[3];
                acc[28] += a0 * tbl03[4];  acc[29] += a0 * tbl03[5];
                acc[30] += a0 * tbl03[6];  acc[31] += a0 * tbl03[7];
                // second element (p+1)
                float a1 = a_row[p+1];
                uint32_t packed1 = B[(p+1)*K_ints + block];
                uint8_t b10 = packed1 & 0xFFu;
                uint8_t b11 = (packed1>>8) & 0xFFu;
                uint8_t b12 = (packed1>>16) & 0xFFu;
                uint8_t b13 = (packed1>>24) & 0xFFu;
                const auto& tbl10 = sign_table[b10];
                const auto& tbl11 = sign_table[b11];
                const auto& tbl12 = sign_table[b12];
                const auto& tbl13 = sign_table[b13];
                acc[0]  += a1 * tbl10[0];  acc[1]  += a1 * tbl10[1];
                acc[2]  += a1 * tbl10[2];  acc[3]  += a1 * tbl10[3];
                acc[4]  += a1 * tbl10[4];  acc[5]  += a1 * tbl10[5];
                acc[6]  += a1 * tbl10[6];  acc[7]  += a1 * tbl10[7];
                acc[8]  += a1 * tbl11[0];  acc[9]  += a1 * tbl11[1];
                acc[10] += a1 * tbl11[2];  acc[11] += a1 * tbl11[3];
                acc[12] += a1 * tbl11[4];  acc[13] += a1 * tbl11[5];
                acc[14] += a1 * tbl11[6];  acc[15] += a1 * tbl11[7];
                acc[16] += a1 * tbl12[0];  acc[17] += a1 * tbl12[1];
                acc[18] += a1 * tbl12[2];  acc[19] += a1 * tbl12[3];
                acc[20] += a1 * tbl12[4];  acc[21] += a1 * tbl12[5];
                acc[22] += a1 * tbl12[6];  acc[23] += a1 * tbl12[7];
                acc[24] += a1 * tbl13[0];  acc[25] += a1 * tbl13[1];
                acc[26] += a1 * tbl13[2];  acc[27] += a1 * tbl13[3];
                acc[28] += a1 * tbl13[4];  acc[29] += a1 * tbl13[5];
                acc[30] += a1 * tbl13[6];  acc[31] += a1 * tbl13[7];
            }
            // handle possible leftover element when K is odd (K is multiple of 32, so even, but keep safety)
            if (p < K){
                float a0 = a_row[p];
                uint32_t packed0 = B[p*K_ints + block];
                uint8_t b00 = packed0 & 0xFFu;
                uint8_t b01 = (packed0>>8) & 0xFFu;
                uint8_t b02 = (packed0>>16) & 0xFFu;
                uint8_t b03 = (packed0>>24) & 0xFFu;
                const auto& tbl00 = sign_table[b00];
                const auto& tbl01 = sign_table[b01];
                const auto& tbl02 = sign_table[b02];
                const auto& tbl03 = sign_table[b03];
                acc[0]  += a0 * tbl00[0];  acc[1]  += a0 * tbl00[1];
                acc[2]  += a0 * tbl00[2];  acc[3]  += a0 * tbl00[3];
                acc[4]  += a0 * tbl00[4];  acc[5]  += a0 * tbl00[5];
                acc[6]  += a0 * tbl00[6];  acc[7]  += a0 * tbl00[7];
                acc[8]  += a0 * tbl01[0];  acc[9]  += a0 * tbl01[1];
                acc[10] += a0 * tbl01[2];  acc[11] += a0 * tbl01[3];
                acc[12] += a0 * tbl01[4];  acc[13] += a0 * tbl01[5];
                acc[14] += a0 * tbl01[6];  acc[15] += a0 * tbl01[7];
                acc[16] += a0 * tbl02[0];  acc[17] += a0 * tbl02[1];
                acc[18] += a0 * tbl02[2];  acc[19] += a0 * tbl02[3];
                acc[20] += a0 * tbl02[4];  acc[21] += a0 * tbl02[5];
                acc[22] += a0 * tbl02[6];  acc[23] += a0 * tbl02[7];
                acc[24] += a0 * tbl03[0];  acc[25] += a0 * tbl03[1];
                acc[26] += a0 * tbl03[2];  acc[27] += a0 * tbl03[3];
                acc[28] += a0 * tbl03[4];  acc[29] += a0 * tbl03[5];
                acc[30] += a0 * tbl03[6];  acc[31] += a0 * tbl03[7];
            }
            // store block results
            for (int t=0; t<32; ++t){
                c_row[base + t] = acc[t];
            }
        }
    }
}
