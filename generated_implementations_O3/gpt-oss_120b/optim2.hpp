#pragma once

#include <cstddef>
#include <cstdint>

// Precomputed sign table: for each 8-bit pattern, provide an array of 8 floats (+1 or -1)
constexpr float sign_table[256][8] = [](){
    float table[256][8] = {};
    for (int byte = 0; byte < 256; ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            // Extract bit (LSB first)
            int b = (byte >> bit) & 1;
            table[byte][bit] = b ? 1.0f : -1.0f;
        }
    }
    return table;
}();

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32; // number of 32-bit blocks per row of B
    // Process each row of A
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // Process columns in blocks of 32 (4 bytes)
        for (size_t block = 0; block < K_ints; ++block) {
            // 32 accumulators for this block
            float acc[32] = {};
            // Iterate over K dimension
            for (size_t p = 0; p < K; ++p) {
                float a_val = a_row[p];
                uint32_t packed = B[p * K_ints + block];
                // treat packed as 4 bytes
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                // accumulate using precomputed signs
                const float* tbl0 = sign_table[b0];
                const float* tbl1 = sign_table[b1];
                const float* tbl2 = sign_table[b2];
                const float* tbl3 = sign_table[b3];
                // unrolled 8 adds per byte
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
            // Store the results for this block
            size_t base = block * 32;
            for (int t = 0; t < 32; ++t) {
                c_row[base + t] = acc[t];
            }
        }
    }
}
