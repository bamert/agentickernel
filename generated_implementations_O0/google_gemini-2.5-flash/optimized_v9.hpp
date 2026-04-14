#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized v9 - Nibble lookup, fully unrolled 32-bit)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    const float sign_nibble_lut[16][4] = {
        {-1.0f, -1.0f, -1.0f, -1.0f}, // 0 (0000)
        { 1.0f, -1.0f, -1.0f, -1.0f}, // 1 (0001)
        {-1.0f,  1.0f, -1.0f, -1.0f}, // 2 (0010)
        { 1.0f,  1.0f, -1.0f, -1.0f}, // 3 (0011)
        {-1.0f, -1.0f,  1.0f, -1.0f}, // 4 (0100)
        { 1.0f, -1.0f,  1.0f, -1.0f}, // 5 (0101)
        {-1.0f,  1.0f,  1.0f, -1.0f}, // 6 (0110)
        { 1.0f,  1.0f,  1.0f, -1.0f}, // 7 (0111)
        {-1.0f, -1.0f, -1.0f,  1.0f}, // 8 (1000)
        { 1.0f, -1.0f, -1.0f,  1.0f}, // 9 (1001)
        {-1.0f,  1.0f, -1.0f,  1.0f}, // 10 (1010)
        { 1.0f,  1.0f, -1.0f,  1.0f}, // 11 (1011)
        {-1.0f, -1.0f,  1.0f,  1.0f}, // 12 (1100)
        { 1.0f, -1.0f,  1.0f,  1.0f}, // 13 (1101)
        {-1.0f,  1.0f,  1.0f,  1.0f}, // 14 (1110)
        { 1.0f,  1.0f,  1.0f,  1.0f}  // 15 (1111)
    };

    for (size_t i = 0; i < M; ++i) {               // For each row in A
        for (size_t col_idx = 0; col_idx < K; ++col_idx) {           // Initialize C[i][col_idx] to 0
            C[i * K + col_idx] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // For each row in B (and column in A)
            float a_val = A[i * K + p];
            const uint32_t* B_row_ptr = B + p * K_ints; 

            for (size_t k_int = 0; k_int < K_ints; ++k_int) { // For each 32-bit chunk in B's row
                uint32_t packed = B_row_ptr[k_int];
                size_t c_base_idx = i * K + k_int * 32;

                // Fully unroll processing for all 8 nibbles (32 bits)
                // Nibble 0 (bits 0-3)
                uint32_t nibble_val_0 = (packed >> 0) & 0xF;
                const float* signs_0 = sign_nibble_lut[nibble_val_0];
                C[c_base_idx + 0] += a_val * signs_0[0];
                C[c_base_idx + 1] += a_val * signs_0[1];
                C[c_base_idx + 2] += a_val * signs_0[2];
                C[c_base_idx + 3] += a_val * signs_0[3];

                // Nibble 1 (bits 4-7)
                uint32_t nibble_val_1 = (packed >> 4) & 0xF;
                const float* signs_1 = sign_nibble_lut[nibble_val_1];
                C[c_base_idx + 4] += a_val * signs_1[0];
                C[c_base_idx + 5] += a_val * signs_1[1];
                C[c_base_idx + 6] += a_val * signs_1[2];
                C[c_base_idx + 7] += a_val * signs_1[3];

                // Nibble 2 (bits 8-11)
                uint32_t nibble_val_2 = (packed >> 8) & 0xF;
                const float* signs_2 = sign_nibble_lut[nibble_val_2];
                C[c_base_idx + 8] += a_val * signs_2[0];
                C[c_base_idx + 9] += a_val * signs_2[1];
                C[c_base_idx + 10] += a_val * signs_2[2];
                C[c_base_idx + 11] += a_val * signs_2[3];

                // Nibble 3 (bits 12-15)
                uint32_t nibble_val_3 = (packed >> 12) & 0xF;
                const float* signs_3 = sign_nibble_lut[nibble_val_3];
                C[c_base_idx + 12] += a_val * signs_3[0];
                C[c_base_idx + 13] += a_val * signs_3[1];
                C[c_base_idx + 14] += a_val * signs_3[2];
                C[c_base_idx + 15] += a_val * signs_3[3];

                // Nibble 4 (bits 16-19)
                uint32_t nibble_val_4 = (packed >> 16) & 0xF;
                const float* signs_4 = sign_nibble_lut[nibble_val_4];
                C[c_base_idx + 16] += a_val * signs_4[0];
                C[c_base_idx + 17] += a_val * signs_4[1];
                C[c_base_idx + 18] += a_val * signs_4[2];
                C[c_base_idx + 19] += a_val * signs_4[3];

                // Nibble 5 (bits 20-23)
                uint32_t nibble_val_5 = (packed >> 20) & 0xF;
                const float* signs_5 = sign_nibble_lut[nibble_val_5];
                C[c_base_idx + 20] += a_val * signs_5[0];
                C[c_base_idx + 21] += a_val * signs_5[1];
                C[c_base_idx + 22] += a_val * signs_5[2];
                C[c_base_idx + 23] += a_val * signs_5[3];

                // Nibble 6 (bits 24-27)
                uint32_t nibble_val_6 = (packed >> 24) & 0xF;
                const float* signs_6 = sign_nibble_lut[nibble_val_6];
                C[c_base_idx + 24] += a_val * signs_6[0];
                C[c_base_idx + 25] += a_val * signs_6[1];
                C[c_base_idx + 26] += a_val * signs_6[2];
                C[c_base_idx + 27] += a_val * signs_6[3];

                // Nibble 7 (bits 28-31)
                uint32_t nibble_val_7 = (packed >> 28) & 0xF;
                const float* signs_7 = sign_nibble_lut[nibble_val_7];
                C[c_base_idx + 28] += a_val * signs_7[0];
                C[c_base_idx + 29] += a_val * signs_7[1];
                C[c_base_idx + 30] += a_val * signs_7[2];
                C[c_base_idx + 31] += a_val * signs_7[3];
            }
        }
    }
}