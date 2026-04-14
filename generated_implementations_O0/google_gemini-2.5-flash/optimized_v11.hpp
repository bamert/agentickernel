#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h> // Though not directly used in core computation

// Calculates Matrix C = Matrix A * Matrix B (Optimized v11 - Nibble lookup, full unroll and streaming packed bits)
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

    float32x4_t zero_vec = vmovq_n_f32(0.0f); // NEON vector of zeros

    for (size_t i = 0; i < M; ++i) {               // For each row in A
        // Initialize C[i] row to 0 using NEON
        for (size_t col_idx = 0; col_idx < K; col_idx += 4) {
            vst1q_f32(&C[i * K + col_idx], zero_vec);
        }

        for (size_t p = 0; p < K; ++p) { // For each row in B (and column in A)
            float a_val = A[i * K + p];
            const uint32_t* B_row_ptr = B + p * K_ints; 

            for (size_t k_int = 0; k_int < K_ints; ++k_int) { // For each 32-bit chunk in B's row
                uint32_t packed = B_row_ptr[k_int];
                size_t c_base_idx = i * K + k_int * 32;

                // Fully unroll processing for all 8 nibbles (32 bits)
                // Use a 'streaming' approach for packed value
                uint32_t current_packed = packed;
                const float* signs;

                // Nibble 0 (bits 0-3)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 0] += a_val * signs[0];
                C[c_base_idx + 1] += a_val * signs[1];
                C[c_base_idx + 2] += a_val * signs[2];
                C[c_base_idx + 3] += a_val * signs[3];
                current_packed >>= 4;

                // Nibble 1 (bits 4-7)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 4] += a_val * signs[0];
                C[c_base_idx + 5] += a_val * signs[1];
                C[c_base_idx + 6] += a_val * signs[2];
                C[c_base_idx + 7] += a_val * signs[3];
                current_packed >>= 4;

                // Nibble 2 (bits 8-11)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 8] += a_val * signs[0];
                C[c_base_idx + 9] += a_val * signs[1];
                C[c_base_idx + 10] += a_val * signs[2];
                C[c_base_idx + 11] += a_val * signs[3];
                current_packed >>= 4;

                // Nibble 3 (bits 12-15)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 12] += a_val * signs[0];
                C[c_base_idx + 13] += a_val * signs[1];
                C[c_base_idx + 14] += a_val * signs[2];
                C[c_base_idx + 15] += a_val * signs[3];
                current_packed >>= 4;

                // Nibble 4 (bits 16-19)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 16] += a_val * signs[0];
                C[c_base_idx + 17] += a_val * signs[1];
                C[c_base_idx + 18] += a_val * signs[2];
                C[c_base_idx + 19] += a_val * signs[3];
                current_packed >>= 4;

                // Nibble 5 (bits 20-23)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 20] += a_val * signs[0];
                C[c_base_idx + 21] += a_val * signs[1];
                C[c_base_idx + 22] += a_val * signs[2];
                C[c_base_idx + 23] += a_val * signs[3];
                current_packed >>= 4;

                // Nibble 6 (bits 24-27)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 24] += a_val * signs[0];
                C[c_base_idx + 25] += a_val * signs[1];
                C[c_base_idx + 26] += a_val * signs[2];
                C[c_base_idx + 27] += a_val * signs[3];
                current_packed >>= 4;

                // Nibble 7 (bits 28-31)
                signs = sign_nibble_lut[current_packed & 0xF];
                C[c_base_idx + 28] += a_val * signs[0];
                C[c_base_idx + 29] += a_val * signs[1];
                C[c_base_idx + 30] += a_val * signs[2];
                C[c_base_idx + 31] += a_val * signs[3];
                // current_packed >>= 4; // Not strictly needed for the last one
            }
        }
    }
}