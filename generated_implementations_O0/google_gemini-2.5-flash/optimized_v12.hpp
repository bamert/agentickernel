#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Calculates Matrix C = Matrix A * Matrix B (Optimized v12 - Nibble lookup with NEON (direct vector LUT))
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Lookup table for 4-bit nibbles (16 entries, each a float32x4_t vector of 4 signs)
    const float32x4_t sign_nibble_lut_neon[16] = {
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
            float32x4_t a_vec = vmovq_n_f32(a_val); // Replicate a_val for NEON operations
            const uint32_t* B_row_ptr = B + p * K_ints; 

            // Initialize current_packed once per k_int loop for streaming
            uint32_t packed; 
            uint32_t current_packed; // Declare to make available to later for loop
            
            for (size_t k_int = 0; k_int < K_ints; ++k_int) { // For each 32-bit chunk in B's row
                packed = B_row_ptr[k_int]; // Load into local variable
                current_packed = packed; // Initialize streaming copy
                size_t c_base_idx = i * K + k_int * 32;

                // Process 8 nibbles
                float32x4_t signs_vec;
                float32x4_t c_values;
                size_t current_c_offset;

                // Nibble 0 (bits 0-3)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 0;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec); // C += A * Sign
                vst1q_f32(&C[current_c_offset], c_values);
                current_packed >>= 4;

                // Nibble 1 (bits 4-7)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 4;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec);
                vst1q_f32(&C[current_c_offset], c_values);
                current_packed >>= 4;

                // Nibble 2 (bits 8-11)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 8;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec);
                vst1q_f32(&C[current_c_offset], c_values);
                current_packed >>= 4;

                // Nibble 3 (bits 12-15)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 12;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec);
                vst1q_f32(&C[current_c_offset], c_values);
                current_packed >>= 4;

                // Nibble 4 (bits 16-19)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 16;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec);
                vst1q_f32(&C[current_c_offset], c_values);
                current_packed >>= 4;

                // Nibble 5 (bits 20-23)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 20;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec);
                vst1q_f32(&C[current_c_offset], c_values);
                current_packed >>= 4;

                // Nibble 6 (bits 24-27)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 24;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec);
                vst1q_f32(&C[current_c_offset], c_values);
                current_packed >>= 4;

                // Nibble 7 (bits 28-31)
                signs_vec = sign_nibble_lut_neon[current_packed & 0xF];
                current_c_offset = c_base_idx + 28;
                c_values = vld1q_f32(&C[current_c_offset]);
                c_values = vfmaq_f32(c_values, a_vec, signs_vec);
                vst1q_f32(&C[current_c_offset], c_values);
                // current_packed >>= 4; // Not needed
            }
        }
    }
}