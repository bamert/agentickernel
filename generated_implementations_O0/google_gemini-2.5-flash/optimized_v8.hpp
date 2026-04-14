#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Calculates Matrix C = Matrix A * Matrix B (Optimized v8 - Nibble lookup with NEON)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Lookup table for 4-bit nibbles (16 entries, each contains 4 signs)
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
            float32x4_t a_vec = vmovq_n_f32(a_val); // Replicate a_val for NEON operations
            const uint32_t* B_row_ptr = B + p * K_ints; 

            for (size_t k_int = 0; k_int < K_ints; ++k_int) { // For each 32-bit chunk in B's row
                uint32_t packed = B_row_ptr[k_int];
                size_t c_base_idx = i * K + k_int * 32;

                // Process 4 bits at a time (8 nibbles in 32 bits)
                for (size_t nibble_idx = 0; nibble_idx < 8; ++nibble_idx) {
                    uint32_t nibble_val = (packed >> (nibble_idx * 4)) & 0xF;
                    
                    // Load 4 signs from the lookup table into a NEON vector
                    float32x4_t signs_vec = vld1q_f32(sign_nibble_lut[nibble_val]);

                    // Load 4 C elements, perform FMA, and store back
                    size_t current_c_offset = c_base_idx + nibble_idx * 4;
                    float32x4_t c_values = vld1q_f32(&C[current_c_offset]);
                    c_values = vfmaq_f32(c_values, a_vec, signs_vec); // C += A * Sign
                    vst1q_f32(&C[current_c_offset], c_values);
                }
            }
        }
    }
}