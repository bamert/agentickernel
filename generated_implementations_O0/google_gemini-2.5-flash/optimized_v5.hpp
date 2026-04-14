#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) { // For each row in A
        // Initialize C row to 0
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // For each row in B (and column in A)
            float a_val = A[i * K + p];
            float32x4_t a_vec = vmovq_n_f32(a_val); // Replicate a_val for NEON operations
            const uint32_t* B_row_ptr = B + p * K_ints; 
            float32x4_t neg_one_vec = vmovq_n_f32(-1.0f);
            float32x4_t two_vec = vmovq_n_f32(2.0f);

            for (size_t k_int = 0; k_int < K_ints; ++k_int) { // For each 32-bit chunk in B's row
                uint32_t packed = B_row_ptr[k_int];
                
                // Process 4 bits at a time using NEON
                for (size_t bit_shift_base = 0; bit_shift_base < 32; bit_shift_base += 4) {
                    size_t col_start_idx = k_int * 32 + bit_shift_base;
                    
                    // Extract 4 relevant bits and put them into a uint32x4_t vector
                    uint32x4_t bits_u32 = {
                        (packed >> bit_shift_base) & 1U,
                        (packed >> (bit_shift_base + 1)) & 1U,
                        (packed >> (bit_shift_base + 2)) & 1U,
                        (packed >> (bit_shift_base + 3)) & 1U
                    };

                    float32x4_t bits_f32 = vcvtq_f32_u32(bits_u32); // Convert 0s/1s to 0.0f/1.0f
                    
                    // Calculate sign_vec: 2.0f * bits_f32 - 1.0f => results in -1.0f or 1.0f
                    float32x4_t sign_vec = vmlaq_f32(neg_one_vec, bits_f32, two_vec); // -1.0f + bits_f32 * 2.0f

                    float32x4_t c_values = vld1q_f32(&C[i * K + col_start_idx]);
                    c_values = vfmaq_f32(c_values, a_vec, sign_vec); // C += A * Sign
                    vst1q_f32(&C[i * K + col_start_idx], c_values);
                }
            }
        }
    }
}