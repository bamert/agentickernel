
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;

        // Zero output row using NEON
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(C_row + j, vdupq_n_f32(0.0f));
        }

        // For each element in the shared dimension p
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            const uint32_t* B_row = B + p * K_ints;

            // Process B[p] row in groups of 32 columns
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* C_out = C_row + g * 32;

                // Process 32 bits = 8 groups of 4 bits using NEON
                // For each group of 4 bits, create a sign vector and do fma
                for (int b = 0; b < 32; b += 4) {
                    uint32_t bits4 = (packed >> b) & 0xF;
                    
                    // Create sign mask: bit=1 -> 0x00000000 (positive), bit=0 -> 0x80000000 (negative)
                    // We want: if bit is 0, flip sign of a_val
                    // sign_bits: for each of 4 lanes, 0x80000000 if bit=0, 0x00000000 if bit=1
                    uint32_t s0 = (bits4 & 1) ? 0 : 0x80000000u;
                    uint32_t s1 = (bits4 & 2) ? 0 : 0x80000000u;
                    uint32_t s2 = (bits4 & 4) ? 0 : 0x80000000u;
                    uint32_t s3 = (bits4 & 8) ? 0 : 0x80000000u;

                    uint32x4_t sign_mask = {s0, s1, s2, s3};
                    
                    // XOR with a_val to flip sign where needed
                    float32x4_t signed_a = vreinterpretq_f32_u32(
                        veorq_u32(vreinterpretq_u32_f32(va), sign_mask)
                    );

                    // Load current C values and accumulate
                    float32x4_t c_vec = vld1q_f32(C_out + b);
                    c_vec = vaddq_f32(c_vec, signed_a);
                    vst1q_f32(C_out + b, c_vec);
                }
            }
        }
    }
}
