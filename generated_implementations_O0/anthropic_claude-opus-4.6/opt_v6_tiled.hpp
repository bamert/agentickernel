
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Build on v1's approach but:
// 1. Process multiple rows of A simultaneously (reduces B reads)
// 2. Use NEON for the add/sub operations on C

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Process 4 rows of A at a time to reuse B reads
    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* C_row0 = C + (i + 0) * K;
        float* C_row1 = C + (i + 1) * K;
        float* C_row2 = C + (i + 2) * K;
        float* C_row3 = C + (i + 3) * K;
        const float* A_row0 = A + (i + 0) * K;
        const float* A_row1 = A + (i + 1) * K;
        const float* A_row2 = A + (i + 2) * K;
        const float* A_row3 = A + (i + 3) * K;

        // Zero output rows
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(C_row0 + j, vdupq_n_f32(0.0f));
            vst1q_f32(C_row1 + j, vdupq_n_f32(0.0f));
            vst1q_f32(C_row2 + j, vdupq_n_f32(0.0f));
            vst1q_f32(C_row3 + j, vdupq_n_f32(0.0f));
        }

        for (size_t p = 0; p < K; ++p) {
            float a0 = A_row0[p];
            float a1 = A_row1[p];
            float a2 = A_row2[p];
            float a3 = A_row3[p];
            
            float32x4_t va0 = vdupq_n_f32(a0);
            float32x4_t va1 = vdupq_n_f32(a1);
            float32x4_t va2 = vdupq_n_f32(a2);
            float32x4_t va3 = vdupq_n_f32(a3);
            
            // For sign flipping via XOR
            uint32x4_t va0_bits = vreinterpretq_u32_f32(va0);
            uint32x4_t va1_bits = vreinterpretq_u32_f32(va1);
            uint32x4_t va2_bits = vreinterpretq_u32_f32(va2);
            uint32x4_t va3_bits = vreinterpretq_u32_f32(va3);
            
            const uint32_t* B_row_ptr = B + p * K_ints;
            const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row_ptr[g];
                size_t base = g * 32;

                for (int b = 0; b < 32; b += 4) {
                    // Extract 4 bits and create sign flip mask
                    uint32_t b0 = (packed >> (b + 0)) & 1u;
                    uint32_t b1 = (packed >> (b + 1)) & 1u;
                    uint32_t b2 = (packed >> (b + 2)) & 1u;
                    uint32_t b3 = (packed >> (b + 3)) & 1u;
                    
                    // flip = sign_bit where bit=0 (to negate), 0 where bit=1
                    uint32x4_t flip = {
                        b0 ? 0u : 0x80000000u,
                        b1 ? 0u : 0x80000000u,
                        b2 ? 0u : 0x80000000u,
                        b3 ? 0u : 0x80000000u
                    };

                    size_t offset = base + b;

                    float32x4_t s0 = vreinterpretq_f32_u32(veorq_u32(va0_bits, flip));
                    float32x4_t s1 = vreinterpretq_f32_u32(veorq_u32(va1_bits, flip));
                    float32x4_t s2 = vreinterpretq_f32_u32(veorq_u32(va2_bits, flip));
                    float32x4_t s3 = vreinterpretq_f32_u32(veorq_u32(va3_bits, flip));

                    vst1q_f32(C_row0 + offset, vaddq_f32(vld1q_f32(C_row0 + offset), s0));
                    vst1q_f32(C_row1 + offset, vaddq_f32(vld1q_f32(C_row1 + offset), s1));
                    vst1q_f32(C_row2 + offset, vaddq_f32(vld1q_f32(C_row2 + offset), s2));
                    vst1q_f32(C_row3 + offset, vaddq_f32(vld1q_f32(C_row3 + offset), s3));
                }
            }
        }
    }

    // Handle remaining rows
    for (; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;

        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(C_row + j, vdupq_n_f32(0.0f));
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            uint32x4_t va_bits = vreinterpretq_u32_f32(va);
            const uint32_t* B_row_ptr = B + p * K_ints;
            const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row_ptr[g];
                size_t base = g * 32;

                for (int b = 0; b < 32; b += 4) {
                    uint32_t b0 = (packed >> (b + 0)) & 1u;
                    uint32_t b1 = (packed >> (b + 1)) & 1u;
                    uint32_t b2 = (packed >> (b + 2)) & 1u;
                    uint32_t b3 = (packed >> (b + 3)) & 1u;
                    
                    uint32x4_t flip = {
                        b0 ? 0u : 0x80000000u,
                        b1 ? 0u : 0x80000000u,
                        b2 ? 0u : 0x80000000u,
                        b3 ? 0u : 0x80000000u
                    };

                    size_t offset = base + b;
                    float32x4_t s = vreinterpretq_f32_u32(veorq_u32(va_bits, flip));
                    vst1q_f32(C_row + offset, vaddq_f32(vld1q_f32(C_row + offset), s));
                }
            }
        }
    }
}
