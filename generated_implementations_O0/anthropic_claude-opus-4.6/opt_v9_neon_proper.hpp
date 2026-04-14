
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Use NEON to expand bits to sign values efficiently.
// For each uint32, we expand to 32 floats (+1/-1) and do 4 FMAs per row.
// Key: use shift + AND + compare to expand bits, process 4 at a time.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    const uint32x4_t one_u = vdupq_n_u32(1);
    const float32x4_t one_f = vdupq_n_f32(1.0f);
    const float32x4_t two_f = vdupq_n_f32(2.0f);

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

        // Zero C rows
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(C_row0 + j, vdupq_n_f32(0.0f));
            vst1q_f32(C_row1 + j, vdupq_n_f32(0.0f));
            vst1q_f32(C_row2 + j, vdupq_n_f32(0.0f));
            vst1q_f32(C_row3 + j, vdupq_n_f32(0.0f));
        }

        for (size_t p = 0; p < K; ++p) {
            float32x4_t va0 = vdupq_n_f32(A_row0[p]);
            float32x4_t va1 = vdupq_n_f32(A_row1[p]);
            float32x4_t va2 = vdupq_n_f32(A_row2[p]);
            float32x4_t va3 = vdupq_n_f32(A_row3[p]);
            
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                size_t base = g * 32;

                // Process 4 bits at a time, 8 iterations for 32 bits
                for (int b = 0; b < 32; b += 4) {
                    // Extract 4 consecutive bits as individual uint32 values
                    // Shift right by b, then extract bits 0,1,2,3
                    uint32_t shifted = packed >> b;
                    // Convert bits to float signs: bit -> 2*bit - 1 = {-1, +1}
                    // Using NEON: extract bit, convert to float, multiply by 2, subtract 1
                    uint32x4_t bits = {shifted & 1u, (shifted >> 1) & 1u, (shifted >> 2) & 1u, (shifted >> 3) & 1u};
                    // sign = 2.0f * float(bit) - 1.0f
                    float32x4_t sign = vsubq_f32(vmulq_f32(two_f, vcvtq_f32_u32(bits)), one_f);
                    
                    size_t off = base + b;
                    vst1q_f32(C_row0 + off, vfmaq_f32(vld1q_f32(C_row0 + off), va0, sign));
                    vst1q_f32(C_row1 + off, vfmaq_f32(vld1q_f32(C_row1 + off), va1, sign));
                    vst1q_f32(C_row2 + off, vfmaq_f32(vld1q_f32(C_row2 + off), va2, sign));
                    vst1q_f32(C_row3 + off, vfmaq_f32(vld1q_f32(C_row3 + off), va3, sign));
                }
            }
        }
    }

    // Remaining rows
    for (; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;

        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(C_row + j, vdupq_n_f32(0.0f));
        }

        for (size_t p = 0; p < K; ++p) {
            float32x4_t va = vdupq_n_f32(A_row[p]);
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                size_t base = g * 32;

                for (int b = 0; b < 32; b += 4) {
                    uint32_t shifted = packed >> b;
                    uint32x4_t bits = {shifted & 1u, (shifted >> 1) & 1u, (shifted >> 2) & 1u, (shifted >> 3) & 1u};
                    float32x4_t sign = vsubq_f32(vmulq_f32(two_f, vcvtq_f32_u32(bits)), one_f);
                    
                    size_t off = base + b;
                    vst1q_f32(C_row + off, vfmaq_f32(vld1q_f32(C_row + off), va, sign));
                }
            }
        }
    }
}
