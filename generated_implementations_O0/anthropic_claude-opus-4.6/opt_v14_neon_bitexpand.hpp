
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>
#include <string.h>

// v7/v11 style but with explicit NEON for the inner loop.
// Expand uint32 bits to sign masks using NEON shifts.
// For 4 lanes at a time: shift packed right by bit positions, AND with 1,
// then use comparison to create 0x80000000 mask for XOR sign flip.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Precompute shift amounts for expanding bits
    // For bits 0-3: shift by 0,1,2,3 then AND with 1
    // For bits 4-7: shift by 4,5,6,7 then AND with 1, etc.
    const int32x4_t shifts[8] = {
        {0, -1, -2, -3},
        {-4, -5, -6, -7},
        {-8, -9, -10, -11},
        {-12, -13, -14, -15},
        {-16, -17, -18, -19},
        {-20, -21, -22, -23},
        {-24, -25, -26, -27},
        {-28, -29, -30, -31}
    };
    const uint32x4_t one_u = vdupq_n_u32(1);
    const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
    const uint32x4_t zero_u = vdupq_n_u32(0);

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

        memset(C_row0, 0, K * sizeof(float));
        memset(C_row1, 0, K * sizeof(float));
        memset(C_row2, 0, K * sizeof(float));
        memset(C_row3, 0, K * sizeof(float));

        for (size_t p = 0; p < K; ++p) {
            float32x4_t va0 = vdupq_n_f32(A_row0[p]);
            float32x4_t va1 = vdupq_n_f32(A_row1[p]);
            float32x4_t va2 = vdupq_n_f32(A_row2[p]);
            float32x4_t va3 = vdupq_n_f32(A_row3[p]);
            uint32x4_t va0u = vreinterpretq_u32_f32(va0);
            uint32x4_t va1u = vreinterpretq_u32_f32(va1);
            uint32x4_t va2u = vreinterpretq_u32_f32(va2);
            uint32x4_t va3u = vreinterpretq_u32_f32(va3);
            
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32x4_t vpacked = vdupq_n_u32(B_row[g]);
                size_t base = g * 32;

                for (int s = 0; s < 8; ++s) {
                    // Shift and extract bits
                    uint32x4_t bits = vandq_u32(vshlq_u32(vpacked, shifts[s]), one_u);
                    // bits is 0 or 1 per lane
                    // flip = sign_bit where bit=0
                    uint32x4_t flip = vandq_u32(vceqq_u32(bits, zero_u), sign_bit);
                    
                    size_t off = base + s * 4;
                    
                    float32x4_t s0 = vreinterpretq_f32_u32(veorq_u32(va0u, flip));
                    float32x4_t s1 = vreinterpretq_f32_u32(veorq_u32(va1u, flip));
                    float32x4_t s2 = vreinterpretq_f32_u32(veorq_u32(va2u, flip));
                    float32x4_t s3 = vreinterpretq_f32_u32(veorq_u32(va3u, flip));

                    vst1q_f32(C_row0 + off, vaddq_f32(vld1q_f32(C_row0 + off), s0));
                    vst1q_f32(C_row1 + off, vaddq_f32(vld1q_f32(C_row1 + off), s1));
                    vst1q_f32(C_row2 + off, vaddq_f32(vld1q_f32(C_row2 + off), s2));
                    vst1q_f32(C_row3 + off, vaddq_f32(vld1q_f32(C_row3 + off), s3));
                }
            }
        }
    }

    for (; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;
        memset(C_row, 0, K * sizeof(float));

        for (size_t p = 0; p < K; ++p) {
            float32x4_t va = vdupq_n_f32(A_row[p]);
            uint32x4_t vau = vreinterpretq_u32_f32(va);
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32x4_t vpacked = vdupq_n_u32(B_row[g]);
                size_t base = g * 32;

                for (int s = 0; s < 8; ++s) {
                    uint32x4_t bits = vandq_u32(vshlq_u32(vpacked, shifts[s]), one_u);
                    uint32x4_t flip = vandq_u32(vceqq_u32(bits, zero_u), sign_bit);
                    size_t off = base + s * 4;
                    float32x4_t sv = vreinterpretq_f32_u32(veorq_u32(vau, flip));
                    vst1q_f32(C_row + off, vaddq_f32(vld1q_f32(C_row + off), sv));
                }
            }
        }
    }
}
