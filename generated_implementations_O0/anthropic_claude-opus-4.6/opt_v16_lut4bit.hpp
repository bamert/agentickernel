
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>
#include <string.h>

// Use a 16-entry LUT to expand 4 bits into 4 sign floats at once.
// Each entry is 4 floats (+1 or -1).
// This avoids per-bit branching entirely.

// Precompute LUT: index is 4-bit nibble value (0-15)
// Each entry is 4 floats: bit0->float0, bit1->float1, etc.
// Stored as uint32x4_t with either 0x00000000 (+1.0f sign bit = 0) 
// or 0x80000000 (-1.0f sign bit = 1) for XOR-based sign flip.

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, 
            float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // LUT: for each 4-bit nibble, store the sign-flip XOR mask
    // bit=1 means +1 (no flip = 0x00000000), bit=0 means -1 (flip = 0x80000000)
    uint32_t lut[16][4];
    for (int n = 0; n < 16; ++n) {
        lut[n][0] = (n & 1) ? 0x00000000u : 0x80000000u;
        lut[n][1] = (n & 2) ? 0x00000000u : 0x80000000u;
        lut[n][2] = (n & 4) ? 0x00000000u : 0x80000000u;
        lut[n][3] = (n & 8) ? 0x00000000u : 0x80000000u;
    }

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* __restrict__ C_row0 = C + (i + 0) * K;
        float* __restrict__ C_row1 = C + (i + 1) * K;
        float* __restrict__ C_row2 = C + (i + 2) * K;
        float* __restrict__ C_row3 = C + (i + 3) * K;
        const float* __restrict__ A_row0 = A + (i + 0) * K;
        const float* __restrict__ A_row1 = A + (i + 1) * K;
        const float* __restrict__ A_row2 = A + (i + 2) * K;
        const float* __restrict__ A_row3 = A + (i + 3) * K;

        memset(C_row0, 0, K * sizeof(float));
        memset(C_row1, 0, K * sizeof(float));
        memset(C_row2, 0, K * sizeof(float));
        memset(C_row3, 0, K * sizeof(float));

        for (size_t p = 0; p < K; ++p) {
            uint32x4_t va0u = vreinterpretq_u32_f32(vdupq_n_f32(A_row0[p]));
            uint32x4_t va1u = vreinterpretq_u32_f32(vdupq_n_f32(A_row1[p]));
            uint32x4_t va2u = vreinterpretq_u32_f32(vdupq_n_f32(A_row2[p]));
            uint32x4_t va3u = vreinterpretq_u32_f32(vdupq_n_f32(A_row3[p]));
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                size_t base = g * 32;

                // Process 8 nibbles (4 bits each = 32 bits total)
                for (int n = 0; n < 8; ++n) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    uint32x4_t flip = vld1q_u32(lut[nibble]);
                    size_t off = base + n * 4;

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
            uint32x4_t vau = vreinterpretq_u32_f32(vdupq_n_f32(A_row[p]));
            const uint32_t* B_row = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                size_t base = g * 32;

                for (int n = 0; n < 8; ++n) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    uint32x4_t flip = vld1q_u32(lut[nibble]);
                    size_t off = base + n * 4;
                    float32x4_t s = vreinterpretq_f32_u32(veorq_u32(vau, flip));
                    vst1q_f32(C_row + off, vaddq_f32(vld1q_f32(C_row + off), s));
                }
            }
        }
    }
}
