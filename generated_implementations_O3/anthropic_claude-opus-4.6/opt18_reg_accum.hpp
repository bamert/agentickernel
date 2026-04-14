#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero C
    {
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (size_t idx = 0; idx < M * K; idx += 16) {
            vst1q_f32(C + idx, zero);
            vst1q_f32(C + idx + 4, zero);
            vst1q_f32(C + idx + 8, zero);
            vst1q_f32(C + idx + 12, zero);
        }
    }

    // Nibble LUT
    uint32_t sign_lut[16 * 4] __attribute__((aligned(16)));
    for (int nibble = 0; nibble < 16; ++nibble) {
        for (int b = 0; b < 4; ++b) {
            int bit = (nibble >> b) & 1;
            sign_lut[nibble * 4 + b] = bit ? 0u : 0x80000000u;
        }
    }

    // Process 2 rows at a time, and for each jb (32 columns), 
    // accumulate all p in registers before writing to C.
    // This means for each (i_block, jb): iterate over all p, accumulate in 8 NEON regs per row.
    // 2 rows * 8 regs = 16 NEON regs for accumulators, leaving room for temporaries.
    
    // For 2 rows: 2 * 8 = 16 accumulators (128-bit each) 
    // Plus va0, va1, mask = 3 more. Total ~19, NEON has 32 registers.
    // We can try 2 rows with 32 columns (1 jb word) accumulated in registers.
    
    size_t i = 0;
    for (; i + 2 <= M; i += 2) {
        const float* __restrict__ a0 = A + (i + 0) * K;
        const float* __restrict__ a1 = A + (i + 1) * K;
        float* __restrict__ c0 = C + (i + 0) * K;
        float* __restrict__ c1 = C + (i + 1) * K;

        for (size_t jb = 0; jb < K_ints; ++jb) {
            // Accumulators for 32 columns, 2 rows
            // 8 float32x4 per row = 32 floats
            float32x4_t acc0_0 = vdupq_n_f32(0.0f);
            float32x4_t acc0_1 = vdupq_n_f32(0.0f);
            float32x4_t acc0_2 = vdupq_n_f32(0.0f);
            float32x4_t acc0_3 = vdupq_n_f32(0.0f);
            float32x4_t acc0_4 = vdupq_n_f32(0.0f);
            float32x4_t acc0_5 = vdupq_n_f32(0.0f);
            float32x4_t acc0_6 = vdupq_n_f32(0.0f);
            float32x4_t acc0_7 = vdupq_n_f32(0.0f);
            
            float32x4_t acc1_0 = vdupq_n_f32(0.0f);
            float32x4_t acc1_1 = vdupq_n_f32(0.0f);
            float32x4_t acc1_2 = vdupq_n_f32(0.0f);
            float32x4_t acc1_3 = vdupq_n_f32(0.0f);
            float32x4_t acc1_4 = vdupq_n_f32(0.0f);
            float32x4_t acc1_5 = vdupq_n_f32(0.0f);
            float32x4_t acc1_6 = vdupq_n_f32(0.0f);
            float32x4_t acc1_7 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                uint32_t ab0, ab1;
                float v0 = a0[p]; __builtin_memcpy(&ab0, &v0, 4);
                float v1 = a1[p]; __builtin_memcpy(&ab1, &v1, 4);
                uint32x4_t va0 = vdupq_n_u32(ab0);
                uint32x4_t va1 = vdupq_n_u32(ab1);

                uint32_t packed = B[p * K_ints + jb];

                uint32x4_t m0 = vld1q_u32(&sign_lut[((packed >>  0) & 0xF) * 4]);
                uint32x4_t m1 = vld1q_u32(&sign_lut[((packed >>  4) & 0xF) * 4]);
                uint32x4_t m2 = vld1q_u32(&sign_lut[((packed >>  8) & 0xF) * 4]);
                uint32x4_t m3 = vld1q_u32(&sign_lut[((packed >> 12) & 0xF) * 4]);
                uint32x4_t m4 = vld1q_u32(&sign_lut[((packed >> 16) & 0xF) * 4]);
                uint32x4_t m5 = vld1q_u32(&sign_lut[((packed >> 20) & 0xF) * 4]);
                uint32x4_t m6 = vld1q_u32(&sign_lut[((packed >> 24) & 0xF) * 4]);
                uint32x4_t m7 = vld1q_u32(&sign_lut[((packed >> 28) & 0xF) * 4]);

                acc0_0 = vaddq_f32(acc0_0, vreinterpretq_f32_u32(veorq_u32(va0, m0)));
                acc0_1 = vaddq_f32(acc0_1, vreinterpretq_f32_u32(veorq_u32(va0, m1)));
                acc0_2 = vaddq_f32(acc0_2, vreinterpretq_f32_u32(veorq_u32(va0, m2)));
                acc0_3 = vaddq_f32(acc0_3, vreinterpretq_f32_u32(veorq_u32(va0, m3)));
                acc0_4 = vaddq_f32(acc0_4, vreinterpretq_f32_u32(veorq_u32(va0, m4)));
                acc0_5 = vaddq_f32(acc0_5, vreinterpretq_f32_u32(veorq_u32(va0, m5)));
                acc0_6 = vaddq_f32(acc0_6, vreinterpretq_f32_u32(veorq_u32(va0, m6)));
                acc0_7 = vaddq_f32(acc0_7, vreinterpretq_f32_u32(veorq_u32(va0, m7)));

                acc1_0 = vaddq_f32(acc1_0, vreinterpretq_f32_u32(veorq_u32(va1, m0)));
                acc1_1 = vaddq_f32(acc1_1, vreinterpretq_f32_u32(veorq_u32(va1, m1)));
                acc1_2 = vaddq_f32(acc1_2, vreinterpretq_f32_u32(veorq_u32(va1, m2)));
                acc1_3 = vaddq_f32(acc1_3, vreinterpretq_f32_u32(veorq_u32(va1, m3)));
                acc1_4 = vaddq_f32(acc1_4, vreinterpretq_f32_u32(veorq_u32(va1, m4)));
                acc1_5 = vaddq_f32(acc1_5, vreinterpretq_f32_u32(veorq_u32(va1, m5)));
                acc1_6 = vaddq_f32(acc1_6, vreinterpretq_f32_u32(veorq_u32(va1, m6)));
                acc1_7 = vaddq_f32(acc1_7, vreinterpretq_f32_u32(veorq_u32(va1, m7)));
            }

            size_t j_base = jb * 32;
            vst1q_f32(c0 + j_base +  0, acc0_0);
            vst1q_f32(c0 + j_base +  4, acc0_1);
            vst1q_f32(c0 + j_base +  8, acc0_2);
            vst1q_f32(c0 + j_base + 12, acc0_3);
            vst1q_f32(c0 + j_base + 16, acc0_4);
            vst1q_f32(c0 + j_base + 20, acc0_5);
            vst1q_f32(c0 + j_base + 24, acc0_6);
            vst1q_f32(c0 + j_base + 28, acc0_7);
            
            vst1q_f32(c1 + j_base +  0, acc1_0);
            vst1q_f32(c1 + j_base +  4, acc1_1);
            vst1q_f32(c1 + j_base +  8, acc1_2);
            vst1q_f32(c1 + j_base + 12, acc1_3);
            vst1q_f32(c1 + j_base + 16, acc1_4);
            vst1q_f32(c1 + j_base + 20, acc1_5);
            vst1q_f32(c1 + j_base + 24, acc1_6);
            vst1q_f32(c1 + j_base + 28, acc1_7);
        }
    }

    for (; i < M; ++i) {
        const float* __restrict__ a_row = A + i * K;
        float* __restrict__ c_row = C + i * K;

        for (size_t jb = 0; jb < K_ints; ++jb) {
            float32x4_t acc_0 = vdupq_n_f32(0.0f);
            float32x4_t acc_1 = vdupq_n_f32(0.0f);
            float32x4_t acc_2 = vdupq_n_f32(0.0f);
            float32x4_t acc_3 = vdupq_n_f32(0.0f);
            float32x4_t acc_4 = vdupq_n_f32(0.0f);
            float32x4_t acc_5 = vdupq_n_f32(0.0f);
            float32x4_t acc_6 = vdupq_n_f32(0.0f);
            float32x4_t acc_7 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                uint32_t a_bits;
                float a_val = a_row[p];
                __builtin_memcpy(&a_bits, &a_val, 4);
                uint32x4_t va = vdupq_n_u32(a_bits);
                uint32_t packed = B[p * K_ints + jb];

                uint32x4_t m0 = vld1q_u32(&sign_lut[((packed >>  0) & 0xF) * 4]);
                uint32x4_t m1 = vld1q_u32(&sign_lut[((packed >>  4) & 0xF) * 4]);
                uint32x4_t m2 = vld1q_u32(&sign_lut[((packed >>  8) & 0xF) * 4]);
                uint32x4_t m3 = vld1q_u32(&sign_lut[((packed >> 12) & 0xF) * 4]);
                uint32x4_t m4 = vld1q_u32(&sign_lut[((packed >> 16) & 0xF) * 4]);
                uint32x4_t m5 = vld1q_u32(&sign_lut[((packed >> 20) & 0xF) * 4]);
                uint32x4_t m6 = vld1q_u32(&sign_lut[((packed >> 24) & 0xF) * 4]);
                uint32x4_t m7 = vld1q_u32(&sign_lut[((packed >> 28) & 0xF) * 4]);

                acc_0 = vaddq_f32(acc_0, vreinterpretq_f32_u32(veorq_u32(va, m0)));
                acc_1 = vaddq_f32(acc_1, vreinterpretq_f32_u32(veorq_u32(va, m1)));
                acc_2 = vaddq_f32(acc_2, vreinterpretq_f32_u32(veorq_u32(va, m2)));
                acc_3 = vaddq_f32(acc_3, vreinterpretq_f32_u32(veorq_u32(va, m3)));
                acc_4 = vaddq_f32(acc_4, vreinterpretq_f32_u32(veorq_u32(va, m4)));
                acc_5 = vaddq_f32(acc_5, vreinterpretq_f32_u32(veorq_u32(va, m5)));
                acc_6 = vaddq_f32(acc_6, vreinterpretq_f32_u32(veorq_u32(va, m6)));
                acc_7 = vaddq_f32(acc_7, vreinterpretq_f32_u32(veorq_u32(va, m7)));
            }

            size_t j_base = jb * 32;
            vst1q_f32(c_row + j_base +  0, acc_0);
            vst1q_f32(c_row + j_base +  4, acc_1);
            vst1q_f32(c_row + j_base +  8, acc_2);
            vst1q_f32(c_row + j_base + 12, acc_3);
            vst1q_f32(c_row + j_base + 16, acc_4);
            vst1q_f32(c_row + j_base + 20, acc_5);
            vst1q_f32(c_row + j_base + 24, acc_6);
            vst1q_f32(c_row + j_base + 28, acc_7);
        }
    }
}
