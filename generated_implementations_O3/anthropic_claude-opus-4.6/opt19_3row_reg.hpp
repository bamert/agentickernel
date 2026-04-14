#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Nibble LUT
    uint32_t sign_lut[16 * 4] __attribute__((aligned(16)));
    for (int nibble = 0; nibble < 16; ++nibble) {
        for (int b = 0; b < 4; ++b) {
            int bit = (nibble >> b) & 1;
            sign_lut[nibble * 4 + b] = bit ? 0u : 0x80000000u;
        }
    }

    // Process 4 rows at a time with register accumulation
    // 4 rows * 8 accumulators = 32 registers... tight but ARM has 32 NEON regs
    // Let's try it - the masks can be reloaded from LUT per nibble
    
    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        const float* __restrict__ a0 = A + (i + 0) * K;
        const float* __restrict__ a1 = A + (i + 1) * K;
        const float* __restrict__ a2 = A + (i + 2) * K;
        const float* __restrict__ a3 = A + (i + 3) * K;
        float* __restrict__ c0 = C + (i + 0) * K;
        float* __restrict__ c1 = C + (i + 1) * K;
        float* __restrict__ c2 = C + (i + 2) * K;
        float* __restrict__ c3 = C + (i + 3) * K;

        for (size_t jb = 0; jb < K_ints; ++jb) {
            // 4 rows * 8 nibble groups = 32 accumulators
            float32x4_t r0_0 = vdupq_n_f32(0.0f), r0_1 = vdupq_n_f32(0.0f);
            float32x4_t r0_2 = vdupq_n_f32(0.0f), r0_3 = vdupq_n_f32(0.0f);
            float32x4_t r0_4 = vdupq_n_f32(0.0f), r0_5 = vdupq_n_f32(0.0f);
            float32x4_t r0_6 = vdupq_n_f32(0.0f), r0_7 = vdupq_n_f32(0.0f);
            
            float32x4_t r1_0 = vdupq_n_f32(0.0f), r1_1 = vdupq_n_f32(0.0f);
            float32x4_t r1_2 = vdupq_n_f32(0.0f), r1_3 = vdupq_n_f32(0.0f);
            float32x4_t r1_4 = vdupq_n_f32(0.0f), r1_5 = vdupq_n_f32(0.0f);
            float32x4_t r1_6 = vdupq_n_f32(0.0f), r1_7 = vdupq_n_f32(0.0f);
            
            float32x4_t r2_0 = vdupq_n_f32(0.0f), r2_1 = vdupq_n_f32(0.0f);
            float32x4_t r2_2 = vdupq_n_f32(0.0f), r2_3 = vdupq_n_f32(0.0f);
            float32x4_t r2_4 = vdupq_n_f32(0.0f), r2_5 = vdupq_n_f32(0.0f);
            float32x4_t r2_6 = vdupq_n_f32(0.0f), r2_7 = vdupq_n_f32(0.0f);
            
            float32x4_t r3_0 = vdupq_n_f32(0.0f), r3_1 = vdupq_n_f32(0.0f);
            float32x4_t r3_2 = vdupq_n_f32(0.0f), r3_3 = vdupq_n_f32(0.0f);
            float32x4_t r3_4 = vdupq_n_f32(0.0f), r3_5 = vdupq_n_f32(0.0f);
            float32x4_t r3_6 = vdupq_n_f32(0.0f), r3_7 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                uint32_t ab0, ab1, ab2, ab3;
                { float v = a0[p]; __builtin_memcpy(&ab0, &v, 4); }
                { float v = a1[p]; __builtin_memcpy(&ab1, &v, 4); }
                { float v = a2[p]; __builtin_memcpy(&ab2, &v, 4); }
                { float v = a3[p]; __builtin_memcpy(&ab3, &v, 4); }
                uint32x4_t va0 = vdupq_n_u32(ab0);
                uint32x4_t va1 = vdupq_n_u32(ab1);
                uint32x4_t va2 = vdupq_n_u32(ab2);
                uint32x4_t va3 = vdupq_n_u32(ab3);

                uint32_t packed = B[p * K_ints + jb];

                uint32x4_t m;
                
                m = vld1q_u32(&sign_lut[((packed >>  0) & 0xF) * 4]);
                r0_0 = vaddq_f32(r0_0, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_0 = vaddq_f32(r1_0, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_0 = vaddq_f32(r2_0, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_0 = vaddq_f32(r3_0, vreinterpretq_f32_u32(veorq_u32(va3, m)));

                m = vld1q_u32(&sign_lut[((packed >>  4) & 0xF) * 4]);
                r0_1 = vaddq_f32(r0_1, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_1 = vaddq_f32(r1_1, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_1 = vaddq_f32(r2_1, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_1 = vaddq_f32(r3_1, vreinterpretq_f32_u32(veorq_u32(va3, m)));

                m = vld1q_u32(&sign_lut[((packed >>  8) & 0xF) * 4]);
                r0_2 = vaddq_f32(r0_2, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_2 = vaddq_f32(r1_2, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_2 = vaddq_f32(r2_2, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_2 = vaddq_f32(r3_2, vreinterpretq_f32_u32(veorq_u32(va3, m)));

                m = vld1q_u32(&sign_lut[((packed >> 12) & 0xF) * 4]);
                r0_3 = vaddq_f32(r0_3, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_3 = vaddq_f32(r1_3, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_3 = vaddq_f32(r2_3, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_3 = vaddq_f32(r3_3, vreinterpretq_f32_u32(veorq_u32(va3, m)));

                m = vld1q_u32(&sign_lut[((packed >> 16) & 0xF) * 4]);
                r0_4 = vaddq_f32(r0_4, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_4 = vaddq_f32(r1_4, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_4 = vaddq_f32(r2_4, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_4 = vaddq_f32(r3_4, vreinterpretq_f32_u32(veorq_u32(va3, m)));

                m = vld1q_u32(&sign_lut[((packed >> 20) & 0xF) * 4]);
                r0_5 = vaddq_f32(r0_5, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_5 = vaddq_f32(r1_5, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_5 = vaddq_f32(r2_5, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_5 = vaddq_f32(r3_5, vreinterpretq_f32_u32(veorq_u32(va3, m)));

                m = vld1q_u32(&sign_lut[((packed >> 24) & 0xF) * 4]);
                r0_6 = vaddq_f32(r0_6, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_6 = vaddq_f32(r1_6, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_6 = vaddq_f32(r2_6, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_6 = vaddq_f32(r3_6, vreinterpretq_f32_u32(veorq_u32(va3, m)));

                m = vld1q_u32(&sign_lut[((packed >> 28) & 0xF) * 4]);
                r0_7 = vaddq_f32(r0_7, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_7 = vaddq_f32(r1_7, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                r2_7 = vaddq_f32(r2_7, vreinterpretq_f32_u32(veorq_u32(va2, m)));
                r3_7 = vaddq_f32(r3_7, vreinterpretq_f32_u32(veorq_u32(va3, m)));
            }

            size_t j_base = jb * 32;
            vst1q_f32(c0+j_base+ 0, r0_0); vst1q_f32(c0+j_base+ 4, r0_1);
            vst1q_f32(c0+j_base+ 8, r0_2); vst1q_f32(c0+j_base+12, r0_3);
            vst1q_f32(c0+j_base+16, r0_4); vst1q_f32(c0+j_base+20, r0_5);
            vst1q_f32(c0+j_base+24, r0_6); vst1q_f32(c0+j_base+28, r0_7);
            
            vst1q_f32(c1+j_base+ 0, r1_0); vst1q_f32(c1+j_base+ 4, r1_1);
            vst1q_f32(c1+j_base+ 8, r1_2); vst1q_f32(c1+j_base+12, r1_3);
            vst1q_f32(c1+j_base+16, r1_4); vst1q_f32(c1+j_base+20, r1_5);
            vst1q_f32(c1+j_base+24, r1_6); vst1q_f32(c1+j_base+28, r1_7);
            
            vst1q_f32(c2+j_base+ 0, r2_0); vst1q_f32(c2+j_base+ 4, r2_1);
            vst1q_f32(c2+j_base+ 8, r2_2); vst1q_f32(c2+j_base+12, r2_3);
            vst1q_f32(c2+j_base+16, r2_4); vst1q_f32(c2+j_base+20, r2_5);
            vst1q_f32(c2+j_base+24, r2_6); vst1q_f32(c2+j_base+28, r2_7);
            
            vst1q_f32(c3+j_base+ 0, r3_0); vst1q_f32(c3+j_base+ 4, r3_1);
            vst1q_f32(c3+j_base+ 8, r3_2); vst1q_f32(c3+j_base+12, r3_3);
            vst1q_f32(c3+j_base+16, r3_4); vst1q_f32(c3+j_base+20, r3_5);
            vst1q_f32(c3+j_base+24, r3_6); vst1q_f32(c3+j_base+28, r3_7);
        }
    }

    // Remaining rows (1 at a time)
    for (; i < M; ++i) {
        const float* __restrict__ a_row = A + i * K;
        float* __restrict__ c_row = C + i * K;

        for (size_t jb = 0; jb < K_ints; ++jb) {
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f), acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f), acc7 = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                uint32_t a_bits;
                float a_val = a_row[p];
                __builtin_memcpy(&a_bits, &a_val, 4);
                uint32x4_t va = vdupq_n_u32(a_bits);
                uint32_t packed = B[p * K_ints + jb];
                uint32x4_t m;

                m = vld1q_u32(&sign_lut[((packed>> 0)&0xF)*4]); acc0 = vaddq_f32(acc0, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((packed>> 4)&0xF)*4]); acc1 = vaddq_f32(acc1, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((packed>> 8)&0xF)*4]); acc2 = vaddq_f32(acc2, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((packed>>12)&0xF)*4]); acc3 = vaddq_f32(acc3, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((packed>>16)&0xF)*4]); acc4 = vaddq_f32(acc4, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((packed>>20)&0xF)*4]); acc5 = vaddq_f32(acc5, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((packed>>24)&0xF)*4]); acc6 = vaddq_f32(acc6, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((packed>>28)&0xF)*4]); acc7 = vaddq_f32(acc7, vreinterpretq_f32_u32(veorq_u32(va, m)));
            }
            size_t j_base = jb * 32;
            vst1q_f32(c_row+j_base+ 0, acc0); vst1q_f32(c_row+j_base+ 4, acc1);
            vst1q_f32(c_row+j_base+ 8, acc2); vst1q_f32(c_row+j_base+12, acc3);
            vst1q_f32(c_row+j_base+16, acc4); vst1q_f32(c_row+j_base+20, acc5);
            vst1q_f32(c_row+j_base+24, acc6); vst1q_f32(c_row+j_base+28, acc7);
        }
    }
}
