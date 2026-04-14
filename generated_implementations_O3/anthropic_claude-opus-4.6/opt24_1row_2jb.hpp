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

    for (size_t i = 0; i < M; ++i) {
        const float* __restrict__ a_row = A + i * K;
        float* __restrict__ c_row = C + i * K;

        // Process 2 jb columns at a time (64 output columns)
        size_t jb = 0;
        for (; jb + 2 <= K_ints; jb += 2) {
            // 16 accumulators for 64 columns
            float32x4_t r0 = vdupq_n_f32(0.0f), r1 = vdupq_n_f32(0.0f);
            float32x4_t r2 = vdupq_n_f32(0.0f), r3 = vdupq_n_f32(0.0f);
            float32x4_t r4 = vdupq_n_f32(0.0f), r5 = vdupq_n_f32(0.0f);
            float32x4_t r6 = vdupq_n_f32(0.0f), r7 = vdupq_n_f32(0.0f);
            float32x4_t r8 = vdupq_n_f32(0.0f), r9 = vdupq_n_f32(0.0f);
            float32x4_t rA = vdupq_n_f32(0.0f), rB = vdupq_n_f32(0.0f);
            float32x4_t rC = vdupq_n_f32(0.0f), rD = vdupq_n_f32(0.0f);
            float32x4_t rE = vdupq_n_f32(0.0f), rF = vdupq_n_f32(0.0f);

            for (size_t p = 0; p < K; ++p) {
                uint32_t ab;
                float v = a_row[p];
                __builtin_memcpy(&ab, &v, 4);
                uint32x4_t va = vdupq_n_u32(ab);

                const uint32_t* b_ptr = B + p * K_ints + jb;
                uint32_t p0 = b_ptr[0];
                uint32_t p1 = b_ptr[1];

                uint32x4_t m;
                
                // First 32 columns
                m = vld1q_u32(&sign_lut[((p0 >>  0) & 0xF) * 4]); r0 = vaddq_f32(r0, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p0 >>  4) & 0xF) * 4]); r1 = vaddq_f32(r1, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p0 >>  8) & 0xF) * 4]); r2 = vaddq_f32(r2, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p0 >> 12) & 0xF) * 4]); r3 = vaddq_f32(r3, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p0 >> 16) & 0xF) * 4]); r4 = vaddq_f32(r4, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p0 >> 20) & 0xF) * 4]); r5 = vaddq_f32(r5, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p0 >> 24) & 0xF) * 4]); r6 = vaddq_f32(r6, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p0 >> 28) & 0xF) * 4]); r7 = vaddq_f32(r7, vreinterpretq_f32_u32(veorq_u32(va, m)));
                
                // Second 32 columns
                m = vld1q_u32(&sign_lut[((p1 >>  0) & 0xF) * 4]); r8 = vaddq_f32(r8, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p1 >>  4) & 0xF) * 4]); r9 = vaddq_f32(r9, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p1 >>  8) & 0xF) * 4]); rA = vaddq_f32(rA, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p1 >> 12) & 0xF) * 4]); rB = vaddq_f32(rB, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p1 >> 16) & 0xF) * 4]); rC = vaddq_f32(rC, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p1 >> 20) & 0xF) * 4]); rD = vaddq_f32(rD, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p1 >> 24) & 0xF) * 4]); rE = vaddq_f32(rE, vreinterpretq_f32_u32(veorq_u32(va, m)));
                m = vld1q_u32(&sign_lut[((p1 >> 28) & 0xF) * 4]); rF = vaddq_f32(rF, vreinterpretq_f32_u32(veorq_u32(va, m)));
            }

            size_t j0 = jb * 32;
            size_t j1 = j0 + 32;
            vst1q_f32(c_row+j0+ 0, r0); vst1q_f32(c_row+j0+ 4, r1);
            vst1q_f32(c_row+j0+ 8, r2); vst1q_f32(c_row+j0+12, r3);
            vst1q_f32(c_row+j0+16, r4); vst1q_f32(c_row+j0+20, r5);
            vst1q_f32(c_row+j0+24, r6); vst1q_f32(c_row+j0+28, r7);
            vst1q_f32(c_row+j1+ 0, r8); vst1q_f32(c_row+j1+ 4, r9);
            vst1q_f32(c_row+j1+ 8, rA); vst1q_f32(c_row+j1+12, rB);
            vst1q_f32(c_row+j1+16, rC); vst1q_f32(c_row+j1+20, rD);
            vst1q_f32(c_row+j1+24, rE); vst1q_f32(c_row+j1+28, rF);
        }

        // Handle remaining jb
        for (; jb < K_ints; ++jb) {
            float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0),a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
            float32x4_t a4=vdupq_n_f32(0),a5=vdupq_n_f32(0),a6=vdupq_n_f32(0),a7=vdupq_n_f32(0);
            for (size_t p = 0; p < K; ++p) {
                uint32_t ab; float v = a_row[p]; __builtin_memcpy(&ab, &v, 4);
                uint32x4_t va = vdupq_n_u32(ab);
                uint32_t packed = B[p * K_ints + jb]; uint32x4_t m;
                m=vld1q_u32(&sign_lut[((packed>>0)&0xF)*4]); a0=vaddq_f32(a0,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vld1q_u32(&sign_lut[((packed>>4)&0xF)*4]); a1=vaddq_f32(a1,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vld1q_u32(&sign_lut[((packed>>8)&0xF)*4]); a2=vaddq_f32(a2,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vld1q_u32(&sign_lut[((packed>>12)&0xF)*4]); a3=vaddq_f32(a3,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vld1q_u32(&sign_lut[((packed>>16)&0xF)*4]); a4=vaddq_f32(a4,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vld1q_u32(&sign_lut[((packed>>20)&0xF)*4]); a5=vaddq_f32(a5,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vld1q_u32(&sign_lut[((packed>>24)&0xF)*4]); a6=vaddq_f32(a6,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vld1q_u32(&sign_lut[((packed>>28)&0xF)*4]); a7=vaddq_f32(a7,vreinterpretq_f32_u32(veorq_u32(va,m)));
            }
            size_t j_base = jb*32;
            vst1q_f32(c_row+j_base+0,a0); vst1q_f32(c_row+j_base+4,a1);
            vst1q_f32(c_row+j_base+8,a2); vst1q_f32(c_row+j_base+12,a3);
            vst1q_f32(c_row+j_base+16,a4); vst1q_f32(c_row+j_base+20,a5);
            vst1q_f32(c_row+j_base+24,a6); vst1q_f32(c_row+j_base+28,a7);
        }
    }
}
