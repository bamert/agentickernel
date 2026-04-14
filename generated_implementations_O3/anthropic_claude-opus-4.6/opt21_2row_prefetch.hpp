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

    size_t i = 0;
    for (; i + 2 <= M; i += 2) {
        const float* __restrict__ a0 = A + (i + 0) * K;
        const float* __restrict__ a1 = A + (i + 1) * K;
        float* __restrict__ c0 = C + (i + 0) * K;
        float* __restrict__ c1 = C + (i + 1) * K;

        for (size_t jb = 0; jb < K_ints; ++jb) {
            float32x4_t r0_0 = vdupq_n_f32(0.0f), r0_1 = vdupq_n_f32(0.0f);
            float32x4_t r0_2 = vdupq_n_f32(0.0f), r0_3 = vdupq_n_f32(0.0f);
            float32x4_t r0_4 = vdupq_n_f32(0.0f), r0_5 = vdupq_n_f32(0.0f);
            float32x4_t r0_6 = vdupq_n_f32(0.0f), r0_7 = vdupq_n_f32(0.0f);
            
            float32x4_t r1_0 = vdupq_n_f32(0.0f), r1_1 = vdupq_n_f32(0.0f);
            float32x4_t r1_2 = vdupq_n_f32(0.0f), r1_3 = vdupq_n_f32(0.0f);
            float32x4_t r1_4 = vdupq_n_f32(0.0f), r1_5 = vdupq_n_f32(0.0f);
            float32x4_t r1_6 = vdupq_n_f32(0.0f), r1_7 = vdupq_n_f32(0.0f);

            // Unroll p by 2 to allow pipelining
            size_t p = 0;
            for (; p + 1 < K; p += 2) {
                // Prefetch next B values
                __builtin_prefetch(&B[(p + 8) * K_ints + jb], 0, 3);
                
                // Iteration 1
                uint32_t ab0_0, ab1_0;
                { float v = a0[p]; __builtin_memcpy(&ab0_0, &v, 4); }
                { float v = a1[p]; __builtin_memcpy(&ab1_0, &v, 4); }
                uint32x4_t va0_0 = vdupq_n_u32(ab0_0);
                uint32x4_t va1_0 = vdupq_n_u32(ab1_0);
                uint32_t packed0 = B[p * K_ints + jb];
                
                // Iteration 2
                uint32_t ab0_1, ab1_1;
                { float v = a0[p+1]; __builtin_memcpy(&ab0_1, &v, 4); }
                { float v = a1[p+1]; __builtin_memcpy(&ab1_1, &v, 4); }
                uint32x4_t va0_1 = vdupq_n_u32(ab0_1);
                uint32x4_t va1_1 = vdupq_n_u32(ab1_1);
                uint32_t packed1 = B[(p+1) * K_ints + jb];

                uint32x4_t m;
                
                #define PROCESS_NIB(idx, shift) \
                    m = vld1q_u32(&sign_lut[((packed0 >> shift) & 0xF) * 4]); \
                    r0_##idx = vaddq_f32(r0_##idx, vreinterpretq_f32_u32(veorq_u32(va0_0, m))); \
                    r1_##idx = vaddq_f32(r1_##idx, vreinterpretq_f32_u32(veorq_u32(va1_0, m))); \
                    m = vld1q_u32(&sign_lut[((packed1 >> shift) & 0xF) * 4]); \
                    r0_##idx = vaddq_f32(r0_##idx, vreinterpretq_f32_u32(veorq_u32(va0_1, m))); \
                    r1_##idx = vaddq_f32(r1_##idx, vreinterpretq_f32_u32(veorq_u32(va1_1, m)));
                
                PROCESS_NIB(0, 0)
                PROCESS_NIB(1, 4)
                PROCESS_NIB(2, 8)
                PROCESS_NIB(3, 12)
                PROCESS_NIB(4, 16)
                PROCESS_NIB(5, 20)
                PROCESS_NIB(6, 24)
                PROCESS_NIB(7, 28)
                
                #undef PROCESS_NIB
            }
            
            // Handle odd p
            for (; p < K; ++p) {
                uint32_t ab0, ab1;
                { float v = a0[p]; __builtin_memcpy(&ab0, &v, 4); }
                { float v = a1[p]; __builtin_memcpy(&ab1, &v, 4); }
                uint32x4_t va0 = vdupq_n_u32(ab0);
                uint32x4_t va1 = vdupq_n_u32(ab1);
                uint32_t packed = B[p * K_ints + jb];
                uint32x4_t m;
                
                m = vld1q_u32(&sign_lut[((packed>> 0)&0xF)*4]); r0_0=vaddq_f32(r0_0,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_0=vaddq_f32(r1_0,vreinterpretq_f32_u32(veorq_u32(va1,m)));
                m = vld1q_u32(&sign_lut[((packed>> 4)&0xF)*4]); r0_1=vaddq_f32(r0_1,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_1=vaddq_f32(r1_1,vreinterpretq_f32_u32(veorq_u32(va1,m)));
                m = vld1q_u32(&sign_lut[((packed>> 8)&0xF)*4]); r0_2=vaddq_f32(r0_2,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_2=vaddq_f32(r1_2,vreinterpretq_f32_u32(veorq_u32(va1,m)));
                m = vld1q_u32(&sign_lut[((packed>>12)&0xF)*4]); r0_3=vaddq_f32(r0_3,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_3=vaddq_f32(r1_3,vreinterpretq_f32_u32(veorq_u32(va1,m)));
                m = vld1q_u32(&sign_lut[((packed>>16)&0xF)*4]); r0_4=vaddq_f32(r0_4,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_4=vaddq_f32(r1_4,vreinterpretq_f32_u32(veorq_u32(va1,m)));
                m = vld1q_u32(&sign_lut[((packed>>20)&0xF)*4]); r0_5=vaddq_f32(r0_5,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_5=vaddq_f32(r1_5,vreinterpretq_f32_u32(veorq_u32(va1,m)));
                m = vld1q_u32(&sign_lut[((packed>>24)&0xF)*4]); r0_6=vaddq_f32(r0_6,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_6=vaddq_f32(r1_6,vreinterpretq_f32_u32(veorq_u32(va1,m)));
                m = vld1q_u32(&sign_lut[((packed>>28)&0xF)*4]); r0_7=vaddq_f32(r0_7,vreinterpretq_f32_u32(veorq_u32(va0,m))); r1_7=vaddq_f32(r1_7,vreinterpretq_f32_u32(veorq_u32(va1,m)));
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
        }
    }

    for (; i < M; ++i) {
        const float* __restrict__ a_row = A + i * K;
        float* __restrict__ c_row = C + i * K;
        for (size_t jb = 0; jb < K_ints; ++jb) {
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
