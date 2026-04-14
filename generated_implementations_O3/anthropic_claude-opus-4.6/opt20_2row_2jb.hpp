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

    // Process 2 rows at a time, 1 jb (32 cols) per tile
    // Inner loop over p: read 1 A value per row, 1 B word, 8 LUT loads
    // Try to reduce by using byte-level extraction instead of nibble
    // and fewer LUT loads: 4 byte LUT loads vs 8 nibble LUT loads
    
    // Actually, let me try to eliminate the LUT entirely.
    // For each bit in packed: mask = (~bit & 1) << 31
    // Using NEON: broadcast packed, right-shift by {0..3}, AND 1, XOR 1, left-shift 31
    // This avoids LUT memory access entirely.
    
    const uint32x4_t one_u = vdupq_n_u32(1);
    const int32x4_t shift_31 = vdupq_n_s32(31);
    
    int32_t shifts_arr[8][4] = {
        {0, -1, -2, -3}, {-4, -5, -6, -7},
        {-8, -9, -10, -11}, {-12, -13, -14, -15},
        {-16, -17, -18, -19}, {-20, -21, -22, -23},
        {-24, -25, -26, -27}, {-28, -29, -30, -31}
    };
    int32x4_t shifts[8];
    for (int n = 0; n < 8; ++n) shifts[n] = vld1q_s32(shifts_arr[n]);

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

            for (size_t p = 0; p < K; ++p) {
                uint32_t ab0, ab1;
                { float v = a0[p]; __builtin_memcpy(&ab0, &v, 4); }
                { float v = a1[p]; __builtin_memcpy(&ab1, &v, 4); }
                uint32x4_t va0 = vdupq_n_u32(ab0);
                uint32x4_t va1 = vdupq_n_u32(ab1);

                uint32_t packed = B[p * K_ints + jb];
                uint32x4_t vpacked = vdupq_n_u32(packed);
                
                uint32x4_t m;
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[0]), one_u), one_u), shift_31);
                r0_0 = vaddq_f32(r0_0, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_0 = vaddq_f32(r1_0, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[1]), one_u), one_u), shift_31);
                r0_1 = vaddq_f32(r0_1, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_1 = vaddq_f32(r1_1, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[2]), one_u), one_u), shift_31);
                r0_2 = vaddq_f32(r0_2, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_2 = vaddq_f32(r1_2, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[3]), one_u), one_u), shift_31);
                r0_3 = vaddq_f32(r0_3, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_3 = vaddq_f32(r1_3, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[4]), one_u), one_u), shift_31);
                r0_4 = vaddq_f32(r0_4, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_4 = vaddq_f32(r1_4, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[5]), one_u), one_u), shift_31);
                r0_5 = vaddq_f32(r0_5, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_5 = vaddq_f32(r1_5, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[6]), one_u), one_u), shift_31);
                r0_6 = vaddq_f32(r0_6, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_6 = vaddq_f32(r1_6, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked, shifts[7]), one_u), one_u), shift_31);
                r0_7 = vaddq_f32(r0_7, vreinterpretq_f32_u32(veorq_u32(va0, m)));
                r1_7 = vaddq_f32(r1_7, vreinterpretq_f32_u32(veorq_u32(va1, m)));
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
            float32x4_t acc0=vdupq_n_f32(0), acc1=vdupq_n_f32(0), acc2=vdupq_n_f32(0), acc3=vdupq_n_f32(0);
            float32x4_t acc4=vdupq_n_f32(0), acc5=vdupq_n_f32(0), acc6=vdupq_n_f32(0), acc7=vdupq_n_f32(0);

            for (size_t p = 0; p < K; ++p) {
                uint32_t a_bits; float a_val = a_row[p]; __builtin_memcpy(&a_bits, &a_val, 4);
                uint32x4_t va = vdupq_n_u32(a_bits);
                uint32_t packed = B[p * K_ints + jb];
                uint32x4_t vpacked = vdupq_n_u32(packed);
                uint32x4_t m;
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[0]),one_u),one_u),shift_31); acc0=vaddq_f32(acc0,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[1]),one_u),one_u),shift_31); acc1=vaddq_f32(acc1,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[2]),one_u),one_u),shift_31); acc2=vaddq_f32(acc2,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[3]),one_u),one_u),shift_31); acc3=vaddq_f32(acc3,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[4]),one_u),one_u),shift_31); acc4=vaddq_f32(acc4,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[5]),one_u),one_u),shift_31); acc5=vaddq_f32(acc5,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[6]),one_u),one_u),shift_31); acc6=vaddq_f32(acc6,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m = vshlq_u32(veorq_u32(vandq_u32(vshlq_u32(vpacked,shifts[7]),one_u),one_u),shift_31); acc7=vaddq_f32(acc7,vreinterpretq_f32_u32(veorq_u32(va,m)));
            }
            size_t j_base = jb * 32;
            vst1q_f32(c_row+j_base+0,acc0); vst1q_f32(c_row+j_base+4,acc1);
            vst1q_f32(c_row+j_base+8,acc2); vst1q_f32(c_row+j_base+12,acc3);
            vst1q_f32(c_row+j_base+16,acc4); vst1q_f32(c_row+j_base+20,acc5);
            vst1q_f32(c_row+j_base+24,acc6); vst1q_f32(c_row+j_base+28,acc7);
        }
    }
}
