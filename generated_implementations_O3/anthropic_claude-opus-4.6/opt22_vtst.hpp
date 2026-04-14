#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Use vtst to test individual bits.
    // Broadcast packed to all 4 lanes, then vtst with {1<<0, 1<<1, 1<<2, 1<<3} etc.
    // vtst returns 0xFFFFFFFF if (a & b) != 0, else 0.
    // Then use vbsl to select +a_val or -a_val.
    
    // Bit test vectors for each group of 4 bits
    uint32_t bit_tests_arr[8][4] = {
        {1u<<0, 1u<<1, 1u<<2, 1u<<3},
        {1u<<4, 1u<<5, 1u<<6, 1u<<7},
        {1u<<8, 1u<<9, 1u<<10, 1u<<11},
        {1u<<12, 1u<<13, 1u<<14, 1u<<15},
        {1u<<16, 1u<<17, 1u<<18, 1u<<19},
        {1u<<20, 1u<<21, 1u<<22, 1u<<23},
        {1u<<24, 1u<<25, 1u<<26, 1u<<27},
        {1u<<28, 1u<<29, 1u<<30, 1u<<31}
    };
    uint32x4_t bit_tests[8];
    for (int n = 0; n < 8; ++n) bit_tests[n] = vld1q_u32(bit_tests_arr[n]);
    
    // sign_bit mask for XOR: 0x80000000
    const uint32x4_t sign_mask = vdupq_n_u32(0x80000000u);

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
                
                // For each group: vtst gives all-1s if bit set, all-0s otherwise.
                // If bit set (sign=+1): we want XOR mask = 0 (keep sign)
                // If bit clear (sign=-1): we want XOR mask = 0x80000000 (flip sign)
                // mask = vbic(sign_mask, vtst_result) = sign_mask AND NOT vtst_result
                
                uint32x4_t m;
                
                #define PROCESS(idx) \
                    m = vbicq_u32(sign_mask, vtstq_u32(vpacked, bit_tests[idx])); \
                    r0_##idx = vaddq_f32(r0_##idx, vreinterpretq_f32_u32(veorq_u32(va0, m))); \
                    r1_##idx = vaddq_f32(r1_##idx, vreinterpretq_f32_u32(veorq_u32(va1, m)));
                
                PROCESS(0) PROCESS(1) PROCESS(2) PROCESS(3)
                PROCESS(4) PROCESS(5) PROCESS(6) PROCESS(7)
                
                #undef PROCESS
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
                uint32_t packed = B[p * K_ints + jb];
                uint32x4_t vpacked = vdupq_n_u32(packed);
                uint32x4_t m;
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[0])); a0=vaddq_f32(a0,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[1])); a1=vaddq_f32(a1,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[2])); a2=vaddq_f32(a2,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[3])); a3=vaddq_f32(a3,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[4])); a4=vaddq_f32(a4,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[5])); a5=vaddq_f32(a5,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[6])); a6=vaddq_f32(a6,vreinterpretq_f32_u32(veorq_u32(va,m)));
                m=vbicq_u32(sign_mask,vtstq_u32(vpacked,bit_tests[7])); a7=vaddq_f32(a7,vreinterpretq_f32_u32(veorq_u32(va,m)));
            }
            size_t j_base = jb*32;
            vst1q_f32(c_row+j_base+0,a0); vst1q_f32(c_row+j_base+4,a1);
            vst1q_f32(c_row+j_base+8,a2); vst1q_f32(c_row+j_base+12,a3);
            vst1q_f32(c_row+j_base+16,a4); vst1q_f32(c_row+j_base+20,a5);
            vst1q_f32(c_row+j_base+24,a6); vst1q_f32(c_row+j_base+28,a7);
        }
    }
}
