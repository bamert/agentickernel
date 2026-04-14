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

    // Nibble LUT: 16 entries x 4 uint32 each
    uint32_t sign_lut[16 * 4] __attribute__((aligned(16)));
    for (int nibble = 0; nibble < 16; ++nibble) {
        for (int b = 0; b < 4; ++b) {
            int bit = (nibble >> b) & 1;
            sign_lut[nibble * 4 + b] = bit ? 0u : 0x80000000u;
        }
    }

    // Process 4 rows of A at a time to reuse B loads
    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* __restrict__ c_row0 = C + i * K;
        float* __restrict__ c_row1 = C + (i + 1) * K;
        float* __restrict__ c_row2 = C + (i + 2) * K;
        float* __restrict__ c_row3 = C + (i + 3) * K;
        const float* __restrict__ a_row0 = A + i * K;
        const float* __restrict__ a_row1 = A + (i + 1) * K;
        const float* __restrict__ a_row2 = A + (i + 2) * K;
        const float* __restrict__ a_row3 = A + (i + 3) * K;
        
        for (size_t p = 0; p < K; ++p) {
            uint32_t a0_bits, a1_bits, a2_bits, a3_bits;
            {
                float a0 = a_row0[p]; __builtin_memcpy(&a0_bits, &a0, 4);
                float a1 = a_row1[p]; __builtin_memcpy(&a1_bits, &a1, 4);
                float a2 = a_row2[p]; __builtin_memcpy(&a2_bits, &a2, 4);
                float a3 = a_row3[p]; __builtin_memcpy(&a3_bits, &a3, 4);
            }
            uint32x4_t va0 = vdupq_n_u32(a0_bits);
            uint32x4_t va1 = vdupq_n_u32(a1_bits);
            uint32x4_t va2 = vdupq_n_u32(a2_bits);
            uint32x4_t va3 = vdupq_n_u32(a3_bits);
            
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                size_t j_base = jb * 32;
                
                // 8 nibbles per uint32
                for (int n = 0; n < 8; ++n) {
                    int nibble = (packed >> (n * 4)) & 0xF;
                    uint32x4_t mask = vld1q_u32(&sign_lut[nibble * 4]);
                    size_t off = j_base + n * 4;
                    
                    float32x4_t c0 = vld1q_f32(c_row0 + off);
                    float32x4_t c1 = vld1q_f32(c_row1 + off);
                    float32x4_t c2 = vld1q_f32(c_row2 + off);
                    float32x4_t c3 = vld1q_f32(c_row3 + off);
                    
                    c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(va0, mask)));
                    c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(va1, mask)));
                    c2 = vaddq_f32(c2, vreinterpretq_f32_u32(veorq_u32(va2, mask)));
                    c3 = vaddq_f32(c3, vreinterpretq_f32_u32(veorq_u32(va3, mask)));
                    
                    vst1q_f32(c_row0 + off, c0);
                    vst1q_f32(c_row1 + off, c1);
                    vst1q_f32(c_row2 + off, c2);
                    vst1q_f32(c_row3 + off, c3);
                }
            }
        }
    }

    // Handle remaining rows
    for (; i < M; ++i) {
        float* __restrict__ c_row = C + i * K;
        const float* __restrict__ a_row = A + i * K;
        
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            uint32_t a_bits;
            __builtin_memcpy(&a_bits, &a_val, 4);
            uint32x4_t va_bits = vdupq_n_u32(a_bits);
            
            const uint32_t* b_row = B + p * K_ints;
            
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                float* c_ptr = c_row + jb * 32;
                
                for (int n = 0; n < 8; ++n) {
                    int nibble = (packed >> (n * 4)) & 0xF;
                    uint32x4_t mask = vld1q_u32(&sign_lut[nibble * 4]);
                    float32x4_t c_vec = vld1q_f32(c_ptr + n * 4);
                    c_vec = vaddq_f32(c_vec, vreinterpretq_f32_u32(veorq_u32(va_bits, mask)));
                    vst1q_f32(c_ptr + n * 4, c_vec);
                }
            }
        }
    }
}
