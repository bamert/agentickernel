#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* __restrict__ A, const uint32_t* __restrict__ B, float* __restrict__ C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero C using NEON
    {
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (size_t idx = 0; idx < M * K; idx += 16) {
            vst1q_f32(C + idx, zero);
            vst1q_f32(C + idx + 4, zero);
            vst1q_f32(C + idx + 8, zero);
            vst1q_f32(C + idx + 12, zero);
        }
    }

    // Precompute lookup: for each byte (0-255), store 8 sign masks (2 x uint32x4_t)
    // Aligned for better load performance
    uint32_t sign_lut[256 * 8] __attribute__((aligned(16)));
    for (int byte = 0; byte < 256; ++byte) {
        for (int b = 0; b < 8; ++b) {
            int bit = (byte >> b) & 1;
            sign_lut[byte * 8 + b] = bit ? 0u : 0x80000000u;
        }
    }

    for (size_t i = 0; i < M; ++i) {
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
                
                // Process 32 bits using byte LUT: 4 bytes per uint32
                int byte0 = packed & 0xFF;
                int byte1 = (packed >> 8) & 0xFF;
                int byte2 = (packed >> 16) & 0xFF;
                int byte3 = (packed >> 24) & 0xFF;
                
                // Byte 0: bits 0-7
                {
                    const uint32_t* lut = &sign_lut[byte0 * 8];
                    uint32x4_t mask0 = vld1q_u32(lut);
                    uint32x4_t mask1 = vld1q_u32(lut + 4);
                    float32x4_t c0 = vld1q_f32(c_ptr);
                    float32x4_t c1 = vld1q_f32(c_ptr + 4);
                    c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(va_bits, mask0)));
                    c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(va_bits, mask1)));
                    vst1q_f32(c_ptr, c0);
                    vst1q_f32(c_ptr + 4, c1);
                }
                // Byte 1: bits 8-15
                {
                    const uint32_t* lut = &sign_lut[byte1 * 8];
                    uint32x4_t mask0 = vld1q_u32(lut);
                    uint32x4_t mask1 = vld1q_u32(lut + 4);
                    float32x4_t c0 = vld1q_f32(c_ptr + 8);
                    float32x4_t c1 = vld1q_f32(c_ptr + 12);
                    c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(va_bits, mask0)));
                    c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(va_bits, mask1)));
                    vst1q_f32(c_ptr + 8, c0);
                    vst1q_f32(c_ptr + 12, c1);
                }
                // Byte 2: bits 16-23
                {
                    const uint32_t* lut = &sign_lut[byte2 * 8];
                    uint32x4_t mask0 = vld1q_u32(lut);
                    uint32x4_t mask1 = vld1q_u32(lut + 4);
                    float32x4_t c0 = vld1q_f32(c_ptr + 16);
                    float32x4_t c1 = vld1q_f32(c_ptr + 20);
                    c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(va_bits, mask0)));
                    c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(va_bits, mask1)));
                    vst1q_f32(c_ptr + 16, c0);
                    vst1q_f32(c_ptr + 20, c1);
                }
                // Byte 3: bits 24-31
                {
                    const uint32_t* lut = &sign_lut[byte3 * 8];
                    uint32x4_t mask0 = vld1q_u32(lut);
                    uint32x4_t mask1 = vld1q_u32(lut + 4);
                    float32x4_t c0 = vld1q_f32(c_ptr + 24);
                    float32x4_t c1 = vld1q_f32(c_ptr + 28);
                    c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(va_bits, mask0)));
                    c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(va_bits, mask1)));
                    vst1q_f32(c_ptr + 24, c0);
                    vst1q_f32(c_ptr + 28, c1);
                }
            }
        }
    }
}
