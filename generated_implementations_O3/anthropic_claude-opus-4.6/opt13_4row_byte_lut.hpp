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

    // Byte LUT: 256 entries x 8 uint32 each (2 x float32x4)
    uint32_t sign_lut[256 * 8] __attribute__((aligned(16)));
    for (int byte_val = 0; byte_val < 256; ++byte_val) {
        for (int b = 0; b < 8; ++b) {
            int bit = (byte_val >> b) & 1;
            sign_lut[byte_val * 8 + b] = bit ? 0u : 0x80000000u;
        }
    }

    // Process 4 rows of A at a time
    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* __restrict__ c0 = C + (i + 0) * K;
        float* __restrict__ c1 = C + (i + 1) * K;
        float* __restrict__ c2 = C + (i + 2) * K;
        float* __restrict__ c3 = C + (i + 3) * K;
        const float* __restrict__ a0 = A + (i + 0) * K;
        const float* __restrict__ a1 = A + (i + 1) * K;
        const float* __restrict__ a2 = A + (i + 2) * K;
        const float* __restrict__ a3 = A + (i + 3) * K;

        for (size_t p = 0; p < K; ++p) {
            uint32_t ab0, ab1, ab2, ab3;
            {
                float v0 = a0[p]; __builtin_memcpy(&ab0, &v0, 4);
                float v1 = a1[p]; __builtin_memcpy(&ab1, &v1, 4);
                float v2 = a2[p]; __builtin_memcpy(&ab2, &v2, 4);
                float v3 = a3[p]; __builtin_memcpy(&ab3, &v3, 4);
            }
            uint32x4_t va0 = vdupq_n_u32(ab0);
            uint32x4_t va1 = vdupq_n_u32(ab1);
            uint32x4_t va2 = vdupq_n_u32(ab2);
            uint32x4_t va3 = vdupq_n_u32(ab3);

            const uint32_t* b_row = B + p * K_ints;

            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = b_row[jb];
                size_t j_base = jb * 32;

                // Process byte 0 (bits 0-7)
                {
                    const uint32_t* lut = &sign_lut[(packed & 0xFF) * 8];
                    uint32x4_t m0 = vld1q_u32(lut);
                    uint32x4_t m1 = vld1q_u32(lut + 4);
                    
                    float32x4_t r;
                    r = vld1q_f32(c0 + j_base); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m0))); vst1q_f32(c0 + j_base, r);
                    r = vld1q_f32(c1 + j_base); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m0))); vst1q_f32(c1 + j_base, r);
                    r = vld1q_f32(c2 + j_base); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m0))); vst1q_f32(c2 + j_base, r);
                    r = vld1q_f32(c3 + j_base); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m0))); vst1q_f32(c3 + j_base, r);

                    r = vld1q_f32(c0 + j_base + 4); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m1))); vst1q_f32(c0 + j_base + 4, r);
                    r = vld1q_f32(c1 + j_base + 4); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m1))); vst1q_f32(c1 + j_base + 4, r);
                    r = vld1q_f32(c2 + j_base + 4); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m1))); vst1q_f32(c2 + j_base + 4, r);
                    r = vld1q_f32(c3 + j_base + 4); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m1))); vst1q_f32(c3 + j_base + 4, r);
                }
                // Process byte 1 (bits 8-15)
                {
                    const uint32_t* lut = &sign_lut[((packed >> 8) & 0xFF) * 8];
                    uint32x4_t m0 = vld1q_u32(lut);
                    uint32x4_t m1 = vld1q_u32(lut + 4);
                    
                    float32x4_t r;
                    r = vld1q_f32(c0 + j_base + 8); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m0))); vst1q_f32(c0 + j_base + 8, r);
                    r = vld1q_f32(c1 + j_base + 8); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m0))); vst1q_f32(c1 + j_base + 8, r);
                    r = vld1q_f32(c2 + j_base + 8); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m0))); vst1q_f32(c2 + j_base + 8, r);
                    r = vld1q_f32(c3 + j_base + 8); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m0))); vst1q_f32(c3 + j_base + 8, r);

                    r = vld1q_f32(c0 + j_base + 12); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m1))); vst1q_f32(c0 + j_base + 12, r);
                    r = vld1q_f32(c1 + j_base + 12); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m1))); vst1q_f32(c1 + j_base + 12, r);
                    r = vld1q_f32(c2 + j_base + 12); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m1))); vst1q_f32(c2 + j_base + 12, r);
                    r = vld1q_f32(c3 + j_base + 12); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m1))); vst1q_f32(c3 + j_base + 12, r);
                }
                // Process byte 2 (bits 16-23)
                {
                    const uint32_t* lut = &sign_lut[((packed >> 16) & 0xFF) * 8];
                    uint32x4_t m0 = vld1q_u32(lut);
                    uint32x4_t m1 = vld1q_u32(lut + 4);
                    
                    float32x4_t r;
                    r = vld1q_f32(c0 + j_base + 16); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m0))); vst1q_f32(c0 + j_base + 16, r);
                    r = vld1q_f32(c1 + j_base + 16); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m0))); vst1q_f32(c1 + j_base + 16, r);
                    r = vld1q_f32(c2 + j_base + 16); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m0))); vst1q_f32(c2 + j_base + 16, r);
                    r = vld1q_f32(c3 + j_base + 16); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m0))); vst1q_f32(c3 + j_base + 16, r);

                    r = vld1q_f32(c0 + j_base + 20); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m1))); vst1q_f32(c0 + j_base + 20, r);
                    r = vld1q_f32(c1 + j_base + 20); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m1))); vst1q_f32(c1 + j_base + 20, r);
                    r = vld1q_f32(c2 + j_base + 20); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m1))); vst1q_f32(c2 + j_base + 20, r);
                    r = vld1q_f32(c3 + j_base + 20); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m1))); vst1q_f32(c3 + j_base + 20, r);
                }
                // Process byte 3 (bits 24-31)
                {
                    const uint32_t* lut = &sign_lut[((packed >> 24) & 0xFF) * 8];
                    uint32x4_t m0 = vld1q_u32(lut);
                    uint32x4_t m1 = vld1q_u32(lut + 4);
                    
                    float32x4_t r;
                    r = vld1q_f32(c0 + j_base + 24); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m0))); vst1q_f32(c0 + j_base + 24, r);
                    r = vld1q_f32(c1 + j_base + 24); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m0))); vst1q_f32(c1 + j_base + 24, r);
                    r = vld1q_f32(c2 + j_base + 24); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m0))); vst1q_f32(c2 + j_base + 24, r);
                    r = vld1q_f32(c3 + j_base + 24); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m0))); vst1q_f32(c3 + j_base + 24, r);

                    r = vld1q_f32(c0 + j_base + 28); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va0, m1))); vst1q_f32(c0 + j_base + 28, r);
                    r = vld1q_f32(c1 + j_base + 28); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va1, m1))); vst1q_f32(c1 + j_base + 28, r);
                    r = vld1q_f32(c2 + j_base + 28); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va2, m1))); vst1q_f32(c2 + j_base + 28, r);
                    r = vld1q_f32(c3 + j_base + 28); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va3, m1))); vst1q_f32(c3 + j_base + 28, r);
                }
            }
        }
    }

    // Remaining rows
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
                size_t j_base = jb * 32;
                for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
                    const uint32_t* lut = &sign_lut[((packed >> (byte_idx * 8)) & 0xFF) * 8];
                    uint32x4_t m0 = vld1q_u32(lut);
                    uint32x4_t m1 = vld1q_u32(lut + 4);
                    size_t off = j_base + byte_idx * 8;
                    float32x4_t r;
                    r = vld1q_f32(c_row + off); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va_bits, m0))); vst1q_f32(c_row + off, r);
                    r = vld1q_f32(c_row + off + 4); r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va_bits, m1))); vst1q_f32(c_row + off + 4, r);
                }
            }
        }
    }
}
