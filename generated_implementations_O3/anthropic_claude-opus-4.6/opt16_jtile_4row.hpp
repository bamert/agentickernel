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

    // J tile: process J_TILE columns at a time
    // 4 rows * J_TILE * 4 bytes should fit in L1 (64KB)
    // J_TILE = 256 -> 4*256*4 = 4KB (very comfortable)
    // J_TILE = 1024 -> 4*1024*4 = 16KB (still fits)
    const size_t J_TILE_INTS = 8; // 8 uint32 = 256 columns
    
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

        for (size_t jt = 0; jt < K_ints; jt += J_TILE_INTS) {
            size_t jb_end = jt + J_TILE_INTS;
            if (jb_end > K_ints) jb_end = K_ints;

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

                for (size_t jb = jt; jb < jb_end; ++jb) {
                    uint32_t packed = b_row[jb];
                    size_t j_base = jb * 32;

                    for (int n = 0; n < 8; ++n) {
                        int nib = (packed >> (n * 4)) & 0xF;
                        uint32x4_t mask = vld1q_u32(&sign_lut[nib * 4]);
                        size_t off = j_base + n * 4;

                        float32x4_t r0 = vld1q_f32(c0 + off);
                        float32x4_t r1 = vld1q_f32(c1 + off);
                        float32x4_t r2 = vld1q_f32(c2 + off);
                        float32x4_t r3 = vld1q_f32(c3 + off);

                        r0 = vaddq_f32(r0, vreinterpretq_f32_u32(veorq_u32(va0, mask)));
                        r1 = vaddq_f32(r1, vreinterpretq_f32_u32(veorq_u32(va1, mask)));
                        r2 = vaddq_f32(r2, vreinterpretq_f32_u32(veorq_u32(va2, mask)));
                        r3 = vaddq_f32(r3, vreinterpretq_f32_u32(veorq_u32(va3, mask)));

                        vst1q_f32(c0 + off, r0);
                        vst1q_f32(c1 + off, r1);
                        vst1q_f32(c2 + off, r2);
                        vst1q_f32(c3 + off, r3);
                    }
                }
            }
        }
    }

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
                    int nib = (packed >> (n * 4)) & 0xF;
                    uint32x4_t mask = vld1q_u32(&sign_lut[nib * 4]);
                    float32x4_t c_vec = vld1q_f32(c_ptr + n * 4);
                    c_vec = vaddq_f32(c_vec, vreinterpretq_f32_u32(veorq_u32(va_bits, mask)));
                    vst1q_f32(c_ptr + n * 4, c_vec);
                }
            }
        }
    }
}
