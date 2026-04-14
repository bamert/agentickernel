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

    // Tile j to keep C rows in L1 cache
    // 8 rows x J_TILE floats = 8 * J_TILE * 4 bytes in C
    // L1 = 64KB, so J_TILE = 512 -> 8 * 512 * 4 = 16KB (fits in L1)
    const size_t J_TILE = 512; // must be multiple of 32

    // Process 8 rows of A at a time
    size_t i = 0;
    for (; i + 8 <= M; i += 8) {
        float* __restrict__ cr[8];
        const float* __restrict__ ar[8];
        for (int r = 0; r < 8; ++r) {
            cr[r] = C + (i + r) * K;
            ar[r] = A + (i + r) * K;
        }

        for (size_t jt = 0; jt < K; jt += J_TILE) {
            size_t jb_start = jt / 32;
            size_t jb_end = (jt + J_TILE) / 32;
            if (jb_end > K_ints) jb_end = K_ints;

            for (size_t p = 0; p < K; ++p) {
                uint32_t ab[8];
                for (int r = 0; r < 8; ++r) {
                    float a = ar[r][p];
                    __builtin_memcpy(&ab[r], &a, 4);
                }
                uint32x4_t va0 = vdupq_n_u32(ab[0]);
                uint32x4_t va1 = vdupq_n_u32(ab[1]);
                uint32x4_t va2 = vdupq_n_u32(ab[2]);
                uint32x4_t va3 = vdupq_n_u32(ab[3]);
                uint32x4_t va4 = vdupq_n_u32(ab[4]);
                uint32x4_t va5 = vdupq_n_u32(ab[5]);
                uint32x4_t va6 = vdupq_n_u32(ab[6]);
                uint32x4_t va7 = vdupq_n_u32(ab[7]);

                const uint32_t* b_row = B + p * K_ints;

                for (size_t jb = jb_start; jb < jb_end; ++jb) {
                    uint32_t packed = b_row[jb];
                    size_t j_base = jb * 32;

                    for (int n = 0; n < 8; ++n) {
                        int nib = (packed >> (n * 4)) & 0xF;
                        uint32x4_t mask = vld1q_u32(&sign_lut[nib * 4]);
                        size_t off = j_base + n * 4;

                        float32x4_t c0 = vld1q_f32(cr[0] + off);
                        float32x4_t c1 = vld1q_f32(cr[1] + off);
                        float32x4_t c2 = vld1q_f32(cr[2] + off);
                        float32x4_t c3 = vld1q_f32(cr[3] + off);

                        c0 = vaddq_f32(c0, vreinterpretq_f32_u32(veorq_u32(va0, mask)));
                        c1 = vaddq_f32(c1, vreinterpretq_f32_u32(veorq_u32(va1, mask)));
                        c2 = vaddq_f32(c2, vreinterpretq_f32_u32(veorq_u32(va2, mask)));
                        c3 = vaddq_f32(c3, vreinterpretq_f32_u32(veorq_u32(va3, mask)));

                        vst1q_f32(cr[0] + off, c0);
                        vst1q_f32(cr[1] + off, c1);
                        vst1q_f32(cr[2] + off, c2);
                        vst1q_f32(cr[3] + off, c3);

                        float32x4_t c4 = vld1q_f32(cr[4] + off);
                        float32x4_t c5 = vld1q_f32(cr[5] + off);
                        float32x4_t c6 = vld1q_f32(cr[6] + off);
                        float32x4_t c7 = vld1q_f32(cr[7] + off);

                        c4 = vaddq_f32(c4, vreinterpretq_f32_u32(veorq_u32(va4, mask)));
                        c5 = vaddq_f32(c5, vreinterpretq_f32_u32(veorq_u32(va5, mask)));
                        c6 = vaddq_f32(c6, vreinterpretq_f32_u32(veorq_u32(va6, mask)));
                        c7 = vaddq_f32(c7, vreinterpretq_f32_u32(veorq_u32(va7, mask)));

                        vst1q_f32(cr[4] + off, c4);
                        vst1q_f32(cr[5] + off, c5);
                        vst1q_f32(cr[6] + off, c6);
                        vst1q_f32(cr[7] + off, c7);
                    }
                }
            }
        }
    }

    // Handle remaining rows (< 8)
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
