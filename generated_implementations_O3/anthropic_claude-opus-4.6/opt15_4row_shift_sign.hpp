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

    // Strategy: For each group of 4 bits from packed, we need to create a sign mask.
    // If bit=0, sign=-1 -> XOR with 0x80000000. If bit=1, sign=+1 -> XOR with 0.
    // mask = (~bit & 1) << 31
    // 
    // Using NEON: broadcast packed, shift right by {0,1,2,3}, AND with 1, 
    // then XOR with 1 (to invert), shift left by 31.
    // Or: shift the bit into position 31 directly, then XOR with 0x80000000.
    //
    // Actually simpler: shift right to put bit in position 0, NOT, AND 1, shift left 31.
    // Or: just use the nibble LUT which was already fast.
    
    // Let's try: for 4 consecutive bits starting at position b:
    // Extract each bit individually and shift to bit 31.
    // Use NEON shifts: vshlq with negative values = right shift.
    
    // Pre-build shift vectors for each nibble position
    // nibble n: bits at positions n*4, n*4+1, n*4+2, n*4+3
    // We shift right by those amounts, AND with 1, XOR with 1, shift left by 31
    
    const uint32x4_t one32 = vdupq_n_u32(1);
    const int32x4_t shift31 = vdupq_n_s32(31);
    
    // Shift amounts for each nibble
    int32x4_t nib_shifts[8];
    for (int n = 0; n < 8; ++n) {
        int base = n * 4;
        int32_t s[4] = {-base, -(base+1), -(base+2), -(base+3)};
        nib_shifts[n] = vld1q_s32(s);
    }

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
                uint32x4_t vpacked = vdupq_n_u32(packed);
                size_t j_base = jb * 32;

                for (int n = 0; n < 8; ++n) {
                    // Extract 4 bits, create sign mask
                    uint32x4_t bits = vandq_u32(vshlq_u32(vpacked, nib_shifts[n]), one32);
                    // bits: 1 means +1, 0 means -1
                    // mask: invert bit, shift to sign position
                    uint32x4_t mask = vshlq_u32(veorq_u32(bits, one32), shift31);
                    
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
                uint32x4_t vpacked = vdupq_n_u32(packed);
                size_t j_base = jb * 32;

                for (int n = 0; n < 8; ++n) {
                    uint32x4_t bits = vandq_u32(vshlq_u32(vpacked, nib_shifts[n]), one32);
                    uint32x4_t mask = vshlq_u32(veorq_u32(bits, one32), shift31);
                    size_t off = j_base + n * 4;
                    float32x4_t r = vld1q_f32(c_row + off);
                    r = vaddq_f32(r, vreinterpretq_f32_u32(veorq_u32(va_bits, mask)));
                    vst1q_f32(c_row + off, r);
                }
            }
        }
    }
}
