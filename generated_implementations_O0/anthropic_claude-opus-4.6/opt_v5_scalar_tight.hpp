
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Back to the approach that worked: accumulate a_val or -a_val into C.
// But now use NEON for the accumulation part with sign flipping via XOR.
// Key: expand each uint32_t of packed bits into 32 sign masks efficiently.
// Process 4 floats at a time. For each 4 bits, create sign mask via table.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Sign bit constant
    const uint32x4_t sign_bit = vdupq_n_u32(0x80000000u);
    
    // Bit position masks for extracting individual bits
    // For bits at positions 0,1,2,3 within a nibble
    const uint32x4_t bit_masks_0 = {1u << 0, 1u << 1, 1u << 2, 1u << 3};
    const uint32x4_t bit_masks_4 = {1u << 4, 1u << 5, 1u << 6, 1u << 7};
    const uint32x4_t bit_masks_8 = {1u << 8, 1u << 9, 1u << 10, 1u << 11};
    const uint32x4_t bit_masks_12 = {1u << 12, 1u << 13, 1u << 14, 1u << 15};
    const uint32x4_t bit_masks_16 = {1u << 16, 1u << 17, 1u << 18, 1u << 19};
    const uint32x4_t bit_masks_20 = {1u << 20, 1u << 21, 1u << 22, 1u << 23};
    const uint32x4_t bit_masks_24 = {1u << 24, 1u << 25, 1u << 26, 1u << 27};
    const uint32x4_t bit_masks_28 = {1u << 28, 1u << 29, 1u << 30, 1u << 31};
    
    const uint32x4_t* all_bit_masks[8] = {
        &bit_masks_0, &bit_masks_4, &bit_masks_8, &bit_masks_12,
        &bit_masks_16, &bit_masks_20, &bit_masks_24, &bit_masks_28
    };

    for (size_t i = 0; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;

        // Zero output row
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(C_row + j, vdupq_n_f32(0.0f));
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            uint32x4_t va_bits = vreinterpretq_u32_f32(va);
            const uint32_t* B_row_ptr = B + p * K_ints;

            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row_ptr[g];
                uint32x4_t vpacked = vdupq_n_u32(packed);
                float* C_out = C_row + g * 32;

                // Process 8 groups of 4 bits each
                for (int nibble = 0; nibble < 8; ++nibble) {
                    uint32x4_t bm = *all_bit_masks[nibble];
                    
                    // Test which bits are set
                    uint32x4_t test = vandq_u32(vpacked, bm);
                    // If bit is 0, we want sign_bit (to flip sign); if 1, we want 0
                    uint32x4_t is_zero = vceqq_u32(test, vdupq_n_u32(0));
                    // is_zero is 0xFFFFFFFF where bit was 0, 0 where bit was 1
                    uint32x4_t flip = vandq_u32(is_zero, sign_bit);
                    
                    // XOR a_val with flip to negate where bit=0
                    float32x4_t signed_a = vreinterpretq_f32_u32(veorq_u32(va_bits, flip));
                    
                    float32x4_t c_vec = vld1q_f32(C_out + nibble * 4);
                    c_vec = vaddq_f32(c_vec, signed_a);
                    vst1q_f32(C_out + nibble * 4, c_vec);
                }
            }
        }
    }
}
