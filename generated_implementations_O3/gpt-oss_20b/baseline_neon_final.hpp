#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Matrix multiplication using NEON intrinsics.
// A is M x K float matrix (row-major).
// B is K x K packed binary matrix: each bit = +1.0f if 1, -1.0f if 0.
// C is M x K output matrix (row-major). K is a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of uint32 words per row of B

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // zero the output row
        for (size_t j = 0; j < K; ++j) c_row[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const float32x4_t val_vec = vdupq_n_f32(a_val);
            size_t base = p * K_ints;
            for (size_t wi = 0; wi < K_ints; ++wi) {
                uint32_t packed = B[base + wi];
                // Process 4 columns at a time
                for (size_t offset = 0; offset < 32; offset += 4) {
                    // Extract 4 bits
                    uint32_t b0 = (packed >> (offset + 0)) & 1U;
                    uint32_t b1 = (packed >> (offset + 1)) & 1U;
                    uint32_t b2 = (packed >> (offset + 2)) & 1U;
                    uint32_t b3 = (packed >> (offset + 3)) & 1U;
                    // Build a uint32x4_t vector from the bits
                    uint32x4_t bit_u32 = vdupq_n_u32(0);
                    bit_u32 = vsetq_lane_u32(b0, bit_u32, 0);
                    bit_u32 = vsetq_lane_u32(b1, bit_u32, 1);
                    bit_u32 = vsetq_lane_u32(b2, bit_u32, 2);
                    bit_u32 = vsetq_lane_u32(b3, bit_u32, 3);
                    // Convert bits to float (0.0 or 1.0)
                    float32x4_t bit_f32 = vcvtq_f32_u32(bit_u32);
                    // sign = 2*bit - 1
                    float32x4_t sign_vec = vsubq_f32(vmulq_n_f32(bit_f32, 2.0f), vdupq_n_f32(1.0f));
                    // Load current C values
                    float32x4_t c_vec = vld1q_f32(c_row + offset);
                    // Accumulate
                    c_vec = vaddq_f32(c_vec, vmulq_f32(val_vec, sign_vec));
                    // Store back
                    vst1q_f32(c_row + offset, c_vec);
                }
            }
        }
    }
}
