#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Matrix multiplication (A * B) where A is M x K float matrix,
// B is packed binary matrix (K x K) with 1 bit = +1.0f, 0 bit = -1.0f.
// C is the M x K output matrix. K is a multiple of 32.
// This implementation uses NEON intrinsics and processes 4 columns at a time.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // words per row in B

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // zero output row
        for (size_t j = 0; j < K; ++j) c_row[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const float32x4_t val_vec = vdupq_n_f32(a_val);
            size_t base = p * K_ints;
            for (size_t wi = 0; wi < K_ints; ++wi) {
                uint32_t packed = B[base + wi];
                // Process 4 columns at a time
                for (size_t offset = 0; offset < 32; offset += 4) {
                    uint32_t bits4 = (packed >> offset) & 0xF;
                    // Prepare sign vector on stack
                    const float sign_vals[16][4] = {
                        {-1.0f,-1.0f,-1.0f,-1.0f}, { 1.0f,-1.0f,-1.0f,-1.0f}, {-1.0f, 1.0f,-1.0f,-1.0f}, { 1.0f, 1.0f,-1.0f,-1.0f},
                        {-1.0f,-1.0f, 1.0f,-1.0f}, { 1.0f,-1.0f, 1.0f,-1.0f}, {-1.0f, 1.0f, 1.0f,-1.0f}, { 1.0f, 1.0f, 1.0f,-1.0f},
                        {-1.0f,-1.0f,-1.0f, 1.0f}, { 1.0f,-1.0f,-1.0f, 1.0f}, {-1.0f, 1.0f,-1.0f, 1.0f}, { 1.0f, 1.0f,-1.0f, 1.0f},
                        {-1.0f,-1.0f, 1.0f, 1.0f}, { 1.0f,-1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f, 1.0f}, { 1.0f, 1.0f, 1.0f, 1.0f}
                    };
                    float32x4_t sign_vec = vld1q_f32(sign_vals[bits4]);
                    float32x4_t out_vec = vld1q_f32(c_row + offset);
                    out_vec = vaddq_f32(out_vec, vmulq_f32(val_vec, sign_vec));
                    vst1q_f32(c_row + offset, out_vec);
                }
            }
        }
    }
}
