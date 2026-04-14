#pragma once
#include <cstdint>
#include <cstddef>
// ARM NEON intrinsics are available via the harness – no explicit include needed.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* rowC = &C[i * K];
        // Zero the destination row
        for (size_t j = 0; j < K; ++j) rowC[j] = 0.0f;

        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            float32x4_t a_vec = vdupq_n_f32(a_val);
            const uint32_t* packed_row = &B[p * K_ints];

            for (size_t blk = 0; blk < K_ints; ++blk) {
                uint32_t bits = packed_row[blk];
                // Process 8 groups of 4 bits (32 bits total)
                for (size_t off = 0; off < 8; ++off) {
                    uint32_t sub = bits >> (off * 4);
                    // Build a vector of +/-1 based on the 4 bits
                    float32x4_t sign_vec = {
                        (float)((sub >> 0) & 1u ? 1.0f : -1.0f),
                        (float)((sub >> 1) & 1u ? 1.0f : -1.0f),
                        (float)((sub >> 2) & 1u ? 1.0f : -1.0f),
                        (float)((sub >> 3) & 1u ? 1.0f : -1.0f)
                    };
                    float32x4_t c_vec = vld1q_f32(&rowC[blk * 32 + off * 4]);
                    c_vec = vfmaq_f32(c_vec, a_vec, sign_vec);
                    vst1q_f32(&rowC[blk * 32 + off * 4], c_vec);
                }
            }
        }
    }
}
