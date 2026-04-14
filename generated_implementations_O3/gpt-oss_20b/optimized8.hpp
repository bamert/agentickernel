#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>

// Matrix multiplication: A (M x K), B packed binary (K x K), C output (M x K)
// Optimized with NEON but processes four columns at a time.  Sign values are computed
// sequentially but the accumulation of 4 partial sums is vectorised, keeping the
// algorithm simple and correct.  The implementation handles 32‑bit word boundaries
// properly.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;
    const size_t block = 4; // process 4 columns with SIMD

    // Zero output matrix C using memset (all bits zero corresponds to 0.0f).
    std::memset(C, 0, M * K * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = &A[i * K];
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_index = 0;            // index into B row
            uint32_t packed = b_row[b_index];
            unsigned shift = 0;            // bit position within packed

            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Load 4 values of C
                float32x4_t c_vec = vld1q_f32(&c_row[j]);

                // Compute signs for the 4 columns
                float signs[4];
                for (int t = 0; t < 4; ++t) {
                    uint32_t bit = (packed >> shift) & 1U;
                    signs[t] = bit ? 1.0f : -1.0f;
                    shift++;
                    if (shift == 32) {
                        shift = 0;
                        ++b_index;
                        packed = b_row[b_index];
                    }
                }
                // Broadcast signs into SIMD registers
                float32x4_t s_vec = vld1q_f32(signs);

                // Multiply and accumulate
                float32x4_t a_vec = vdupq_n_f32(a_val);
                c_vec = vaddq_f32(c_vec, vmulq_f32(a_vec, s_vec));

                // Store result
                vst1q_f32(&c_row[j], c_vec);
            }

            // Handle any remaining columns that do not fill a full block
            for (; j < K; ++j) {
                uint32_t bit = (packed >> shift) & 1U;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;
                shift++;
                if (shift == 32) {
                    shift = 0;
                    ++b_index;
                    packed = b_row[b_index];
                }
            }
        }
    }
}
