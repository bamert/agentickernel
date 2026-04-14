#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <arm_neon.h>
#include <array>

// Pre‑computed sign table: for every 8‑bit mask we have 8 floats (+1.0f or -1.0f)
// The table is generated at compile time.
constexpr auto make_sign_table() {
    std::array<std::array<float, 8>, 256> table{};
    for (int m = 0; m < 256; ++m) {
        for (int b = 0; b < 8; ++b) {
            table[m][b] = ((m >> b) & 1) ? 1.0f : -1.0f;
        }
    }
    return table;
}
constexpr auto sign_table = make_sign_table();

// Matrix multiplication: A (M × K) · B (K × K) → C (M × K)
// B is packed binary – 1 bit represents +1.0f, 0 bit represents –1.0f.
// This implementation uses NEON intrinsics and looks up sign vectors from the
// pre‑computed table, processing 8 columns per iteration.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32; // number of 32‑bit words per row of B
    const size_t block  = 8;       // number of columns processed per SIMD step

    // Zero the output matrix C once.
    std::memset(C, 0, M * K * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        float* c_row = &C[i * K];
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* b_row = &B[p * K_ints];
            size_t b_index = 0;
            uint32_t packed = b_row[b_index];
            unsigned shift  = 0;

            size_t j = 0;
            for (; j + block <= K; j += block) {
                // Load current C block
                float32x4_t c_low  = vld1q_f32(&c_row[j]);
                float32x4_t c_high = vld1q_f32(&c_row[j + 4]);

                float32x4_t a_vec = vdupq_n_f32(a_val);

                // Compute mask of 8 bits
                unsigned mask = (packed >> shift) & 0xFFU;

                // Load sign vectors from pre‑computed table
                float32x4_t s_low  = vld1q_f32(&sign_table[mask][0]);
                float32x4_t s_high = vld1q_f32(&sign_table[mask][4]);

                // Accumulate
                c_low  = vaddq_f32(c_low,  vmulq_f32(a_vec, s_low));
                c_high = vaddq_f32(c_high, vmulq_f32(a_vec, s_high));

                // Store back
                vst1q_f32(&c_row[j], c_low);
                vst1q_f32(&c_row[j + 4], c_high);

                shift += 8;
                if (shift >= 32) {
                    shift -= 32;
                    ++b_index;
                    packed = b_row[b_index];
                }
            }

            // Handle remaining columns that do not fit in an 8‑column block
            for (; j < K; ++j) {
                unsigned bit = (packed >> shift) & 1U;
                float sign = bit ? 1.0f : -1.0f;
                c_row[j] += a_val * sign;

                shift = (shift + 1) & 31U;
                if (shift == 0) {
                    ++b_index;
                    packed = b_row[b_index];
                }
            }
        }
    }
}
