#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Initialize the row of C to zero (for C2)
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
        float rsum = 0.0f;
        // Process p in chunks of 16
        for (size_t p_start = 0; p_start < K; p_start += 16) {
            // Load 16 a_vals
            float a_val[16];
            for (int ii = 0; ii < 16; ++ii) {
                a_val[ii] = A[i * K + p_start + ii];
                rsum += a_val[ii];
            }

            // Process each chunk (each chunk corresponds to 32 columns)
            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                size_t base_j = chunk_idx * 32;
                // We'll process the 32 columns in groups of 4 using NEON
                for (size_t j = 0; j < 32; j += 4) {
                    // Neon accumulator for 4 columns
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    // For each of the 16 p's
                    for (int ii = 0; ii < 16; ++ii) {
                        uint32_t packed = B[(p_start + ii) * K_ints + chunk_idx];
                        // Extract 4 bits at positions j..j+3
                        uint32_t bits4 = (packed >> j) & 0xF;
                        // Create a vector of 0.0f or a_val[ii] for each bit
                        float32x4_t val = vdupq_n_f32(0.0f);
                        // Set each lane conditionally (branchless)
                        val = vsetq_lane_f32(((bits4 >> 0) & 1) * a_val[ii], val, 0);
                        val = vsetq_lane_f32(((bits4 >> 1) & 1) * a_val[ii], val, 1);
                        val = vsetq_lane_f32(((bits4 >> 2) & 1) * a_val[ii], val, 2);
                        val = vsetq_lane_f32(((bits4 >> 3) & 1) * a_val[ii], val, 3);
                        // Accumulate
                        acc = vaddq_f32(acc, val);
                    }
                    // Store the accumulated 4 values back to C
                    float32x4_t c_vec = vld1q_f32(C + i * K + base_j + j);
                    c_vec = vaddq_f32(c_vec, acc);
                    vst1q_f32(C + i * K + base_j + j, c_vec);
                }
            }
        }
        // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }
}