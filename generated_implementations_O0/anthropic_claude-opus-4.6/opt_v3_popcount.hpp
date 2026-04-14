
#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// Approach: For each output C[i][j] = 2 * sum(A[i][p] where B_bit(p,j)=1) - sum(A[i][p])
// 
// We precompute row_sum = sum of A's row.
// Then accumulate only where bits are 1.
// C[i][j] = 2 * positive_sum[j] - row_sum
//
// Loop structure: for each row i of A:
//   1. Compute row_sum
//   2. Zero C_row (will store positive_sum)
//   3. For each p: scatter-add A[i][p] to C positions where B[p][j] bit is 1
//   4. C[i][j] = 2 * C[i][j] - row_sum

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;

        // Compute row sum using NEON
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (size_t p = 0; p < K; p += 4) {
            sum_vec = vaddq_f32(sum_vec, vld1q_f32(A_row + p));
        }
        float row_sum = vaddvq_f32(sum_vec);

        // Zero C_row (will accumulate positive_sum)
        for (size_t j = 0; j < K; j += 4) {
            vst1q_f32(C_row + j, vdupq_n_f32(0.0f));
        }

        // For each element in shared dimension
        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            float32x4_t va = vdupq_n_f32(a_val);
            const uint32_t* B_row_ptr = B + p * K_ints;

            // For each group of 32 output columns
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row_ptr[g];
                float* C_out = C_row + g * 32;

                // For each set of 4 bits, conditionally add a_val
                // Use NEON bit manipulation: expand bits to float mask
                for (int b = 0; b < 32; b += 4) {
                    uint32_t bits = (packed >> b);
                    
                    // Create mask: lane k has all 1s if bit k is set, else all 0s
                    // bits & 1, bits & 2, bits & 4, bits & 8
                    uint32x4_t bit_vals = {bits & 1u, (bits >> 1) & 1u, (bits >> 2) & 1u, (bits >> 3) & 1u};
                    // Convert to mask: 0 -> 0x00000000, 1 -> 0xFFFFFFFF
                    uint32x4_t mask = vceqq_u32(bit_vals, vdupq_n_u32(1));
                    
                    // Conditional add: only add where bit is 1
                    float32x4_t masked_a = vreinterpretq_f32_u32(
                        vandq_u32(vreinterpretq_u32_f32(va), mask)
                    );
                    
                    float32x4_t c_vec = vld1q_f32(C_out + b);
                    c_vec = vaddq_f32(c_vec, masked_a);
                    vst1q_f32(C_out + b, c_vec);
                }
            }
        }

        // Final transform: C[i][j] = 2 * C[i][j] - row_sum
        float32x4_t vrow_sum = vdupq_n_f32(row_sum);
        float32x4_t vtwo = vdupq_n_f32(2.0f);
        for (size_t j = 0; j < K; j += 4) {
            float32x4_t c_vec = vld1q_f32(C_row + j);
            c_vec = vfmsq_f32(vmulq_f32(vtwo, c_vec), vdupq_n_f32(1.0f), vrow_sum);
            // Actually simpler: 2*c - row_sum
            // = fma(2, c, -row_sum) but let me just do it directly
            vst1q_f32(C_row + j, vmlsq_f32(vmulq_f32(vtwo, c_vec), vdupq_n_f32(1.0f), vrow_sum));
        }
    }
}
