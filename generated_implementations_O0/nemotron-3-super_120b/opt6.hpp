#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>
#include <cstdlib> // for malloc/free

// Calculates Matrix C = Matrix A * Matrix B (Optimized using 0/1 bits, row sum, and NEON for accumulation)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Allocate temporary array for row sums of A
    float* row_sum = static_cast<float*>(std::malloc(M * sizeof(float)));
    if (!row_sum) return; // allocation failed (unlikely)
    for (size_t i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            sum += A[i * K + p];
        }
        row_sum[i] = sum;
    }

    // Initialize C to zero for accumulation
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }
    }

    // Compute C2[i,j] = sum_p A[i,p] * B_bit[p,j] (where B_bit is 0 or 1)
    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* B_row = B + p * K_ints;
            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t packed = B_row[chunk_idx];
                size_t base_j = chunk_idx * 32;
                // Process 4 floats at a time using NEON
                for (size_t j = 0; j < 32; j += 4) {
                    // Load 4 floats from C
                    float32x4_t c_vec = vld1q_f32(C + i * K + base_j + j);
                    // Extract 4 bits from packed at positions j..j+3
                    uint32_t bits4 = (packed >> j) & 0xF;
                    // Create a vector of values to add: a_val if bit is set, 0.0f if not
                    float32x4_t add_vec = vdupq_n_f32(0.0f);
                    if (bits4 & 0x1) add_vec = vsetq_lane_f32(a_val, add_vec, 0);
                    if (bits4 & 0x2) add_vec = vsetq_lane_f32(a_val, add_vec, 1);
                    if (bits4 & 0x4) add_vec = vsetq_lane_f32(a_val, add_vec, 2);
                    if (bits4 & 0x8) add_vec = vsetq_lane_f32(a_val, add_vec, 3);
                    // Now: c_vec += add_vec
                    c_vec = vaddq_f32(c_vec, add_vec);
                    // Store back
                    vst1q_f32(C + i * K + base_j + j, c_vec);
                }
            }
        }
    }

    // Final conversion: C[i,j] = 2.0f * C2[i,j] - row_sum[i]
    for (size_t i = 0; i < M; ++i) {
        float rsum = row_sum[i];
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }

    std::free(row_sum);
}