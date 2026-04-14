#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdlib> // for malloc/free

// Calculates Matrix C = Matrix A * Matrix B (Optimized using 0/1 bits and row sum)
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

    // Compute C2[i,j] = sum_p A[i,p] * B_bit[p,j] (where B_bit is 0 or 1)
    for (size_t i = 0; i < M; ++i) {
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* B_row = B + p * K_ints;
            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t packed = B_row[chunk_idx];
                size_t base_j = chunk_idx * 32;
                // Unroll by 8
                for (size_t b = 0; b < 32; b += 8) {
                    uint32_t bits0 = (packed >> (b+0)) & 1; C[i * K + base_j + b+0] += a_val * bits0;
                    uint32_t bits1 = (packed >> (b+1)) & 1; C[i * K + base_j + b+1] += a_val * bits1;
                    uint32_t bits2 = (packed >> (b+2)) & 1; C[i * K + base_j + b+2] += a_val * bits2;
                    uint32_t bits3 = (packed >> (b+3)) & 1; C[i * K + base_j + b+3] += a_val * bits3;
                    uint32_t bits4 = (packed >> (b+4)) & 1; C[i * K + base_j + b+4] += a_val * bits4;
                    uint32_t bits5 = (packed >> (b+5)) & 1; C[i * K + base_j + b+5] += a_val * bits5;
                    uint32_t bits6 = (packed >> (b+6)) & 1; C[i * K + base_j + b+6] += a_val * bits6;
                    uint32_t bits7 = (packed >> (b+7)) & 1; C[i * K + base_j + b+7] += a_val * bits7;
                }
            }
        }
    }

    // Final conversion: C[i,j] = 2 * C2[i,j] - row_sum[i]
    for (size_t i = 0; i < M; ++i) {
        float rsum = row_sum[i];
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 2.0f * C[i * K + j] - rsum;
        }
    }

    std::free(row_sum);
}