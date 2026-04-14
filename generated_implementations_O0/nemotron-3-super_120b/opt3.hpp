#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized with loop unrolling)
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t p = 0; p < K; ++p) {     // Inner dimension
            float a_val = A[i * K + p];
            const uint32_t* B_row = B + p * K_ints;
            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t packed = B_row[chunk_idx];
                size_t base_j = chunk_idx * 32;
                // Unroll by 8
                for (size_t b = 0; b < 32; b += 8) {
                    uint32_t bits0 = (packed >> (b+0)) & 1; float sign0 = bits0 ? 1.0f : -1.0f; C[i * K + base_j + b+0] += a_val * sign0;
                    uint32_t bits1 = (packed >> (b+1)) & 1; float sign1 = bits1 ? 1.0f : -1.0f; C[i * K + base_j + b+1] += a_val * sign1;
                    uint32_t bits2 = (packed >> (b+2)) & 1; float sign2 = bits2 ? 1.0f : -1.0f; C[i * K + base_j + b+2] += a_val * sign2;
                    uint32_t bits3 = (packed >> (b+3)) & 1; float sign3 = bits3 ? 1.0f : -1.0f; C[i * K + base_j + b+3] += a_val * sign3;
                    uint32_t bits4 = (packed >> (b+4)) & 1; float sign4 = bits4 ? 1.0f : -1.0f; C[i * K + base_j + b+4] += a_val * sign4;
                    uint32_t bits5 = (packed >> (b+5)) & 1; float sign5 = bits5 ? 1.0f : -1.0f; C[i * K + base_j + b+5] += a_val * sign5;
                    uint32_t bits6 = (packed >> (b+6)) & 1; float sign6 = bits6 ? 1.0f : -1.0f; C[i * K + base_j + b+6] += a_val * sign6;
                    uint32_t bits7 = (packed >> (b+7)) & 1; float sign7 = bits7 ? 1.0f : -1.0f; C[i * K + base_j + b+7] += a_val * sign7;
                }
            }
        }
    }
}