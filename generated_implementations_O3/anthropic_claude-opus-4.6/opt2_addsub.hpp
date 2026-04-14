#pragma once
#include <cstdint>
#include <cstddef>

// Instead of a_val * sign, use: bit=1 -> C += a_val, bit=0 -> C -= a_val
// Also compute rowsum = sum of all A values in the row, then:
// C[i][j] = 2 * (sum of A[i][p] where B[p][j]=1) - rowsum
// This way we only accumulate when bit=1.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Compute row sum
        float rowsum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            rowsum += A[i * K + p];
        }

        // Initialize C[i][j] = -rowsum (assuming all bits are 0, i.e., all -1)
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = -rowsum;
        }

        // Now add 2*a_val for every bit that is 1
        for (size_t p = 0; p < K; ++p) {
            float two_a = 2.0f * A[i * K + p];
            for (size_t jb = 0; jb < K_ints; ++jb) {
                uint32_t packed = B[p * K_ints + jb];
                size_t j_base = jb * 32;
                while (packed) {
                    int bit = __builtin_ctz(packed);
                    C[i * K + j_base + bit] += two_a;
                    packed &= packed - 1; // clear lowest set bit
                }
            }
        }
    }
}
