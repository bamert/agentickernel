
#pragma once
#include <cstdint>
#include <cstddef>

// Key insight: for each output C[i][j], we compute dot(A_row, B_col_sign)
// where B_col_sign[p] = (bit(p,j) ? +1 : -1)
// = 2*bit(p,j) - 1
// So dot = sum_p A[i][p] * (2*bit(p,j) - 1)
//        = 2 * sum_{p where bit=1} A[i][p] - sum_p A[i][p]
//
// Alternative approach: process B row by row (packed), accumulating into C.
// For each row p of A and packed row p of B:
//   For each group of 32 columns j..j+31 of B:
//     packed = B[p * K_ints + g]
//     For each bit b in packed:
//       if bit set: C[i][g*32+b] += A[i][p]
//       else:       C[i][g*32+b] -= A[i][p]
//
// Better: C[i][g*32+b] += a_val if bit set, -= a_val if not
// Which equals: C[i][g*32+b] += (2*bit - 1) * a_val

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Zero output row
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }

        // For each element in A's row (iterating over the shared dimension p)
        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* B_row = B + p * K_ints;

            // Process B[p] row in groups of 32 columns
            for (size_t g = 0; g < K_ints; ++g) {
                uint32_t packed = B_row[g];
                float* C_out = C + i * K + g * 32;

                // Process all 32 bits
                // If all bits are 1, add a_val to all 32; if all 0, subtract from all 32
                // We can check special cases or just iterate
                
                // Fast path: check if all 1s or all 0s
                if (packed == 0xFFFFFFFF) {
                    for (int b = 0; b < 32; ++b) {
                        C_out[b] += a_val;
                    }
                } else if (packed == 0x00000000) {
                    for (int b = 0; b < 32; ++b) {
                        C_out[b] -= a_val;
                    }
                } else {
                    for (int b = 0; b < 32; ++b) {
                        float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                        C_out[b] += a_val * sign;
                    }
                }
            }
        }
    }
}
