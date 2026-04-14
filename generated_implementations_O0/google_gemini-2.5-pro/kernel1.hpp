#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
// Optimization Strategy:
// 1. Loop Reordering: The baseline's i,j,p loop order results in poor cache
//    utilization for matrix B. By changing the conceptual loop order to i,p,j,
//    we iterate through A and B with sequential memory access.
// 2. Cache-Friendly Accumulation: The formula C[i] = sum(A[i][p] * B[p]) is
//    used. We compute one full row of C at a time. For each row of C, we
//    iterate through the rows of B, scaled by elements of the corresponding A row,
//    and accumulate the results. This ensures data loaded into cache is reused.
// 3. Chunking B: We process B in 32-bit chunks, which aligns with the
//    storage format and the constraint on K.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // We compute one row of C at a time.
    // C_row[i] = sum over p (A[i][p] * B_row[p])
    for (size_t i = 0; i < M; ++i) {
        // Initialize the output row C[i] to all zeros.
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }

        const float* A_row_i = A + i * K;
        float* C_row_i = C + i * K;

        // For each element A[i][p] in the input row A[i]...
        for (size_t p = 0; p < K; ++p) {
            const float a_val = A_row_i[p];
            const uint32_t* B_row_p = B + p * K_ints;

            // ...add its contribution to the output row C[i].
            // C[i][j] += a_val * B[p][j] for all j
            for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
                uint32_t packed_val = B_row_p[j_chunk];
                
                // Process 32 columns of C for the current A[i][p] and B[p] chunk.
                for (int k = 0; k < 32; ++k) {
                    // Extract the k-th bit from packed_val to determine the sign.
                    if ((packed_val >> k) & 1) {
                        C_row_i[j_chunk * 32 + k] += a_val;
                    } else {
                        C_row_i[j_chunk * 32 + k] -= a_val;
                    }
                }
            }
        }
    }
}
