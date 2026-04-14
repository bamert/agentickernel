// Optimized Matrix Multiplication
// Approach: Using the identity C[i][j] = 2.0f * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p] for all p)
// This implementation avoids the heavy bit extraction/shifting inside the inner loop.
// We iterate over set bits using __builtin_ctz for speed.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        // Precalculate the sum of elements in the current row of A
        float sumA = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            sumA += rowA[p];
        }

        // Initialize row C[i][j] with the negative sum of row A
        // This covers the case where the bit is 0 (sign = -1.0f)
        // result = 2 * (sum of A where bit is 1) - sumA
        float base_val = -sumA;
        for (size_t j = 0; j < K; ++j) {
            rowC[j] = base_val;
        }

        // Now add 2.0f * A[i][p] for every bit that is set to 1 in B
        for (size_t p = 0; p < K; ++p) {
            const float dualA = 2.0f * rowA[p];
            const uint32_t* rowB = &B[p * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits = rowB[chunk_idx];
                if (bits == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];

                // Efficient bit scanning
                while (bits != 0) {
                    int bit_pos = __builtin_ctz(bits);
                    target_C[bit_pos] += dualA;
                    bits &= (bits - 1); // Clear the lowest set bit
                }
            }
        }
    }
}
