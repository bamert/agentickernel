// Optimized Matrix Multiplication
// Approach: Use the arithmetic identity:
// sum(A[i][p] * sign[p][j]) where sign is 1 if bit is 1, else -1.
// Let bits_set = sum(A[i][p] where bit is 1)
// Let bits_unset = sum(A[i][p] where bit is 0)
// Total sum = bits_set - bits_unset
// We know: bits_set + bits_unset = sum(A[i][p] for all p)
// So: bits_unset = sum(A[i][p]) - bits_set
// Result = bits_set - (sum(A[i][p]) - bits_set) = 2 * bits_set - sum(A[i][p])

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        // Calculate total sum of row A once
        float sumA = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            sumA += rowA[p];
        }

        // Initialize row C with the base value: -sumA
        // because result = 2 * bits_set - sumA
        float base_val = -sumA;
        for (size_t j = 0; j < K; ++j) {
            rowC[j] = base_val;
        }

        // Add 2 * A[i][p] to the columns where bit is 1
        for (size_t p = 0; *p < K; ++p) {
            const float dualA = 2.0f * rowA[p];
            const uint32_t* rowB = &B[p * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits = rowB[chunk_idx];
                if (bits == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];
                
                // Process all set bits in this 32-bit word
                while (bits != 0) {
                    int bit_pos = __builtin_ctz(bits);
                    target_C[bit_pos] += dualA;
                    bits &= (bits - 1); // Clear the lowest set bit
                }
            }
        }
    }
}
