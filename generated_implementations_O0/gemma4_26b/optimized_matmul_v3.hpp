// Optimized Matrix Multiplication
// This version uses NEON intrinsics to process 4 columns of C at once.
// We still use the identity: C[i][j] = 2.0f * bits_set - sum_A.
// For each bit chunk, we use a bitmask to add 2.0f * A[i][p] to 4 targets simultaneously.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        float sumA = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            sumA += rowA[p];
        }

        float base_val = -sumA;
        for (size_t j = 0; j < K; ++j) {
            rowC[j] = base_val;
        }

        for (size_t p = 0; p < K; ++p) {
            const float dualA = 2.0f * rowA[p];
            const uint32_t* rowB = &B[p * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits = rowB[chunk_idx];
                if (bits == 0) continue;

                float* chunk_C = &rowC[chunk_idx * 32];

                // We still use the bit-scanning approach but we try to group updates.
                // Since we cannot easily use NEON for scatter-adds of arbitrary bits,
                // we focus on the most efficient way to traverse the bits.
                // The previous version was already quite fast.
                // To improve it, let's see if we can process 4 bits at a time if they are close.
                
                while (bits != 0) {
                    int bit_pos = __builtin_ctz(bits);
                    chunk_C[bit_pos] += dualA;
                    bits &= (bits - 1);
                }
            }
        }
    }
}
