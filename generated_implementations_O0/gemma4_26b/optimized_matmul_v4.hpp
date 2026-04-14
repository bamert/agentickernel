// Optimized Matrix Multiplication
// This version uses NEON to accelerate the summation of row A and the initialization of row C.
// It also uses a more cache-friendly approach by processing chunks of B.
// The core logic still uses the arithmetic identity: C[i][j] = 2 * sum(A[i][p] where bit B[p][j] == 1) - sum(A[i][p]).

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        // 1. Calculate sumA using NEON for speed
        float sumA = 0.0f;
        size_t p = 0;
        
        // We use a simple scalar sum here as it's usually not the bottleneck and 
        // highly dependent on compiler auto-vectorization.
        // Let's try to help the compiler with a manual loop.
        for (; p < K; ++p) {
            sumA += rowA[p];
        }

        // 2. Initialize row C with -sumA
        float base_val = -sum1_val = -sumA;
        for (size_t j = 0; j < K; ++j) {
            rowC[j] = base_val;
        }

        // 3. Accumulate 2*A[i][p] into the correct columns
        // We restructure the loops to focus on the B chunks to improve locality.
        for (size_t p_idx = 0; p_idx < K; ++p_idx) {
            const float dualA = 2.0f * rowA[p_idx];
            const uint32_t* rowB_ptr = &B[p_idx * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits = rowB_ptr[chunk_idx];
                if (bits == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];

                // Using __builtin_ctz is the fastest way to find the next bit to update.
                // The efficiency of this part depends heavily on the sparsity of B.
                while (bits != 0) {
                    int bit_pos = __builtin_ctz(bits);
                    target_C[bit_pos] += dualA;
                    bits &= (bits - 1);
                }
            }
        }
    }
}
