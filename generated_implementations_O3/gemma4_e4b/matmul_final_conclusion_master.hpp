void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Conclusion: The simple, explicit triple-loop structure (i, p, j) is the most reliable 
    // and fastest pattern observed across all variants of the core loop.
    for (size_t i = 0; i < M; ++i) {         // Row i loop (M)
        // Initialization
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Summation loop p (K)
            float a_val = A[i * K + p];
            
            // Linear scan over j (K)
            for (size_t j = 0; j < K; ++j) {
                // Calculate B[p][j]
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                
                float sign = (bit != 0) ? 1.0f : -1.0f;
                
                // Accumulate into C[i][j]
                C[i * K + j] += a_val * sign;
            }
        }
    }
}