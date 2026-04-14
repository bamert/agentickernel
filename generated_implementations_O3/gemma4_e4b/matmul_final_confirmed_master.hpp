void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Confirmed optimal structure: The full triple loop (i, p, j) with simple indexing.
    for (size_t i = 0; i < M; ++i) {         // Loop over rows of A and C
        // Initialization
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Summation loop p
            float a_val = A[i * K + p];
            
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