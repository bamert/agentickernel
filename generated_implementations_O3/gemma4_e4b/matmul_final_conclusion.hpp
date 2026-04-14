void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // This structure represents the pattern that has consistently generated the best results
    // through multiple passes, suggesting the inherent complexity of the B matrix access 
    // limits the benefit of loop unrolling/blocking beyond the simple O(M*K^2) structure.
    for (size_t i = 0; i < M; ++i) {         // Row i loop
        // 1. Initialization (Safest to keep for correctness check)
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Summation loop p
            float a_val = A[i * K + p];
            
            // Inner loop accumulates the entire dot product for C[i, j] summation
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