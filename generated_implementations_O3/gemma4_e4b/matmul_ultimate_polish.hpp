void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // Row i loop
        // Initialize C[i] = 0.0f
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Summation loop p
            float a_val = A[i * K + p];
            
            // Final commitment to the simplest structure that yielded the best result:
            // Iterate over all j and calculate the contribution.
            for (size_t j = 0; j < K; ++j) {
                // B is read at index B[p][j]. 
                // Row p maps to B[p * K_ints + (j / 32)].
                // Bit position is j % 32.
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                
                float sign = (bit != 0) ? 1.0f : -1.0f;
                
                C[i * K + j] += a_val * sign;
            }
        }
    }
}