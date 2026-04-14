void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A (and C)
        // Initialize C[i] = 0.0f
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Summation index p
            float a_val = A[i * K + p];
            
            // Process j in blocks of 32 (Leveraging K multiple of 32 constraint)
            size_t j_start = 0;
            
            // Looping over 32-wide blocks of j
            for (; j_start <= K - 32; j_start += 32) {
                // Process 32 columns simultaneously
                for (size_t j_offset = 0; j_offset < 32; ++j_offset) {
                    size_t j = j_start + j_offset;
                    
                    // Extract bit at B[p][j]
                    uint32_t packed = B[p * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    
                    // Accumulate into C[i][j]
                    C[i * K + j] += a_val * sign;
                }
            }
            
            // Handle remaining iterations j (linear scan)
            for (size_t j = j_start; j < K; ++j) {
                // Extract bit at B[p][j]
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val * sign;
            }
        }
    }
}