void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; ++j) {     // For each column in C
            float sum = 0.0f;
            
            // Precalculate constants for bit access for column j:
            size_t j_word = j / 32;
            size_t j_bit = j % 32;
            
            // Optimization: Process summation over p in blocks of 32
            for (size_t k = 0; k < K / 32; ++k) {
                float block_sum = 0.0f;
                
                // Process 32 sequential p values within the block k
                for (size_t b = 0; b < 32; ++b) {
                    size_t p = k * 32 + b;
                    
                    // Calculate B[p][j]: B row p, word j_word, bit j_bit
                    uint32_t packed = B[p * K_ints + j_word];
                    // Extract the specific bit using shift and mask
                    uint32_t bit = (packed >> j_bit) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    
                    // Accumulate A[i][p] * sign
                    block_sum += A[i * K + p] * sign;
                }
                sum += block_sum;
            }
            
            C[i * K + j] = sum;
        }
    }
}