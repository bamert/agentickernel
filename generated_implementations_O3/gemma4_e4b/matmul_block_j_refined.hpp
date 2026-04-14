void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Optimization Strategy: Block the calculation over columns j (J_BLOCK_SIZE = 4).
    // This allows us to compute 4 dot products concurrently in the inner accumulators.
    
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Loop over blocks of 4 columns (0, 4, 8, ...)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Accumulators for the 4 target columns (j_block_start to j_block_start + 3)
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension)
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Calculate contribution for all 4 target columns
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Stop if we exceed the matrix boundary K
                    if (j >= K) break; 

                    // 1. Determine necessary indices for B[p][j]
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    // 2. Access B: B is row-major. Row p. Word j_word.
                    uint32_t packed = B[p * K_ints + j_word];
                    
                    // 3. Extract the specific bit
                    uint32_t bit = (packed >> j_bit) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    
                    // Accumulate A[i, p] * sign into the t-th sum container
                    sums[t] += a_val * sign;
                }
            }
            
            // Write out results for j_block_start to j_block_start + 3
            for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                size_t j = j_block_start + t;
                if (j < K) {
                    C[i * K + j] = sums[t];
                }
            }
        }
        
        // Critical refinement: If K is not divisible by 4, the loop above overshoots/misses the end.
        // This handles the perfect remaining columns in a clean single loop after the main block loop finishes.
        // We only execute this if the last block's start was less than K.
        size_t last_j_handled = ((K - 1) / J_BLOCK_SIZE) * J_BLOCK_SIZE;
        if (last_j_handled < K) {
            for (size_t j = last_j_handled; j < K; ++j) {
                float sum = 0.0f;
                for (size_t p = 0; p < K; ++p) {
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    uint32_t packed = B[p * K_ints + j_word];
                    uint32_t bit = (packed >> j_bit) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    sum += A[i * K + p] * sign;
                }
                C[i * K + j] = sum;
            }
        }
    }
}