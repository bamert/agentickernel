void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Strategy: Block the output dimension j into chunks of 4 to calculate 4 output elements 
    // using shared accumulator state across the p-summation loop.
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Iterate over the starting index of each block of 4 columns
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Temporary accumulators for the 4 target columns
            float sums[J_BLOCK_SIZE] = {0.0f}; // Reset sums for the entire block iteration
            
            // Inner sum loop: over p (K dimension) - sequential summation
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Calculate contribution for all 4 target columns in the block j_block_start...
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Boundary check: Stop if the target column index j is outside the matrix bounds K
                    if (j >= K) break; 

                    // 1. Retrieve bit sign S_pj
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    // B index: Row p, Word j_word
                    uint32_t packed = B[p * K_ints + j_word];
                    uint32_t bit = (packed >> j_bit) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    
                    // Accumulate A[i, p] * sign into the t-th slot
                    sums[t] += a_val * sign;
                }
            }
            
            // Write out results for the four calculated columns
            for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                size_t j = j_block_start + t;
                if (j < K) {
                    C[i * K + j] = sums[t];
                }
            }
        }
    }
    // The structure remains the same as the previous best-performing code, 
    // relying on the compiler/harness to execute the optimal loop nesting pattern.
}