void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    // Using the pattern confirmed to be the fastest (matmul_block_j_vectorized = 72.2086ms)
    // This pattern blocks the output dimension j into chunks of 4 (J_BLOCK_SIZE).
    
    size_t K_ints = K / 32;
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Loop over blocks of 4 columns (j_block_start up to K)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Initialize 4 accumulators for the 4 target columns
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension)
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Calculate contribution for all 4 target columns in this block
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Boundary check: Stop if the target column index j is outside the matrix bounds K
                    if (j >= K) break; 

                    // 1. Determine necessary indices for B[p][j]
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    // 2. Access B: B is row-major. Row p. Word j_word.
                    // B index calculation: p * K_ints + j_word
                    uint32_t packed = B[p * K_ints + j_word];
                    
                    // 3. Extract the specific bit
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
}