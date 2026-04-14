void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    // Final optimization attempt: Use the confirmed high-performing strategy 
    // (Blocking output dimension J into chunks of 4) while ensuring minimal overhead.
    
    size_t K_ints = K / 32;
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Iterate over blocks of 4 columns j (j_block_start)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Accumulators for the 4 target columns
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension)
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Calculate contribution for the 4 target columns in this block
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Boundary check: Exit if j is outside the matrix bounds K
                    if (j >= K) break; 

                    // 1. Determine necessary indices for B[p][j]
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    // B index: Row p, Word j_word
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