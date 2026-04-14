void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // We process columns in blocks of 4 (J_BLOCK_SIZE = 4) to compute four dot products simultaneously.
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Loop over blocks of 4 columns (j_block_start up to K)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Accumulators for the 4 target columns (j_block_start to j_block_start + 3)
            // These hold the partial sum for the current row i across 4 columns.
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension)
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Calculate contribution for all 4 target columns (t=0 to 3)
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Boundary check: Stop if the target column index j exceeds K-1
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
                    
                    // Accumulate A[i, p] * sign into the t-th sum container
                    sums[t] += a_val * sign;
                }
            }
            
            // Write out results for j_block_start to j_block_start + 3
            for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                size_t j = j_block_start + t;
                // Only write if the index hasn't been handled by the boundary break
                if (j < K) {
                    C[i * K + j] = sums[t];
                }
            }
        }
        
        // Final cleanup loop for the remainder of columns (K % 4 != 0).
        // Since the main loop handles j_block_start + t, we only need this if the last block was partial.
        // The previous loop ensures all columns j < K are covered by j_block_start + t, 
        // so the only necessary fallback is if K is not divisible by 4.
        // To avoid duplicating logic, we rely on the incremental nature above, but explicitly check for coverage.
        
        // The logic for checking the remainder in the previous iteration was complex. 
        // A simple final loop covering all indices j is the most robust fallback.
        // Since the main loop already ran for j=0 up to j=K-1 if K was a multiple of 4, 
        // we only need to run this fallback if the last iteration of j_block_start < K 
        // and there were remaining columns that couldn't be fully covered by the block structure.
        
        // However, if the block structure passes j_block_start up to K - (K % 4) or K - 1, 
        // the final loop block already covers everything up to K - 1 correctly.
        // We must ensure that the logic above handles the remainder K-1 correctly.
        // Let's ensure the last column after the loop finishes is covered without redundancy.
        
        // Since the main loop structure handles all j < K correctly by checking 'if (j >= K) break;',
        // no separate remainder handling loop should be necessary if the main loop logic is correct.
    }
}