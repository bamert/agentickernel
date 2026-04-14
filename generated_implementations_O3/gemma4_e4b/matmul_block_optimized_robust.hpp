void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // We process columns in blocks of 4 for efficiency (J blocking).
    const size_t J_BLOCK_SIZE = 4;
    size_t num_j_blocks = (K + J_BLOCK_SIZE - 1) / J_BLOCK_SIZE;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Loop over blocks of 4 columns (j_block_start up to K)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Initialize 4 accumulators for the 4 target columns j, j+1, j+2, j+3
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension), processing 32 elements at a time (P blocking)
            const size_t P_BLOCK_SIZE = 32;
            for (size_t p_start = 0; p_start < K; p_start += P_BLOCK_SIZE) {
                
                // Process 32 consecutive p values within the block (p_start to p_start+31)
                for (size_t p_offset = 0; p_offset < P_BLOCK_SIZE && (p_start + p_offset) < K; ++p_offset) {
                    size_t p = p_start + p_offset;
                    float a_val = A[i * K + p];
                    
                    // Calculate contribution for all 4 target columns (j_offsets[t])
                    for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                        size_t j = j_block_start + t;
                        
                        // Check if this column index is out of bounds
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
            }
            
            // Write out results for j_block_start to j_block_start + 3
            for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                size_t j = j_block_start + t;
                // Only write if the index hasn't overflowed K
                if (j < K) {
                    C[i * K + j] = sums[t];
                }
            }
        }
        
        // Final cleanup loop for the very last column if K % 4 != 0
        // This handles the remainder columns that were not covered by the J_BLOCK_SIZE loop
        for (size_t j = (K / J_BLOCK_SIZE) * J_BLOCK_SIZE; j < K; ++j) {
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