/*
 * Based on successful previous runs, the best performance was achieved by blocking the column dimension (j)
 * into chunks of 4 (J_BLOCK_SIZE=4), which allows us to accumulate 4 dot products simultaneously 
 * using local arrays, minimizing loop overhead while maintaining straightforward arithmetic.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Block columns j into groups of 4
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A
        
        // Iterate over the starting index of each block of 4 columns
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Sums array holds the running total for the 4 columns in this block: [sum_j, sum_{j+1}, sum_{j+2}, sum_{j+3}]
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension)
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Calculate contribution for all 4 target columns in the block
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Boundary check: If the target column index j is outside the matrix bounds K
                    if (j >= K) break; 

                    // 1. Determine necessary indices for B[p][j]
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    // 2. Access B: B is row-major. Row p. Word j_word.
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
        
        // Note: Since the loop structure inherently handles the boundary checks 
        // (the 'if (j >= K) break;' handles partial blocks ending near K), 
        // no explicit remainder loop is necessary if the logic is sound, 
        // as the block stepping handles coverage robustly.
    }
}