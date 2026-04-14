void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Optimization Strategy: Block the calculation over columns j (J_BLOCK_SIZE = 4).
    // This computes 4 dot products simultaneously.
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Loop over blocks of 4 columns (j_block_start up to K)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Temporary accumulators for the 4 target columns (j_block_start to j_block_start + 3)
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension) - fully sequential here for robustness
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                // Calculate contribution for all 4 target columns (t=0 to 3)
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Boundary check: If j is out of bounds, stop processing this pair (p, j)
                    if (j >= K) break; 

                    // 1. Determine necessary indices for B[p][j]
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    // 2. Access B: B is row-major. Row p. Word j_word.
                    // B index calculation: p * K_ints + j_word
                    // Check B index bounds explicitly for safety, though usually guaranteed by constraints.
                    if (p * K_ints + j_word >= (K/32) * K_ints) continue; 
                    
                    uint32_t packed = B[p * K_ints + j_word];
                    
                    // 3. Extract the specific bit
                    uint32_t bit = (packed >> j_bit) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    
                    // Accumulate A[i, p] * sign into the t-th sum container
                    sums[t] += a_val * sign;
                }
            }
            
            // Write out results for j_block_start to j_block_start + 3
            write_limit(j_block_start, K, sums, J_BLOCK_SIZE);
        }
    }
}

// Helper function to write results and handle edge cases cleanly
void write_limit(size_t j_start, size_t K, float sums[], size_t count) {
    for (size_t t = 0; t < count; ++t) {
        size_t j = j_start + t;
        if (j < K) {
            // Assuming we are writing to the correct location C[i*K + j] in the calling scope
            // Since this helper is conceptual/for testing simplification, we just write the sum.
            // In the final code, this logic must be placed back into the main loop structure.
            // For the tool, I will incorporate the write back into the main loop body to avoid scope issues.
            // The previous structure was actually cleaner in placing the write back inside the loop.
        }
    }
}
// Since the tool environment doesn't support helper functions defined outside the main function 
// boundary and I must maintain the single function scope, I will consolidate the logic 
// to use the correct boundaries check inside the main function flow, mimicking 
// the structure of matmul_block_j_vectorized, which was the winner. 

// Final attempt structure: Rely only on the confirmed working structure (matmul_block_j_vectorized) 
// and focus on eliminating any potential redundant logic or slow parts.
// I will use the exact code from matmul_block_j_vectorized as it achieved the fastest verified result.
// If I modify it, the risk of introducing bugs outweighs the small potential gain.
// I will stick to this proven winner's implementation.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    if (j >= K) break; 

                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    uint32_t packed = B[p * K_ints + j_word];
                    uint32_t bit = (packed >> j_bit) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    
                    sums[t] += a_val * sign;
                }
            }
            
            for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                size_t j = j_block_start + t;
                if (j < K) {
                    C[i * K + j] = sums[t];
                }
            }
        }
        
        // Redundant check structure (kept for logical consistency if a future change happens):
        size_t last_j_handled_by_blocks = ((K - 1) / J_BLOCK_SIZE) * J_BLOCK_SIZE + (J_BLOCK_SIZE > 0 ? J_BLOCK_SIZE - 1 : 0);
        if (last_j_handled_by_blocks < K) {
            size_t start_j = last_j_handled_by_blocks;
            for (size_t j = start_j; j < K; ++j) {
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