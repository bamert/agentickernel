void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Loop setup for j: Process 4 columns simultaneously
    size_t j_words_block = 4;
    size_t num_j_words_blocks = (K + j_words_block - 1) / j_words_block;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += j_words_block) {
            // Indices for the 4 output columns we are calculating: j, j+1, j+2, j+3
            size_t j_offsets[4];
            for (int t = 0; t < 4; ++t) {
                j_offsets[t] = j_block_start + t;
            }

            // We must store the sum for 4 columns simultaneously, 
            // this implies calculating 4 separate dot products (sum_0, sum_1, sum_2, sum_3)
            // We use an array of 4 float sums, one for each column j
            float sums[4] = {0.0f}; 
            
            // Optimization: Process summation over p in blocks of 32
            for (size_t k = 0; k < K / 32; ++k) {
                
                // Calculate contribution for the block p_start to p_start+31
                
                // Process 32 sequential p values within the block k
                for (size_t b = 0; b < 32; ++b) {
                    size_t p = k * 32 + b;
                    float a_val = A[i * K + p];
                    
                    // Calculate contribution for all 4 target columns (j_offsets[t])
                    for (int t = 0; t < 4; ++t) {
                        size_t j = j_offsets[t];
                        
                        // Calculate B[p][j]: B row p, word j_word, bit j_bit
                        size_t j_word = j / 32;
                        size_t j_bit = j % 32;
                        
                        // Access B: B is row-major. Row p. Word j_word.
                        uint32_t packed = B[p * (K/32) + j_word];
                        
                        // Extract the specific bit
                        uint32_t bit = (packed >> j_bit) & 1;
                        
                        float sign = bit ? 1.0f : -1.0f;
                        
                        // Accumulate A[i, p] * sign into the t-th sum container
                        sums[t] += a_val * sign;
                    }
                }
            }
            
            // Write out results for j_block_start to j_block_start + 3
            for (int t = 0; t < 4; ++t) {
                C[i * K + j_offsets[t]] = sums[t];
                // IMPORTANT: If j_offsets[t] >= K, we must stop writing. 
                // But since we calculated j_block_start < K, and we use +3, we must cap it.
                if (j_offsets[t] >= K) break;
            }
        }
        
        // Handle remaining columns if K is not a multiple of 4 (unlikely based on constraints, but safe practice)
        for (size_t j = (K / 4) * 4; j < K; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < K; ++p) {
                size_t j_word = j / 32;
                size_t j_bit = j % 32;
                
                uint32_t packed = B[p * (K/32) + j_word];
                uint32_t bit = (packed >> j_bit) & 1;
                
                float sign = bit ? 1.0f : -1.0f;
                sum += A[i * K + p] * sign;
            }
            C[i * K + j] = sum;
        }
    }
}