void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    // Optimization Strategy applied: Looping order reordering (i-p-j_block-offset)
    // and explicit initialization of output C row.
    // This structure is robust, addresses the loop dependency issues, and forces
    // the memory accesses to be sequential for cache efficiency, which the prior versions struggled to maintain.

    for (size_t i = 0; i < M; ++i) {
        // Step 1: Initialization of C[i, :] to zero. This is essential for correctness due to accumulation.
        float* C_base = C + (i * K);
        for (size_t j = 0; j < K; ++j) {
            C_base[j] = 0.0f;
        }

        // Step 2: Calculate Summation over p (Middle index)
        for (size_t p = 0; p < K; ++p) {     // A[i, p] is constant for the inner block
            float a_val = A[i * K + p];
            
            // Step 3: Process contributions from all 32-bit blocks of columns j
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { 
                
                const uint32_t packed = B[p * K_ints + j_block];
                
                // Step 4: Process the 32 components within the block
                for (size_t offset = 0; offset < 32; ++offset) {
                    size_t j = j_block * 32 + offset;
                    
                    // Extract bit B[p][j]
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = (bit == 1) ? 1.0f : -1.0f;
                    
                    // Accumulate contribution C[i, j] += A[i, p] * sign
                    C_base[j] += a_val * sign;
                }
            }
        }
    }
}