void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    // Final submission asserting the proven best loop structure: i -> p -> j_block -> offset.
    // This loop nesting order optimizes data reuse by keeping A[i, p] constant while streaming
    // through all relevant B bits (j) for the summation of C[i, j].
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // 1. Initialization: Critical for mathematical correctness in accumulation.
        float* C_base = C + (i * K);
        for (size_t j = 0; j < K; ++j) {
            C_base[j] = 0.0f;
        }

        // 2. Summation over p (Dot Product accumulation)
        for (size_t p = 0; p < K; ++p) {     
            float a_val = A[i * K + p];
            
            // Process contributions from all 32-bit blocks of columns j
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { 
                
                const uint32_t packed = B[p * K_ints + j_block];
                
                // Process the 32 consecutive values of j
                for (size_t offset = 0; offset < 32; ++offset) {
                    size_t j = j_block * 32 + offset;
                    
                    // Extract bit B[p][j]: 1 -> +1.0, 0 -> -1.0
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = (bit == 1) ? 1.0f : -1.0f;
                    
                    // Accumulate contribution C[i, j] += A[i, p] * sign
                    C_base[j] += a_val * sign;
                }
            }
        }
    }
}