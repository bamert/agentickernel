void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    // Final attempt, reiterating the pattern found to be the most effective:
    // i (Row) -> p (Summation/K) -> j_block (Column block) -> offset (Within block)
    // This order ensures that A[i, p] is repeatedly used to update C[i, j] for all j, promoting temporal and spatial locality.

    for (size_t i = 0; i < M; ++i) {
        // 1. Initialization (Essential step for correct accumulation)
        float* C_base = C + (i * K);
        for (size_t j = 0; j < K; ++j) {
            C_base[j] = 0.0f;
        }

        // 2. Summation over the intermediate dimension p
        for (size_t p = 0; p < K; ++p) {     // A[i, p] is constant 'a_val' for the contribution
            float a_val = A[i * K + p];
            
            // Process contributions from all 32-bit blocks of columns j
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { 
                
                const uint32_t packed = B[p * K_ints + j_block];
                
                // Process the 32 consecutive values of j
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