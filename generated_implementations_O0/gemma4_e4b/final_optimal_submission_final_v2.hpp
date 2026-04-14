void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    // Providing the same loop structure one last time.
    // Conclusion: The i -> p -> j_block -> offset order is the most effective high-level optimization
    // achievable without hardware intrinsics, as it capitalizes on minimizing pointer/index arithmetic
    // overhead while adhering to the accumulation pattern.
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // 1. Initialization
        float* C_base = C + (i * K);
        for (size_t j = 0; j < K; ++j) {
            C_base[j] = 0.0f;
        }

        // 2. Summation over p
        for (size_t p = 0; p < K; ++p) {     
            float a_val = A[i * K + p];
            
            // Process all 32-bit blocks of columns j
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