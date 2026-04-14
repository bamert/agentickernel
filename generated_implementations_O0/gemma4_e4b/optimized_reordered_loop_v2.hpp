void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Initialize C row i to zero
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {     // Summation index p (A[i, p] is fixed)
            float a_val = A[i * K + p];
            
            // Process all 32 blocks of j columns for this A[i, p] contribution
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { 
                
                // Get the 32-bit word representing B row p, chunk j_block
                uint32_t packed = B[p * K_ints + j_block];
                
                // Process 32 consecutive values of j (j = j_block*32 + offset)
                for (size_t offset = 0; offset < 32; ++offset) {
                    size_t j = j_block * 32 + offset;
                    
                    // Extract the specific bit at B[p][j]
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    
                    // Accumulate contribution: C[i, j] += A[i, p] * Sign(B[p, j])
                    C[i * K + j] += a_val * sign;
                }
            }
        }
    }
}