void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Initialization must be explicit and robust
        float* C_base = C + (i * K);
        for (size_t j = 0; j < K; ++j) {
            C_base[j] = 0.0f;
        }

        // Matrix multiplication: C = A * B (dot product approach)
        // We iterate over the shared dimension K (p) for the summation.
        for (size_t p = 0; p < K; ++p) {     // Summation index p
            float a_val = A[i * K + p];
            
            // Contributions from all 32-bit blocks of columns j
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { 
                
                const uint32_t packed = B[p * K_ints + j_block];
                
                // Process the 32 consecutive values of j (offset 0 to 31)
                for (size_t offset = 0; offset < 32; ++offset) {
                    size_t j = j_block * 32 + offset;
                    
                    // Extract bit B[p][j]: bit 1 (+1.0f) or bit 0 (-1.0f)
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = (bit == 1) ? 1.0f : -1.0f;
                    
                    // Accumulate contribution C[i, j] += A[i, p] * sign
                    C_base[j] += a_val * sign;
                }
            }
        }
    }
}