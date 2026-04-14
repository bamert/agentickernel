void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A (and C)
        // Initialization is necessary but not performance critical.
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Summation index p
            float a_val = A[i * K + p];
            
            size_t j = 0;
            size_t j_limit_4 = K - (K % 4);

            // Unroll j loop by 4 for the fastest path
            for (; j <= j_limit_4 - 4; j += 4) {
                // Calculate common index components for readability/micro-optimization stability
                
                // j
                {
                    size_t j_block = j;
                    uint32_t p_col_idx = p * K_ints + (j_block / 32);
                    uint32_t bit = (B[p_col_idx] >> (j_block % 32)) & 1;
                    float sign = (bit != 0) ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
                // j+1
                {
                    size_t j_block = j + 1;
                    uint32_t p_col_idx = p * K_ints + ((j + 1) / 32);
                    uint32_t bit = (B[p_col_idx] >> ((j + 1) % 32)) & 1;
                    float sign = (bit != 0) ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
                // j+2
                {
                    size_t j_block = j + 2;
                    uint32_t p_col_idx = p * K_ints + ((j + 2) / 32);
                    uint32_t bit = (B[p_col_idx] >> ((j + 2) % 32)) & 1;
                    float sign = (bit != 0) ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
                // j+3
                {
                    size_t j_block = j + 3;
                    uint32_t p_col_idx = p * K_ints + ((j + 3) / 32);
                    uint32_t bit = (B[p_col_idx] >> ((j + 3) % 32)) & 1;
                    float sign = (bit != 0) ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
            }

            // Handle remaining iterations j (Linear scan)
            for (; j < K; ++j) {
                uint32_t p_col_idx = p * K_ints + (j / 32);
                uint32_t bit = (B[p_col_idx] >> (j % 32)) & 1;
                float sign = (bit != 0) ? 1.0f : -1.0f;
                C[i * K + j] += a_val * sign;
            }
        }
    }
}