void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // Outer loop: Row of A (and C)
        
        // Optimization strategy: Calculate C[i][j] = sum_p (A[i][p] * S[p][j])
        // We iterate over p and, for each p, we update all K elements of row i in C.
        
        size_t p = 0;
        size_t p_limit = K - (K % 4);

        // Unroll summation dimension p by 4 (if possible)
        for (; p <= p_limit - 4; p += 4) {
            // Process contribution from A[i][p] and A[i][p+1]...
            
            // Calculate contribution from p
            float a_val_p = A[i * K + p];
            // Accumulate for all j in blocks of 4 (unrolled over j)
            size_t j = 0;
            size_t j_limit_4 = K - (K % 4);
            for (; j <= j_limit_4 - 4; j += 4) {
                // j
                {
                    size_t j_block = j;
                    uint32_t packed = B[p * K_ints + (j_block / 32)];
                    uint32_t bit = (packed >> (j_block % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val_p * sign;
                }
                // j+1
                {
                    size_t j_block = j + 1;
                    uint32_t packed = B[p * K_ints + ((j + 1) / 32)];
                    uint32_t bit = (packed >> ((j + 1) % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val_p * sign;
                }
                // j+2
                {
                    size_t j_block = j + 2;
                    uint32_t packed = B[p * K_ints + ((j + 2) / 32)];
                    uint32_t bit = (packed >> ((j + 2) % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val_p * sign;
                }
                // j+3
                {
                    size_t j_block = j + 3;
                    uint32_t packed = B[p * K_ints + ((j + 3) / 32)];
                    uint32_t bit = (packed >> ((j + 3) % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val_p * sign;
                }
            }
            // Remainder for j (linear scan)
            for (; j < K; ++j) {
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val_p * sign;
            }

            // Simplified: Since we calculate the full dot product sum for each j separately in the outer loops, 
            // it is better to stick to the most successful structure: dot product accumulation.
            // Reverting to the structure of matmul_p_j_unrolled_j which worked best.
        }


        // Re-implementing the structure that yielded the best result (matmul_p_j_unrolled_j)
        // Outer loop: i
        // Intermediate loop: p (summation index)
        // Inner loop: j (output column), unrolled by 4
        
        for (size_t p = 0; p < K; ++p) { // Summation index p (re-initializing p counter)
            float a_val = A[i * K + p];
            
            size_t j = 0;
            size_t j_limit_4 = K - (K % 4);

            // Unroll j loop by 4
            for (; j <= j_limit_4 - 4; j += 4) {
                // j
                {
                    size_t j_block = j;
                    uint32_t packed = B[p * K_ints + (j_block / 32)];
                    uint32_t bit = (packed >> (j_block % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
                // j+1
                {
                    size_t j_block = j + 1;
                    uint32_t packed = B[p * K_ints + ((j + 1) / 32)];
                    uint32_t bit = (packed >> ((j + 1) % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
                // j+2
                {
                    size_t j_block = j + 2;
                    uint32_t packed = B[p * K_ints + ((j + 2) / 32)];
                    uint32_t bit = (packed >> ((j + 2) % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
                // j+3
                {
                    size_t j_block = j + 3;
                    uint32_t packed = B[p * K_ints + ((j + 3) / 32)];
                    uint32_t bit = (packed >> ((j + 3) % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j_block] += a_val * sign;
                }
            }

            // Handle remaining iterations j
            for (; j < K; ++j) {
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val * sign;
            }
        }

    }
}