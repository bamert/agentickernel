void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Initialize C row i to zero
        float* C_row_i = C + (i * K);
        for (size_t j = 0; j < K; ++j) {
            C_row_i[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {     // Summation index p (A[i, p] is fixed)
            float a_val = A[i * K + p];
            
            // Process all 32 blocks of j columns for this A[i, p] contribution
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { 
                
                // 1. Load the 32-bit word representing B row p, chunk j_block
                const uint32_t packed = B[p * K_ints + j_block];
                
                // Temporary buffer to hold the 32 sign floats derived from the packed word
                float sign_vector[32];
                
                // 2. Unpack the 32 bits into 32 sign floats
                for (size_t offset = 0; offset < 32; ++offset) {
                    // j = j_block*32 + offset
                    size_t j = j_block * 32 + offset;
                    
                    // Extract the bit at B[p][j]
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    
                    // Determine sign: 1 -> 1.0f, 0 -> -1.0f
                    sign_vector[offset] = (bit == 1) ? 1.0f : -1.0f;
                }
                
                // 3. Accumulate: C[i, j] += a_val * sign_vector[offset]
                // Since we are staying within the structure found to be fast, we accumulate element-wise 
                // to avoid complex intrinsic manipulation for sign conversion on 32 lanes.
                for (size_t offset = 0; offset < 32; ++offset) {
                    size_t j = j_block * 32 + offset;
                    C_row_i[j] += a_val * sign_vector[offset];
                }
            }
        }
    }
}