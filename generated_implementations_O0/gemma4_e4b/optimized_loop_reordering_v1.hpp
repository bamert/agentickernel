void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Restructured loop order: Iterate over rows i, then summation index p, then output columns j.
    // This pattern computes C_i = Sum_p (A_i,p * [Signs_p])
    for (size_t i = 0; i < M; ++i) {         // For each row in A/C
        // Initialize C[i, :] to zero (since we are accumulating contributions)
        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {     // Summation index p (Corresponds to A[i, p])
            float a_val = A[i * K + p];
            
            // Calculate contribution of A[i, p] across all j
            uint32_t* B_row_ptr = B + p * K_ints; // Base address for B row p/block
            
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { // Iterate over blocks of 32 columns j
                uint32_t packed = B_row_ptr + j_block;
                
                // Process 32 consecutive values of j (j_block*32 to (j_block+1)*32 - 1)
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