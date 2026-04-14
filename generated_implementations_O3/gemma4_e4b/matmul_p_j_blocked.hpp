void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        // Initialize C[i] = 0.0f by assumption, or explicitly if needed, 
        // but since we accumulate here, we must clear C's row first.
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Iterate over the intermediate dimension p
            float a_val = A[i * K + p];
            
            // Now update C[i][j] for all j, using the contribution from A[i][p]
            for (size_t j_start = 0; j_start < K; j_start += 32) {
                
                // Process 32 columns (j_start to j_start + 31)
                for (size_t j_offset = 0; j_offset < 32; ++j_offset) {
                    size_t j = j_start + j_offset;
                    
                    if (j >= K) break;

                    // Extract the specific bit at B[p][j]
                    // p is the row index (B array index)
                    // j defines the column index
                    uint32_t packed = B[p * K_ints + (j / 32)];
                    // Bit position within the integer (j % 32)
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    
                    // Accumulate into C[i][j]
                    C[i * K + j] += a_val * sign;
                }
            }
        }
    }
}