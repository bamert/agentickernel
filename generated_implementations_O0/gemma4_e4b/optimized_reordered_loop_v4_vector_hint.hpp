void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* C_row_i = C + (i * K);
        // Initialize C row i to zero
        for (size_t j = 0; j < K; ++j) {
            C_row_i[j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {     // Summation index p
            float a_val = A[i * K + p];
            
            // Vector approach: Process contribution of A[i, p] across all j in blocks of 32
            for (size_t j_block = 0; j_block < K_ints; ++j_block) { 
                
                const uint32_t packed = B[p * K_ints + j_block];
                
                // 1. Load the 32 signs (this requires careful bit manipulation to map to floats)
                float sign_vector[32];
                for (size_t offset = 0; offset < 32; ++offset) {
                    size_t j = j_block * 32 + offset;
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    sign_vector[offset] = (bit == 1) ? 1.0f : -1.0f;
                }
                
                // 2. Calculate the contribution: (a_val * sign_vector)
                // Since I cannot use intrinsics directly here without seeing the function signature match the intrinsic usage, 
                // I will vectorize the scalar accumulation step using basic vector pointer offsetting, 
                // which often guides the compiler towards auto-vectorization better than the previous loops.
                
                // Use a pointer arithmetic pattern to maximize the chance of compiler vectorization
                float* C_offset_ptr = C_row_i + j_block * 32;
                float* a_val_broadcast = const_cast<float*>(A); // Necessary hack if not using intrinsics directly

                for (size_t offset = 0; offset < 32; ++offset) {
                    float contribution = a_val * sign_vector[offset];
                    C_offset_ptr[offset] += contribution;
                }
            }
        }
    }
}