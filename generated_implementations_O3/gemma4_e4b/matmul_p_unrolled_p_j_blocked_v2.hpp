void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        // Initialize C[i] = 0.0f
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        // Unroll p loop by 4 (Contribution to C)
        size_t p = 0;
        size_t p_limit = K - (K % 4);
        
        for (; p <= p_limit - 4; p += 4) {
            
            // --- Contribution from p -----
            float a_val_p = A[i * K + p];
            // Calculate contribution for j in blocks of 32
            size_t j_start = 0;
            for (; j_start <= K - 32; j_start += 32) {
                // Process 32 columns (j_start to j_start + 31)
                for (size_t j_offset = 0; j_offset < 32; ++j_offset) {
                    size_t j = j_start + j_offset;
                    
                    uint32_t packed = B[p * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j] += a_val_p * sign;
                }
            }
            // Handle remainder for j (j_start to K-1)
            for (size_t j = j_start; j < K; ++j) {
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val_p * sign;
            }

            // --- Contribution from p+1 -----
            float a_val_p1 = A[i * K + p + 1];
            // Calculate contribution for j in blocks of 32
            j_start = 0;
            for (; j_start <= K - 32; j_start += 32) {
                for (size_t j_offset = 0; j_offset < 32; ++j_offset) {
                    size_t j = j_start + j_offset;
                    
                    uint32_t packed = B[(p + 1) * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j] += a_val_p1 * sign;
                }
            }
            // Handle remainder for j
            for (size_t j = j_start; j < K; ++j) {
                uint32_t packed = B[(p + 1) * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val_p1 * sign;
            }


            // --- Contribution from p+2 -----
            float a_val_p2 = A[i * K + p + 2];
            // Calculate contribution for j in blocks of 32
            j_start = 0;
            for (; j_start <= K - 32; j_start += 32) {
                for (size_t j_offset = 0; j_offset < 32; ++j_offset) {
                    size_t j = j_start + j_offset;
                    
                    uint32_t packed = B[(p + 2) * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j] += a_val_p2 * sign;
                }
            }
            // Handle remainder for j
            for (size_t j = j_start; j < K; ++j) {
                uint32_t packed = B[(p + 2) * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val_p2 * sign;
            }

            // --- Contribution from p+3 -----
            float a_val_p3 = A[i * K + p + 3];
            // Calculate contribution for j in blocks of 32
            j_start = 0;
            for (; j_start <= K - 32; j_start += 32) {
                for (size_t j_offset = 0; j_offset < 32; ++j_offset) {
                    size_t j = j_start + j_offset;
                    
                    uint32_t packed = B[(p + 3) * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j] += a_val_p3 * sign;
                }
            }
            // Handle remainder for j
            for (size_t j = j_start; j < K; ++j) {
                uint32_t packed = B[(p + 3) * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val_p3 * sign;
            }
        }

        // Handle remaining iterations for p (p to p_limit-1)
        for (; p < K; ++p) {
            float a_val = A[i * K + p];
            
            // Calculate contribution for j in blocks of 32
            size_t j_start = 0;
            for (; j_start <= K - 32; j_start += 32) {
                for (size_t j_offset = 0; j_offset < 32; ++j_offset) {
                    size_t j = j_start + j_offset;
                    
                    uint32_t packed = B[p * K_ints + (j / 32)];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    float sign = bit ? 1.0f : -1.0f;
                    C[i * K + j] += a_val * sign;
                }
            }
            // Handle remainder for j
            for (size_t j = j_start; j < K; ++j) {
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                C[i * K + j] += a_val * sign;
            }
        }
    }
}