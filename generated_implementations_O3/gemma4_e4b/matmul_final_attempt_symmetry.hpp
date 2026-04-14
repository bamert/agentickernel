void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A (and C)
        // Initialize C[i] = 0.0f
        for(size_t j_init = 0; j_init < K; ++j_init) {
            C[i * K + j_init] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) { // Summation index p
            float a_val = A[i * K + p];
            
            size_t j = 0;
            size_t j_limit_4 = K - (K % 4);

            // Unroll j loop by 4
            for (; j <= j_limit_4 - 4; j += 4) {
                // Calculate contribution for j, j+1, j+2, j+3 using block indexing
                
                // j
                {
                    size_t j_block = j;
                    uint32_t packed = B[p * K_ints + (j_block / 32)];
                    float sign = (((packed >> (j_block % 32)) & 1) ? 1.0f : -1.0f);
                    C[i * K + j_block] += a_val * sign;
                }
                // j+1
                {
                    size_t j_block = j + 1;
                    uint32_t packed = B[p * K_ints + ((j + 1) / 32)];
                    float sign = (((B[p * K_ints + ((j + 1) / 32)] >> ((j + 1) % 32)) & 1) ? 1.0f : -1.0f);
                    C[i * K + j_block] += a_val * sign;
                }
                // j+2
                {
                    size_t j_block = j + 2;
                    uint32_t packed = B[p * K_ints + ((j + 2) / 32)];
                    float sign = (((B[p * K_ints + ((j + 2) / 32)] >> ((j + 2) % 32)) & 1) ? 1.0f : -1.0f);
                    C[i * K + j_block] += a_val * sign;
                }
                // j+3
                {
                    size_t j_block = j + 3;
                    uint32_t packed = B[p * K_ints + ((j + 3) / 32)];
                    float sign = (((B[p * K_ints + ((j + 3) / 32)] >> ((j + 3) % 32)) & 1) ? 1.0f : -1.0f);
                    C[i * K + j_block] += a_val * sign;
                }
            }

            // Handle remaining iterations j (Linear scan)
            for (; j < K; ++j) {
                uint32_t packed = B[p * K_ints + (j / 32)];
                float sign = (((packed >> (j % 32)) & 1) ? 1.0f : -1.0f);
                C[i * K + j] += a_val * sign;
            }
        }
    }
}