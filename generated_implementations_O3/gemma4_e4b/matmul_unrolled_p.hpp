void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; ++j) {     // For each column in B (and C)
            float sum = 0.0f;
            
            size_t p = 0;
            size_t p_limit = K - (K % 4);

            // Manual unrolling of the innermost loop over p (factor of 4)
            for (; p <= p_limit - 4; p += 4) {
                // p
                float a_val_p = A[i * K + p];
                uint32_t packed_p = B[p * K_ints + (j / 32)];
                uint32_t bit_p = (packed_p >> (j % 32)) & 1;
                float sign_p = bit_p ? 1.0f : -1.0f;
                sum += a_val_p * sign_p;

                // p+1
                float a_val_p1 = A[i * K + p + 1];
                uint32_t packed_p1 = B[(p + 1) * K_ints + (j / 32)];
                uint32_t bit_p1 = (packed_p1 >> (j % 32)) & 1;
                float sign_p1 = bit_p1 ? 1.0f : -1.0f;
                sum += a_val_p1 * sign_p1;

                // p+2
                float a_val_p2 = A[i * K + p + 2];
                uint32_t packed_p2 = B[(p + 2) * K_ints + (j / 32)];
                uint32_t bit_p2 = (packed_p2 >> (j % 32)) & 1;
                float sign_p2 = bit_p2 ? 1.0f : -1.0f;
                sum += a_val_p2 * sign_p2;
                
                // p+3
                float a_val_p3 = A[i * K + p + 3];
                uint32_t packed_p3 = B[(p + 3) * K_ints + (j / 32)];
                uint32_t bit_p3 = (packed_p3 >> (j % 32)) & 1;
                float sign_p3 = bit_p3 ? 1.0f : -1.0f;
                sum += a_val_p3 * sign_p3;
            }

            // Handle remaining iterations
            for (; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed = B[p * K_ints + (j / 32)];
                uint32_t bit = (packed >> (j % 32)) & 1;
                float sign = bit ? 1.0f : -1.0f;
                sum += a_val * sign;
            }
            
            C[i * K + j] = sum;
        }
    }
}