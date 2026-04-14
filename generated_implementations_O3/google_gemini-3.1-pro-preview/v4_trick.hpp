void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    for (size_t i = 0; i < M; ++i) {
        float S_i = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            S_i += A[i * K + p];
        }

        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
        }

        for (size_t p = 0; p < K; ++p) {
            float a_val = A[i * K + p];
            const uint32_t* B_row = B + p * K_ints;
            float* C_row = C + i * K;
            
            for (size_t j_int = 0; j_int < K_ints; ++j_int) {
                uint32_t packed = B_row[j_int];
                for (size_t bit = 0; bit < 32; ++bit) {
                    if ((packed >> bit) & 1) {
                        C_row[j_int * 32 + bit] += a_val;
                    }
                }
            }
        }

        for (size_t j = 0; j < K; ++j) {
            C[i * K + j] = C[i * K + j] * 2.0f - S_i;
        }
    }
}
