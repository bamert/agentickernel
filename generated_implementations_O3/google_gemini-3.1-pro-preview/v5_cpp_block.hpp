void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j_int = 0; j_int < K_ints; ++j_int) {
            float c[32] = {0};
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed = B[p * K_ints + j_int];
                
                #pragma unroll
                for (int bit = 0; bit < 32; ++bit) {
                    if ((packed >> bit) & 1) {
                        c[bit] += a_val;
                    } else {
                        c[bit] -= a_val;
                    }
                }
            }
            for (int bit = 0; bit < 32; ++bit) {
                C[i * K + j_int * 32 + bit] = c[bit];
            }
        }
    }
}
