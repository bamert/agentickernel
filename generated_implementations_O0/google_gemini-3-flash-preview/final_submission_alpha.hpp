
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];
        
        for (size_t bj = 0; bj < K_ints; ++bj) {
            float sums[32] = {0};
            
            for (size_t p = 0; p < K; ++p) {
                float aval = row_A[p];
                uint32_t bval = B[p * K_ints + bj];
                
                for (int b = 0; b < 32; ++b) {
                    float sign = (bval & (1u << b)) ? 1.0f : -1.0f;
                    sums[b] += aval * sign;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                row_C[bj * 32 + b] = sums[b];
            }
        }
    }
}
