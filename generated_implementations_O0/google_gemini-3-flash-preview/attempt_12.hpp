
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* __restrict__ row_C = &C[i * K];
        const float* __restrict__ row_A = &A[i * K];
        
        for (size_t bj = 0; bj < K_ints; ++bj) {
            float sums[32] = {0};
            
            // Loop optimized for compiler auto-vectorization and branchless sign selection.
            // Using a strictly simple form similar to Attempt 3.
            for (size_t p = 0; p < K; ++p) {
                const float aval = row_A[p];
                const uint32_t bval = B[p * K_ints + bj];
                
                for (int b = 0; b < 32; ++b) {
                    float s = (bval & (1u << b)) ? 1.0f : -1.0f;
                    sums[b] += aval * s;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                row_C[bj * 32 + b] = sums[b];
            }
        }
    }
}
