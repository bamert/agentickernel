
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];
        
        for (size_t bj = 0; bj < K_ints; ++bj) {
            float sums[32] = {0};
            
            // Further unroll the P loop to reduce loop overhead and improve ILP
            size_t p = 0;
            for (; p + 3 < K; p += 4) {
                float a0 = row_A[p];
                float a1 = row_A[p+1];
                float a2 = row_A[p+2];
                float a3 = row_A[p+3];
                
                uint32_t b0 = B[p * K_ints + bj];
                uint32_t b1 = B[(p+1) * K_ints + bj];
                uint32_t b2 = B[(p+2) * K_ints + bj];
                uint32_t b3 = B[(p+3) * K_ints + bj];

                for (int b = 0; b < 32; ++b) {
                    sums[b] += ((b0 >> b) & 1) ? a0 : -a0;
                    sums[b] += ((b1 >> b) & 1) ? a1 : -a1;
                    sums[b] += ((b2 >> b) & 1) ? a2 : -a2;
                    sums[b] += ((b3 >> b) & 1) ? a3 : -a3;
                }
            }
            
            for (; p < K; ++p) {
                float a = row_A[p];
                uint32_t b = B[p * K_ints + bj];
                for (int bit = 0; bit < 32; ++bit) {
                    sums[bit] += ((b >> bit) & 1) ? a : -a;
                }
            }
            
            for (int b = 0; b < 32; ++b) {
                row_C[bj * 32 + b] = sums[b];
            }
        }
    }
}
