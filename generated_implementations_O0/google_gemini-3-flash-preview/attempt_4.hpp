
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];
        
        // Sum up row A once
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += row_A[p];
        }

        for (size_t bj = 0; bj < K_ints; ++bj) {
            float sums_ones[32] = {0};
            
            for (size_t p = 0; p < K; ++p) {
                float aval = row_A[p];
                uint32_t bval = B[p * K_ints + bj];
                
                // Unroll in chunks to help compiler
                #define STEP(b) if (bval & (1u << (b))) sums_ones[b] += aval;
                STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
                STEP(8) STEP(9) STEP(10) STEP(11) STEP(12) STEP(13) STEP(14) STEP(15)
                STEP(16) STEP(17) STEP(18) STEP(19) STEP(20) STEP(21) STEP(22) STEP(23)
                STEP(24) STEP(25) STEP(26) STEP(27) STEP(28) STEP(29) STEP(30) STEP(31)
                #undef STEP
            }
            
            for (int b = 0; b < 32; ++b) {
                // val = ones - zeros = ones - (row_sum - ones) = 2*ones - row_sum
                row_C[bj * 32 + b] = 2.0f * sums_ones[b] - row_sum;
            }
        }
    }
}
