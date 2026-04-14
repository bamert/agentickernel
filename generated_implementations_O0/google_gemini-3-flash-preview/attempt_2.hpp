
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    // Zero out C initially to use addition
    for (size_t i = 0; i < M * K; ++i) C[i] = 0.0f;

    for (size_t p = 0; p < K; ++p) {
        const float* row_A = &A[p]; // We want A[i][p]. A is M x K.
        // Actually, A is M rows, K cols. A[i][p] is A[i * K + p].
        // Let's iterate i as the outer loop to keep A access sequential as much as possible,
        // but switch B access to process multiple columns via the bitmask.
    }

    // Attempt 2: Block-based approach to improve cache hits and reduce bit extraction overhead.
    for (size_t i = 0; i < M; ++i) {
        const float* row_A = &A[i * K];
        float* row_C = &C[i * K];

        for (size_t bj = 0; bj < K_ints; ++bj) {
            // Process 32 columns of C at once for this row A[i]
            float local_C[32] = {0};
            
            for (size_t p = 0; p < K; ++p) {
                float a_val = row_A[p];
                uint32_t packed_B = B[p * K_ints + bj];
                
                // Unroll bit extraction
                for (int bit = 0; bit < 32; ++bit) {
                    if ((packed_B >> bit) & 1) {
                        local_C[bit] += a_val;
                    } else {
                        local_C[bit] -= a_val;
                    }
                }
            }
            
            for (int bit = 0; bit < 32; ++bit) {
                row_C[bj * 32 + bit] = local_C[bit];
            }
        }
    }
}
