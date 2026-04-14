
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        // Prepare row i of C. Initialize with -1.0f * sum(row i of A) 
        // because each bit 0 acts as -1.0f and bit 1 acts as +1.0f.
        // sum = sum(A[i][p] * (2*bit - 1)) = sum(2*A[i][p]*bit - A[i][p])
        // sum = 2 * sum(A[i][p] where bit is 1) - sum(A[i][p] for all p)
        
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += A[i * K + p];
        }

        for (size_t j = 0; j < K; ++j) {
            float sum_bit_ones = 0.0f;
            size_t b_idx_base = j / 32;
            uint32_t b_mask = (1u << (j % 32));

            for (size_t p = 0; p < K; ++p) {
                uint32_t packed = B[p * K_ints + b_idx_base];
                if (packed & b_mask) {
                    sum_bit_ones += A[i * K + p];
                }
            }
            C[i * K + j] = 2.0f * sum_bit_ones - row_sum;
        }
    }
}
