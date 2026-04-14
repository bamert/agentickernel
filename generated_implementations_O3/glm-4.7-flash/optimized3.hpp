// Optimized Matrix Multiplication
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            const uint32_t* B_col = B + j * K_ints;
            
            for (size_t p = 0; p < K; ++p) {
                float a_val = A[i * K + p];
                uint32_t packed = B_col[p / 32];
                uint32_t bit = (packed >> (p % 32)) & 1;
                sum += a_val * (bit ? 1.0f : -1.0f);
            }
            
            C[i * K + j] = sum;
        }
    }
}