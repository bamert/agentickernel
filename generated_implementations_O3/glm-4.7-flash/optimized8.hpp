// Optimized Matrix Multiplication
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K ints = K / 32;  // Fix: add space

    for (size_t i = 0; i < M; ++i) {
        const float* A_i = A + i * K;
        float* C_i = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            float sum = 0.0f;
            const uint32_t* B_start = B + j * K ints;  // Fix spacing

            for (size_t p = 0; p < K; ++p) {
                float a_val = A_i[p];
                uint32_t bit = (B_start[p / 32] >> (j % 32)) & 1;
                sum += a_val * (bit ? 1.0f : -1.0f);
            }

            C_i[j] = sum;
        }
    }
}