// Optimized Matrix Multiplication
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_i = A + i * K;
        float* C_i = C + i * K;

        for (size_t j = 0; j < K_ints; ++j) {
            const uint32_t* B_col = B + j * K_ints;
            float* C_out = C_i + j * 32;

            for (size_t k = 0; k < K_ints; ++k) {
                uint32_t packed = B_col[k];
                float a_val = A_i[k * 32];

                for (int b = 31; b >= 0; --b) {
                    uint32_t bit = (packed >> b) & 1;
                    int idx = k * 32 + (31 - b);
                    C_out[idx] += a_val * (bit ? 1.0f : -1.0f);
                }
            }
        }
    }
}