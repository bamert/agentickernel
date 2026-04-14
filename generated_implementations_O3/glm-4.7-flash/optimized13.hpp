// Optimized Matrix Multiplication
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K_ints; ++j) {
            int bit_offset = (j * 32) % 32;  // Bit position in the packed uint32
            for (size_t p = 0; p < K_ints; ++p) {
                float sum = 0.0f;
                const uint32_t* B_col = B + p * K_ints + j;
                for (int b = 0; b < 32; ++b) {
                    uint32_t bit = (B_col[b / 32] >> bit_offset) & 1;
                    float a_val = A[i * K + p * 32 + b];
                    sum += a_val * (bit ? 1.0f : -1.0f);
                }
                C[i * K + bit_offset] = sum;
                bit_offset = (bit_offset + 1) % 32;
            }
        }
    }
}