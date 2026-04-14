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
        
        for (size_t p = 0; p < K_ints; ++p) {
            uint32_t a_val = *(uint32_t*)&A_i[p * 32];
            const uint32_t* B_row = B + p * K_ints;
            
            for (size_t j = 0; j < K_ints; ++j) {
                uint32_t packed = B_row[j];
                for (int b = 0; b < 32; ++b) {
                    uint32_t bit = (packed >> b) & 1;
                    float val = *(float*)&a_val;
                    C_i[j * 32 + b] = val * (bit ? 1.0f : -1.0f);
                }
            }
        }
    }
}