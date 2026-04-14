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
            const uint32_t* B_p = B + p * K_ints;
            float a_val = A_i[p * 32];
            
            for (size_t j = 0; j < K; ++j) {
                uint32_t bit = (B_p[j / 32] >> (j % 32)) & 1;
                C_i[j] += a_val * (bit ? 1.0f : -1.0f);
            }
        }
    }
}