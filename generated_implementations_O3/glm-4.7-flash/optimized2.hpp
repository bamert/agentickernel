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
            
            for (size_t p = 0; p < K; p += 8) {
                // Load and process 8 uint32_t values
                uint32_t packed0 = B_col[p / 32];
                float a_val0 = A[i * K + p];
                uint32_t bit0 = (packed0 >> (j & 31)) & 1;
                sum += a_val0 * (bit0 ? 1.0f : -1.0f);
                
                if (p + 1 < K) {
                    uint32_t packed1 = B_col[(p + 1) / 32];
                    float a_val1 = A[i * K + p + 1];
                    uint32_t bit1 = (packed1 >> (j & 31)) & 1;
                    sum += a_val1 * (bit1 ? 1.0f : -1.0f);
                }
                
                if (p + 2 < K) {
                    uint32_t packed2 = B_col[(p + 2) / 32];
                    float a_val2 = A[i * K + p + 2];
                    uint32_t bit2 = (packed2 >> (j & 31)) & 1;
                    sum += a_val2 * (bit2 ? 1.0f : -1.0f);
                }
                
                if (p + 3 < K) {
                    uint32_t packed3 = B_col[(p + 3) / 32];
                    float a_val3 = A[i * K + p + 3];
                    uint32_t bit3 = (packed3 >> (j & 31)) & 1;
                    sum += a_val3 * (bit3 ? 1.0f : -1.0f);
                }
                
                if (p + 4 < K) {
                    uint32_t packed4 = B_col[(p + 4) / 32];
                    float a_val4 = A[i * K + p + 4];
                    uint32_t bit4 = (packed4 >> (j & 31)) & 1;
                    sum += a_val4 * (bit4 ? 1.0f : -1.0f);
                }
                
                if (p + 5 < K) {
                    uint32_t packed5 = B_col[(p + 5) / 32];
                    float a_val5 = A[i * K + p + 5];
                    uint32_t bit5 = (packed5 >> (j & 31)) & 1;
                    sum += a_val5 * (bit5 ? 1.0f : -1.0f);
                }
                
                if (p + 6 < K) {
                    uint32_t packed6 = B_col[(p + 6) / 32];
                    float a_val6 = A[i * K + p + 6];
                    uint32_t bit6 = (packed6 >> (j & 31)) & 1;
                    sum += a_val6 * (bit6 ? 1.0f : -1.0f);
                }
                
                if (p + 7 < K) {
                    uint32_t packed7 = B_col[(p + 7) / 32];
                    float a_val7 = A[i * K + p + 7];
                    uint32_t bit7 = (packed7 >> (j & 31)) & 1;
                    sum += a_val7 * (bit7 ? 1.0f : -1.0f);
                }
            }
            
            C[i * K + j] = sum;
        }
    }
}