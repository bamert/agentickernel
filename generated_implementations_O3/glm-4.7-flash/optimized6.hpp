// Optimized Matrix Multiplication
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    const size_t unroll = 8;
    
    for (size_t i = 0; i < M; ++i) {
        const float* A_i = A + i * K;
        float* C_i = C + i * K;
        
        for (size_t j = 0; j < K; ++j) {
            const uint32_t* B_start = B + j * K_ints;
            float sum = 0.0f;
            
            for (size_t p = 0; p < K; p += unroll) {
                size_t remain = K - p;
                if (remain >= 4) {
                    // Process 4 items
                    uint32_t packed0 = B_start[(p + 0) / 32];
                    uint32_t packed1 = B_start[(p + 1) / 32];
                    uint32_t packed2 = B_start[(p + 2) / 32];
                    uint32_t packed3 = B_start[(p + 3) / 32];
                    
                    float a0 = A_i[p + 0];
                    float a1 = A_i[p + 1];
                    float a2 = A_i[p + 2];
                    float a3 = A_i[p + 3];
                    
                    uint32_t bit0 = (packed0 >> (j % 32)) & 1;
                    uint32_t bit1 = (packed1 >> (j % 32)) & 1;
                    uint32_t bit2 = (packed2 >> (j % 32)) & 1;
                    uint32_t bit3 = (packed3 >> (j % 32)) & 1;
                    
                    sum += a0 * (bit0 ? 1.0f : -1.0f);
                    sum += a1 * (bit1 ? 1.0f : -1.0f);
                    sum += a2 * (bit2 ? 1.0f : -1.0f);
                    sum += a3 * (bit3 ? 1.0f : -1.0f);
                    
                    p += 4;
                } else if (remain >= 2) {
                    // Process 2 items
                    uint32_t packed4 = B_start[(p + 0) / 32];
                    uint32_t packed5 = B_start[(p + 1) / 32];
                    
                    float a4 = A_i[p + 0];
                    float a5 = A_i[p + 1];
                    
                    uint32_t bit4 = (packed4 >> (j % 32)) & 1;
                    uint32_t bit5 = (packed5 >> (j % 32)) & 1;
                    
                    sum += a4 * (bit4 ? 1.0f : -1.0f);
                    sum += a5 * (bit5 ? 1.0f : -1.0f);
                    
                    p += 2;
                }
                
                // Process remaining single items
                for (; p < K; ++p) {
                    float a_val = A_i[p];
                    uint32_t packed = B_start[p / 32];
                    uint32_t bit = (packed >> (j % 32)) & 1;
                    sum += a_val * (bit ? 1.0f : -1.0f);
                }
            }
            
            C_i[j] = sum;
        }
    }
}