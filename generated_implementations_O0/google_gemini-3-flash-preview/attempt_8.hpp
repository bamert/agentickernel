
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        float* row_C = &C[i * K];
        const float* row_A = &A[i * K];
        
        for (size_t bj = 0; bj < K_ints; ++bj) {
            float sums0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sums1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sums2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sums3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sums4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sums5[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sums6[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sums7[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            
            for (size_t p = 0; p < K; ++p) {
                float32x4_t aval = vdupq_n_f32(row_A[p]);
                uint32_t bval = B[p * K_ints + bj];
                
                auto process_chunk = [&](uint32_t bits) {
                    float32x4_t s;
                    // Bit 0 corresponds to 1.0f, Bit 1 to -1.0f
                    // Mask and compare to create a mask for selecting signs
                    // However, it's easier to just use the logic from attempt 3
                    // but vectorized.
                    float vals[4];
                    for(int b=0; b<4; ++b) vals[b] = ((bits >> b) & 1) ? 1.0f : -1.0f;
                    return vld1q_f32(vals);
                };

                // This is still quite manual. Let's try a different NEON approach.
            }
        }
    }
}
