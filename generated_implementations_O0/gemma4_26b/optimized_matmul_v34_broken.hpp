// Optimized Matrix Multiplication
// Target: Beat v21 (169.67ms)
// Strategy: 
// 1. Use the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// 2. NEON for fast sumA and row C initialization.
// 3. Unroll p-loop by 4.
// 4. The bottleneck is the many 'while' loops and scattered writes.
// 5. Critical Optimization: Instead of 4 separate 'while' loops that each 
//    perform 'target_chunk[bit] += dualA_x', let's try to process 
//    the bits of all 4 rows at once. 
//    We can combine the bits into one 128-bit-like structure or just 
//    iterate through all bits of the 4 words together.
//    Wait, the most expensive part is the pointer arithmetic/load for target_chunk.
//    Let's try to process all 4 'b' words in a single loop over the bits of (b1|b2|b3|b4).

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* target_rowC = &C[i * K];

        float sumA = 0.0f;
        size_t p = 0;
        float32x4_t vsum = vdupq_n_f32(0.0f);
        
        for (; p + 3 < K; p += 4) {
            vsum = vaddq_f32(vsum, vld1q_f32(&rowA[p]));
        }
        
        float temp_lanes[4];
        vst1q_f32(temp_lanes, vsum);
        sumA = temp_lanes[0] + temp_lanes[1] + temp_lanes[2] + temp_lanes[3];

        for (; p < K; ++p) {
            sumA += row_A_val; // Error: sum_A_val
            // Let's fix the sumA logic properly.
        }
    }
}
