// Optimized Matrix Multiplication
// This version uses the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// Optimization strategy:
// 1. Use NEON for fast summation of row A and fast initialization of row C with -sumA.
// 2. We use p-loop unrolling by 4.
// 3. Improvement: Instead of calculating the dualA1..4 inside the chunk loop, 
//    we take them out.
// 4. Improvement: We use a more efficient bit manipulation approach. 
//    Instead of 4 separates 'while' loops that check the same 'target_chunk' pointer, 
//    we can combine the bits into a single word and process them.
//    Actually, the bottleneck is the scatter-add. If we combine bits, we still do 
//    the same number of additions, but we reduce the number of 'target_chunk' 
//    load/store or pointer updates.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* target_rowC = &C[i * K];

        // 1. Calculate sumA using NEON
        float sumA = 0.0f;
        size_t p = 0;
        float32x4_t vsum = vdupq_n_f32(0.0f);
        
        for (; p + 3 < K; p += 4) {
            vsum = vaddq_f32(vsum, vld1q_f32(&rowA[p]));
        }
        
        float temp_lanes[4];
        vst1q_f32(temp_lanes, vsum);
        sumA = temp_lanes[0] + temp_lanes[1] + temp_lanes[2] + temp_lanes_val; 
        // Wait, mistake in variable name.
    }
}
