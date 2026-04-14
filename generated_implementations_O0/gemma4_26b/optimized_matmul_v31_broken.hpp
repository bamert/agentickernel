// Optimized Matrix Multiplication
// Target: Beat v21 (169.67ms)
// Strategy:
// 1. Maintain the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// 2. Use NEON for fast sumA and initialization of row C.
// 3. Optimization: Instead of unrolling the p-loop, we unroll the chunk loop or use a more efficient way to load B.
// 4. Critical: The bottleneck is the scatter-add. We can try to process the bits of B using a technique that reduces the number of branches.
// 5. We use the property that we can load 4 integers from B at once (p, p+1, p+2, p+3) and check their combined OR.

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
        sumA = temp_lanes[0] + temp_lanes[1] + temp_lanes[2] + temp_lanes[3];

        for (; p < K; ++p) {
            sum_A_val = sumA + rowA[p]; 
            sumA = sum_A_val;
        }
        // ERROR in logic above, fixing...
    }
}
