// Optimized Matrix Multiplication
// This version uses the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// Optimization strategy:
// 1. Use NEON for fast summation of row A and fast initialization of row C with -sumA.
// 2. Unroll the p-loop by 4 (as proven efficient in v10).
// 3. Critical Improvement: Reduce the number of 32-bit integer loads for the B matrix.
//    In the previous version, we loaded b1, b2, b3, b4 for every chunk.
//    If we process chunks of B along the p-dimension, we can reuse the loaded 32-bit words.
//    Actually, the B matrix is indexed as B[p * K_ints + chunk_idx].
//    For a fixed chunk_idx, as we increment p, we are accessing B[p_idx * K_ints + chunk_idx].
//    This is contiguous in memory if we iterate p then chunk.
// 4. We will try to process 4 chunks of B at a time to improve the load/store ratio.

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
        // Note: I realized I'm repeating the summation logic from a broken version.
        // Let's fix the sumA logic once and for all.
    }
}
