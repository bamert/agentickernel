// Optimized Matrix Multiplication
// This version uses the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// Optimization strategy:
// 1. Use NEON for fast summation of row A and fast initialization of row C with -sumA.
// 2. We use p-loop unrolling by 4 (as proven efficient in v10).
// 3. We optimize the bit-scanning loop. 
// 4. To improve cache locality, we process B in chunks.

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
            sumA += rowA[p];
        }

        // 2. Initialize row C with -sumA using NEON
        float base_val = -sumA;
        float32x4_t vbase = vdupq_n_f32(base_val);
        size_t j = 0;
        for (; j + 3 < K; j += 4) {
            vst1_f32(&target_rowC[j], vbase); // Wait, vst1q_f32
        }
        // Let's use the correct NEON function names.
    }
}
