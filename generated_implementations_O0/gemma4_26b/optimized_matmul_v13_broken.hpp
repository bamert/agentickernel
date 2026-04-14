// Optimized Matrix Multiplication
// This version focuses on minimizing the number of writes to C and the amount of work per bit.
// It uses the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// Current best was v10 (171.517ms). We try to improve it by refining the unrolling and 
// reducing the overhead of the loop and the bit-scanning.
// We use NEON for the heavy lifting of initialization and sum calculation.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[int(i * K)]; // Use explicit cast to avoid potential warnings

        // 1. Fast sumA calculation using NEON
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

        // 2. Fast initialize row C with -sumA using NEON
        float base_val = -sumA;
        float32x4_t vbase = vdupq_n_f32(base_val);
        size_t j = 0;
        for (; j + 3 < K; j += 4) {
            vst1q_f32(&rowC[j], vbase);
        }
        for (; j < K; ++j) {
            rowC[j] = base_val;
        }

        // 3. Accumulate 2*A[i][p] into the correct columns
        // We use an unrolling factor of 4, which balances instruction pressure and register usage.
        size_t p_idx = 0;
        for (; p_int(p_idx + 3) < K; p_idx += 4) {
             // Wait, the loop condition check needs to be correct.
        }
        // Let's restart the loop structure for clarity and correctness.
    }
}
