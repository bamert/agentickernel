// Optimized Matrix Multiplication using NEON and Bit-Scanning
// This version uses NEON for fast summation of A and fast initialization of C.
// It also uses NEON to skip zeroed chunks of B 4-at-a-time.
// The core identity is: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

        // 1. Fast sumA calculation using NEON
        float sumA = 0.0f;
        size_t p = 0;
        float32x4_t vsum = vdupq_n_f32(0.0f);
        for (; p + 3 < K; p += 4) {
            vsum = vaddq_f32(vsum, vld1q_f32(&rowA[p]));
        }
        
        float s_res = vgetq_lane_f32(vsum, 0) + vget_lane_f32(vsum, 1) +
                      vget_lane_f32(vsum, 2) + vget_lane_f32(vsum, 3);
        // wait, error in my mental code, let's use the correct name
        // (re-evaluating)
    }
}
