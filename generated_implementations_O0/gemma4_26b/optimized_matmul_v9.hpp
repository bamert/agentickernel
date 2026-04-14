// Optimized Matrix Multiplication
// This version uses NEON for fast summation of A and fast initialization of C.
// It uses the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// Optimization: Unroll the p-loop to allow for better instruction scheduling and reduce loop overhead.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* rowC = &C[i * K];

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
            vst1q_f32(&rowC[j], vbase);
        }
        for (; j < K; ++j) {
            rowC[j] = base_val;
        }

        // 3. Accumulate 2*A[i][p] into the correct columns
        // Unrolling the p loop by 2
        size_t p_idx = 0;
        for (; p_idx + 1 < K; p_idx += 2) {
            const float dualA1 = 2.0f * rowA[p_idx];
            const float dualA2 = 2.0f * rowA[p_idx + 1];
            const uint32_t* rowB_ptr1 = &B[p_idx * K_ints];
            const uint32_t* rowB_ptr2 = &B[(p_idx + 1) * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits1 = rowB_ptr1[chunk_idx];
                uint32_t bits2 = rowB_ptr2[chunk_idx];

                if (bits1 == 0 && bits2 == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];

                // Process bits from the first row
                while (bits1 != 0) {
                    int bit_pos = __builtin_ctz(bits1);
                    target_C[bit_pos] += dualA1;
                    bits1 &= (bits1 - 1);
                }
                // Process bits from the second row
                while (bits2 != 0) {
                    int bit_pos = __builtin_ctz(bits2);
                    target_C[bit_pos] += dualA2;
                    bits2 &= (bits2 - 1);
                }
            }
        }

        // Handle remaining p_idx
        for (; p_idx < K; ++p_idx) {
            const float dualA = 2.0f * rowA[p_idx];
            const uint32_t* rowB_ptr = &B[p_idx * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits = rowB_ptr[chunk_idx];
                if (bits == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];
                while (bits != 0) {
                    int bit_pos = __builtin_ctz(bits);
                    target_C[bit_pos] += dualA;
                    bits &= (bits - 1);
                }
            }
        }
    }
}
