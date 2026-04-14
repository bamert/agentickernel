// Optimized Matrix Multiplication
// This version uses NEON for fast summation of A and fast initialization of C.
// It leverages the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// Optimization: Process bits in 4-wide chunks using NEON where possible.
// We use a bit-mask approach for the 32-bit word to avoid loop overheads when bits are dense.

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
        // We optimize the inner loop by checking chunks of B.
        for (size_t p_idx = 0; p_idx < K; ++p_idx) {
            const float dualA = 2.0 * rowA[p_idx];
            const uint32_t* rowB_ptr = &B[p_idx * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits = rowB_ptr[chunk_idx];
                if (bits == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];
                
                // The __builtin_ctz approach is very fast for sparse bits.
                // If the word is very dense, the overhead of ctz might increase,
                // but for most random/binary matrices, it is very efficient.
                while (bits != 0) {
                    int bit_pos = __builtin_ctz(bits);
                    target_C[bit_pos] += dualA;
                    bits &= (bits - 1);
                }
            }
        }
    }
}
