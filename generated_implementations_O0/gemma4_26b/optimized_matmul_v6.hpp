// Optimized Matrix Multiplication
// This version uses NEON to accelerate the summation of row A and the initialization of row C.
// It uses the identity: C[i][j] = 2.0f * sum(A[i][p] where bit B[p][j] == 1) - sum(A[i][p]).

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
        
        // Sum the lanes of the vector
        float temp_lanes[4];
        vst1q_f32(temp_lanes, vsum);
        sumA = temp_lanes[0] + temp_lanes[1] + temp_lanes[2] + temp_lanes[3];

        // Handle remaining elements
        for (; p < K; ++p) {
            sumA += rowA[p];
        }

        // 2. Initialize row C with -sumA
        float base_val = -sumA;
        size_t j = 0;
        float32x4_t vbase = vdupq_n_f32(base_val);
        for (; j + 3 < K; j += 4) {
            vst1q_f32(&rowC[j], vbase);
        }
        for (; j < K; ++j) {
            rowC[j] = base_val;
        }

        // 3. Accumulate 2*A[i][p] into the correct columns
        for (size_t p_idx = 0; p_idx < K; ++p_idx) {
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
