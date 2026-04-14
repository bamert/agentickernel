// Optimized Matrix Multiplication
// This version uses the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// It uses NEON for fast summation of A and fast initialization of C.
// To improve upon v10, we use the p-loop unrolling by 4 but with a more optimized 
// inner loop for the bit-scanning to reduce the number of loads and jumps.

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
            sum_A_val = sumA + rowA[p]; // Correcting my local logic
            sumA = sum_A_val;
        }
        // Resetting correctly
        p = 0;
        sumA = 0.0f;
        vsum = vdupq_n_f32(0.0f);
        for (; p + 3 < K; p += 4) {
            vsum = vaddq_f32(vsum, vld1q_f32(&rowA[p]));
        }
        float temp_lanes_final[4];
        vst1q_f32(temp_lanes_final, vsum);
        sumA = temp_lanes_final[0] + temp_lanes_final[1] + temp_lanes_final[2] + temp_lanes_final[3];
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
        // We use an unrolling factor of 4.
        size_t p_idx = 0;
        for (; p_idx + 3 < K; p_idx += 4) {
            const float dualA1 = 2.0f * rowA[p_idx];
            const float dualA2 = 2.0f * rowA[p_idx + 1];
            const float dualA3 = 2.0f * rowA[p_idx + 2];
            const float dualA4 = 2.0f * rowA[p_idx + 3];
            
            const uint32_t* rowB_ptr1 = &B[p_idx * K_ints];
            const uint32_t* rowB_ptr2 = &B[(p_idx + 1) * K_ints];
            const uint32_t* rowB_ptr3 = &B[(p_idx + 2) * K_ints];
            const uint32_t* rowB_ptr4 = &B[(p_idx + 3) * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits1 = rowB_ptr1[chunk_idx];
                uint32_t bits2 = rowB_ptr2[chunk_idx];
                uint32_t bits3 = rowB_ptr3[chunk_idx];
                uint32_t bits4 = rowB_ptr4[chunk_idx];

                if ((bits1 | bits2 | bits3 | bits4) == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];

                while (bits1 != 0) {
                    int bit_pos = __builtin_auclz(bits1 ^ 0xFFFFFFFF) ^ 31; // Note: ctz is better
                    // Using builtin_ctz directly
                    int pos = __builtin_ctz(bits1);
                    target_C[pos] += dualA1;
                    bits1 &= (bits1 - 1);
                }
                while (bits2 != 0) {
                    int pos = __builtin_ctz(bits2);
                    target_C[pos] += dualA2;
                    bits2 &= (bits2 - 1);
                }
                while (bits3 != 0) {
                    int pos = __builtin_ctz(bits3);
                    target_C[pos] += dualA3;
                    bits3 &= (bits3 - 1);
                }
                while (bits4 != 0) {
                    int pos = __builtin_ctz(bits4);
                    target_C[pos] += dualA4;
                    bits4 &= (bits4 - 1);
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
                    int pos = __builtin_ctz(bits);
                    target_C[pos] += dualA;
                    bits &= (bits - 1);
                }
            }
        }
    }
}
