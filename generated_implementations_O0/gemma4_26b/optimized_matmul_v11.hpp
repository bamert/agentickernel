// Optimized Matrix Multiplication
// This version uses NEON for fast summation of A entries and fast initialization of C entries.
// It uses the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// Optimization: Unrolling the p-loop by 8 to further reduce loop overhead and improve instruction scheduling.
// We also use the __builtin_ctz approach for efficient bit-scanning within the unrolled chunks.

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
        // Unrolling the p loop by 8
        size_t p_idx = 0;
        for (; p_idx + 7 < K; p_idx += 8) {
            const float dualA1 = 2.0f * rowA[p_idx];
            const float dualA2 = 2.0f * rowA[p_idx + 1];
            const float dualA3 = 2.0f * rowA[p_idx + 2];
            const float dualA4 = 2.0f * rowA[p_idx + 3];
            const float dualA5 = 2.0f * rowA[p_idx + 4];
            const float dualA6 = 2.0f * rowA[p_idx + 5];
            const float dualA7 = 2.0f * rowA[p_idx + 6];
            const float dualA8 = 2.0f * rowA[p_idx + 7];
            
            const uint32_t* rowB_ptr1 = &B[p_idx * K_ints];
            const uint32_t* rowB_ptr2 = &B[(p_idx + 1) * K_ints];
            const uint32_t* rowB_ptr3 = &B[(p_idx + 2) * K_ints];
            const uint32_t* rowB_ptr4 = &B[(p_idx + 3) * K_ints];
            const uint32_t* rowB_ptr5 = &B[(p_idx + 4) * K_ints];
            const uint32_t* rowB_ptr6 = &B[(p_idx + 5) * K_ints];
            const uint32_t* rowB_ptr7 = &B[(p_idx + 6) * K_ints];
            const uint32_t* rowB_ptr8 = &B[(p_idx + 7) * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits1 = rowB_ptr1[chunk_idx];
                uint32_t bits2 = rowB_ptr2[chunk_idx];
                uint32_t bits3 = rowB_ptr3[chunk_idx];
                uint32_t bits4 = rowB_ptr4[chunk_idx];
                uint32_t bits5 = rowB_ptr5[chunk_idx];
                uint32_t bits6 = rowB_ptr6[chunk_idx];
                uint32_t bits7 = rowB_ptr7[chunk_idx];
                uint32_t bits8 = rowB_ptr8[chunk_idx];

                if (bits1 == 0 && bits2 == 0 && bits3 == 0 && bits4 == 0 && 
                    bits5 == 0 && bits6 == 0 && bits2 == 0 && bits8 == 0) { // Wait, typo in my safety check
                    // Let's just check if all are zero
                }
                // Corrected check:
                if ((bits1 | bits2 | bits3 | bits4 | bits5 | bits6 | bits7 | bits8) == 0) continue;

                float* target_C = &rowC[chunk_idx * 32];

                // Helper macro to avoid repetition and keep code clean
                #define PROCESS_BITS(bits, val) \
                while (bits != 0) { \
                    int bit_pos = __builtin_ctz(bits); \
                    target_C[bit_pos] += val; \
                    bits &= (bits - 1); \
                }

                PROCESS_BITS(bits1, dualA1);
                PROCESS_BITS(bits2, dualA2);
                PROCESS_BITS(bits3, dualA3);
                PROCESS_BITS(bits4, dualA4);
                PROCESS_BITS(bits5, dualA5);
                PROCESS_BITS(bits6, dualA6);
                PROCESS_BITS(bits7, dualA7);
                PROCESS_BITS(bits8, dualA8);

                #undef PROCESS_BITS
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
