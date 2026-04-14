// Optimized Matrix Multiplication
// Strategy:
// 1. Use the arithmetic identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// 2. NEON for fast row A summation and row C initialization.
// 3. Loop unrolling (p_idx) by 4.
// 4. Optimization: Instead of unrolling the p-loop and doing independent while loops,
//    we process the B chunks with a single combined bitmask. 
//    By ORing all 4 bits (b1|b2|b3|b4) into a single mask, we can iterate over 
//    the 'set' bits in the combined mask, and for each set bit, determine 
//    which of the 4 source rows it came from. 
//    However, that requires more logic. 
//    Let's try a different tactic: 
//    Flattening the p-loop unrolling and using a single 'while' loop 
//    over the combined bits to reduce branchy code in the inner loop.
//    Wait, if we expand the bits into one large mask, we actually lose 
//    the ability to easily know which dualA to use without more logic.
//    Let's stick to the proven p-loop unrolling but optimize the B loading.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* rowA = &A[i * K];
        float* target_rowC = &C[i * K];

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

        float base_val = -sumA;
        float32x4_t vbase = vdupq_n_f32(base_val);
        size_t j = 0;
        for (; j + 3 < K; j += 4) {
            vst1q_f32(&target_rowC[j], vbase);
        }
        for (; j < K; ++j) {
            target_rowC[j] = base_val;
        }

        size_t p_idx = 0;
        for (; p_idx + 3 < K; p_idx += 4) {
            const float dualA1 = 2.0F * rowA[p_idx];
            const float dualA2 = 2.0F * rowA[p_idx + 1];
            const float dualA3 = 2.0F * rowA[p_idx + 2];
            const float dual_A4 = 2.0F * rowA[p_idx + 3];
            
            const uint32_t* rowB1 = &B[p_idx * K_ints];
            const uint32_t* rowB2 = &B[(p_idx + 1) * K_ints];
            const uint32_t* rowB3 = &B[(p_idx + 2) * K_ints];
            const uint32_t* rowB4 = &B[(p_idx + 3) * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t b1 = rowB1[chunk_idx];
                uint32_t b2 = rowB2[chunk_idx];
                uint32_t b3 = rowB3[chunk_idx];
                uint32_t b4 = rowB4[chunk_idx];

                uint32_t combined = b1 | b2 | b3 | b4;
                if (combined == 0) continue;

                float* __restrict__ target_chunk = &target_rowC[chunk_idx * 32];

                // We process bits of b1, then b2, etc. 
                // To optimize, we loop over the bits of b1, then b2...
                // This minimizes the number of times we load 'target_chunk'.
                while (b1 != 0) {
                    int bit = __builtin_ctz(b1);
                    target_chunk[bit] += dualA1;
                    b1 &= (b1 - 1);
                }
                while (b2 != 0) {
                    int bit = __builtin_ctz(b2);
                    target_chunk[bit] += dualA2;
                    b2 &= (b2 - 1);
                }
                while (b3 != 0) {
                    int bit = __builtin_ctz(b3);
                    target_chunk[bit] += dualA3;
                    b3 &= (b3 - 1);
                }
                while (b4 != 0) {
                    int bit = __builtin_ctz(b4);
                    target_chunk[bit] += dual_A4;
                    b4 &= (b4 - 1);
                }
            }
        }

        for (; p_idx < K; ++p_idx) {
            const float dualA = 2.0f * rowA[p_idx];
            const uint32_t* rowB = &B[p_idx * K_ints];
            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t bits = rowB[chunk_idx];
                if (bits == 0) continue;
                float* __restrict__ target_chunk = &target_rowC[chunk_idx * 32];
                while (bits != 0) {
                    int bit = __builtin_ctz(bits);
                    target_chunk[bit] += dualA;
                    bits &= (bits - 1);
                }
            }
        }
    }
}
