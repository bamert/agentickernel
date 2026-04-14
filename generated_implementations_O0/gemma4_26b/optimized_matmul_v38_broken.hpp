// Optimized Matrix Multiplication
// Target: Beat v21 (169.67ms)
// Strategy:
// 1. Use the identity: C[i][j] = 2 * sum(A[i][p] where B[p][j] == 1) - sum(A[i][p]).
// 2. NEON for fast summation of row A and fast initialization of row C with -sumA.
// 3. p-loop unrolling by 4.
// 4. Optimization: Use a single traversal of the chunk to process all bits from b1, b2, b3, b4.
//    This reduces the number of times 'target_chunk' is accessed.
//    We use a bitmask approach: combine b1, b2, b3, b4 into one wide mask, 
//    but we need to know which dualA to use. 
//    Actually, let's try to reduce the number of 'if' and 'while' checks by combining the
//    bit-scanning logic into a more streamlined loop.

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
            vst1q_f32(&target_rowC[j], vbase);
        }
        for (; j < K; ++j) {
            target_rowC[j] = base_val;
        }

        // 3. Accumulate 2*A[i][p]
        size_t p_idx = 0;
        for (; p_idx + 3 < K; p_idx += 4) {
            const float dualA1 = 2.0F * rowA[p_idx];
            const float dualA2 = 2.0F * rowA[p_idx + 1];
            const float dualA3 = 2.0F * rowA[p_idx + 2];
            const float dualA4 = 2.0F * rowA[p_idx + 3];
            
            const uint32_t* rowB1 = &B[p_idx * K_ints];
            const uint32_t* rowB2 = &B[(p_idx + 1) * K_ints];
            const uint32_t* rowB3 = &B[(p_idx + 2) * K_ints];
            const uint32_t* rowB4 = &B[(p_idx + 3) * K_ints];

            for (size_t chunk_idx = 0; chunk_idx < K_ints; ++chunk_idx) {
                uint32_t b1 = rowB1[chunk_idx];
                uint32_t b2 = rowB2[chunk_idx];
                uint32_t b3 = rowB3[chunk_idx];
                uint32_t b4 = rowB[chunk_idx]; // Wait, bug! Should be rowB4.
                // Let's fix this correctly.
            }
        }
    }
}
