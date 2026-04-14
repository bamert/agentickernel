#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = &A[i * K];
        float* C_row = &C[i * K];

        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) {
            row_sum += A_row[p];
        }

        float neg_row_sum = -row_sum;
        
        // Step 2: Initialize C_row with -row_sum using NEON
        float32x4_t v_neg_S = vdupq_n_f32(neg_row_sum);
        size_t j_init = 0;
        for (; j_init <= K - 4; j_init += 4) {
            vst1q_f32(&C_row[j_init], v_neg_S);
        }
        for (; j_init < K; ++j_init) {
            C_row[j_init] = neg_row_sum;
        }

        // Step 3: Accumulate 2 * A_ip for every set bit in B
        for (size_t p = 0; p < K; ++p) {
            const float val = 2.0f * A_row[p];
            if (val == 0.0f) continue;

            const uint32_t* B_row_p = &B[p * K_ints];
            float32x4_t v_val = vdupq_n_f16(0.0f); // Placeholder, will use vdupq_n_f32
            // wait, I need vdupq_n_f32(val)
            float32x4_t v_val_f32 = vdupq_n_f32(val);

            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row_p[chunk];
                if (word == 0) continue;
                
                float* C_chunk_ptr = &C_row[chunk * 32];
                if (word == 0xFFFFFFFF) {
                    // Optimization for dense bits: use NEON to add val to all 32 elements
                    // We unroll this to 8 vector additions
                    vst1q_f32(C_chunk_ptr, vaddq_f32(vld1q_f32(C_chunk_ptr), v_val_f32));
                    vst1q_f32(C_chunk_ptr + 4, vaddq_f32(vld1q_f32(C_chunk_ptr + 4), v_val_f32));
                    vst1q_f32(C_chunk_ptr + 8, vaddq_f32(vld1q_f32(C_chunk_ptr + 8), v_val_f32));
                    vst1q_f32(C_chunk_ptr + 12, vaddq_f32(vld1q_f32(C_chunk_ptr + 12), v_val_f32));
                    vst1q_f32(C_chunk_ptr + 16, vaddq_f32(vld1q_f32(C_chunk_ptr + 16), v_val_f32));
                    vst1q_f32(C_chunk_ptr + 20, vaddq_f32(vld1q_f32(C_chunk_ptr + 20), v_val_f32));
                    vst1q_f32(C_chunk_ptr + 24, vaddq_f32(vld1q_f32(C_chunk_ptr + 24), v_val_f32));
                    vst1q_f32(C_chunk_ptr + 28, vaddq_f32(vld1q_f32(C_chunk_ptr + 28), v_val_f32));
                } else {
                    while (word != 0) {
                        int bit_idx = __builtin_ctz(word);
                        C_chunk_ptr[bit_idx] += val;
                        word &= (word - 1);
                    }
                }
            }
        }
    }
}
