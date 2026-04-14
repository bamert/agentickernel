#pragma once
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;               // K / 32

    // nibble -> four signs (+1 or -1)
    const float signTable[16][4] = {
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1},
        {-1,-1,+1,-1}, {+1,-1,+1,-1}, {-1,+1,+1,-1}, {+1,+1,+1,-1},
        {-1,-1,-1,+1}, {+1,-1,-1,+1}, {-1,+1,-1,+1}, {+1,+1,-1,+1},
        {-1,-1,+1,+1}, {+1,-1,+1,+1}, {-1,+1,+1,+1}, {+1,+1,+1,+1}
    };

    for (size_t i = 0; i < M; ++i) {
        float* Ci = C + i * K;                 // row i of C (assumed zeroed before kernel call)
        const float* Ai = A + i * K;           // row i of A

        for (size_t p = 0; p < K; ++p) {
            float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints; // start of row p in packed B

            // Process the row in 32‑bit chunks
            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                size_t   col_base = chunk * 32;    // first column index of this chunk

                // Process four columns at a time
                #pragma unroll 8
                for (int offset = 0; offset < 32; offset += 4) {
                    // Extract the 4‑bit nibble that encodes the signs for these columns
                    uint32_t nibble = (word >> offset) & 0xF;
                    const float* signs = signTable[nibble]; // points to 4 pre‑computed signs

                    // Global column indices for the four elements we will update
                    size_t idx0 = col_base + offset + 0;
                    size_t idx1 = col_base + offset + 1;
                    size_t idx2 = col_base + offset + 2;
                    size_t idx3 = col_base + offset + 3;

                    // Load current C values (they start at zero)
                    float32x4_t cur0 = vld1q_f32(&Ci[idx0]);
                    float32x4_t cur1 = vld1q_f32(&Ci[idx1]);
                    float32x4_t cur2 = vld1q_f32(&Ci[idx2]);
                    float32x4_t cur3 = vld1q_f32(&Ci[idx3]);

                    // Convert signs to a NEON vector
                    float32x4_t sign_vec = vld1q_f32(&signs[0]);

                    // Broadcast a_val to a NEON register
                    float32x4_t a_vec = vdupq_n_f32(a_val);

                    // Multiply and add
                    float32x4_t term0 = vmulq_f32(sign_vec, a_vec);
                    float32x4_t term1 = vmulq_f32(sign_vec, a_vec);
                    float32x4_t term2 = vmulq_f32(sign_vec, a_vec);
                    float32x4_t term3 = vmulq_f32(sign_vec, a_vec);

                    // Accumulate
                    cur0 = vaddq_f32(cur0, term0);
                    cur1 = vaddq_f32(cur1, term1);
                    cur2 = vaddq_f32(cur2, term2);
                    cur3 = vaddq_f32(cur3, term3);

                    // Store back
                    vst1q_f32(&Ci[idx0], cur0);
                    vst1q_f32(&Ci[idx1], cur1);
                    vst1q_f32(&Ci[idx2], cur2);
                    vst1q_f32(&Ci[idx3], cur3);
                }
            }
        }
    }
}