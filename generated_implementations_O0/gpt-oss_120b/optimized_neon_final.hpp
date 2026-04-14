#pragma once

using uint32_t = unsigned int;
using size_t = unsigned long;

// NEON‑based matrix multiplication using the identity:
//   C[i][j] = 2 * Σ_{p where B[p][j]=1} A[i][p] - Σ_p A[i][p]
// Processes four columns at a time with SIMD. The mask table is created on the stack.
void matmul(const float* A, const uint32_t* B, float* C,
            size_t M, size_t K) {
    const size_t K_ints = K / 32; // uint32 per row of B

    // 4‑bit lookup table: each entry holds four floats (0.0f or 1.0f).
    const float maskLUT[16][4] = {
        {0.f,0.f,0.f,0.f}, {1.f,0.f,0.f,0.f}, {0.f,1.f,0.f,0.f}, {1.f,1.f,0.f,0.f},
        {0.f,0.f,1.f,0.f}, {1.f,0.f,1.f,0.f}, {0.f,1.f,1.f,0.f}, {1.f,1.f,1.f,0.f},
        {0.f,0.f,0.f,1.f}, {1.f,0.f,0.f,1.f}, {0.f,1.f,0.f,1.f}, {1.f,1.f,0.f,1.f},
        {0.f,0.f,1.f,1.f}, {1.f,0.f,1.f,1.f}, {0.f,1.f,1.f,1.f}, {1.f,1.f,1.f,1.f}
    };

    // Reusable accumulator for Σ_{bit=1} A[i][p] per column.
    float* sum_one = new float[K];

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        // Compute total sum of this row.
        float row_sum = 0.0f;
        for (size_t p = 0; p < K; ++p) row_sum += a_row[p];

        // Zero accumulator.
        for (size_t j = 0; j < K; ++j) sum_one[j] = 0.0f;

        // Accumulate contributions where B bit == 1.
        for (size_t p = 0; p < K; ++p) {
            float a_val = a_row[p];
            const uint32_t* b_row = B + p * K_ints;
            for (size_t block = 0; block < K_ints; ++block) {
                uint32_t packed = b_row[block];
                // Process 32 bits as 8 nibbles (4 bits each).
                for (size_t g = 0; g < 8; ++g) {
                    uint32_t nibble = (packed >> (g * 4)) & 0xF;
                    float* base = sum_one + block * 32 + g * 4;
                    // Load current accumulator.
                    float32x4_t acc = vld1q_f32(base);
                    // Load mask for this nibble (0/1 floats).
                    float32x4_t mask = vld1q_f32(maskLUT[nibble]);
                    // acc += mask * a_val;
                    acc = vmlaq_n_f32(acc, mask, a_val);
                    vst1q_f32(base, acc);
                }
            }
        }

        // Write final results for this row.
        float* c_row = C + i * K;
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 2.0f * sum_one[j] - row_sum;
        }
    }
    delete[] sum_one;
}
