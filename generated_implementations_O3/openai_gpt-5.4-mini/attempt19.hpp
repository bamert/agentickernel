#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_words = K >> 5;

    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const uint32_t* b_col = B + (j >> 5);
            const uint32_t mask = 1u << (j & 31);

            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;
            float sum4 = 0.0f;
            float sum5 = 0.0f;
            float sum6 = 0.0f;
            float sum7 = 0.0f;

            size_t p = 0;
            for (; p + 7 < K; p += 8) {
                const float a0 = a_row[p + 0];
                const float a1 = a_row[p + 1];
                const float a2 = a_row[p + 2];
                const float a3 = a_row[p + 3];
                const float a4 = a_row[p + 4];
                const float a5 = a_row[p + 5];
                const float a6 = a_row[p + 6];
                const float a7 = a_row[p + 7];

                const uint32_t b0 = b_col[(p + 0) * K_words];
                const uint32_t b1 = b_col[(p + 1) * K_words];
                const uint32_t b2 = b_col[(p + 2) * K_words];
                const uint32_t b3 = b_col[(p + 3) * K_words];
                const uint32_t b4 = b_col[(p + 4) * K_words];
                const uint32_t b5 = b_col[(p + 5) * K_words];
                const uint32_t b6 = b_col[(p + 6) * K_words];
                const uint32_t b7 = b_col[(p + 7) * K_words];

                const uint32_t m0 = (b0 & mask) != 0;
                const uint32_t m1 = (b1 & mask) != 0;
                const uint32_t m2 = (b2 & mask) != 0;
                const uint32_t m3 = (b3 & mask) != 0;
                const uint32_t m4 = (b4 & mask) != 0;
                const uint32_t m5 = (b5 & mask) != 0;
                const uint32_t m6 = (b6 & mask) != 0;
                const uint32_t m7 = (b7 & mask) != 0;

                sum0 += a0 * (float)(m0 + m0 - 1u);
                sum1 += a1 * (float)(m1 + m1 - 1u);
                sum2 += a2 * (float)(m2 + m2 - 1u);
                sum3 += a3 * (float)(m3 + m3 - 1u);
                sum4 += a4 * (float)(m4 + m4 - 1u);
                sum5 += a5 * (float)(m5 + m5 - 1u);
                sum6 += a6 * (float)(m6 + m6 - 1u);
                sum7 += a7 * (float)(m7 + m7 - 1u);
            }

            float sum = (((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7)));
            for (; p < K; ++p) {
                const uint32_t packed = b_col[p * K_words];
                sum += a_row[p] * ((packed & mask) ? 1.0f : -1.0f);
            }

            c_row[j] = sum;
        }
    }
}
