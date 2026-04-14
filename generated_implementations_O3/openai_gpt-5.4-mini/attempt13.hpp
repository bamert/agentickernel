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
            float sum8 = 0.0f;
            float sum9 = 0.0f;
            float sum10 = 0.0f;
            float sum11 = 0.0f;
            float sum12 = 0.0f;
            float sum13 = 0.0f;
            float sum14 = 0.0f;
            float sum15 = 0.0f;

            size_t p = 0;
            for (; p + 15 < K; p += 16) {
                const uint32_t b0 = b_col[(p + 0) * K_words];
                const uint32_t b1 = b_col[(p + 1) * K_words];
                const uint32_t b2 = b_col[(p + 2) * K_words];
                const uint32_t b3 = b_col[(p + 3) * K_words];
                const uint32_t b4 = b_col[(p + 4) * K_words];
                const uint32_t b5 = b_col[(p + 5) * K_words];
                const uint32_t b6 = b_col[(p + 6) * K_words];
                const uint32_t b7 = b_col[(p + 7) * K_words];
                const uint32_t b8 = b_col[(p + 8) * K_words];
                const uint32_t b9 = b_col[(p + 9) * K_words];
                const uint32_t b10 = b_col[(p + 10) * K_words];
                const uint32_t b11 = b_col[(p + 11) * K_words];
                const uint32_t b12 = b_col[(p + 12) * K_words];
                const uint32_t b13 = b_col[(p + 13) * K_words];
                const uint32_t b14 = b_col[(p + 14) * K_words];
                const uint32_t b15 = b_col[(p + 15) * K_words];

                sum0 += a_row[p + 0] * (float)((b0 & mask) ? 1 : -1);
                sum1 += a_row[p + 1] * (float)((b1 & mask) ? 1 : -1);
                sum2 += a_row[p + 2] * (float)((b2 & mask) ? 1 : -1);
                sum3 += a_row[p + 3] * (float)((b3 & mask) ? 1 : -1);
                sum4 += a_row[p + 4] * (float)((b4 & mask) ? 1 : -1);
                sum5 += a_row[p + 5] * (float)((b5 & mask) ? 1 : -1);
                sum6 += a_row[p + 6] * (float)((b6 & mask) ? 1 : -1);
                sum7 += a_row[p + 7] * (float)((b7 & mask) ? 1 : -1);
                sum8 += a_row[p + 8] * (float)((b8 & mask) ? 1 : -1);
                sum9 += a_row[p + 9] * (float)((b9 & mask) ? 1 : -1);
                sum10 += a_row[p + 10] * (float)((b10 & mask) ? 1 : -1);
                sum11 += a_row[p + 11] * (float)((b11 & mask) ? 1 : -1);
                sum12 += a_row[p + 12] * (float)((b12 & mask) ? 1 : -1);
                sum13 += a_row[p + 13] * (float)((b13 & mask) ? 1 : -1);
                sum14 += a_row[p + 14] * (float)((b14 & mask) ? 1 : -1);
                sum15 += a_row[p + 15] * (float)((b15 & mask) ? 1 : -1);
            }

            float sum = ((((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7))) +
                         (((sum8 + sum9) + (sum10 + sum11)) + ((sum12 + sum13) + (sum14 + sum15))));
            for (; p < K; ++p) {
                const uint32_t packed = b_col[p * K_words];
                sum += a_row[p] * (float)((packed & mask) ? 1 : -1);
            }

            c_row[j] = sum;
        }
    }
}
