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
                const uint32_t p0 = b_col[(p + 0) * K_words];
                const uint32_t p1 = b_col[(p + 1) * K_words];
                const uint32_t p2 = b_col[(p + 2) * K_words];
                const uint32_t p3 = b_col[(p + 3) * K_words];
                const uint32_t p4 = b_col[(p + 4) * K_words];
                const uint32_t p5 = b_col[(p + 5) * K_words];
                const uint32_t p6 = b_col[(p + 6) * K_words];
                const uint32_t p7 = b_col[(p + 7) * K_words];
                const uint32_t p8 = b_col[(p + 8) * K_words];
                const uint32_t p9 = b_col[(p + 9) * K_words];
                const uint32_t p10 = b_col[(p + 10) * K_words];
                const uint32_t p11 = b_col[(p + 11) * K_words];
                const uint32_t p12 = b_col[(p + 12) * K_words];
                const uint32_t p13 = b_col[(p + 13) * K_words];
                const uint32_t p14 = b_col[(p + 14) * K_words];
                const uint32_t p15 = b_col[(p + 15) * K_words];

                sum0 += a_row[p + 0] * ((p0 & mask) ? 1.0f : -1.0f);
                sum1 += a_row[p + 1] * ((p1 & mask) ? 1.0f : -1.0f);
                sum2 += a_row[p + 2] * ((p2 & mask) ? 1.0f : -1.0f);
                sum3 += a_row[p + 3] * ((p3 & mask) ? 1.0f : -1.0f);
                sum4 += a_row[p + 4] * ((p4 & mask) ? 1.0f : -1.0f);
                sum5 += a_row[p + 5] * ((p5 & mask) ? 1.0f : -1.0f);
                sum6 += a_row[p + 6] * ((p6 & mask) ? 1.0f : -1.0f);
                sum7 += a_row[p + 7] * ((p7 & mask) ? 1.0f : -1.0f);
                sum8 += a_row[p + 8] * ((p8 & mask) ? 1.0f : -1.0f);
                sum9 += a_row[p + 9] * ((p9 & mask) ? 1.0f : -1.0f);
                sum10 += a_row[p + 10] * ((p10 & mask) ? 1.0f : -1.0f);
                sum11 += a_row[p + 11] * ((p11 & mask) ? 1.0f : -1.0f);
                sum12 += a_row[p + 12] * ((p12 & mask) ? 1.0f : -1.0f);
                sum13 += a_row[p + 13] * ((p13 & mask) ? 1.0f : -1.0f);
                sum14 += a_row[p + 14] * ((p14 & mask) ? 1.0f : -1.0f);
                sum15 += a_row[p + 15] * ((p15 & mask) ? 1.0f : -1.0f);
            }

            float sum = ((((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7))) +
                         (((sum8 + sum9) + (sum10 + sum11)) + ((sum12 + sum13) + (sum14 + sum15))));
            for (; p < K; ++p) {
                const uint32_t packed = b_col[p * K_words];
                sum += a_row[p] * ((packed & mask) ? 1.0f : -1.0f);
            }

            c_row[j] = sum;
        }
    }
}
