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
                const uint32_t b0 = b_col[(p + 0) * K_words];
                const uint32_t b1 = b_col[(p + 1) * K_words];
                const uint32_t b2 = b_col[(p + 2) * K_words];
                const uint32_t b3 = b_col[(p + 3) * K_words];
                const uint32_t b4 = b_col[(p + 4) * K_words];
                const uint32_t b5 = b_col[(p + 5) * K_words];
                const uint32_t b6 = b_col[(p + 6) * K_words];
                const uint32_t b7 = b_col[(p + 7) * K_words];

                const float s0 = 1.0f - 2.0f * float((b0 & mask) == 0);
                const float s1 = 1.0f - 2.0f * float((b1 & mask) == 0);
                const float s2 = 1.0f - 2.0f * float((b2 & mask) == 0);
                const float s3 = 1.0f - 2.0f * float((b3 & mask) == 0);
                const float s4 = 1.0f - 2.0f * float((b4 & mask) == 0);
                const float s5 = 1.0f - 2.0f * float((b5 & mask) == 0);
                const float s6 = 1.0f - 2.0f * float((b6 & mask) == 0);
                const float s7 = 1.0f - 2.0f * float((b7 & mask) == 0);

                sum0 += a_row[p + 0] * s0;
                sum1 += a_row[p + 1] * s1;
                sum2 += a_row[p + 2] * s2;
                sum3 += a_row[p + 3] * s3;
                sum4 += a_row[p + 4] * s4;
                sum5 += a_row[p + 5] * s5;
                sum6 += a_row[p + 6] * s6;
                sum7 += a_row[p + 7] * s7;
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
