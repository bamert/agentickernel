#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;

    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const size_t packed_idx = j >> 5;
            const uint32_t bit_mask = 1u << (j & 31);
            const uint32_t* brow = B + packed_idx;
            float sum = 0.0f;

            size_t p = 0;
            for (; p + 3 < K; p += 4) {
                const float a0 = arow[p + 0];
                const float a1 = arow[p + 1];
                const float a2 = arow[p + 2];
                const float a3 = arow[p + 3];

                const uint32_t b0 = brow[(p + 0) * K_ints];
                const uint32_t b1 = brow[(p + 1) * K_ints];
                const uint32_t b2 = brow[(p + 2) * K_ints];
                const uint32_t b3 = brow[(p + 3) * K_ints];

                sum += a0 * ((b0 & bit_mask) ? 1.0f : -1.0f);
                sum += a1 * ((b1 & bit_mask) ? 1.0f : -1.0f);
                sum += a2 * ((b2 & bit_mask) ? 1.0f : -1.0f);
                sum += a3 * ((b3 & bit_mask) ? 1.0f : -1.0f);
            }
            for (; p < K; ++p) {
                const uint32_t packed = brow[p * K_ints];
                sum += arow[p] * ((packed & bit_mask) ? 1.0f : -1.0f);
            }

            crow[j] = sum;
        }
    }
}
