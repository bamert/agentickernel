#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    const float pos = 1.0f;
    const float neg = -1.0f;

    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;

        for (size_t j = 0; j < K; ++j) {
            const size_t packed_idx = j >> 5;
            const uint32_t bit_mask = 1u << (j & 31);
            float sum = 0.0f;

            size_t p = 0;
            for (; p + 15 < K; p += 16) {
                const uint32_t b0  = B[(p + 0)  * K_ints + packed_idx];
                const uint32_t b1  = B[(p + 1)  * K_ints + packed_idx];
                const uint32_t b2  = B[(p + 2)  * K_ints + packed_idx];
                const uint32_t b3  = B[(p + 3)  * K_ints + packed_idx];
                const uint32_t b4  = B[(p + 4)  * K_ints + packed_idx];
                const uint32_t b5  = B[(p + 5)  * K_ints + packed_idx];
                const uint32_t b6  = B[(p + 6)  * K_ints + packed_idx];
                const uint32_t b7  = B[(p + 7)  * K_ints + packed_idx];
                const uint32_t b8  = B[(p + 8)  * K_ints + packed_idx];
                const uint32_t b9  = B[(p + 9)  * K_ints + packed_idx];
                const uint32_t b10 = B[(p + 10) * K_ints + packed_idx];
                const uint32_t b11 = B[(p + 11) * K_ints + packed_idx];
                const uint32_t b12 = B[(p + 12) * K_ints + packed_idx];
                const uint32_t b13 = B[(p + 13) * K_ints + packed_idx];
                const uint32_t b14 = B[(p + 14) * K_ints + packed_idx];
                const uint32_t b15 = B[(p + 15) * K_ints + packed_idx];

                sum += arow[p + 0]  * ((b0  & bit_mask) ? pos : neg);
                sum += arow[p + 1]  * ((b1  & bit_mask) ? pos : neg);
                sum += arow[p + 2]  * ((b2  & bit_mask) ? pos : neg);
                sum += arow[p + 3]  * ((b3  & bit_mask) ? pos : neg);
                sum += arow[p + 4]  * ((b4  & bit_mask) ? pos : neg);
                sum += arow[p + 5]  * ((b5  & bit_mask) ? pos : neg);
                sum += arow[p + 6]  * ((b6  & bit_mask) ? pos : neg);
                sum += arow[p + 7]  * ((b7  & bit_mask) ? pos : neg);
                sum += arow[p + 8]  * ((b8  & bit_mask) ? pos : neg);
                sum += arow[p + 9]  * ((b9  & bit_mask) ? pos : neg);
                sum += arow[p + 10] * ((b10 & bit_mask) ? pos : neg);
                sum += arow[p + 11] * ((b11 & bit_mask) ? pos : neg);
                sum += arow[p + 12] * ((b12 & bit_mask) ? pos : neg);
                sum += arow[p + 13] * ((b13 & bit_mask) ? pos : neg);
                sum += arow[p + 14] * ((b14 & bit_mask) ? pos : neg);
                sum += arow[p + 15] * ((b15 & bit_mask) ? pos : neg);
            }
            for (; p < K; ++p) {
                const uint32_t packed = B[p * K_ints + packed_idx];
                sum += arow[p] * ((packed & bit_mask) ? pos : neg);
            }

            crow[j] = sum;
        }
    }
}
