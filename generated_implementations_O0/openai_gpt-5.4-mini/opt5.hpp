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
            const uint32_t* brow = B + packed_idx;
            float sum = 0.0f;

            size_t p = 0;
            for (; p + 15 < K; p += 16) {
                sum += arow[p + 0]  * ((brow[(p + 0)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 1]  * ((brow[(p + 1)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 2]  * ((brow[(p + 2)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 3]  * ((brow[(p + 3)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 4]  * ((brow[(p + 4)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 5]  * ((brow[(p + 5)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 6]  * ((brow[(p + 6)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 7]  * ((brow[(p + 7)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 8]  * ((brow[(p + 8)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 9]  * ((brow[(p + 9)  * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 10] * ((brow[(p + 10) * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 11] * ((brow[(p + 11) * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 12] * ((brow[(p + 12) * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 13] * ((brow[(p + 13) * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 14] * ((brow[(p + 14) * K_ints] & bit_mask) ? pos : neg);
                sum += arow[p + 15] * ((brow[(p + 15) * K_ints] & bit_mask) ? pos : neg);
            }
            for (; p < K; ++p) {
                sum += arow[p] * ((brow[p * K_ints] & bit_mask) ? pos : neg);
            }

            crow[j] = sum;
        }
    }
}
