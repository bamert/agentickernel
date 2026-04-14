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

            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
            size_t p = 0;
            for (; p + 31 < K; p += 32) {
                sum0 += arow[p + 0]  * ((brow[(p + 0)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 1]  * ((brow[(p + 1)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 2]  * ((brow[(p + 2)  * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 3]  * ((brow[(p + 3)  * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 4]  * ((brow[(p + 4)  * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 5]  * ((brow[(p + 5)  * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 6]  * ((brow[(p + 6)  * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 7]  * ((brow[(p + 7)  * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 8]  * ((brow[(p + 8)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 9]  * ((brow[(p + 9)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 10] * ((brow[(p + 10) * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 11] * ((brow[(p + 11) * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 12] * ((brow[(p + 12) * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 13] * ((brow[(p + 13) * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 14] * ((brow[(p + 14) * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 15] * ((brow[(p + 15) * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 16] * ((brow[(p + 16) * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 17] * ((brow[(p + 17) * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 18] * ((brow[(p + 18) * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 19] * ((brow[(p + 19) * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 20] * ((brow[(p + 20) * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 21] * ((brow[(p + 21) * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 22] * ((brow[(p + 22) * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 23] * ((brow[(p + 23) * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 24] * ((brow[(p + 24) * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 25] * ((brow[(p + 25) * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 26] * ((brow[(p + 26) * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 27] * ((brow[(p + 27) * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 28] * ((brow[(p + 28) * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 29] * ((brow[(p + 29) * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 30] * ((brow[(p + 30) * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 31] * ((brow[(p + 31) * K_ints] & bit_mask) ? pos : neg);
            }
            for (; p < K; ++p) {
                sum0 += arow[p] * ((brow[p * K_ints] & bit_mask) ? pos : neg);
            }

            crow[j] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
        }
    }
}
