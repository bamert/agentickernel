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
            for (; p + 63 < K; p += 64) {
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
                sum0 += arow[p + 16] * ((brow[(p + 16)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 17] * ((brow[(p + 17)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 18] * ((brow[(p + 18)  * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 19] * ((brow[(p + 19)  * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 20] * ((brow[(p + 20)  * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 21] * ((brow[(p + 21)  * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 22] * ((brow[(p + 22)  * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 23] * ((brow[(p + 23)  * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 24] * ((brow[(p + 24)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 25] * ((brow[(p + 25)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 26] * ((brow[(p + 26)  * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 27] * ((brow[(p + 27)  * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 28] * ((brow[(p + 28)  * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 29] * ((brow[(p + 29)  * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 30] * ((brow[(p + 30)  * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 31] * ((brow[(p + 31)  * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 32] * ((brow[(p + 32)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 33] * ((brow[(p + 33)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 34] * ((brow[(p + 34)  * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 35] * ((brow[(p + 35)  * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 36] * ((brow[(p + 36)  * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 37] * ((brow[(p + 37)  * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 38] * ((brow[(p + 38)  * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 39] * ((brow[(p + 39)  * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 40] * ((brow[(p + 40)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 41] * ((brow[(p + 41)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 42] * ((brow[(p + 42)  * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 43] * ((brow[(p + 43)  * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 44] * ((brow[(p + 44)  * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 45] * ((brow[(p + 45)  * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 46] * ((brow[(p + 46)  * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 47] * ((brow[(p + 47)  * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 48] * ((brow[(p + 48)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 49] * ((brow[(p + 49)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 50] * ((brow[(p + 50)  * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 51] * ((brow[(p + 51)  * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 52] * ((brow[(p + 52)  * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 53] * ((brow[(p + 53)  * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 54] * ((brow[(p + 54)  * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 55] * ((brow[(p + 55)  * K_ints] & bit_mask) ? pos : neg);
                sum0 += arow[p + 56] * ((brow[(p + 56)  * K_ints] & bit_mask) ? pos : neg);
                sum1 += arow[p + 57] * ((brow[(p + 57)  * K_ints] & bit_mask) ? pos : neg);
                sum2 += arow[p + 58] * ((brow[(p + 58)  * K_ints] & bit_mask) ? pos : neg);
                sum3 += arow[p + 59] * ((brow[(p + 59)  * K_ints] & bit_mask) ? pos : neg);
                sum4 += arow[p + 60] * ((brow[(p + 60)  * K_ints] & bit_mask) ? pos : neg);
                sum5 += arow[p + 61] * ((brow[(p + 61)  * K_ints] & bit_mask) ? pos : neg);
                sum6 += arow[p + 62] * ((brow[(p + 62)  * K_ints] & bit_mask) ? pos : neg);
                sum7 += arow[p + 63] * ((brow[(p + 63)  * K_ints] & bit_mask) ? pos : neg);
            }
            for (; p < K; ++p) {
                sum0 += arow[p] * ((brow[p * K_ints] & bit_mask) ? pos : neg);
            }

            crow[j] = (((sum0 + sum1) + (sum2 + sum3)) + ((sum4 + sum5) + (sum6 + sum7)));
        }
    }
}
