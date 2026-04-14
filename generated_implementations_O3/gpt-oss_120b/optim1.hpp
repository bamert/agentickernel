#pragma once

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    // Process each row of A
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // Process columns in blocks of 32
        for (size_t block = 0; block < K_ints; ++block) {
            // accumulator for 32 columns
            float acc[32];
            // initialize accumulators to zero
            for (int t = 0; t < 32; ++t) acc[t] = 0.0f;
            // iterate over K dimension
            for (size_t p = 0; p < K; ++p) {
                float a_val = a_row[p];
                uint32_t packed = B[p * K_ints + block];
                // expand bits to signs and accumulate
                // unrolled 8 bits per iteration for slight speed-up
                uint32_t bits = packed;
                for (int b = 0; b < 32; ++b) {
                    // extract sign: 1 -> +1, 0 -> -1
                    float sign = (bits & 1u) ? 1.0f : -1.0f;
                    acc[b] += a_val * sign;
                    bits >>= 1u;
                }
            }
            // store results
            size_t base = block * 32;
            for (int t = 0; t < 32; ++t) {
                c_row[base + t] = acc[t];
            }
        }
    }
}
