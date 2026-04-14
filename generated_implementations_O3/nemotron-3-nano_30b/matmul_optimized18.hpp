#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5; // K / 32

    // nibble->sign lookup table
    const float nibbleSigns[16][4] = {
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1},
        {-1,-1,+1,-1}, {+1,-1,+1,-1}, {-1,+1,+1,-1}, {+1,+1,+1,-1},
        {-1,-1,-1,+1}, {+1,-1,-1,+1}, {-1,+1,-1,+1}, {+1,+1,-1,+1},
        {-1,-1,+1,+1}, {+1,-1,+1,+1}, {-1,+1,+1,+1}, {+1,+1,+1,+1},
        {-1,-1,-1,-1}, {+1,-1,-1,-1}, {-1,+1,-1,-1}, {+1,+1,-1,-1}
    };

    for (size_t i = 0; i < M; ++i) {
        float accum[/*K*/];
        for (size_t j = 0; j < K; ++j) accum[j] = 0.0f;

        const float* A_row = A + i * K;
        float*       C_row = C + i * K;

        for (size_t p = 0; p < K; ++p) {
            float a_val = A_row[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t chunk = 0; chunk < K_ints; ++chunk) {
                uint32_t word = B_row[chunk];
                size_t   col_base = chunk * 32;

#pragma unroll 8
                for (int offset = 0; offset < 32; offset += 4) {
                    size_t idx0 = col_base + offset + 0;
                    size_t idx1 = col_base + offset + 1;
                    size_t idx2 = col_base + offset + 2;
                    size_t idx3 = col_base + offset + 3;

                    uint32_t nibble = (word >> offset) & 0xF;
                    const float* signs = nibbleSigns[nibble];

                    accum[idx0] += a_val * signs[0];
                    accum[idx1] += a_val * signs[1];
                    accum[idx2] += a_val * signs[2];
                    accum[idx3] += a_val * signs[3];
                }
            }
        }

        for (size_t j = 0; j < K; ++j) {
            C_row[j] = accum[j];
        }
    }
}