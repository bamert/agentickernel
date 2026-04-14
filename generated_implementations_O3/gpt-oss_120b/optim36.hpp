#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time sign table: 8 floats (+1.0f or -1.0f) for each possible byte value.
constexpr std::array<std::array<float, 8>, 256> sign_table = [](){
    std::array<std::array<float, 8>, 256> tbl{};
    for (size_t b = 0; b < 256; ++b) {
        for (size_t bit = 0; bit < 8; ++bit) {
            tbl[b][bit] = ((b >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}();

// Matrix multiply processing two rows of A (and C) together.
// Inner K loop is unrolled by 8 to minimise loop overhead.
// A: M×K float (row‑major)
// B: K×K binary packed (1 → +1.0f, 0 → -1.0f) stored as K/32 uint32_t per row
// C: M×K float (row‑major)
// K is a multiple of 32 (therefore also a multiple of 8).
void matmul(const float* __restrict A,
            const uint32_t* __restrict B,
            float* __restrict C,
            size_t M,
            size_t K)
{
    const size_t K_ints = K / 32; // words per row of B
    size_t i = 0;

    // Process pairs of rows.
    for (; i + 1 < M; i += 2) {
        const float* a_row0 = A + i * K;
        const float* a_row1 = A + (i + 1) * K;
        float* c_row0 = C + i * K;
        float* c_row1 = C + (i + 1) * K;

        for (size_t block = 0; block < K_ints; ++block) {
            float acc0[32] = {};
            float acc1[32] = {};
            const size_t base = block * 32;
            const uint32_t* b_ptr = B + block; // start of this 32‑column block in row 0 of B

            // Inner K loop: K is a multiple of 8.
            for (size_t p = 0; p + 7 < K; p += 8) {
                // Prefetch next B words and A values.
                __builtin_prefetch(b_ptr + 64);
                __builtin_prefetch(a_row0 + p + 16);
                __builtin_prefetch(a_row1 + p + 16);

                // ---- element p ----
                float a0 = a_row0[p];
                float a1 = a_row1[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8)  & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& t0 = sign_table[b0];
                const auto& t1 = sign_table[b1];
                const auto& t2 = sign_table[b2];
                const auto& t3 = sign_table[b3];
                // Unrolled column accumulation (8 columns per byte).
                acc0[0]  += a0 * t0[0];  acc1[0]  += a1 * t0[0];  acc0[8]  += a0 * t1[0];  acc1[8]  += a1 * t1[0];
                acc0[1]  += a0 * t0[1];  acc1[1]  += a1 * t0[1];  acc0[9]  += a0 * t1[1];  acc1[9]  += a1 * t1[1];
                acc0[2]  += a0 * t0[2];  acc1[2]  += a1 * t0[2];  acc0[10] += a0 * t1[2];  acc1[10] += a1 * t1[2];
                acc0[3]  += a0 * t0[3];  acc1[3]  += a1 * t0[3];  acc0[11] += a0 * t1[3];  acc1[11] += a1 * t1[3];
                acc0[4]  += a0 * t0[4];  acc1[4]  += a1 * t0[4];  acc0[12] += a0 * t1[4];  acc1[12] += a1 * t1[4];
                acc0[5]  += a0 * t0[5];  acc1[5]  += a1 * t0[5];  acc0[13] += a0 * t1[5];  acc1[13] += a1 * t1[5];
                acc0[6]  += a0 * t0[6];  acc1[6]  += a1 * t0[6];  acc0[14] += a0 * t1[6];  acc1[14] += a1 * t1[6];
                acc0[7]  += a0 * t0[7];  acc1[7]  += a1 * t0[7];  acc0[15] += a0 * t1[7];  acc1[15] += a1 * t1[7];
                acc0[16] += a0 * t2[0];  acc1[16] += a1 * t2[0];  acc0[24] += a0 * t3[0];  acc1[24] += a1 * t3[0];
                acc0[17] += a0 * t2[1];  acc1[17] += a1 * t2[1];  acc0[25] += a0 * t3[1];  acc1[25] += a1 * t3[1];
                acc0[18] += a0 * t2[2];  acc1[18] += a1 * t2[2];  acc0[26] += a0 * t3[2];  acc1[26] += a1 * t3[2];
                acc0[19] += a0 * t2[3];  acc1[19] += a1 * t2[3];  acc0[27] += a0 * t3[3];  acc1[27] += a1 * t3[3];
                acc0[20] += a0 * t2[4];  acc1[20] += a1 * t2[4];  acc0[28] += a0 * t3[4];  acc1[28] += a1 * t3[4];
                acc0[21] += a0 * t2[5];  acc1[21] += a1 * t2[5];  acc0[29] += a0 * t3[5];  acc1[29] += a1 * t3[5];
                acc0[22] += a0 * t2[6];  acc1[22] += a1 * t2[6];  acc0[30] += a0 * t3[6];  acc1[30] += a1 * t3[6];
                acc0[23] += a0 * t2[7];  acc1[23] += a1 * t2[7];  acc0[31] += a0 * t3[7];  acc1[31] += a1 * t3[7];

                // ---- element p+1 ----
                a0 = a_row0[p + 1];
                a1 = a_row1[p + 1];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8)  & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& t0_1 = sign_table[b0];
                const auto& t1_1 = sign_table[b1];
                const auto& t2_1 = sign_table[b2];
                const auto& t3_1 = sign_table[b3];
                acc0[0]  += a0 * t0_1[0];  acc1[0]  += a1 * t0_1[0];  acc0[8]  += a0 * t1_1[0];  acc1[8]  += a1 * t1_1[0];
                acc0[1]  += a0 * t0_1[1];  acc1[1]  += a1 * t0_1[1];  acc0[9]  += a0 * t1_1[1];  acc1[9]  += a1 * t1_1[1];
                acc0[2]  += a0 * t0_1[2];  acc1[2]  += a1 * t0_1[2];  acc0[10] += a0 * t1_1[2];  acc1[10] += a1 * t1_1[2];
                acc0[3]  += a0 * t0_1[3];  acc1[3]  += a1 * t0_1[3];  acc0[11] += a0 * t1_1[3];  acc1[11] += a1 * t1_1[3];
                acc0[4]  += a0 * t0_1[4];  acc1[4]  += a1 * t0_1[4];  acc0[12] += a0 * t1_1[4];  acc1[12] += a1 * t1_1[4];
                acc0[5]  += a0 * t0_1[5];  acc1[5]  += a1 * t0_1[5];  acc0[13] += a0 * t1_1[5];  acc1[13] += a1 * t1_1[5];
                acc0[6]  += a0 * t0_1[6];  acc1[6]  += a1 * t0_1[6];  acc0[14] += a0 * t1_1[6];  acc1[14] += a1 * t1_1[6];
                acc0[7]  += a0 * t0_1[7];  acc1[7]  += a1 * t0_1[7];  acc0[15] += a0 * t1_1[7];  acc1[15] += a1 * t1_1[7];
                // columns 16‑23
                for (int c = 0; c < 8; ++c) {
                    acc0[16 + c] += a0 * t2_1[c];
                    acc1[16 + c] += a1 * t2_1[c];
                    acc0[24 + c] += a0 * t3_1[c];
                    acc1[24 + c] += a1 * t3_1[c];
                }
                // ---- element p+2 ----
                a0 = a_row0[p + 2]; a1 = a_row1[p + 2];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& t0_2 = sign_table[b0];
                const auto& t1_2 = sign_table[b1];
                const auto& t2_2 = sign_table[b2];
                const auto& t3_2 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * t0_2[c];
                    acc1[c]      += a1 * t0_2[c];
                    acc0[8 + c]  += a0 * t1_2[c];
                    acc1[8 + c]  += a1 * t1_2[c];
                    acc0[16 + c] += a0 * t2_2[c];
                    acc1[16 + c] += a1 * t2_2[c];
                    acc0[24 + c] += a0 * t3_2[c];
                    acc1[24 + c] += a1 * t3_2[c];
                }
                // ---- element p+3 ----
                a0 = a_row0[p + 3];
                a1 = a_row1[p + 3];
                packed = *b_ptr; b_ptr += K_ints;
                b0 = packed & 0xFFu; b1 = (packed >> 8) & 0xFFu; b2 = (packed >> 16) & 0xFFu; b3 = (packed >> 24) & 0xFFu;
                const auto& t0_3 = sign_table[b0];
                const auto& t1_3 = sign_table[b1];
                const auto& t2_3 = sign_table[b2];
                const auto& t3_3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc0[c]      += a0 * t0_3[c];
                    acc1[c]      += a1 * t0_3[c];
                    acc0[8 + c]  += a0 * t1_3[c];
                    acc1[8 + c]  += a1 * t1_3[c];
                    acc0[16 + c] += a0 * t2_3[c];
                    acc1[16 + c] += a1 * t2_3[c];
                    acc0[24 + c] += a0 * t3_3[c];
                    acc1[24 + c] += a1 * t3_3[c];
                }
            }
            // Store results for the two rows.
            for (int t = 0; t < 32; ++t) {
                c_row0[base + t] = acc0[t];
                c_row1[base + t] = acc1[t];
            }
        }
    }

    // Handle remaining single row if M is odd.
    if (i < M) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        for (size_t block = 0; block < K_ints; ++block) {
            float acc[32] = {};
            const size_t base = block * 32;
            const uint32_t* b_ptr = B + block;
            for (size_t p = 0; p + 7 < K; p += 8) {
                __builtin_prefetch(b_ptr + 32);
                for (size_t off = 0; off < 8; ++off) {
                    float a_val = a_row[p + off];
                    uint32_t packed = *b_ptr; b_ptr += K_ints;
                    uint8_t b0 = packed & 0xFFu;
                    uint8_t b1 = (packed >> 8)  & 0xFFu;
                    uint8_t b2 = (packed >> 16) & 0xFFu;
                    uint8_t b3 = (packed >> 24) & 0xFFu;
                    const auto& t0 = sign_table[b0];
                    const auto& t1 = sign_table[b1];
                    const auto& t2 = sign_table[b2];
                    const auto& t3 = sign_table[b3];
                    for (int c = 0; c < 8; ++c) {
                        acc[c]      += a_val * t0[c];
                        acc[8 + c]  += a_val * t1[c];
                        acc[16 + c] += a_val * t2[c];
                        acc[24 + c] += a_val * t3[c];
                    }
                }
            }
            for (size_t p = K - (K % 8); p < K; ++p) {
                float a_val = a_row[p];
                uint32_t packed = *b_ptr; b_ptr += K_ints;
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8)  & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& t0 = sign_table[b0];
                const auto& t1 = sign_table[b1];
                const auto& t2 = sign_table[b2];
                const auto& t3 = sign_table[b3];
                for (int c = 0; c < 8; ++c) {
                    acc[c]      += a_val * t0[c];
                    acc[8 + c]  += a_val * t1[c];
                    acc[16 + c] += a_val * t2[c];
                    acc[24 + c] += a_val * t3[c];
                }
            }
            for (int t = 0; t < 32; ++t) {
                c_row[base + t] = acc[t];
            }
        }
    }
}
