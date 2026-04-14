#pragma once

#include <cstddef>
#include <cstdint>
#include <array>

// Compile‑time generated table: for each 8‑bit pattern we store 8 floats (+1.0f or -1.0f).
constexpr auto generate_sign_table() {
    std::array<std::array<float, 8>, 256> tbl{};
    for (size_t byte = 0; byte < 256; ++byte) {
        for (size_t bit = 0; bit < 8; ++bit) {
            tbl[byte][bit] = ((byte >> bit) & 1u) ? 1.0f : -1.0f;
        }
    }
    return tbl;
}
constexpr auto sign_table = generate_sign_table();

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32; // number of 32‑bit words per row of B
    for (size_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * K;
        // Initialise output row to zeros
        for (size_t j = 0; j < K; ++j) {
            c_row[j] = 0.0f;
        }
        // Process each 32‑bit block (32 columns) of B
        for (size_t block = 0; block < K_ints; ++block) {
            size_t base = block * 32;
            for (size_t p = 0; p < K; ++p) {
                float a_val = a_row[p];
                uint32_t packed = B[p * K_ints + block];
                // split into four bytes
                uint8_t b0 = packed & 0xFFu;
                uint8_t b1 = (packed >> 8) & 0xFFu;
                uint8_t b2 = (packed >> 16) & 0xFFu;
                uint8_t b3 = (packed >> 24) & 0xFFu;
                const auto& tbl0 = sign_table[b0];
                const auto& tbl1 = sign_table[b1];
                const auto& tbl2 = sign_table[b2];
                const auto& tbl3 = sign_table[b3];
                // unrolled accumulation directly into C
                c_row[base + 0]  += a_val * tbl0[0];  c_row[base + 1]  += a_val * tbl0[1];
                c_row[base + 2]  += a_val * tbl0[2];  c_row[base + 3]  += a_val * tbl0[3];
                c_row[base + 4]  += a_val * tbl0[4];  c_row[base + 5]  += a_val * tbl0[5];
                c_row[base + 6]  += a_val * tbl0[6];  c_row[base + 7]  += a_val * tbl0[7];
                c_row[base + 8]  += a_val * tbl1[0];  c_row[base + 9]  += a_val * tbl1[1];
                c_row[base +10]  += a_val * tbl1[2];  c_row[base +11]  += a_val * tbl1[3];
                c_row[base +12]  += a_val * tbl1[4];  c_row[base +13]  += a_val * tbl1[5];
                c_row[base +14]  += a_val * tbl1[6];  c_row[base +15]  += a_val * tbl1[7];
                c_row[base +16]  += a_val * tbl2[0];  c_row[base +17]  += a_val * tbl2[1];
                c_row[base +18]  += a_val * tbl2[2];  c_row[base +19]  += a_val * tbl2[3];
                c_row[base +20]  += a_val * tbl2[4];  c_row[base +21]  += a_val * tbl2[5];
                c_row[base +22]  += a_val * tbl2[6];  c_row[base +23]  += a_val * tbl2[7];
                c_row[base +24]  += a_val * tbl3[0];  c_row[base +25]  += a_val * tbl3[1];
                c_row[base +26]  += a_val * tbl3[2];  c_row[base +27]  += a_val * tbl3[3];
                c_row[base +28]  += a_val * tbl3[4];  c_row[base +29]  += a_val * tbl3[5];
                c_row[base +30]  += a_val * tbl3[6];  c_row[base +31]  += a_val * tbl3[7];
            }
        }
    }
}
