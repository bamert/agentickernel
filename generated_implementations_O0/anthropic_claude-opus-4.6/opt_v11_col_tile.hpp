
#pragma once
#include <cstdint>
#include <cstddef>

// v7 was best at 188ms. Let's try tiling the output columns to fit in L1 cache.
// L1 is 64KB. With 4 rows of C, each tile of T columns = 4*T*4 bytes for C.
// Plus we read A (4 floats per p) and B (T/32 uint32s per p).
// For T=512: C tiles = 4*512*4 = 8KB. Good for L1.
// B tile per p = 16 uint32 = 64 bytes. 
// Over K=3072 p iterations: B = 3072*64 = 192KB (doesn't fit L1 but fits L2).

// Actually the main issue may be that K=3072 columns means C row = 12KB.
// 4 C rows = 48KB which is close to L1 64KB. That fits.
// The B matrix is 3072 * 96 * 4 = ~1.1MB per scan through all rows.
// 
// Let me try a different approach: transpose the computation.
// Instead of iterating p in the inner loop (which streams through B row by row),
// accumulate the output column-tile by column-tile.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    // Tile size in output columns (in units of 32 to align with packed B)
    // 256 columns = 8 uint32_t per B row per tile
    // C tile: 4 rows * 256 cols * 4 bytes = 4KB (fits L1 easily)
    const size_t TILE_COLS = 256;
    const size_t TILE_INTS = TILE_COLS / 32;  // 8

    size_t i = 0;
    for (; i + 4 <= M; i += 4) {
        float* C_row0 = C + (i + 0) * K;
        float* C_row1 = C + (i + 1) * K;
        float* C_row2 = C + (i + 2) * K;
        float* C_row3 = C + (i + 3) * K;
        const float* A_row0 = A + (i + 0) * K;
        const float* A_row1 = A + (i + 1) * K;
        const float* A_row2 = A + (i + 2) * K;
        const float* A_row3 = A + (i + 3) * K;

        // Zero C rows
        for (size_t j = 0; j < K; ++j) {
            C_row0[j] = 0.0f;
            C_row1[j] = 0.0f;
            C_row2[j] = 0.0f;
            C_row3[j] = 0.0f;
        }

        // Tile over output columns
        for (size_t gt = 0; gt < K_ints; gt += TILE_INTS) {
            size_t g_end = gt + TILE_INTS;
            if (g_end > K_ints) g_end = K_ints;

            // For each element in the shared dimension
            for (size_t p = 0; p < K; ++p) {
                float a0 = A_row0[p];
                float a1 = A_row1[p];
                float a2 = A_row2[p];
                float a3 = A_row3[p];
                const uint32_t* B_row = B + p * K_ints;

                for (size_t g = gt; g < g_end; ++g) {
                    uint32_t packed = B_row[g];
                    float* c0 = C_row0 + g * 32;
                    float* c1 = C_row1 + g * 32;
                    float* c2 = C_row2 + g * 32;
                    float* c3 = C_row3 + g * 32;

                    for (int b = 0; b < 32; ++b) {
                        float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                        c0[b] += a0 * sign;
                        c1[b] += a1 * sign;
                        c2[b] += a2 * sign;
                        c3[b] += a3 * sign;
                    }
                }
            }
        }
    }

    for (; i < M; ++i) {
        float* C_row = C + i * K;
        const float* A_row = A + i * K;

        for (size_t j = 0; j < K; ++j) C_row[j] = 0.0f;

        for (size_t gt = 0; gt < K_ints; gt += TILE_INTS) {
            size_t g_end = gt + TILE_INTS;
            if (g_end > K_ints) g_end = K_ints;

            for (size_t p = 0; p < K; ++p) {
                float a_val = A_row[p];
                const uint32_t* B_row = B + p * K_ints;

                for (size_t g = gt; g < g_end; ++g) {
                    uint32_t packed = B_row[g];
                    float* c_out = C_row + g * 32;
                    for (int b = 0; b < 32; ++b) {
                        float sign = (packed & (1u << b)) ? 1.0f : -1.0f;
                        c_out[b] += a_val * sign;
                    }
                }
            }
        }
    }
}
