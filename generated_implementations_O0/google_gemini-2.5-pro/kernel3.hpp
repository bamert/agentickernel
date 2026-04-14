#pragma once
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B
// A: Float matrix (M rows, K cols)
// B: Packed binary matrix (K rows, K cols). 1 bit = +1.0f, 0 bit = -1.0f.
// C: Output float matrix (M rows, K cols)
// Constraint: K is guaranteed to be a multiple of 32.
// ---------------------------------------------------------
// Optimization Strategy:
// 1. Tiling/Chunking: Retains the successful structure from kernel2, computing
//    32 elements of C simultaneously to align with B's packing and reuse A's values.
// 2. Branch Elimination: The key improvement over kernel2. The data-dependent
//    `if/else` statement inside the innermost loop is a major source of branch
//    mispredictions. It is replaced with a branchless arithmetic computation.
//    - The bit (0 or 1) is extracted: `(packed_b >> bit) & 1`.
//    - It is arithmetically converted to a sign: `(float)(bit * 2 - 1)` turns
//      1 into 1.0f and 0 into -1.0f.
//    - This removes conditional logic, preventing pipeline stalls and leading to
//      a much faster and more predictable inner loop.
// 3. Register Accumulation: Continues to use a local `sums[32]` array to
//    reduce memory traffic to the final C matrix.
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {
        const float* A_row = A + i * K;
        float* C_row = C + i * K;

        // Iterate through columns of B/C in chunks of 32
        for (size_t j_chunk = 0; j_chunk < K_ints; ++j_chunk) {
            float sums[32] = {0.0f};

            // This is the dot product dimension
            for (size_t p = 0; p < K; ++p) {
                const float a_val = A_row[p];
                const uint32_t packed_b = B[p * K_ints + j_chunk];
                
                // Update 32 accumulators at once using the bits from packed_b
                // This loop is now branchless.
                for (int bit = 0; bit < 32; ++bit) {
                    // Arithmetically determine the sign (+1.0f or -1.0f)
                    int bit_val = (packed_b >> bit) & 1;
                    float sign = (float)(bit_val * 2 - 1);
                    sums[bit] += a_val * sign;
                }
            }

            // Write the final 32 sums to the C matrix
            for (int k = 0; k < 32; ++k) {
                C_row[j_chunk * 32 + k] = sums[k];
            }
        }
    }
}
