#include <arm_neon.h>
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized using NEON for the summation loop)
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        for (size_t j = 0; j < K; ++j) {     // For each column in C (K)
            
            // Initialize Neon accumulator for the dot product sum
            float sum_array[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float* v_sum = &sum_array[0]; 

            // Inner loop: Summation over p (K dimension)
            // Process p in blocks of 4 (vector size)
            for (size_t p_start = 0; p_start < K; p_start += 4) {
                
                // 1. Load A[i][p_start:p_start+3]
                float v_a_storage[4] = {0.0f};
                vst1q_f32(v_a_storage, A + i * K + p_start);
                float v_a = vld1q_f32(v_a_storage);

                // 2. Calculate signs for 4 consecutive p values (p_start to p_start+3)
                float v_s_storage[4];
                
                // Calculate signs for 4 p values: S[p_start], S[p_start+1], S[p_start+2], S[p_start+3]
                for(size_t offset = 0; offset < 4; ++offset) {
                    size_t p = p_start + offset;
                    
                    // Calculate B[p][j]: B row p, word j_word, bit j_bit
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    // Access B: B is row-major. Row p. Word j_word.
                    // Check bounds before calculating B access
                    if (p * K_ints + j_word >= (K/32) * K_ints) {
                         // Should not happen given K >= 1, but good practice
                         v_s_storage[offset] = 0.0f; // Default sign
                         continue;
                    }
                    
                    uint32_t packed = B[p * K_ints + j_word];
                    uint32_t bit = (packed >> j_bit) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    v_s_storage[offset] = sign;
                }
                
                float* v_s_output = v_s_storage;
                float v_s = vld1q_f32(v_s_output);
                
                // 3. Calculate the product vector: A * Sign
                float v_prod = vmulq_f32(v_a, v_s);
                
                // 4. Accumulate the product into the sum vector
                v_sum = vaddq_f32(v_sum, v_prod);
            }
            
            // 5. Sum the four lanes of the resulting vector sum
            float total_sum = 0.0f;
            total_sum += vgetq_lane_f32(v_sum, 0);
            total_sum += vgetq_lane_f32(v_sum, 1);
            total_sum += vgetq_lane_f32(v_sum, 2);
            total_sum += vgetq_lane_f32(v_sum, 3);
            
            C[i * K + j] = total_sum;
        }
    }
}