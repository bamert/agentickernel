#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;

    for (size_t i = 0; i < M; ++i) {         // For each row in A
        for (size_t j = 0; j < K; ++j) {     // For each column in C
            // Initialize Neon accumulator for the dot product sum
            float sum_arr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float* v_sum = &sum_arr[0]; 

            // Iterate over p using NEON intrinsics (processing 4 elements of p at a time)
            for (size_t p_start = 0; p_start < K; p_start += 4) {
                // Vector to hold the 4 values of A[i][p_start:p_end-1]
                float v_a = vld1q_f32(A + i * K + p_start);
                
                // Vector to hold the 4 signs S[p_start:p_end-1, j]
                float v_s_arr[4];
                
                // Calculate signs for 4 consecutive p values (p_start to p_start+3)
                for(size_t offset = 0; offset < 4; ++offset) {
                    size_t p = p_start + offset;
                    
                    // Calculate B[p][j]: B row p, word j_word, bit j_bit
                    size_t j_word = j / 32;
                    size_t j_bit = j % 32;
                    
                    // We need B[rank p] up to B[rank p+3]. 
                    // The correct access for the p-th row is: B[p * K_ints + j_word]

                    uint32_t packed = B[p * K_ints + j_word];
                    uint32_t bit = (packed >> j_bit) & 1;
                    
                    float sign = bit ? 1.0f : -1.0f;
                    v_s_arr[offset] = sign;
                }

                v_s = vld1q_f32(v_s_arr);
                
                // Calculate the product vector: A * Sign
                float v_prod = vmulq_f32(v_a, v_s);
                
                // Accumulate the product into the sum vector
                v_sum = vaddq_f32(v_sum, v_prod);
            }
            
            // Sum the four lanes of the resulting vector sum
            float total_sum = 0.0f;
            float temp_sum = vgetq_lane_f32(v_sum, 0);
            temp_sum += vgetq_lane_f32(v_sum, 1);
            temp_sum += vgetq_lane_f32(v_sum, 2);
            temp_sum += vgetq_lane_f32(v_sum, 3);
            
            C[i * K + j] = total_sum + temp_sum;
        }
    }
}