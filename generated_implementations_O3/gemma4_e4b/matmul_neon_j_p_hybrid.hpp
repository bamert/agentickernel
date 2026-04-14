#include <arm_neon.h>
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized using NEON, blocking on J and P)
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    const size_t J_BLOCK_SIZE = 4;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Loop over blocks of 4 columns (j_block_start up to K)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += J_BLOCK_SIZE) {
            
            // Accumulators for the 4 target columns (j_block_start to j_block_start + 3)
            float sums[J_BLOCK_SIZE] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension), processing 4 elements at a time (P_VECTOR_SIZE)
            const size_t P_VECTOR_SIZE = 4;
            for (size_t p_start = 0; p_start < K; p_start += P_VECTOR_SIZE) {
                
                // 1. Load A[i][p_start:p_start+3]
                float v_a_storage[4] = {0.0f};
                // We must handle bounds check implicitly by only writing/loading available elements.
                size_t actual_p_vec_size = (p_start + P_VECTOR_SIZE > K) ? (K - p_start) : P_VECTOR_SIZE;
                
                for (size_t offset = 0; offset < actual_p_vec_size; ++offset) {
                     v_a_storage[offset] = A[i * K + p_start + offset];
                }
                float v_a = vld1q_f32(v_a_storage);

                // 2. Calculate 4 sign vectors S[p_start:p_start+3, j_block_start:j_block_start+3]
                float v_s_storage[4 * 4]; // 4 target j columns * 4 p elements
                int s_idx = 0;

                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Calculate signs for p = p_start to p_start + actual_p_vec_size - 1, targeting column j
                    for (size_t offset = 0; offset < actual_p_vec_size; ++offset) {
                        size_t p = p_start + offset;
                        
                        // Calculate B[p][j]: B row p, word j_word, bit j_bit
                        size_t j_word = j / 32;
                        size_t j_bit = j % 32;
                        
                        // Access B: Row p. Word j_word.
                        uint32_t packed = B[p * K_ints + j_word];
                        uint32_t bit = (packed >> j_bit) & 1;
                        float sign = bit ? 1.0f : -1.0f;
                        
                        v_s_storage[s_idx++] = sign;
                    }
                }
                
                // 3. Accumulation
                for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Load the 4 sign signs for column j and the current p block
                    float* v_s_output = &v_s_storage[t * 4];
                    float v_s = vld1q_f32(v_s_output);
                    
                    // Product: A * Sign
                    float v_prod = vmulq_f32(v_a, v_s);
                    
                    // Accumulate dot product: sums[t] += sum(v_prod)
                    // Since NEON operates on 4 elements, we use the first lane's value for accumulation,
                    // which works if we treat the total sum as a sequence of 4 independent float accumulations.
                    sums[t] += vgetq_lane_f32(v_prod, 0);
                }
            }
            
            // 4. Write out results for the 4 columns
            for (size_t t = 0; t < J_BLOCK_SIZE; ++t) {
                size_t j = j_block_start + t;
                if (j < K) {
                    C[i * K + j] = sums[t];
                }
            }
        }
        
        // Fallback: Handle the remaining columns if K % 4 != 0
        // This part remains the sequential (non-vectorized) version for correctness at the edge.
        size_t start_j = (K / J_BLOCK_SIZE) * J_BLOCK_SIZE;
        for (size_t j = start_j; j < K; ++j) {
            float sum = 0.0f;
            for (size_t p = 0; p < K; ++p) {
                size_t j_word = j / 32;
                size_t j_bit = j % 32;
                
                uint32_t packed = B[p * K_ints + j_word];
                uint32_t bit = (packed >> j_bit) & 1;
                
                float sign = bit ? 1.0f : -1.0f;
                sum += A[i * K + p] * sign;
            }
            C[i * K + j] = sum;
        }
    }
}