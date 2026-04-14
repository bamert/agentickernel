#include <arm_neon.h>
#include <cstdint>
#include <cstddef>

// Calculates Matrix C = Matrix A * Matrix B (Optimized using NEON, blocking on J and P)
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    size_t K_ints = K / 32;
    
    size_t j_vector_width = 4;
    size_t num_j_blocks = (K + j_vector_width - 1) / j_vector_width;
    
    for (size_t i = 0; i < M; ++i) {         // For each row in A (M)
        
        // Loop over blocks of 4 columns (j_block_start up to K)
        for (size_t j_block_start = 0; j_block_start < K; j_block_start += j_vector_width) {
            
            // Initialize 4 accumulators for the 4 target columns
            float sums[4] = {0.0f}; 
            
            // Inner sum loop: over p (K dimension), processing 4 elements at a time
            size_t p_vector_width = 4;
            for (size_t p_start = 0; p_start < K; p_start += p_vector_width) {
                
                // 1. Load A[i][p_start:p_start+3]
                float v_a_storage[4] = {0.0f};
                // Write 4 consecutive float values to temporary storage
                vst1q_f32(v_a_storage, A + i * K + p_start);
                float v_a = vld1q_f32(v_a_storage);

                // 2. Calculate 4 sign vectors S[p_start:p_start+3, j_block_start:j_block_start+3]
                // We calculate the signs for 4 p-indices, and for 4 j-indices, all in one go.
                
                // Initialize the signs buffer pointers
                float v_s_storage[4 * 4]; // 4 target j columns * 4 p elements
                int s_idx = 0;

                for (size_t t = 0; t < j_vector_width; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Calculate signs for p = p_start to p_start + 3, targeting column j
                    for (size_t offset = 0; offset < p_vector_width; ++offset) {
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
                
                // 3. Perform the accumulation (Dot product vectorization)
                // This requires 4 separate accumulations paths, managed by the `sums` array,
                // using the loaded A vector (v_a) and the sign blocks (v_s_t).
                
                for (size_t t = 0; t < j_vector_width; ++t) {
                    size_t j = j_block_start + t;
                    
                    // Load the 4 sign signs for column j and the current p block
                    float* v_s_output = &v_s_storage[t * p_vector_width];
                    float v_s = vld1q_f32(v_s_output);
                    
                    // Product: A * Sign
                    float v_prod = vmulq_f32(v_a, v_s);
                    
                    // Accumulate: sums[t] += v_prod
                    // Use NEON for accumulation if possible, but since we access 'sums[t]' 
                    // individually, it's safer and clearer to use float accumulation here.
                    // If we were confident in L1 cache size, we could use a vector accumulator array for sums.
                    sums[t] += vgetq_lane_f32(v_prod, 0);
                }
            }
            
            // 4. Write out results for the 4 columns
            for (size_t t = 0; t < j_vector_width; ++t) {
                size_t j = j_block_start + t;
                // Check bounds *before* writing
                if (j < K) {
                    C[i * K + j] = sums[t];
                }
            }
        }
        
        // Handle cleanup for the remaining columns if K % 4 != 0
        for (size_t j = (K / 4) * 4; j < K; ++j) {
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