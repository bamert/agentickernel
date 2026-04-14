#pragma once
#include <cstdint>
#include <cstddef>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K/32;
    // Precomputed sign table as a constexpr local array
    constexpr auto create_table = [](){
        const float table[256][8] = []{
            float t[256][8];
            for(int v=0; v<256; ++v){
                for(int k=0; k<8; ++k){
                    t[v][k] = ((v >> k) & 1U) ? 1.0f : -1.0f;
                }
            }
            return t;
        }();
        return table;
    }();

    for(size_t i=0; i<M; ++i){
        const float* a_row = A + i*K;
        float* rowC = C + i*K;
        for(size_t j=0; j<K; ++j) rowC[j] = 0.0f;

        for(size_t p=0; p<K; ++p){
            float a_val = a_row[p];
            const uint32_t* packed_row = B + p*K_ints;
            for(size_t blk=0; blk<K_ints; ++blk){
                uint32_t bits = packed_row[blk];
                float* c_ptr = rowC + blk*32;
                for(size_t byte=0; byte<4; ++byte){
                    uint8_t sub = (bits >> (byte*8)) & 0xFFu;
                    const float* signs = create_table[sub];
                    for(size_t k=0; k<8; ++k){
                        c_ptr[byte*8 + k] += a_val * signs[k];
                    }
                }
            }
        }
    }
}
