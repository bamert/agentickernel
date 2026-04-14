#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K/32;
    for(size_t i=0;i<M;++i){
        const float* a_row = A + i*K;
        float* rowC = C + i*K;
        std::memset(rowC,0,sizeof(float)*K);
        for(size_t p=0;p<K;++p){
            float a_val = a_row[p];
            const uint32_t* packed_row = B + p*K_ints;
            for(size_t blk=0; blk<K_ints; ++blk){
                uint32_t bits = packed_row[blk];
                float* c_ptr = rowC + blk*32;
                for(size_t bit=0; bit<32; ++bit){
                    if(bits & 1u){
                        c_ptr[bit] += a_val;
                    } else {
                        c_ptr[bit] -= a_val;
                    }
                    bits >>= 1u;
                }
            }
        }
    }
}
