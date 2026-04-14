#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Vectorised over 4 rows of A (and therefore 4 rows of C) to reduce loop overhead.
// Works when M is a multiple of 4; our test harness uses M = 32.
// It processes 32 columns at once using bit extraction per column.

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K/32;
    // Process rows in groups of 4
    for(size_t i=0; i+3 < M; i+=4){
        const float* a0 = A + (i+0)*K;
        const float* a1 = A + (i+1)*K;
        const float* a2 = A + (i+2)*K;
        const float* a3 = A + (i+3)*K;
        float* c0 = C + (i+0)*K;
        float* c1 = C + (i+1)*K;
        float* c2 = C + (i+2)*K;
        float* c3 = C + (i+3)*K;
        std::memset(c0,0,sizeof(float)*K);
        std::memset(c1,0,sizeof(float)*K);
        std::memset(c2,0,sizeof(float)*K);
        std::memset(c3,0,sizeof(float)*K);

        for(size_t p=0; p<K; ++p){
            float av0 = a0[p];
            float av1 = a1[p];
            float av2 = a2[p];
            float av3 = a3[p];
            const uint32_t* packed = B + p*K_ints;
            for(size_t blk=0; blk<K_ints; ++blk){
                uint32_t bits = packed[blk];
                size_t base = blk*32;
                float* cc0 = c0 + base;
                float* cc1 = c1 + base;
                float* cc2 = c2 + base;
                float* cc3 = c3 + base;
                uint32_t cur = bits;
                for(size_t bit=0; bit<32; ++bit){
                    float sign = 1.0f - 2.0f * float(cur & 1u);
                    cc0[bit] += av0 * sign;
                    cc1[bit] += av1 * sign;
                    cc2[bit] += av2 * sign;
                    cc3[bit] += av3 * sign;
                    cur >>= 1u;
                }
            }
        }
    }
    // Handle remaining rows if any (shouldn't for 32)
    for(size_t i=M&~3; i<M; ++i){
        const float* a_row = A + i*K;
        float* rowC = C + i*K;
        std::memset(rowC,0,sizeof(float)*K);
        for(size_t p=0; p<K; ++p){
            float a_val = a_row[p];
            const uint32_t* packed = B + p*K_ints;
            for(size_t blk=0; blk<K_ints; ++blk){
                uint32_t bits = packed[blk];
                size_t base = blk*32;
                float* cc = rowC + base;
                uint32_t cur = bits;
                for(size_t bit=0; bit<32; ++bit){
                    float sign = 1.0f - 2.0f * float(cur & 1u);
                    cc[bit] += a_val * sign;
                    cur >>= 1u;
                }
            }
        }
    }
}
