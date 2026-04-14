#pragma once

using uint32_t = unsigned int;
using size_t   = unsigned long;

/*
 * Matrix multiplication – scalar implementation.
 * This version builds a sign lookup table on the stack and
 * unrolls the inner 32‑bit word loop for zero branch overhead.
 */
void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K)
{
    const size_t K_ints = K / 32;

    /* Sign lookup table: 256×8 floats (+1.0f or –1.0f). */
    float sign_tbl[256*8];
    for(int bv=0; bv<256; ++bv)
        for(int bit=0; bit<8; ++bit)
            sign_tbl[(bv<<3)|bit] = (bv & (1<<bit)) ? 1.0f : -1.0f;

    for(size_t i=0; i<M; ++i){
        const float* Ai = A + i*K;
        float* Ci = C + i*K;
        for(size_t j=0; j<K; ++j) Ci[j]=0.0f;

        for(size_t p=0; p<K; ++p){
            float a_val=Ai[p];
            const uint32_t* B_row = B + p*K_ints;
            for(size_t w=0; w<K_ints; ++w){
                uint32_t word=B_row[w];
                size_t base=w*32;
                // unrolled 4 blocks of 8 bits
                {
                    uint8_t b0 = (uint8_t)(word>>0);
                    const float* sp=&sign_tbl[b0<<3];
                    Ci[base+0+=a_val*sp[0];
                }
            }
        }
    }
}
