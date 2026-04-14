#pragma once
#include <cstdint>
#include <cstddef>
#include <array>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K/32;
    // Precompute a lookup table mapping a byte to 8 sign values.
    constexpr std::array<float, 256*8> sign_table = []{
        std::array<float, 256*8> t{};
        for(int v=0; v<256; ++v){
            for(int k=0; k<8; ++k){
                t[v*8 + k] = ((v >> k) & 1u) ? 1.0f : -1.0f;
            }
        }
        return t;
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
                    const float* signs = &sign_table[sub*8];
                    for(size_t k=0; k<8; ++k){
                        c_ptr[byte*8 + k] += a_val * signs[k];
                    }
                }
            }
        }
    }
}
