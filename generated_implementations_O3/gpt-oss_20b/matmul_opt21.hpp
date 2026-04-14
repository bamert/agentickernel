#pragma once
#include <cstdint>
#include <cstddef>
#include <array>

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K){
    const size_t K_ints = K / 32;

    // Build lookup table once per call using a constexpr lambda.
    const std::array<float, 256 * 8> sign_table = [](){
        std::array<float, 256 * 8> t{};
        for(int v=0; v<256; ++v){
            for(int k=0; k<8; ++k){
                t[v*8 + k] = ((v >> k) & 1u) ? 1.0f : -1.0f;
            }
        }
        return t;
    }();

    for(size_t i=0; i<M; ++i){
        const float* a_row = A + i * K;
        float* rowC = C + i * K;
        std::memset(rowC, 0, sizeof(float) * K);

        for(size_t p=0; p<K; ++p){
            float a_val = a_row[p];
            const uint32_t* packed_row = B + p * K_ints;
            for(size_t blk=0; blk<K_ints; ++blk){
                uint32_t bits = packed_row[blk];
                float* c_ptr = rowC + blk * 32;
                uint8_t sub0 = bits & 0xFFu;
                uint8_t sub1 = (bits >> 8) & 0xFFu;
                uint8_t sub2 = (bits >> 16) & 0xFFu;
                uint8_t sub3 = (bits >> 24) & 0xFFu;
                const float* s0 = sign_table.data() + sub0 * 8;
                const float* s1 = sign_table.data() + sub1 * 8;
                const float* s2 = sign_table.data() + sub2 * 8;
                const float* s3 = sign_table.data() + sub3 * 8;
                // unrolled updates
                c_ptr[0]  += a_val * s0[0];  c_ptr[1]  += a_val * s0[1];  c_ptr[2]  += a_val * s0[2];  c_ptr[3]  += a_val * s0[3];
                c_ptr[4]  += a_val * s0[4];  c_ptr[5]  += a_val * s0[5];  c_ptr[6]  += a_val * s0[6];  c_ptr[7]  += a_val * s0[7];
                c_ptr[8]  += a_val * s1[0];  c_ptr[9]  += a_val * s1[1];  c_ptr[10] += a_val * s1[2];  c_ptr[11] += a_val * s1[3];
                c_ptr[12] += a_val * s1[4];  c_ptr[13] += a_val * s1[5];  c_ptr[14] += a_val * s1[6];  c_ptr[15] += a_val * s1[7];
                c_ptr[16] += a_val * s2[0];  c_ptr[17] += a_val * s2[1];  c_ptr[18] += a_val * s2[2];  c_ptr[19] += a_val * s2[3];
                c_ptr[20] += a_val * s2[4];  c_ptr[21] += a_val * s2[5];  c_ptr[22] += a_val * s2[6];  c_ptr[23] += a_val * s2[7];
                c_ptr[24] += a_val * s3[0];  c_ptr[25] += a_val * s3[1];  c_ptr[26] += a_val * s3[2];  c_ptr[27] += a_val * s3[3];
                c_ptr[28] += a_val * s3[4];  c_ptr[29] += a_val * s3[5];  c_ptr[30] += a_val * s3[6];  c_ptr[31] += a_val * s3[7];
            }
        }
    }
}
