#pragma once

typedef unsigned int uint32_t;
typedef decltype(sizeof(0)) size_t;

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K) {
    const size_t K_ints = K >> 5;
    for (size_t i = 0; i < M; ++i) {
        const float* arow = A + i * K;
        float* crow = C + i * K;
        for (size_t jb = 0; jb < K; jb += 32) {
            float s0=0.0f,s1=0.0f,s2=0.0f,s3=0.0f,s4=0.0f,s5=0.0f,s6=0.0f,s7=0.0f;
            float s8=0.0f,s9=0.0f,s10=0.0f,s11=0.0f,s12=0.0f,s13=0.0f,s14=0.0f,s15=0.0f;
            float s16=0.0f,s17=0.0f,s18=0.0f,s19=0.0f,s20=0.0f,s21=0.0f,s22=0.0f,s23=0.0f;
            float s24=0.0f,s25=0.0f,s26=0.0f,s27=0.0f,s28=0.0f,s29=0.0f,s30=0.0f,s31=0.0f;
            const size_t widx = jb >> 5;
            for (size_t p = 0; p < K; ++p) {
                const float a = arow[p];
                const float na = -a;
                const uint32_t bits = B[p * K_ints + widx];
                s0  += (bits & (1u << 0))  ? a : na;
                s1  += (bits & (1u << 1))  ? a : na;
                s2  += (bits & (1u << 2))  ? a : na;
                s3  += (bits & (1u << 3))  ? a : na;
                s4  += (bits & (1u << 4))  ? a : na;
                s5  += (bits & (1u << 5))  ? a : na;
                s6  += (bits & (1u << 6))  ? a : na;
                s7  += (bits & (1u << 7))  ? a : na;
                s8  += (bits & (1u << 8))  ? a : na;
                s9  += (bits & (1u << 9))  ? a : na;
                s10 += (bits & (1u << 10)) ? a : na;
                s11 += (bits & (1u << 11)) ? a : na;
                s12 += (bits & (1u << 12)) ? a : na;
                s13 += (bits & (1u << 13)) ? a : na;
                s14 += (bits & (1u << 14)) ? a : na;
                s15 += (bits & (1u << 15)) ? a : na;
                s16 += (bits & (1u << 16)) ? a : na;
                s17 += (bits & (1u << 17)) ? a : na;
                s18 += (bits & (1u << 18)) ? a : na;
                s19 += (bits & (1u << 19)) ? a : na;
                s20 += (bits & (1u << 20)) ? a : na;
                s21 += (bits & (1u << 21)) ? a : na;
                s22 += (bits & (1u << 22)) ? a : na;
                s23 += (bits & (1u << 23)) ? a : na;
                s24 += (bits & (1u << 24)) ? a : na;
                s25 += (bits & (1u << 25)) ? a : na;
                s26 += (bits & (1u << 26)) ? a : na;
                s27 += (bits & (1u << 27)) ? a : na;
                s28 += (bits & (1u << 28)) ? a : na;
                s29 += (bits & (1u << 29)) ? a : na;
                s30 += (bits & (1u << 30)) ? a : na;
                s31 += (bits & (1u << 31)) ? a : na;
            }
            crow[jb + 0] = s0; crow[jb + 1] = s1; crow[jb + 2] = s2; crow[jb + 3] = s3;
            crow[jb + 4] = s4; crow[jb + 5] = s5; crow[jb + 6] = s6; crow[jb + 7] = s7;
            crow[jb + 8] = s8; crow[jb + 9] = s9; crow[jb + 10] = s10; crow[jb + 11] = s11;
            crow[jb + 12] = s12; crow[jb + 13] = s13; crow[jb + 14] = s14; crow[jb + 15] = s15;
            crow[jb + 16] = s16; crow[jb + 17] = s17; crow[jb + 18] = s18; crow[jb + 19] = s19;
            crow[jb + 20] = s20; crow[jb + 21] = s21; crow[jb + 22] = s22; crow[jb + 23] = s23;
            crow[jb + 24] = s24; crow[jb + 25] = s25; crow[jb + 26] = s26; crow[jb + 27] = s27;
            crow[jb + 28] = s28; crow[jb + 29] = s29; crow[jb + 30] = s30; crow[jb + 31] = s31;
        }
    }
}
