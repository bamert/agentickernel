#pragma once

using uint32_t = unsigned int;
using size_t   = unsigned long;

void matmul(const float* A, const uint32_t* B, float* C, size_t M, size_t K)
{
    const size_t K_ints = K / 32;          // 32‑bit words per row of B

    /* Sign lookup table: 256 × 8 floats (+1.0f or –1.0f). */
    alignas(16) float sign_tbl[256 * 8];
    for (int bv = 0; bv < 256; ++bv)
        for (int bit = 0; bit < 8; ++bit)
            sign_tbl[(bv << 3) | bit] = (bv & (1 << bit)) ? 1.0f : -1.0f;

    for (size_t i = 0; i < M; ++i)
    {
        const float* Ai = A + i * K;   // row i of A
        float*       Ci = C + i * K;   // row i of C

        /* Zero the output row. */
        for (size_t j = 0; j < K; ++j)
            Ci[j] = 0.0f;

        for (size_t p = 0; p < K; ++p)
        {
            const float a_val = Ai[p];
            const uint32_t* B_row = B + p * K_ints;

            for (size_t w = 0; w < K_ints; ++w)
            {
                const uint32_t word  = B_row[w];
                const size_t   base  = w * 32;

                /* Process four 8‑bit blocks of the 32‑bit word. */
                for (size_t byte_off = 0; byte_off < 4; ++byte_off)
                {
                    uint8_t byte_val = static_cast<uint8_t>(word >> (byte_off * 8));
                    const float* signs = &sign_tbl[(byte_val << 3)];

                    /* Load two groups of four C elements. */
                    float32x4_t c0 = vld1q_f32(&Ci[base + byte_off * 8 + 0]);
                    float32x4_t c1 = vld1q_f32(&Ci[base + byte_off * 8 + 4]);

                    /* Load corresponding sign vectors. */
                    float32x4_t s0 = vld1q_f32(signs + 0);
                    float32x4_t s1 = vld1q_f32(signs + 4);

                    /* Broadcast a_val. */
                    float32x4_t a_vec = vdupq_n_f32(a_val);

                    /* Accumulate. */
                    c0 = vaddq_f32(c0, vmulq_f32(a_vec, s0));
                    c1 = vaddq_f32(c1, vmulq_f32(a_vec, s1));

                    /* Store results back. */
                    vst1q_f32(&Ci[base + byte_off * 8 + 0], c0);
                    vst1q_f32(&Ci[base + byte_off * 8 + 4], c1);
                }
            }
        }
    }
}
