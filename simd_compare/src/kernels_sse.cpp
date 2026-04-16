// SPDX-License-Identifier: MIT
// x86 SSE (SSE2 + SSSE3) implementations of the SIMD kernels.
#include "simd_ops.hpp"

#if defined(SIMD_COMPARE_HAVE_SSE)

#include <cassert>
#include <cstdint>
#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3

namespace simd::sse
{

void audio_gain(std::span<const float> in, float gain, std::span<float> out) noexcept
{
    assert(in.size() == out.size());
    const std::size_t n = in.size();
    const std::size_t vec_end = n & ~std::size_t{3};

    const __m128 vg = _mm_set1_ps(gain);
    for (std::size_t i = 0; i < vec_end; i += 4)
    {
        const __m128 v = _mm_loadu_ps(in.data() + i);
        _mm_storeu_ps(out.data() + i, _mm_mul_ps(v, vg));
    }
    for (std::size_t i = vec_end; i < n; ++i)
        out[i] = in[i] * gain;
}

void audio_mix(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept
{
    assert(a.size() == b.size() && a.size() == out.size());
    const std::size_t n = a.size();
    const std::size_t vec_end = n & ~std::size_t{3};

    for (std::size_t i = 0; i < vec_end; i += 4)
    {
        const __m128 va = _mm_loadu_ps(a.data() + i);
        const __m128 vb = _mm_loadu_ps(b.data() + i);
        _mm_storeu_ps(out.data() + i, _mm_add_ps(va, vb));
    }
    for (std::size_t i = vec_end; i < n; ++i)
        out[i] = a[i] + b[i];
}

void rgb_to_gray(std::span<const std::uint8_t> rgb, std::span<std::uint8_t> gray) noexcept
{
    assert(rgb.size() == gray.size() * 3);
    const std::size_t npix = gray.size();

    // Process 8 pixels (24 bytes in, 8 bytes out) per iteration.
    // We load 24 bytes (rounded up to a 16-byte loadu that reads past
    // the 24-byte window but only uses the first 24) and use pshufb
    // (SSSE3) to deinterleave R/G/B into three 128-bit lanes with one
    // 16-bit value per pixel.
    const std::size_t block = 8;
    const std::size_t vec_end = (npix >= 16) ? ((npix - 16) / block) * block : 0;

    //   byte layout of 24 RGB bytes:
    //     R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5 G5 B5 R6 G6 B6 R7 G7 B7
    alignas(16) static const std::int8_t shuf_r[16] = {
        0, -1, 3, -1, 6, -1, 9, -1, 12, -1, 15, -1, -1, -1, -1, -1};
    alignas(16) static const std::int8_t shuf_g[16] = {
        1, -1, 4, -1, 7, -1, 10, -1, 13, -1, -1, -1, -1, -1, -1, -1};
    alignas(16) static const std::int8_t shuf_b[16] = {
        2, -1, 5, -1, 8, -1, 11, -1, 14, -1, -1, -1, -1, -1, -1, -1};
    const __m128i sR = _mm_load_si128(reinterpret_cast<const __m128i*>(shuf_r));
    const __m128i sG = _mm_load_si128(reinterpret_cast<const __m128i*>(shuf_g));
    const __m128i sB = _mm_load_si128(reinterpret_cast<const __m128i*>(shuf_b));
    // Remaining lanes (R6, R7 / G5, G6, G7 / B5, B6, B7) live in the second
    // 16-byte chunk which is `rgb + 3*i + 8`, so we index into `hi` using
    // (original_byte_offset - 8).
    //   R6 @ byte 18 -> hi[10], R7 @ byte 21 -> hi[13]
    //   G5 @ byte 16 -> hi[ 8], G6 @ byte 19 -> hi[11], G7 @ byte 22 -> hi[14]
    //   B5 @ byte 17 -> hi[ 9], B6 @ byte 20 -> hi[12], B7 @ byte 23 -> hi[15]
    alignas(16) static const std::int8_t shuf_r2[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, -1, 13, -1};
    alignas(16) static const std::int8_t shuf_g2[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  8, -1, 11, -1, 14, -1};
    alignas(16) static const std::int8_t shuf_b2[16] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  9, -1, 12, -1, 15, -1};
    const __m128i sR2 = _mm_load_si128(reinterpret_cast<const __m128i*>(shuf_r2));
    const __m128i sG2 = _mm_load_si128(reinterpret_cast<const __m128i*>(shuf_g2));
    const __m128i sB2 = _mm_load_si128(reinterpret_cast<const __m128i*>(shuf_b2));

    const __m128i wR    = _mm_set1_epi16(77);
    const __m128i wG    = _mm_set1_epi16(150);
    const __m128i wB    = _mm_set1_epi16(29);
    const __m128i bias  = _mm_set1_epi16(128);

    std::size_t i = 0;
    for (; i < vec_end; i += block)
    {
        const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb.data() + 3 * i));
        const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb.data() + 3 * i + 8));

        // R/G/B as 8 lanes of 16-bit unsigned values.
        const __m128i R = _mm_or_si128(_mm_shuffle_epi8(lo, sR), _mm_shuffle_epi8(hi, sR2));
        const __m128i G = _mm_or_si128(_mm_shuffle_epi8(lo, sG), _mm_shuffle_epi8(hi, sG2));
        const __m128i B = _mm_or_si128(_mm_shuffle_epi8(lo, sB), _mm_shuffle_epi8(hi, sB2));

        __m128i y = _mm_add_epi16(_mm_mullo_epi16(R, wR),
                    _mm_add_epi16(_mm_mullo_epi16(G, wG),
                                  _mm_mullo_epi16(B, wB)));
        y = _mm_add_epi16(y, bias);
        y = _mm_srli_epi16(y, 8);
        // Pack 8 x u16 -> 8 x u8 (saturating to unsigned); values are always <=255.
        const __m128i packed = _mm_packus_epi16(y, _mm_setzero_si128());
        _mm_storel_epi64(reinterpret_cast<__m128i*>(gray.data() + i), packed);
    }
    // Tail: scalar fallback.
    for (; i < npix; ++i)
    {
        const unsigned r = rgb[3 * i + 0];
        const unsigned g = rgb[3 * i + 1];
        const unsigned b = rgb[3 * i + 2];
        gray[i] = static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
    }
}

} // namespace simd::sse

#endif // SIMD_COMPARE_HAVE_SSE
