// SPDX-License-Identifier: MIT
// AArch64/ARM NEON implementations of the SIMD kernels.
#include "simd_ops.hpp"

#if defined(SIMD_COMPARE_HAVE_NEON)

#include <arm_neon.h>
#include <cassert>
#include <cstdint>

namespace simd::neon
{

void audio_gain(std::span<const float> in, float gain, std::span<float> out) noexcept
{
    assert(in.size() == out.size());
    const std::size_t n = in.size();
    const std::size_t vec_end = n & ~std::size_t{3};

    const float32x4_t vg = vdupq_n_f32(gain);
    for (std::size_t i = 0; i < vec_end; i += 4)
    {
        const float32x4_t v = vld1q_f32(in.data() + i);
        vst1q_f32(out.data() + i, vmulq_f32(v, vg));
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
        const float32x4_t va = vld1q_f32(a.data() + i);
        const float32x4_t vb = vld1q_f32(b.data() + i);
        vst1q_f32(out.data() + i, vaddq_f32(va, vb));
    }
    for (std::size_t i = vec_end; i < n; ++i)
        out[i] = a[i] + b[i];
}

void rgb_to_gray(std::span<const std::uint8_t> rgb, std::span<std::uint8_t> gray) noexcept
{
    assert(rgb.size() == gray.size() * 3);
    const std::size_t npix = gray.size();

    // NEON makes this embarrassingly simple: vld3q_u8 deinterleaves
    // 48 bytes (16 RGB pixels) into three 16-byte R/G/B vectors.
    const std::size_t block = 16;
    const std::size_t vec_end = (npix / block) * block;

    std::size_t i = 0;
    for (; i < vec_end; i += block)
    {
        const uint8x16x3_t p = vld3q_u8(rgb.data() + 3 * i);

        // Widen to 16-bit, multiply by Rec. 601 weights, sum, add rounding,
        // then narrow back to u8.
        const uint16x8_t rlo = vmovl_u8(vget_low_u8(p.val[0]));
        const uint16x8_t rhi = vmovl_u8(vget_high_u8(p.val[0]));
        const uint16x8_t glo = vmovl_u8(vget_low_u8(p.val[1]));
        const uint16x8_t ghi = vmovl_u8(vget_high_u8(p.val[1]));
        const uint16x8_t blo = vmovl_u8(vget_low_u8(p.val[2]));
        const uint16x8_t bhi = vmovl_u8(vget_high_u8(p.val[2]));

        uint16x8_t ylo = vmulq_n_u16(rlo, 77);
        ylo = vmlaq_n_u16(ylo, glo, 150);
        ylo = vmlaq_n_u16(ylo, blo, 29);
        ylo = vaddq_u16(ylo, vdupq_n_u16(128));
        ylo = vshrq_n_u16(ylo, 8);

        uint16x8_t yhi = vmulq_n_u16(rhi, 77);
        yhi = vmlaq_n_u16(yhi, ghi, 150);
        yhi = vmlaq_n_u16(yhi, bhi, 29);
        yhi = vaddq_u16(yhi, vdupq_n_u16(128));
        yhi = vshrq_n_u16(yhi, 8);

        const uint8x16_t y = vcombine_u8(vmovn_u16(ylo), vmovn_u16(yhi));
        vst1q_u8(gray.data() + i, y);
    }
    for (; i < npix; ++i)
    {
        const unsigned r = rgb[3 * i + 0];
        const unsigned g = rgb[3 * i + 1];
        const unsigned b = rgb[3 * i + 2];
        gray[i] = static_cast<std::uint8_t>((77u * r + 150u * g + 29u * b + 128u) >> 8);
    }
}

} // namespace simd::neon

#endif // SIMD_COMPARE_HAVE_NEON
