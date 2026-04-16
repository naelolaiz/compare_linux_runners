// SPDX-License-Identifier: MIT
// Portable scalar reference implementations of the SIMD kernels.
#include "simd_ops.hpp"

#include <cassert>
#include <cstdint>

namespace simd::scalar
{

void audio_gain(std::span<const float> in, float gain, std::span<float> out) noexcept
{
    assert(in.size() == out.size());
    for (std::size_t i = 0; i < in.size(); ++i)
        out[i] = in[i] * gain;
}

void audio_mix(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept
{
    assert(a.size() == b.size() && a.size() == out.size());
    for (std::size_t i = 0; i < a.size(); ++i)
        out[i] = a[i] + b[i];
}

void rgb_to_gray(std::span<const std::uint8_t> rgb, std::span<std::uint8_t> gray) noexcept
{
    assert(rgb.size() == gray.size() * 3);
    for (std::size_t i = 0; i < gray.size(); ++i)
    {
        const unsigned r = rgb[3 * i + 0];
        const unsigned g = rgb[3 * i + 1];
        const unsigned b = rgb[3 * i + 2];
        // Rec. 601 luma with 8.8 fixed-point weights (77, 150, 29).
        const unsigned y = (77u * r + 150u * g + 29u * b + 128u) >> 8;
        gray[i] = static_cast<std::uint8_t>(y);
    }
}

} // namespace simd::scalar
