// SPDX-License-Identifier: MIT
//
// simd_ops.hpp -- Public interface for the SIMD comparison kernels.
//
// For each kernel we expose three implementations:
//   * scalar  -- a portable plain-C++ baseline (always available),
//   * sse     -- x86/x86_64 SSE2/SSSE3 intrinsics (only on x86),
//   * neon    -- AArch64/ARM NEON intrinsics (only on ARM with NEON).
//
// The `simd` namespace exposes `has_sse()` / `has_neon()` so callers can
// discover at runtime which intrinsic implementations were compiled in,
// plus a `best` alias for each kernel that picks the fastest available
// implementation for the current build target.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace simd
{

// -------- build-time capability flags -----------------------------------

constexpr bool has_sse() noexcept
{
#if defined(SIMD_COMPARE_HAVE_SSE)
    return true;
#else
    return false;
#endif
}

constexpr bool has_neon() noexcept
{
#if defined(SIMD_COMPARE_HAVE_NEON)
    return true;
#else
    return false;
#endif
}

constexpr std::string_view target_arch() noexcept
{
#if defined(__aarch64__)
    return "aarch64";
#elif defined(__arm__)
    return "arm";
#elif defined(__x86_64__)
    return "x86_64";
#elif defined(__i386__)
    return "i386";
#else
    return "unknown";
#endif
}

// -------- Kernel 1: audio gain ------------------------------------------
// out[i] = in[i] * gain   (typical audio volume / envelope stage)

namespace scalar { void audio_gain(std::span<const float> in, float gain, std::span<float> out) noexcept; }
#if defined(SIMD_COMPARE_HAVE_SSE)
namespace sse    { void audio_gain(std::span<const float> in, float gain, std::span<float> out) noexcept; }
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
namespace neon   { void audio_gain(std::span<const float> in, float gain, std::span<float> out) noexcept; }
#endif

// -------- Kernel 2: audio mix -------------------------------------------
// out[i] = a[i] + b[i]    (summing two mono audio streams)

namespace scalar { void audio_mix(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept; }
#if defined(SIMD_COMPARE_HAVE_SSE)
namespace sse    { void audio_mix(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept; }
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
namespace neon   { void audio_mix(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept; }
#endif

// -------- Kernel 3: RGB -> grayscale (Rec. 601 integer) -----------------
// Given packed 24-bit RGB pixels, produce 8-bit luminance:
//     gray = (77 * R + 150 * G + 29 * B + 128) >> 8
// Picked because it's the canonical scalar pipeline you find in image /
// video codecs and maps naturally onto 16-bit SIMD multiply-add lanes.

namespace scalar { void rgb_to_gray(std::span<const std::uint8_t> rgb, std::span<std::uint8_t> gray) noexcept; }
#if defined(SIMD_COMPARE_HAVE_SSE)
namespace sse    { void rgb_to_gray(std::span<const std::uint8_t> rgb, std::span<std::uint8_t> gray) noexcept; }
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
namespace neon   { void rgb_to_gray(std::span<const std::uint8_t> rgb, std::span<std::uint8_t> gray) noexcept; }
#endif

// -------- "best available" aliases --------------------------------------
namespace best
{
    inline void audio_gain(std::span<const float> in, float gain, std::span<float> out) noexcept
    {
#if defined(SIMD_COMPARE_HAVE_NEON)
        neon::audio_gain(in, gain, out);
#elif defined(SIMD_COMPARE_HAVE_SSE)
        sse::audio_gain(in, gain, out);
#else
        scalar::audio_gain(in, gain, out);
#endif
    }

    inline void audio_mix(std::span<const float> a, std::span<const float> b, std::span<float> out) noexcept
    {
#if defined(SIMD_COMPARE_HAVE_NEON)
        neon::audio_mix(a, b, out);
#elif defined(SIMD_COMPARE_HAVE_SSE)
        sse::audio_mix(a, b, out);
#else
        scalar::audio_mix(a, b, out);
#endif
    }

    inline void rgb_to_gray(std::span<const std::uint8_t> rgb, std::span<std::uint8_t> gray) noexcept
    {
#if defined(SIMD_COMPARE_HAVE_NEON)
        neon::rgb_to_gray(rgb, gray);
#elif defined(SIMD_COMPARE_HAVE_SSE)
        sse::rgb_to_gray(rgb, gray);
#else
        scalar::rgb_to_gray(rgb, gray);
#endif
    }

    constexpr std::string_view name() noexcept
    {
#if defined(SIMD_COMPARE_HAVE_NEON)
        return "neon";
#elif defined(SIMD_COMPARE_HAVE_SSE)
        return "sse";
#else
        return "scalar";
#endif
    }
} // namespace best

} // namespace simd
