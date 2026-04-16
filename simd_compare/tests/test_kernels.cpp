// SPDX-License-Identifier: MIT
//
// Correctness tests: every SIMD implementation that is compiled in must
// produce the exact same output as the scalar reference over randomized
// inputs, including sizes that are not a multiple of the SIMD width so
// that tail-handling is exercised.
#include "simd_ops.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace
{

int g_failures = 0;

void expect(bool cond, const char* what)
{
    if (!cond)
    {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

bool floats_equal(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        // Both are produced by the exact same arithmetic (single multiply
        // or add on the same inputs), so results must be bit-identical.
        if (a[i] != b[i]) return false;
    }
    return true;
}

bool bytes_equal(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b)
{
    return a == b;
}

} // namespace

int main()
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_f(-10.0f, 10.0f);
    std::uniform_int_distribution<int>    dist_u(0, 255);

    // Pick sizes that are NOT multiples of 4/8/16 to exercise the tails.
    const std::vector<std::size_t> audio_sizes = {0, 1, 3, 4, 7, 15, 16, 17, 63, 1000, 4099};
    const std::vector<std::size_t> pixel_sizes = {0, 1, 7, 8, 15, 16, 17, 31, 33, 257, 1000, 4099};

    for (std::size_t n : audio_sizes)
    {
        std::vector<float> a(n), b(n);
        for (auto& x : a) x = dist_f(rng);
        for (auto& x : b) x = dist_f(rng);

        // audio_gain
        std::vector<float> ref(n), got(n);
        const float gain = 0.375f;
        simd::scalar::audio_gain(a, gain, ref);
#if defined(SIMD_COMPARE_HAVE_SSE)
        simd::sse::audio_gain(a, gain, got);
        expect(floats_equal(ref, got), "sse::audio_gain matches scalar");
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
        simd::neon::audio_gain(a, gain, got);
        expect(floats_equal(ref, got), "neon::audio_gain matches scalar");
#endif
        // audio_mix
        simd::scalar::audio_mix(a, b, ref);
#if defined(SIMD_COMPARE_HAVE_SSE)
        simd::sse::audio_mix(a, b, got);
        expect(floats_equal(ref, got), "sse::audio_mix matches scalar");
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
        simd::neon::audio_mix(a, b, got);
        expect(floats_equal(ref, got), "neon::audio_mix matches scalar");
#endif
    }

    for (std::size_t npix : pixel_sizes)
    {
        std::vector<std::uint8_t> rgb(npix * 3);
        for (auto& x : rgb) x = static_cast<std::uint8_t>(dist_u(rng));
        std::vector<std::uint8_t> ref(npix), got(npix);
        simd::scalar::rgb_to_gray(rgb, ref);
#if defined(SIMD_COMPARE_HAVE_SSE)
        simd::sse::rgb_to_gray(rgb, got);
        expect(bytes_equal(ref, got), "sse::rgb_to_gray matches scalar");
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
        simd::neon::rgb_to_gray(rgb, got);
        expect(bytes_equal(ref, got), "neon::rgb_to_gray matches scalar");
#endif
    }

    if (g_failures == 0)
    {
        std::printf("All SIMD kernel tests passed (impl: %s).\n",
                    std::string(simd::best::name()).c_str());
        return 0;
    }
    std::printf("%d test failure(s).\n", g_failures);
    return 1;
}
