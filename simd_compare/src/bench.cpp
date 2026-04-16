// SPDX-License-Identifier: MIT
//
// Micro-benchmark that runs every available implementation (scalar,
// plus SSE or NEON depending on the build target) over the three
// kernels and prints throughput in GB/s processed.
#include "simd_ops.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using clock_type = std::chrono::steady_clock;

template <class Fn>
double time_seconds(std::size_t iterations, Fn&& fn)
{
    const auto t0 = clock_type::now();
    for (std::size_t i = 0; i < iterations; ++i)
        fn();
    const auto t1 = clock_type::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void print_row(std::string_view kernel, std::string_view impl,
               double seconds, std::size_t iters, std::size_t bytes_per_iter)
{
    const double total_bytes = static_cast<double>(iters) * static_cast<double>(bytes_per_iter);
    const double gibps = (total_bytes / seconds) / (1024.0 * 1024.0 * 1024.0);
    std::printf("  %-14s %-8s %10.3f ms    %8.2f GiB/s\n",
                std::string(kernel).c_str(), std::string(impl).c_str(),
                seconds * 1000.0, gibps);
}

} // namespace

int main()
{
    std::printf("SIMD kernel benchmark\n");
    std::printf("  target arch : %s\n", std::string(simd::target_arch()).c_str());
    std::printf("  SSE build   : %s\n", simd::has_sse() ? "yes" : "no");
    std::printf("  NEON build  : %s\n", simd::has_neon() ? "yes" : "no");
    std::printf("  best impl   : %s\n\n", std::string(simd::best::name()).c_str());

    constexpr std::size_t kAudioSamples = 1u << 20;  // 1 Mi floats  = 4 MiB
    constexpr std::size_t kAudioIters   = 200;
    constexpr std::size_t kPixels       = 1u << 20;  // 1 Mi pixels = 3 MiB RGB
    constexpr std::size_t kPixelIters   = 200;

    std::vector<float> in_a(kAudioSamples);
    std::vector<float> in_b(kAudioSamples);
    std::vector<float> out_f(kAudioSamples);
    std::vector<std::uint8_t> rgb(kPixels * 3);
    std::vector<std::uint8_t> gray(kPixels);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);
    for (auto& x : in_a) x = dist_f(rng);
    for (auto& x : in_b) x = dist_f(rng);
    std::uniform_int_distribution<int> dist_u(0, 255);
    for (auto& x : rgb)  x = static_cast<std::uint8_t>(dist_u(rng));

    std::printf("  %-14s %-8s %13s    %13s\n", "kernel", "impl", "time", "throughput");
    std::printf("  -----------------------------------------------------------\n");

    // audio_gain
    {
        const std::size_t bytes = kAudioSamples * sizeof(float) * 2; // read+write
        const double s = time_seconds(kAudioIters, [&]{ simd::scalar::audio_gain(in_a, 0.75f, out_f); });
        print_row("audio_gain", "scalar", s, kAudioIters, bytes);
#if defined(SIMD_COMPARE_HAVE_SSE)
        const double ss = time_seconds(kAudioIters, [&]{ simd::sse::audio_gain(in_a, 0.75f, out_f); });
        print_row("audio_gain", "sse",    ss, kAudioIters, bytes);
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
        const double sn = time_seconds(kAudioIters, [&]{ simd::neon::audio_gain(in_a, 0.75f, out_f); });
        print_row("audio_gain", "neon",   sn, kAudioIters, bytes);
#endif
    }
    // audio_mix
    {
        const std::size_t bytes = kAudioSamples * sizeof(float) * 3; // two in + one out
        const double s = time_seconds(kAudioIters, [&]{ simd::scalar::audio_mix(in_a, in_b, out_f); });
        print_row("audio_mix",  "scalar", s, kAudioIters, bytes);
#if defined(SIMD_COMPARE_HAVE_SSE)
        const double ss = time_seconds(kAudioIters, [&]{ simd::sse::audio_mix(in_a, in_b, out_f); });
        print_row("audio_mix",  "sse",    ss, kAudioIters, bytes);
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
        const double sn = time_seconds(kAudioIters, [&]{ simd::neon::audio_mix(in_a, in_b, out_f); });
        print_row("audio_mix",  "neon",   sn, kAudioIters, bytes);
#endif
    }
    // rgb_to_gray
    {
        const std::size_t bytes = kPixels * 3 + kPixels;
        const double s = time_seconds(kPixelIters, [&]{ simd::scalar::rgb_to_gray(rgb, gray); });
        print_row("rgb_to_gray","scalar", s, kPixelIters, bytes);
#if defined(SIMD_COMPARE_HAVE_SSE)
        const double ss = time_seconds(kPixelIters, [&]{ simd::sse::rgb_to_gray(rgb, gray); });
        print_row("rgb_to_gray","sse",    ss, kPixelIters, bytes);
#endif
#if defined(SIMD_COMPARE_HAVE_NEON)
        const double sn = time_seconds(kPixelIters, [&]{ simd::neon::rgb_to_gray(rgb, gray); });
        print_row("rgb_to_gray","neon",   sn, kPixelIters, bytes);
#endif
    }
    // Consume results so the optimizer can't drop them.
    volatile float sink_f = out_f.front() + out_f.back();
    volatile std::uint8_t sink_u = gray.front() ^ gray.back();
    (void)sink_f;
    (void)sink_u;
    return 0;
}
