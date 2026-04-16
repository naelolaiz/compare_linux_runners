# compare_linux_runners

Projects that exercise and compare the different Linux runners available
on GitHub Actions (x86_64 vs arm64, SSE vs NEON, …).

## Projects

- [`simd_compare/`](simd_compare/README.md) — modern C++23 SIMD kernels
  (audio gain, audio mix, RGB→gray) implemented with a scalar baseline
  plus **SSE** (x86_64) and **NEON** (aarch64) intrinsics. Includes a
  CMake build, correctness tests, a throughput micro-benchmark and a
  GitHub Actions workflow that runs everything on both `ubuntu-latest`
  and `ubuntu-24.04-arm` runners.