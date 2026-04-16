# simd_compare — SSE vs NEON kernel comparison

A small, modern-C++ (C++23, CMake ≥ 3.20) project that implements three
audio/video-style kernels in three flavours each — portable scalar,
x86 **SSE** intrinsics, and ARM **NEON** intrinsics — and runs a
correctness test plus a throughput micro-benchmark on both x86_64 and
arm64 Linux GitHub-hosted runners.

## Kernels

| Kernel        | Signature                                                                 | Use case                |
| ------------- | ------------------------------------------------------------------------- | ----------------------- |
| `audio_gain`  | `out[i] = in[i] * gain`                                                   | audio volume / envelope |
| `audio_mix`   | `out[i] = a[i] + b[i]`                                                    | summing audio streams   |
| `rgb_to_gray` | `gray[i] = (77·R + 150·G + 29·B + 128) >> 8` (Rec. 601 8.8 fixed-point)   | video luminance         |

Each kernel is exposed under `simd::scalar::…`, `simd::sse::…` (on x86)
and `simd::neon::…` (on ARM), plus a `simd::best::…` alias that selects
the fastest available implementation at compile time.

## Layout

```
simd_compare/
├── CMakeLists.txt
├── include/simd_ops.hpp            # public kernel declarations + capability flags
├── src/
│   ├── kernels_scalar.cpp          # always built
│   ├── kernels_sse.cpp             # built on x86/x86_64
│   ├── kernels_neon.cpp            # built on aarch64 / armv7+neon
│   └── bench.cpp                   # micro-benchmark (`simd_bench`)
└── tests/test_kernels.cpp          # cross-impl correctness tests (`simd_tests`)
```

## Build & run locally

```sh
cmake -S simd_compare -B simd_compare/build -DCMAKE_BUILD_TYPE=Release
cmake --build simd_compare/build --parallel
ctest --test-dir simd_compare/build --output-on-failure
./simd_compare/build/simd_bench
```

Expected output on an x86_64 machine (numbers obviously vary):

```
SIMD kernel benchmark
  target arch : x86_64
  SSE build   : yes
  NEON build  : no
  best impl   : sse

  kernel         impl              time       throughput
  -----------------------------------------------------------
  audio_gain     scalar       28.823 ms       54.21 GiB/s
  audio_gain     sse          33.012 ms       47.33 GiB/s
  audio_mix      scalar       43.546 ms       53.82 GiB/s
  audio_mix      sse          42.346 ms       55.35 GiB/s
  rgb_to_gray    scalar      196.886 ms        3.97 GiB/s
  rgb_to_gray    sse          61.190 ms       12.77 GiB/s
```

For the simple float kernels the compiler's auto-vectorizer already
reaches near-memory-bandwidth on the scalar code; the intrinsics pay
off most visibly on `rgb_to_gray`, where deinterleaving packed 24-bit
RGB is awkward for the auto-vectorizer.

## CI

The `.github/workflows/simd.yml` workflow builds, tests, and benchmarks
the project on two GitHub-hosted Linux runners:

- `ubuntu-latest`      → **x86_64** → SSE path
- `ubuntu-24.04-arm`   → **aarch64** → NEON path

giving a direct side-by-side comparison of SSE vs NEON on real hardware.
