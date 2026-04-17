# compare_linux_runners

Projects that exercise and compare the different Linux runners available
on GitHub Actions — ARM (`aarch64`) vs Intel (`x86_64`), SSE vs NEON, etc.

## Projects

- **Python CPU benchmark** (this directory) — a suite of CPU-bound Python
  benchmarks (NumPy BLAS, prime sieve, SHA-256, sorting, zlib, JSON, …)
  run in parallel on `ubuntu-24.04` and `ubuntu-24.04-arm` to produce a
  side-by-side comparison table. See below.
- [`simd_compare/`](simd_compare/README.md) — modern **C++23** SIMD
  kernels (audio gain, audio mix, RGB→gray) implemented with a scalar
  baseline plus **SSE** (x86_64) and **NEON** (aarch64) intrinsics.
  Includes a CMake build, correctness tests, a throughput micro-benchmark
  and a GitHub Actions workflow that runs everything on both
  `ubuntu-latest` and `ubuntu-24.04-arm` runners.

### Integrated workflow — `all-tests`

The [`all-tests`](.github/workflows/all_tests.yml) workflow is the
**single, orchestrated pipeline** that runs every suite in this
repository. Each matrix job (`All tests — x86_64 (Python + SSE)` and
`All tests — aarch64 (Python + NEON)`) executes **both** the Python CPU
benchmarks and the C++ SIMD build / correctness tests / micro-benchmarks
on its architecture, and emits its own focused Markdown summary right on
the job page. A final **🏁 Integrated Summary (x86_64 vs aarch64)** job
then downloads the artifacts from both matrix jobs and renders one
unified side-by-side summary containing:

1. an **🏆 At-a-glance scoreboard** (overall winner, per-arch wins, geomean speed ratio),
2. C++ SIMD correctness test status per architecture,
3. the Python CPU benchmark comparison table,
4. a scalar x86_64 vs aarch64 C++ throughput table,
5. an SSE vs NEON C++ throughput table.

On pull requests, the integrated summary is also **posted (and updated
in place) as a PR comment** so the cross-architecture comparison shows
up right on the PR conversation — no need to click through to the
Actions tab.

Open any run of the `all-tests` workflow and you can either scroll to
any matrix job to see the results for that architecture, or open the
**🏁 Integrated Summary** job to see every suite rendered side by side.

---

## Python CPU benchmark

Compare CPU performance of GitHub-hosted **ARM (aarch64)** vs **Intel (x86_64)** Linux runners using a suite of CPU-bound Python benchmarks.

### Why?

GitHub provides free ARM runners (`ubuntu-24.04-arm`) for **public repositories** — no extra cost, no quota concerns. This project lets you see how ARM stacks up against Intel for common computational tasks right inside GitHub Actions.

### Benchmarks

| Benchmark | What it tests |
|-----------|--------------|
| `matrix_multiplication` | NumPy 1024×1024 double-precision matrix multiply (BLAS) |
| `prime_sieve` | Sieve of Eratosthenes up to 5_000_000 (integer/memory throughput) |
| `sha256_hashing` | SHA-256 over 200 MB of data (crypto / hash throughput) |
| `fibonacci_memoized` | Recursive Fibonacci up to n=10_000 with `lru_cache` (function-call overhead) |
| `list_sorting` | Sort 5_000_000 random floats with Python's Timsort |
| `float_math_trig` | NumPy `sin`, `cos`, `tan` over 10_000_000 elements (SIMD / FPU throughput) |
| `zlib_compression` | zlib compress + decompress ~9 MB payload at level 6 |
| `json_serialization` | `json.dumps` + `json.loads` on a 100_000-item list of dicts |

### How to run locally

```bash
pip install numpy
python benchmark.py
```

Results are printed to stdout and also saved to `benchmark_results_<arch>.json`.

### How to trigger the workflow

1. Push a commit to `main`, open a pull request targeting `main`, or navigate to **Actions → all-tests → Run workflow**.
2. The matrix runs in parallel on both `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm` (aarch64), each executing the Python and C++ SIMD suites.
3. Each matrix job uploads an `all-results-<arch>` artifact and renders a per-architecture Markdown summary on its own job page.
4. The **🏁 Integrated Summary** job downloads both artifacts and writes the unified comparison tables to its job summary.

#### Viewing results

- Open the workflow run on GitHub.
- Click any matrix job (e.g. **All tests — x86_64 (Python + SSE)**) to see the per-architecture summary, or
- click the **🏁 Integrated Summary (x86_64 vs aarch64)** job to see the side-by-side comparison.
- Raw JSON artifacts are available under **Artifacts** at the bottom of the run page.

### ARM runners are free on public repos

> GitHub-hosted ARM runners (`ubuntu-24.04-arm`, `ubuntu-22.04-arm`) are **free and unlimited on public repositories**.  
> For private repositories they consume your included-minutes allotment and are then billed at standard rates.  
> See the [GitHub Actions billing docs](https://docs.github.com/en/billing/reference/actions-minute-multipliers) for details.

### Example comparison table

| Benchmark | x86_64 (s) | aarch64 (s) | Faster |
|-----------|------------|-------------|--------|
| matrix_multiplication | 0.4123 | 0.3891 | 🟠 aarch64 (5.6% faster) |
| prime_sieve | 0.8210 | 0.7654 | 🟠 aarch64 (6.8% faster) |
| sha256_hashing | 1.2045 | 1.1832 | 🟠 aarch64 (1.8% faster) |
| fibonacci_memoized | 0.0531 | 0.0487 | 🟠 aarch64 (8.3% faster) |
| list_sorting | 1.9321 | 2.0104 | 🔵 x86_64 (3.9% faster) |
| float_math_trig | 0.3654 | 0.3412 | 🟠 aarch64 (6.6% faster) |
| zlib_compression | 0.6789 | 0.7012 | 🔵 x86_64 (3.2% faster) |
| json_serialization | 0.9234 | 0.8876 | 🟠 aarch64 (3.9% faster) |

> Numbers above are illustrative. Run the workflow to see real results for your workload.
