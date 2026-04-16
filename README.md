# compare_linux_runners

Compare CPU performance of GitHub-hosted **ARM (aarch64)** vs **Intel (x86_64)** Linux runners using a suite of CPU-bound Python benchmarks.

## Why?

GitHub provides free ARM runners (`ubuntu-24.04-arm`) for **public repositories** — no extra cost, no quota concerns. This project lets you see how ARM stacks up against Intel for common computational tasks right inside GitHub Actions.

## Benchmarks

| Benchmark | What it tests |
|-----------|--------------|
| `matrix_multiplication` | NumPy 1024×1024 double-precision matrix multiply (BLAS) |
| `prime_sieve` | Sieve of Eratosthenes up to 5 000 000 (integer/memory throughput) |
| `sha256_hashing` | SHA-256 over 200 MB of data (crypto / hash throughput) |
| `fibonacci_memoized` | Recursive Fibonacci up to n=10 000 with `lru_cache` (function-call overhead) |
| `list_sorting` | Sort 5 000 000 random floats with Python's Timsort |
| `float_math_trig` | NumPy `sin`, `cos`, `tan` over 10 000 000 elements (SIMD / FPU throughput) |
| `zlib_compression` | zlib compress + decompress ~9 MB payload at level 6 |
| `json_serialization` | `json.dumps` + `json.loads` on a 100 000-item list of dicts |

## How to run locally

```bash
pip install numpy
python benchmark.py
```

Results are printed to stdout and also saved to `benchmark_results_<arch>.json`.

## How to trigger the workflow

1. Push a commit to `main`, open a pull request targeting `main`, or navigate to **Actions → ARM vs Intel Benchmark → Run workflow**.
2. The `benchmark` job runs in parallel on both `ubuntu-24.04` (x86_64) and `ubuntu-24.04-arm` (aarch64).
3. Each job uploads a `benchmark_results_<arch>.json` artifact.
4. The `summary` job downloads both artifacts and writes a Markdown comparison table to the **job summary**.

### Viewing results

- Open the workflow run on GitHub.
- Click the **Comparison Summary** job.
- Scroll down to the **job summary** pane to see the side-by-side table.
- Raw JSON artifacts are available under **Artifacts** at the bottom of the run page.

## ARM runners are free on public repos

> GitHub-hosted ARM runners (`ubuntu-24.04-arm`, `ubuntu-22.04-arm`) are **free and unlimited on public repositories**.  
> For private repositories they consume your included-minutes allotment and are then billed at standard rates.  
> See the [GitHub Actions billing docs](https://docs.github.com/en/billing/reference/actions-minute-multipliers) for details.

## Example comparison table

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
