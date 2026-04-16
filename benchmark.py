#!/usr/bin/env python3
"""
CPU-bound benchmark suite for comparing ARM (aarch64) vs Intel (x86_64) runners.
Results are printed to stdout as JSON for easy collection in CI pipelines.
"""

import json
import math
import platform
import random
import time
import zlib
import hashlib
import sys
from functools import lru_cache

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not available; matrix and float benchmarks will be skipped.", file=sys.stderr)


ARCH = platform.machine()
results = []


def bench(name):
    """Decorator that times a benchmark function and appends the result."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            record = {"benchmark": name, "elapsed_seconds": round(elapsed, 6), "arch": ARCH}
            results.append(record)
            print(f"  {name}: {elapsed:.4f}s  [{ARCH}]")
            return elapsed
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 1. Matrix multiplication (NumPy)
# ---------------------------------------------------------------------------
if HAS_NUMPY:
    @bench("matrix_multiplication")
    def run_matrix_multiplication():
        size = 1024
        a = np.random.rand(size, size).astype(np.float64)
        b = np.random.rand(size, size).astype(np.float64)
        np.dot(a, b)


# ---------------------------------------------------------------------------
# 2. Prime number sieve (Sieve of Eratosthenes)
# ---------------------------------------------------------------------------
@bench("prime_sieve")
def run_prime_sieve():
    limit = 5_000_000
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    # consume the result so it is not optimised away
    _ = sum(sieve)


# ---------------------------------------------------------------------------
# 3. SHA-256 hashing
# ---------------------------------------------------------------------------
@bench("sha256_hashing")
def run_sha256_hashing():
    chunk = b"x" * (1024 * 1024)  # 1 MB chunk
    h = hashlib.sha256()
    for _ in range(200):          # hash 200 MB total
        h.update(chunk)
    h.hexdigest()


# ---------------------------------------------------------------------------
# 4. Fibonacci with memoization
# ---------------------------------------------------------------------------
@bench("fibonacci_memoized")
def run_fibonacci():
    @lru_cache(maxsize=None)
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    # Warm up cache then time a large value
    for n in range(0, 10_001):
        fib(n)


# ---------------------------------------------------------------------------
# 5. Large list sorting
# ---------------------------------------------------------------------------
@bench("list_sorting")
def run_list_sorting():
    rng = random.Random(42)
    data = [rng.random() for _ in range(5_000_000)]
    data.sort()


# ---------------------------------------------------------------------------
# 6. Floating-point math (trigonometric operations)
# ---------------------------------------------------------------------------
if HAS_NUMPY:
    @bench("float_math_trig")
    def run_float_math():
        arr = np.linspace(0, 2 * np.pi, 10_000_000)
        np.sin(arr)
        np.cos(arr)
        np.tan(arr)


# ---------------------------------------------------------------------------
# 7. Compression (zlib compress / decompress)
# ---------------------------------------------------------------------------
@bench("zlib_compression")
def run_zlib_compression():
    payload = b"The quick brown fox jumps over the lazy dog. " * 200_000
    compressed = zlib.compress(payload, level=6)
    zlib.decompress(compressed)


# ---------------------------------------------------------------------------
# 8. JSON serialization / deserialization
# ---------------------------------------------------------------------------
@bench("json_serialization")
def run_json_serialization():
    rng = random.Random(0)
    data = [
        {
            "id": i,
            "name": f"item_{i}",
            "value": rng.random(),
            "tags": [f"tag_{j}" for j in range(5)],
        }
        for i in range(100_000)
    ]
    serialized = json.dumps(data)
    json.loads(serialized)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nRunning benchmarks on {ARCH}\n")

    if HAS_NUMPY:
        run_matrix_multiplication()
    run_prime_sieve()
    run_sha256_hashing()
    run_fibonacci()
    run_list_sorting()
    if HAS_NUMPY:
        run_float_math()
    run_zlib_compression()
    run_json_serialization()

    print("\nResults (JSON):")
    print(json.dumps(results, indent=2))

    # Write results to a file for artifact upload
    output_file = f"benchmark_results_{ARCH}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_file}")
