#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Generate an integrated Markdown summary that combines the Python CPU
benchmark results and the C++ SIMD test/benchmark results produced by
the `all-tests` workflow into a single pretty job-summary page.

Usage:
    generate_summary.py <results_root> <output_file>
    generate_summary.py --single-arch <arch> <arch_dir> <output_file>

In the default (aggregated) mode, `<results_root>` is expected to
contain one sub-directory per architecture (e.g. `x86_64/`, `aarch64/`),
each holding the JSON and log files produced by a single matrix run.
Missing directories or files are handled gracefully so the summary can
still be emitted if one arch fails.

In `--single-arch` mode, a focused summary is emitted for a single
architecture (`<arch>`) using the files directly inside `<arch_dir>`.
This is meant to be rendered on the per-matrix-job summary page.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple, TypedDict


ARCHES: List[str] = ["x86_64", "aarch64"]


class Scoreboard(TypedDict):
    wins_x: int
    wins_a: int
    ties: int
    total: int
    geomean_ratio: float
    details: List[Tuple[str, float, float]]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def load_python_results(arch_dir: str) -> Dict[str, float]:
    """Return {benchmark_name: elapsed_seconds} for the given arch dir."""
    out: Dict[str, float] = {}
    for path in sorted(glob.glob(os.path.join(arch_dir, "benchmark_results_*.json"))):
        data = _load_json(path)
        if not isinstance(data, list):
            continue
        for item in data:
            if "benchmark" in item and "elapsed_seconds" in item:
                out[item["benchmark"]] = float(item["elapsed_seconds"])
    return out


def load_simd_results(arch_dir: str) -> Optional[dict]:
    """Return the parsed simd_bench JSON document, or None."""
    for path in sorted(glob.glob(os.path.join(arch_dir, "simd_bench_*.json"))):
        data = _load_json(path)
        if isinstance(data, dict):
            return data
    return None


def load_tests_status(arch_dir: str) -> Tuple[str, Optional[str]]:
    """Return ('pass'|'fail'|'missing', log_snippet_or_none)."""
    for path in sorted(glob.glob(os.path.join(arch_dir, "simd_tests_*.log"))):
        try:
            with open(path) as f:
                content = f.read()
        except OSError:
            continue
        if "100% tests passed" in content and "0 tests failed" in content:
            return "pass", None
        return "fail", content[-1000:]
    return "missing", None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _faster_cell(a: Optional[float], b: Optional[float],
                 a_label: str, b_label: str,
                 higher_is_better: bool = False) -> str:
    """Produce a 'faster' cell comparing two numbers.

    For timings, lower is better. For throughput, higher is better.
    """
    if a is None or b is None:
        return "N/A"
    if a == b:
        return "🟰 Tie"
    if higher_is_better:
        winner_label, winner_val, loser_val = (
            (a_label, a, b) if a > b else (b_label, b, a)
        )
        pct = (winner_val - loser_val) / loser_val * 100 if loser_val else 0.0
    else:
        winner_label, winner_val, loser_val = (
            (a_label, a, b) if a < b else (b_label, b, a)
        )
        pct = (loser_val - winner_val) / loser_val * 100 if loser_val else 0.0
    emoji = "🔵" if winner_label == "x86_64" else "🟠"
    return f"{emoji} {winner_label} ({pct:.1f}% faster)"


def format_python_table(x86: Dict[str, float], arm: Dict[str, float]) -> List[str]:
    lines: List[str] = []
    lines.append("### 🐍 Python CPU benchmarks")
    lines.append("")
    if not x86 and not arm:
        lines.append("_No Python benchmark results were produced._")
        lines.append("")
        return lines
    lines.append("| Benchmark | x86_64 (s) | aarch64 (s) | Faster |")
    lines.append("|-----------|------------|-------------|--------|")
    for bm in sorted(set(list(x86.keys()) + list(arm.keys()))):
        x_time = x86.get(bm)
        a_time = arm.get(bm)
        x_str = f"{x_time:.4f}" if x_time is not None else "N/A"
        a_str = f"{a_time:.4f}" if a_time is not None else "N/A"
        faster = _faster_cell(x_time, a_time, "x86_64", "aarch64",
                              higher_is_better=False)
        lines.append(f"| {bm} | {x_str} | {a_str} | {faster} |")
    lines.append("")
    lines.append("> Timing measured with `time.perf_counter()`. Lower is better.")
    lines.append("")
    return lines


def format_simd_table(x86: Optional[dict], arm: Optional[dict]) -> List[str]:
    lines: List[str] = []
    lines.append("### ⚡ C++ SIMD micro-benchmarks")
    lines.append("")
    if not x86 and not arm:
        lines.append("_No SIMD benchmark results were produced._")
        lines.append("")
        return lines

    def _index(doc: Optional[dict]) -> Dict[Tuple[str, str], dict]:
        if not doc:
            return {}
        return {(r["kernel"], r["impl"]): r for r in doc.get("results", [])}

    x_idx = _index(x86)
    a_idx = _index(arm)

    best_x = x86.get("best_impl") if x86 else None
    best_a = arm.get("best_impl") if arm else None
    if best_x or best_a:
        lines.append(
            f"- **Best implementation:** x86_64 → `{best_x or 'n/a'}`, "
            f"aarch64 → `{best_a or 'n/a'}`"
        )
        lines.append("")

    lines.append("#### Scalar baseline — x86_64 vs aarch64")
    lines.append("")
    lines.append("| Kernel | x86_64 GiB/s | aarch64 GiB/s | Faster |")
    lines.append("|--------|--------------|---------------|--------|")
    kernels = sorted({k for (k, _i) in list(x_idx.keys()) + list(a_idx.keys())})
    for kernel in kernels:
        xr = x_idx.get((kernel, "scalar"))
        ar = a_idx.get((kernel, "scalar"))
        x_tp = xr["throughput_gibps"] if xr else None
        a_tp = ar["throughput_gibps"] if ar else None
        x_str = f"{x_tp:.2f}" if x_tp is not None else "N/A"
        a_str = f"{a_tp:.2f}" if a_tp is not None else "N/A"
        faster = _faster_cell(x_tp, a_tp, "x86_64", "aarch64",
                              higher_is_better=True)
        lines.append(f"| {kernel} | {x_str} | {a_str} | {faster} |")
    lines.append("")

    lines.append("#### SSE (x86_64) vs NEON (aarch64)")
    lines.append("")
    lines.append("| Kernel | SSE GiB/s (x86_64) | NEON GiB/s (aarch64) | Faster |")
    lines.append("|--------|--------------------|----------------------|--------|")
    for kernel in kernels:
        xr = x_idx.get((kernel, "sse"))
        ar = a_idx.get((kernel, "neon"))
        x_tp = xr["throughput_gibps"] if xr else None
        a_tp = ar["throughput_gibps"] if ar else None
        x_str = f"{x_tp:.2f}" if x_tp is not None else "N/A"
        a_str = f"{a_tp:.2f}" if a_tp is not None else "N/A"
        faster = _faster_cell(x_tp, a_tp, "x86_64", "aarch64",
                              higher_is_better=True)
        lines.append(f"| {kernel} | {x_str} | {a_str} | {faster} |")
    lines.append("")
    lines.append("> Throughput measured over read+write bytes. Higher is better.")
    lines.append("")
    return lines


def format_tests_section(statuses: Dict[str, Tuple[str, Optional[str]]]) -> List[str]:
    lines: List[str] = []
    lines.append("### ✅ C++ SIMD correctness tests")
    lines.append("")
    lines.append("| Architecture | Status |")
    lines.append("|--------------|--------|")
    icon = {"pass": "🟢 passed", "fail": "🔴 failed", "missing": "⚪ not run"}
    for arch in ARCHES:
        status, _ = statuses.get(arch, ("missing", None))
        lines.append(f"| {arch} | {icon[status]} |")
    lines.append("")
    for arch in ARCHES:
        status, snippet = statuses.get(arch, ("missing", None))
        if status == "fail" and snippet:
            lines.append(f"<details><summary>{arch} failure log (tail)</summary>")
            lines.append("")
            lines.append("```")
            lines.append(snippet)
            lines.append("```")
            lines.append("</details>")
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_summary(results_root: str) -> str:
    py: Dict[str, Dict[str, float]] = {}
    simd: Dict[str, Optional[dict]] = {}
    tests: Dict[str, Tuple[str, Optional[str]]] = {}
    for arch in ARCHES:
        arch_dir = os.path.join(results_root, arch)
        py[arch] = load_python_results(arch_dir) if os.path.isdir(arch_dir) else {}
        simd[arch] = load_simd_results(arch_dir) if os.path.isdir(arch_dir) else None
        tests[arch] = (
            load_tests_status(arch_dir) if os.path.isdir(arch_dir) else ("missing", None)
        )

    lines: List[str] = []
    lines.append("## 🏁 Integrated ARM vs Intel test results")
    lines.append("")
    lines.append(
        "Side-by-side results for **all** suites in this repository — "
        "Python CPU benchmarks, C++ SIMD correctness tests, and C++ SIMD "
        "micro-benchmarks — on `ubuntu-24.04` (x86_64) and "
        "`ubuntu-24.04-arm` (aarch64) GitHub-hosted runners."
    )
    lines.append("")
    scoreboard = _compute_scoreboard(
        py.get("x86_64", {}), py.get("aarch64", {}),
        simd.get("x86_64"), simd.get("aarch64"),
    )
    lines.extend(format_scoreboard(scoreboard, tests))
    lines.extend(format_tests_section(tests))
    lines.extend(format_python_table(py.get("x86_64", {}), py.get("aarch64", {})))
    lines.extend(format_simd_table(simd.get("x86_64"), simd.get("aarch64")))
    return "\n".join(lines) + "\n"


def _compute_scoreboard(
    py_x: Dict[str, float], py_a: Dict[str, float],
    simd_x: Optional[dict], simd_a: Optional[dict],
) -> Scoreboard:
    """Compute per-arch wins and an aggregate geometric-mean speed ratio.

    All metrics are converted to "higher is better" values before being
    compared, so a single ``aarch64 / x86_64`` ratio is meaningful
    across both Python timings and SIMD throughputs.
    """
    details: List[Tuple[str, float, float]] = []

    # Python benchmarks: lower seconds is better → convert to speed (1/t).
    for bm in sorted(set(list(py_x.keys()) + list(py_a.keys()))):
        x = py_x.get(bm)
        a = py_a.get(bm)
        if x and a and x > 0 and a > 0:
            details.append((f"py:{bm}", 1.0 / x, 1.0 / a))

    # SIMD: higher throughput is better → use throughput directly.
    def _idx(doc: Optional[dict]) -> Dict[Tuple[str, str], float]:
        if not doc:
            return {}
        return {
            (r["kernel"], r["impl"]): float(r["throughput_gibps"])
            for r in doc.get("results", [])
            if r.get("throughput_gibps") is not None
        }

    xi = _idx(simd_x)
    ai = _idx(simd_a)
    kernels = sorted({k for (k, _i) in list(xi.keys()) + list(ai.keys())})
    for kernel in kernels:
        for impl_x, impl_a, tag in (
            ("scalar", "scalar", f"simd-scalar:{kernel}"),
            ("sse",    "neon",   f"simd-isa:{kernel}"),
        ):
            x = xi.get((kernel, impl_x))
            a = ai.get((kernel, impl_a))
            if x and a and x > 0 and a > 0:
                details.append((tag, x, a))

    wins_x = wins_a = ties = 0
    ratios: List[float] = []
    for _name, x, a in details:
        if a > x:
            wins_a += 1
        elif x > a:
            wins_x += 1
        else:
            ties += 1
        ratios.append(a / x)

    if ratios:
        geom = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    else:
        geom = 1.0

    return {
        "wins_x": wins_x,
        "wins_a": wins_a,
        "ties": ties,
        "total": len(details),
        "geomean_ratio": geom,
        "details": details,
    }


def format_scoreboard(sb: Scoreboard,
                      tests: Dict[str, Tuple[str, Optional[str]]]) -> List[str]:
    lines: List[str] = []
    lines.append("### 🏆 At-a-glance scoreboard")
    lines.append("")

    wins_x = sb["wins_x"]
    wins_a = sb["wins_a"]
    ties = sb["ties"]
    total = sb["total"]
    geom = sb["geomean_ratio"]

    if total == 0:
        lines.append("_Not enough paired results to compute a scoreboard._")
        lines.append("")
        return lines

    if geom > 1.0:
        headline_arch = "🟠 **aarch64** is faster overall"
        headline_pct = f"{(geom - 1.0) * 100:.1f}% faster (geomean across {total} metrics)"
    elif geom < 1.0:
        headline_arch = "🔵 **x86_64** is faster overall"
        headline_pct = f"{(1.0 / geom - 1.0) * 100:.1f}% faster (geomean across {total} metrics)"
    else:
        headline_arch = "🟰 **Tied** overall"
        headline_pct = f"across {total} metrics"

    status_icon = {"pass": "🟢", "fail": "🔴", "missing": "⚪"}
    x_status = status_icon[tests.get("x86_64", ("missing", None))[0]]
    a_status = status_icon[tests.get("aarch64", ("missing", None))[0]]

    lines.append(f"- **Overall winner:** {headline_arch} — {headline_pct}.")
    lines.append(
        f"- **Wins:** 🔵 x86_64 {wins_x} · 🟠 aarch64 {wins_a} · 🟰 ties {ties} "
        f"(out of {total} paired metrics)."
    )
    lines.append(
        f"- **C++ SIMD tests:** 🔵 x86_64 {x_status} · 🟠 aarch64 {a_status}."
    )
    lines.append("")
    lines.append("| | 🔵 x86_64 (SSE) | 🟠 aarch64 (NEON) |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Wins across paired metrics | **{wins_x}** | **{wins_a}** |"
    )
    lines.append(
        f"| C++ SIMD correctness tests | {x_status} | {a_status} |"
    )
    if geom > 1.0:
        lines.append(
            f"| Overall speed (geomean) | 1.00× | **{geom:.2f}×** |"
        )
    elif geom < 1.0:
        lines.append(
            f"| Overall speed (geomean) | **{1.0 / geom:.2f}×** | 1.00× |"
        )
    else:
        lines.append("| Overall speed (geomean) | 1.00× | 1.00× |")
    lines.append("")
    lines.append(
        "> Speed ratios use `1/elapsed` for Python timings and raw "
        "throughput for SIMD kernels, aggregated with a geometric mean. "
        "Detailed per-benchmark tables follow below."
    )
    lines.append("")
    return lines



    """Build a focused Markdown summary for one architecture."""
    isa = {"x86_64": "SSE", "aarch64": "NEON"}.get(arch, "scalar")
    emoji = {"x86_64": "🔵", "aarch64": "🟠"}.get(arch, "⚪")
    have_dir = os.path.isdir(arch_dir)
    py = load_python_results(arch_dir) if have_dir else {}
    simd = load_simd_results(arch_dir) if have_dir else None
    status, snippet = (
        load_tests_status(arch_dir) if have_dir else ("missing", None)
    )

    lines: List[str] = []
    lines.append(f"## {emoji} All tests — {arch} (Python + {isa})")
    lines.append("")
    lines.append(
        f"Results from the Python CPU benchmarks and the C++ SIMD "
        f"build / test / micro-benchmark suites on this `{arch}` runner."
    )
    lines.append("")

    # ---- C++ SIMD correctness tests -----------------------------------
    icon = {"pass": "🟢 passed", "fail": "🔴 failed", "missing": "⚪ not run"}
    lines.append("### ✅ C++ SIMD correctness tests")
    lines.append("")
    lines.append(f"**Status:** {icon[status]}")
    lines.append("")
    if status == "fail" and snippet:
        lines.append("<details><summary>Failure log (tail)</summary>")
        lines.append("")
        lines.append("```")
        lines.append(snippet)
        lines.append("```")
        lines.append("</details>")
        lines.append("")

    # ---- Python benchmarks --------------------------------------------
    lines.append("### 🐍 Python CPU benchmarks")
    lines.append("")
    if not py:
        lines.append("_No Python benchmark results were produced._")
        lines.append("")
    else:
        lines.append("| Benchmark | Elapsed (s) |")
        lines.append("|-----------|-------------|")
        for bm in sorted(py):
            lines.append(f"| {bm} | {py[bm]:.4f} |")
        lines.append("")
        lines.append("> Timing measured with `time.perf_counter()`. Lower is better.")
        lines.append("")

    # ---- SIMD micro-benchmarks ----------------------------------------
    lines.append(f"### ⚡ C++ SIMD micro-benchmarks ({isa})")
    lines.append("")
    if not simd:
        lines.append("_No SIMD benchmark results were produced._")
        lines.append("")
    else:
        best = simd.get("best_impl")
        if best:
            lines.append(f"- **Best implementation:** `{best}`")
            lines.append("")
        results = simd.get("results", [])
        kernels = sorted({r["kernel"] for r in results})
        impls = sorted({r["impl"] for r in results})
        index = {(r["kernel"], r["impl"]): r for r in results}
        if kernels and impls:
            header = "| Kernel | " + " | ".join(f"{i} GiB/s" for i in impls) + " |"
            sep = "|--------|" + "|".join(["------"] * len(impls)) + "|"
            lines.append(header)
            lines.append(sep)
            for kernel in kernels:
                row = [kernel]
                for impl in impls:
                    r = index.get((kernel, impl))
                    row.append(f"{r['throughput_gibps']:.2f}" if r else "N/A")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
            lines.append(
                "> Throughput measured over read+write bytes. Higher is better."
            )
            lines.append("")

    lines.append(
        "> The cross-architecture comparison is available on the "
        "**🏁 Integrated Summary** job of this workflow run."
    )
    return "\n".join(lines) + "\n"


def main(argv: List[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--single-arch":
        if len(argv) != 5:
            print(
                f"usage: {argv[0]} --single-arch <arch> <arch_dir> <output_file>",
                file=sys.stderr,
            )
            return 2
        arch, arch_dir, output_file = argv[2], argv[3], argv[4]
        summary = build_single_arch_summary(arch, arch_dir)
    else:
        if len(argv) != 3:
            print(
                f"usage: {argv[0]} <results_root> <output_file>\n"
                f"       {argv[0]} --single-arch <arch> <arch_dir> <output_file>",
                file=sys.stderr,
            )
            return 2
        results_root, output_file = argv[1], argv[2]
        summary = build_summary(results_root)

    # Print to stdout and also append to the chosen output file.
    print(summary)
    try:
        with open(output_file, "a") as f:
            f.write(summary)
    except OSError as exc:
        print(f"warning: could not write to {output_file}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
