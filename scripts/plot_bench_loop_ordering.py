#!/usr/bin/env python3
"""Plot Google Benchmark loop-ordering results for mm-test-library.

Input is the JSON emitted by:

    ./build/mm_bench --benchmark_format=json --benchmark_out=bench/results.json

The script expects benchmark names like:

    run_variant/ipj/256
    run_variant/ipj/256_median

and plots GFLOP/s vs problem size, one line per variant.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "missing Python dependency: matplotlib\n"
        "Install it with one of:\n"
        "  python3 -m pip install matplotlib\n"
        "  sudo apt install python3-matplotlib\n"
    ) from exc


VARIANT_ORDER = ["ijp", "ipj", "jip", "jpi", "pij", "pji"]
VARIANT_STYLE = {
    "ipj": {"linestyle": "-", "marker": "o"},
    "pij": {"linestyle": "-", "marker": "s"},
    "ijp": {"linestyle": "--", "marker": "o"},
    "jip": {"linestyle": "--", "marker": "s"},
    "jpi": {"linestyle": ":", "marker": "o"},
    "pji": {"linestyle": ":", "marker": "s"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot GEMM loop-ordering GFLOP/s from Google Benchmark JSON."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="bench/results.json",
        help="Google Benchmark JSON file (default: bench/results.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="bench/loop_ordering_gflops.png",
        help="output plot path (default: bench/loop_ordering_gflops.png)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["iteration", "mean", "median"],
        default="iteration",
        help=(
            "which benchmark rows to plot. Use median/mean when the JSON was "
            "created with --benchmark_repetitions."
        ),
    )
    parser.add_argument(
        "--title",
        default="Row-major GEMM loop-order comparison",
        help="plot title",
    )
    return parser.parse_args()


def wanted_row(name: str, aggregate: str) -> tuple[str, bool]:
    """Return (base_name, keep) for a benchmark row name."""
    aggregate_suffixes = ("_mean", "_median", "_stddev", "_cv")

    if aggregate == "iteration":
        if name.endswith(aggregate_suffixes):
            return name, False
        return name, True

    suffix = f"_{aggregate}"
    if name.endswith(suffix):
        return name[: -len(suffix)], True
    return name, False


def counter_to_gflops(value: float) -> float:
    """Google Benchmark JSON stores rate counters as raw units/second.

    The console pretty-prints them as G/s, but the JSON value is commonly
    around 1e9 for 1 GFLOP/s. Convert raw FLOP/s to GFLOP/s.
    """
    return value / 1e9


def load_series(path: Path, aggregate: str) -> dict[str, list[tuple[int, float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    pattern = re.compile(r"^run_variant/([^/]+)/([0-9]+)$")

    for row in data.get("benchmarks", []):
        name = row.get("name", "")
        base_name, keep = wanted_row(name, aggregate)
        if not keep:
            continue

        match = pattern.match(base_name)
        if not match:
            continue

        variant = match.group(1)
        size = int(match.group(2))

        raw_rate = row.get("GFLOP/s")
        if raw_rate is None:
            continue

        series[variant].append((size, counter_to_gflops(float(raw_rate))))

    return series


def plot(series: dict[str, list[tuple[int, float]]], output: Path, title: str) -> None:
    if not series:
        raise SystemExit("no benchmark rows found; check input path/name pattern")

    plt.figure(figsize=(10, 6))

    ordered = [v for v in VARIANT_ORDER if v in series]
    ordered += sorted(v for v in series if v not in VARIANT_ORDER)

    for variant in ordered:
        points = sorted(series[variant])
        xs = [size for size, _ in points]
        ys = [gflops for _, gflops in points]
        plt.plot(xs, ys, label=variant, **VARIANT_STYLE.get(variant, {}))

    plt.xlabel("Matrix size N (square M=N=K)")
    plt.ylabel("GFLOP/s")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Loop order")
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    series = load_series(input_path, args.aggregate)
    plot(series, output_path, args.title)

    total_points = sum(len(points) for points in series.values())
    print(
        f"wrote {output_path} "
        f"({len(series)} variants, {total_points} benchmark points)"
    )


if __name__ == "__main__":
    main()
