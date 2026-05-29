# mm-test-library

A small, layered study of **dense matrix multiplication (GEMM)** in **row-major**
layout. It implements all **six loop orderings** (`ijp`, `ipj`, `jip`, `jpi`,
`pij`, `pji`) on top of a clean BLAS-style layering, so you can see — and
measure — how loop order maps to BLAS-1/BLAS-2 primitives and to cache behavior.

All kernels compute the accumulating product:

```
C(MxN) += A(MxK) * B(KxN)        (row-major: ldA=K, ldB=N, ldC=N)
```

## The layering

The library is built in three layers, each implemented on top of the one below:

```
GEMM  (6 loop orders)        src/mm/gemm.cpp
  └── BLAS-2  (gemv / ger)   src/mm/blas2.cpp
        └── BLAS-1 (dot/axpy) src/mm/blas1.cpp
```

Each loop order maps to a specific BLAS-2 operation, which is in turn realized
with either `dot` or `axpy`:

| variant | outer loop | BLAS-2 body                | BLAS-1 | row-major speed |
|---------|------------|----------------------------|--------|-----------------|
| `ijp`   | i          | `gemv_row_dot`             | dot    | mid             |
| `ipj`   | i          | `gemv_row_axpy`            | axpy   | **fast**        |
| `jip`   | j          | `gemv_col_dot`             | dot    | mid             |
| `jpi`   | j          | `gemv_col_axpy`            | axpy   | **slow**        |
| `pij`   | p          | `ger_row` (rank-1 update)  | axpy   | **fast**        |
| `pji`   | p          | `ger_col` (rank-1 update)  | axpy   | **slow**        |

**Row-major prediction:** the variants whose innermost loop streams both the
output `C` and the read operand with unit stride (`ipj`, `pij`) are fastest;
those that walk `C` down columns (`jpi`, `pji`) are slowest; the `dot`-based
ones (`ijp`, `jip`) sit in between.

## Project layout

```
include/mm/        public headers (blas1.hpp, blas2.hpp, gemm.hpp, log.hpp)
src/mm/            implementations (blas1.cpp, blas2.cpp, gemm.cpp)
test/              gtest correctness suite (test_mm_correctness.cpp)
bench/             google-benchmark harness (mm_bench.cpp)  [work in progress]
examples/mm/       one runnable sample per variant (sample_mm_*.cpp)
cmake/             dependencies.cmake (FetchContent: gtest, benchmark)
third_party/       fetched sources (git-ignored)
```

## Building

Requires CMake (>= 3.16), a C++17 compiler, and network access on first
configure (to fetch GoogleTest and Google Benchmark via FetchContent).

```bash
cmake -S . -B build
cmake --build build -j
```

## Running

```bash
# Correctness tests (all six variants vs a reference)
ctest --test-dir build --output-on-failure

# Examples: a tiny, hand-verifiable 2x3 * 3x2 multiply
./build/sample_mm_ijp        # ... ipj, jip, jpi, pij, pji

# Benchmark (placeholder; see below)
./build/mm_bench
```

## Logging

The library has a compile-time log level (logs go to **stderr**):

| `-DMM_LOG_LEVEL` | output                                   |
|------------------|------------------------------------------|
| `0` (default)    | nothing (zero runtime cost)              |
| `1`              | one line per GEMM call + call summary    |
| `2`              | + each BLAS-2 (`gemv`/`ger`) call        |
| `3`              | + each BLAS-1 (`dot`/`axpy`) call        |

```bash
cmake -S . -B build -DMM_LOG_LEVEL=1
cmake --build build --clean-first -j
./build/sample_mm_ipj 2>log.txt     # logs to stderr, result to stdout
```

The level-1 summary shows how a variant decomposed into BLAS calls, e.g.:

```
[mm] mm_ipj: done M=2 N=2 K=3 flops=24 gemv_row_axpy=2 axpy=6
```

## Status / TODO

- [x] BLAS-1 (`dot`, `axpy`), BLAS-2 (`gemv_*`, `ger_*`), six GEMM variants
- [x] Correctness suite (shapes, identity, zero, accumulation, padded strides, negative dims)
- [x] Per-variant examples and layered logging
- [ ] Full benchmark sweep over all six variants (current `mm_bench.cpp` is a placeholder)
- [ ] Plotting / analysis of GFLOP/s vs size
