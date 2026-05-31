# mm-test-library

A small C++17 study of dense matrix multiplication (GEMM) in row-major layout.
It implements all six loop orderings (`ijp`, `ipj`, `jip`, `jpi`, `pij`, `pji`)
using a simple layered structure so the connection between GEMM, BLAS-2, and
BLAS-1 is explicit.

All kernels compute the accumulating product:

```text
C(MxN) += A(MxK) * B(KxN)
```

For tightly packed row-major matrices:

```text
ldA = K, ldB = N, ldC = N
```

The kernels intentionally use `+=`, so callers should zero `C` first when they
want plain `C = A * B`.

## Layering
The implementation is organized as:

```text
GEMM  (6 loop orders)          src/mm/gemm.cpp
  -> BLAS-2 (gemv / ger)       src/mm/blas2.cpp
       -> BLAS-1 (dot / axpy)  src/mm/blas1.cpp
```

Each loop order maps to one BLAS-2 view:

| Variant | Outer loop | BLAS-2 body | BLAS-1 primitive | Row-major expectation |
|---|---|---|---|---|
| `ijp` | `i` | `gemv_row_dot` | `dot` | middle |
| `ipj` | `i` | `gemv_row_axpy` | `axpy` | fast |
| `jip` | `j` | `gemv_col_dot` | `dot` | middle |
| `jpi` | `j` | `gemv_col_axpy` | `axpy` | slow |
| `pij` | `p` | `ger_row` rank-1 update | `axpy` | fast |
| `pji` | `p` | `ger_col` rank-1 update | `axpy` | slow |

For row-major storage, `ipj` and `pij` are expected to be fastest because their
inner `axpy` walks rows of `B` and `C` with unit stride. `jpi` and `pji` are
expected to be slowest because they walk columns of row-major matrices.

## Layout

```text
include/mm/        public headers
src/mm/            library implementation
test/              GoogleTest correctness tests
bench/             Google Benchmark harness and benchmark outputs
examples/mm/       one runnable sample per loop ordering
scripts/           plotting scripts
cmake/             dependency setup via FetchContent
third_party/       fetched GoogleTest / Google Benchmark sources, git-ignored
```

## Dependencies
The project uses CMake `FetchContent` for:

- GoogleTest `v1.17.0`
- Google Benchmark `v1.9.5`

The first configure needs network access. Downloaded sources are placed under
`third_party/` and are ignored by git.

For plotting, install `matplotlib`:

```bash
python3 -m pip install matplotlib
```

or:

```bash
sudo apt install python3-matplotlib
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMM_LOG_LEVEL=0
cmake --build build -j
```

To see exact compiler commands:

```bash
cmake --build build --verbose
```

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Or run GoogleTest directly:

```bash
./build/test_mm_correctness
./build/test_mm_correctness --gtest_list_tests
./build/test_mm_correctness --gtest_filter='*M64_N64_K64*'
```

The correctness suite checks all six variants against `mm_ref`, plus exact
small examples, identity, zero matrix, accumulation semantics, padded leading
dimensions, and non-positive dimensions.

## Examples
Each sample performs the same hand-verifiable multiplication:

```text
A = [  1  -2   3 ]      B = [ 2   1 ]
    [ -1   0   2 ]          [ 0  -1 ]
                             [ 1   3 ]

C = A * B = [ 5  12 ]
            [ 0   5 ]
```

Run any loop ordering:

```bash
./build/sample_mm_ijp
./build/sample_mm_ipj
./build/sample_mm_jip
./build/sample_mm_jpi
./build/sample_mm_pij
./build/sample_mm_pji
```

## Benchmark
The benchmark target compares the six variants over square sizes:

```text
N = 64, 128, 256, 512, 1024
M = N, K = N
```

Run:

```bash
./build/mm_bench
```

For cleaner single-thread measurements, pin the benchmark to one logical CPU:

```bash
taskset -c 7 ./build/mm_bench
```

For more stable output and JSON export:

```bash
taskset -c 7 ./build/mm_bench \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true \
  --benchmark_format=json \
  --benchmark_out=bench/results.json
```

Google Benchmark reports average time per iteration and the custom `GFLOP/s`
counter:

```text
GFLOP/s = (2 * M * N * K) / seconds_per_iteration
```

The benchmark currently resets `C` with `std::fill` inside each timed iteration
because the kernels accumulate into `C`.

## Plotting
After creating `bench/results.json`, generate a plot:

```bash
python3 scripts/plot_bench_loop_ordering.py bench/results.json
```

This writes:

```text
bench/loop_ordering_gflops.png
```

If the JSON was generated with repetitions and aggregate rows, plot the median:

```bash
python3 scripts/plot_bench_loop_ordering.py bench/results.json --aggregate median
```

Choose a custom output path:

```bash
python3 scripts/plot_bench_loop_ordering.py bench/results.json \
  -o bench/my_loop_ordering_plot.png
```

Generated benchmark data and plots (`*.json`, `*.csv`, `*.png`) are ignored by
git by default.

## Logging
The library has a compile-time log level. Logs go to `stderr`.

| `-DMM_LOG_LEVEL` | Output |
|---|---|
| `0` | off, default |
| `1` | one line per GEMM call and a summary |
| `2` | plus each BLAS-2 call |
| `3` | plus each BLAS-1 `dot` / `axpy` call |

Example:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMM_LOG_LEVEL=1
cmake --build build --clean-first -j
./build/sample_mm_ipj 2>log.txt
```

A level-1 summary looks like:

```text
[mm] mm_ipj: done M=2 N=2 K=3 flops=24 gemv_row_axpy=2 axpy=6
```