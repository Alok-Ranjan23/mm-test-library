#include <benchmark/benchmark.h>
#include <vector>

#include "gemm.hpp"

// Placeholder benchmark so the project configures. Replace with the full
// sweep over all six variants (GFLOP/s via custom counters).
static void BM_mm_ijp(benchmark::State& state) {
    const int N = static_cast<int>(state.range(0));
    const int M = N, K = N;
    const int ldA = K, ldB = N, ldC = N;

    std::vector<float> A(static_cast<size_t>(M) * K, 1.0f);
    std::vector<float> B(static_cast<size_t>(K) * N, 1.0f);
    std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);

    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);
        mm_ijp(M, N, K, A.data(), ldA, B.data(), ldB, C.data(), ldC);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    state.counters["GFLOPs"] = benchmark::Counter(
        2.0 * M * N * K, benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
}
BENCHMARK(BM_mm_ijp)->Arg(128)->Arg(256)->Arg(512);

BENCHMARK_MAIN();
