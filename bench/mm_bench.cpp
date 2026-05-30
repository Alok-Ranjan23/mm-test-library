#include <benchmark/benchmark.h>
#include <vector>
#include <algorithm>
#include "gemm.hpp"

using mm_fn = void(*)(int,int,int,const float*,int,const float*,int,float*,int);

static void run_variant(benchmark::State& state, mm_fn fn) {
    const int N = static_cast<int>(state.range(0));   // square: M=N=K=N
    const int M = N, K = N;
    std::vector<float> A(size_t(M)*K, 1.0f);
    std::vector<float> B(size_t(K)*N, 1.0f);
    std::vector<float> C(size_t(M)*N, 0.0f);
    for (auto _ : state) {
        std::fill(C.begin(), C.end(), 0.0f);          // kernels accumulate
        fn(M, N, K, A.data(), K, B.data(), N, C.data(), N);
        benchmark::DoNotOptimize(C.data());
        benchmark::ClobberMemory();
    }
    // GFLOP/s = 2*M*N*K flops per call, reported as a rate over time
    state.counters["GFLOP/s"] = benchmark::Counter(
        2.0 * M * N * K,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000);
}

// One registration per variant. ->RangeMultiplier(2)->Range(64,1024)
// sweeps N = 64,128,...,1024.
BENCHMARK_CAPTURE(run_variant, ijp, &mm_ijp)->RangeMultiplier(2)->Range(64,1024);
BENCHMARK_CAPTURE(run_variant, ipj, &mm_ipj)->RangeMultiplier(2)->Range(64,1024);
BENCHMARK_CAPTURE(run_variant, jip, &mm_jip)->RangeMultiplier(2)->Range(64,1024);
BENCHMARK_CAPTURE(run_variant, jpi, &mm_jpi)->RangeMultiplier(2)->Range(64,1024);
BENCHMARK_CAPTURE(run_variant, pij, &mm_pij)->RangeMultiplier(2)->Range(64,1024);
BENCHMARK_CAPTURE(run_variant, pji, &mm_pji)->RangeMultiplier(2)->Range(64,1024);

BENCHMARK_MAIN();
