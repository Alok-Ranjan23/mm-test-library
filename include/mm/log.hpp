#ifndef MM_LOG_HPP
#define MM_LOG_HPP

#include <iostream>

// Compile-time log level for the mm library:
//   0 = off, 1 = info (per GEMM), 2 = debug (per BLAS-2), 3 = trace (per BLAS-1)
// Set from the build, e.g. -DMM_LOG_LEVEL=2.
#ifndef MM_LOG_LEVEL
#define MM_LOG_LEVEL 0
#endif

// Streamed log, compile-time gated: when (level > MM_LOG_LEVEL) the body is
// discarded by `if constexpr`, leaving zero runtime cost. Logs go to stderr
// so they stay separate from program output on stdout.
#define MM_LOG(level, msg)                                                 \
    do {                                                                   \
        if constexpr ((level) <= MM_LOG_LEVEL)                             \
            std::cerr << "[mm] " << __func__ << ": " << msg << '\n';       \
    } while (0)

namespace mm_detail {

// Per-GEMM call counters: how the multiply decomposed into BLAS-1/2 calls.
struct Counters {
    long dot = 0,          axpy = 0;
    long gemv_row_dot = 0, gemv_row_axpy = 0;
    long gemv_col_dot = 0, gemv_col_axpy = 0;
    long ger_row = 0,      ger_col = 0;
    void reset() { *this = Counters{}; }
};

// One shared instance (C++17 inline variable => a single definition across TUs).
// Not thread-safe; intended for single-threaded study/benchmark runs.
inline Counters g_counters{};

// One-line summary of a GEMM call (only emits at level >= 1).
inline void log_summary(const char* tag, int M, int N, int K) {
#if MM_LOG_LEVEL >= 1
    const Counters& c = g_counters;
    std::cerr << "[mm] " << tag << ": done M=" << M << " N=" << N
              << " K=" << K << " flops=" << (2L * M * N * K);
    if (c.gemv_row_dot)  std::cerr << " gemv_row_dot="  << c.gemv_row_dot;
    if (c.gemv_row_axpy) std::cerr << " gemv_row_axpy=" << c.gemv_row_axpy;
    if (c.gemv_col_dot)  std::cerr << " gemv_col_dot="  << c.gemv_col_dot;
    if (c.gemv_col_axpy) std::cerr << " gemv_col_axpy=" << c.gemv_col_axpy;
    if (c.ger_row)       std::cerr << " ger_row="       << c.ger_row;
    if (c.ger_col)       std::cerr << " ger_col="       << c.ger_col;
    if (c.dot)           std::cerr << " dot="           << c.dot;
    if (c.axpy)          std::cerr << " axpy="          << c.axpy;
    std::cerr << '\n';
#else
    (void)tag; (void)M; (void)N; (void)K;
#endif
}

}  // namespace mm_detail

// Counter macros: no-ops (zero cost) when logging is off.
#if MM_LOG_LEVEL >= 1
#define MM_COUNT(field)   (++mm_detail::g_counters.field)
#define MM_COUNT_RESET()  (mm_detail::g_counters.reset())
#else
#define MM_COUNT(field)   ((void)0)
#define MM_COUNT_RESET()  ((void)0)
#endif

#endif  // MM_LOG_HPP
