#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "gemm.hpp"

namespace {

using mm_fn = void (*)(int, int, int,
                       const float*, int,
                       const float*, int,
                       float*, int);

struct Variant {
    const char* name;
    mm_fn       fn;
};

const Variant kVariants[] = {
    {"ijp", mm_ijp},
    {"ipj", mm_ipj},
    {"jip", mm_jip},
    {"jpi", mm_jpi},
    {"pij", mm_pij},
    {"pji", mm_pji},
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

std::vector<float> random_matrix(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> m(static_cast<size_t>(n));
    for (auto& v : m) v = dist(rng);
    return m;
}

// Compare two buffers with a combined absolute+relative tolerance.
::testing::AssertionResult MatricesClose(const std::vector<float>& got,
                                         const std::vector<float>& want,
                                         double tol) {
    if (got.size() != want.size()) {
        return ::testing::AssertionFailure()
               << "size mismatch: got " << got.size() << " want " << want.size();
    }
    double max_rel = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = std::abs(static_cast<double>(got[i]) - want[i]);
        const double rel  = diff / (1.0 + std::abs(static_cast<double>(want[i])));
        if (rel > max_rel) { max_rel = rel; worst = i; }
    }
    if (max_rel < tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "max relative error " << max_rel << " at index " << worst
           << " (got " << got[worst] << ", want " << want[worst] << ")";
}

// ===========================================================================
// 1. Every variant matches the reference across many shapes.
// ===========================================================================

struct Shape {
    int M, N, K;
};

class MMCorrectness : public ::testing::TestWithParam<Shape> {};

TEST_P(MMCorrectness, MatchesReference) {
    const Shape s = GetParam();
    const int M = s.M, N = s.N, K = s.K;
    const int ldA = K, ldB = N, ldC = N;   // tightly packed, row-major

    const std::vector<float> A = random_matrix(M * K, 1);
    const std::vector<float> B = random_matrix(K * N, 2);

    std::vector<float> Cref(static_cast<size_t>(M) * N, 0.0f);
    mm_ref(M, N, K, A.data(), ldA, B.data(), ldB, Cref.data(), ldC);

    for (const Variant& var : kVariants) {
        SCOPED_TRACE(testing::Message()
                     << "variant=" << var.name
                     << " M=" << M << " N=" << N << " K=" << K);
        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        var.fn(M, N, K, A.data(), ldA, B.data(), ldB, C.data(), ldC);
        EXPECT_TRUE(MatricesClose(C, Cref, 1e-3));
    }
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, MMCorrectness,
    ::testing::Values(
        Shape{1, 1, 1},        // scalar
        Shape{1, 8, 8},        // single row of C
        Shape{8, 1, 8},        // single column of C
        Shape{8, 8, 1},        // rank-1 (K=1)
        Shape{2, 2, 2},        // tiny square
        Shape{16, 16, 16},     // small square
        Shape{32, 48, 40},     // rectangular, non-power-of-2
        Shape{17, 13, 23},     // primes
        Shape{64, 64, 64},     // square
        Shape{128, 96, 72},    // larger rectangular
        Shape{200, 1, 200},    // very tall/thin
        Shape{1, 200, 200}),   // very wide/flat
    [](const testing::TestParamInfo<Shape>& info) {
        const Shape s = info.param;
        return "M" + std::to_string(s.M) +
               "_N" + std::to_string(s.N) +
               "_K" + std::to_string(s.K);
    });

// ===========================================================================
// 2. Exact, hand-verifiable result (matches the examples/).
// ===========================================================================

TEST(MMExact, KnownSmallProduct) {
    const int M = 2, K = 3, N = 2;
    const std::vector<float> A = {1, -2, 3,
                                  -1, 0, 2};
    const std::vector<float> B = {2,  1,
                                  0, -1,
                                  1,  3};
    const std::vector<float> expected = {5, 12,
                                         0, 5};
    for (const Variant& var : kVariants) {
        SCOPED_TRACE(testing::Message() << "variant=" << var.name);
        std::vector<float> C(M * N, 0.0f);
        var.fn(M, N, K, A.data(), K, B.data(), N, C.data(), N);
        EXPECT_TRUE(MatricesClose(C, expected, 1e-5));
    }
}

// ===========================================================================
// 3. Multiply by identity:  A * I == A.
// ===========================================================================

TEST(MMIdentity, RightMultiplyIdentity) {
    const int M = 4, K = 3, N = 3;          // N must equal K for A*I
    const std::vector<float> A = random_matrix(M * K, 7);
    std::vector<float> I(static_cast<size_t>(K) * N, 0.0f);
    for (int d = 0; d < K; ++d) I[d * N + d] = 1.0f;

    for (const Variant& var : kVariants) {
        SCOPED_TRACE(testing::Message() << "variant=" << var.name);
        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        var.fn(M, N, K, A.data(), K, I.data(), N, C.data(), N);
        EXPECT_TRUE(MatricesClose(C, A, 1e-5));   // A is MxK == MxN here
    }
}

// ===========================================================================
// 4. Multiply by zero:  A * 0 == 0.
// ===========================================================================

TEST(MMZero, MultiplyByZeroMatrix) {
    const int M = 3, K = 4, N = 5;
    const std::vector<float> A = random_matrix(M * K, 9);
    const std::vector<float> Bzero(static_cast<size_t>(K) * N, 0.0f);
    const std::vector<float> expected(static_cast<size_t>(M) * N, 0.0f);

    for (const Variant& var : kVariants) {
        SCOPED_TRACE(testing::Message() << "variant=" << var.name);
        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        var.fn(M, N, K, A.data(), K, Bzero.data(), N, C.data(), N);
        EXPECT_TRUE(MatricesClose(C, expected, 1e-6));
    }
}

// ===========================================================================
// 5. Accumulation semantics:  C must be (C_init + A*B), since kernels do +=.
// ===========================================================================

TEST(MMAccumulate, AddsOntoExistingC) {
    const int M = 6, K = 5, N = 7;
    const std::vector<float> A     = random_matrix(M * K, 11);
    const std::vector<float> B     = random_matrix(K * N, 12);
    const std::vector<float> Cinit = random_matrix(M * N, 13);

    std::vector<float> Cref = Cinit;        // start from same initial C
    mm_ref(M, N, K, A.data(), K, B.data(), N, Cref.data(), N);

    for (const Variant& var : kVariants) {
        SCOPED_TRACE(testing::Message() << "variant=" << var.name);
        std::vector<float> C = Cinit;       // same non-zero starting point
        var.fn(M, N, K, A.data(), K, B.data(), N, C.data(), N);
        EXPECT_TRUE(MatricesClose(C, Cref, 1e-3));
    }
}

// ===========================================================================
// 6. Non-unit leading dimensions (padded / sub-matrix storage).
//    Exercises the stride arguments: ld > number of columns.
// ===========================================================================

TEST(MMLeadingDim, PaddedStrides) {
    const int M = 5, N = 7, K = 3;
    const int pad = 4;
    const int ldA = K + pad, ldB = N + pad, ldC = N + pad;

    const std::vector<float> A = random_matrix(M * ldA, 21);
    const std::vector<float> B = random_matrix(K * ldB, 22);

    std::vector<float> Cref(static_cast<size_t>(M) * ldC, 0.0f);
    mm_ref(M, N, K, A.data(), ldA, B.data(), ldB, Cref.data(), ldC);

    for (const Variant& var : kVariants) {
        SCOPED_TRACE(testing::Message() << "variant=" << var.name);
        std::vector<float> C(static_cast<size_t>(M) * ldC, 0.0f);
        var.fn(M, N, K, A.data(), ldA, B.data(), ldB, C.data(), ldC);
        EXPECT_TRUE(MatricesClose(C, Cref, 1e-3));   // padding stays 0 in both
    }
}

// ===========================================================================
// 7. Negative tests: non-positive dimensions must be a safe no-op.
//    The library guards `if (M<=0 || N<=0 || K<=0) return;`, so every kernel
//    must leave C byte-for-byte untouched and must not crash / read OOB.
// ===========================================================================

TEST(MMRobustness, NonPositiveDimsAreNoOp) {
    const int cap = 16;                                   // generous capacity
    const std::vector<float> A(static_cast<size_t>(cap) * cap, 1.5f);
    const std::vector<float> B(static_cast<size_t>(cap) * cap, -0.5f);

    const std::vector<Shape> bad = {
        Shape{-2, 6, 9},   // negative M (the example from the request)
        Shape{6, -3, 9},   // negative N
        Shape{6, 9, -1},   // negative K
        Shape{0, 6, 9},    // zero M
        Shape{6, 0, 9},    // zero N
        Shape{6, 9, 0},    // zero K
        Shape{-1, -1, -1}, // all negative
    };

    for (const Variant& var : kVariants) {
        for (const Shape& s : bad) {
            SCOPED_TRACE(testing::Message()
                         << "variant=" << var.name
                         << " M=" << s.M << " N=" << s.N << " K=" << s.K);
            const std::vector<float> sentinel(static_cast<size_t>(cap) * cap, 7.0f);
            std::vector<float> C = sentinel;
            // Use a safe, in-bounds leading dimension regardless of sign.
            var.fn(s.M, s.N, s.K, A.data(), cap, B.data(), cap, C.data(), cap);
            EXPECT_EQ(C, sentinel) << "output buffer was modified";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
