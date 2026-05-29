#include <iostream>
#include <vector>

#include "gemm.hpp"

static void print_matrix(const char* name, const float* m, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) std::cout << "  " << m[i * cols + j];
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // Row-major:  C(MxN) += A(MxK) * B(KxN)
    const int M = 2, K = 3, N = 2;
    const int ldA = K, ldB = N, ldC = N;

    // Easy to verify by hand:   A * B = [[5, 12],
    //                                    [0,  5]]
    std::vector<float> A = {1, -2, 3,
                            -1, 0, 2};     // 2x3
    std::vector<float> B = {2,  1,
                            0, -1,
                            1,  3};        // 3x2
    std::vector<float> C(M * N, 0.0f);     // 2x2, zeroed (kernels accumulate)

    mm_ijp(M, N, K, A.data(), ldA, B.data(), ldB, C.data(), ldC);

    std::cout << "Example: mm_ijp   (C = A * B)\n\n";
    print_matrix("A", A.data(), M, K);
    print_matrix("B", B.data(), K, N);
    print_matrix("C", C.data(), M, N);
    return 0;
}
