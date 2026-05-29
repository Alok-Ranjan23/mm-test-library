#include "blas1.hpp"
#include "blas2.hpp"
#include "log.hpp"

// GEMV producing one ROW of C:   c_row[0..N) += a_row[0..K) * B(K x N)
// JP
void gemv_row_dot (
        int N, 
        int K, 
        const float* a_row,
        const float* B, 
        int ldB, 
        float* c_row ) {

    MM_COUNT(gemv_row_dot);
    MM_LOG(2, "gemv_row_dot N=" << N << " K=" << K << " ldB=" << ldB);

    for (int j=0;j<N;++j) {
        dot(K, a_row, 1, B + j, ldB, c_row + j);
    }

    return;
}

//PJ
void gemv_row_axpy (
        int N, 
        int K, 
        const float* a_row,
        const float* B, 
        int ldB, 
        float* c_row ) {

    MM_COUNT(gemv_row_axpy);
    MM_LOG(2, "gemv_row_axpy N=" << N << " K=" << K << " ldB=" << ldB);

    for(int p=0;p<K;++p) {
        axpy(N, a_row[p], B + p*ldB, 1, c_row, 1);
    }
   
    return;
}

// GEMV producing one COLUMN of C: c_col[0..M) += A(M x K) * b_col[0..K)
// IP
void gemv_col_dot (
        int M, 
        int K, 
        const float* A, 
        int ldA,
        const float* b_col, 
        int incb, 
        float* c_col, 
        int incc ) {

    MM_COUNT(gemv_col_dot);
    MM_LOG(2, "gemv_col_dot M=" << M << " K=" << K << " ldA=" << ldA);

    for (int i=0;i<M;++i) {
        dot(K, A + i*ldA, 1, b_col, incb, c_col + i*incc);
    }
    
    return;
}

//PI
void gemv_col_axpy(
        int M, int K, 
        const float* A, 
        int ldA,
        const float* b_col, 
        int incb, 
        float* c_col, 
        int incc) {

    MM_COUNT(gemv_col_axpy);
    MM_LOG(2, "gemv_col_axpy M=" << M << " K=" << K << " ldA=" << ldA);

    for(int p=0;p<K;++p) {
        axpy(M, b_col[p*incb], A+p, ldA, c_col, incc);
    }

    return;
}

// rank-1 update: C += a_col[0..M) (outer) b_row[0..N)
// IJ
void ger_row (
        int M, 
        int N, 
        const float* a_col, 
        int inca,
        const float* b_row, 
        float* C, 
        int ldC ) {  // axpy across rows

    MM_COUNT(ger_row);
    MM_LOG(2, "ger_row M=" << M << " N=" << N << " ldC=" << ldC);

    for(int i=0;i<M;++i) {
        axpy(N, a_col[i*inca], b_row, 1, C + i*ldC, 1);
    }

    return;
}

//JI
void ger_col (
        int M, 
        int N, 
        const float* a_col, 
        int inca,
        const float* b_row, 
        float* C, 
        int ldC ) {  // axpy down cols

    MM_COUNT(ger_col);
    MM_LOG(2, "ger_col M=" << M << " N=" << N << " ldC=" << ldC);

    for(int j=0;j<N;++j) {
        axpy(M, b_row[j] , a_col, inca, C + j, ldC);
    }

    return;
}
