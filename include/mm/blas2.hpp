#ifndef blas2_hpp
#define blas2_hpp

// GEMV producing one ROW of C:   c_row[0..N) += a_row[0..K) * B(K x N)
void gemv_row_dot (
        int N, 
        int K, 
        const float* a_row,
        const float* B, 
        int ldB, 
        float* c_row
);

void gemv_row_axpy (
        int N, 
        int K, 
        const float* a_row,
        const float* B, 
        int ldB, 
        float* c_row
);

// GEMV producing one COLUMN of C: c_col[0..M) += A(M x K) * b_col[0..K)
void gemv_col_dot (
        int M, 
        int K, 
        const float* A, 
        int ldA,
        const float* b_col, 
        int incb, 
        float* c_col, 
        int incc
);

void gemv_col_axpy(
        int M, int K, 
        const float* A, 
        int ldA,
        const float* b_col, 
        int incb, 
        float* c_col, 
        int incc
);

// rank-1 update: C += a_col[0..M) (outer) b_row[0..N)
void ger_row (
        int M, 
        int N, 
        const float* a_col, 
        int inca,
        const float* b_row, 
        float* C, 
        int ldC);  // axpy across rows

void ger_col (
        int M, 
        int N, 
        const float* a_col, 
        int inca,
        const float* b_row, 
        float* C, 
        int ldC
);  // axpy down cols

#endif
