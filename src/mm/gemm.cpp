#include "blas2.hpp"
#include "gemm.hpp"
#include "log.hpp"

void mm_ijp (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* dst, 
        int ldC ) {

    if (M <= 0 || N <= 0 || K <= 0) return;

    MM_COUNT_RESET();
    MM_LOG(1, "variant=ijp M=" << M << " N=" << N << " K=" << K
              << " ldA=" << ldA << " ldB=" << ldB << " ldC=" << ldC);

    for(int i=0;i<M;++i) {
        gemv_row_dot (N, K, &src[i*ldA], wei, ldB, &dst[i*ldC]);
    }

    mm_detail::log_summary("mm_ijp", M, N, K);
    return;
}

void mm_ipj (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* dst, 
        int ldC ) { //fastest

    if (M <= 0 || N <= 0 || K <= 0) return;

    MM_COUNT_RESET();
    MM_LOG(1, "variant=ipj M=" << M << " N=" << N << " K=" << K
              << " ldA=" << ldA << " ldB=" << ldB << " ldC=" << ldC);

    for(int i=0;i<M;++i) {
        gemv_row_axpy (N, K, &src[i*ldA], wei, ldB, &dst[i*ldC]);
    }

    mm_detail::log_summary("mm_ipj", M, N, K);
    return;
}

void mm_jpi (
        int M, int N, int K,
        const float* src,
        int ldA, 
        const float* wei, 
        int ldB, 
        float* dst, 
        int ldC) {  //slowest

    if (M <= 0 || N <= 0 || K <= 0) return;

    MM_COUNT_RESET();
    MM_LOG(1, "variant=jpi M=" << M << " N=" << N << " K=" << K
              << " ldA=" << ldA << " ldB=" << ldB << " ldC=" << ldC);

    for(int j=0;j<N;++j) {
        gemv_col_axpy (M, K, src, ldA, &wei[j], ldB, &dst[j], ldC);
    }

    mm_detail::log_summary("mm_jpi", M, N, K);
    return;
}

void mm_jip (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* dst, 
        int ldC ) {

    if (M <= 0 || N <= 0 || K <= 0) return;

    MM_COUNT_RESET();
    MM_LOG(1, "variant=jip M=" << M << " N=" << N << " K=" << K
              << " ldA=" << ldA << " ldB=" << ldB << " ldC=" << ldC);

    for(int j=0;j<N;++j) {
        gemv_col_dot (M, K, src, ldA, &wei[j], ldB, &dst[j], ldC);
    }

    mm_detail::log_summary("mm_jip", M, N, K);
    return;
}

void mm_pij (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* dst, 
        int ldC ) { //fastest 

    if (M <= 0 || N <= 0 || K <= 0) return;

    MM_COUNT_RESET();
    MM_LOG(1, "variant=pij M=" << M << " N=" << N << " K=" << K
              << " ldA=" << ldA << " ldB=" << ldB << " ldC=" << ldC);

    for(int p=0;p<K;++p) {
        ger_row (M, N, &src[p], ldA, &wei[p*ldB], dst, ldC);
    }

    mm_detail::log_summary("mm_pij", M, N, K);
    return;
}

void mm_pji (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* dst, 
        int ldC ) { //slowest
    
    if (M <= 0 || N <= 0 || K <= 0) return;

    MM_COUNT_RESET();
    MM_LOG(1, "variant=pji M=" << M << " N=" << N << " K=" << K
              << " ldA=" << ldA << " ldB=" << ldB << " ldC=" << ldC);

    for(int p=0;p<K;++p) {
        ger_col (M, N, &src[p], ldA, &wei[p*ldB], dst, ldC);
    }

    mm_detail::log_summary("mm_pji", M, N, K);
    return;
}


void mm_ref (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* dst, 
        int ldC ) { //reference              

    if (M <= 0 || N <= 0 || K <= 0) return;

    for(int i=0;i<M;++i) {
        for(int j=0;j<N;++j) {
            for(int p=0;p<K;++p) {
                dst[i*ldC+j] += (src[i*ldA + p] * wei[p*ldB + j]);
            }
        }
    }

    return;
} 
