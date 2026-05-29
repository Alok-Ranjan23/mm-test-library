#ifndef gemm_hpp
#define gemm_hpp

// Row-major GEMM:  C += A * B
//   A is M x K   (ldA = K for tightly packed)
//   B is K x N   (ldB = N)
//   C is M x N   (ldC = N)

void mm_ijp (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* C, 
        int ldC
);

void mm_ipj (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* C, 
        int ldC
); //fastest

void mm_jpi (
        int M, int N, int K,
        const float* src,
        int ldA, 
        const float* wei, 
        int ldB, 
        float* C, 
        int ldC
); //slowest

void mm_jip (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* C, 
        int ldC
);

void mm_pij (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* C, 
        int ldC
); //fastest 
  
void mm_pji (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* C, 
        int ldC
); //slowest

void mm_ref (
        int M, int N, int K,
        const float* src, 
        int ldA, 
        const float* wei, 
        int ldB, 
        float* C, 
        int ldC
); //reference               

/*
 * typedef enum {
 * MM_IJP, MM_IPJ, MM_JIP, MM_JPI, MM_PIJ, MM_PJI, MM_COUNT
 * } mm_variant_t;
 *
 * const char* mm_name(mm_variant_t v);
 * void mm_dispatch(
 *                  mm_variant_t v,
 *                  int M, int N, int K,
 *                  const float* src, int ldA,
 *                  const float* wei, int ldB,
 *                  float* C,int ldC); 
*/

#endif
