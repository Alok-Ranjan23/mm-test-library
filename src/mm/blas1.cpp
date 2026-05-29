#include "blas1.hpp"
#include "log.hpp"

void dot (
        int n,
        const float* x,
        int incx,
        const float* y,
        int incy,
        float* gamma ) {

    MM_COUNT(dot);
    MM_LOG(3, "dot n=" << n << " incx=" << incx << " incy=" << incy);

    float sum = 0.0f;
    for(int i=0;i<n;++i) {
        sum += (x[(i)*incx]* y[(i)*incy]) ;
    }
    *gamma += sum;

    return;
}

void axpy (
        int n,
        float alpha,
        const float* x,
        int incx,
        float* y,
        int incy ) {

    MM_COUNT(axpy);
    MM_LOG(3, "axpy n=" << n << " alpha=" << alpha);

    for(int j=0;j<n;++j) {
        y[(j)*incy] = (alpha * x[(j)*incx] + y[(j)*incy]) ;  
    }

    return;
} 
