#ifndef blas1_hpp
#define blas1_hpp

void dot (
        int n,
        const float* x,
        int incx,
        const float* y,
        int incy,
        float* gamma
);

void axpy (
        int n,
        float alpha,
        const float* x,
        int incx,
        float* y,
        int incy
); 

#endif
