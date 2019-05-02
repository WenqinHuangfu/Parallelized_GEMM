//#include <acml.h> //assumes AMD platform
#include <mkl.h>

const char* dgemm_desc = "BLAS dgemm.";
void square_dgemm( int n, double *A, double *B, double *C)
{
    //dgemm( 'N','N', n,n,n, 1, A,n, B,n, 1, C,n );
    const double BETA = 1.0;
    const double negOne = -1.0;
    char *ntran = "N";
    dgemm(ntran, ntran, &n, &n, &n, &BETA, A, &n, B, &n, &BETA, C, &n);
}
