#include<omp.h>
#include<stdlib.h>
#include<stdio.h>
const char *dgemm_desc = "openMP, three-loop dgemm.";
void square_dgemm( int n, double *A, double *B, double *C)
{
  #pragma omp parallel
  {
  int i,j,k;
  #pragma omp for
  for( i = 0; i < n; i++ )
       for( j = 0; j < n; j++ )
       {
            double cij = C[i+j*n];
            for( k = 0; k < n; k++ )
                 cij += A[i+k*n] * B[k+j*n];
            C[i+j*n] = cij;
       }
  }
}
