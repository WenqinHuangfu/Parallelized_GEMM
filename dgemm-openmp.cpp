#include<omp.h>
#include<stdlib.h>
#include<stdio.h>
const char *dgemm_desc = "openMP, three-loop dgemm.";
void square_dgemm( int n, double *A, double *B, double *C)
{
  //omp_set_num_threads(thread_count);
  printf("enter matrix multiplication using openmp\n");
  printf("start computation, num_threads=%i\n",omp_get_num_threads());
  #pragma omp parallel
  {
  printf("thread_num=%i\n",omp_get_thread_num());
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
