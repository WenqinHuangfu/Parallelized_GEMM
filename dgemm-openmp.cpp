//
//  dgemm-omp.cpp
//  
//
//  Created by Nan Wu on 5/4/19.
//

#include<omp.h>
#include<stdlib.h>
#include<stdio.h>
#include <mkl.h>

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 40
#endif

#define min(a,b) (((a)<(b))?(a):(b))

const char *dgemm_desc = "OMP, blocked dgemm.";
const double BETA = 1.0;
const char *ntran = "N";
void square_dgemm( int lda, double *A, double *B, double *C)
{
    //omp_set_num_threads(thread_count);
    //printf("enter matrix multiplication using openmp\n");
    //printf("start computation, num_threads=%i\n",omp_get_num_threads());
#pragma omp parallel
    {
        //printf("thread_num=%i\n",omp_get_thread_num());
        int j;
#pragma omp for
        /*For each block combination*/
        for( int j = 0; j < lda; j += BLOCK_SIZE )
        {
            /*This gets the correct block size (for fringe blocks also)*/
            int N = min( BLOCK_SIZE, lda-j );
            
            //double *b=(double*) malloc( lda * N * sizeof(double) );
            double *c=(double*) malloc( lda * N * sizeof(double) );
            //for( int kb = 0; kb < lda; kb++ )
            //    for( int nb = 0; nb < N; nb++ )
            //    {
            //        b[kb+nb*lda] = B[kb+j*lda+nb*lda];
            //    }
            
            for( int mc = 0; mc < lda; mc++ )
                for( int nc = 0; nc < N; nc++ )
                {
                    c[mc+nc*lda] = C[mc+j*lda+nc*lda];
                }
            dgemm(ntran, ntran, &lda, &N, &lda, &BETA, A, &lda, B+j*lda, &lda, &BETA, c, &lda);
            for( int mc = 0; mc < lda; mc++ )
                for( int nc = 0; nc < N; nc++ )
                {
                    C[mc+j*lda+nc*lda]=c[mc+nc*lda];
                }
            free(c);
            //free(b);
        }
    }
}

