//
//  dgemm-pthread.cpp
//  
//
//  Created by Nan Wu on 5/4/19.
//

#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <pthread.h>
#include <mkl.h>
#define thread_count 24
const char *dgemm_desc = "pthread, three-loop dgemm.";
const double BETA = 1.0;
const char *ntran = "N";
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 40
#endif

#define min(a,b) (((a)<(b))?(a):(b))

struct param{
    int id;
    int lda;
    double *A;
    double *B;
    double *C;
};

void *runner(void *par) {
    printf("Enter working thread..\n");
    param *data = (param *) par;
    double *A = data->A;
    double *B = data->B;
    double *C = data->C;
    int lda = data->lda;
    int id = data->id;
    int avecol = lda/thread_count;
    int extra = lda%thread_count;
    int cols = (id<extra) ? avecol+1 : avecol;
    int col_start = (id<extra) ? (id*(avecol+1)) : (id-extra)*avecol+extra*(avecol+1);
    int col_end = col_start + cols;
    //printf("thread=%d created, computing from column %d to column %d\n",id,col_start,col_end);
    
    for( int j = col_start; j < col_end; j += BLOCK_SIZE )
    {
        /*This gets the correct block size (for fringe blocks also)*/
        int N = min( BLOCK_SIZE, col_end-j );
        
        double *b=(double*) malloc( lda * N * sizeof(double) );
        double *c=(double*) malloc( lda * N * sizeof(double) );
        for( int kb = 0; kb < lda; kb++ )
            for( int nb = 0; nb < N; nb++ )
            {
                b[kb+nb*lda] = B[kb+j*lda+nb*lda];
            }
        
        for( int mc = 0; mc < lda; mc++ )
            for( int nc = 0; nc < N; nc++ )
            {
                c[mc+nc*lda] = C[mc+j*lda+nc*lda];
            }
        dgemm(ntran, ntran, &lda, &N, &lda, &BETA, A, &lda, b, &lda, &BETA, c, &lda);
        for( int mc = 0; mc < lda; mc++ )
            for( int nc = 0; nc < N; nc++ )
            {
                C[mc+j*lda+nc*lda]=c[mc+nc*lda];
            }
        free(c);
        free(b);
    }
    //printf("end computation from id=%d\n",data->id);
    free(data);
    pthread_exit(0);
}
void square_dgemm( int lda, double *A, double *B, double *C)
{
    printf("Enter matrix multiplication using pthread..\n");
    int thread;
    //pthread_t *thread_handlers = (pthread_t *)malloc(thread_count*sizeof(pthread_t));
    pthread_t thread_handlers[thread_count];
    for (thread = 0; thread < thread_count; thread++){
        param *data = (param *)malloc(sizeof(param));
        data->A = A;
        data->B = B;
        data->C = C;
        data->lda = lda;
        data->id = thread;
        pthread_create(&(thread_handlers[thread]),NULL,runner,(void*)data);
        //printf("thread=%d allocated successful\n",thread);
        //pthread_join(thread_handlers[thread],NULL);
    }
    for (thread = 0; thread < thread_count; thread++){
        //printf("start destroying thread = %d\n", thread);
        pthread_join(thread_handlers[thread],NULL);
        //printf("end destroying thread = %d\n", thread);
    }
}





