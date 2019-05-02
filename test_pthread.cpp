#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 24

struct threadData {
    int id;
    int n;
    double *A;
    double *B;
    double *C;
};

void *thread_mm( void * td ) {
  threadData * tD = (threadData*) td;
  int n = tD->n;
  int id = tD->id;
  double *A = tD->A;
  double *B = tD->B;
  double *C = tD->C;
  int rowsPerThread = n/NUM_THREADS;
  int startRow = id * rowsPerThread;
  int endRow = startRow + rowsPerThread;
  if (id + 1 == NUM_THREADS)
    endRow = n;
  for( int i = startRow; i < endRow ; i++ )
       for( int j = 0; j < n; j++ )
       {
            double cij = 0.0;
            for( int k = 0; k < n; k++ )
                 cij += A[i+k*n] * B[k+j*n];
            C[i+j*n] = cij;
       }
    pthread_exit(0);
}

const char *dgemm_desc = "Naive, three-loop dgemm with pthreads";
void square_dgemm( int n, double *A, double *B, double *C , int invalid )
{

    pthread_t p[NUM_THREADS];
    int t = 0;

    for (t=0;t<NUM_THREADS;t++) {

        threadData * tD = (threadData*)malloc(sizeof(threadData));
        tD->id = t;
        tD->n = n;
        tD->A = A;
        tD->B = B;
        tD->C = C;
        pthread_create(&p[t],NULL,thread_mm, (void *) tD);
    }
    for (t=0;t<NUM_THREADS;t++) {
        pthread_join(p[t],NULL);
    }

}
