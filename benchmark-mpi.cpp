#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <float.h>
#include <math.h>

#include <sys/time.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

double read_timer( )
{
    static bool initialized = false;
    static struct timeval start;
    struct timeval end;
    if( !initialized )
    {
        gettimeofday( &start, NULL );
        initialized = true;
    }

    gettimeofday( &end, NULL );

    return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

void fill( double *p, int n )
{
    for( int i = 0; i < n; i++ )
        p[i] = 2 * drand48( ) - 1;
}

void absolute_value( double *p, int n )
{
    for( int i = 0; i < n; i++ )
        p[i] = fabs( p[i] );
}

MPI_Status status;

int main( int argc, char **argv )
{
    /* These sizes should highlight performance dips at multiples of certain powers-of-two */
    /*int test_sizes[] = {
        31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
        319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769,
    };*/
    int numprocs, numworkers, myid, begin, cols;
    const int n = 1600;
    double *A = (double*) malloc( n * n * sizeof(double) );
    double *B = (double*) malloc( n * n * sizeof(double) );
    double *C = (double*) malloc( n * n * sizeof(double) );
    //memset( C, 0, sizeof( double ) * n * n );
    //int test_sizes[] = {100, 200, 400, 800, 1600};
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &numprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &myid );
    numworkers = numprocs - 1;

    if (myid == 0){
        fill( A, n * n );
        fill( B, n * n );
        printf("using %d processes", numprocs);

        double seconds = read_timer( );
        begin = 0;
        if(n % numworkers == 0)
            cols = n / numworkers;
        else
            cols = n / numworkers + 1;
        for (int worker=1; worker <= numworkers; worker++){
            MPI_Send(&begin, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            int real_cols = 0;
            if(begin + cols > n)
                real_cols = n - begin;
            else
                real_cols = cols;
            MPI_Send(&real_cols, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
            MPI_Send(&B[begin * n], real_cols * n, MPI_DOUBLE, worker, 1, MPI_COMM_WORLD);
            MPI_Send(&A[0], n*n, MPI_DOUBLE, worker, 1, MPI_COMM_WORLD);
            begin += real_cols;
        }

        begin = 0;
        for (int worker=1; worker <= numworkers; worker++){
            MPI_Recv(&begin, 1, MPI_INT, worker, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols, 1, MPI_INT, worker, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[begin * n], cols * n, MPI_DOUBLE, worker, 2, MPI_COMM_WORLD, &status);
            begin += cols;
        }
        seconds = read_timer( ) - seconds;
        double Mflop_s = 2e-6 * n * n * n / seconds;
        printf ("Size: %d\tTime: %f\tMflop/s: %g\n", n, seconds, Mflop_s);    
    	double beta_num = -1;
    	double gamma_num = 1;
    	const double* beta = &beta_num;
    	const double* gamma = &gamma_num;
        const int * pointer_n = &n;
        dgemm( "N","N", pointer_n, pointer_n, pointer_n, beta, A, pointer_n, B, pointer_n, gamma, C, pointer_n);
    	/*Subtract the maximum allowed roundoff from each element of C*/
        absolute_value( A, n * n );
        absolute_value( B, n * n );
        absolute_value( C, n * n );
    	beta_num = -3.0*DBL_EPSILON*n;
    	beta = &beta_num;
        dgemm( "N","N", pointer_n, pointer_n, pointer_n, beta, A, pointer_n, B, pointer_n, gamma, C, pointer_n);
    	/*After this test if any element in C is still positive something went wrong in square_dgemm*/
        for( int i = 0; i < n * n; i++ )
        {
            if( C[i] > 0 )
            {
                printf( "FAILURE: error in matrix multiply exceeds an acceptable margin\n" );
                exit(-1);
            }
        }
    }

    if (myid > 0){
        MPI_Recv(&begin, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&cols, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[0], cols * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&A[0], n * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);

        for( int i = 0; i < n; i++ ){
            for( int j = 0; j < cols; j++ ){
                C[i+j*n] = 0.0;
                for( int k = 0; k < n; k++ ){
                    C[i+j*n] += A[i+k*n] * B[k+j*n];
                }
            }
        }

        MPI_Send(&begin, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&cols, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&C[0], cols * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    
    return 0;
}
