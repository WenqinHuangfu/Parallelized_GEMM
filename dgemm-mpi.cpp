//
//  dgemm-mpi.cpp
//  
//
//  Created by Nan Wu on 5/5/19.
//
#include <mkl.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 40
#endif

#define min(a,b) (((a)<(b))?(a):(b))


const char *dgemm_desc = "MPI, three-loop dgemm.";
const double BETA = 1.0;
const char *ntran = "N";
MPI_Status status;

void square_dgemm( int lda, double *A, double *B, double *C, int *argc, char ***argv )
{
    printf("enter\n");
    int taskid,numtasks,numworkers,source,dest,mtype;
    int cols,avecol,extra,offset;
    int i,j,k;
    MPI_Init(argc,argv);
    if ( MPI_Comm_rank(MPI_COMM_WORLD,&taskid) != MPI_SUCCESS )
    {
        printf("Error MPI_RANK\n");
        return;
    }
    if ( MPI_Comm_size(MPI_COMM_WORLD,&numtasks) != MPI_SUCCESS )
    {
        printf("Error MPI_SIZE\n");
        return;
    }
    numworkers = numtasks - 1;
    //master
    if (taskid == MASTER){
        printf("master started.. num_tasks= %d \n", numtasks);
        avecol = lda/numworkers;
        extra = lda%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworkers; dest++)
        {
            cols = (dest <= extra) ? avecol+1 :avecol;
            printf("sending %d columns to task %d offset = %d\n",cols, dest, offset);
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&cols, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&B[offset*lda], cols*lda, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(A, lda*lda, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset += cols;
            printf("sending successful to task %d\n",dest);
        }
        mtype = FROM_WORKER;
        for (i=1; i<=numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset*lda], cols*lda, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
            printf("receiving results from task %d\n",source);
        }
        
    }
    if (taskid > MASTER)
    {
        mtype = FROM_MASTER;
        printf("workerid = %d starts receiving job\n",taskid);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[offset*lda], cols*lda, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(A, lda*lda, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        printf("workerid = %d receiving job successful!\n",taskid);
        
        for( int j = offset; j < offset+cols; j += BLOCK_SIZE )
        {
            /*This gets the correct block size (for fringe blocks also)*/
            int N = min( BLOCK_SIZE, offset+cols-j );
            
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
        
        mtype = FROM_WORKER;
        printf("workerid = %d starts sending finished job\n",taskid);
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&C[offset*lda], cols*lda, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        printf("workerid = %d sending finishes\n",taskid);
    }
}
