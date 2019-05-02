#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#define MASTER 0
#define FROM_MASTER 1
#define FROM_WORKER 2

const char *dgemm_desc = "MPI, three-loop dgemm.";
MPI_Status status;
void square_dgemm( int n, double *A, double *B, double *C, int *argc, char ***argv )
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
        avecol = n/numworkers;
        extra = n%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest = 1; dest <= numworkers; dest++)
        {
            cols = (dest <= extra) ? avecol+1 :avecol;
            MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&cols, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(&B[offset*n], cols*n, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            MPI_Send(A, n*n, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
            offset += cols;
        }
        mtype = FROM_WORKER;
        for (i=1; i<=numworkers; i++)
        {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&cols, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&C[offset*n], cols*n, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        }
        
    }
    if (taskid > MASTER)
    {
        mtype = FROM_MASTER;
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&B[offset*n], cols*n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(A, n*n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        for( i = 0; i < n; i++ )
        for( j = offset; j < offset+cols; j++ )
        {
            double cij = 0;
            for( k = 0; k < n; k++ )
            cij += A[i+k*n] * B[k+j*n];
            C[i+j*n] = C[i+j*n]+cij;
        }
        mtype = FROM_WORKER;
        MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&cols, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&C[offset*n], cols*n, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
    }
}
