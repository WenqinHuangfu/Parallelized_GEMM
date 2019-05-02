#include <stdlib.h>
#include <sys/types.h>
#include <stdio.h>
#include <pthread.h>
#define thread_count 24
const char *dgemm_desc = "pthread, three-loop dgemm.";
struct param{
	int id;
	int n;
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
	int n = data->n;
	int id = data->id;
	int avecol = n/thread_count;
	int extra = n%thread_count;
	int cols = (id<extra) ? avecol+1 : avecol;
	int col_start = (id<extra) ? (id*(avecol+1)) : (id-extra)*avecol+extra*(avecol+1);
	int col_end = col_start + cols;
	printf("thread=%d created, computing from column %d to column %d\n",id,col_start,col_end);
	for (int i = 0; i < n; i++){
		for (int j = col_start; j < col_end; j++){
			C[i+j*n] = 0;
			for(int k = 0; k < n; k++ ){
				C[i+j*n] += A[i+k*n] * B[k+j*n];
			}
		}
	}
	printf("end computation from id=%d\n",data->id);
	free(data);
	pthread_exit(0);
}
void square_dgemm( int n, double *A, double *B, double *C)
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
	data->n = n;
	data->id = thread;
	pthread_create(&(thread_handlers[thread]),NULL,runner,(void*)data);
	printf("thread=%d allocated successful\n",thread);
	//pthread_join(thread_handlers[thread],NULL);
}
  for (thread = 0; thread < thread_count; thread++){
	printf("start destroying thread = %d\n", thread);
	pthread_join(thread_handlers[thread],NULL);
	printf("end destroying thread = %d\n", thread);
  }
}




