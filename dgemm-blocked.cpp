/*In case you're wondering, dgemm stands for Double-precision, GEneral Matrix-Matrix multiplication.*/

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 41
#endif

#define min(a,b) (((a)<(b))?(a):(b))

void do_block( int lda, int M, int N, int K, double *A, double *B, double *C )
{
/* To optimize this, think about loop unrolling and software
    pipelining.  Hint:  For the majority of the matmuls, you
    know exactly how many iterations there are (the block size)...  */
	for( int i = 0; i < M; i++ )
		for( int j = 0; j < N; j++ ) 
		{
			double cij = C[i+j*lda];
			for( int k = 0; k < K; k++ )
				cij += A[i+k*lda] * B[k+j*lda];
			C[i+j*lda] = cij;
       }
}

void square_dgemm( int lda, double *A, double *B, double *C)
{
	/*For each block combination*/
	for( int i = 0; i < lda; i += BLOCK_SIZE )
		for( int j = 0; j < lda; j += BLOCK_SIZE )
			for( int k = 0; k < lda; k += BLOCK_SIZE )
			{
				/*This gets the correct block size (for fringe blocks also)*/
				int M = min( BLOCK_SIZE, lda-i );
				int N = min( BLOCK_SIZE, lda-j );
				int K = min( BLOCK_SIZE, lda-k );
				
				do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
			}
}
