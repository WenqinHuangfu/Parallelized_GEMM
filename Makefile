#Comet
#MKLPATH=/opt/intel/Compiler/11.1/072/mkl
MKLPATH=$MKL_ROOT
CC=icpc
MPICC=mpicc
CFLAGS = -I$(MKLPATH)/include
LDFLAGS = -mkl -lpthread -lm
OMPFLAGS = -openmp

all: benchmark-blocked benchmark-blas benchmark-naive benchmark-openmp benchmark-mpi benchmark-pthread

benchmark-blocked: benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS)
benchmark-blas: benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDFLAGS)
benchmark-naive: benchmark.o dgemm-naive.o
	$(CC) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)
benchmark-pthread: benchmark.o dgemm-pthread.o
	$(CC) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)
benchmark-openmp: benchmark.o dgemm-openmp.o
	$(CC) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)
benchmark-mpi: test.o dgemm-mpi.o
	$(MPICC) -o $@ $^ $(LDFLAGS)

benchmark.o: benchmark.cpp
	$(CC) $(OMPFLAGS) $(CFLAGS) -c -g -Wall $<
dgemm-blocked.o:dgemm-blocked.cpp
	$(CC) -c -g -Wall $<
dgemm-blas.o:dgemm-blas.cpp
	$(CC) -c -g -Wall $<
dgemm-naive.o:dgemm-naive.cpp
	$(CC) -c -g -Wall $<
dgemm-pthread.o:dgemm-pthread.cpp
	$(CC) -c -g -Wall $<
dgemm-openmp.o:dgemm-openmp.cpp
	$(CC) $(OMPFLAGS) -c -g -Wall $<
test.o: test.cpp
	$(MPICC) -c -g $<
dgemm-mpi.o: dgemm-mpi.cpp
	$(MPICC) -c -g $<

clean:
	rm -f benchmark-naive benchmark-blocked benchmark-blas benchmark-mpi benchmark-pthread benchmark-openmp *.o

