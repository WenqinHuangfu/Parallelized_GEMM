#Comet
#MKLPATH=/opt/intel/Compiler/11.1/072/mkl
MKLPATH=$MKL_ROOT
MPICC = mpicc
CC=icpc
GCC = gcc
CFLAGS = -I$(MKLPATH)/include  

MKLFLAGS=  -I$MKL_ROOT/include ${MKL_ROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKL_ROOT}/lib/intel64/libmkl_intel_lp64.a ${MKL_ROOT}/lib/intel64/libmkl_core.a ${MKL_ROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group ${MKL_ROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a 

LDFLAGS = $(MKLFLAGS) -lpthread -lm 
LDFLAGS1 = -std=gnu99 -O2 DNDEBUG -g0 -msse4.2 -masm=intel -lpthread -lm
OMPFLAGS = -openmp

all:	benchmark-naive benchmark-blocked benchmark-blas 

benchmark-naive: benchmark.o dgemm-naive.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-blocked: benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-blas: benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDFLAGS)

benchmark-omp: benchmark.o dgemm-omp.o
	$(CC) $(OMPFLAGS) -o $@ $^ $(LDFLAGS)

benchmark-mpi: benchmark-mpi.o
	$(MPICC) -o $@ $^ $(LDFLAGS)

benchmark-pthread: benchmark-pthread.o dgemm-pthread.o
	$(CC) -o $@ $^ -O3 $(LDFLAGS)

dgemm-omp.o: dgemm-omp.cpp
	$(CC) $(OMPFLAGS) -c $<

benchmark-mpi.o: benchmark-mpi.cpp
	$(MPICC) -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<

run-benchmark:
	sbatch -v run-benchmark

run-benchmark-mpi:
	sbatch -v run-benchmark-mpi4-comet

run-benchmark-omp4:
	sbatch -v run-benchmark-omp4

run-benchmark-pth:
	sbatch -v run-benchmark-pth

dgemm-pthread.o: dgemm-pthread.cpp
	$(GCC) -c -O3 -msse4.2 -masm=intel -lpthread -lm $<

status:
	squeue -u `whoami`


clean:
	rm -f benchmark-naive benchmark-blocked benchmark-blas benchmark-omp benchmark-mpi benchmark-pthread *.o
