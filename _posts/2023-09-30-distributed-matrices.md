---
layout:	post
title:	"Distributed matrices using MPI"
date:	2023-09-30 02:05:00 +0300
categories:	Category
---
{% include_relative _includes/head-custom.html %}

## Introduction

Parallelized sparse and dense matrix algorithms serve as an important basis for many scientific computing applications. While OpenMP and other parallel programming interfaces can be used on shared memory devices (e.g. laptop, cluster node), for larger problems, a single machine can seldom provide enough memory or enough CPUs to parallelize over. To write concurrent programs that use several machines, devices or cluster nodes (what supercomputers essentially are), one could use a Message Passing Interface (MPI). The key difference between MPI and OpenMP is that OpenMP threads can access the same memory, where each MPI process lives on a separate machine and cannot directly access others' memory. To communicate, we need to send chunks of data back and forth between physical machines. The speed at which this happens is very slow compared to a CPU speed, and thus, one cannot send too much over a network. Understanding this bottleneck is important for writing better MPI programs.

In this post, we will implement distributed (parallel) dense matrices in C using MPI. The C language has simple syntax and is explicit about memory, therefore it is our choice. We note that although MPI is mainly used for applications run on supercomputers, we can write and test them on shared memory devices too. That is, we can use it to run parallel programs on our laptops. Sometimes applications choose to use MPI instead of OpenMP for their shared memory implementations. For the exercise, we can use both MPICH or Open MPI (do not confuse with OpenMP) implementations of the MPI. If you do not have, you can always `sudo apt install` them. This will let you use MPI compilers `mpicc` instead of usual `gcc` to compile C source codes into a parallel program, e.g.: `mpicc main.c -o main`. Then we can run it by specifying the number of processes as `mpiexec -n 2 ./main`.

Let us create a new project folder `distributed_matrix`, then add 3 files: `main.c`, `matrix.h` and `matrix.c`. It is better to organize the code in modules. In C, we achieve this by creating separate header files `.h` and corresponding source files `.c` for the functionality we intend to implement. In this case, for a distributed matrix. This also provides some basic encapsulation and is a good exercise for writing APIs and libraries. The structure must look as follows:

```
.
├── main.c
├── matrix.c
└── matrix.h
```

Now let us write a simple program which prints the process rank, a unique number every MPI process gets at the initialization. Go into `main.c` file and add the following lines:

```c
#include <stdio.h>
#include <openmpi/mpi.h>

int main(int argc, char **argv){
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    printf("The process %i says hello world!\n", rank);

    MPI_Finalize();
    return 0;
}
```

Now compile the code with

```
mpicc main.c -o main
```

and run via

```
mpiexec -n 2 ./main
```

to get something like (order might be different)

```
The process 0 says hello world!
The process 1 says hello world!
```

We see that our program just duplicates the code using two processes, with the `rank` variable having different values, 0 and 1. This is essentially how all MPI programs work, they execute the same code with some variables set to defferent values. Depending on the value of `rank` and using `if` statements, for example, we can force processes to work on different parts of a matrix. This is how the parallelization is achieved.

## Distributed matrix

Now we will implement a parallel matrix. For this exercise, we will pretend that MPI processes reside on different machines (although they do not), and each will store only one part of the matrix. First, there are several ways to split matrices. A common way is to assign row blocks to different MPI processes. Let $$A$$ be a $$4\times4$$ matrix. We will split its rows evenly as follows:

$$
A = \left[
\begin{array}{cccc}
	a_{0,0} & a_{0,1} & a_{0,2} & a_{0,3} \\
	a_{1,0} & a_{1,1} & a_{1,2} & a_{1,3} \\
	\hline
	a_{2,0} & a_{2,1} & a_{2,2} & a_{2,3} \\
	a_{3,0} & a_{3,1} & a_{3,2} & a_{3,3} \\
\end{array}
\right]
$$

The rank 0 process gets rows 0 and 1, while the rank 1 process gets to store rows 2 and 3, the horizontal line indicates that the rows belong to different processes. Another apporach would be to split according to column indices, in which case the line would be vertical. We will stick to the first approach. Note that the submatrices are of size $$2\times4$$, that is, despite owning only 2 rows, they own all the columns. Later, this would be beneficial for performing matrix-vector products.

Go into the file `matrix.h` and add 

```c
typedef struct _Matrix *Matrix;
```

Here we do two things. First, we define a new `struct` with a pretty name `_Matrix`, where we will store all our matrix data. This structure is incomplete, and the exact contents will be given in `matrix.c` source file. However, when the user (programmer writing the `main.c` file) includes the header file `matrix.h`, one will only see the incomplete matrix data type. The compiler and the user do not know the true size of this data type and cannot allocate memory for it. Therefore, secondly, we will give the user a handle - a pointer `*Matrix` to our matrix structure. That is, `Matrix` is an alias for `*_Matrix`, a pointer to `_Matrix`. Since pointers are not the data itself, but are just 64 bit addresses, the compiler does not have a problem with allocating extra 64 bits of memory. All of this is done to abstract the user from the internals and implementation details, and provide just enough knowledge about what matrix is. For more, google opaque data types.

Thanks to our incomplete definition, we can use our new data type to forward declare the functionality we want from our parallel matrices. For instance, we want to be able to allocate memory for our matrix, to free it when needed, to set entries of the matrix, and possibly to print the matrix. These functions can be defined in `matrix.h` file, and will be the interface the users will use to interact with matrices.

```c
void MatrixCreate(MPI_Comm comm, int N, int M, Matrix* A);
void MatrixDestroy(Matrix* A);
void MatrixSetValue(Matrix A, int i, int j, double val);
void MatrixView(Matrix A);
```

The implementations will follow in `matrix.c` source file. We start a file by defining a structure that will hold all the necessary data for our parallel matrix. This structure is internal and is not seen from outside of `matrix.c` file.

```c
#include "matrix.h"
#include <openmpi/mpi.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct _Matrix {
    MPI_Comm comm;
    int rank, size;
    int n, m, N, M;
    int *isize, *istart, *iend;
    double* val;
} _Matrix;
```

Here, `comm` is an MPI communicator, we will need it for communication between processes for various matrix operations. The `rank` and `size` hold the process rank and overall number of processes. Integers `N` and `M` are global matrix dimensions, 4 and 4 for the matrix $$A$$. Whereas `n` and `m` are local matrix sizes, which are 2 and 4 (`m = M` in this exercise). When we run our program, each process will have to allocate its own memory, and local sizes might differ for each rank. One would often need additional information about other ranks. This info can be stored in arrays `isize`, `istart` and `iend`, each of size `size`. These are duplicated on each process, and contain the number of owned rows, the starting row index, the ending row index (exclusive), correspondingly. Lastly, we need an array `val` for storing local matrix entries.

## MatrixCreate

Let's first implement the `MatrixCreate` function. The full code of the function will be given later. The first thing we need to do is to understand our place in the world. We use `MPI_Comm_rank` and `MPI_Comm_size` functions to get the rank and overall size of the communicator. 

```c
int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);
```

Next, we allocate the memory for our data using `malloc`, as the argument, we give size of `_Matrix` structure. We also allocate memory for our arrays `isize`, `istart` and `iend` using `calloc`. This initializes arrays to 0, other fields related to the communicator are initalized as well. Note that `A` is a pointer to `Matrix`, which is itself a pointer to `_Matrix`, i.e. `A` is of type `Matrix**`. To access our fields, we do something like `(**A).comm` or `(*A)->comm`.

```c
(*A) = malloc(sizeof(_Matrix));
(*A)->comm = comm;
(*A)->rank = rank;
(*A)->size = size;
(*A)->isize  = calloc(size, sizeof(int));
(*A)->istart = calloc(size, sizeof(int));
(*A)->iend   = calloc(size, sizeof(int));
```

Now we have to decide how we split our matrix. The number of local and global columns is equal to `M` in our implementation. So, given global row dimensions `N`, we have to calculate how many will be local `n` rows. Since we are working with integers, each process should get at least `N / size` rows. E.g, for `N = 7` and `size = 3`, each rank gets `7 / 3 = 2` rows at minimum. Further, we must somehow distribute the remaining rows, of which, there are `N % size` or `7 % 3 = 1`. We can make the following rule: if the remainder is `r`, the first `r` ranks will get one additional row. This can nicely be written as `rank < N % size`, which is 1 for the first `r` ranks and 0 for the rest. To get the intuition, you can play this game on a paper using other numbers.

```c
(*A)->n = N / size + (rank < N % size);
(*A)->m = M;
(*A)->N = N;
(*A)->M = M;
```

The row split is fully deterministic and does not require any communication. That means we can generate this information on every process, so that everyone knows who owns how many rows, and the ownership ranges. Here, the `isize` array stores the number of rows each rank owns. The arrays `istart` and `iend` store the start and end indices of the local submatrix using the global ordering. For instance, for two ranks and a $$4\times4$$ example we provided: `isize = [2, 2]`, `istart = [0, 2]` and `iend = [2, 4]`.

```c
(*A)->isize[0] = N / size + (0 < N % size);
(*A)->iend[0] = (*A)->isize[0];
for(int k=1; k<size; k++){
    (*A)->isize[k]   = N / size + (k < N % size);
    (*A)->istart[k]  = (*A)->istart[k-1]  + (*A)->isize[k-1];
    (*A)->iend[k]    = (*A)->iend[k-1]    + (*A)->isize[k];
}
```

Finally, we know how much memory must be allocated for the local portion of the matrix. On each process, the number of local rows is `n` and the number of columns is `M`. Therefore, we allocate `n * M` doubles to store matrix entries.


```c
(*A)->val = calloc((*A)->n * M, sizeof(double));
```

The full implementation is:

```c
void MatrixCreate(MPI_Comm comm, int N, int M, Matrix* A){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    (*A) = malloc(sizeof(_Matrix));
    (*A)->comm = comm;
    (*A)->rank = rank;
    (*A)->size = size;
    (*A)->isize  = calloc(size, sizeof(int));
    (*A)->istart = calloc(size, sizeof(int));
    (*A)->iend   = calloc(size, sizeof(int));

    (*A)->n = N / size + (rank < N % size);
    (*A)->m = M;
    (*A)->N = N;
    (*A)->M = M;

    (*A)->isize[0] = N / size + (0 < N % size);
    (*A)->iend[0] = (*A)->isize[0];
    for(int k=1; k<size; k++){
        (*A)->isize[k]   = N / size + (k < N % size);
        (*A)->istart[k]  = (*A)->istart[k-1]  + (*A)->isize[k-1];
        (*A)->iend[k]    = (*A)->iend[k-1]    + (*A)->isize[k];
    }

    (*A)->val = calloc((*A)->n * M, sizeof(double));
}
```

## MatrixDestroy

The memory allocated dynamically using `malloc` or `calloc` cannot be reclaimed by the operating system, unless we `free` it or the program terminates. If we provide a function for creating matrices, we must provide a function for destroying them too. This will help the user to get better control of one's memory consumption. The following code deallocates the contents of a matrix in reverse order: first it frees arrays `val`, `isize`, etc., and only after that the structure itself. If we `free(*A)` first, `(*A)->val` and friends now point to nowhere and we cannot deallocate them. Since the arrays are still unreclaimed, this would be a memory leak.

```c
void MatrixDestroy(Matrix* A){
    free((*A)->isize);
    free((*A)->istart);
    free((*A)->iend);
    free((*A)->val);
    free((*A));
}
```

## MatrixSetValue

Navigating C arrays to set individual matrix entries might be too tiresome for the user. Especially when the matrix is distributed and its indices do not correspond to those of local arrays. For example, if we want to insert a value in the position `(0,0)` in matrix $$A$$, only the owner (rank 0) must do so. Luckily, we can access the ownership information we saved in `istart` and `iend` arrays. If the index `i` is in the ownership range, we can insert the value on `i - A->istart[A->rank]`th row of the local matrix. For our matrix $$A$$, the offsets are equal to 0 and 2 correspondingly.

```c
void MatrixSetValue(Matrix A, int i, int j, double val){
    if(i >= A->istart[A->rank] && i < A->iend[A->rank]){
        A->val[(i - A->istart[A->rank]) * A->M + j] = val;
    }
}
```

## MatrixView

At last, we implement a function to print our matrix. Input and output in an MPI program is quite tricky. The reason is that all MPI processes run the same code simultaneously, and if we want to naively print array contents, the output would be a mess. A slightly more successful approach is to force processes to wait for each other. This can be accomplished with `MPI_Barrier` function.


```c
void MatrixView(Matrix A){
    if(!A->rank) printf("Matrix of size (%i, %i) on %i processes:\n", A->N, A->M, A->size);
    for(int k=0; k<A->size; k++){
        if(A->rank == k){
            printf("[%i]\n", A->rank);
            for(int i=0; i<A->n; i++){
                for(int j=0; j<A->M; j++){
                    printf("%f\t", A->val[i*A->M+j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(A->comm);
    }
}
```

The code above runs a `k` loop from 0 to `size` on each process. Inside, we check if the `rank` is equal to current `k`. If yes, we print the local submatrix. If not, we do nothing. The `MPI_Barrier` at the end makes sure that no other process continues until we are done printing. As a result, on iteration `k = 0`, rank 0 process prints its own submatrix, all others wait. On iteration `k = 1`, rank 1 prints, all others wait. And so on, until we reach the last process and the end of the matrix.

We must note that the output will be much more structured, but not always be in order. In general, it is impossible to synchronize the output of an MPI program, since the output is handled by the operating system. It might be possible to write into a file in a correct order, or to send all the data to a single process for printing. However, for large scale problems, printing millions of entries or sending the entire matrix over the network would be unwise in any case.

## Test

Now, let us test the implemented functions on an example matrix $$A$$ of size $$4\times4$$. Let $$A$$ be an identity matrix:

$$
A = \left[
\begin{array}{cccc}
	1 & 0 & 0 & 0 \\
	0 & 1 & 0 & 0 \\
	\hline
    0 & 0 & 1 & 0 \\
	0 & 0 & 0 & 1 \\
\end{array}
\right]
$$

We test our code in `main.c` as follows:

```c
#include <stdio.h>
#include <openmpi/mpi.h>
#include "matrix.h"

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);
    
    int N = 4;
    Matrix A;
    MatrixCreate(MPI_COMM_WORLD, N, N, &A);
    for(int i=0; i<N; i++) MatrixSetValue(A, i, i, 1.0);
    MatrixView(A);
    MatrixDestroy(&A);

    MPI_Finalize();

    return 0;
}
```

We compile the code as `mpicc main.c matrix.c -o main` and run it as `mpiexec -n 2 ./main`. The output should be something in the lines of (might be wrong order)

```
Matrix of size (4, 4) on 2 processes:
[0]
1.000000	0.000000	0.000000	0.000000	
0.000000	1.000000	0.000000	0.000000	
[1]
0.000000	0.000000	1.000000	0.000000	
0.000000	0.000000	0.000000	1.000000
```

## Conclusion

To sum up, MPI is a powerful tool for writing programs for distributed memory, where each "device" runs its own copy of the executable and has its own memory. By distinguishing processes based on rank, one could implement sophisticated parallel data structures and algorithms. Next time, we will implement parallel vectors and parallelized matrix-vector products.