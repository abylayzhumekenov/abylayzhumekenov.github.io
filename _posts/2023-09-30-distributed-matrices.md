---
layout:	post
title:	"Distributed matrices using MPI"
date:	2023-09-30 02:05:00 +0300
categories:	Category
---
{% include_relative _includes/head-custom.html %}

## Introduction

Parallelized sparse and dense matrix algorithms serve as an important basis for many scientific computing applications. While OpenMP and other parallel programming interfaces can be used on shared memory devices (e.g. laptop, cluster node), for larger problems, a single machine can seldom provide enough memory or CPUs to parallelize over. To write concurrent programs that use several machines, devices or cluster nodes (what supercomputers essentially are), one could use a Message Passing Interface (MPI). The key difference between MPI and OpenMP is that OpenMP threads access the same memory, where each MPI process lives on a separate machine and cannot directly access others memory. To communicate, we need to send chunks of data back and forth between physical machines. The speed at which this happens is very slow compared to a CPU speed, and thus, one cannot send too much over a network. Understanding this bottleneck is important for writing better MPI programs.

In this post, we will implement distributed (parallel) dense matrices in C using MPI. The C language has simple syntax and is explicit about memory, therefore it is our choice. We note that although MPI is mainly used for applications run on supercomputers, we can write and test them on shared memory devices too. That is, we can use it run parallel programs on out laptops. Some applications even use MPI instead of OpenMP for their shared memory implementations. For the exercise, we can use both MPICH or Open MPI (do not confuse with OpenMP) implementations of the MPI. If you do not have, you can always `sudo apt install` them. This will let you use MPI compilers `mpicc` instead of usual `gcc` to compile C source codes into a parallel program, e.g.: `mpicc main.c -o main`. Then we can run it by specifying the number of processes as `mpiexec -n 2 ./main`.

Let us create a new project folder `distributed_matrix`, then add 3 files: `main.c`, `matrix.h` and `matrix.c`. It is better to organize the code in modules. In C, we achieve this by creating separate header files `.h` and corresponding source files `.c` for the functionality we intend to implement. In this case, for a distributed matrix. This also provides some basic encapsulation and is a good exercise for writing APIs and libraries. The structure must look as follows:

```
.
├── main.c
├── matrix.c
└── matrix.h
```

Now let us write a simple program which print the process rank, a unique number every process gets at the initialization. Go into `main.c` file and add the following lines:

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
\begin{array}{cc:cc}
	a_{0,0} & a_{0,1} & a_{0,2} & a_{0,3} \\
	a_{1,0} & a_{1,1} & a_{1,2} & a_{1,3} \\
	\hline
	a_{2,0} & a_{2,1} & a_{2,2} & a_{2,3} \\
	a_{3,0} & a_{3,1} & a_{3,2} & a_{3,3} \\
\end{array}
\right]
$$

The rank 0 process gets rows 0 and 1, while the rank 1 process gets to store rows 2 and 3, the horizontal line indicates that the rows belong to different processes. Another apporach would be to split according to column indices, in which case the dashed line would be the border. We will stick to the first approach. Note that the submatrix on rank 0 is of size $$2\times4$$, that is, despite owning only 2 rows, it owns all the columns. This would be beneficial for performing matrix-vector products.

Go into the file `matrix.h` and add 

```c
typedef struct _Matrix *Matrix;
```

Here we do two things. First, we define a new `struct` with a pretty name `_Matrix`, where we will store all our matrix data. This structure is incomplete, and the exact contents will be given in `matrix.c` source file. However, when the user (programmer writing the `main.c` file) includes the header file `matrix.h`, one will only see the incomplete matrix data type. The compiler and the user do not know the true size of this data type and cannot allocate memory for it. Therefore, second, we will give the user a handle - a pointer `*Matrix` to our matrix structure. That is, `Matrix` is an alias for `*_Matrix`, a pointer to `_Matrix`. All of this is done to abstract the user from the internals and implementation details, and provide just enough knowledge about what matrix is. For more, google opaque data types.

Thanks to our incomplete definition, we can use our new data type to forward declare the functionility we want from our parallel matrices. For instance, we want to be able to allocate memory for our matrix, to free it when needed, to set entries of the matrix, and possibly to print the matrix. These functions can be defined in `matrix.h` file, and will be the interface the users will use to interact with matrices.

```c
void MatrixCreate(MPI_Comm comm, int N, int M, Matrix* A);
void MatrixDestroy(Matrix* A);
void MatrixSetValue(Matrix A, int i, int j, double val);
void MatrixView(Matrix A);
```

The implementations will follow in `matrix.c` source file. We start by defining a structure that will hold all the necessary data for our parallel matrix. This structure is internal and is not seen from outside of `matrix.c` file.

```c
typedef struct _Matrix {
    MPI_Comm comm;
    int rank, size;
    int n, m, N, M;
    int *isize, *istart, *iend;
    double* val;
} _Matrix;
```

Here, `comm` is an MPI communicator, we will need it for communication between processes for various matrix operations. The `rank` and `size` hold the process rank and overall number of processes. Integers `N` and `M` are global matrix dimensions, 4 and 4 for the matrix $$A$$. Whereas `n` and `m` are local matrix sizes, which are 2 and 4 (`m = M` for now). When we run our program, each process will have allocate its own memory, and local sizes might differ for each rank. One would often need additional information about other ranks. This info can be stored in arrays `isize`, `istart` and `iend`, each of size `size`. These are duplicated on each process, and contain the number of owned rows, the starting row index, the ending row index (exclusive), correspondingly. Lastly, we need an array `val` for storing our matrix entries.

