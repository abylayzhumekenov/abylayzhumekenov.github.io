---
layout:	post
title:	"Distributed matrices using MPI: part 2"
date:	2023-10-07 22:50:00 +0300
categories: Computing
---
{% include_relative _includes/head-custom.html %}


## Introduction

Last time we have implemented distributed vector using MPI in C. Today, we will do the same for distributed dense matrices. The approach to MPI dense matrices is essentially the same as for vectors, and can even be extended to sparse matrices. For the sake of simplicity, we will consider only dense matrices.


## Distributed matrices

First, with additional dimension, there are now two ways to split matrices. A common approach is to assign row blocks to different MPI processes, similar to what we did to vectors. Alternatively, one could split based on column blocks. For constitency and more efficient matrix-vector products, we will use the row based partitioning, which implies a row-oriented dense matrix format. Let $$A$$ be a $$8\times8$$ dense matrix split between 3 processes.

$$
A = \left[
\begin{array}{cccccccc}
	a_{0,0} & a_{0,1} & a_{0,2} & a_{0,3} & a_{0,4} & a_{0,5} & a_{0,6} & a_{0,7} \\
	a_{1,0} & a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4} & a_{1,5} & a_{1,6} & a_{1,7} \\
	a_{2,0} & a_{2,1} & a_{2,2} & a_{2,3} & a_{2,4} & a_{2,5} & a_{2,6} & a_{2,7} \\
    \hline
	a_{3,0} & a_{3,1} & a_{3,2} & a_{3,3} & a_{3,4} & a_{3,5} & a_{3,6} & a_{3,7} \\
	a_{4,0} & a_{4,1} & a_{4,2} & a_{4,3} & a_{4,4} & a_{4,5} & a_{4,6} & a_{4,7} \\
	a_{5,0} & a_{5,1} & a_{5,2} & a_{5,3} & a_{5,4} & a_{5,5} & a_{5,6} & a_{5,7} \\
    \hline
	a_{6,0} & a_{6,1} & a_{6,2} & a_{6,3} & a_{6,4} & a_{6,5} & a_{6,6} & a_{6,7} \\
	a_{7,0} & a_{7,1} & a_{7,2} & a_{7,3} & a_{7,4} & a_{7,5} & a_{7,6} & a_{7,7} \\
\end{array}
\right]
$$

As we see, the row-orientation better fits the row-partitioning, as we have more contiguous data layout. Note that despite owning only several rows, ranks own all the columns. That is, there is no concept of local columns here. However, that is not always the case. For PETSc matrices, one can also indicate the local column sizes. Altought this does not really change the memory layout, it is usually done to speed up matrix-vector products by multiplying block diagonals while communicating.

$$
A = \left[
\begin{array}{ccc:ccc:cc}
	a_{0,0} & a_{0,1} & a_{0,2} & a_{0,3} & a_{0,4} & a_{0,5} & a_{0,6} & a_{0,7} \\
	a_{1,0} & a_{1,1} & a_{1,2} & a_{1,3} & a_{1,4} & a_{1,5} & a_{1,6} & a_{1,7} \\
	a_{2,0} & a_{2,1} & a_{2,2} & a_{2,3} & a_{2,4} & a_{2,5} & a_{2,6} & a_{2,7} \\
    \hline
	a_{3,0} & a_{3,1} & a_{3,2} & a_{3,3} & a_{3,4} & a_{3,5} & a_{3,6} & a_{3,7} \\
	a_{4,0} & a_{4,1} & a_{4,2} & a_{4,3} & a_{4,4} & a_{4,5} & a_{4,6} & a_{4,7} \\
	a_{5,0} & a_{5,1} & a_{5,2} & a_{5,3} & a_{5,4} & a_{5,5} & a_{5,6} & a_{5,7} \\
    \hline
	a_{6,0} & a_{6,1} & a_{6,2} & a_{6,3} & a_{6,4} & a_{6,5} & a_{6,6} & a_{6,7} \\
	a_{7,0} & a_{7,1} & a_{7,2} & a_{7,3} & a_{7,4} & a_{7,5} & a_{7,6} & a_{7,7} \\
\end{array}
\right]
$$

Let us go into the file `matrix.h` and add 

```c
#ifndef MATRIX_H
#define MATRIX_H

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct _Matrix *Matrix;

// functions

#endif
```

As with vectors, we use opaque data types. We define a new `struct` with a pretty name `_Matrix`, where we will store all our matrix data. This structure is incomplete, and the exact contents will be given in `matrix.c` source file. However, when the user (programmer writing the `main.c` file) includes the header file `matrix.h`, one will only see the incomplete matrix data type. The compiler and the user do not know the true size of this data type and cannot allocate memory for it. Therefore, secondly, we will give the user a handle - a pointer `*Matrix` to our matrix structure. That is, `Matrix` is an alias for `*_Matrix`, a pointer to `_Matrix`. Since pointers are not the data itself, but are just 64 bit addresses, the compiler does not have a problem with allocating extra 64 bits of memory.

Thanks to our incomplete definition, we can use our new data type to forward declare the functionality we want from our parallel matrices.

```c
void MatrixCreate(MPI_Comm comm, int N, int M, Matrix* A);
void MatrixDestroy(Matrix* A);
void MatrixSetValue(Matrix A, int i, int j, double val);
void MatrixView(Matrix A);
```

The implementations in `matrix.c` file look as follows. The fields are the same as for `_Vector`, except integers `m` and `M` are local and global column dimensions. In our case, we set them to be equal `m = M`.

```c
#include "matrix.h"

typedef struct _Matrix {
    MPI_Comm comm;
    int rank, size;
    int n, m, N, M;
    int *isize, *istart, *iend;
    double* val;
} _Matrix;
```

## MatrixCreate

The function has a few extra things. First, we set `m` and `M`. Second, we now allocate `n * M` doubles for the matrix data.

```c
void MatrixCreate(MPI_Comm comm, int N, int M, Matrix* A){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int n = N / size + (rank < N % size);

    (*A) = malloc(sizeof(_Matrix));
    (*A)->comm = comm;
    (*A)->rank = rank;
    (*A)->size = size;
    (*A)->n    = n;
    (*A)->m    = M;
    (*A)->N    = N;
    (*A)->M    = M;
    (*A)->isize  = calloc(size, sizeof(int));
    (*A)->istart = calloc(size, sizeof(int));
    (*A)->iend   = calloc(size, sizeof(int));
    (*A)->val    = calloc(n * M, sizeof(double));

    (*A)->isize[0] = N / size + (0 < N % size);
    (*A)->iend[0] = (*A)->isize[0];
    for(int k=1; k<size; k++){
        (*A)->isize[k]   = N / size + (k < N % size);
        (*A)->istart[k]  = (*A)->istart[k-1]  + (*A)->isize[k-1];
        (*A)->iend[k]    = (*A)->iend[k-1]    + (*A)->isize[k];
    }

}
```


## MatrixDestroy

The destroy function for deallocating the memory

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

Navigating C arrays to set individual matrix entries might be too tiresome for the user. Especially when the matrix is distributed and its indices do not correspond to those of local arrays. We use the information from `istart` to correctly decide when we need to modify the local data. If the index `i` is in the ownership range, we can insert the value on `i - A->istart[A->rank]`th row of the local matrix.

```c
void MatrixSetValue(Matrix A, int i, int j, double val){
    if(i >= A->istart[A->rank] && i < A->iend[A->rank]){
        A->val[(i - A->istart[A->rank]) * A->M + j] = val;
    }
}
```


## MatrixView

The function to print the matrix values. Again, order can be random sometimes.

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


## Testing the functions

Finally, we test the functions on an example matrix $$A$$ of size $$8\times8$$. Let $$A$$ be an identity matrix:

$$
A = \left[
\begin{array}{cccccccc}
	1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
	0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
    \hline
	0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
	0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
	0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
    \hline
	0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
	0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{array}
\right]
$$

Replace the old test code from `main.c` to:

```c
#include <stdio.h>
#include <mpi.h>
#include "vector.h"
#include "matrix.h"

int main(int argc, char **argv){

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int N = 8;

    Matrix A;
    MatrixCreate(MPI_COMM_WORLD, N, N, &A);
    for(int i=0; i<N; i++) MatrixSetValue(A, i, i, 1.0);
    MatrixView(A);
    MatrixDestroy(&A);    

    MPI_Finalize();

    return 0;
}
```

Compile with `mpicc matrix.c main.c -o main` and run as `mpiexec -n 3 ./main`. The output should be something along the lines of (might be wrong order):

```
Matrix of size (8, 8) on 3 processes:
[0]
1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	
0.000000	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	
0.000000	0.000000	1.000000	0.000000	0.000000	0.000000	0.000000	0.000000	
[1]
0.000000	0.000000	0.000000	1.000000	0.000000	0.000000	0.000000	0.000000	
0.000000	0.000000	0.000000	0.000000	1.000000	0.000000	0.000000	0.000000	
0.000000	0.000000	0.000000	0.000000	0.000000	1.000000	0.000000	0.000000	
[2]
0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	1.000000	0.000000	
0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000	1.000000
```


## Conclusion

It took only some minor changes to our previous code for parallel vectors to implement parallel matrices. Ideally, to reduce the code duplication, one would unite these two interfaces (`Vector` and `Matrix`) into a single one. For more complex cases though, such as sparse matrices, data layout and required functionalitites might be too different from the implementation standpoint. 