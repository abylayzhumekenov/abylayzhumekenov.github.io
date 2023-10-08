---
layout:	post
title:	"Distributed matrices using MPI: part 1"
date:	2023-09-30 21:50:00 +0300
categories: Computing
---
{% include_relative _includes/head-custom.html %}


## Introduction

Previously we gave a basic introduction to compiling and running MPI programs, as well as structuring C projects. For today, we will implement distributed vectors, so-called MPI vectors. We will primarily work with `main.c`, `vector.h` and `vector.c` files.


## Distributed vectors

For this exercise, we will pretend that MPI processes reside on different machines (although they do not), and each will store only one part of the vector. Vectors can be split into contiguous parts. This ensures the data locality and, consequently, higher cache hit rate. Let us consider a vector $$x$$ of size 8 and let the number of processes to be 3. We must have a consistent way to determine how many elements will be in each of 3 partitions.
We can prioritize lower ranks to have more elements. For example,

$$
x = \left[
\begin{array}{c}
    x_0\\
    x_1\\
    x_2\\
    \hline
    x_3\\
    x_4\\
    x_5\\
    \hline
    x_6\\
    x_7\\
\end{array}
\right]
$$

The rank 0 gets 3 local elements $$x_0, x_1, x_2$$, the rank 1 gets 3 elements $$x_3, x_4, x_5$$ and the last rank 2 gets 2 elements $$x_6, x_7$$. The horizontal line indicates that the rows belong to different processes. The exact ownership rule can be specified when creating the vector. Without further adew, let us go into `vector.h` and define our vector data structure.

```c
#ifndef VECTOR_H
#define VECTOR_H

#include <openmpi/mpi.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct _Vector *Vector;

// functions

#endif
```

The code above does several things. The `#ifndef`, `#define` and `#endif` are preprocessor keywords, and are used here to define `VECTOR_H` variable. This is called a header guard, which prevents the `vector.h` file to be repeatedly included. It is a good practice, but for now, we can ignore it. The `#include` includes the MPI and standard library headers. Second, using `typedef`, we define a new type name (alias) called `Vector`, which is a pointer to `struct _Vector`. The `_Vector` structure is still not defined, which makes it incomplete. We will provide the exact structure in `vector.c` file. This is called opaque data types, and it is a popular approach in many C libraries. The benefit of this approach is that the user (programmer) is abstracted from the internals and implementations of the `_Vector` data structure, and is provided just enough knowledge to interact with the vector. 

After `typedef`ing `Vector`, we now can define functions to be applied on vectors. For instance, we want to allocate memory for our vector, to free it when needed, to set its entries and possibly to print it. These functions can be defined in `vector.h` as the interface for users to interact, we place them instead of `// functions` comment.

```c
void VectorCreate(MPI_Comm comm, int N, Vector* x);
void VectorDestroy(Vector* x);
void VectorSetValue(Vector x, int i, double val);
void VectorView(Vector x);
```

The implementations follow in `vector.c` source file. We start by completing the structure definition

```c
#include "vector.h"

typedef struct _Vector {
    MPI_Comm comm;
    int rank, size;
    int n, N;
    int *isize, *istart, *iend;
    double* val;
} _Vector;
```

The structure holds all the necessary data for our parallel vector and is seen only in `vector.c`. Here, `comm` is an MPI communicator, we will need it for communication between processes for various matrix operations. The `rank` and `size` hold the process rank and overall number of processes. Integer `N` is the global vector dimension, whereas `n` is the local size on this `rank`. When we run our program, each process will have to allocate its own memory, and local sizes might differ for each rank. One would often need additional information about other ranks. This info can be stored in arrays `isize`, `istart` and `iend`, each of size `size`. These are duplicated on each process, and contain the number of owned rows, the starting row index, the ending row index (exclusive), correspondingly. Lastly, we need an array `val` for storing local vector entries.

## VectorCreate

Let us implement the `VectorCreate` function. The very first thing to do is to understand our place in the world by using `MPI_Comm_rank` and `MPI_Comm_size` functions, which extract the `rank` and the `size` of the communicator.

```c
int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);
```

Now we have to decide how we split our vector. Given global row dimensions `N`, we have to calculate how many will be local `n` rows. Since we are working with integers, each process should get at least `N / size` rows. E.g, for `N = 8` and `size = 3`, each rank gets `8 / 3 = 2` rows at minimum. Further, we must somehow distribute the remaining rows, of which, there are `N % size` or `8 % 3 = 2`. We can make the following rule: if the remainder is `r`, the first `r` ranks will get one additional row. This can nicely be written as `rank < N % size`, which is 2 for the first `r` ranks and 0 for the rest. To get the intuition, you can play this game on a paper using other numbers.

```c
int n = N / size + (rank < N % size);
```

Next, using the `malloc` function, we allocate memory to hold our vector. We also allocate memory for our arrays `isize`, `istart`, `iend` and `val` using `calloc`, which initializes arrays to 0. Since each process stores only its own portion of the vector, we need to allocate only `n` doubles for the array `val`. Other fields related to the communicator and dimensions are initalized as well. Note that `x` is a pointer to `Vector`, which is itself a pointer to `_Vector`, i.e. `x` is of type `Vector**`. To access our fields, we do something like `(**x).comm` or `(*x)->comm`.

```c
(*x) = malloc(sizeof(_Vector));
(*x)->comm = comm;
(*x)->rank = rank;
(*x)->size = size;
(*x)->n    = n;
(*x)->N    = N;
(*x)->isize  = calloc(size, sizeof(int));
(*x)->istart = calloc(size, sizeof(int));
(*x)->iend   = calloc(size, sizeof(int));
(*x)->val    = calloc(n, sizeof(double));
```

The row split is fully deterministic and does not require any communication. That means we can generate this information on every process, so that everyone knows who owns how many rows, and the ownership ranges. Here, the `isize` array stores the number of rows each rank owns. The arrays `istart` and `iend` store the start and end indices of the local submatrix using the global ordering. For instance, in the example we provided `size = 3` and `N = 8`, which means `isize = [3, 3, 2]`, `istart = [0, 3, 6]` and `iend = [3, 6, 8]`.

```c
(*x)->isize[0] = N / size + (0 < N % size);
(*x)->iend[0] = (*x)->isize[0];
for(int k=1; k<size; k++){
    (*x)->isize[k]   = N / size + (k < N % size);
    (*x)->istart[k]  = (*x)->istart[k-1]  + (*x)->isize[k-1];
    (*x)->iend[k]    = (*x)->iend[k-1]    + (*x)->isize[k];
}
```

The full implementation is

```c
void VectorCreate(MPI_Comm comm, int N, Vector* x){
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int n = N / size + (rank < N % size);

    (*x) = malloc(sizeof(_Vector));
    (*x)->comm = comm;
    (*x)->rank = rank;
    (*x)->size = size;
    (*x)->n    = n;
    (*x)->N    = N;
    (*x)->isize  = calloc(size, sizeof(int));
    (*x)->istart = calloc(size, sizeof(int));
    (*x)->iend   = calloc(size, sizeof(int));
    (*x)->val    = calloc(n, sizeof(double));

    (*x)->isize[0] = N / size + (0 < N % size);
    (*x)->iend[0] = (*x)->isize[0];
    for(int k=1; k<size; k++){
        (*x)->isize[k]   = N / size + (k < N % size);
        (*x)->istart[k]  = (*x)->istart[k-1]  + (*x)->isize[k-1];
        (*x)->iend[k]    = (*x)->iend[k-1]    + (*x)->isize[k];
    }
}
```


## VecDestroy

The memory allocated dynamically using `malloc` or `calloc` cannot be reclaimed by the operating system, unless we `free` it or the program terminates. If we provide a function for creating vectors, we must provide a function for destroying them too. This will help the user to get better control of one's memory consumption. The following code deallocates the contents of a vector in reverse order: first it frees arrays `isize`, `istart`, `iend` and `val`, and only after that the structure itself. If we `free(*A)` first, we lose the reference to `(*A)->val` and we cannot deallocate them. Since the arrays are still unreclaimed, this would be a memory leak.

```c
void VectorDestroy(Vector* x){
    free((*x)->isize);
    free((*x)->istart);
    free((*x)->iend);
    free((*x)->val);
    free((*x));
}
```


## VecSetValue

Since each `Vector` instance on different ranks stores its own local data, setting or accessing elements from the vector is not straightforward anymore. For example, if we want to insert a value in row `4`, only the corresponding process owning the row (`rank = 1`) has to modify its local array. Luckily, we can access the ownership information we saved in `istart` and `iend` arrays. If the index `i` is in the ownership range, we can insert the value on `i - x->istart[x->rank]`th row of the local vector. For the example vector, the offsets are equal to `istart = [0, 3, 6]` correspondingly.

```c
void VectorSetValue(Vector x, int i, double val){
    if(i >= x->istart[x->rank] && i < x->iend[x->rank]){
        x->val[(i - x->istart[x->rank])] = val;
    }
}
```


## VecView

At last, we implement a function to print our vector. Input and output in an MPI program is quite tricky. The reason is that all MPI processes run the same code simultaneously, and if we want to naively print array contents, the output would be a mess. A slightly more successful approach is to force processes to wait for each other. This can be accomplished with `MPI_Barrier` function.

```c
void VectorView(Vector x){
    if(!x->rank) printf("Vector of size (%i,) on %i processes:\n", x->N, x->size);
    for(int k=0; k<x->size; k++){
        if(x->rank == k){
            printf("[%i]\n", x->rank);
            for(int i=0; i<x->n; i++){
                printf("%f\n", x->val[i]);
            }
        }
        MPI_Barrier(x->comm);
    }
}
```

The code above runs a `k` loop from 0 to `size` on each process. Inside, we check if the `rank` is equal to current `k`. If yes, we print the local subvector. If not, we do nothing. The `MPI_Barrier` at the end makes sure that no other process continues until we are done printing. As a result, on iteration `k = 0`, rank 0 process prints its own subvector, all others wait. On iteration `k = 1`, rank 1 prints, all others wait. And so on, until we reach the last process and the end of the vector.

We must note that the output will be much more structured, but not always be in order. In general, it is impossible to synchronize the output of an MPI program, since the output buffer is handled by the operating system. It might be possible to write into a file in a correct order, or to send all the data to a single process for printing. However, for large scale problems, printing millions of entries or sending the entire matrix over the network would be unwise in any case. So we restrict ourselves to an imperfect, but probably good enough implementation.


## Testing the functions

Now, let us test our implementations using the vector

$$
x = \left[
\begin{array}{c}
    0\\
    1\\
    2\\
    \hline
    3\\
    4\\
    5\\
    \hline
    6\\
    7\\
\end{array}
\right]
$$

We test our code in `main.c` as follows

```c
#include <stdio.h>
#include <openmpi/mpi.h>
#include "vector.h"

int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    int N = 8;
    Vector x;
    VectorCreate(MPI_COMM_WORLD, N, &x);
    for(int i=0; i<N; i++) VectorSetValue(x, i, i);
    VectorView(x);
    VectorDestroy(&x);

    MPI_Finalize();

    return 0;
}
```

To compile the program, run `mpicc vector.c main.c -o main`. To run the executable using 3 processes, the command is `mpiexec -n 3 ./main`. The output should be something alone these lines (order might be different):

```
Vector of size (8,) on 3 processes:
[0]
0.000000
1.000000
2.000000
[1]
3.000000
4.000000
5.000000
[2]
6.000000
7.000000
```


## Conclusion

To sum up, MPI is a powerful tool for writing programs for distributed memory, where each "device" runs its own copy of the executable and has its own memory. By distinguishing processes based on rank, one could implement sophisticated parallel data structures and algorithms. Next time, we will implement parallel matrices and parallelized matrix-vector products.