---
layout:	post
title:	"Distributed matrices using MPI: part 0"
date:	2023-09-29 22:34:00 +0300
categories: Computing
---
{% include_relative _includes/head-custom.html %}


## Introduction

Parallelized sparse and dense matrix algorithms serve as an important basis for many scientific computing applications. While OpenMP and other parallel programming interfaces can be used on shared memory devices (e.g. laptop, cluster node), for larger problems, a single machine can seldom provide enough memory or enough CPUs to parallelize over. To write concurrent programs that use several machines, devices or cluster nodes (what supercomputers essentially are), one could use a Message Passing Interface (MPI). The key difference between MPI and OpenMP is that OpenMP threads can access the same memory, where each MPI process lives on a separate machine and cannot directly access others' memory. To communicate, we need to send chunks of data back and forth between physical machines. The speed at which this happens is very slow compared to a CPU speed, and thus, one cannot send too much over a network. Understanding this bottleneck is important for writing better MPI programs.

In following posts, we will implement distributed vectors and dense matrices, as well as parallelized matrix-vector products, in C language using MPI. The C language has simple syntax and is explicit about memory, therefore it is our choice. We note that although MPI is mainly used for applications run on supercomputers, we can write and test them on shared memory devices too. That is, we can use it to run parallel programs on our laptops. Sometimes applications choose to use MPI instead of OpenMP for their shared memory implementations. 


## Installing MPI

For the exercise, we can use both MPICH or Open MPI (do not confuse with OpenMP) implementations of the MPI. If you do not have, you can use 

```
sudo apt install mpich
```

or

```
sudo apt install libopenmpi-dev openmpi-bin
```

to install the respective libraries. This will let you use MPI compilers `mpicc` instead of usual `gcc` to compile C source codes into a parallel program, e.g.: `mpicc main.c -o main`. Then we can run it by specifying the number of processes as `mpiexec -n 2 ./main`.


## Project directory

Let us create a new project folder `distributed_matrices`, then add 7 files to it.

```
.
├── main.c
├── matrix.c
├── matrix.h
├── operations.c
├── operations.h
├── vector.c
└── vector.h
```

It is better to organize the code in modules. In C, we achieve this by creating separate header files `.h` and corresponding source files `.c` for the functionality we intend to implement. In this case, for a distributed vector. This also provides some basic encapsulation and is a good exercise for writing APIs and libraries.


## Compiling and running MPI programs

Now let us write a simple program which prints the process rank, a unique number every MPI process gets at the initialization. Go into `main.c` file and add the following lines:

```c
#include <mpi.h>
#include <stdio.h>

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
mpiexec -n 3 ./main
```

to get something like (order might be different)

```
The process 0 says hello world!
The process 1 says hello world!
The process 2 says hello world!
```


## Conclusion

We see that our program just duplicates the code using three processes, with the `rank` variable having different values, 0, 1 and 2. This is essentially how all MPI programs work, they execute the same code with some variables set to defferent values. Depending on the value of `rank` and using `if` statements, for example, we can force processes to work on different parts of a matrix. This is how the parallelization is achieved.