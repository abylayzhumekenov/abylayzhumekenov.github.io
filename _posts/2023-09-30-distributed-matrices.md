---
layout:	post
title:	"Distributed matrices using MPI"
date:	2023-09-30 02:05:00 +0300
categories:	Category
---
{% include_relative _includes/head-custom.html %}

Parallelized sparse and dense matrix algorithms serve as an important basis for many scientific computing applications.


In this post, we will implement distributed (parallel) matrices using MPI (Message Passing Interface).

First, there are several ways to split a large matrix. A common way is to assign row blocks to different MPI processes.
Let $$A$$ be a $$4\times4$$ matrix. We will split its rows evenly as follows:

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

The rank 0 process gets rows 0 and 1, while the rank 1 process gets to store rows 2 and 3, the horizontal line indicates
that the rows belong to different processes. Another apporach would be to split according to column indices, in which case
the dashed line would be the border. We will stick to the first approach. Note that the submatrix on rank 0 is of size $$2\times4$$,
that is, despite owning only 2 rows, it owns all the columns. This would be beneficial for performing matrix-vector products.

We start by defining a structure that will hold all the necessary data for our parallel matrix.

```
typedef struct _Matrix {
    MPI_Comm comm;
    int rank, size;
    int n, m, N, M;
    int *isize, *istart, *iend;
    double* val;
} _Matrix;
```

Here, `comm` is an MPI communicator, we will need it for communication between processes for various matrix operations. The `rank` and
`size` hold the process rank and overall number of processes. Integers `N` and `M` are global matrix dimensions, 4 and 4 for the matrix $$A$$.
Whereas `n` and `m` are local matrix sizes, which are 2 and 4 (`m = M` for now). When we run our program, each process will have allocate
its own memory, and local sizes might differ for each rank. One would often need additional information about other ranks. This info can be
stored in arrays `isize`, `istart` and `iend`, each of size `size`. These are duplicated on each process, and contain the number of owned rows,
the starting row index, the ending row index (exclusive), correspondingly. Lastly, we need an array `val` for storing our matrix entries.




```
git clone -b release https://gitlab.com/petsc/petsc.git petsc
git pull # obtain new release fixes (since a prior clone or pull)
```

For more visit their [official website][petsc-website]

Also we could write math like $$ c = \sqrt{a^2+b^2} $$

And some code in C:

{% highlight c %}
int main(){
	return 0;
}
{% endhighlight %}

[petsc-website]: https://petsc.org/
