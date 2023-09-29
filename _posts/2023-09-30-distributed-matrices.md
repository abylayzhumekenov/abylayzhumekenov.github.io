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
Let $A$ be a $4\times4$ matrix. We will split its rows evenly as follows:

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
