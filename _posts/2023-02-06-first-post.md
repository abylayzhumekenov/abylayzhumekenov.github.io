---
layout:	post
title:	"First post"
date:	2023-02-06 01:39:00 +0300
categories:	Category
---
{% include_relative _includes/head-custom.html %}

How to install `PETSc` library

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
