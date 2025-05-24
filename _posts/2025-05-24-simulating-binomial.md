---
layout:	post
title:	"Fast simulation of binomial random variables in Rcpp"
date:	2025-04-24 15:25:00 +0800
categories: Computing
---
{% include_relative _includes/head-custom.html %}


## Introduction

R's default `rbinom` function to generate random variables from a binomial density is slow (even slower than `rnorm`). We implement a custom sampler given fixed parameters `size=64` and `prob=0.5`. We use `Rcpp` to generate 64 random bits and sum the bit fields using [Hamming weight][https://en.wikipedia.org/wiki/Hamming_weight]. 

```R
library(Rcpp)
cppFunction(includes="
#include <random>
", "
IntegerVector rbinom64(int n) {
  IntegerVector vector(n);
  uint64_t x;
  std::random_device rd;
  std::mt19937_64 gen(rd());
  for(int i=0; i<n; i++){
    x = gen();
    x -= (x >> 1) & 0x5555555555555555;
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f;
    vector[i] = (x * 0x0101010101010101) >> 56;
  }
  return vector;  
}")
```

Let us run the `rbinom` and our custom `rbinom64` functions for various sample sizes. We call each function 20 times and save the elapsed time.

```R
samples = 2^(0:25)
runs = 20
res1 = res2 = matrix(NA, runs, length(samples))
for(i in seq_along(samples)){
  cat("#samples =", samples[i], "\n")
  for(k in 1:runs){
    cat("\trun =", k, "\n")
    
    start_time <- Sys.time()
    x1 = rbinom(samples[i], size = 64, prob = 0.5)
    res1[k,i] = difftime(Sys.time(), start_time, units = "secs")
    
    start_time <- Sys.time()
    x2 = rbinom64(samples[i])
    res2[k,i] = difftime(Sys.time(), start_time, units = "secs")
    
    if(k%%(1e9%/%samples[i])==0) gc()
  }
  gc()
}
```

Now plot the speedup against the sample size. Additionally, plot the histogram to check if they match.

```R
speedup = res1/res2
speedup_mean = colMeans(speedup)
speedup_quantile = t(sapply(1:dim(speedup)[2], FUN=function(i) quantile(speedup[,i], c(0.5, 0.05, 0.95))))

par(mfrow=c(1,2))
plot(log2(samples), colMeans(speedup), ylim=range(speedup), t="l", col="red", main=paste0("rbinom64 vs rbinom, ", runs, " runs"), xlab="log2(#samples)", ylab="Speedup")
lines(log2(samples), speedup_quantile[,1], lty=1)
lines(log2(samples), speedup_quantile[,2], lty=2)
lines(log2(samples), speedup_quantile[,3], lty=2)
legend("topleft", legend=c("Mean","Mode","Q(0.05)","Q(0.95)"), lty=c(1,2,1,2), col=c("red","black","black","black"))
hist(x1, 50, prob=TRUE, col=rgb(1,0,0,0.5), border=NA, main=paste0("rbinom64 vs rbinom, n=", 2^i), xlab="x", ylab="Density")
hist(x2, 50, prob=TRUE, col=rgb(0,1,0,0.5), border=NA, add=TRUE)
par(mfrow=c(1,1))
```

<object data="/_includes/pdf/rbinom64.pdf" type="application/pdf" width="100%" height="500px">
  <p>Your browser doesn't support embedded PDFs. 
    <a href="/_includes/pdf/rbinom64.pdf">Download the PDF instead.</a>
  </p>
</object>

We see that our implementation is faster for almost all sample sizes, reaching a speedup of about x15. There is a potential to generalize the results for `size>64`, implementation would involve summing bits from multiple 64 bit fields.