
library(cart)
library(rpart.plot)
library(stringr)
library(dplyr)
set.seed(1)
d = 4
n = 40
d = 12
n = 1000
X = mvtnorm::rmvnorm(n, rep(0, d), diag(1, d)) # covariates
beta = 1:d
eps = rnorm(n, 0, 3)
y = X %*% beta + eps # response
z = cbind(y, X)

## faster version that uses the rpart optimization
# part_new = f(z, FALSE) # old tree building routine
part_new = f_partition(z, FALSE) # old tree building routine
dim(part_new) # (2d x n_leaf)

## slower version where we sort every iteration:
part_old = old_partition(z) # old tree building routine
dim(part_old) # (2d x n_leaf)

## rpart code
cart_r = function(y, X) {
  tree_model = rpart(y ~ X)
  # rpart.plot(tree_model)
  # param_support = extractSupport(X, d) ## get the support of the data
  ## partition in good format
  # rpart_out = t(extractPartition(tree_model, param_support)[,-c(1:2)])
  # rpart_out
}


microbenchmark::microbenchmark(
  cpp_old = old_build(z),
  cpp_new = f_build(z, FALSE),
  r       = cart_r(y, X),
  times = 50
)





# d = 5
# n = 100
# X = matrix(rnorm(n * d), n, d)
# b = runif(d, 0, 10)
# y = X %*% b + rnorm(d)
# z = cbind(y, X)





## rpart output
tree_model = rpart(y ~ X)
rpart.plot(tree_model)
param_support = extractSupport(X, d) ## get the support of the data
## partition in good format
rpart_out = t(extractPartition(tree_model, param_support)[,-c(1:2)])
dim(rpart_out)

# verify output against rpart() partition output


tree_model = rpart(y ~ X)
param_support = extractSupport(X, d) ## get the support of the data
## partition in good format
rpart_out = t(extractPartition(tree_model, param_support)[,-c(1:2)])

