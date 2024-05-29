# MatMul

## About

After a series of machine learning algos to predict instead of evaluate on addition and multipliction, now we try to do something more worth a while.  

We try to predict matrix multiplications! The success of this project can be enormous... CNNs work on O(n2) asymptotically. With this prediction model we will be able to compute the matrix multiplication in O(n2).

We then will be able to reduce the order of matrix multiplication by an order O(n3) -> O(n2)!!!

## Results

- So far we have built the model only for integer matrices.
- Integers are limited to 1, 2, 3
- We considered only 2 x 2 matrices
- MSE Loss achieved so far is 0.044
- Therefore all the elemnets of the matrices differ from the actual value only by 0.2
- t1 = [[1, 1], [1, 1]], t1 @ t2 = (pred) [[2.4431, 2.5671], [3.4695, 3.4248]]

## Vision

We want this model not to learn pair wise data, but to learn the broader concpet matrix multiplication
