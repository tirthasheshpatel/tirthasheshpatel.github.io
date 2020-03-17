---
type: post
title: A review of Gaussian Processes
category: [Machine Learning, GSoC 2020]
tags: [gsoc, machine learning]
---

### Abstract

I have collected and reviewed proposed literature for Gaussian Processes and thier applications. I have also provided code to implement them in Python.

### 1. Introduction

Gaussian Processes are non-parametric models being used extensively used for spatial as well as temporal regression and prediction tasks [1], [2]. The idea of Gaussian Process Modelling is to infer a latent function underlying the data by assuming a prior over a space of functions. Gaussian Processes can be thought of as a Gaussian Distribution over an infinite vector space. The authors of [2] compare a Gaussain Process Model with a neural network having a single layer of infinite neurons.

> Gaussian Process is in fact one HUGE gaussian!

Just like a Gaussian Distribution, a Gaussian Process can be parameterized by a mean and a covariance function. The mean function, $m$, is usually assumed to be a zero function but it can be made to reflect the most information of the data known apriori. Literature has esspecially been interested in covariance functions, $k$, due to the positive semi-definite constraint and they have been explored extensively. We review all these mean and covariace functions in the latter sections.

$$f \sim \mathcal{GP}(m, k)$$

This article follows a notation provided below.

- $X$: Upper case letters in an equation denote a matrix.
- $x$: Lower case letters in an equation denote a vector.

### 2. Mean functions

There hasn't been a huge interest in exploring the nature of mean functions in Gaussaian Process literature. I present a few mean functions that have been proposed to work well in practice.

1. Zero Mean function: This mean function is a mapping of data to zero vector $\R^{m \times n} \to \R^{m}$ where $m$ are the number of samples and $n$ are the number of features. This is usually a choice for most of the tasks concerning big data as it is extreamly difficult to encode apriori knowledge in such a function. Following is a code snippet to code a mean function in ``python`` using ``numpy`` library.

$$f(X) = \overrightarrow{0}$$

```python
import numpy as np

def zero_mean_function(X):
    """The zero mean function

    Parameters
    -----
    X: array-like
        Data matrix of shape (n_samples, n_features)
    
    Returns
    -----
    A zero vector of shape (n_samples, )
    """
    return np.zeros(X.shape[0])
```

2. Constant mean: This mean function is a generalization over zero mean. It maps the input to a vector with a constant value in all its dimensions. Often, a good practicle choice of the constant to map to is the mean of the data at hand. It statistically holds more information that zero mean and hence may lead to a better posterior (though in practice it doesn't make much difference). Zero mean can easily be derieved by setting the coefficient to zero.

$$f(X) = c$$

```python
def constant_mean_function(X, coef):
    """The constant mean function

    Parameters
    -----
    X: array-like
        Data matrix of shape (n_samples, n_features)

    coef: float
        The coefficient in each dimension.

    Returns
    -----
    A vector of shape (n_samples, ) with a
    constant `coef` in all the dimension.
    """
    return coef * np.ones(X.shape[0])
```

3. Linear Mean: This mean function is a generalization over constant mean. It is a mapping $\R^{m \times n} \to \R^{m}$ parametrized by a weight vector $w$ and a bias or intercept $b$. The following equation shows the mapping.

$$f(X) = w^{T}X + b$$

```python
def linear_mean_function(X, w, b):
    """Linear mean function
    m(x) = wx + b

    Parameters
    -----
    w: array-like
        A vector of weights of shape (n_features, )

    b: float
        Intercept in the linear equation

    Returns
    -----
    A vector of shape (n_samples, )
    """
    w = w.reshape(-1, 1)
    return w.T @ X + b
```

### References

[1] Rasmussen C.E. (2004) Gaussian Processes in Machine Learning. In: Bousquet O., von Luxburg U., Rätsch G. (eds) Advanced Lectures on Machine Learning. ML 2003. Lecture Notes in Computer Science, vol 3176. Springer, Berlin, Heidelberg

[2] MacKay, David J. C.. “Introduction to Gaussian processes.” (1998).
