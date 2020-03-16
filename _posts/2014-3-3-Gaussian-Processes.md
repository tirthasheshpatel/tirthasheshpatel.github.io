---
layout: post
title: Gaussian Processes for Machine Learning
category: Machine Learning
tags: [gaussian process, machine learning, statistics]
---

Gaussian Processes are stationary ramdom processes that can be used for spatial as well as temporal predictions. They have been used extensively by diciplines such as astronomy and biology for prediction tasks and meansuring uncertainty in thier data. In this article, I have collected many different approaches to Gaussian Processes and their variations.

### What are Gaussian Processes

The formal definition of Gaussian Processes may be scary for many beginners, so here is a very simple definition by Rasmussen and Williams that I love:

> *A Gaussian Process is a collection of random variables,any finite number of which have (consistent) joint Gaussian distributions.*

A Gaussian Process is fully parameterized by its mean function (not a mean vector) \\( m \\) and a covariance function \\( k \\) and can be written as:

\\[ f \sim \mathcal{GP}\left( m, k \right) \\]

These functions take a bunch of data points, commonly called "index" points and spit out a mean and covariance function and the underlying (latent) function responsible for the generation of data can then be modelled as a multivariate normal distribution.

\\[ f(X) \sim \mathcal{N}\left(m(X), k(X, X^\prime)\right) \\]

### Implementation

Let's see different ways of implementing Gaussain Processes in Python.

```python

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Let's generate some data
X = np.linspace(-5, 5, num=200).reshape(-1, 1)
m = tf.zeros(X.shape[0])
k = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=2., length_scale=1.5, feature_ndims=1)(X, X)
# A noisy latent function
f_true = np.random.multivariate_normal(loc=m, scale=k, size=1).ravel() + np.random.randn(X.shape[0])

# Let's see how the data looks
plt.scatter(X, f)
plt.xlabel("data")
plt.ylabel("latent function")
plt.title("Synthetic data")
plt.show()

# Let's see how to implement Gaussian Processes in tensorflow_probability
kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(amplitude=1., length_scale=1., feature_ndims=1)
# The line below creates for us a Gaussian Process object
# which is quvivalent to equation (1) we saw earlier.
gp = tfp.distributions.GaussianProcesses(index_points=X)
# Now, we can use the `gp` object to sample from the
# underlying function.
samples = gp.sample((1,)).numpy().ravel()

```

### Referances

[Gaussian Processes in Machine Learning by Rasmussen et. al.](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
