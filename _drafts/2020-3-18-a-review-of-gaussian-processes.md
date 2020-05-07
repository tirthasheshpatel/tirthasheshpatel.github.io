---
type: post
title: A review of Gaussian Processes
subtitle: A theoritical introduction to Gaussian Processes and implementation from scratch using python!
image: /images/gaussian_process_files/exp_sin_sq_cov_func.png
gsoc_post: true
tags: [Machine Learning, GSoC 2020, Python]
---

### Abstract

I have collected and reviewed the proposed literature for Gaussian Processes and their applications. I have also provided code to implement them in Python.

### 1. Introduction

Gaussian Processes are non-parametric models being used extensively used for spatial as well as temporal regression and prediction tasks [1], [2]. The idea of Gaussian Process Modelling is to infer a latent function underlying the data by assuming a prior over a space of functions. Gaussian Processes can be thought of as a Gaussian Distribution over an infinite vector space. The authors of [2] compare a Gaussian Process Model with a neural network having a single layer of infinite neurons.

> Gaussian Process is, in fact, one HUGE gaussian!

Just like a Gaussian Distribution, a Gaussian Process can be parameterized by a mean and a covariance function. The mean function, $m$, is usually assumed to be a zero function but it can be made to reflect the most information of the data known apriori. Literature has especially been interested in covariance functions, $k$, due to the positive semi-definite constraint and they have been explored extensively. We review all these mean and covariance functions in the latter sections.

$$f \sim \mathcal{GP}\left(m, k\right)$$

This article follows a notation provided below.

- $X$: Upper case letters in an equation denote a matrix.
- $x$: Lower case letters in an equation denote a vector.

### 2. Mean functions

There hasn't been a huge interest in exploring the nature of mean functions in Gaussian Process literature. I present a few mean functions that have been proposed to work well in practice.

- **Zero Mean function**: This mean function is a mapping of data to zero vector $\mathbb{R}^{m \times n} \to \mathbb{R}^{m}$ where $m$ are the number of samples and $n$ are the number of features. This is usually a choice for most of the tasks concerning big data as it is extremely difficult to encode apriori knowledge in such a function. Following is a code snippet to code a mean function in ``python`` using the ``numpy`` library.

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

- **Constant mean**: This mean function is a generalization over zero mean. It maps the input to a vector with a constant value in all its dimensions. Often, a good practical choice of the constant to map to is the mean of the data at hand. It statistically holds more information than zero mean and hence may lead to a better posterior (though in practice it doesn't make much difference). Zero mean can easily be derived by setting the coefficient to zero.

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
    constant `coef` in all the dimensions.
    """
    return coef * np.ones(X.shape[0])
```

- **Linear Mean**: This mean function is a generalization over constant mean. It is a mapping $\mathbb{R}^{m \times n} \to \mathbb{R}^{m}$ parametrized by a weight vector $w$ and a bias or intercept $b$. The following equation shows the mapping.

$$f(X) = Xw + b$$

```python
def linear_mean_function(X, w, b):
    """Linear mean function
    m(X) = Xw + b

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
    return X @ w + b
```

### 3. Covariance functions

Covariance functions (or positive semi-definite kernels) have been a huge topic of interest for researchers as Gaussian processes are known primarily for predicting uncertainty in the data. This property of Gaussian processes is key to exploration in the Bayesian optimization algorithm [3], [4], [5]. The inference is very sensitive to the choice of covariance functions and hence they have to be designed carefully. Moreover, the positive semi-definite constraint on covariance functions makes them even more difficult to design. Several stationary and non-stationary covariance functions have been introduced in [6] and [7]. Here, I implement all the stationary kernels followed by periodic and non-stationary kernels.

- **Exponentiated Quadratic Kernel**: This stationary kernel is widely used and is commonly known as the Radial Basis Function (RBF) kernel. The following equation shows the RBF kernel where. This function is parametrized by an amplitude $\sigma$ and a length scale $l$.

$$K(x, x^{\prime}) = \sigma^2 \mathcal{exp}\left(-\frac{||x-x^{\prime}||^2}{2l^2}\right)$$

```python
def exponentiated_quadratic_kernel(X, X_prime, length_scale, amplitude):
    """The exponentiated quadratic kernel

    Parameters
    -----
    X: array-like
        Prior data matrix of shape (n_samples, n_features)

    X_prime: array_like
        New data matrix of shape (n_new_samples, n_features)

    length_scale: float
        The `l` parameter in the equation

    amplitude: float
        The `sigma^2` parameter in the equation

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    l2 = np.sum(X ** 2, 1).reshape(-1, 1) + (np.sum(X_prime ** 2, 1).reshape(1, -1) - 2 * X @ X_prime.T)
    return amplitude * np.exp( 0.5 * l2 / length_scale ** 2 )
```

![Exponential Quadratic Covariance Function](/images/gaussian_process_files/exp_quad_cov_func.svg)

- **Constant Kernel**: This stationary kernel is also often used for computationally light modeling. It just maps the data matrix to a covariance matrix with constant values in all the rows and columns analogous to the constant mean function. It is computationally easy to compute with the disadvantage of poor posterior covariance. It often models the uncertainty very poorly and hence is not suitable for tasks like Bayesian Optimization.

$$K(x, x^{\prime}) = C$$

where $C$ is a matrix with dimensions $\mathbb{R}^{m \times m}$ with constant entries in all its rows and columns

```python
def constant_kernel(X, X_prime, coef):
    """The constant kernel

    Parameters
    -----
    X: array-like
        Prior data matrix of shape (n_samples, n_features)

    X_prime: array_like
        New data matrix of shape (n_new_samples, n_features)

    coef: float
        The constant in all rows and columns of
        the covariance function

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    return coef * np.ones((X.shape[0], X_prime.shape[0]))
```

![Constant Kernel](/images/gaussian_process_files/constant_cov_func.svg)

- **Linear Kernel**: This is a stationary kernel which is parameterized by bias variance ($\sigma_b$), slope variance ($\sigma_w$) and shift ($s$). It is analogous to linear mean function and the equation is given by

$$K(x, x^{\prime}) = \sigma_b^2+\sigma_w^2(x-s)(x^{\prime}-s)$$

```python
def linear_kernel(X, X_prime, bias_var, slope_var, shift):
    """A linear kernel

    Parameteres
    -----
    X: array-like
        Prior Data matrix of shape (n_samples, n_features)

    X_prime: array-like
        New data matrix of shape (n_new_samples, n_features)

    bias_var: float
        Bias variance to control the variance from the origin

    slope_var: float
        Slope variance to control the variance of the slope of the line

    shift: float
        Offset from available data

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    dot_prod = (X - shift) @ (X_prime - shift).T
    return bias_var ** 2 + slope_var ** 2 * dot_prod
```

![Linear kernel](/images/gaussian_process_files/linear_cov_func.svg)

- **Polynomial Kernel**: This is a stationary kernel. The linear kernel introduced above is a special case of polynomial kernel when the exponent is one. Hence the polynomial function is given by

$$K(x, x^{\prime}) = \sigma_b^2+\sigma_w^2{(x-s)(x^{\prime}-s)}^{v}$$

where $v$ is the exponent of the dot product term.

```python
def polynomial_kernel(X, X_prime, bias_var, slope_var, shift, exponent):
    """A polynomial kernel

    Parameteres
    -----
    X: array-like
        Prior Data matrix of shape (n_samples, n_features)

    X_prime: array-like
        New data matrix of shape (n_new_samples, n_features)

    bias_var: float
        Bias variance to control the variance from the origin

    slope_var: float
        Slope variance to control the variance of the slope of the line

    shift: float
        Offset from available data

    exponent: float
        The exponent of the dot product term

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    dot_prod = (X - shift) @ (X_prime - shift).T
    return bias_var ** 2 + slope_var ** 2 * dot_prod ** exponent
```

![Polynomial Kernel](/images/gaussian_process_files/poly_cov_func.svg)

- **Rational Quadratic**: A Rational Quadratic Kernel is a stationary kernel and a generalization over the Exponential Quadratic kernel introduced above. It is parameterized by amplitude ($\sigma$), length_scale ($l$), and scale mixture rate ($\mathcal{M}$). Its functional form can be given by

$$K(x, x^{\prime}) = \sigma^2\left(1+\frac{||x-x^{\prime}||^2}{2\mathcal{M}l^2}\right)^{-\mathcal{M}}$$

This kernel acts like an Exponentiated Quadratic Kernel when the scale mixture rate parameter $\mathcal{M}$ approaches infinity.

```python
def rational_quadratic_kernel(X, X_prime, length_scale, amplitude, scale_mixture_rate):
    """The rational quadratic kernel

    Parameters
    -----
    X: array-like
        Prior data matrix of shape (n_samples, n_features)

    X_prime: array_like
        New data matrix of shape (n_new_samples, n_features)

    length_scale: float
        The `l` parameter in the equation

    amplitude: float
        The `sigma^2` parameter in the equation

    scale_mixture_rate: float
        The `M` parameter in the equation

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    l2 = np.sum(X ** 2, 1).reshape(-1, 1) + (np.sum(X_prime ** 2, 1).reshape(1, -1) - 2 * X @ X_prime.T)
    return amplitude * (1. + 0.5 * l2 / (scale_mixture_rate * length_scale ** 2)) ** (-scale_mixture_rate)
```

![Rational Quadratic Kernel](/images/gaussian_process_files/rat_quad_cov_func.svg)

- **Exponentiated Sine Squared Kernel**: This is a periodic kernel and is known to model periodic data that is often a case for time series tasks. It is paramerized by amplitude ($\sigma$), length scale ($l$), and period ($T$). It functional form can be given as

$$K(x, x^{\prime}) = \sigma^2\exp\left(\frac{-2\sum_{k=0}^{N}\sin\left(\frac{\pi|x_k-x^{\prime}_k|}{T}\right)}{l^2}\right)$$

```python
def exponentiated_sine_squared(X, X_prime, length_scale, amplitude, period):
    """The Exponentiated Sine Squared Kernel

    Parameters
    -----
    X: array-like
        Prior data matrix of shape (n_samples, n_features)

    X_prime: array_like
        New data matrix of shape (n_new_samples, n_features)

    length_scale: float
        The `l` parameter in the equation

    amplitude: float
        The `sigma^2` parameter in the equation

    period: float
        The `T` parameter in the equation

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    sine_term = np.empty((X.shape[0], X_prime.shape[0]))
    for i, x_prime in enumerate(X_prime):
        sine_term[:, i] = np.sin(np.pi * np.sum(np.abs(X - x_prime), axis=1) / period)
    return amplitude * np.exp( -2 * sine_term / length_scale ** 2 )
```

![Exponentiated Sine Squared Kernel](/images/gaussian_process_files/exp_sin_sq_cov_func.svg)

- **Matern One-Halves Kernel**: The matern family is a generalization over the Exponential Quadratic kernels. It adds one degree of freedom nu ($\nu$) which controls the smoothness of the kernel. It can take values 0.5 (1/2), 1.5 (3/2) and 2.5 (5/2). For all other values the kernel is not continuous and hence not diffrentiable which make it analytically impossible to tune the heyperparameters. For Matern 1/2 kernel, the value of smoothness paramter ($\nu$) is 1/2. This kenel is also known as Laplacian kernel. It becomes Exponentiated Quadratic kernel when $\nu \to \inf$. Its functional form is shown below.

$$K(x, x^{\prime}) = \sigma^2\exp\left(\frac{||x - x^{\prime}||}{l}\right)$$

```python
def matern_one_halves_kernel(X, X_prime, length_scale, amplitude):
    """Matern kernel with nu=1/2

    Parameters
    -----
    X: array-like
        Prior data matrix of shape (n_samples, n_features)

    X_prime: array_like
        New data matrix of shape (n_new_samples, n_features)

    length_scale: float
        The `l` parameter in the equation

    amplitude: float
        The `sigma^2` parameter in the equation

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    l2 = np.sum(X ** 2, 1).reshape(-1, 1) + (np.sum(X_prime ** 2, 1).reshape(1, -1) - 2 * X @ X_prime.T)
    return amplitude * np.exp( l2 ** 0.5 / length_scale )
```

![Matern 1/2 Kernel](/images/gaussian_process_files/matern_12_cov_func.svg)

- **Matern Three-Halves kernel**: Another member of Matern family with the value of $\nu=\frac{3}{2}$. Its functional form is given below.

$$K(x, x^{\prime}) = \sigma(1+z)\exp(-z)$$

where

$$z = \frac{\sqrt{3}||x-x^{\prime}||}{l}$$

```python
def matern_three_halves_kernel(X, X_prime, length_scale, amplitude):
    """Matern kernel with nu=3/2

    Parameters
    -----
    X: array-like
        Prior data matrix of shape (n_samples, n_features)

    X_prime: array_like
        New data matrix of shape (n_new_samples, n_features)

    length_scale: float
        The `l` parameter in the equation

    amplitude: float
        The `sigma^2` parameter in the equation

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    l2 = np.sum(X ** 2, 1).reshape(-1, 1) + (np.sum(X_prime ** 2, 1).reshape(1, -1) - 2 * X @ X_prime.T)
    z = (3 * l2) ** 0.5 / length_scale
    return amplitude * (1 + z) * np.exp(-z)
```

![Matern 3/2 Kernel](/images/gaussian_process_files/matern_32_cov_func.svg)

- **Matern Five-Halves kernel**: Another member of Matern family with the value of $\nu=\frac{5}{2}$. Its functional form is given below.

$$K(x, x^{\prime}) = \sigma\left(1+z+\sqrt{z^2}{3}\right)\exp(-z)$$

where

$$z = \frac{\sqrt{5}||x-x^{\prime}||}{l}$$

```python
def matern_three_halves_kernel(X, X_prime, length_scale, amplitude):
    """Matern kernel with nu=5/2

    Parameters
    -----
    X: array-like
        Prior data matrix of shape (n_samples, n_features)

    X_prime: array_like
        New data matrix of shape (n_new_samples, n_features)

    length_scale: float
        The `l` parameter in the equation

    amplitude: float
        The `sigma^2` parameter in the equation

    Returns
    -----
    A covariance matrix of shape (n_samples, n_new_samples)
    """
    l2 = np.sum(X ** 2, 1).reshape(-1, 1) + (np.sum(X_prime ** 2, 1).reshape(1, -1) - 2 * X @ X_prime.T)
    z = (5 * l2) ** 0.5 / length_scale
    return amplitude * (1 + z + z ** 2 / 3) * np.exp(-z)
```

![Matern 5/2 Kernel](/images/gaussian_process_files/matern_52_cov_func.svg)

- **Composite Kernel**: Two or more positive semi-definite kernels can be added, multiplied, or exponentiated to create a new positive semi-definite kernel. This property of kernel functions makes all sorts of different combinations open to model more and more complex data. Below are some plots showing different combinations of kernel functions inreoduced in the former sections.

### 4. Gaussian Processes

``to be continued``

### References

[1] Rasmussen C.E. (2004) Gaussian Processes in Machine Learning. In: Bousquet O., von Luxburg U., Rätsch G. (eds) Advanced Lectures on Machine Learning. ML 2003. Lecture Notes in Computer Science, vol 3176. Springer, Berlin, Heidelberg

[2] MacKay, David J. C.. “Introduction to Gaussian processes.” (1998).