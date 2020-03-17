---
layout: post
title: Project Proposal - Adding Gaussian Process in PyMC4
category: GSoC 2020
tags: [gsoc, gaussian process, machine learning]
---

### Introduction

I have decided to work with PyMC4 on the project statement **Adding Gaussian Process in PyMC4** for *GSoC 2020*. In this article, I present my ideas for development during the community bonding period.

> My goal for GSoC 2020 is to implement, test, and maintain a higher-level API for Gaussian Processes in PyMC4 using TensorFlow and TensorFlow Probability and write tutorials/articles and notebooks explaining their usage.

I am Tirth Patel, an undergraduate computer science student at Nirma University, Ahmedabad, India. I have previously worked on many Machine Learning projects and particularly on Bayesian Methods for Machine Learning like Facial Composites that use Variational AutoEncoders and Gaussian Processes to generate faces of a suspect! My motivation to work with PyMC is driven by my passion for open-source projects and Machine Learning.

### Project Proposal

Gaussian Processes are well established in both temporal and spatial interpolation tasks. During summer I aim to create a ``gp`` submodule with the following functionality.

#### 1. Mean Functions

The choice of mean function is highly dependent on the user and his/her knowledge about the data at hand. This leads to a design choice of the base class that makes it convenient for the user to flexibly create new mean functions of his/her choice. Such a base class is shown below:

```python
class Mean:
    r"""Base Class for all the mean functions in GP."""

    def __init__(self, feature_ndims=1):
        self.feature_ndims = feature_ndims

    def __call__(self, X):
        raise NotImplementedError("Your mean function should override this method")

    def __add__(self, mean2):
        return MeanAdd(self, mean2)

    def __mul__(self, mean2):
        return MeanProd(self, mean2)
```

The above base class can be easily used to create new mean functions by just overriding the ``__call__`` and optionally the ``__init__`` method. Moreover, mean functions created using this base are additive and multiplicative and hence provide the user maximum flexibility to create and combine all sorts of mean functions!

The mean functions that I aim to create are:

- [Zero mean function](https://github.com/pymc-devs/pymc3/blob/6c5254fe8fe7cbcf7fdf775db67191beb1dd7c90/pymc3/gp/mean.py#L42)
- [Constant mean function](https://github.com/pymc-devs/pymc3/blob/6c5254fe8fe7cbcf7fdf775db67191beb1dd7c90/pymc3/gp/mean.py#L51)
- [Linear mean function](https://github.com/pymc-devs/pymc3/blob/6c5254fe8fe7cbcf7fdf775db67191beb1dd7c90/pymc3/gp/mean.py#L69)

#### 2. Covariance Functions

The computational speed and inference quality are generally very sensitive to the choice of covariance function and hence have to be designed carefully. One more necessity of the covariance functions is that they must be positive semi-definite. The ``tfp.math.pad_kernels`` module contains a lot of covariance functions to work with. We can wrap these covariance functions in our base class as shown below

```python
class Covariance:
    r"""Base class of all Covariance functions for Gaussian Process"""

    def __init__(self, feature_ndims=1, **kwargs):
        # TODO: Implement the `diag` parameter as in PyMC3.
        self._feature_ndims = feature_ndims
        self._kernel = self._init_kernel(feature_ndims=self.feature_ndims, **kwargs)
        if self._kernel is not None:
            # wrap the kernel in FeatureScaled kernel for ARD
            self._scale_diag = kwargs.pop("scale_diag", 1.0)
            self._kernel = tfp.math.psd_kernels.FeatureScaled(
                self._kernel, scale_diag=self._scale_diag
            )

    @abstractmethod
    def _init_kernel(self, feature_ndims, **kwargs):
        raise NotImplementedError("Your Covariance class should override this method")

    def __call__(self, X1, X2, **kwargs):
        return self._kernel.matrix(X1, X2, **kwargs)

    def evaluate_kernel(self, X1, X2, **kwargs):
        """Evaluate kernel at certain points

        Parameters
        ----------
        X1 : tensor, array-like
            First point(s)
        X2 : tensor, array-like
            Second point(s)
        """
        return self._kernel.apply(X1, X2, **kwargs)

    def __add__(self, cov2):
        return CovarianceAdd(self, cov2)

    def __mul__(self, cov2):
        return CovarianceProd(self, cov2)

    @property
    def feature_ndims(self):
        return self._feature_ndims
```

The advantage of this design is that the covariance functions are additive and multiplicative which gives the user high flexibility to experiment with different combinations of covariance functions. Moreover, [ARD (Automatic Relevance Determination)](http://www.gaussianprocess.org/gpml/chapters/RW5.pdf) is possible using ``tfp.math.psd_kernel.FeatureScaled`` kernel and hence user get one more degree of freedom which is to set and infer the ``scale_diag`` parameter during inference using hir/her favorite inference methods like MCMC or ``find_MAP``.

The only disadvantage of this design is that it is inconvenient for the user to create his/her covariance function as it requires the ``self._kernel`` to be an instance of ``tfp.math.psd_kernels.PositiveSemidefiniteKernel``. As a result, the user has to first inherit ``tfp.math.psd_kernels.PositiveSemidefiniteKernel`` and then wrap it in ``pm.gp.Covariance`` base.

Following are the covariance functions that I aim to create:

- Stationary kernels
  - Exponentiated Quadratic kernel
  - Rational Quadratic kernel
  - Matern 1/2 kernel
  - Matern 3/2 kernel
  - Matern 5/2 kernel
  - Polynomial kernel
  - Linear kernel
- Transformed Kernels
  - Kumaraswamy Transformed kernel
- Non Stationary Kernels
  - Exponentiated Sine Squared Kernel
- Other Kernels
  - Kroneker Product Kernel

**References**

- [Introduction to Gaussian Processes by David J.C. Mackay](http://www.inference.org.uk/mackay/gpB.pdf)
- [Random Walk Kernels and Learning Curves for Gaussian Process Regression on Random Graphs](https://arxiv.org/pdf/1211.1328.pdf)
- [Nonstationary Covariance Functions for Gaussian Process Regression](https://papers.nips.cc/paper/2350-nonstationary-covariance-functions-for-gaussian-process-regression.pdf)

#### 3. Gaussian Process Models

GP models are extensively used for both spatial and temporal predictions and have to be carefully designed. I propose to develop a base that can easily be used to create new GP models. GP models created on this base are very simple and naive to use.

```python
class BaseGP:
    def __init__(self, cov_fn, mean_fn=Zero(1)):
        if mean_fn.feature_ndims != cov_fn.feature_ndims:
            raise ValueError("The feature_ndims of mean and covariance functions should be equal")
        self.feature_ndims = mean_fn.feature_ndims
        self.mean_fn = mean_fn
        self.cov_fn = cov_fn

    def prior(self, name, X, **kwargs):
        raise NotImplementedError

    def conditional(self, name, Xnew, **kwargs):
        raise NotImplementedError

    def predict(self, name, Xnew, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, **kwargs):
        raise NotImplementedError

    def __add__(self, model2):
        ...
```

The above design can be used to create all sorts of GP models proposed in the literature and these models are made additive so the user can experiment with different combinations of GP models of his/her choice and end up with a model most suitable for his/her data.

I propose to add the following GP models from their respective references:

1. ``LatentGP``: Noiseless GP model that can infer the underlying latent function.
    - [PyMC3 Source](https://github.com/pymc-devs/pymc3/blob/6c5254fe8fe7cbcf7fdf775db67191beb1dd7c90/pymc3/gp/gp.py#L65)
2. ``MarginalGP``: Noisy GP models that can be used for regression, classification, and prediction.
    - [Gaussian Process in Machine Learning by Rasmussen et. al.](https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers05/rasmussen_gps_in_ml.pdf)
3. ``StudentTP``: A alternative to gaussian processes using the t distribution
    - [Student-t Processes as Alternatives to Gaussian Processes](https://www.cs.cmu.edu/~andrewgw/tprocess.pdf)
    - [PyMC3 Source](https://github.com/pymc-devs/pymc3/blob/6c5254fe8fe7cbcf7fdf775db67191beb1dd7c90/pymc3/gp/gp.py#L226)
4. ``MarginalSparse``: I will implement all the sparse models for GP alongwith those implemented in PyMC3.
    - [A Unifying View of Sparse Approximate Gaussian Process Regression](http://www.jmlr.org/papers/volume6/quinonero-candela05a/quinonero-candela05a.pdf)
    - [Sparse Spectrum Gaussian Process Regression](http://quinonero.net/Publications/lazaro-gredilla10a.pdf)
    - [Fast Forward Selection to Speed Up Sparse Gaussian Process Regression](https://pdfs.semanticscholar.org/1455/12a08a7cd79a0efb1f0503ddc6a4e4ef02dc.pdf)
    - [PyMC3 Source](https://github.com/pymc-devs/pymc3/blob/6c5254fe8fe7cbcf7fdf775db67191beb1dd7c90/pymc3/gp/gp.py#L572)
5. ``VariationalGP``: I aim to implement Variational Gaussian Process models that can be used for classification tasks.
    - [Variational Gaussian Process Classifiers](https://ieeexplore.ieee.org/document/883477)
    - [TFP source](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/variational_gaussian_process.py)
6. ``MarginalKronGP``: Marginal Gaussian process whose covariance is a tensor product kernel.
    - [PyMC3 Source](https://github.com/pymc-devs/pymc3/blob/6c5254fe8fe7cbcf7fdf775db67191beb1dd7c90/pymc3/gp/gp.py#L965)

#### Working Prototype

I have created a working prototype of the ``LatentGP`` model that can be used to perform regression and the parameters can be inferred using MCMC. This prototype is [available here on my branch](https://github.com/tirthasheshpatel/pymc4/tree/xlpatch/gp) of the project and the [here is the corresponding PR on GitHub](https://github.com/pymc-devs/pymc4/pull/235). The results and the code to reproduce them are shown below.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pymc4 as pm
import matplotlib.pyplot as plt
import arviz as az
%matplotlib inline

# set the seed
np.random.seed(None)

n = 100 # The number of data points
X = np.linspace(0, 10, n)[:, None] # The inputs to the GP, they must be arranged as a column vector
n_new = 200
X_new = np.linspace(0, 15, n_new)[:,None]

# Define the true covariance function and its parameters
l_true = 3.
cov_func = pm.gp.cov.ExpQuad(amplitude=1., length_scale=l_true, feature_ndims=1)

# A mean function that is zero everywhere
mean_func = pm.gp.mean.Zero()

# The latent function values are one sample from a multivariate normal
f_true = np.random.multivariate_normal(mean_func(X).numpy(),
                                       cov_func(X, X).numpy() + 1e-6*np.eye(n), 1).flatten()

# The observed data is the latent function plus a small amount of T distributed noise
# The degrees of freedom is `nu`
ν_true = 10.0
y = f_true + 0.1*np.random.standard_t(ν_true, size=n)

## Plot the data and the unobserved latent function
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
ax.plot(X, f_true, "dodgerblue", lw=3, label="True f")
ax.plot(X, y, 'ok', ms=3, label="Data")
ax.set_xlabel("X"); ax.set_ylabel("y"); plt.legend()
```

![svg](/images/gaussian_process_files/gaussian_process_16_1.svg)

```python
@pm.model
def latent_gp_model(X, y, X_new):
    """A latent GP model with unknown length_scale and student-t noise

    Parameters
    ----------
    X: np.ndarray, tensor
        The prior data
    y: np.ndarray, tensor
        The function corresponding to the prior data
    X_new: np.ndarray, tensor
        The new data points to evaluate the conditional

    Returns
    -------
    y_: tensor
        A random sample from inferred function and its noise.
    """
    # We define length_scale of RBF kernel as a random variable
    l = yield pm.HalfCauchy("l", scale=5.)
    # We can now define a GP with mean as zeros and covariance function
    # as RBF kernel with ``length_scale=l`` and ``amplitude=1``
    cov_fn = pm.gp.cov.ExpQuad(length_scale=l, amplitude=1., feature_ndims=1)
    latent_gp = pm.gp.LatentGP(cov_fn=cov_fn)
    # f is the prior and f_pred is the conditional which we discussed in theory section
    f = yield latent_gp.prior("f", X=X)
    f_pred = yield latent_gp.conditional("f_pred", X_new, given={'X': X, 'f': f})
    # finally, we model the noise and our outputs.
    ν = yield pm.Gamma("ν", concentration=1., rate=0.1)
    y_ = yield pm.StudentT("y", loc=f, scale=0.1, df=ν, observed=y)
    return y_


gp = latent_gp_model(X, y, X_new)
trace = pm.sample(gp, num_samples=1000, num_chains=1)

az.plot_trace(trace, lines=lines, var_names=["latent_gp_model/l", "latent_gp_model/ν"])
```

![svg](/images/gaussian_process_files/gaussian_process_20_1.svg)

```python
from pymc4.gp.util import plot_gp_dist
# plot the results
fig = plt.figure(figsize=(12,5)); ax = fig.gca()

# plot the samples from the gp posterior with samples and shading
plot_gp_dist(ax, np.array(trace.posterior["latent_gp_model/f"])[0], X)

# plot the data and the true latent function
plt.plot(X, f_true, "dodgerblue", lw=3, label="True f")
plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data")

# axis labels and title
plt.xlabel("X"); plt.ylabel("True f(x)")
plt.title("Posterior distribution over $f(x)$ at the observed values"); plt.legend()
```

![svg](/images/gaussian_process_files/gaussian_process_21_1.svg)

```python
pred_samples = pm.sample_posterior_predictive(gp, trace=trace, var_names=["latent_gp_model/f_pred"])
fig = plt.figure(figsize=(12,5)); ax = fig.gca()
plot_gp_dist(ax, pred_samples.posterior_predictive["latent_gp_model/f_pred"][0], X_new)
plt.plot(X, f_true, "dodgerblue", lw=3, label="True f")
plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data")
plt.xlabel("$X$"); plt.ylabel("True $f(x)$")
plt.title("Conditional distribution of $f_*$, given $f$"); plt.legend()
```

![svg](/images/gaussian_process_files/gaussian_process_24_1.svg)

***Note**: I have not implemented additive models and only implemented the full covariance functions without diagonal approximation. This work is only a prototype of the actual implementation.*

#### Potential for Future work

New mean and covariance functions can be added easily in the future. One of the most interesting projects possible based on my proposed implementation is Bayesian Optimization. Bayesian Optimization is used extensively in tuning hyperparameters in machine learning models and is highly based on Gaussian Processes. It has not been implemented in both PyMC3 and PyMC4 and would make a great future project. I will try to take this up during the community bonding period and look forward to it even after the community bonding period. Below are some reference papers that can be referred to implement Bayesian Optimization Algorithms:

- [Taking the Human Out of the Loop: A Review of Bayesian Optimization](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- [A Tutorial on Bayesian Optimization of Expensive Cost Functions, with Application to Active User Modeling and Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1012.2599.pdf)
- [BOA: The Bayesian Optimization Algorithm](http://www.medal-lab.org/hboa/boa.pdf)
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)

### Project Timeline

- **May 18 - June 19: Phase 1**
  - Week 1: May 18 - May 24
    - Create a few mean and covariance functions with their base classes.
    - Implement a fully functional Latent Gaussian Process Model and a base for other models.
  - Week 2: May 25 - May 31
    - Complete migrating all the covariance functions from PyMC3
    - Complete migrating all the mean functions from PyMC3.
  - Week 3: June 1 - June 7
    - Implement the ``MarginalGP`` model for basic regression and classification tasks.
    - Add fixtures and tests for all the covariance functions, mean functions, and GP models.
  - Week 4: June 8 - June 14
    - Get all the PRs ready for review.
  - June 15 - June 19
    - Mentors review the basic functional API and suggest changes
    - Make the changes and get all the PRs ready to merge.
- **June 20 - July 17: Phase 2**
  - Week 1: June 20 - June 26
    - Implement all the remaining mean and covariance functions mentioned in former sections
  - Week 2: June 27 - July 3
    - Implement the Student-t Process model.
    - Start implementing the sparse models.
  - Week 3: July 4 - July 10
    - Complete the implementation of sparse GP models.
    - Add tests and fixtures for all the new models and functions.
    - Get all the PRs ready for review.
  - Week 4: July 11 - July 17
    - Mentors review the API and suggest changes
    - Make the changes and get all the PRs ready to merge.
- **July 18 - August 24: Phase 3**
  - Week 1: July 18 - July 24
    - Implement the Variational GP model.
  - Week 2: July 25 - July 31
    - Start implementing all the remaining models from PyMC3 and other proposed models
    - Mentors review the PRs and suggest changes.
    - Make the changes and get all the PRs ready to merge.
  - Week 3: August 1 - August 7
    - Add functionality to all the implemented GP models and mean and covariance functions.
    - Complete all the test suites.
    - Get all the PRs ready for review
  - Final phase: August 7 - August 16
    - Mentors review the API and suggest changes
    - Make the changes and get all the PRs ready to merge.
- **August 17 - August 25: Final Evaluation**
  - Final work evaluation by mentors
  - Results

### Commitments and Conclusion

I aim to work full time for at least 60 to 70 hours a week on my project and write a blog on the work done every week.

I am working on a research paper with my university professor to which I am willing to give not more than 4 to 5 hours a week. It will not affect my performance on the project.

I will notify beforehand if I am not available during some time slot and compensate for the time lost afterward during the community bonding period. I will maintain and review future code contributions even after the community bonding period.

## PyMCheers!
