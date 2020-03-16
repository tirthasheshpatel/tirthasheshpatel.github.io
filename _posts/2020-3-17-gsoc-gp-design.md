---
layout: post
title: Gaussian Process in PyMC4
category: GSoC 2020
tags: [gsoc, gaussian process, machine learning]
---

### Introduction

I have decided to work with PyMC4 on the project statement **Adding Gaussian Process in PyMC4** for *GSoC 2020*. In this article, I present my ideas for development during the community bonding period.

> My goal for GSoC 2020 is to implement, test, and maintain a higher level API for Gaussian Processes in PyMC4 using TensorFlow and TensorFlow Probability and write tutorials/articles and notebooks explaining their usage.

I am Tirth Patel, an undergraduate computer science student at Nirma University, Ahmedabad, India. I have previously worked on many Machine Learning projects and particularly on Bayesian Methods for Machine Learning like Facial Composites that use Variational AutoEncoders and Gaussian Processes to generate faces of a suspect! My motivation to work with PyMC is driven by my passion for open-source projects and Machine Learning.

### Project Proposal

Gaussian Processes are well established in both temporal and spatial interpolation tasks. During summer I aim to create a ``gp`` submodule with following functinality.

#### 1. Mean Functions

The choice of mena function is highly dependent on the user and his/her knowledge about the data at hand. This leads to a design choice of the base class that makes it convenient for the user to flexibly create new mean functions of his/her choice. Such a base class is shown below:

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

The above base class can be easily used to create new mean functions by just overriding the ``__call__`` and optionaly the ``__init__`` method. Moreover, mean functions created using this base are additive and multiplicative and hence provide the user maximum flexibility to create and combine all sorts of mean functions!

The mean functions that I aim to create are:

- Zero mean function
- Constant mean function
- Linear mean function

#### 2. Covariance Functions

The computational speed and inference quality is generally very sensitive to the choice of covariance function and hence have to be designed carefully. One more necessity for of the covariance functions is that they must be positive semi-definite. The ``tfp.math.pad_kernels`` module contains a lot of covariance functions to work with. We can wrap these covariance functions in own own base class as shown below

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

The advantages of this design is that the covariance functions are additive and multiplicative which gives the user high flexibility to experiment with different combinations of covariance functions. Moreover, [ARD (Automatic Relevence Determination)](http://www.gaussianprocess.org/gpml/chapters/RW5.pdf) is possible using ``tfp.math.psd_kernel.FeatureScaled`` kernel.

The only disadvantage of this design is that it is inconvinient for the user to create his/her own covariance function as it requires the ``self._kernel`` to be an instance of ``tfp.math.psd_kernels.PositiveSemidefiniteKernel``. As a result, the user has to first inherit ``tfp.math.psd_kernels.PositiveSemidefiniteKernel`` and then wrap it in ``pm.gp.Covariance`` base.
