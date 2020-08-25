---
title: GSoC'20 Phase 1 Report
date: 2020-7-3
categories:
  - GSoC 2020
tags:
  - GSoC 2020
permalink: /gsoc2020/gsoc-phase-1-summary
---

### Tasks Completed

**Week 1**: [#235: [MRG] ENH: Add Basic Gaussian Process Interface](https://github.com/pymc-devs/pymc4/pull/235)


**Blog**: [GSoC Week 1 - Latent GP model and Covariance functions!](https://tirthasheshpatel.github.io/gsoc2020/latent-gp-model-and-covariance-functions)

  - [x] Create a base class for kernel/covariance functions.
  - [x] Create an API for combining covariance functions.
  - [x] Create a base class for mean functions.
  - [x] Create an API for combining mean functions.
  - [x] Create a base class for GP Models.
  - [x] Implement `ExponentiatedQuadratic` kernel function.
  - [x] Implement `Zero` and `Constant` mean functions.
  - [x] Implement `LatentGP` model using the mean and covariance functions.
  - [x] Create a notebook explaining the design and working of the basic API.
  - [x] Add duck typing for easy maintenance.
  - [x] Add a documentation suite for the subpackage.
  - [x] Add a test suite for GP Models, Mean functions, and Kernel functions.

**Week 2**: [#272: ENH: add constant and white noise kernel, fix docs and tests](https://github.com/pymc-devs/pymc4/pull/272)


**Blog**: [GSoC Week 2 and Week 3 - Implementing GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/implementing-gp-kernels)

  - [x] Add `Constant` kernel with batched input support.
  - [x] Add `WhiteNoise` kernel with batched input support.
  - [x] Add `diag` parameter to only evaluate the diagonal of the full covariance matrix.
  - [x] Add `feature_ndims` parameter to the covariance base class to allow tensors with multiple feature dimensions as inputs.
  - [x] Optimize the base class for faster combination of kernel functions.
  - [x] Refactor the test suite of the entire submodule.
    - [x] Better tests for covariance functions.
    - [x] Create a template for testing of new covariance functions.
    - [x] Create a template for testing of new mean functions.
  - [x] Refactor the documentation suite according to the `pydocstyle` linting rules.

**Week 3**: [#285 ENH: add all covariance functions for gp from PyMC3](https://github.com/pymc-devs/pymc4/pull/285)


**Blog**: [GSoC Week 2 and Week 3 - Implementing GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/implementing-gp-kernels)

  - [x] Create following Kernel functions with tests and documentation for each of them:
    - [x] `RatQuad`
    - [x] `Exponential`
    - [x] `Matern52`
    - [x] `Matern32`
    - [x] `Matern12`
    - [x] `Linear`
    - [x] `Polynomial`
    - [x] `Cosine`
    - [x] `Periodic`
    - [x] `Gibbs`
    - [x] `WarpedInput`
    - [x] ~~`Coregion`~~
    - [x] `ScaledCov`
    - [x] ~~`Kron`~~

**Week 4**: [#285 ENH: add all covariance functions for gp from PyMC3](https://github.com/pymc-devs/pymc4/pull/285)


**Blog**: [GSoC Week 4 - Writing Notebooks on GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/writing-notebooks-on-gp-kernels)

  - Add notebook for the following Kernel Functions:
    - [x] Basic Kernel API.
    - [x] `RatQuad` kernel.
    - [x] `Exponential` kernel.
    - [x] `Matern52` kernel.
    - [x] `Matern32` kernel.
    - [x] `Matern12` kernel.
    - [x] `Linear` kernel.
    - [x] `Polynomial` kernel.
    - [x] `Cosine` kernel.
    - [x] `Periodic` kernel.
    - [x] `Gibbs` kernel.
    - [x] `WarpedInput` kernel.
    - [x] `ScaledCov` kernel.

### Week 1

![PR 235 Overview](/images/gsoc_files/pr-235.png)

I laid down some of the most basic aspects of the API I was going to develop throughout the GSoC period in this pull request. It was well received by the PyMC team. The internals are quite different but the overall usage of the API is very similar to PyMC3. This has its own set of advantages and disadvantages. Some of which I list here:


Pros:
  - Easy to port from PyMC3 to PyMC4.
  - TFP computational backend for speed.


Cons:
  - Prior needs to be explicitly passed in the conditional method (though there is a workaround!)
  - Custom covariance functions are not straight-forward to make because the base class relies on TFP. So, the user has to first implement a TFP kernel and then wrap it using PyMC4's base class (though there is also a workaround for this).
  - `Conditional` method can't be separated otherwise, that variable will not be recorded. OTOH, It is highly expensive to compute the gradients of the conditional distribution making inference almost impossible on big datasets.

I have tried to address the cons the next few weeks by refactoring and rewriting some base classes.

### Week 2

![PR 272 Overview](/images/gsoc_files/pr-272.png)

This PR adds a new base class for combinations of kernels and introduces multiple new parameters in the kernel functions API. It also adds Constant and WhiteNoise kernel functions not present in TFP.

As these changes are pretty big, I cannot summarize them here. You can take a look at my blog post for week 2 where I give a detailed view of the changes and their use.

I also refactored the test suite so new tests can be added in just a few lines. The base class for wrapping TFP kernels is not very well defined and needs a lot of refactoring to be done.

### Week 3 and 4

![PR 285 Overview](/images/gsoc_files/pr-285.png)

I proposed [#285](https://github.com/pymc-devs/pymc4/pull/285) in Week 3 porting all the kernel functions from PyMC3 to PyMC4.

Week 4 was spent writing a notebook on the Kernel functions API. It was fun to work with [Bill Engels](https://github.com/bwengals) who developed the GP API for PyMC3. His reviews were very helpful in making the notebook robust and correcting some of my misunderstandings about the kernels I wrote! I aim to write an article on my blog page about the kernel functions API and how to port to PyMC4 from PyMC3.

### Conclusion!

![Alto's adventure Picture](/images/random/alto1.png)


**Wow! This month went by very fast and we already are in the second phase of GSoC! Good luck to other GSoC students with their projects! And thank you mentors for your continuous support!**
