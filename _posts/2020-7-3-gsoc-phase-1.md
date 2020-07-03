---
layout: post
title: GSoC'20 Phase 1 - A complete overview!
subtitle: A complete and detailed overview of the work done during Phase 1 of GSoC 2020
gsoc_post: true
tags: [GSoC 2020]
comments: true
permalink: /gsoc2020/gsoc-phase-1-ends
---

### Tasks Completed

Week 1: [#235: [MRG] ENH: Add Basic Gaussian Process Interface](https://github.com/pymc-devs/pymc4/pull/235)
  - [x] &nbsp;&nbsp;&nbsp; Create a base class for kernel/covariance functions.
  - [x] &nbsp;&nbsp;&nbsp; Create an API for combining covariance functions.
  - [x] &nbsp;&nbsp;&nbsp; Create a base class for mean functions.
  - [x] &nbsp;&nbsp;&nbsp; Create an API for combining mean functions.
  - [x] &nbsp;&nbsp;&nbsp; Create a base class for GP Models.
  - [x] &nbsp;&nbsp;&nbsp; Implement `ExponentiatedQuadratic` kernel function.
  - [x] &nbsp;&nbsp;&nbsp; Implement `Zero` and `Constant` mean functions.
  - [x] &nbsp;&nbsp;&nbsp; Implement `LatentGP` model using the mean and covariance functions.
  - [x] &nbsp;&nbsp;&nbsp; Create a notebook explaining the design and working of the basic API.
  - [x] &nbsp;&nbsp;&nbsp; Add duck typing for easy maintenance.
  - [x] &nbsp;&nbsp;&nbsp; Add a documentation suite for the subpackage.
  - [x] &nbsp;&nbsp;&nbsp; Add a test suite for GP Models, Mean functions, and Kernel functions.

{: .box-warning}
**Blog**: [GSoC Week 1 - Latent GP model and Covariance functions!](https://tirthasheshpatel.github.io/gsoc2020/latent-gp-model-and-covariance-functions)

Week 2: [#272: ENH: add constant and white noise kernel, fix docs and tests](https://github.com/pymc-devs/pymc4/pull/272)
  - [x] &nbsp;&nbsp;&nbsp; Add `Constant` kernel with batched input support.
  - [x] &nbsp;&nbsp;&nbsp; Add `WhiteNoise` kernel with batched input support.
  - [x] &nbsp;&nbsp;&nbsp; Add `diag` parameter to only evaluate the diagonal of the full covariance matrix.
  - [x] &nbsp;&nbsp;&nbsp; Add `feature_ndims` parameter to the covariance base class to allow tensors with multiple feature dimensions as inputs.
  - [x] &nbsp;&nbsp;&nbsp; Optimize the base class for faster combination of kernel functions.
  - [x] &nbsp;&nbsp;&nbsp; Refactor the test suite of the entire submodule.
    - [x] &nbsp;&nbsp;&nbsp; Better tests for covariance functions.
    - [x] &nbsp;&nbsp;&nbsp; Create a template for testing of new covariance functions.
    - [x] &nbsp;&nbsp;&nbsp; Create a template for testing of new mean functions.
  - [x] &nbsp;&nbsp;&nbsp; Refactor the documentation suite according to the `pydocstyle` linting rules.

{: .box-warning}
**Blog**: [GSoC Week 2 and Week 3 - Implementing GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/implementing-gp-kernels)

Week 3: [#285 ENH: add all covariance functions for gp from PyMC3](https://github.com/pymc-devs/pymc4/pull/285)
  - Create following Kernel functions with tests and documentation for each of them:
    - [x] &nbsp;&nbsp;&nbsp; `RatQuad`
    - [x] &nbsp;&nbsp;&nbsp; `Exponential`
    - [x] &nbsp;&nbsp;&nbsp; `Matern52`
    - [x] &nbsp;&nbsp;&nbsp; `Matern32`
    - [x] &nbsp;&nbsp;&nbsp; `Matern12`
    - [x] &nbsp;&nbsp;&nbsp; `Linear`
    - [x] &nbsp;&nbsp;&nbsp; `Polynomial`
    - [x] &nbsp;&nbsp;&nbsp; `Cosine`
    - [x] &nbsp;&nbsp;&nbsp; `Periodic`
    - [x] &nbsp;&nbsp;&nbsp; `Gibbs`
    - [x] &nbsp;&nbsp;&nbsp; `WarpedInput`
    - [x] &nbsp;&nbsp;&nbsp; ~~`Coregion`~~
    - [x] &nbsp;&nbsp;&nbsp; `ScaledCov`
    - [x] &nbsp;&nbsp;&nbsp; ~~`Kron`~~

{: .box-warning}
**Blog**: [GSoC Week 2 and Week 3 - Implementing GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/implementing-gp-kernels)

Week 4: [#285 ENH: add all covariance functions for gp from PyMC3](https://github.com/pymc-devs/pymc4/pull/285)
  - Add notebook for the following Kernel Functions:
    - [x] &nbsp;&nbsp;&nbsp; Basic Kernel API.
    - [x] &nbsp;&nbsp;&nbsp; `RatQuad` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Exponential` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Matern52` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Matern32` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Matern12` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Linear` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Polynomial` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Cosine` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Periodic` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `Gibbs` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `WarpedInput` kernel.
    - [x] &nbsp;&nbsp;&nbsp; `ScaledCov` kernel.

{: .box-warning}
**Blog**: [GSoC Week 4 - Writing Notebooks on GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/writing-notebooks-on-gp-kernels)

### Week 1

![PR 235 Overview](/images/gsoc_files/pr-235.png)

### Week 2

![PR 272 Overview](/images/gsoc_files/pr-272.png)

### Week 3 and 4

![PR 285 Overview](/images/gsoc_files/pr-285.png)
