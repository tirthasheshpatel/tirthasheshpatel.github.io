---
layout: post
title: GSoC'20 Phase 1 -- A complete overview!
subtitle: A complete and detailed overview of the work done during Phase 1 of GSoC 2020
gsoc_post: true
tags: [GSoC 2020]
comments: true
permalink: /gsoc2020/gsoc-phase-1-ends
---

### Tasks Completed

Week 1: [#235: [MRG] ENH: Add Basic Gaussian Process Interface](https://github.com/pymc-devs/pymc4/pull/235)
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

{: .box-warning}
**Blog**: [GSoC Week 1 - Latent GP model and Covariance functions!](https://tirthasheshpatel.github.io/gsoc2020/latent-gp-model-and-covariance-functions)

Week 2: [#272: ENH: add constant and white noise kernel, fix docs and tests](https://github.com/pymc-devs/pymc4/pull/272)
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

{: .box-warning}
**Blog**: [GSoC Week 2 and Week 3 - Implementing GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/implementing-gp-kernels)

Week 3: [#285 ENH: add all covariance functions for gp from PyMC3](https://github.com/pymc-devs/pymc4/pull/285)
  - Create following Kernel functions with tests and documentation for each of them:
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

{: .box-warning}
**Blog**: [GSoC Week 2 and Week 3 - Implementing GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/implementing-gp-kernels)

Week 4: [#285 ENH: add all covariance functions for gp from PyMC3](https://github.com/pymc-devs/pymc4/pull/285)
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

{: .box-warning}
**Blog**: [GSoC Week 4 - Writing Notebooks on GP Kernels!](https://tirthasheshpatel.github.io/gsoc2020/writing-notebooks-on-gp-kernels)

### Week 1

![PR 235 Overview](/images/gsoc_files/pr-235.png)

### Week 2

![PR 272 Overview](/images/gsoc_files/pr-272.png)

### Week 3 and 4

![PR 285 Overview](/images/gsoc_files/pr-285.png)
