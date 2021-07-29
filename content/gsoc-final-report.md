title: GSoC'20 Final Report
author: Tirth Patel
date: 2020-08-26
category: GSoC 2020
tags: gsoc 2020
alias: /gsoc2020/gsoc-final-report/index.html, /gsoc2020/gsoc-final-report.html

### Major Pull Requests

- [pymc-devs/pymc4#235][1]: Implemented the basic API structure of Gaussian Processes. Implemented Latent Gaussian Process Model. Created a notebook explaining them.
- [pymc-devs/pymc4#272][2]: Implemented/Refactored the covariance functions API for Gaussian Processes. Introduced multiple new features on top of Tensorflow Probability's PSD API.
- [pymc-devs/pymc4#285][3]: Implemented 16 new covariance functions. Created a notebook explaining each of them.
- [pymc-devs/pymc4#309][4]: Implemented Marginal Gaussian Process. Created a tutorial notebook for it.

The above PRs are core to my GSoC project. I started with [pymc-devs/pymc4#235][1] before the official coding round. This PR proposed a basic API for performing GP Modelling in PyMC4. I also implemented a Latent GP model on top of it. It closely follows PyMC3's GP API.

During the first few weeks, I worked on refactoring the Covariance/Kernel functions API in [pymc-devs/pymc4#272][2]. This PR introduced multiple features on top of Tensorflow Probability's PSD kernels API.

By the end of Phase 1 and the commencement of Phase 2, I proposed [pymc-devs/pymc4#285][3]. This PR implemented 8 stationary, 2 periodic, 5 non-stationary, and 2 special kernel functions. I also implemented a huge notebook explaining each of these kernel functions with the help of Bill Engels and Alex Andorra. It was completed and merged by the end of the second phase.

During Phase 3, I started working on the Marginal GP Model in [pymc-devs/pymc4#309][4]. I also implemented a notebook explaining the Marginal GP Model using the GP-LVM example (Gaussian Process Latent Variable Model) and got good results with the Variational Inference API.

### Other PRs

- [pymc-devs/pymc4#296][5]: Fix the `MvNormal` distribution and add reparametrize to GP.
- [pymc-devs/pymc4#303][6]: Fix docs in GP submodule.
- [pymc-devs/pymc4#304][7]: Fix docs in kernel functions API.
- [pymc-devs/pymc4#305][8]: Add `jitter` argument to GP models.

### Blogs Written

Blog Page : [https://tirthasheshpatel.github.io/gsoc2020/](https://tirthasheshpatel.github.io/gsoc2020/)

- [GSoC'20 Phase 2 Report (August 3, 2020)](https://tirthasheshpatel.github.io/gsoc2020/gsoc-phase-2-summary)
- [GSoC'20 Phase 1 Report (July 3, 2020)](https://tirthasheshpatel.github.io/gsoc2020/gsoc-phase-1-summary)
- [GSoC Week 4 - Writing Notebooks on GP Kernels! (July 1, 2020)](https://tirthasheshpatel.github.io/gsoc2020/writing-notebooks-on-gp-kernels)
- [GSoC Week 2 and Week 3 - Implementing GP Kernels! (June 21, 2020)](https://tirthasheshpatel.github.io/gsoc2020/implementing-gp-kernels)
- [GSoC Week 1 - Latent GP model and Covariance functions! (June 4, 2020)](https://tirthasheshpatel.github.io/gsoc2020/latent-gp-model-and-covariance-functions)
- [Pre-GSoC Period - I am excited to get started! (May 8, 2020)](https://tirthasheshpatel.github.io/gsoc2020/pre-gsoc-period-i-am-excited-to-get-started)

### Tutorials Written

- [Marginal GP model in PyMC4 (August 10, 2020)](https://tirthasheshpatel.github.io/gsoc2020/marginal-gp-model-in-pymc4)
- [Kernels for GP Modelling in PyMC4. (July 3, 2020)](https://tirthasheshpatel.github.io/gsoc2020/kernels-for-gp-modelling-in-pymc4)
- [Getting started with Gaussian Process in PyMC4 (March 16, 2020)](https://tirthasheshpatel.github.io/gsoc2020/getting-started-with-gaussian-process-in-pymc4)

### Some things I noticed

- GPs work best with Variational Inference.
- **Always** use float64 datatype!
- Sampling fails on large datasets and large models! (probably because tensorflow probability doesn't do mass matrix adaptation)
- Marginal GP is hard to infer using sampling...

# Goals accomplished (as per proposal)

### What's implemented

- Constant Kernel
- White Noise Kernel
- Exponential Quadratic Kernel
- Rational Quadratic Kernel
- Matern 1/2 Kernel
- Matern 3/2 Kernel
- Matern 5/2 Kernel
- Linear Kernel
- Polynomial Kernel
- Exponential Kernel
- Exponential Sine Squared Kernel
- Scaled Covariance Kernel
- Gibbs Kernel
- Warped Input Kernel
- Additive Kernels
- Multiplicative Kernels
- Docs and Tests for all the Kernels
- Notebook explaining all the Kernel functions
- Latent GP Model
- Latent GP example notebook
- Marginal GP Model
- Marginal GP example notebook
- Docs and Tests for GP Models

### What's left

- Kronecker Kernels
- ARD API for Kernel functions
- Co-region Kernels (for Multi-Output GPs)
- Student's T Process (WIP)
- Sparse Marginal GP
- Kronecker GPs
- Some more GP examples present in PyMC3

### Some Potential Post GSoC Projects

- Multi-Output GPs
- Bayes Optimization Example Notebook
- Black Box Matrix Multiplication GP

[1]: https://github.com/pymc-devs/pymc4/pull/235
[2]: https://github.com/pymc-devs/pymc4/pull/272
[3]: https://github.com/pymc-devs/pymc4/pull/285
[4]: https://github.com/pymc-devs/pymc4/pull/309
[5]: https://github.com/pymc-devs/pymc4/pull/296
[6]: https://github.com/pymc-devs/pymc4/pull/303
[7]: https://github.com/pymc-devs/pymc4/pull/304
[8]: https://github.com/pymc-devs/pymc4/pull/305
