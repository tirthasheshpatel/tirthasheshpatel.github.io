title: GSoC'20 Phase 2 Report
author: Tirth Patel
date: 2020-08-03
category: GSoC 2020
tags: gsoc 2020
alias: /gsoc2020/gsoc-phase-2-summary/index.html, /gsoc2020/gsoc-phase-2-summary.html
---

### Tasks Completed

**Week 5**: [#303: Fix docs in GP submodule](https://github.com/pymc-devs/pymc4/pull/303)

Small fixes in the documentation suite of GP submodule so that `pytest` passes

**Week 6**: [#304: fix docs of kernel functions](https://github.com/pymc-devs/pymc4/pull/304)

Fixed typos in the documentation suite of kernel functions. One inconsistency still remains
which I have planned to remove later (probably in the last phase of GSoC).

**Week 7**: [#305: add jitter argument to gp models](https://github.com/pymc-devs/pymc4/pull/305)

Added `jitter` argument to the GP models using which, the user can choose the amount of deterministic
noise to add in the diagonal of the evaluated covariance functions. This helps to deal with cholesky
decomposition errors that commonly occur with 32-bit floating-point numbers.

**Week 8**: [#309 [WIP] ENH: add MarginalGP model](https://github.com/pymc-devs/pymc4/pull/309)

This is the major PR of the phase. It adds the Marginal GP model to the gp submodule. I have also
written a Gaussian Process Latent Variable Model example using Marginal GP and got good results.

The following tasks are to be completed before the next phase begins:
    - Create a predict and predictt method using which point predictions are possible.
    - Create a notebook with an example of the GP-LVM model.
    - Enhance the documentation suite of the Marginal GP model.

## Conclusion

This phase has been another beast for me! I am running a little late on my schedule but I will finish all the listed goals (hopefully) by the end of phase 3. Nevertheless, I have enjoyed my work on GPs from the last couple of months and making enhancements to the GP submodule in the PyMC4 project.
