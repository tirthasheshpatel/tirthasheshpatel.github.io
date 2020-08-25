---
title: GSoC Week 4 - Writing Notebooks on GP Kernels!
categories:
  - GSoC 2020
tags:
  - GSoC 2020
permalink: /gsoc2020/writing-notebooks-on-gp-kernels
---

### Overview of this week

I opened a huge PR in week 3, porting all the kernel functions from PyMC3 to PyMC4. This week, I continued my work on the PR to write a tutorial notebook on the kernel API. Thanks to [Bill Engels](https://github.com/bwengals) and [Alex Andorra](https://github.com/AlexAndorra) for reviewing the notebook multiple times and providing helpful suggestions!

The other task that I have completed (on my local branch) is to implement ARD on the kernels. I will probably propose a PR by the end of this week.

### Writing Notebooks

![Week 4 Amazing Plot](/images/gaussian_process_files/week_4_work.png)


[#285: ENH: add all covariance functions for gp from PyMC3](https://github.com/pymc-devs/pymc4/pull/285)

This was a WIP in week 3 because it was missing a notebook explaining the kernel functions API implemented so far. So, I continued and completed almost all my work on this PR. I was able to write a very robust and helpful notebook with the help of [Bill Engels](https://github.com/bwengals) and [Alex Andorra][https://github.com/AlexAndorra]. After going through multiple reviews, it has matured enough to be merged. The current implementation doesn't support ARD. I aim to complete it by the end of this week.

### Adding ARD to the implemented kernels

I refactored the existing base class and created a separate module for wrapping TFP kernels. This was necessary as TFP doesn't provide an API for performing ARD with **batched** tensors. TFP's implementation breaks when multiple batches are to be processed in parallel, making it impossible to add ARD by externally warping its kernels.

I don't like the resulting API too much and will look for workarounds during the next few weeks.

The only job remaining now is to implement a test suite for this new feature before proposing a PR.

### EndNote

I am almost done with the kernel functions API so I can start working on the GP models in the next phase. Everything went as planned this week which is amazing!

Now, I need to get ready for the second coding phase during which I have planned to implement GP models in my proposal! I am very excited about the second phase as it is going to be the core part of my project! Hope everything goes as planned!
