---
title: Week -3 of GSoC 2020 with PyMC4
subtitle: This blog articulates my ideas to implement Gaussian Processes in PyMC4
gsoc_post: true
image: /images/gsoc-logo.png
gh-repo: tirthasheshpatel/pymc4
gh-badge: [star, fork, follow]
tags: [GSoC 2020]
comments: true
---

#### Abstract

*My goal for GSoC 2020 is to implement, test, and maintain a higher-level API for Gaussian Processes in PyMC4 using TensorFlow and TensorFlow Probability and write tutorials/articles and notebooks explaining their usage.*

*My work consists of implementing many Gaussian Process Models including Student's T Process Model and writing optimization methods like Neuton's Method and Powell's Method to find the maximum a-posteriori of the models that can be used to infer parameters of the model. My goal is also to implement at least one approximation technique when full GP modelling becomes impractical.*

### 1. Previous Attempts

Gaussian Processes (GPs) are very widely used in the field of spatial as well as temporal prediction tasks. Recently, there have been a lot of developments dedicated purely to GP modelling. Some of the most popular frameworks are listed below.

1. [SheffieldML/GPy][1]: A fully fleged, highly scalable, and flexible GP library that relies on pure numpy and scipy.
2. [cornellius-gp/gpytorch][2]: A fully fleged GP library with PyTorch backend.
3. [GPflow/GPflow][3]: Somewhat recent project for GP modelling that relies on TensorFlow (supposts both 1.x and 2.x).
4. [dfm/george][4]: A standalone fast gaussian process library (in python) which mostly relies on `numpy` and `scipy` backend. It also uses C++ ***Eigen*** library to speed up optimizations of complex log-likelihood functions.
5. [pymc-devs/pymc3][5]: A standalone gaussian process module that can be embedded in a Hierarchial Bayesian Model. Uses theano backend.
6. Others: There have been several other attempts to implement flexible gaussian processes in deep learning, machine learning, and bayesian workflows like [alshedivat/keras-gp][6] (gp in deep learning workflow), [scikit-learn/scikit-learn][7] (gp in machine learning workflows), [mblum/libgp][8] (fast gp in c++ using Eigen backend), [gpstuff-dev/gpstuff][9] (gp in matlab), etc. A full list of open source packages can be found at [awesomeopensource][10] website.

#### 1.1 GPyTorch and GPFlow - Gaussian Processes in Deep Learning Workflows

Two big players in the deep learning paradigm, TensorFlow and PyTorch, have been indulged in GP Modelling since the launch of these frameworks. Deep Learning's success in the field of machine learning has motivated many researches and developers to build Gaussian Process models that can fit in a deep learning environment without having to switch between packages. `GPyTorch` is one such package that allows the user to easily build and embed GP models in PyTorch's deep learning models.

``to be ccontinued...``

<!-- *****  References  ***** -->

[1]: https://github.com/SheffieldML/GPy
[2]: https://github.com/cornellius-gp/gpytorch
[3]: https://github.com/GPflow/GPflow
[4]: https://github.com/dfm/george
[5]: https://github.com/pymc-devs/pymc3
[6]: https://github.com/alshedivat/keras-gp
[7]: https://github.com/scikit-learn/scikit-learn
[8]: https://github.com/mblum/libgp
[9]: https://github.com/gpstuff-dev/gpstuff
[10]: https://awesomeopensource.com/projects/gaussian-processes
