---
layout: post
title: Gaussian Processes for Machine Learning
category: Machine Learning
---

Gaussian Processes are stationary ramdom processes that can be used for spacial as well as temporal predictions. They have been used extensively by diciplines such as astronomy and biology for prediction and meansuring uncertainty in thier data. In this blog, I have collected many different approaches to Gaussian Processes and their variations.

### What are Gaussian Processes

The formal definition of Gaussian Processes may be scary for many beginners, so here is a very simple definition by Rasmussen and Williams that I love:

*A Gaussian Process is a collection of random variables,any finite number of which have (consistent) joint Gaussian distributions.*

A Gaussian Process is fully parameterized by its mean function (not a mean vector) \\( m \\) and a covariance function \\( k \\) and can be written as:

\\[ f \sim \mathcal{GP}\left( m, k \right) \\]

These functions take a bunch of data points, commonly called "index" points and spit out a mean and covariance function and the underlying (latent) function responsible for the generation of data can then be modelled as a multivariate normal distribution.

\\[ f(X) \sim \mathcal{N}\left(m(X), k(X, X^\prime)\right) \\]

### Referances

[Gaussian Processes in Machine Learning by Rasmussen et. al.](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
