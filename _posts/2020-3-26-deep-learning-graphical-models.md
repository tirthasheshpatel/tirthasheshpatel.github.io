---
type: post
title: Graphical Models for Deep Learning
image: /images/graphical_models/logo_graph.jpg
tags: [Machine Learning, Deep Learning]
---

### Table of Contents

- [Recap of Probability Thoery](#recap-of-probability-theory)
- [Why are we interested in Joint Distributions](#why-are-we-interested-in-joit-distributions)
- [How to represent Joint Distributions](#how-to-represent-joint-distributions)
- [Can we use a Graph to represent a Joint Distribution](#can-we-use-a-graph-to-represent-a-joint-distribution)
- [Different types of Reasoning encoded in a Bayesian Network](#different-types-of-reasoning-encoded-in-a-bayesian-network)
- [Independencies encoded by Bayesian Networks](#independencies-encoded-by-bayesian-networks)
- [Bayesian Networks](#bayesian-networks)
- [I Maps](#i-maps)
- [Markov Networks](#markov-networks)
- [Factors in Markov Networks](#factors-in-markov-networks)
- [Local Independencies in a Markov Network](#local-independencies-in-a-markov-network)

### Recap of Probability Thoery

**Marginal distribution** is a Probability distribution over a single Random Variable represented by $P(X)$.

**Joint distribution** is a probability distribution over all possble combinations of two or more RVs represented by $P(X, Y, ...)$

**Conditional distribution** is a probability distribution over some varables given other variables represented by $P(X \mid Y)$

We can also have a combination of joint and conditional called the **Joint Conditional Distribution** represented by $P(X, Y, ... \mid Z)$

Joint distribution can be factorized as a product of mariginals and some conditional distributions.

$$P(X, Y) = P(X \mid Y)P(Y)$$

In case of a conditional independence, the above factorization reduces to $P(X, y)=P(X)P(Y)$. In a more general sense,

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-1} \mid X_n)P(X_n)$$

which can be further factorized into

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-2} \mid X_{n-1}, X_n)P(X_{n-1} \mid X_n)P(X_n)$$

and so on. Now, by eliminating the factors that are independent, we can end up with a compact factorization which nicely describes the system.

We can get a marginal distribution from a joint distribution by summing up the value of all possible combinations of all other variables other than the variable over which the marginal is desired. An example is shown below:

$$P(X) = \sum_{Y}P(X, Y)$$

Now, suppose that we have $n$ number of random variables $X_1, X_2, ..., X_n$. We can now specify the number of ***parameters*** required to completely determine a joint probability distribution as $2^n - 1$.

### Why are we interested in Joint Distributions

### How to represent Joint Distributions

### Can we use a Graph to represent a Joint Distribution

### Different types of Reasoning encoded in a Bayesian Network

### Independencies encoded by Bayesian Networks

### Bayesian Networks

### I Maps

### Markov Networks

### Factors in Markov Networks

### Local Independencies in a Markov Network
