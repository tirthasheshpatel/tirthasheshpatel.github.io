---
type: post
title: Graphical Models for Deep Learning
subtitle: Deep Learning Course - Part 2
bgimg: /images/graphical_models/logo_graph.jpg
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

**Conditional distribution** is a probability distribution over some variables given other variables represented by $P(X \mid Y)$

We can also have a combination of joint and conditional called the **Joint Conditional Distribution** represented by $P(X, Y, ... \mid Z)$

Joint distribution can be factorized as a product of mariginals and some conditional distributions.

$$P(X, Y) = P(X \mid Y)P(Y)$$

In case of a conditional independence, the above factorization reduces to $P(X, Y)=P(X)P(Y)$. In a more general sense,

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-1} \mid X_n)P(X_n)$$

which can be further factorized into

$$P(X_1, X_2, ..., X_n) = P(X_1, X_2, ..., X_{n-2} \mid X_{n-1}, X_n)P(X_{n-1} \mid X_n)P(X_n)$$

and so on. Now, by eliminating the factors that are independent, we can end up with a compact factorization which nicely describes the system.

We can get a marginal distribution from a joint distribution by summing up the value of all possible combinations of all other variables other than the variable over which the marginal is desired. An example is shown below:

$$P(X) = \sum_{Y}P(X, Y)$$

Now, suppose that we have $n$ number of random variables $X_1, X_2, ..., X_n$. We can now determine the number of ***parameters*** required to completely determine a joint probability distribution as $2^n - 1$.

### Why are we interested in Joint Distributions

A Joint distribution encodes in it all the required information of a system to fully describe it and answer all sorts of different questions that could arise.

***Example***: Assume you are running a oil company and you want to determine the locations that have the maximum probability of finding oil. You can use one of the systems below to describe your setting.

![Oil Graph](/images/graphical_models/oil_graph.svg)

$Y$ is the binary random variable describing the availibility of oil in a particular location. Other parameters $X_1, X_2, ..., X_6$ are the factors on which it depends. Now, we are primarily interested in $P(Y \mid X_1, X_2, ..., X_6)$ which can be determined using the joint probability over all the rvs in the system.

$$P(Y \mid X_1, X_2, ..., X_6) = \frac{P(Y, X_1, X_2, ..., X_6)}{\sum_{X_1, X_2, .., X_6}P(Y, X_1, X_2, ..., X_6)}$$

Now, using the joint distribution, we cna find the marginal

$$P(Y) = \sum_{X_1, X_2, ..., X_6}P(Y, X_1, X_2, ..., X_6)$$

We can also determine conditional independies

$$P(X_1, Y) \stackrel{?}{=} P(X)P(Y)$$

The joint distributions can in turn be used to ask how high or low the temperature ($X_4$) is at some location given we found oil there. In general,

> The joint distribution is an encyclopedia containing all the possible information about a system.

### How to represent Joint Distributions

To determine a joint distributions, with say $n$ number of variables $X_1, X_2, X_3, ..., X_n$, we have to get the all the explicit probabilities of all the combinations of RVs. This means that we need to specify $\mid X_1 \mid \times \mid X_2 \mid \times \mid X_3 \mid \times ... \times \mid X_n \mid - 1$. This means for $n$ binary variables, we need $2^n-1$ such values.

#### Challenges of explicit representation

The number of parameters required to represent the joint distribution increases exponentially and hence becomes intractable even with a very small number of parameters. Formally writing

- **Computational**: It is $\mathcal{O}(2^n-1)$ in both space and time. We are quickly going to run out of computational resources to calculate and store the joint.

- **Cognitive**: Impossible to aquire so many numbers from a human or even a expert.

- **Statistical**: Need huge amount of prior data to calculate the joint.

### Can we use a Graph to represent a Joint Distribution



### Different types of Reasoning encoded in a Bayesian Network

### Independencies encoded by Bayesian Networks

### Bayesian Networks

### I Maps

### Markov Networks

### Factors in Markov Networks

### Local Independencies in a Markov Network
