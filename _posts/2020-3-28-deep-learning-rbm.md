---
type: post
title: Restricted  Boltzmann Machines in Deep Learning
subtitle: Deep Learning Course - Part 2
image: /images/graphical_models/logo_rbm.jpeg
tags: [Deep Learning, Machine Learning]
---

## Influence and Reference

This article is highly influenced by the [NPTEL's Deep Learning - Part 2 course by Mitesh Khapra](https://nptel.ac.in/courses/106/106/106106201/) and uses its material reserving all the rights to their corresponding authors. This article contains a full implementation of Restricted Boltzmann Machines using pure NumPy and no other 3rd party frameworks.

### Table of Contents

- [Joint Distributions](#joint-distributions)
- [The concept of Latent Variables](#the-concept-of-latent-variables)
- [Restricted Boltzmann Machines](#restricted-boltzmann-machines)
- [RBMs as Stochastic Neural Networks](#rbms-as-stochastic-neural-networks)
- [Unsupervised Learning with RBMs](#unsupervised-learning-with-rbms)
- [Computing the gradient of the log likelihood](#computing-the-gradient-of-the-log-likelihood)
- [Sampling](#sampling)
- [Markov Chain](#markov-chain)
- [Why do we care about Markov Chains](#why-do-we-care-about-markov-chains)
- [Setting up a Markov Chain for RBMs](#setting-up-a-markov-chain-for-rbms)
- [Training RBMs using Gibbs Sampling](#training-rbms-using-gibbs-sampling)
- [Training RBMs using Contrastive Divergence](#training-rbms-using-contrastive-divergence)

### Joint Distributions

#### The Movie Critic Example

We have collected reviews of a movie that are $5$ words long and these words come from a vocabulary of $50$ words. Five such reviews are shown below:

- An unexpected and necessary masterpiece
- delightfully merged information and comedy
- Directors first true masterpiece movie
- Sci-fi perfection, tryly mesmerizing film
- Waste of time and money
- Best lame historical movie ever

We create a Bayesian Network assuming that each word is a random variable and each word depends only on previous two words in the review. One such graph is shown below.

![review graph](/images/graphical_models/reviews.svg)

Now, we can write the joint distribution as

$$P(X_1, X_2, X_3, X_4, X_5) = P(X_1)P(X_2 \mid X_1)P(X_3 \mid X_2, X_1)P(X_4 \mid X_3, X_2)P(X_5 \mid X_4, X_3)$$

Using this distribution, we can sample new reviews, we can complete the incomplete reviews and also determine if the review comes from the same author! The example of determining weather the review came from the same author is shown below.

![Review Example](/images/graphical_models/review_example.png)

Similarly, we can sample new reviews using a simple code shown below

```python
import numpy as np

review = [None, None]

words = np.array(
            [
                'the',
                'movie',
                'amazing',
                'useless',
                'was',
                'is',
                'masterpiece',
                'i',
                'liked',
                'descent'
            ]
        )

probs = dict()

probs[('the', 'movie')] = [.01, .01, .01, .03, .60, .30, .01, .01, .01, .01]

# Add conditional independencies for all users

for _ in range(5):
    outcomes = np.random.choice(np.arange(10), p=probs[(review[-2], review[-1])])
    review.append(words[outcome])

print(''.join(review[2:]))
```

The above code generates following reviews:

1. I liked the amazing movie
2. The movie is a masterpiece
3. The movie was liked descent

#### The Bedroom Images Example

![The bedroom](/images/graphical_models/bedroom.jpg)

Suppose we are given some ${1024} \times {1024}$ images of different bedrooms like the one above and want to train the machine to generate new bedroom images. This task is very similar to the reviews example except that there is no sense of direction in case of images, unlike reviews. Assuming the pixels as our RVs, we need a Markov Network to represent the connections between the neighboring pixels instead of a Bayesian Network.

![Bedroom Network](/images/graphical_models/bedroom_network.png)

Using the above graph, we can factorize the joint as

$$P(X_1, X_2, ..., X_{1024}) = \frac{1}{Z}\prod_i\phi_1(D_i)$$

where $D_i$ is a set of variables that form a maximum clique (groups of neighboring pixels).

Using the joint distributions, again, we can generate images of a bedroom, denoise given images and even inpute the values of missing pixels in the image. Such models are called Generative Models and we are going to look at one such generative model called **Restricted Boltzmann Machines**.

### The concept of Latent Variables

> **Latent variables are hidden variables in a model which are responsible for the generation of the observed data.**

Such variables have no special meaning whatsoever. We can't predeict that the hidden variable $h_1$ represents sunny day, $h_2$ represents trees, etc. We can just think of them as some variables that provide an ***abstract*** representation of the data we observe.

Such variables can also be used to ***generate*** an observation by tweaking their values a little.

Suppose, we have $32 \times 32$ images of a beach and we assume there are $n$ latent variables responsible for the generation of those images. More formally, we have a set $V$ of $\{V_1, V_2, ..., V_{1024}\}$ visible variables and a set $H$ of $\{H_1 H_2, ..., H_n\}$ latent variables. Can you now think of a Markov Network that has the joint distribution $P(V, H)$?

Our original Markov Network assumed we had connections between the neighboring pixels of an image. We can now assume connections between all the visible variables $V$ with all the hidden variables $H$, eliminating the original connections between neighboring pixels. This means that we try to capture the relations between neighboring pixels through the latent variables rather than directly connection them. This gives us the advantage of ***abstraction*** which is not possible to achieve by directly assuming connections between pixles. An intuition behisd this is that images may vary differently on per pizel basis while being similar to each other in terms of what they represent. This behaviour can be captured by the latent variables and not by assuming direct connections between pixles.

This concept is very similar to PCA and auto encoders.

For our case, we assume these visible and hidden variables to only take up binary values $\{0, 1\}$. In general, if we have $m$ visible variables and $n$ hidden variables then $V$ and $H$ can take up $2^m$ and $2^n$ unique values respectively and there are $2^{n+m}$ unique configurations possible.

$$V \belongsto \{0, 1\}^{m}$$

$$H \belongsto \{0, 1\}^{n}$$

### Restricted Boltzmann Machines

According to our previous discussion, we have the following Markov Network

![Markov Network](/images/graphical_models/markov_rbm.png)

There are connections between each visible and hidden variable but no connections between two visible or two hidden variables. So, we can write the joint distirbution as the product of the following factors

$$P(V, H) = \frac{1}{Z}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)$$

We can also introduce factors tied to each visible and hidden unit until we normalize the whole expression appropriately.

$$P(V, H) = \frac{1}{Z}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j)$$

Normalization contant $Z$ is a partition function which is a sum of products over $2^{m+n}$ values as $V$ and $H$ can take upto $2^m$ and $2^n$ unique values respectively.

$$Z = \sum_V\sum_H\prod_{i}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j)$$

In our case, the visible varibles in $V$ and hidden variables in $H$ can take on only binary values and partition function $Z$ is a sum over $2^{m+n}$ values.

Now, we need a representation that can be learned by a machine. And we know that machine learns... **parameters**. Hence, we have to introduce paramters in order to make the machine learn this joint distribution.

$$\phi_{ij}(v_i, h_j) = \exp(W_{ij}v_ih_j)$$

$$\psi_{i}(v_i) = \exp(b_iv_i)$$

$$\xi_{j}(h_j) = \exp(c_jh_j)$$

This particular choice of parameters leads to a joint distribution of the following form

$$
\begin{align*}
P(V, H) &= \frac{1}{Z}\prod_{i}\prod_{j}\left[ \phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j) \right]\\
        &= \frac{1}{Z}\prod_{i}\prod_{j}\left[ \exp(W_{ij}v_ih_j)\exp(b_iv_i)\exp(c_jh_j) \right]\\
        &= \frac{1}{Z}\exp \left[ \sum_{i}\sum_{j} W_{ij}v_ih_j + \sum_{i} b_iv_i + \sum_{j} c_jh_j \right]\\
        &= \frac{1}{Z}\exp\left[ - E(V, H) \right]\\
\end{align*}
$$

where $E(V, H)$  is the energy function and is given by

$$ E(V, H) = - \sum_{i}\sum_{j} W_{ij}v_ih_j - \sum_{i} b_iv_i - \sum_{j} c_jh_j $$

The resulting joint distribution $P(V, H)$ is called a ***Boltzmann distribution*** or ***Gibb's Distribution***. We have further restricted our connections only between visible and hidden variables. Hence, this models are called **Restriced Boltzmann Machines**.

### RBMs as Stochastic Neural Networks

Let's derive some formulas that show that these RBMs are just plain old neural networks

The energy function is given by

$$
\begin{align*}
E(V, H) &=  - \sum_{i}\sum_{j} W_{ij}v_ih_j - \sum_{i} b_iv_i - \sum_{j} c_jh_j
\end{align*}
$$

Now, let's say $V_{-l}$ denote all the visible variables except the $l$'th varaible. Then

$$\alpha_l(H) = - \sum_{j=1}^{n} W_{lj}h_j - b_l$$

$$\beta(V_{-l}, H) = - \sum_{i=1, i \neq l}^{m} \sum_{j=1}^{n} W_{ij}v_ih_j - \sum_{i=1, i \neq l}^{m} b_iv_i - \sum_{j=1}^{n} c_jh_j$$

$$E(V, H) = \alpha_l(H)v_l + \beta(V_{-l}, H)$$

$$
\begin{align*}
P(v_l=1 \mid H) &= P(v_l=1 \mid V_{-l}, H) \\
                &= \frac{P(v_l=1, V_{-l}, H)}{P(v_l=0, V_{-l}, H) + P(v_l=1, V_{-l}, H)} \\
                &= \frac{e^{\alpha_l(H)1 + \beta(V_{-l}, H)}}{e^{\alpha_l(H)1 + \beta(V_{-l}, H)} + e^{\alpha_l(H)0 + \beta(V_{-l}, H)}} \\
                &= \frac{1}{1 + e^{ - \alpha_l(H) }} \\
                &= \sigma(\alpha_l(H)) \\
                &= \sigma(- \sum_{j=1}^{n} W_{lj}h_j - b_l) \\
\end{align*}
$$

Similarly, we have

$$
\begin{align*}
P(v_l=0 \mid H) &= \sigma(-\alpha_l(H)) \\
                &= \sigma(\sum_{j=1}^{n} W_{lj}h_j + b_l) \\
\end{align*}
$$

We can similarly calculate the P(h_l=1 \mid V, H_{-l}).

$$\alpha_l(V) = - \sum_{i=1}^{m} W_{il}v_i - c_l$$

$$\beta(V, H_{-l}) = - \sum_{i=1}^{m} \sum_{j=1, j \neq l}^{n} W_{ij}v_ih_j - \sum_{i=1}^{m} b_iv_i - \sum_{j=1, \neq l}^{n} c_jh_j$$

$$E(V, H) = \alpha_l(V)h_l + \beta(V, H_{-l})$$

$$
\begin{align*}
P(h_l=1 \mid V) &= P(h_l=1 \mid V, H_{-l}) \\
                &= \frac{P(h_l=1, V, H_{-l})}{P(h_l=0, V, H_{-l}) + P(h_l=1, V, H_{-l})} \\
                &= \frac{e^{\alpha_l(V)1 + \beta(V, H_{-l})}}{e^{\alpha_l(V)1 + \beta(V, H_{-l})} + e^{\alpha_l(V)0 + \beta(V, H_{-l})}} \\
                &= \frac{1}{1 + e^{ - \alpha_l(V) }} \\
                &= \sigma(\alpha_l(V)) \\
                &= \sigma(- \sum_{i=1}^{m} W_{il}v_i - c_l) \\
\end{align*}
$$

Similarly, we have

$$
\begin{align*}
P(h_l=0 \mid V) &= \sigma(-\alpha_l(V)) \\
                &= \sigma(\sum_{i=1}^{m} W_{il}v_i + c_l) \\
\end{align*}
$$

We have derived all the formulas we need to show that the model can be thought of as a neural network with each variables being a neuron! Let's go ahead to the last step of completely transforming this model into a neural network by deriveing a loss function...

### Unsupervised Learning with RBMs

Just like in any other probabilistic framework, we want to maximize the ***likelihood*** which is the probability of observing the data. As $\log(.)$ function is a concave function, we can maximize the log likelihood instead.

$$
\begin{align}
& arg\,max_{W, b, c} \log P(V) \\
& arg\,max_{W, b, c} \log \frac{1}{Z} \sum_{H}P(V, H) \\
& arg\,max_{W, b, c} \log \sum_{H}P(V, H) - \log Z \\
& arg\,max_{W, b, c} \log \sum_{H}P(V, H) - \log \sum_{V, H} P(V, H) \\
& arg\,max_{W, b, c} \log \sum_{H} \exp(-E(V, H)) - \log \sum_{V, H} \exp(-E(V, H)) \\
\end{align}
$$

### Computing the gradient of the log likelihood

Let's pretend for a second that $\theta$ is a collection of all the paramters and we want to maximize the above function wrt them. Let's start by evaluating the gradient first.

$$
\begin{align*}
\frac{\partial\mathcal{L}(\theta)}{\partial\theta} &= \frac{\partial}{\partial\theta}\left(\log \sum_{H} \exp(-E(V, H)) - \log \sum_{V, H} \exp(-E(V, H))\right) \\
                                                   &= \frac{1}{\sum_{H}\exp(-E(V, H))}\frac{\partial}{\partial\theta}\left( \sum_{H} \exp(-E(V, H)) \right) - \frac{1}{\sum_{V, H}\exp(-E(V, H))}\frac{\partial}{\partial\theta}\left( \sum_{H, V} \exp(-E(V, H)) \right) \\
                                                   &= -\frac{1}{\sum_{H}\exp(-E(V, H))}\left( \sum_{H} \exp(-E(V, H)) \frac{\partial E(V, H)}{\partial \theta} \right) + \frac{1}{\sum_{V, H} \exp(-E(V, H))}\left(\sum_{V, H} \exp(\exp(-E(V, H))\frac{\partial E(V, H)}{\partial \theta}\right) \\
                                                   &= - \sum_{H}\left( \frac{\exp(-E(V, H))}{\sum_{H} \exp(-E(V, H))}\frac{\partial E(V, H)}{\partial \theta} \right) + \sum_{V, H}\left( \frac{\exp(-E(V, H))}{\sum_{V, H} \exp(-E(V, H))}\frac{\partial E(V, H)}{\partial \theta} \right) \\
                                                   &= - \sum_{H}\left( P(H \mid V) \frac{\partial E(V, H)}{\partial \theta} \right) + \sum_{V, H}\left( P(H, V) \frac{\partial E(V, H)}{\partial \theta} \right)
                                                   &= - \mathbb{E}_{P(H \mid V)}\left( \frac{\partial E(V, H)}{\partial \theta} \right) + \mathbb{E}_{P(V, H)}\left( \frac{\partial E(V, H)}{\partial \theta} \right)
\end{align*}
$$

### Sampling

### Markov Chain

### Why do we care about Markov Chains

### Setting up a Markov Chain for RBMs

### Training RBMs using Gibbs Sampling

### Training RBMs using Contrastive Divergence
