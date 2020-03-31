---
type: post
title: Restricted  Boltzmann Machines in Deep Learning
subtitle: Deep Learning Course - Part 2. See how to implement a generative model called RBM in Python
image: /images/graphical_models/rbm_generated_3.png
gh-repo: tirthasheshpatel/Generative-Models
gh-badge: [star, fork, follow]
tags: [Deep Learning, Machine Learning]
---

## Influence and Reference

This article is highly influenced by the [NPTEL's Deep Learning - Part 2 course by Mitesh Khapra](https://nptel.ac.in/courses/106/106/106106201/) and uses its material reserving all the rights to their corresponding authors. This article contains a full implementation of Restricted Boltzmann Machines using pure NumPy and no other 3rd party frameworks.

### Code

The code is available both on colab and GitHub.

1. **Colab Link** : [![launch in Colab](https://img.shields.io/badge/Open%20in-Colab-yellowgreen)](https://colab.research.google.com/drive/1dGjafQOqi2wdXvZK_QfLCrGa4hjtz_EA)

2. **GitHub Link**: [![launch on GitHub](https://img.shields.io/badge/Open%20on-GitHub-blue)](https://github.com/tirthasheshpatel/Generative-Models)

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
- [Implementing RBMs in Python](#implementation)

### Joint Distributions

#### The Movie Critic Example

We have collected reviews of a movie that are $5$ words long and these words come from a vocabulary of $50$ words. Five such reviews are shown below:

- An unexpected and necessary masterpiece
- delightfully merged information and comedy
- Directors first true masterpiece movie
- Sci-fi perfection, truly mesmerizing film
- Waste of time and money
- Best lame historical movie ever

We create a Bayesian Network assuming that each word is a random variable and each word depends only on the previous two words in the review. One such graph is shown below.

![review graph](/images/graphical_models/reviews.svg)

Now, we can write the joint distribution as

$$P(X_1, X_2, X_3, X_4, X_5) = P(X_1)P(X_2 \mid X_1)P(X_3 \mid X_2, X_1)P(X_4 \mid X_3, X_2)P(X_5 \mid X_4, X_3)$$

Using this distribution, we can sample new reviews, we can complete the incomplete reviews and also determine if the review comes from the same author! The example of determining whether the review came from the same author is shown below.

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

The above code generates the following reviews:

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

Using the joint distributions, again, we can generate images of a bedroom, denoise images and even impute the values of missing pixels in the image. Such models are called Generative Models and we are going to look at one such generative model called **Restricted Boltzmann Machines**.

### The concept of Latent Variables

> **Latent variables are hidden variables in a model which are responsible for the generation of the observed data.**

Such variables have no special meaning whatsoever. We can't predict that the hidden variable $h_1$ represents "sunny day", $h_2$ represents "trees", etc. We can just think of them as some variables that provide an ***abstract*** representation of the data we observe.

Such variables can also be used to ***generate*** an observation by tweaking their values a little.

Suppose, we have $32 \times 32$ images of a beach and we assume there are $n$ latent variables responsible for the generation of those images. More formally, we have a set $V$ of $\{V_1, V_2, ..., V_{1024}\}$ visible variables and a set $H$ of $\{H_1 H_2, ..., H_n\}$ latent variables. Can you now think of a Markov Network that has the joint distribution $P(V, H)$?

Our original Markov Network assumed we had connections between the neighboring pixels of an image. We can now assume connections between all the visible variables $V$ with all the hidden variables $H$, eliminating the original connections between neighboring pixels. This means that we try to capture the relations between neighboring pixels through the latent variables rather than direct connection them. This gives us the advantage of ***abstraction*** which is not possible to achieve by directly assuming connections between pixels. The intuition behind this is that images may vary differently on a per-pixel basis while being similar to each other in terms of what they represent. This behavior can be captured by the latent variables and not by assuming direct connections between pixels.

This concept is very similar to PCA and autoencoders.

For our case, we assume these visible and hidden variables to only take up binary values $\{0, 1\}$. In general, if we have $m$ visible variables and $n$ hidden variables then $V$ and $H$ can take up $2^m$ and $2^n$ unique values respectively and there are $2^{n+m}$ unique configurations possible.

$$V \in \{0, 1\}^{m}$$

$$H \in \{0, 1\}^{n}$$

### Restricted Boltzmann Machines

According to our previous discussion, we have the following Markov Network

![Markov Network](/images/graphical_models/markov_rbm.png)

There are connections between each visible and hidden variable but no connections between two visible or two hidden variables. So, we can write the joint distribution as the product of the following factors

$$P(V, H) = \frac{1}{Z}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)$$

We can also introduce factors tied to each visible and hidden unit until we normalize the whole expression appropriately.

$$P(V, H) = \frac{1}{Z}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j)$$

The normalization constant $Z$ is a partition function which is a sum of products over $2^{m+n}$ values as $V$ and $H$ can take up to $2^m$ and $2^n$ unique values respectively.

$$Z = \sum_V\sum_H\prod_{i}\prod_{i}\prod_{j}\phi_{ij}(v_i, h_j)\psi_i(v_i)\xi_j(h_j)$$

In our case, the visible variables in $V$ and hidden variables in $H$ can take on only binary values and partition function $Z$ is a sum over $2^{m+n}$ values.

Now, we need a representation that can be learned by a machine. And we know that machine learns... **parameters**. Hence, we have to introduce parameters to make the machine learn this joint distribution.

$$
\begin{gather*}
\phi_{ij}(v_i, h_j) = \exp(W_{ij}v_ih_j) \\
\psi_{i}(v_i) = \exp(b_iv_i) \\
\xi_{j}(h_j) = \exp(c_jh_j) \\
\end{gather*}
$$

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

The resulting joint distribution $P(V, H)$ is called a ***Boltzmann distribution*** or ***Gibb's Distribution***. We have further restricted our connections only between visible and hidden variables. Hence, this model is called **Restricted Boltzmann Machines**.

### RBMs as Stochastic Neural Networks

Let's derive some formulas that show that these RBMs are just plain old neural networks

The energy function is given by

$$E(V, H) =  - \sum_{i}\sum_{j} W_{ij}v_ih_j - \sum_{i} b_iv_i - \sum_{j} c_jh_j$$

Now, let's say $V_{-l}$ denote all the visible variables except the $l'th$ variable. Then

$$
\begin{gather*}
\alpha_l(H) = - \sum_{j=1}^{n} W_{lj}h_j - b_l \\
\beta(V_{-l}, H) = - \sum_{i=1, i \neq l}^{m} \sum_{j=1}^{n} W_{ij}v_ih_j - \sum_{i=1, i \neq l}^{m} b_iv_i - \sum_{j=1}^{n} c_jh_j \\
E(V, H) = \alpha_l(H)v_l + \beta(V_{-l}, H) \\
\end{gather*}
$$

$$
\begin{align*}
P(v_l=1 \mid H) &= P(v_l=1 \mid V_{-l}, H) \\
                &= \frac{P(v_l=1, V_{-l}, H)}{P(v_l=0, V_{-l}, H) + P(v_l=1, V_{-l}, H)} \\
                &= \frac{e^{- \alpha_l(H)1 - \beta(V_{-l}, H)}}{e^{- \alpha_l(H)1 - \beta(V_{-l}, H)} + e^{- \alpha_l(H)0 - \beta(V_{-l}, H)}} \\
                &= \frac{1}{1 + e^{ \alpha_l(H) }} \\
                &= \sigma(- \alpha_l(H)) \\
                &= \sigma(\sum_{j=1}^{n} W_{lj}h_j + b_l) \\
\end{align*}
$$

Similarly, we have

$$
\begin{align*}
P(v_l=0 \mid H) &= \sigma(\alpha_l(H)) \\
                &= \sigma(- \sum_{j=1}^{n} W_{lj}h_j - b_l) \\
\end{align*}
$$

We can similarly calculate the $P(h_l=1 \mid V, H_{-l})$.

$$
\alpha_l(V) = - \sum_{i=1}^{m} W_{il}v_i - c_l \\
\beta(V, H_{-l}) = - \sum_{i=1}^{m} \sum_{j=1, j \neq l}^{n} W_{ij}v_ih_j - \sum_{i=1}^{m} b_iv_i - \sum_{j=1, \neq l}^{n} c_jh_j \\
E(V, H) = \alpha_l(V)h_l + \beta(V, H_{-l}) \\
$$

$$
\begin{align*}
P(h_l=1 \mid V) &= P(h_l=1 \mid V, H_{-l}) \\
                &= \frac{P(h_l=1, V, H_{-l})}{P(h_l=0, V, H_{-l}) + P(h_l=1, V, H_{-l})} \\
                &= \frac{e^{- \alpha_l(V)1 - \beta(V, H_{-l})}}{e^{- \alpha_l(V)1 - \beta(V, H_{-l})} + e^{- \alpha_l(V)0 - \beta(V, H_{-l})}} \\
                &= \frac{1}{1 + e^{ \alpha_l(V) }} \\
                &= \sigma(- \alpha_l(V)) \\
P(h_l=1 \mid V) &= \sigma(\sum_{i=1}^{m} W_{il}v_i + c_l) \\
\end{align*}
$$

Similarly, we have

$$
\begin{align*}
P(h_l=0 \mid V) &= \sigma(\alpha_l(V)) \\
                &= \sigma(- \sum_{i=1}^{m} W_{il}v_i - c_l) \\
\end{align*}
$$

We have derived all the formulas we need to show that the model can be thought of as a neural network with each variable being a neuron! Let's go ahead to the last step of completely transforming this model into a neural network by deriving a loss function...

### Unsupervised Learning with RBMs

Just like in any other probabilistic framework, we want to maximize the ***likelihood*** which is the probability of observing the data. As $\log(.)$ a function is a concave function, we can maximize the log-likelihood instead.

$$
\begin{gather*}
& arg\,max_{W, b, c} \log P(V) \\
& arg\,max_{W, b, c} \log \frac{1}{Z} \sum_{H}P(V, H) \\
& arg\,max_{W, b, c} \log \sum_{H}P(V, H) - \log Z \\
& arg\,max_{W, b, c} \log \sum_{H}P(V, H) - \log \sum_{V, H} P(V, H) \\
& arg\,max_{W, b, c} \log \sum_{H} \exp(-E(V, H)) - \log \sum_{V, H} \exp(-E(V, H)) \\
\end{gather*}
$$

### Computing the gradient of the log-likelihood

Let's pretend for a second that $\theta$ is a collection of all the parameters and we want to maximize the above function wrt them. Let's start by evaluating the gradient first.

$$
\begin{align*}
\frac{\partial\mathcal{L}(\theta)}{\partial\theta} &= \frac{\partial}{\partial\theta}\left(\log \sum_{H} \exp(-E(V, H)) - \log \sum_{V, H} \exp(-E(V, H))\right) \\
                                                   &= \frac{1}{\sum_{H}\exp(-E(V, H))}\frac{\partial}{\partial\theta}\left( \sum_{H} \exp(-E(V, H)) \right) - \frac{1}{\sum_{V, H}\exp(-E(V, H))}\frac{\partial}{\partial\theta}\left( \sum_{H, V} \exp(-E(V, H)) \right) \\
                                                   &= -\frac{1}{\sum_{H}\exp(-E(V, H))}\left( \sum_{H} \exp(-E(V, H)) \frac{\partial E(V, H)}{\partial \theta} \right) + \frac{1}{\sum_{V, H} \exp(-E(V, H))}\left(\sum_{V, H} \exp(\exp(-E(V, H))\frac{\partial E(V, H)}{\partial \theta}\right) \\
                                                   &= - \sum_{H}\left( \frac{\exp(-E(V, H))}{\sum_{H} \exp(-E(V, H))}\frac{\partial E(V, H)}{\partial \theta} \right) + \sum_{V, H}\left( \frac{\exp(-E(V, H))}{\sum_{V, H} \exp(-E(V, H))}\frac{\partial E(V, H)}{\partial \theta} \right) \\
                                                   &= - \sum_{H}\left( P(H \mid V) \frac{\partial E(V, H)}{\partial \theta} \right) + \sum_{V, H}\left( P(V, H) \frac{\partial E(V, H)}{\partial \theta} \right) \\
                                                   &= - \mathbb{E}_{P(H \mid V)}\left( \frac{\partial E(V, H)}{\partial \theta} \right) + \mathbb{E}_{P(V, H)}\left( \frac{\partial E(V, H)}{\partial \theta} \right) \\
\end{align*}
$$

Now we can calculate the gradients with respect to our original parameters $W$, $b$, and $c$.

$$
\begin{gather*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} = \mathbb{E}_{P(H \mid V)}\left( v_ih_j \right) - \mathbb{E}_{P(V, H)}\left( v_ih_j \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial b_i} = \mathbb{E}_{P(H \mid V)}\left( v_i \right) - \mathbb{E}_{P(V, H)}\left( v_i \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial c_j} = \mathbb{E}_{P(H \mid V)}\left( h_j \right) - \mathbb{E}_{P(V, H)}\left( h_j \right) \\
\end{gather*}
$$

### Sampling

The gradients with respect to our parameters are expectations that are summations over an exponential number of terms and hence interactable. They can't be evaluated analytically and we have to fall back to some approximation methods. Expectations almost immediately point towards sampling. So, we need to sample from this multimodal probability distribution space of $2^{m+n}$ dimensions. This is a really difficult task that we will tackle using Gibbs Sampling. So, let's get started!

### Markov Chain

**Goal 1**: Given a random variable $X \in \mathbb{R}^n$, we want to sample from a joint distribution $P(X)$.

**Goal 2**: Given a function $f(X)$ of some random variable $X$, we want to calculate the expectation $\mathbb{E}_{P(X)}(f(X))$.

Suppose that instead of a single random variable $X$, we have a chain of random variables $X_1, ..., X_k$ where $X_i \in \mathbb{R}^n \forall i \in \{1, 2, ..., k\}$.

Instead of looking at the real values variables $X$, we will stick to our binary-valued variables.

Let's define the state of a random variable as any value from its domain. For our example, the RV $X_i$ is a $n$ dimensional vector and each dimension can take on a $0$ or a $1$. This means the size of the state space is $2^{n}$ for our case.

Suppose we are initially in some *state* $X_0$ at time $0$. We now *transition* from this state to some other state $X_1$ at time $1$. Similarly, we keep on transitioning from the previous state to some new state at each time step. Eventually, we would end up with a *chain* of $k$ random variables if we did this for $k$ time steps. We are interested in calculating the probability of transitioning to $X_{k+1}$ given all the previous states $X_1, X_2, ..., X_k$.

$$P(X_{k+1}=x_{k+1} \mid X_1=x_1, X_2, x_2, ..., X_k=x_k)$$

Assuming that the next state only depends on the previous state, we end up with a distribution

$$P(X_{k+1}=x_{k+1} \mid X_k=x_k)$$

This property of $X_{k+1} \perp \{X_1, X_2, ..., X_{k-1}\} \mid X_k$ is called the ***Markov Property*** and the resulting chain is called a ***Markov Chain***. As the variables we have are binary, we can specify the distribution $P(X_{k+1}=x_{k+1} \mid X_k=x_k)$ by defining its value for each value our RV can take in time step $k+1$ and in time step $k$. We can further arrange these probabilities in a matrix $\mathbb{T}$ where an entry $\mathbb{T}_{ab}$ denote the probability of moving from state $a$ to state $b$. Such a matrix is called a ***Transition Matrix*** and denoted by $\mathbb{T}$. For our case, the size of this matrix is $2^{n} \times 2^{n}$.

This transition matrix can change at each time step but, for our task, we assume it to be constant at all time steps. Such a Markov chain is called ***time-homogeneous Markov chain***.

Now suppose that the starting distribution at time step $0$ is given by $\mu_0$. Just to be clear, $mu^0$ is a $2^n$ dimensional vector such that $\mu^{0}_{a} = P(X_0=a)$ where $a$ is a value $X_0$ can take at time step $0$. Hence, $\mu^0$ contains probabilities of all the values $X$ can take at any time step and hence is $2^n$ dimensional vector. $\mu^{0}_a$ is the probability that the random variable takes on the value $a$ among all possible $2^n$ values. Now, we can evaluate $P(X_1=b)$ as

$$P(X_1=b) = \sum_{a}P(X_0=a, X_1=b)$$

The above formula tries to find every possible path from which we could have reached $b$ and sum up over all such probabilities in order to get the final probability. Quite intuitive, right?

$$
\begin{align*}
P(X_1=b) &= \sum_{a} P(X_0=a)P(X_1=b \mid X_0=a) \\
         &= \sum_{a} \mu^{0}_{a} \mathbb{T}_{ab}
\end{align*}
$$

We can nicely represent the distribution as a matrix multiplication shown below.

$$
\mu^{1} = P(X_1) = \mu^{0}\mathbb{T}
$$

Let's move on to $P(X_2=b)$

$$
\begin{align*}
P(X_2=b) &= \sum_{a} P(X_1=a)P(X_2=b \mid X_1=a) \\
         &= \sum_{a} \mu^{1}_{a} \mathbb{T}_{ab} \\
\end{align*}
$$

$$
\begin{align*}
P(X_2) &= \mu^{1}\mathbb{T} \\
       &= (\mu^{0}\mathbb{T})\mathbb{T} \\
       &= \mu^{0}\mathbb{T}^2
\end{align*}
$$

Similarly, for $k'th$ time step

$$P(X_k) = \mu^{0}\mathbb{T}^k$$

Now, the above equation has exponential number of terms ($\mu^{k} \in \mathbb{R}^{2^n}$ and $\mathbb{T} \in \mathbb{R}^{2^n \times 2^n}$) and hence intractable to store or calculate. Let's keep that at the back of our minds and we will revisit it later.

If at certain time step $t$, $\mu^{t}$ reashes distribution $\pi$ such that $\pi \mathbb{T} = \pi$, then for all subsequent time steps

$$P(X_j) = \mu^{j} = \pi, \forall j \geq t$$

Such a distribution $\pi$ is called the stationary distribution of the Markov Chain. Now if we continue the chainby sampling $X_{t+1} = x_{t+1}, X_{t+1}=x_{t+1}, ...$ then we can think of $x_{t+1}, x_{t+2}$ as samples drawn from the same distribution $\pi$.

***Conclusion: If we run our chain from any initial distribution $\mu^{0}$ for large number of time steps then after some arbitary point $t$, we start getting samples $x_{t+1}, x_{t+2}, ...$ which come from the stationary distribution of the Markov chain.***

### Why do we care about Markov Chains

#### Our Goal

1. Sample from $P(X)$
2. Compute the expectation $\mathbb{E}_{P(X)}(f(X))$

#### Attack Plan

Now suppose we have a Markov Chain whose stationary distribution $\pi$ is our desired probability distribution $P(X)$ then
  - we can sample from the stationary distribution easily.
  - we can use those samples to calculate the empirical estimate for our expectation
    $$\mathbb{E}(f(X)) \approx \frac{1}{n} \sum_{k}^{k+n} f(X_i) $$
    where $X_i$ is drawn from the stationary distribution $\pi$ of the markov chain.

*Theorem 1: If $X_1, X_2, ..., X_n$ is a **irreducible** time homogeneous markov chain with stationary distribution $\pi$, then*

$$\frac{1}{t}\sum_{i=1}^{t} f(X_i) \xrightarrow{t \to \infty} \mathbb{E}_{P(X)}(f(X)) \text{ where } X \in \mathbb{X} \text{ and } X \sim \pi$$

*Further, if the Markov Chain is non-periodic then*

$$P(X_t=x_t \mid X_{0}=x_{0}) \to \pi \text{ as } t \to \infty, \forall x_{t}, x_{0} \in \mathbb{X}$$

#### Adding onto our goal list

1. Define the Markov Chain for our RBMs.
2. Define what the transition matrix for our Markov Chain is.
3. Show that it is easy to sample from this chain.
4. Show that the stationary distribution $\pi$ is the desired distribution $P(V, H)$.
5. Show that the chain is ***irreducible*** and ***aperiodic***.

Let's also define $X = \{V, H\}$ which means that $\{X_1, X_2, X_3, ..., X_{m+n}\} = \{V_1, V_2, ..., V_m, H_1, H_2, ..., H_n\}$.

### Setting up a Markov Chain for RBMs

#### Procedure

1. At time step $0$, we start be randomly initializing $X_0 \in \{0, 1\}^{m+n}$ where $X = \{V, H\}$
2. We randomly choose some $i \in {1, 2, ..., m+n}$ from a uniform distribution, say, $q(i)$
3. Fix values of all the variables except $X_i$.
4. Sample the new value for $X_i$ (could be a $V$ or a $H$) using the following conditional distribution.
    - $P(X_{i}=y_i \mid X_{-i}=X_{-i})$
5. Repeat the steps 2 to 4 for many many time steps.

#### Identifying the transition matrix

If we were to provide the entire transition matrix, then we would need $\mathbb{R}^{2^{m+n} \times 2^{m+n}}$ real values. Such a matrix is impossible to store in practice. So, in the procedure above, we have defined a very very sparse matrix by only allowing those transitions where the value of only one variable changes at a time step. It can formally be written as

$$
\mathbb{T}_{xy}
\begin{cases}
q(i)P(y_i \mid x_{-i}), & \text{where $y \in \{0, 1\}^{m+n}$ and $\exists i \in \{1, 2, ..., m+n\}$ such that $X_j=y_j \forall j \in \{1, 2, ..., m+n\} - \{i\}$} \\
0,                      & otherwise
\end{cases}
$$

The above expression says that the probability of the state of more than one random variable being different is $0$. While the probability that the $i'th$ random variable would change given values of all the other random variables is $q(i)P(y_i \mid x_{-i})$.

#### How is it easy to sample from this chain?

At eash step, we need to compute $P(X_i=y_i \mid X_{-i}=x_{-i})$.

We saw how to compute in the [RBMs as Stochastic Neural Networks section](#rbms-as-stochastic-neural-networks). Let's write those out assuming $y_i=1$. (Note that $y_i$ is also binary and take values $0$ or $1$)

$$
\begin{gather*}
P(v_l=1 \mid H) = \sigma(\sum_{j=1}^{n} W_{lj}h_j + b_l) \\
P(h_l=1 \mid V) = \sigma(\sum_{i=1}^{m} W_{il}v_i + c_l) \\
\end{gather*}
$$

We can now sample from a uniform distribution and use $P(v_l=1 \mid H)$ and $P(h_l=1 \mid V)$ values as thresholds to decide weather to assign a $1$ or a $0$ to the hidden or visible variable. Formally, we are just sampling from Bernoulli's distribution with probability of success $P(v_l=1 \mid H)$ or $P(h_l=1 \mid V)$ depending on $i \leq m$ or $i > m$.

#### How do we show that the stationary distribution is P(X)?

To prove that our chain converges to our desired distribution, we need to define a theorem.

***Detailed Balance Theorem**: To show that the distribution $\pi$ of a Markov Chain described by the transition probabilities $\mathbb{T}_{xy}, x, y \in \Omega$, it is sufficient to show that $\forall x, y \in \Omega$, the following condition holds*

$$\pi(x)\mathbb{T}_{xy} = \pi(y)\mathbb{T}_{yx}$$

Let's prove this theorem for 3 different cases.

- ***Case 1: When $X$ and $Y$ are different in more than two dimensions.***

We clearly stated that we can only transition to those states where either one of the variables changes its value or all the variables remain the same. This means that, for this case, $\mathbb{T}_{xy} = 0$.

$$
\begin{gather*}
\pi(x)\mathbb{T}_{xy} \stackrel{?}{=} \pi(y)\mathbb{T}_{yx}\\
\pi(x)(0) \stackrel{?}{=} \pi(y)(0)\\
0 = 0 & (\text{prove me wrong hehe})\\
\end{gather*}
$$

- ***Case 2: When $X$ and $Y$ are the same***

$$
\begin{gather*}
\pi(x)\mathbb{T}_{xy} \stackrel{?}{=} \pi(y)\mathbb{T}_{yx}\\
\pi(x)\mathbb{T}_{xx} = \pi(x)\mathbb{T}_{xx}\\
\end{gather*}
$$

- ***Case 3: When $X$ and $Y$ differ in exactly one dimension***

$$
\begin{align*}
\pi(x)\mathbb{T}_{xy} &= P(x) (q(i)P(y_i \mid X_{-i}))\\
                      &= P(x_i, X_{-i}) (q(i)\frac{P(y_i, X_{-i})}{P(X_{-i})})\\
                      &= P(y_i, X_{-i}) (q(i)\frac{P(x_i, X_{-i})}{P(X_{-i})})\\
                      &= P(y) (q(i)P(x_i \mid X_{-i}))\\
                      &= P(y) (q(i)P(x_i \mid Y_{-i}))\\
                      &= \pi(y) \mathbb{T}_{yx}\\
\end{align*}
$$

The above case proves that the stationary distribution has converged to our distribution. All these three cases combined determine that the ***Detailed Balance Condition*** holds and we can use it as a proxy for our original distribution. Now, we have everything that we need to compute, or rather approximate, the gradient. So, let's move on to training and algorithm!

### Training RBMs using Gibbs Sampling

Let's first catch up to the last discussion of gradients that led to all these fucken of Markov chains and Gibb's Sampling.

$$
\begin{gather*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} = \mathbb{E}_{P(H \mid V)}\left( v_ih_j \right) - \mathbb{E}_{P(V, H)}\left( v_ih_j \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial b_i} = \mathbb{E}_{P(H \mid V)}\left( v_i \right) - \mathbb{E}_{P(V, H)}\left( v_i \right) \\
\frac{\partial\mathcal{L}(\theta)}{\partial c_j} = \mathbb{E}_{P(H \mid V)}\left( h_j \right) - \mathbb{E}_{P(V, H)}\left( h_j \right) \\
\end{gather*}
$$

Let's try to simplify the gradient a little bit more and then we can move on to the training part.

$$
\begin{align*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} &= \mathbb{E}_{P(H \mid V)}\left( v_ih_j \right) - \mathbb{E}_{P(V, H)}\left( v_ih_j \right) \\
                                                    &= \sum_{H}\left(P(H \mid V)v_ih_j\right) - \sum_{V, H}\left(P(V, H) v_ih_j \right) \\
                                                    &= \sum_{H}\left(P(H \mid V)v_ih_j\right) - \sum_{V} P(V) \sum_{H}\left(P(H \mid V)v_ih_j\right) \\
\end{align*}
$$

We can further simplify $\sum_{H}\left(P(H \mid V)v_ih_j\right)$ as

$$
\begin{align*}
\sum_{H}\left(P(H \mid V)v_ih_j\right) &= \sum_{h_j}\sum_{H_{-j}}\left( P(h_j \mid H_{-j}, V)P(H_{-j} \mid V)v_ih_j \right) \\
                                       &= \sum_{h_j}\left( P(h_j \mid H_{-j}, V) v_ih_j \right) \left( \sum_{H_{-j}}P(H_{-j} \mid V) \right) \\
                                       &= \sum_{h_j}\left( P(h_j \mid H_{-j}, V) v_ih_j \right) \\
                                       &= P(h_j=1 \mid H_{-j}, V)v_i \\
                                       &= \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i \\
\end{align*}
$$

Substituting this in our gradient equation, we have

$$
\begin{align*}
\frac{\partial\mathcal{L}(\theta)}{\partial W_{ij}} &= \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i - \sum_{V} P(V)\left( \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i \right) \\
                                                    &= \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i - \mathbb{E}_{P(V)}\left( \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right)v_i \right) \\
\end{align*}
$$

We can similarly derive the equations for the gradients with respect to $b_i$ and $c_j$

$$
\begin{gather*}
\frac{\partial\mathcal{L}(\theta)}{\partial b_i} = v_i - \mathbb{E}_{P(V)} v_i \\
\frac{\partial\mathcal{L}(\theta)}{\partial c_j} = \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right) - \mathbb{E}_{P(V)}\left( \sigma\left( \sum_{i=1}^{m} W_{ij}v_i + c_j \right) \right) \\
\end{gather*}
$$

We have removed the expectation with respect to $P(H, V)$ and $P(H \mid V)$ but still the expectation with respect to $P(V)$ remains and is exponential in time. Hence, we have to use Gibb's sampling procedure to compute the empirical estimate of the gradient. So, let's look at the practical implementation of this algorithm!

### Implementation

#### 1. Vectorize

We have to first vectorize all the equations we have seen so far to implement them. Remember vectorization just means we convert all the scalars to vectors and matrices to speed up the computation.

$$
\begin{gather*}
P(v=1 \mid H) = \sigma( W h + b) \\
P(h=1 \mid V) = \sigma( W^T v + c) \\
\nabla_{W} \mathcal{L}(\theta) = v \sigma\left( W^{T}v + c \right)^{T} - \mathbb{E}_{P(V)} \left( v \sigma\left( W^{T}v + c \right)^{T} \right) \\
\nabla_{b} \mathcal{L}(\theta) = v - \mathbb{E}_{P(V)} \left( v \right) \\
\nabla_{c} \mathcal{L}(\theta) = \sigma\left( W^{T}v + c \right)^{T} - \mathbb{E}_{P(V)} \left( \sigma\left( W^{T}v + c \right)^{T} \right) \\
\end{gather*}
$$

#### 2. Block Gibb's Sampling

Instead of sampling just one of the variables in one time step, we sample all of them at once and use it to update all the variables in the previous sample at once. This means that we eliminate a for loop that could cause efficiency problems. We run the chain for many many time steps (say 1 million) and use all those samples to estimate the expectation. In such a long run, all the variables are going to get updated anyways so the result is not affected.

#### 3. Code like a real man

Below is the **fully vectorized** code for implementing RBMs in python using numpy only. I have created a ``fit`` method that takes as a parameter a dataset of shape ``(n_samples, n_features)`` and trains the RBM on the dataset. Once fitted, the model can be used for two tasks:

1. Generate instances similar to those in the training dataset
2. Encode an instance to its latent space to perform dimensionality reduction.

``decode()`` method can be used to perform generation and ``encode()`` method can be used to map an instance to its latent space. Some examples and experiments are shown below.

```python
import sys
import numpy as np

def sigmoid(X):
    r"""Evaluate the sigmoid function elementwise on
    a vector or matrix X

    Parameters
    ----------
    X: array_like
        Array on which the function needs to be applied

    Returns
    -------
    sigma_X: array_like of shape `X.shape`
        Array on which sigmoid function is applied elementwise
    """
    sigma_X = 1. / (1. + np.exp(-X))
    return sigma_X

class BinaryRestrictedBoltzmannMachine(object):
    r"""A restricted boltzmann machine model that takes
    binary inputs and maps it to a binary latent space.

    Parameters
    ----------
    hidden_dims: int
        The number of hidden or latent variables in your model

    Returns
    -------
    self : object
    """

    def __init__(self, hidden_dims):
        self.hidden_dims = hidden_dims

    def _init_params(self, visible_dims):
        r"""Initialize the parameters of the model

        Parameters
        ----------
        visible_dims: int
            The number of visible dimentations.

        Returns
        -------
        None
        """
        self.visible_dims = visible_dims

        m = self.visible_dims
        n = self.hidden_dims

        self.W = np.random.randn(m, n)
        self.b = np.random.randn(m, 1)
        self.c = np.random.randn(n, 1)

        return None

    def __gibbs_step(self, V_t):
        r"""Take one gibbs sampling step.

        Parameters
        ----------
        V_t: array_like
            The values of the visible variables at time step `t`.

        Returns
        -------
        V_tplus1: array_like
            The value of variables at time step `t+1`.
        """
        # We will first sample the hidden variables using
        # P(H_t | V_t) => probability of observing H given V at time step t.
        # P(V_tplus1 | H_t) => Sample new visible varaibles at time step t+1.
        probs_H = sigmoid(self.W.T @ V_t + self.c)

        # One more thing, this is called "block" gibb's
        # sampling where we vectorize over all dimensions
        # and sample from all the dimensions at the same time.
        H_t = 1. * (np.random.rand(*probs_H.shape) <= probs_H)

        probs_V = sigmoid(self.W @ H_t + self.b)
        V_tplus1 = 1. * (np.random.rand(*probs_V.shape) <= probs_V)

        return V_tplus1

    def _gibbs_sampling(self, V_0, burn_in, tune):
        r"""The gibb's sampling step in training to calculate
        the estimates of the expectation in the gradient.

        Parameters
        ----------
        V_0: array_like
            The visible variables at time step 0.

        burn_in: int
            The number of samples to disregard from
            the chain.

        tune: int
            The number of samples to use to estimate
            the actual expectation

        Returns
        -------
        expectation_w, expectation_b, expectation_c: array_like
            The expecation term appearing in the gradients wrt W, b and c
            respectively.
        """

        # We first find the total number of training instances
        # present in the array and then the number of visible
        # and hidden dimentions.
        num_examples = V_0.shape[-1]
        m = self.visible_dims
        n = self.hidden_dims

        # We start sampling from the markov chain.
        V_sampled = self.__gibbs_step(V_0)

        # This for loop just "warms up" the chain to reach
        # its stationary distribution. Please try to keep
        # these steps as large as possible to converge to
        # the desired distribution!
        for _ in range(burn_in):
            V_sampled = self.__gibbs_step(V_sampled)

        # The chain has now reached its stationary distribution
        # and we can start collecting the samples and estimate
        # required estimations.
        expectation_b = np.sum(V_sampled,
                               axis=-1,
                               keepdims=True)
        expectation_c = np.sum(sigmoid(self.W.T @ V_sampled + self.c),
                               axis=-1,
                               keepdims=True)
        expectation_w = V_sampled @ sigmoid(self.W.T @ V_sampled + self.c).T

        # Collect a `tune` number of samples and find the
        # sum over them. We will normalize it with `tune`
        # later on...
        for i in range(tune):
            V_sampled = self.__gibbs_step(V_sampled)

            expectation_b += np.sum(V_sampled,
                                    axis=-1,
                                    keepdims=True)
            expectation_c += np.sum(sigmoid(self.W.T @ V_sampled + self.c),
                                    axis=-1,
                                    keepdims=True)
            expectation_w += V_sampled @ sigmoid(self.W.T @ V_sampled + self.c).T

        # Finally, we have to devide by the number of samples
        # we have drawn to calculate the expectation
        return (
            expectation_w / float(tune+num_examples),
            expectation_b / float(tune+num_examples),
            expectation_c / float(tune+num_examples)
        )

    def _contrastive_divergence(self, V_0, burn_in, tune):
        r"""Train using contrastive divergence method

        Parameters
        ----------
        V_0: array_like
            A training sample

        burn_in: int
            Present for API consistency.

        tune: int
            `k` term in `k-contrastive-divergence` algorithm.

        Returns
        -------
        expectation_w, expectation_b, expectation_c: array_like
            The expecation term appearing in the gradients wrt W, b and c
            respectively.
        """
        V_tilt = V_0
        for _ in range(tune):
            V_tilt = self.__gibbs_step(V_tilt)

        expectation_b = np.sum(V_tilt,
                               axis=-1,
                               keepdims=True)
        expectation_c = np.sum(sigmoid(self.W.T @ V_tilt + self.c),
                               axis=-1,
                               keepdims=True)
        expectation_w = V_tilt @ sigmoid(self.W.T @ V_tilt + self.c).T
        return expectation_w, expectation_b, expectation_c

    def _param_grads(self, V, expectation_w, expectation_b, expectation_c):
        r"""Calculate the emperical estimates of the gradients of the energy
        function with respect to [W, b, c].

        Parameters
        ----------
        V: array_like
            Visible variables/data.

        expectation_w: array_like
            Expectation term in the equation for gradient wrt W.

        expectation_b: array_like
            Expectation term in the equation for gradient wrt b.

        expectation_c: array_like
            Expectation term in the equation for gradient wrt c.

        Returns
        -------
        dloss_dW, dloss_db, dloss_dc: tuple
            Gradients wrt all the parameters in the order [W, b, c].
        """
        dloss_dW = V @ sigmoid(self.W.T @ V + self.c).T - expectation_w
        dloss_db = np.sum(V, axis=-1, keepdims=True) - expectation_b
        dloss_dc = sigmoid(self.W.T @ V + self.c) - expectation_c

        return dloss_dW, dloss_db, dloss_dc

    def _apply_grads(self, lr, dloss_dW, dloss_db, dloss_dc):
        """Update the parameters [W, b, c] of the model using
        stochastic gradient descent.

        Parameters
        ----------
        lr: int
            Learning rate of the model

        dloss_dW: array_like
            The gradient of energy function wrt W.

        dloss_db: array_like
            The gradient of energy function wrt b.

        dloss_dc: array_like
            The gradient of energy function wrt c.

        Returns
        -------
        None
        """
        # Remember we are perfoming gradient ASSCENT (not descent)
        # to MAXIMIZE (not minimize) the energy function!
        self.W = self.W + lr * dloss_dW
        self.b = self.b + lr * dloss_db
        self.c = self.c + lr * dloss_dc

    def fit(self, X, lr=0.1, epochs=10, method="contrastive_divergence", burn_in=1000, tune=2000, verbose=False):
        r"""Train the model on provided data

        Parameters
        ----------
        X: array_like
            The data array of shape (n_samples, n_features)

        lr: float, optional
            The learning rate of the model. Defaults to 0.1

        epochs: int, optional
            The number of steps to train your model

        method: string, optional
            Can be either "gitbbs_sampling" or "constrastive_divergence".
            Defaults to "constrastive_divergence"

        burn_in: int, optional
            The number of steps to warm the markov chain up

        tune: int, optional
            The number of samples to generate from the merkov chain

        verbose: bool, optional
            Weather to log the epochs or not.
        """
        # We want to vectorize over multiple batches
        # and so we have to reshape our data to `(n_features, n_samples)`
        X = X.T
        num_examples = X.shape[-1]
        self.visible_dims = X.shape[0]

        m = self.visible_dims
        n = self.hidden_dims

        # Initialize the parameters [W, b, c] of our model
        self._init_params(m)

        # Run the training for provided number of epochs
        for _ in range(epochs):
            # Emperically calculate the expectation using our markov chain.
            if method == "gibbs_sampling":
                _method = self._gibbs_sampling
            elif method == "contrastive_divergence":
                _method = self._contrastive_divergence
            else:
                raise ValueError(f"invalid method: {method}. You sholud inherit this "
                                 f"class and implement the method with an `_` at"
                                 f"the start to use it instead of built-in methods.")

            V_0 = X
            Ew, Eb, Ec = _method(V_0, burn_in=burn_in, tune=tune)

            # Using the emperical estimates of the expectation, calculate
            # the gradients wrt all our parameters
            dloss_dW, dloss_db, dloss_dc = self._param_grads(X, Ew, Eb, Ec)

            # Update the parameters
            self._apply_grads(lr, dloss_dW, dloss_db, dloss_dc)

            if verbose:
                sys.stdout.write(f"\rEpoch {_+1}")

        return self

    def decode(self, H=None):
        """Move from latent space to data space. Acts like a generator.

        Parameters
        ----------
        H: array_like, optional
            A vector of latent/hidden variables. If `None`, then it is
            randomly initialized

        Returns
        -------
        decoded: array_like
            The generated data from given latent space
        """
        # We generate a random latent space if not given
        if H is None:
            H = 1. * (np.random.rand(self.hidden_dims, 1) >= 0.5)

        # We sample the Vs given Hs.
        probs_V = sigmoid(self.W @ H + self.b)
        return 1. * (np.random.rand(*probs_V.shape) <= probs_V)

    def encode(self, V):
        """Encode the given data in its latent variables.

        Parameters
        ----------
        V: array_like
            The data to be encoded

        Returns
        -------
        encoded: array_like
            An encoded vector of the given data
        """
        # We will sampe a random H for a given V.
        probs_H = sigmoid(self.W.T @ V + self.c)
        return 1. * (np.random.rand(*probs_H.shape) <= probs_H)

```

#### 4. Experiment

- **Excellent Generator**: Let's train our rbm on one training instance of the mnist dataset and see its performance. I have trained the below model with 1 training instance with $784 (28 \times 28)$ visible variables and $3$ hidden/latent variables for 10 ``epochs``, 1000 ``burn_in`` steps (the samples we are going to disregard), 2000 ``tune`` steps (samples we are going to use to estimate the gradient), and a learning rate of ``0.1``.

```python
import numpy as np
import matplotlib.pyplot as plt
import rbm
from keras.datasets import mnist

(X_train, y), (_, _) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(60000, -1)
X_train = 1. * ((X_train[1] / 255.) >= 0.5)

# Plot the image
plt.imshow(X_train.reshape(28, 28))
plt.title("Training Instance")
plt.show()

# Convert the dimensions to format `(n_samples, n_features)`.
# For an image, pixels are its `n_features` and we have only
# one training instance.
X = X_train.reshape(1, -1)

# We will mainly experiment with different latent space
# dimensions. For this instance, i have a 30-D latent space.
hidden_dims = 3

# Define our model
model = rbm.BinaryRestrictedBoltzmannMachine(hidden_dims)

# Train the model on our dataset with learning rate 0.5
model.fit(X, method="gibbs_sampling", 0.1)

# Use the `decode()` method to generate an image.
image = model.decode()

# Plot the generated image.
plt.imshow(image.reshape(28, 28))
plt.title("Generated Instance")
plt.show()
```

This is the image I used for training.

![Training Instance](/images/graphical_models/rbm_train_instance.png)

And this is the generated image.

![Generated instance](/images/graphical_models/rbm_generated_instance.png)

You can see the generated image is very similar to the one on which the model was trained! This is because there is only one image and we have set a very high dimensional latent space.

- Good Generator: Let's train our model on 100 images of a handwritten 3 and see its performance. I have trained the below model with 1 training instance with $784 (28 \times 28)$ visible variables and $3$ hidden/latent variables for 20 ``epochs``, 1000 ``burn_in`` steps (the samples we are going to disregard), 2000 ``tune`` steps (samples we are going to use to estimate the gradient), and a learning rate of ``0.005``.

```python
import numpy as np
import matplotlib.pyplot as plt
import rbm
from keras.datasets import mnist

(X_train, y), (_, _) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(60000, -1)
X_train = 1. * ((X_train[y == 3][:101] / 255.) >= 0.5)

# Plot some training isntances
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
ax[0, 0].imshow(X_train[0].reshape(28, 28))
ax[0, 1].imshow(X_train[1].reshape(28, 28))
ax[0, 2].imshow(X_train[2].reshape(28, 28))
ax[1, 0].imshow(X_train[3].reshape(28, 28))
ax[1, 1].imshow(X_train[4].reshape(28, 28))
ax[1, 2].imshow(X_train[5].reshape(28, 28))
ax[2, 0].imshow(X_train[6].reshape(28, 28))
ax[2, 1].imshow(X_train[7].reshape(28, 28))
ax[2, 2].imshow(X_train[8].reshape(28, 28))
fig.suptitle("Training instances")
plt.show()

# We will mainly experiment with different latent space
# dimensions. For this instance, i have a 30-D latent space.
hidden_dims = 3

# Define our model
model = rbm.BinaryRestrictedBoltzmannMachine(hidden_dims)

# Train the model on our dataset with learning rate 0.5
model.fit(X_train, lr=0.005, method="gibbs_sampling", burn_in=1000, tune=2000, epochs=20, verbose=True)

# Use the `decode()` method to generate an image.
images = [model.decode() for _ in range(9)]

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
ax[0, 0].imshow(images[0].reshape(28, 28))
ax[0, 1].imshow(images[1].reshape(28, 28))
ax[0, 2].imshow(images[2].reshape(28, 28))
ax[1, 0].imshow(images[3].reshape(28, 28))
ax[1, 1].imshow(images[4].reshape(28, 28))
ax[1, 2].imshow(images[5].reshape(28, 28))
ax[2, 0].imshow(images[6].reshape(28, 28))
ax[2, 1].imshow(images[7].reshape(28, 28))
ax[2, 2].imshow(images[8].reshape(28, 28))
fig.suptitle("Generated instances")
plt.show()
```

Some training instances are shown below.

![Training instances](/images/graphical_models/rbm_train_3.png)

This model generates the following images!

![Generated images](/images/graphical_models/rbm_generated_3.png)

As you can see the model generates very good instances and can be, more or less, be used as a generative model. But some images are not too good. You can try to generate better images by setting up the hyperparameters like ``tune``, ``burn_in``, ``epochs``, ``lr``, etc. Don't forget to show your results off in the comment section on my GitHub page. Let's move ahead to the last experiment of this model.

- **Bad Generator**: The model performs very well on training instances that are similar to just training on images of $5$ or $0$. But what happens if we train it on more than one type of image (like a mix of all 10 digits)?? Let's see.

```python
import numpy as np
import matplotlib.pyplot as plt
import rbm
from keras.datasets import mnist

(X_train, y), (_, _) = mnist.load_data()

# Normalize and reshape
X_train = X_train.reshape(60000, -1)
X_train = 1. * ((X_train[:2] / 255.) >= 0.5)

# Plot the image
plt.imshow(X_train[0].reshape(28, 28))
plt.title("Training Instance")
plt.show()

plt.imshow(X_train[1].reshape(28, 28))
plt.title("Training Instance")
plt.show()

# We will mainly experiment with different latent space
# dimensions. For this instance, i have a 30-D latent space.
hidden_dims = 3

# Define our model
model = rbm.BinaryRestrictedBoltzmannMachine(hidden_dims)

# Train the model on our dataset with learning rate 0.5
model.fit(X_train, method="gibbs_sampling", lr=0.05, burn_in=1000, tune=2000, epochs=20, verbose=True)

# Use the `decode()` method to generate an image.
image = model.decode()

# Plot the generated image.
plt.imshow(image.reshape(28, 28))
plt.title("Generated Instance")
plt.show()
```

The training instances are shown below. We have trained the model on only two instances. One instance is a $0$ and the other is a $5$.

![Training instance 1](/images/graphical_models/rbm_mm_train_1.png)
![Training instance 2](/images/graphical_models/rbm_mm_train_2.png)

The generated images are

![Generated image](/images/graphical_models/rbm_mm_generate.png)

What?! The model just learned to put one image onto the other! This is not expected, right? Let me explain. The problem is not training but a sampling. This is one of the limitations of Gibb's Sampling. This sampling method sometimes thinks that the expectation lies at the peak of the distribution. But as explained in the Betancourt's paper on MCMC methods, the actual samples must lie inside the so-called, "***typical set***" of the expectation and not the mode of the distribution. Well, this can be solved using more advanced sampling methods like Hamiltonian Monte Carlo or No U-Turn Sampler. PyMC3 folks have done a great job on those sampling methods and you should definitely check it out!!

Even though we have removed the exponent from the time complexity, we need to sample thousands and thousands of samples from our chain which is very inefficient! So, let's try to tackle that last thing!!

### Training using Contrastive Divergence

``WIP``

Code is ready. You can add this method to the class and use it instead of ``_gibbs_sampling()``.

```python
def _contrastive_divergence(self, V_0, burn_in, tune):
    r"""Train using contrastive divergence method

    Parameters
    ----------
    V_0: array_like
        A training sample

    burn_in: int
        Present for API consistency.

    tune: int
        `k` term in `k-contrastive-divergence` algorithm.

    Returns
    -------
    expectation_w, expectation_b, expectation_c: array_like
        The expecation term appearing in the gradients wrt W, b and c
        respectively.
    """
    V_tilt = V_0
    for _ in range(tune):
        V_tilt = self.__gibbs_step(V_tilt)

    expectation_b = np.sum(V_tilt,
                            axis=-1,
                            keepdims=True)
    expectation_c = np.sum(sigmoid(self.W.T @ V_tilt + self.c),
                            axis=-1,
                            keepdims=True)
    expectation_w = V_tilt @ sigmoid(self.W.T @ V_tilt + self.c).T
    return expectation_w, expectation_b, expectation_c
```

Having done so much mathematics, coding, and experimenting, I hope you guys took something home!

Signing out!
Tirth Patel.
